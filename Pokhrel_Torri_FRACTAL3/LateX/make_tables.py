"""
make_tables.py
--------------
Writes the LaTeX tables of the report (LateX/tables/*.tex) from the result JSON files of the
experiments, so that every number in the report can be traced back to a run.

    python LateX/make_tables.py --results_dir results --out_dir LateX/tables

Result folders used (day:month of the run, as named in results/):
    13:11 results ...            final matrix: 1, 5, 10, 25 %, 6 partitions per core, 32 nodes
    13:11 testing on validation  one run that also evaluates the validation split
    12:11 results 5% times 6 ... 5 %, 6 partitions per core, 24 nodes
    old/12:11 results 5% times 3 ...  5 %, 3 partitions per core, 24 nodes
    old/10:11 results 1% ...     first configuration (100 trees, class weights)
"""
import argparse
import glob
import json
import os
import statistics

STAGES = ["preprocessing", "training", "evaluation"]
CLASSES = ["other", "ground", "vegetation", "building", "water", "bridge", "structure"]


def load(results_dir, pattern):
    """All runs of the folders matching `pattern`, as {(fraction, executors granted): run dict}."""
    runs = {}
    for path in sorted(glob.glob(os.path.join(results_dir, pattern, "*.json"))):
        with open(path) as f:
            r = json.load(f)
        granted = [v for v in (r["cluster"].get("executors_granted") or {}).values() if v > 0]
        n = min(granted) if granted else int(r["cluster"]["n_executors"])
        r["n"] = n
        r["fraction"] = float(r["params"]["fraction"])
        r["t"] = dict(r["timings_sec"])
        r["t"]["prep_excl"] = r["t"]["preprocessing"] - r["t"].get("file_listing", 0.0)
        runs[(r["fraction"], n)] = r
    return runs


def f0(x):
    return f"{x:.0f}"


def write(out_dir, name, text):
    with open(os.path.join(out_dir, name), "w") as f:
        f.write(text)
    print("wrote", name)


def util(r, stage):
    """Core utilisation = executor run time / (wall time x cores)."""
    cores = r["n"] * int(r["cluster"]["executor_cores"])
    return r["spark_metrics"][stage]["totals"]["executorRunTime"] / 1000.0 / (r["t"][stage] * cores)


def core_time(r, stage="training"):
    """Total executor run time of a stage, in seconds of core time."""
    return r["spark_metrics"][stage]["totals"]["executorRunTime"] / 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_dir", default="LateX/tables")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    final = load(a.results_dir, "13:11 results*")
    val = load(a.results_dir, "13:11 testing on validation set")
    first = load(a.results_dir, "old/10:11 results 1%*")
    c3 = load(a.results_dir, "old/12:11 results 5% times 3*")
    c6 = load(a.results_dir, "12:11 results 5% times 6*")
    fractions = sorted({k[0] for k in final})

    # ---- Table: model quality (test set, largest executor count of each fraction) ----
    def qrow(label, r, split="test"):
        m = r["metrics"][split]
        ious = " & ".join(f"{m['per_class_iou'][c]:.3f}" for c in CLASSES)
        return f"{label} & {100 * m['overall_accuracy']:.1f} & {m['mIoU']:.3f} & {ious} \\\\\n"

    q = qrow("A: 100 trees, class weights, 1\\,\\%", first[(1.0, 20)])
    q += "\\hline\n"
    for fr in fractions:
        n = max(k[1] for k in final if k[0] == fr)
        q += qrow(f"Final, {fr:g}\\,\\%", final[(fr, n)])
    q += "\\hline\n"
    q += qrow("Final, 1\\,\\%, validation split", val[(1.0, 60)], "val")
    write(a.out_dir, "tab_quality.tex", q + "\\hline\n")  # closing rule here: a \hline after \input breaks in recent LaTeX

    # spread of the quality over the executor counts of one fraction (RF depends slightly on partitioning)
    spread = {fr: [final[k]["metrics"]["test"]["mIoU"] for k in final if k[0] == fr] for fr in fractions}
    write(a.out_dir, "quality_spread.txt", "\n".join(
        f"{fr:g}% mIoU min {min(v):.3f} max {max(v):.3f}" for fr, v in spread.items()) + "\n")

    # ---- Table: stage times of the final matrix ----
    ns = sorted({k[1] for k in final})
    t = "Fraction & Stage & " + " & ".join(str(n) for n in ns) + " \\\\\n\\hline\n"
    for fr in fractions:
        rows = [("preprocessing", "Preprocessing", "preprocessing"), ("listing", "\\quad of which file listing", "file_listing"),
                ("training", "Training", "training"), ("evaluation", "Evaluation", "evaluation")]
        for i, (_, label, key) in enumerate(rows):
            cells = [f0(final[(fr, n)]["t"][key]) if (fr, n) in final else "--" for n in ns]
            first_col = f"{fr:g}\\,\\%" if i == 0 else ""
            t += f"{first_col} & {label} & " + " & ".join(cells) + " \\\\\n"
        t += "\\hline\n"
    write(a.out_dir, "tab_times.tex", t)

    # ---- Table: speedups (and parallel efficiency) at the largest executor count ----
    s = ""
    for fr in fractions:
        by_n = sorted(k[1] for k in final if k[0] == fr)
        n0, n1 = by_n[0], by_n[-1]
        cells = []
        for key in ["preprocessing", "prep_excl", "training", "evaluation"]:
            sp = final[(fr, n0)]["t"][key] / final[(fr, n1)]["t"][key]
            cells.append(f"{sp:.1f} ({100 * sp / (n1 / n0):.0f}\\,\\%)")
        s += f"{fr:g}\\,\\% & {n0}$\\to${n1} & {n1 / n0:.1f} & " + " & ".join(cells) + " \\\\\n"
    write(a.out_dir, "tab_speedup.tex", s + "\\hline\n")  # closing rule here: a \hline after \input breaks in recent LaTeX

    # ---- Table: why training does not reach the ideal speedup ----
    # measured speedup = ideal x (utilisation(n) / utilisation(n0)) / (core time(n) / core time(n0))
    w = ""
    for fr in fractions:
        by_n = sorted(k[1] for k in final if k[0] == fr)
        n0, n1 = by_n[0], by_n[-1]
        r0, r1 = final[(fr, n0)], final[(fr, n1)]
        growth = core_time(r1) / core_time(r0)
        measured = r0["t"]["training"] / r1["t"]["training"]
        w += (f"{fr:g}\\,\\% & {n0}$\\to${n1} & {n1 / n0:.1f} & {100 * util(r0, 'training'):.0f}\\,\\% $\\to$ "
              f"{100 * util(r1, 'training'):.0f}\\,\\% & {growth:.2f} & {measured:.1f} \\\\\n")
    write(a.out_dir, "tab_work.tex", w + "\\hline\n")  # closing rule here: a \hline after \input breaks in recent LaTeX

    # ---- Table: same executors, 24 or 32 nodes (5 %, training) ----
    nd = ""
    for label, runs in [("24 nodes", c6), ("32 nodes", {k: v for k, v in final.items() if k[0] == 5.0})]:
        for n in sorted(k[1] for k in runs if k[0] == 5.0):
            r = runs[(5.0, n)]
            hosts = r["spark_metrics"]["training"]["hosts_used"]
            nd += (f"{label if n == min(k[1] for k in runs if k[0] == 5.0) else ''} & {n} & {hosts} & "
                   f"{r['data']['train_input_partitions']} & {r['t']['training']:.0f} & "
                   f"{core_time(r) / 1000:.1f} \\\\\n")
        nd += "\\hline\n"
    write(a.out_dir, "tab_nodes.tex", nd)

    # ---- Table: 3 or 6 input partitions per core (5 %, 24 nodes) ----
    p = ""
    for n in sorted({k[1] for k in c3 if k[0] == 5.0} & {k[1] for k in c6 if k[0] == 5.0}):
        a3, a6 = c3[(5.0, n)], c6[(5.0, n)]
        p += (f"{n} & {a3['data']['train_input_partitions']} / {a6['data']['train_input_partitions']} & "
              f"{100 * util(a3, 'training'):.0f} / {100 * util(a6, 'training'):.0f} & "
              f"{a3['t']['preprocessing']:.0f} / {a6['t']['preprocessing']:.0f} & "
              f"{a3['t']['training']:.0f} / {a6['t']['training']:.0f} & "
              f"{a3['t']['total']:.0f} / {a6['t']['total']:.0f} \\\\\n")
    write(a.out_dir, "tab_partitions.tex", p + "\\hline\n")  # closing rule here: a \hline after \input breaks in recent LaTeX

    # ---- a few numbers quoted in the text ----
    e = final[(25.0, 60)]
    lines = [
        f"25% e60 file listing {e['t']['file_listing']:.1f} s, preprocessing {e['t']['preprocessing']:.1f} s, "
        f"training {e['t']['training']:.1f} s, evaluation {e['t']['evaluation']:.1f} s, total {e['t']['total']:.1f} s",
        f"25% points train {e['data']['train_points']} test {e['data']['test_points']}",
        "feature importances 25% e60: " + ", ".join(f"{k} {v:.3f}" for k, v in sorted(
            e["feature_importances"].items(), key=lambda kv: -kv[1])),
    ]
    write(a.out_dir, "numbers.txt", "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
