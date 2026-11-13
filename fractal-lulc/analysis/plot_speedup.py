"""
plot_speedup.py
---------------
Builds the mandatory speedup figures from the results JSON files written by
main.py. Runs on a normal machine (no Spark needed); copy the results first:

    aws s3 sync s3://my-bucket/fractal-results ./results

    python analysis/plot_speedup.py --results_dir ./results --out_dir ./figures

Output: one figure per stage (preprocessing, training, evaluation), black and
white as required, with
    x : number of executors
    y : speedup T(n0) / T(n), n0 = smallest executor count run for that fraction
    one curve per dataset fraction + the ideal linear speedup n / n0.
Also writes summary.csv (times and quality of every run).
"""
import argparse
import csv
import glob
import json
import os
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt

STAGES = ["preprocessing", "training", "evaluation"]
# extra figure: preprocessing without the driver-only file listing (a constant that executors cannot shorten)
PLOT_STAGES = STAGES + ["preprocessing_excl_listing"]
# Without colour, curves are told apart by marker and line style
MARKERS = ["o", "s", "^", "D", "v", "P"]
LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]


def load_runs(results_dir):
    """Read every *.json file of the folder; runs without a numeric executor count are skipped."""
    runs = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            r = json.load(f)
        n_exec = r["cluster"].get("n_executors")
        if n_exec is None:
            print(f"skipping {path}: no executor count (local run?)")
            continue
        # x coordinate = executors actually granted by YARN (smallest count seen during the run), not
        # just requested; older result files without this information fall back to the requested count.
        granted = [v for v in (r["cluster"].get("executors_granted") or {}).values() if v > 0]
        n_used = min(granted) if granted else int(n_exec)
        if n_used != int(n_exec):
            print(f"warning: {r['run_name']} requested {n_exec} executors but ran with {n_used}")
        times = dict(r["timings_sec"])
        times["preprocessing_excl_listing"] = times["preprocessing"] - times.get("file_listing", 0.0)
        runs.append({
            "run_name": r["run_name"],
            "fraction": float(r["params"]["fraction"]),
            "n_exec": n_used,
            "times": times,
            "test": r["metrics"]["test"],
        })
    return runs


def median_times(runs, stage):
    """{fraction: {n_exec: time}}; if a configuration was run several times, take the median."""
    grouped = defaultdict(lambda: defaultdict(list))
    for r in runs:
        grouped[r["fraction"]][r["n_exec"]].append(r["times"][stage])
    return {f: {n: statistics.median(ts) for n, ts in by_n.items()} for f, by_n in grouped.items()}


def plot_stage(runs, stage, out_dir):
    """One speedup figure for one stage."""
    times = median_times(runs, stage)
    fig, ax = plt.subplots(figsize=(5, 4))

    n0_values = []
    all_n = set()
    for i, fraction in enumerate(sorted(times)):
        by_n = times[fraction]
        ns = sorted(by_n)
        n0 = ns[0]                       # smallest executor count that ran this fraction
        speedup = [by_n[n0] / by_n[n] for n in ns]
        ax.plot(ns, speedup, color="black", marker=MARKERS[i % len(MARKERS)],
                linestyle=LINESTYLES[i % len(LINESTYLES)], label=f"{fraction:g} %")
        n0_values.append(n0)
        all_n.update(ns)

    # Ideal linear speedup n / n0, drawn from the smallest n0 over all fractions
    n0_ref = min(n0_values)
    ideal_x = [min(all_n), max(all_n)]
    ax.plot(ideal_x, [n / n0_ref for n in ideal_x], color="gray", linestyle=(0, (1, 3)),
            linewidth=1.5, label="ideal (n / n0)")

    ax.set_xlabel("Number of executors")
    ax.set_ylabel("Speedup T(n0) / T(n)")
    ax.set_title(f"Speedup - {stage}")
    ax.set_xticks(sorted(all_n))
    ax.grid(True, linestyle=":", color="lightgray")
    ax.legend(title="Dataset fraction")
    fig.tight_layout()
    for ext in ("pdf", "png"):  # pdf for the LaTeX report
        fig.savefig(os.path.join(out_dir, f"speedup_{stage}.{ext}"), dpi=200)
    plt.close(fig)


def write_summary(runs, out_dir):
    """summary.csv: one row per run with the stage times and the test-set quality."""
    path = os.path.join(out_dir, "summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "fraction_pct", "executors", *[f"{s}_sec" for s in STAGES],
                    "total_sec", "test_OA", "test_mIoU"])
        for r in sorted(runs, key=lambda r: (r["fraction"], r["n_exec"])):
            w.writerow([r["run_name"], r["fraction"], r["n_exec"],
                        *[round(r["times"][s], 1) for s in STAGES], round(r["times"]["total"], 1),
                        round(r["test"]["overall_accuracy"], 4), round(r["test"]["mIoU"], 4)])
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser(description="Speedup figures from FRACTAL run results")
    p.add_argument("--results_dir", required=True, help="folder with the results JSON files")
    p.add_argument("--out_dir", default="figures", help="where to write the figures")
    args = p.parse_args()

    runs = load_runs(args.results_dir)
    if not runs:
        raise SystemExit("no usable result files found")
    os.makedirs(args.out_dir, exist_ok=True)

    for stage in PLOT_STAGES:
        plot_stage(runs, stage, args.out_dir)
        print(f"wrote speedup_{stage}.pdf/png")
    write_summary(runs, args.out_dir)


if __name__ == "__main__":
    main()
