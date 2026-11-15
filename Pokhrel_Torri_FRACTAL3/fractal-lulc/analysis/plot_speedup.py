"""
plot_speedup.py
---------------
Makes the speedup figures from the results JSON files written by main.py.
No Spark needed, copy the results first:

    aws s3 sync s3://my-bucket/fractal-results ./results
    python analysis/plot_speedup.py --results_dir ./results --out_dir ./figures

One black and white figure per stage (preprocessing, training, evaluation, plus
preprocessing without the file listing and the total), with
    x : number of executors
    y : speedup T(n0) / T(n), n0 = smallest executor count run for that fraction
    one curve per dataset fraction + the ideal speedup n / n0.
Also makes the total time figure and summary.csv (times and quality of each run).
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

plt.rcParams.update({"font.size": 9})  # small figures: the report shows four of them per page

STAGES = ["preprocessing", "training", "evaluation"]
# extra figure: preprocessing without the driver-only file listing (a constant that executors cannot shorten)
PLOT_STAGES = STAGES + ["preprocessing_excl_listing", "total"]  # "total": whole run (the three stages)
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
        # x = executors really given by YARN (smallest value seen during the run);
        # old result files without it use the requested number
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
    fig, ax = plt.subplots(figsize=(4.2, 3.4))

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

    # Ideal speedup n / n0. The curves do not all start at the same n0 (8 for 1-10 %, 40 for 25 %),
    # so one dotted line is drawn for each n0.
    max_n = max(all_n)
    for n0 in sorted(set(n0_values)):
        ax.plot([n0, max_n], [1.0, max_n / n0], color="gray", linestyle=(0, (1, 3)), linewidth=1.5,
                label=f"ideal n / n0 (n0 = {n0})")

    ax.set_xlabel("Number of executors")
    ax.set_ylabel("Speedup T(n0) / T(n)")
    ax.set_title(f"Speedup - {stage}")
    ax.set_xticks(sorted(all_n))
    ax.grid(True, linestyle=":", color="lightgray")
    ax.legend(title="Dataset fraction", fontsize=7, title_fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):  # pdf for the LaTeX report
        fig.savefig(os.path.join(out_dir, f"speedup_{stage}.{ext}"), dpi=200)
    plt.close(fig)


def plot_time(runs, stage, out_dir):
    """Time (seconds) against the number of executors for one stage. Log axis: the fractions differ by 25x."""
    times = median_times(runs, stage)
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    all_n = set()
    for i, fraction in enumerate(sorted(times)):
        by_n = times[fraction]
        ns = sorted(by_n)
        ax.plot(ns, [by_n[n] for n in ns], color="black", marker=MARKERS[i % len(MARKERS)],
                linestyle=LINESTYLES[i % len(LINESTYLES)], label=f"{fraction:g} %")
        all_n.update(ns)
    ax.set_yscale("log")
    ax.set_xlabel("Number of executors")
    ax.set_ylabel("Time (s)")
    ax.set_title(f"Execution time - {stage}")
    ax.set_xticks(sorted(all_n))
    ax.grid(True, which="both", linestyle=":", color="lightgray")
    # legend below the axes: the curves fill the whole plot area
    ax.legend(title="Dataset fraction", fontsize=7, title_fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.2), ncol=4)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"time_{stage}.{ext}"), dpi=200)
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
    """Read the results and write all the figures."""
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
    plot_time(runs, "total", args.out_dir)
    print("wrote time_total.pdf/png")
    write_summary(runs, args.out_dir)


if __name__ == "__main__":
    main()
