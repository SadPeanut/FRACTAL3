"""
diagnose.py
-----------
Reads the sparkMeasure metrics saved in the results JSON files and prints, for
each run and stage, a few numbers that help to understand the scaling:

  cores     executors x cores
  parts     number of tasks of the biggest Spark stage
  t/core    parts / cores (below 2-3, some cores have nothing to do)
  util      busy time of the executors / (wall time x cores)
  skew      max / median task time (above 1.5, a few slow tasks hold up the stage)
  crit      slowest tasks / wall time (close to 1: the stage waits for its slowest tasks)
  gc        share of the time spent in garbage collection (above 10 % = memory pressure)
  spill_MB  data written to disk because the memory was full

    python analysis/diagnose.py --results_dir ./results
"""
import argparse
import glob
import json
import os
import re

STAGES = ["preprocessing", "training", "evaluation"]


def total_cores(cluster):
    """Cores of the run: executors x cores on a cluster, N for a local[N] master (None for local[*])."""
    if cluster.get("n_executors") and cluster.get("executor_cores"):
        return int(cluster["n_executors"]) * int(cluster["executor_cores"])
    m = re.fullmatch(r"local\[(\d+)\]", cluster.get("master") or "")
    return int(m.group(1)) if m else None


def stage_row(run, stage):
    """The numbers above for one stage of one run (None if the run has no metrics)."""
    metrics = (run.get("spark_metrics") or {}).get(stage)
    if not metrics or not metrics["stages"]:
        return None
    timings = run["timings_sec"]
    # the file listing runs on the driver only, so it is removed from the preprocessing time
    wall = timings[stage] - (timings.get("file_listing", 0.0) if stage == "preprocessing" else 0.0)
    cores = total_cores(run["cluster"])
    stages, totals = metrics["stages"], metrics["totals"]
    # ignore the very short stages (e.g. the file listing job of Spark, with tasks of a few ms)
    real =[s for s in stages if s["median_run_ms"] >= 500] or stages
    parts = max(s["n_partitions"] for s in real)
    big = [s for s in real if s["n_partitions"] >= 0.5 * parts]
    skew = sum(s["max_run_ms"] / s["median_run_ms"] for s in big) / len(big) if big else float("nan")
    run_ms = max(totals["executorRunTime"], 1)
    return {
        "cores": cores, "parts": parts, "t/core": parts / cores,
        "util": totals["executorRunTime"] / 1000 / (wall * cores),
        "skew": skew,
        "crit": sum(s["max_run_ms"] for s in stages) / 1000 / wall,
        "gc": totals["jvmGCTime"] / run_ms,
        "spill_MB": (totals["diskBytesSpilled"] + totals["memoryBytesSpilled"]) / 1e6,
    }


def hints(row):
    """Short advice when a number looks bad."""
    out = []
    if row["t/core"] < 2:
        out.append("few tasks per core -> more partitions (spark.sql.files.minPartitionNum)")
    if row["util"] < 0.7:
        out.append("idle cores")
    if row["skew"] > 1.5:
        out.append("skewed tasks")
    if row["gc"] > 0.10:
        out.append("GC pressure -> more memory / partitions")
    if row["spill_MB"] > 0:
        out.append("spilling -> more partitions or memory")
    return "; ".join(out)


def main():
    """Print one line per run and stage."""
    p = argparse.ArgumentParser(description="Scaling diagnosis from sparkMeasure metrics")
    p.add_argument("--results_dir", required=True)
    args = p.parse_args()
    runs = []
    for path in glob.glob(os.path.join(args.results_dir, "*.json")):
        with open(path) as f:
            r = json.load(f)
        if total_cores(r["cluster"]):  # local[*] runs are skipped: the number of cores is not recorded
            runs.append(r)
    runs.sort(key=lambda r: (float(r["params"]["fraction"]), total_cores(r["cluster"])))
    # header, then one line per run and stage
    print(f"{'run':<14}{'stage':<14}{'cores':>6}{'parts':>7}{'t/core':>8}{'util':>7}{'skew':>6}{'crit':>6}{'gc':>6}{'spill_MB':>10}  hints")
    for r in runs:
        for stage in STAGES:
            row = stage_row(r, stage)
            if row is None:
                continue
            print(f"{r['run_name']:<14}{stage:<14}{row['cores']:>6}{row['parts']:>7}{row['t/core']:>8.1f}"
                  f"{row['util']:>7.0%}{row['skew']:>6.2f}{row['crit']:>6.2f}{row['gc']:>6.1%}{row['spill_MB']:>10.0f}  {hints(row)}")


if __name__ == "__main__":
    main()
