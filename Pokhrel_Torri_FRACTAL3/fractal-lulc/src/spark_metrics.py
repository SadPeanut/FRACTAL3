"""
spark_metrics.py
----------------
Task metrics with sparkMeasure (seen in the Spark Tuning slides), collected for
each stage of the pipeline and saved in the results JSON: number of tasks, task
times, GC, spill, executors and hosts used. analysis/diagnose.py prints them.

The sparkMeasure jar is added  by run_experiment.sh (--packages). We call it
through the JVM, so the Python package is not needed. If the jar is missing,
the pipeline still runs, just without these metrics.
"""
from contextlib import contextmanager

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

# name in our results -> possible column names in sparkMeasure (they change a bit between versions)
SUM_COLUMNS = {
    "executorRunTime": ["executorRunTime"],
    "executorCpuTime": ["executorCpuTime"],
    "jvmGCTime": ["jvmGCTime"],
    "schedulerDelayTime": ["schedulerDelay", "schedulerDelayTime"],
    "shuffleFetchWaitTime": ["shuffleFetchWaitTime"],
    "bytesRead": ["bytesRead"],
    "shuffleTotalBytesRead": ["shuffleTotalBytesRead"],
    "shuffleBytesWritten": ["shuffleBytesWritten"],
    "memoryBytesSpilled": ["memoryBytesSpilled"],
    "diskBytesSpilled": ["diskBytesSpilled"],
}


def _pick(df, names):
    """First of `names` that is a column of df, else None."""
    return next((n for n in names if n in df.columns), None)


def summarize_tasks(tasks):
    """Turn sparkMeasure's task table (one row per task) into a small dict.

    totals: sums over all tasks, number of tasks, executors and hosts used.
    stages: for each Spark stage, number of tasks, median and max task time (ms),
            biggest shuffle read and spilled bytes. max / median shows the skew.
    """
    cols = {key: _pick(tasks, names) for key, names in SUM_COLUMNS.items()}
    # totals over the whole pipeline stage
    aggs = [F.count(F.lit(1)).alias("numTasks"),
            F.countDistinct("executorId").alias("executors_used"),
            F.countDistinct("host").alias("hosts_used")]
    aggs += [F.sum(c).alias(key) for key, c in cols.items() if c]
    row = tasks.agg(*aggs).collect()[0].asDict()
    totals = {k: int(v or 0) for k, v in row.items() if k not in ("executors_used", "hosts_used")}

    # same kind of numbers, but for each Spark stage
    run, shuffle, spill = cols["executorRunTime"], cols["shuffleTotalBytesRead"], cols["diskBytesSpilled"]
    stage_aggs = [F.count(F.lit(1)).alias("n_partitions"),
                  F.expr(f"percentile_approx({run}, 0.5)").alias("median_run_ms"),
                  F.max(run).alias("max_run_ms")]
    if shuffle:
        stage_aggs.append(F.max(shuffle).alias("max_shuffle_read_bytes"))
    if spill:
        stage_aggs.append(F.sum(spill).alias("spill_bytes"))
    stages = [{k: (int(v) if v is not None else 0) for k, v in r.asDict().items()}
              for r in tasks.groupBy("stageId").agg(*stage_aggs).orderBy("stageId").collect()]
    return {"totals": totals, "executors_used": int(row["executors_used"] or 0),
            "hosts_used": int(row["hosts_used"] or 0), "stages": stages}


class SparkMetrics:
    """Usage: `with metrics.stage("training"): ...` (does nothing if sparkMeasure is unavailable)."""

    def __init__(self, spark, enabled=True):
        self.spark = spark
        self.summary = {}  # stage name -> summarize_tasks(...) result
        self._tm = None
        if not enabled:
            return
        try:
            # same call as sparkmeasure.TaskMetrics(spark) in the Python package
            self._tm = spark._jvm.ch.cern.sparkmeasure.TaskMetrics(spark._jsparkSession)
        except Exception as exc:  # the jar is not on the cluster
            print(f"[spark_metrics] sparkMeasure not available, metrics skipped ({type(exc).__name__}). "
                  "Run through run_experiment.sh (it adds --packages) or set SPARK_MEASURE_PACKAGE.", flush=True)

    @contextmanager
    def stage(self, name):
        """Collect the metrics of the tasks run inside the `with` block."""
        if self._tm is None:
            yield
            return
        self._tm.begin()  # begin() also empties the task list, so we only get this stage
        try:
            yield
        finally:
            self._tm.end()
            try:
                tasks = DataFrame(self._tm.createTaskMetricsDF(f"task_metrics_{name}"), self.spark)
                self.summary[name] = summarize_tasks(tasks)
            except Exception as exc:  # a problem with the metrics must not stop the run
                print(f"[spark_metrics] could not summarise stage {name}: {exc}", flush=True)
