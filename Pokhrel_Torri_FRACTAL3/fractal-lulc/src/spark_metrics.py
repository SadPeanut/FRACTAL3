"""
spark_metrics.py
----------------
Task-level performance metrics with sparkMeasure (the tool of the Spark Tuning slides, p.32-37), collected
separately for each pipeline stage and stored in the results JSON. They show, per stage, how many partitions
(= tasks) there were, how uneven the tasks were, whether anything spilled to disk and how many executors and
hosts took part, i.e. the information needed to retune the configuration (analysis/diagnose.py prints it).

The sparkMeasure jar must be on the cluster: run_experiment.sh adds it with --packages. The Python package
`sparkmeasure` is NOT needed: the same JVM calls are made directly, so nothing has to be installed on the
worker node that runs the driver. If the jar is missing the pipeline still runs and the metrics are left out.
"""
from contextlib import contextmanager

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

# result key -> possible column names in sparkMeasure's task table (differs a little between versions)
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
    """Reduce sparkMeasure's task table (one row per task) to a small dict.

    totals : sums over all tasks + number of tasks, of executors and of hosts that ran at least one task
    stages : per Spark stage, n_partitions (= tasks), median / max task run time (ms), largest shuffle read
             of one task, spilled bytes. The ratio max / median shows skew (stragglers).
    Only small aggregates are collected to the driver.
    """
    cols = {key: _pick(tasks, names) for key, names in SUM_COLUMNS.items()}
    aggs = [F.count(F.lit(1)).alias("numTasks"),
            F.countDistinct("executorId").alias("executors_used"),
            F.countDistinct("host").alias("hosts_used")]
    aggs += [F.sum(c).alias(key) for key, c in cols.items() if c]
    row = tasks.agg(*aggs).collect()[0].asDict()
    totals = {k: int(v or 0) for k, v in row.items() if k not in ("executors_used", "hosts_used")}

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
        except Exception as exc:  # jar not on the classpath: py4j returns a non-callable JavaPackage
            print(f"[spark_metrics] sparkMeasure not available, metrics skipped ({type(exc).__name__}). "
                  "Run through run_experiment.sh (it adds --packages) or set SPARK_MEASURE_PACKAGE.", flush=True)

    @contextmanager
    def stage(self, name):
        if self._tm is None:
            yield
            return
        self._tm.begin()
        try:
            yield
        finally:
            self._tm.end()
            try:
                tasks = DataFrame(self._tm.createTaskMetricsDF(f"task_metrics_{name}"), self.spark)
                self.summary[name] = summarize_tasks(tasks)
            except Exception as exc:  # never let the measurement break a run
                print(f"[spark_metrics] could not summarise stage {name}: {exc}", flush=True)
