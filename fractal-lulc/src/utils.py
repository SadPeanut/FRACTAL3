"""
utils.py
--------
Small helpers shared by the pipeline modules:
  * StageTimer       - wall-clock timing of named stages
  * build_spark      - creates the SparkSession
  * cluster_settings - reads the executor setup back from the Spark conf
  * write_text       - writes a small text file to local disk, HDFS or S3
"""
import time
from contextlib import contextmanager

from pyspark.sql import SparkSession


class StageTimer:
    """Collects wall-clock durations (in seconds) of named code blocks.

    Spark is lazy: a block is only timed correctly if it ends with an action
    (count, collect, fit, write...) that forces the work to actually happen.
    """

    def __init__(self):
        # name of the stage -> elapsed seconds
        self.seconds = {}

    @contextmanager
    def stage(self, name):
        """Usage: `with timer.stage("training"): ...`"""
        start = time.perf_counter()
        try:
            yield
        finally:
            # Recorded even if the block raises, so a failed run shows where it stopped
            self.seconds[name] = time.perf_counter() - start
            print(f"[timer] {name}: {self.seconds[name]:.1f} s", flush=True)


def build_spark(app_name):
    """Create (or get) the SparkSession.

    Executor count, cores and memory are NOT set here: they are given to
    spark-submit by run_experiment.sh. In YARN cluster mode the driver is
    already running when this code starts, so the driver settings could not
    be changed from inside the application anyway.
    """
    spark = SparkSession.builder.appName(app_name).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")  # keep the logs readable

    # Size the number of shuffle partitions from the cluster size (4 per core).
    # We read the configured values rather than sc.defaultParallelism because
    # executors may not have registered yet when the session starts.
    conf = spark.sparkContext.getConf()
    n_exec = conf.get("spark.executor.instances", None)
    cores = conf.get("spark.executor.cores", None)
    if n_exec and cores:
        total_cores = int(n_exec) * int(cores)
    else:  # e.g. local mode
        total_cores = spark.sparkContext.defaultParallelism
    spark.conf.set("spark.sql.shuffle.partitions", str(total_cores * 4))
    # Input partitions: by default Spark makes about one per core, i.e. a single wave of tasks. Then one
    # slow patch decides the stage time and extra executors have nothing to do. Asking for 3 per core
    # (slide "3 more partitions than # of cores", Spark Tuning p.6) gives the scheduler tasks to rebalance.
    # It is a lower bound: big inputs still get ~256 MB partitions (maxPartitionBytes).
    # An explicit --conf spark.sql.files.minPartitionNum=N (EXTRA_SPARK_ARGS) wins, e.g. to keep N fixed in a
    # strong-scaling run where only the number of executors changes.
    if conf.get("spark.sql.files.minPartitionNum", None) is None:
        spark.conf.set("spark.sql.files.minPartitionNum", str(total_cores * 6))
    return spark


def cluster_settings(spark):
    """Return the executor setup actually used by this run (None if not set, e.g. local mode)."""
    conf = spark.sparkContext.getConf()
    keys = {
        "master": "spark.master",
        "n_executors": "spark.executor.instances",
        "executor_cores": "spark.executor.cores",
        "executor_memory": "spark.executor.memory",
        "driver_memory": "spark.driver.memory",
    }
    return {name: conf.get(key, None) for name, key in keys.items()}


def registered_executors(spark):
    """Number of executors that YARN actually granted and that are alive right now.

    With dynamic allocation off, `spark.executor.instances` is only the number requested. On a shared
    cluster fewer may be granted, which silently changes what a "16 executors" run measures
    (Spark Tuning p.32). PySpark does not expose getExecutorInfos, so the JVM object is used.
    Returns 0 in local mode (only the driver is listed).
    """
    infos = spark.sparkContext._jsc.sc().statusTracker().getExecutorInfos()
    return len(infos) - 1  # the list also holds the driver


def write_text(spark, path, text):
    """Write `text` to `path` through the Hadoop FileSystem API.

    Works for local paths (file:///...), HDFS and S3 (s3:// or s3a://), so the
    results JSON can be written from the driver wherever it runs.
    """
    jvm = spark._jvm
    hadoop_path = jvm.org.apache.hadoop.fs.Path(path)
    fs = hadoop_path.getFileSystem(spark._jsc.hadoopConfiguration())
    out = fs.create(hadoop_path, True)  # True = overwrite; parent folders are created
    try:
        out.write(bytearray(text.encode("utf-8")))
    finally:
        out.close()
