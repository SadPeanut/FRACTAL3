"""
utils.py
--------
Small helpers shared by the pipeline modules:
  * StageTimer           - wall-clock timing of named stages
  * build_spark          - creates the SparkSession
  * cluster_settings     - reads the executor setup back from the Spark conf
  * registered_executors - number of executors really running
  * write_text           - writes a small text file to local disk, HDFS or S3
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
            # saved even if the block fails, so we can see where a failed run stopped
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

    # Shuffle partitions = 4 per core (the slides advise 2-4 per core).
    # We use the values given to spark-submit because the executors may not be
    # registered yet when the session starts.
    conf = spark.sparkContext.getConf()
    n_exec = conf.get("spark.executor.instances", None)
    cores = conf.get("spark.executor.cores", None)
    if n_exec and cores:
        total_cores = int(n_exec) * int(cores)
    else:  # e.g. local mode
        total_cores = spark.sparkContext.defaultParallelism
    spark.conf.set("spark.sql.shuffle.partitions", str(total_cores * 4))
    # Input partitions: at least 6 per core. By default Spark makes about one per core, so one slow task
    # decides the stage time. With more (smaller) tasks the scheduler can spread the work better.
    # It is only a lower bound: big inputs are still cut in 256 MB partitions (maxPartitionBytes).
    # Can be changed with --conf spark.sql.files.minPartitionNum=N in EXTRA_SPARK_ARGS.
    if conf.get("spark.sql.files.minPartitionNum", None) is None:
        spark.conf.set("spark.sql.files.minPartitionNum", str(total_cores * 6))
    return spark


def cluster_settings(spark):
    """Return the executor setup actually used by this run (None if not set, e.g. local mode)."""
    conf = spark.sparkContext.getConf()
    # our name -> spark setting (set by run_experiment.sh with spark-submit)
    keys = {
        "master": "spark.master",
        "n_executors": "spark.executor.instances",
        "executor_cores": "spark.executor.cores",
        "executor_memory": "spark.executor.memory",
        "driver_memory": "spark.driver.memory",
    }
    return {name: conf.get(key, None) for name, key in keys.items()}


def registered_executors(spark):
    """Number of executors YARN really gave us (alive right now).

    `spark.executor.instances` is only what we asked for, and we can get fewer. PySpark has no
    function for this, so we go through the JVM. Returns 0 in local  mode.
    """
    infos = spark.sparkContext._jsc.sc().statusTracker().getExecutorInfos()
    return len(infos) - 1  # the list also holds the driver


def write_text(spark, path, text):
    """Write `text` to `path` through the Hadoop FileSystem API.

    Works for local paths (file:///...), HDFS and S3 (s3:// or s3a://), so the
    results JSON can be written from the driver wherever it runs.
    """
    # same as list_parquet_files in data_io.py
    jvm = spark._jvm
    hadoop_path = jvm.org.apache.hadoop.fs.Path(path)
    fs = hadoop_path.getFileSystem(spark._jsc.hadoopConfiguration())
    out = fs.create(hadoop_path, True)  # True = overwrite; parent folders are created
    try:
        out.write(bytearray(text.encode("utf-8")))  # java wants bytes
    finally:
        out.close()
