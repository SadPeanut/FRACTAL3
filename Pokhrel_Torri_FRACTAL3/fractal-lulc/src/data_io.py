"""
data_io.py
----------
Loading of the FRACTAL Parquet files.

In FRACTAL every Parquet file is one 50 m x 50 m patch, so sampling files is
the same as sampling patches. A "fraction" of the dataset therefore means a
fraction of the files of each split.
"""
import random

from pyspark.sql import functions as F

# Only these columns are read from the Parquet files (columnar storage lets
# Spark skip the others, e.g. the heavy `wkb` geometry column).
RAW_COLUMNS = [
    "xyz",              # array<double>: x, y, z coordinates
    "Intensity",
    "ReturnNumber",
    "NumberOfReturns",
    "Red", "Green", "Blue", "Infrared",
    "Classification",   # raw LAS class code (the target)
]


def list_parquet_files(spark, path):
    """List the .parquet files directly under `path` (local, HDFS or S3), sorted by name."""
    jvm = spark._jvm
    hadoop_path = jvm.org.apache.hadoop.fs.Path(path)
    fs = hadoop_path.getFileSystem(spark._jsc.hadoopConfiguration())
    statuses = fs.listStatus(hadoop_path)
    files = [str(s.getPath()) for s in statuses if str(s.getPath()).endswith(".parquet")]
    return sorted(files)  # sorting makes the sample independent of the listing order


def sample_files(files, fraction, seed):
    """Pick `fraction` percent of the files, reproducibly.

    The whole list is shuffled once with a fixed seed and the first k files are
    kept. Consequence: the samples are nested (the 1% sample is contained in
    the 2% sample, which is contained in the 5% sample...), so runs with
    different fractions are comparable.
    """
    if fraction >= 100:
        return list(files)
    k = max(1, int(round(len(files) * fraction / 100.0)))  # at least one file
    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:k]


def load_split(spark, path, fraction, seed):
    """Read a sample of one split (train / val / test).

    Returns (DataFrame, number_of_files_read). The DataFrame holds RAW_COLUMNS
    plus `patch`, the path of the file a point comes from (= its patch id).
    """
    files = list_parquet_files(spark, path)
    if not files:
        raise ValueError(f"No .parquet files found under {path}")
    selected = sample_files(files, fraction, seed)

    # When everything is selected, point Spark at the folder: avoids passing
    # tens of thousands of explicit paths.
    inputs = [path] if len(selected) == len(files) else selected
    df = spark.read.parquet(*inputs).select(
        *RAW_COLUMNS,
        F.col("_metadata.file_path").alias("patch"),  # hidden file-source column
    )
    return df, len(selected)
