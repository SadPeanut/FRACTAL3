"""
preprocessing.py
----------------
Turns the raw point clouds into a compact table of features + label.

All transformations are stateless (nothing is learned from the training set),
so the same function is applied to train, val and test. A fitted
`Pipeline` is therefore not needed here; the model pipeline (assembler +
classifier) lives in training.py.
"""
from pyspark import StorageLevel
from pyspark.sql import functions as F

# The 7 target classes of the project, in label order (label 0 ... 6).
CLASS_NAMES = ["other", "ground", "vegetation", "building", "water", "bridge", "structure"]

# Raw LAS class code -> label. Any code not listed (65 artefacts, 66 virtual
# points, ...) goes to "other" (label 0). Checked against the class shares of
# the FRACTAL dataset card (e.g. vegetation = codes 3+4+5 = 57.0 % of train).
RAW_TO_CLASS = {1: 0, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 9: 4, 17: 5, 64: 6}

# Model inputs. The absolute X/Y coordinates are deliberately NOT used: the
# train/val/test split is geographic, so coordinates would only let the model
# memorise places instead of learning what the classes look like.
FEATURE_COLS = [
    "Height",           # height above the lowest point of the patch (see preprocess)
    "Intensity",
    "ReturnNumber",
    "NumberOfReturns",
    "Red", "Green", "Blue", "Infrared",
    "NDVI",             # (NIR - Red) / (NIR + Red)
    "Brightness",       # mean of R, G, B
]


def label_expression():
    """Spark column expression mapping the raw class code to the 0..6 label (as double, as MLlib expects)."""
    expr = None
    for raw_code, label in RAW_TO_CLASS.items():
        condition = F.col("Classification") == raw_code
        expr = F.when(condition, F.lit(label)) if expr is None else expr.when(condition, F.lit(label))
    return expr.otherwise(F.lit(0)).cast("double")


def preprocess(df, height_norm=True):
    """Compute the features and the label. Returns a DataFrame with FEATURE_COLS + `label`.

    height_norm=True : Height = Z - (lowest Z of the same patch). A tree can
                       then separate ground / vegetation / building by height
                       above ground instead of by absolute elevation, which
                       changes from one region of France to another.
    height_norm=False: Height = absolute Z.
    """
    df = df.withColumn("Z", F.col("xyz").getItem(2))

    if height_norm:
        # One row per patch (at most 80 000 rows): small enough to broadcast, so
        # the big table is joined without being shuffled.
        patch_min = df.groupBy("patch").agg(F.min("Z").alias("Z_min"))
        df = (df.join(F.broadcast(patch_min), on="patch", how="inner")
                .withColumn("Height", (F.col("Z") - F.col("Z_min")).cast("float")))
    else:
        df = df.withColumn("Height", F.col("Z").cast("float"))

    red, green, blue, nir = F.col("Red"), F.col("Green"), F.col("Blue"), F.col("Infrared")
    # Guard against a zero denominator (black pixels) instead of using a Python UDF,
    # which would be much slower on billions of points.
    ndvi = F.when((nir + red) == 0, F.lit(0.0)).otherwise((nir - red) / (nir + red))

    df = (df.withColumn("NDVI", ndvi.cast("float"))
            .withColumn("Brightness", ((red + green + blue) / 3).cast("float"))
            .withColumn("label", label_expression()))

    return df.select(*FEATURE_COLS, "label")


def materialize(df):
    """Keep the result in memory (spilling to disk if needed) and force its computation.

    Spark is lazy: without this, the cost of the preprocessing would be paid
    (and timed) inside the training stage. Returns (cached DataFrame, number of rows).
    """
    df = df.persist(StorageLevel.MEMORY_AND_DISK)
    n_rows = df.count()  # action: runs the whole preprocessing now
    return df, n_rows
