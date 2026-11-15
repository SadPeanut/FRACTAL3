"""
training.py
-----------
Model definition and fitting (Spark MLlib).
"""
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F


def add_class_weights(df, n_classes):
    """Add a `weight` column to counter class imbalance.

    weight(class) = sqrt( total / (n_classes * count(class)) ). Plain inverse
    frequency would give the rarest class (0.04 % of the points) a weight in the
    thousands and destabilise the trees, so the square root is used.
    Returns (DataFrame with `weight`, dict label -> weight).
    """
    # One small aggregation over the (cached) training data
    counts = {int(r["label"]): r["count"] for r in df.groupBy("label").count().collect()}
    total = sum(counts.values())
    weights = {c: (total / (n_classes * n)) ** 0.5 for c, n in counts.items()}
    # label -> weight lookup as a Spark map literal
    lookup = F.create_map(*[x for c, w in weights.items() for x in (F.lit(float(c)), F.lit(float(w)))])
    return df.withColumn("weight", lookup[F.col("label")]), weights


def fit_model(train_df, args, feature_cols, n_classes):
    """Fit assembler + classifier as one MLlib Pipeline.

    Returns (PipelineModel, class weights dict or None).
    """
    # Packs the feature columns into the single vector column MLlib expects
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    common = dict(featuresCol="features", labelCol="label", maxDepth=args.max_depth,
                  maxBins=args.max_bins, seed=args.seed)
    if args.model == "rf":
        classifier = RandomForestClassifier(numTrees=args.num_trees,
                                            subsamplingRate=args.subsampling_rate, **common)
    else:  # "dt"
        classifier = DecisionTreeClassifier(**common)

    weights = None
    if args.use_class_weights:
        train_df, weights = add_class_weights(train_df, n_classes)
        classifier.setWeightCol("weight")

    model = Pipeline(stages=[assembler, classifier]).fit(train_df)
    return model, weights


def feature_importances(model, feature_cols):
    """Feature importance of the fitted tree model as {feature name: importance}."""
    values = model.stages[-1].featureImportances.toArray().tolist()
    return dict(zip(feature_cols, values))
