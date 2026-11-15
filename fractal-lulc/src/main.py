"""
main.py
-------
Entry point of the FRACTAL land-cover classification pipeline (PySpark / MLlib).

Three stages are timed separately, as required for the scalability study:
    1. preprocessing : file listing, reading, feature computation (train and test)
    2. training      : fitting the model on the training features
    3. evaluation    : predictions on the test set + confusion matrix + metrics

One run writes one JSON file (parameters, stage times, metrics) to --output_dir.
plot_speedup.py then turns a folder of such files into the speedup curves.

Normally launched through run_experiment.sh, which sets the executor setup.
"""
import argparse
import datetime
import json

from data_io import load_split
from evaluation import evaluate
from spark_metrics import SparkMetrics
from preprocessing import CLASS_NAMES, FEATURE_COLS, materialize, preprocess
from training import feature_importances, fit_model
from utils import StageTimer, build_spark, cluster_settings, registered_executors, write_text

DATA_ROOT = "s3a://ubs-datasets/FRACTAL/data"


def parse_args():
    """Command line options (run_experiment.sh passes here everything it does not use itself)."""
    p = argparse.ArgumentParser(description="FRACTAL land-cover classification with Spark MLlib")

    # --- data ---
    p.add_argument("--train_path", default=f"{DATA_ROOT}/train/", help="folder with the training .parquet files")
    p.add_argument("--val_path", default=f"{DATA_ROOT}/val/", help="folder with the validation .parquet files")
    p.add_argument("--test_path", default=f"{DATA_ROOT}/test/", help="folder with the test .parquet files")
    p.add_argument("--fraction", type=float, default=1.0,
                   help="percentage of the files (patches) of each split to use, e.g. 1, 5, 25, 100")
    p.add_argument("--eval_fraction", type=float, default=None,
                   help="percentage of val/test files; default: same as --fraction. Set to 100 to always "
                        "evaluate on the full test set")
    p.add_argument("--eval_val", action="store_true", help="also evaluate on the validation split")
    p.add_argument("--seed", type=int, default=42, help="seed for file sampling and for the model")

    # --- features ---
    p.add_argument("--no_height_norm", action="store_true",
                   help="use the absolute Z instead of the height above the lowest point of the patch")

    # --- model ---
    p.add_argument("--model", choices=["rf", "dt"], default="rf", help="random forest or decision tree")
    p.add_argument("--num_trees", type=int, default=30, help="number of trees (rf only)")
    p.add_argument("--max_depth", type=int, default=10)
    p.add_argument("--max_bins", type=int, default=32)
    p.add_argument("--subsampling_rate", type=float, default=1.0,
                   help="fraction of the training rows used by each tree (rf only); lowering it saves time")
    p.add_argument("--use_class_weights", action="store_true", help="weight rows by sqrt inverse class frequency")

    # --- outputs ---
    p.add_argument("--output_dir", default="results",
                   help="folder for the results JSON (local, hdfs:// or s3:// path). On a cluster use an S3 path")
    p.add_argument("--repeat", type=int, default=1,
                   help="repeat number of this configuration; part of the default run name, so repeated "
                        "runs do not overwrite each other's result file")
    p.add_argument("--run_name", default=None,
                   help="name of the run; default e<executors>_f<fraction>_r<repeat>")
    p.add_argument("--no_spark_metrics", action="store_true",
                   help="do not collect the sparkMeasure task metrics (stored under spark_metrics in the results)")
    p.add_argument("--save_model", action="store_true", help="also save the fitted model under <output_dir>/models/")
    return p.parse_args()


def main():
    """Run the three timed stages and write the results JSON."""
    args = parse_args()
    # by default val/test use the same fraction as train
    eval_fraction = args.eval_fraction if args.eval_fraction is not None else args.fraction

    spark = build_spark("fractal-lulc")
    settings = cluster_settings(spark)
    # e.g. e32_f10_r1 = 32 executors, 10 % of the data, repeat 1
    run_name = args.run_name or f"e{settings['n_executors'] or 'local'}_f{args.fraction:g}_r{args.repeat}"
    print(f"=== run {run_name} | cluster settings: {settings} ===", flush=True)

    timer = StageTimer()
    metrics = SparkMetrics(spark, enabled=not args.no_spark_metrics)  # task metrics per stage (sparkMeasure)
    data_info = {}  # file and point counts, stored in the results

    # Executors really given by YARN, checked at the start and after each stage (we can get fewer than
    # requested). plot_speedup.py uses the smallest value as the x coordinate of the run.
    granted = {}

    def snapshot(label):
        # save and print the number of executors at this point of the run
        granted[label] = registered_executors(spark)
        print(f"executors granted ({label}): {granted[label]} (requested: {settings['n_executors']})", flush=True)

    snapshot("start")

    with timer.stage("total"):
        # ------------------------------------------------------------------
        # Stage 1: preprocessing (train + test, and val if requested)
        # ------------------------------------------------------------------
        with metrics.stage("preprocessing"), timer.stage("preprocessing"):
            with timer.stage("file_listing"):  # driver-only work, kept visible as a sub-step
                train_raw, data_info["train_files"] = load_split(spark, args.train_path, args.fraction, args.seed)
                test_raw, data_info["test_files"] = load_split(spark, args.test_path, eval_fraction, args.seed)
                if args.eval_val:
                    val_raw, data_info["val_files"] = load_split(spark, args.val_path, eval_fraction, args.seed)

            # Number of input partitions = number of tasks of the first stage. Saved because it helps to
            # explain the scaling results (the slides advise 128-256 MB per partition).
            data_info["train_input_partitions"] = train_raw.rdd.getNumPartitions()
            data_info["test_input_partitions"] = test_raw.rdd.getNumPartitions()
            if args.eval_val:
                data_info["val_input_partitions"] = val_raw.rdd.getNumPartitions()
            partitions = {k: v for k, v in data_info.items() if k.endswith("_input_partitions")}
            print(f"input partitions: {partitions}", flush=True)

            # features + label, cached (materialize ends with a count, so the work is done in this stage)
            height_norm = not args.no_height_norm
            train_df, data_info["train_points"] = materialize(preprocess(train_raw, height_norm))
            test_df, data_info["test_points"] = materialize(preprocess(test_raw, height_norm))
            if args.eval_val:
                val_df, data_info["val_points"] = materialize(preprocess(val_raw, height_norm))

        snapshot("after_preprocessing")

        # ------------------------------------------------------------------
        # Stage 2: training
        # ------------------------------------------------------------------
        with metrics.stage("training"), timer.stage("training"):
            model, class_weights = fit_model(train_df, args, FEATURE_COLS, len(CLASS_NAMES))

        snapshot("after_training")

        # ------------------------------------------------------------------
        # Stage 3: evaluation
        # ------------------------------------------------------------------
        metrics_by_split = {}
        with metrics.stage("evaluation"), timer.stage("evaluation"):
            metrics_by_split["test"] = evaluate(model, test_df, CLASS_NAMES)
            if args.eval_val:
                metrics_by_split["val"] = evaluate(model, val_df, CLASS_NAMES)

    snapshot("end")
    # total = sum of the three stages, so the time spent reading the sparkMeasure metrics is not counted
    timer.seconds["total"] = sum(timer.seconds[s] for s in ("preprocessing", "training", "evaluation"))

    # ----------------------------------------------------------------------
    # Results: one JSON per run
    # ----------------------------------------------------------------------
    result = {
        "run_name": run_name,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "params": {**vars(args), "eval_fraction": eval_fraction},
        "cluster": {**settings, "executors_granted": granted},
        "data": data_info,
        "timings_sec": timer.seconds,
        "class_weights": class_weights,
        "feature_importances": feature_importances(model, FEATURE_COLS),
        "metrics": metrics_by_split,
        "spark_metrics": metrics.summary,
    }
    result_json = json.dumps(result, indent=2)
    write_text(spark, f"{args.output_dir.rstrip('/')}/{run_name}.json", result_json)
    # also printed on one line, so the result can be found in the YARN logs if writing to S3 fails
    print("RESULT_JSON " + json.dumps(result), flush=True)

    if args.save_model:  # optional, not used for the experiments
        model.write().overwrite().save(f"{args.output_dir.rstrip('/')}/models/{run_name}")

    spark.stop()


if __name__ == "__main__":
    main()
