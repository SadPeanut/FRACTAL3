#!/usr/bin/env bash
# =============================================================================
# run_experiment.sh - run the FRACTAL pipeline once, with a chosen dataset
# fraction and executor setup.
#
# Usage:
#   ./run_experiment.sh --fraction 10 --n_executors 16 --output_dir s3://my-bucket/fractal-results
#
# Options handled here (they become spark-submit settings):
#   --fraction          % of the dataset files to use              (default 1)
#   --n_executors       number of executors                        (default 8)
#   --executor_cores    cores per executor                         (default 3)
#   --executor_memory   heap memory per executor                   (default 20g)
#   --memory_overhead   off-heap memory per executor               (default 5g)
#   --driver_memory     driver memory                              (default 6g)
#   --driver_overhead   off-heap memory for the driver             (default 2g)
#   --deploy_mode       cluster | client                           (default cluster)
# Any other option (--model, --num_trees, --output_dir, --use_class_weights...)
# is passed unchanged to src/main.py; see `python src/main.py --help`.
#
# The Spark settings of the course's "Spark Tuning" slides (p.28-31) are applied
# to every run (see SPARK_TUNING_ARGS below), so results do not depend on the
# cluster's defaults.
#
# Environment variables:
#   MASTER            spark master (default: yarn). Use local[4] for a local test.
#   EXTRA_SPARK_ARGS  extra spark-submit arguments. A --conf given here overrides the same
#                     key set by this script.
#   SPARK_MEASURE_PACKAGE  Maven coordinates of sparkMeasure, added with --packages so that the
#                     task metrics are collected (default ch.cern.sparkmeasure:spark-measure_2.12:0.27).
#                     Set it to "" if the cluster cannot reach Maven; the run then has no spark_metrics.
#   EXTRA_PACKAGES    more Maven packages, comma separated, e.g. "org.apache.hadoop:hadoop-aws:3.3.1"
#                     on a cluster that does not already ship the S3A connector.
# =============================================================================
set -euo pipefail

# ---- defaults ----
FRACTION=1
N_EXECUTORS=8
EXECUTOR_CORES=3      # slide p.28: 3 cores -> 2 executors per 8-vCPU node, 2 vCPU left for OS/YARN/HDFS
EXECUTOR_MEMORY=20g  # slide p.28: (54 GB - 10 % - 6 GB) / 2 = 21 GB max per executor
MEMORY_OVERHEAD=5g   # slide p.28; covers Python/Arrow buffers on top of the JVM heap
DRIVER_MEMORY=6g
DRIVER_OVERHEAD=2g
DEPLOY_MODE=cluster
MASTER="${MASTER:-yarn}"
EXTRA_SPARK_ARGS="${EXTRA_SPARK_ARGS:-}"
SPARK_MEASURE_PACKAGE="${SPARK_MEASURE_PACKAGE-ch.cern.sparkmeasure:spark-measure_2.13:0.27}"
EXTRA_PACKAGES="${EXTRA_PACKAGES:-}"
PIPELINE_ARGS=()  # everything this script does not recognise goes to main.py

# print the comment block at the top of this file (stops at its closing "# ====" line)
usage() { awk 'NR>=2 {print} NR>2 && /^# =+$/ {exit}' "$0"; }

# ---- parse the command line ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fraction)        FRACTION="$2"; shift 2 ;;
    --n_executors)     N_EXECUTORS="$2"; shift 2 ;;
    --executor_cores)  EXECUTOR_CORES="$2"; shift 2 ;;
    --executor_memory) EXECUTOR_MEMORY="$2"; shift 2 ;;
    --memory_overhead) MEMORY_OVERHEAD="$2"; shift 2 ;;
    --driver_memory)   DRIVER_MEMORY="$2"; shift 2 ;;
    --driver_overhead) DRIVER_OVERHEAD="$2"; shift 2 ;;
    --deploy_mode)     DEPLOY_MODE="$2"; shift 2 ;;
    -h|--help)         usage; exit 0 ;;
    *)                 PIPELINE_ARGS+=("$1"); shift ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
mkdir -p "$BUILD_DIR"

# ---- package the helper modules so they are shipped to the driver and executors ----
(cd "$SCRIPT_DIR/src" && python3 -m zipfile -c "$BUILD_DIR/deps.zip" \
    utils.py data_io.py preprocessing.py training.py evaluation.py spark_metrics.py)

# ---- resources: a fixed number of executors, no dynamic allocation ----
# (dynamic allocation is on by default on EMR and would make the speedup
# measurements meaningless)
SPARK_RESOURCE_ARGS=()
if [[ "$MASTER" == local* ]]; then
  DEPLOY_MODE=client  # cluster deploy mode does not exist for a local master
else
  SPARK_RESOURCE_ARGS=(
    --num-executors "$N_EXECUTORS"
    --executor-cores "$EXECUTOR_CORES"
    --executor-memory "$EXECUTOR_MEMORY"
    --conf "spark.executor.memoryOverhead=$MEMORY_OVERHEAD"
    --conf "spark.driver.memoryOverhead=$DRIVER_OVERHEAD"
    --conf "spark.dynamicAllocation.enabled=false"
    --conf "spark.speculation=true"  # re-run straggler tasks elsewhere (slide p.30)
    # Start the job only when every requested executor is registered (the default on YARN is to start
    # at 80 % after 30 s), otherwise the first stage runs on fewer executors than the run claims.
    --conf "spark.scheduler.minRegisteredResourcesRatio=1.0"
    --conf "spark.scheduler.maxRegisteredResourcesWaitingTime=120s"
  )
fi

# ---- Spark settings from the "Spark Tuning" slides (p.29-31), applied to every run ----
SPARK_TUNING_ARGS=(
  --conf "spark.driver.maxResultSize=2g"  # cap on the results returned to the driver
  --conf "spark.serializer=org.apache.spark.serializer.KryoSerializer"  # faster than Java serialization
  --conf "spark.sql.files.maxPartitionBytes=268435456"  # at most 256 MB per input partition
  --conf "spark.sql.adaptive.enabled=true"  # adaptive query execution
  --conf "spark.sql.adaptive.advisoryPartitionSizeInBytes=134217728"  # target 128 MB partitions
)

# one --packages option holding every package (a second --packages would replace the first)
PACKAGES="${SPARK_MEASURE_PACKAGE}${SPARK_MEASURE_PACKAGE:+${EXTRA_PACKAGES:+,}}${EXTRA_PACKAGES}"
if [[ -n "$PACKAGES" ]]; then
  SPARK_TUNING_ARGS+=(--packages "$PACKAGES")
fi

echo "Run: fraction=${FRACTION}% | executors=${N_EXECUTORS} x ${EXECUTOR_CORES} cores x ${EXECUTOR_MEMORY} | master=${MASTER} (${DEPLOY_MODE})"

# EXTRA_SPARK_ARGS is left unquoted on purpose so it splits into separate arguments.
# The ${arr[@]+...} form avoids an "unbound variable" error on old bash when an array is empty.
# shellcheck disable=SC2086
spark-submit \
  --master "$MASTER" \
  --deploy-mode "$DEPLOY_MODE" \
  --driver-memory "$DRIVER_MEMORY" \
  ${SPARK_RESOURCE_ARGS[@]+"${SPARK_RESOURCE_ARGS[@]}"} \
  "${SPARK_TUNING_ARGS[@]}" \
  $EXTRA_SPARK_ARGS \
  --py-files "$BUILD_DIR/deps.zip" \
  "$SCRIPT_DIR/src/main.py" \
  --fraction "$FRACTION" \
  ${PIPELINE_ARGS[@]+"${PIPELINE_ARGS[@]}"}
