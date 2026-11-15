#!/usr/bin/env bash
# =============================================================================
# run_matrix.sh - run the full scalability experiment: every dataset fraction
# with every executor count, one after the other.
#
# Usage:
#   ./run_matrix.sh --output_dir s3://my-bucket/fractal-results [other run_experiment.sh options]
#
# Environment variables:
#   FRACTIONS   space-separated percentages     (default: "1 5")
#   EXECUTORS   space-separated executor counts (default: "60 40 32 20 8")
#   REPEATS     how many times each configuration is run (default: 1)
#
# A run that fails (e.g. out of memory with too few executors for a large
# fraction) is written in failed_runs.txt and the loop goes on.
#
# With REPEATS > 1 the whole matrix is run again, so the repeats of one
# configuration are spread over time. plot_speedup.py takes their median.
# =============================================================================
set -uo pipefail  # no -e: one failed run must not stop the whole matrix

FRACTIONS="${FRACTIONS:-1 5}"
EXECUTORS="${EXECUTORS:-60 40 32 20 8}"
REPEATS="${REPEATS:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# one run_experiment.sh call per (repeat, fraction, executors)
for repeat in $(seq 1 "$REPEATS"); do
  for fraction in $FRACTIONS; do
    for n_exec in $EXECUTORS; do
      echo "=============== repeat=${repeat}  fraction=${fraction}%  executors=${n_exec} ==============="
      if ! "$SCRIPT_DIR/run_experiment.sh" --fraction "$fraction" --n_executors "$n_exec" --repeat "$repeat" "$@"; then
        echo "repeat=${repeat} fraction=${fraction} executors=${n_exec}" >> failed_runs.txt
        echo "!! run failed, continuing with the next one"
      fi
    done
  done
done
