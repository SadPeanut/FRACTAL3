#!/usr/bin/env bash
# =============================================================================
# run_matrix.sh - run the full scalability experiment: every dataset fraction
# with every executor count, one after the other.
#
# Usage:
#   ./run_matrix.sh --output_dir s3://my-bucket/fractal-results [other run_experiment.sh options]
#
# Environment variables:
#   FRACTIONS   space-separated percentages   (default: "1 2 5 10 25")
#   EXECUTORS   space-separated executor counts (default: "4 8 12 16 20")
#   REPEATS     how many times each configuration is run (default: 1)
#
# A run that fails (typically out of memory with too few executors for a large
# fraction) is recorded in failed_runs.txt and the loop continues. The smallest
# executor count that succeeds for a fraction is its n0 for the speedup curves.
#
# On a shared cluster the timings are noisy, so use REPEATS=3 or more: the whole
# matrix is run once, then again, so the repeats of one configuration are spread
# over time (not back to back). plot_speedup.py uses the median of the repeats.
# =============================================================================
set -uo pipefail  # no -e: one failed run must not stop the whole matrix

FRACTIONS="${FRACTIONS:-1 5}"
EXECUTORS="${EXECUTORS:-60 40 32 20 8}"
REPEATS="${REPEATS:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
