# FRACTAL land-cover classification with PySpark / MLlib

A random forest (MLlib) classifies the points of the FRACTAL lidar dataset into 7 classes. The three stages
(preprocessing, training, evaluation) are timed separately to measure how the pipeline scales with the
number of executors.

```
run_experiment.sh          one run: dataset fraction + executor setup as parameters
run_matrix.sh              several fractions x executor counts (calls run_experiment.sh)
src/                       the pipeline (src/main.py is the entry point)
analysis/plot_speedup.py   speedup figures + summary.csv from the result JSON files
analysis/diagnose.py       idle cores, skew and spill per stage, from the sparkMeasure metrics
```

## Requirements

- An AWS EMR cluster with Spark (tested on Spark 4.1.1 / Scala 2.13), with read access to
  `s3a://ubs-datasets/FRACTAL/data/` and an S3 folder you can write to.
- `run_experiment.sh` adds sparkMeasure from Maven (`spark-measure_2.13`). On Spark 3 (Scala 2.12), set
  `SPARK_MEASURE_PACKAGE=ch.cern.sparkmeasure:spark-measure_2.12:0.27`. Without it, the run still works but has
  no task metrics.
- For the figures: Python 3 with `matplotlib`, on any machine.

## Run one experiment

On the cluster master node, from this folder:

```bash
./run_experiment.sh --fraction 10 --n_executors 32 --output_dir s3://<your-bucket>/fractal-results
```

| Option | Default | Meaning |
|---|---|---|
| `--fraction` | 1 | % of the files (patches) of each split |
| `--n_executors` | 8 | number of executors |
| `--executor_cores` | 3 | cores per executor |
| `--executor_memory` | 20g | heap memory per executor |
| `--memory_overhead` | 5g | off-heap memory per executor |
| `--driver_memory` / `--driver_overhead` | 6g / 2g | driver memory |
| `--output_dir` | `results` | where the result JSON is written (give an S3 path) |

Each run writes one JSON file (`e<N>_f<fraction>_r1.json`) with the time of each stage, the quality metrics
(OA, IoU per class, mIoU) and the task metrics. Other options go to `src/main.py` (`python src/main.py --help`).

## 50 % and 100 % on a larger cluster

```bash
./run_experiment.sh --fraction 50  --n_executors 60 --output_dir s3://<your-bucket>/fractal-results
./run_experiment.sh --fraction 100 --n_executors 60 --output_dir s3://<your-bucket>/fractal-results
```

- With the default sizing (25 GB per executor), a 54 GB node holds two executors and the node that runs the
  driver holds one, so at most 2 x nodes - 1 executors (63 on 32 nodes).
- Use at least 32 executors. Expected time on 32 nodes with 60 executors, extrapolated from our 25 % run:
  about 30 min for 50 % and about 1 h for 100 %.

## Reproduce our results

```bash
FRACTIONS="1 5 10" EXECUTORS="8 20 32 40 60" ./run_matrix.sh --output_dir s3://<your-bucket>/fractal-results
FRACTIONS=25 EXECUTORS="40 60" ./run_matrix.sh --output_dir s3://<your-bucket>/fractal-results
aws s3 sync s3://<your-bucket>/fractal-results ./results
python analysis/plot_speedup.py --results_dir ./results --out_dir ./figures
```

Stopping the script does not stop a job already submitted to YARN: `yarn application -kill <application_id>`.
