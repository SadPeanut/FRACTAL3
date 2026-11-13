# FRACTAL land-cover classification with PySpark / MLlib

Classifies the points of the FRACTAL aerial-lidar dataset into 7 classes (other, ground,
vegetation, building, water, bridge, structure) with a Spark MLlib tree model, and measures
how the pipeline scales with the number of executors.

Three stages are timed separately, as required:

| Stage | What it does |
|---|---|
| preprocessing | lists and reads the sampled files, computes features and labels, caches train and test |
| training | fits the model (random forest by default) |
| evaluation | predicts on the test set, builds the confusion matrix, computes OA, per-class IoU, mIoU |

```
run_experiment.sh      one run: dataset fraction + executor setup as parameters
run_matrix.sh          all fractions x executor counts (the scalability experiment)
src/main.py            entry point (argument parsing, the 3 timed stages, results JSON)
src/data_io.py         file listing and seeded sampling
src/preprocessing.py   class mapping, features
src/training.py        model + optional class weights
src/evaluation.py      confusion matrix -> OA / IoU / mIoU
src/utils.py           Spark session, stage timer, file writer
analysis/plot_speedup.py   speedup figures + summary.csv from the results JSONs (no Spark needed)
analysis/diagnose.py       partitions per core, idle cores, skew, spill per stage, from the sparkMeasure metrics
src/spark_metrics.py       sparkMeasure task metrics per stage
```

## Requirements

- On the cluster (AWS EMR): Spark 3.x with PySpark, Python 3 with `numpy` (installed with PySpark ML).
- Access to the data: `s3a://ubs-datasets/FRACTAL/data/{train,val,test}/` (the defaults).
- For the plots, on any machine: Python 3 with `matplotlib`.

- sparkMeasure (task metrics): `run_experiment.sh` adds the jar with `--packages ch.cern.sparkmeasure:spark-measure_2.12:0.27`,
  so the cluster must be able to reach Maven. If it cannot, run with `SPARK_MEASURE_PACKAGE=""` (the results then have no
  `spark_metrics`); the Python package `sparkmeasure` is not needed.

If the cluster does not already ship the S3A connector, pass it to spark-submit:
`EXTRA_PACKAGES="org.apache.hadoop:hadoop-aws:3.3.1" ./run_experiment.sh ...`

## Run one experiment

```bash
./run_experiment.sh --fraction 10 --n_executors 16 --output_dir s3://<your-bucket>/fractal-results
```

Always launch through `run_experiment.sh`, not `spark-submit src/main.py` directly: the executor setup and the
Spark settings listed below are applied there (dynamic allocation, for instance, is on by default on EMR and
would distort the speedup measurements).

Options of `run_experiment.sh` (become spark-submit settings):

| Option | Default | Meaning |
|---|---|---|
| `--fraction` | 1 | % of the files (patches) of each split |
| `--n_executors` | 8 | number of executors (dynamic allocation is switched off) |
| `--executor_cores` | 3 | cores per executor (Spark Tuning p.28: 2 executors per 8-vCPU node) |
| `--executor_memory` | 20g | heap memory per executor (p.28: (54 GB - 10 % - 6 GB) / 2 = 21 GB max) |
| `--memory_overhead` | 5g | off-heap memory per executor (the Spark Tuning slides use 5G) |
| `--driver_memory` | 6g | driver memory |
| `--driver_overhead` | 2g | off-heap memory for the driver |
| `--deploy_mode` | cluster | `cluster` or `client` |

Spark settings applied to every run (from the course's "Spark Tuning" slides, p.28-31), so the results do
not depend on the cluster's defaults:

| Setting | Value |
|---|---|
| `spark.dynamicAllocation.enabled` | `false` (cluster only) |
| `spark.speculation` | `true` (cluster only) |
| `spark.scheduler.minRegisteredResourcesRatio` | `1.0`, wait up to 120 s: start only when all executors are up (cluster only) |
| `spark.driver.maxResultSize` | `2g` |
| `spark.serializer` | `KryoSerializer` |
| `spark.sql.files.maxPartitionBytes` | 256 MB |
| `spark.sql.adaptive.enabled` | `true` |
| `spark.sql.adaptive.advisoryPartitionSizeInBytes` | 128 MB |
| `spark.sql.shuffle.partitions` | 4 x total executor cores (set in `src/utils.py`) |
| `spark.sql.files.minPartitionNum` | 3 x total executor cores (set in `src/utils.py`) |

To change one, pass the same key in `EXTRA_SPARK_ARGS`; the last `--conf` wins, e.g.
`EXTRA_SPARK_ARGS="--conf spark.speculation=false" ./run_experiment.sh ...`

Any other option is passed to `src/main.py` (`python src/main.py --help`):

| Option | Default | Meaning |
|---|---|---|
| `--output_dir` | `results` | folder for the results JSON; on a cluster give an S3 path |
| `--train_path`, `--val_path`, `--test_path` | FRACTAL S3 folders | input folders of Parquet files |
| `--eval_fraction` | same as `--fraction` | % of val/test files; use 100 for a fixed, full test set |
| `--eval_val` | off | also evaluate on the validation split |
| `--model` | `rf` | `rf` (random forest) or `dt` (decision tree) |
| `--num_trees`, `--max_depth`, `--max_bins` | 30, 10, 32 | model hyperparameters (not tuned; kept moderate so that runs on 50-100 % of the data stay affordable) |
| `--subsampling_rate` | 1.0 | share of rows used by each tree; lower it to cut training time |
| `--use_class_weights` | off | weight rows by sqrt inverse class frequency |
| `--no_spark_metrics` | off | skip the sparkMeasure task metrics |
| `--no_height_norm` | off | use absolute Z instead of height above the patch minimum |
| `--seed` | 42 | seed for file sampling and the model |
| `--repeat` | 1 | repeat number of this configuration (see the scalability study below) |
| `--run_name` | `e<N>_f<fraction>_r<repeat>` | name of the run / of its JSON file |
| `--save_model` | off | save the fitted model under `<output_dir>/models/` |

Larger fractions on a bigger cluster, e.g. 100 % on 64 executors:
`./run_experiment.sh --fraction 100 --n_executors 64 --output_dir s3://...`

Logs: in cluster mode the driver output is in the YARN logs (`yarn logs -applicationId <id>`).
The full result is also printed on one line starting with `RESULT_JSON`.

In cluster mode, interrupting `run_experiment.sh` or `run_matrix.sh` does **not** stop a job that was already
submitted to YARN. The cluster is shared, so list and stop your own jobs with:

```bash
yarn application -list
yarn application -kill <application_id>
```

## Reproduce the scalability study

```bash
# 1. all fractions x executor counts, repeated 3 times (edit the lists with environment variables)
FRACTIONS="1 2 5 10 25" EXECUTORS="4 8 12 16 20" REPEATS=3 ./run_matrix.sh --output_dir s3://<your-bucket>/fractal-results

# 2. get the results and build the figures
aws s3 sync s3://<your-bucket>/fractal-results ./results
python analysis/plot_speedup.py --results_dir ./results --out_dir ./figures

# 3. why a curve bends: partitions per core, idle cores, stragglers, spill (from the sparkMeasure metrics)
python analysis/diagnose.py --results_dir ./results
```

Big fractions need a larger smallest executor count (a 25 % run on 4 executors takes hours): run them separately, e.g.
`FRACTIONS=25 EXECUTORS="8 12 16 20" ./run_matrix.sh ...`; n0 is then 8 for that curve.

`run_matrix.sh` keeps going when a run fails (usually out of memory with too few executors for a large
fraction) and lists it in `failed_runs.txt`. The smallest executor count that succeeds for a fraction is
its n0. The plots show, for each stage, T(n0)/T(n) against the number of executors, one black-and-white
curve per fraction, plus the ideal line n/n0 (drawn from the smallest n0 over all fractions).
`figures/summary.csv` lists times and quality for every run. A fourth figure, `speedup_preprocessing_excl_listing`,
leaves out the driver-only file listing (about 26 s whatever the fraction), which executors cannot shorten.

The timings on a shared cluster are noisy (other users' jobs compete for the same resources), so each
configuration should be repeated (`REPEATS`). The whole matrix is run once and then again, so the repeats of
one configuration are spread over time. Each repeat has its own result file
(`e<N>_f<fraction>_r<k>.json`) and the plots use the median time of the repeats. Running the same repeat
number again overwrites its file.

## Method

- **Sampling.** Each Parquet file is one 50 x 50 m patch. A fraction selects that share of the files of each
  split with a fixed seed. The files are shuffled once and the first k are kept, so samples are nested
  (the 1 % sample is part of the 2 % sample, etc.).
- **Labels.** Raw LAS codes 1 -> other, 2 -> ground, 3/4/5 -> vegetation, 6 -> building, 9 -> water,
  17 -> bridge, 64 -> structure; everything else (65, 66...) -> other. Checked against the class shares
  of the FRACTAL dataset card.
- **Features.** Height above the lowest point of the patch, intensity, return number, number of returns,
  R, G, B, infrared, NDVI, brightness. Absolute X/Y are not used because the split is geographic.
- **Timing.** Spark is lazy, so preprocessing results are cached and forced with a `count()` inside the
  preprocessing timer, and evaluation ends with the `collect()` of the confusion matrix. The file listing
  (driver only) is included in preprocessing and also reported on its own (`file_listing`).
- **Metrics.** OA, per-class IoU = TP/(TP+FP+FN) and mIoU come from the confusion matrix. A class absent from
  both reference and prediction (possible on tiny samples) has no IoU and is left out of the mIoU
  (`n_classes_in_mIoU` says how many were averaged).

## Results file

One JSON per run in `--output_dir`: parameters, executor setup (requested, and `executors_granted` as counted at the start and after each stage), file, point and input-partition counts, stage timings
(`timings_sec`), sparkMeasure task metrics per stage (`spark_metrics`), feature importances, and for each evaluated split OA, mIoU, per-class IoU, support and the
confusion matrix.

## Local smoke test (no cluster, no S3)

```bash
MASTER='local[4]' ./run_experiment.sh --fraction 100 \
  --train_path file:///abs/path/train --val_path file:///abs/path/val --test_path file:///abs/path/test \
  --eval_val --output_dir file:///tmp/fractal-out
```

Use `file://` with absolute paths: on EMR a plain path would point to HDFS.

### Checking partitions, skew and memory on a laptop

Local mode is one JVM, so it shows how the work is split, not how the cluster scales (no S3, no network, no
other users). Use a few hundred patches, an explicit `local[N]` master (not `local[*]`) and a distinct run name:

```bash
for N in 2 4 8; do
  MASTER="local[$N]" ./run_experiment.sh --fraction 100 --driver_memory 12g --run_name local${N}_f \
    --train_path file:///abs/path/train --test_path file:///abs/path/test --output_dir file:///abs/path/out
done
python analysis/diagnose.py --results_dir /abs/path/out
```

`diagnose.py` reads the cores from `local[N]`. Leave `SPARK_MEASURE_PACKAGE` at its default to also test the
sparkMeasure jar (needs internet for Maven); without `spark_metrics` the diagnosis has nothing to show.
