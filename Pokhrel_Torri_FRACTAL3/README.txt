Report.pdf           final report
LateX/               LaTeX source of the report (compile: latexmk -pdf Report.tex)
fractal-lulc/        the PySpark pipeline, see fractal-lulc/README.md.
                     To run 50 % or 100 % of the data on the cluster (about 30 min / 1 h on 60 executors):
                       cd fractal-lulc
                       ./run_experiment.sh --fraction 100 --n_executors 60 --output_dir s3://<your-bucket>/fractal-results
