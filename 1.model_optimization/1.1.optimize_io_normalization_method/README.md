# This folder contains code/notebook for running optimziation trials on the normalization method used for input and target images working with FNet models.

This optimization analysis (and other optimizations in general) is dependent on output from running the notebooks under 0.data_preprocessing

For the analysis in this folder to work, the config.yml file at the repo home will need to be configured with the correct pathes to the pediatric_cancer_atlas_profiling repo (`pediatric_cancer_atlas_profiling_path`), the location where the single cell level features parquet files (`sc_features_path`) as well as the location where the virtual_stain_flow_software (dev-0.1) branch clone lives (`software_path`) as the software is currently in active development and is not installable. 

## The FNet input/output(target) normalization optimization trails can be run by running the following notebook:
1.1.optimize_io_normalization_method/1.1.1.optimize_fnet_io_norm.ipynb

## ALternatively, run
```bash
# Ensure config.yml at repo root is properly configured
source io_norm_optimization.sh
```