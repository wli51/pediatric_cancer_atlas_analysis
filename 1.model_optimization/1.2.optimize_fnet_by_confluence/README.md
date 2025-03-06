# This folder contains code for optimziation on the basic training hyperparameters for FNet on two separate datasets: the high confluence U2-OS and the low confluence U2-OS

This optimization analysis (and other optimizations in general) is dependent on output from running the notebooks under 0.data_preprocessing

For the analysis in this folder to work, the config.yml file at the repo home will need to be configured with the correct pathes to the pediatric_cancer_atlas_profiling repo (`pediatric_cancer_atlas_profiling_path`), the location where the single cell level features parquet files (`sc_features_path`) as well as the location where the virtual_stain_flow_software (dev-0.1) branch clone lives (`software_path`) as the software is currently in active development and is not installable.

## The FNet training hyper-parameter optimization can performed by sequentially running the following notebooks, where the first notebook runs the optuna optimization trials to produce optuna joblib files and the second notebook performs evaluation on the train and heldout U2-OS datasets and produces plots:
1.2.1.optimize_fnet.ipynb
1.2.2.validate_fnet_optimization.ipynb

## Alternatively, run
```bash
# Ensure config.yml at repo root is properly configured
source optimize_fnet.sh
```