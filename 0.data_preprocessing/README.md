# This folder contains the notebooks to preprocess and divide data for use with training and evaluation. 

This entire data pre-processing step (and the repo in general) will be dependent on a local pediatric_cancer_atlas_profiling repo (https://github.com/WayScience/pediatric_cancer_atlas_profiling) that is ran up to 2.feature_extraction. 

The config.yml file will NEED to be configured with the correct path to the pediatric_cancer_atlas_profiling repo for every notebook in this folder to work.

## Data preprocessing workflow can be performed by sequentially running the following notebooks:
0.1.filter_low_quality_sites.ipynb
0.2.data_split.ipynb

## Alternatively, run
```bash
# Ensure config.yml at repo root is properly configured
source preprocessing.sh
```