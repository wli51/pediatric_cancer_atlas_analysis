#!/usr/bin/env python
# coding: utf-8

# # This notebook generate data splits for training, heldout and evaluation dataset based on cell line, plate and seeding density
# This entire data pre-processing step (and the repo in general) will be dependent on a local pediatric_cancer_atlas_profiling repo (https://github.com/WayScience/pediatric_cancer_atlas_profiling) that is ran up to 2.feature_extraction. The config.yml file will need to be configured with the correct path to the pediatric_cancer_atlas_profiling repo for this notebook to work.
# 
# This notebook relies the loaddata csv file generated from the pediatric_cancer_atlas_profiling to index images and the QC output from 0.1.filter_low_quality_sites.ipynb in this repo. It will take the loaddata csv file, remove sites marked for exclusion, and divide the loaddata csv into 3 csv files for train, heldout and evaluation dataset. 
# Specifically, the U2-OS cell line on plate1 of all cell plating densities will be selected as the trianing set and one random well per seedinng density will be saved as heldout, every thing else (different cell lines across 2 plates and U2-OS on plate 2 will be used as evaluation dataset to compare differential model performance).

# In[1]:


import pathlib
import yaml

import numpy as np
import pandas as pd


# ## Read config

# In[2]:


with open(pathlib.Path('.').absolute().parent / "config.yml", "r") as file:
    config = yaml.safe_load(file)


# ## Define paths to metadata, loaddata csvs and sc features

# In[3]:


## Access profiling repo path from config
PROFILING_DIR = pathlib.Path(config['paths']['pediatric_cancer_atlas_profiling_path'])

## Output path for the data split loaddata csvs generated in this notebook 
DATASPLIT_OUTPUT_DIR = pathlib.Path('.') / 'data_split_loaddata'
DATASPLIT_OUTPUT_DIR.mkdir(exist_ok=True)

## Path to platemap level metadata in pediatric_cancer_atlas_profiling repo
# this associates Platemap-cell_line-seeding_density-well information
platemap_csv_path = PROFILING_DIR \
    / "0.download_data" / "metadata" / "platemaps"
assert platemap_csv_path.exists()

## Path to loaddata csvs in pediatric_cancer_atlas_profiling repo
# this associates well with image_path
loaddata_csv_path = PROFILING_DIR \
    / "1.illumination_correction" / "loaddata_csvs"
assert loaddata_csv_path.exists()

## Path to QC excluded site csv file
qc_path = pathlib.Path('.').absolute() \
    / "preprocessing_output" / "qc_exclusion.csv"
assert qc_path.exists()


# ## Define columns that uniquely identifies well, condition (cell_line + seeding_density) and the train U2-OS condition

# In[1]:


## Whether to remove sites with low QC score
QC = True


## Wells are uniquely identified by the combination of these columns
## Define columns in loaddata
SITE_COLUMN = 'Metadata_Site'
WELL_COLUMN = 'Metadata_Well'
PLATE_COLUMN = 'Metadata_Plate'
UNIQUE_IDENTIFIERS = [SITE_COLUMN, WELL_COLUMN, PLATE_COLUMN]

## Condition for train and heldout data (every other condition will be left for evaluation)
TRAIN_CONDITION_KWARGS = {
    'cell_line': 'U2-OS',
    'platemap_file': 'Assay_Plate1_platemap', # plate 1 only
    'seeding_density': [1_000, 2_000, 4_000, 8_000, 12_000]
}

## Conditions are uniquely identified by the combination of keys from TRAIN_CONDITION_KWARGS
CONDITIONS = list(TRAIN_CONDITION_KWARGS.keys())


# ## Load all barcode/platemap metadata and all loaddata csv files and merge

# In[5]:


## Read platemap and well cell line metadata
barcode_df = pd.concat([pd.read_csv(f) for f in platemap_csv_path.glob('Barcode_*.csv')])

## Infers from barcode_df how many plates exist, retrieve all plate metadata and merge with barcode_df
platemap_df = pd.DataFrame()
for platemap in barcode_df['platemap_file'].unique():
    df = pd.read_csv(platemap_csv_path / f'{platemap}.csv')
    df['platemap_file'] = platemap
    platemap_df = pd.concat([platemap_df, df])    
barcode_platemap_df = pd.merge(barcode_df, platemap_df, on='platemap_file', how='inner')

## Read QC file
remove_sites = pd.read_csv(qc_path)

## Read loaddata csvs
loaddata_df = pd.concat(
    [pd.read_csv(f) for f in loaddata_csv_path.glob('*.csv')], 
    ignore_index=True)

## Merge loaddata with barcode/platemap metadata to map condition to well
loaddata_barcode_platemap_df = pd.merge(
    barcode_platemap_df.rename(columns={'barcode': PLATE_COLUMN, 'well': WELL_COLUMN}),
    loaddata_df,
    on=[PLATE_COLUMN, WELL_COLUMN], 
    how='left')

## Perform QC removal per site
if QC:
    print(f"{loaddata_barcode_platemap_df.shape[0]} sites prior to QC")
    # Merge to correctly identify rows to be removed
    qc_merge_df = loaddata_barcode_platemap_df.merge(
        remove_sites, 
        on=UNIQUE_IDENTIFIERS, 
        how='left', 
        indicator=True
        )

    # Keep only rows that were NOT found in remove_sites
    loaddata_barcode_platemap_df = qc_merge_df[qc_merge_df['_merge'] == 'left_only'].drop(columns=['_merge'])
    print(f"{loaddata_barcode_platemap_df.shape[0]} sites after QC")


# ## Generate Data split

# In[6]:


loaddata_barcode_platemap_train_df = loaddata_barcode_platemap_df.copy()

## Filter load data csvs dynamically with TRAIN_CONDITION_KWARGS
for k, v in TRAIN_CONDITION_KWARGS.items():
    if isinstance(v, list):
        loaddata_barcode_platemap_train_df = loaddata_barcode_platemap_train_df[loaddata_barcode_platemap_train_df[k].isin(v)]
    else:
        loaddata_barcode_platemap_train_df = loaddata_barcode_platemap_train_df[loaddata_barcode_platemap_train_df[k] == v]
    if len(loaddata_barcode_platemap_train_df) == 0:
        raise ValueError(f'No data found for {k}={v}')
print(f"{loaddata_barcode_platemap_train_df.shape[0]} sites for train and heldout")

## Everything else is used for eval 
loaddata_barcode_platemap_eval_df = loaddata_barcode_platemap_df.loc[
    ~loaddata_barcode_platemap_df.index.isin(loaddata_barcode_platemap_train_df.index)
]
print(f"{loaddata_barcode_platemap_eval_df.shape[0]} sites for evaluation")


# ## For each unique condition combation in train/heldout split, hold out one well at random

# In[7]:


## Reproducibility
seed = 42
np.random.seed(seed)

## Group by seeding density and cell line (condition)
grouped = loaddata_barcode_platemap_train_df.groupby(CONDITIONS)

## Initialize lists to store holdout and train data
heldout_list = []
train_list = []

## Iterate over each group (condition)
for _, group in grouped:

    # sample one well in each group at random
    held_out_well = [np.random.choice(group[WELL_COLUMN].unique())]
    train_wells = group[~group[WELL_COLUMN].isin(held_out_well)][WELL_COLUMN].unique()

    # subset group into train and heldout
    loaddata_held_out_df = group[group[WELL_COLUMN].isin(held_out_well)].copy()
    loaddata_train_df = group[group[WELL_COLUMN].isin(train_wells)].copy()

    # print which well is heldout
    condition = group[CONDITIONS].iloc[0].to_dict()
    print(f"For Condition: {condition} Heldout well: {held_out_well} Train wells: {train_wells}")

    # append subset groups to lists
    heldout_list.append(loaddata_held_out_df)
    train_list.append(loaddata_train_df)

# Concatenate the lists into final train and heldout loaddata dataframes
loaddata_heldout_df = pd.concat(heldout_list).reset_index(drop=True)
print(f"{loaddata_heldout_df.shape[0]} sites Heldout")
loaddata_train_df = pd.concat(train_list).reset_index(drop=True)
print(f"{loaddata_train_df.shape[0]} sites for Training")


# In[8]:


loaddata_heldout_df.to_csv(DATASPLIT_OUTPUT_DIR / 'loaddata_heldout.csv')
loaddata_train_df.to_csv(DATASPLIT_OUTPUT_DIR / 'loaddata_train.csv')
loaddata_barcode_platemap_eval_df.to_csv(DATASPLIT_OUTPUT_DIR / 'loaddata_eval.csv')

