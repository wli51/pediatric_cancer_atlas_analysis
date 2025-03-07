#!/usr/bin/env python
# coding: utf-8

# ## In this notebook the FNet optimization results are validating against train and heldout data

# In[1]:


import random
import pathlib
import sys
import yaml
from collections import defaultdict

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.artifacts
from optuna.visualization import plot_param_importances, plot_optimization_history
import joblib


# ## Read config

# In[2]:


with open(pathlib.Path('.').absolute().parent.parent / "config.yml", "r") as file:
    config = yaml.safe_load(file)


# ## Import virtual_stain_flow software 

# In[3]:


sys.path.append(config['paths']['software_path'])
print(str(pathlib.Path('.').absolute().parent.parent))

## Dataset
from virtual_stain_flow.datasets.PatchDataset import PatchDataset

## FNet training
from virtual_stain_flow.models.fnet import FNet

from virtual_stain_flow.transforms.MinMaxNormalize import MinMaxNormalize
from virtual_stain_flow.transforms.PixelDepthTransform import PixelDepthTransform

## Metrics
from virtual_stain_flow.metrics.PSNR import PSNR
from virtual_stain_flow.metrics.SSIM import SSIM

## Visualization software
from virtual_stain_flow.evaluation.visualization_utils import plot_predictions_grid_from_model
from virtual_stain_flow.evaluation.evaluation_utils import evaluate_per_image_metric
from virtual_stain_flow.evaluation.predict_utils import predict_image


# ## Define paths and other train parameters

# In[4]:


## Loaddata for train and heldout set
LOADDATA_FILE_PATH = pathlib.Path('.').absolute().parent.parent \
    / '0.data_preprocessing' / 'data_split_loaddata' / 'loaddata_train.csv'
assert LOADDATA_FILE_PATH.exists(), f"File not found: {LOADDATA_FILE_PATH}"

LOADDATA_HELDOUT_FILE_PATH = pathlib.Path('.').absolute().parent.parent \
    / '0.data_preprocessing' / 'data_split_loaddata' / 'loaddata_heldout.csv'
assert LOADDATA_HELDOUT_FILE_PATH.exists(), f"Directory not found: {LOADDATA_HELDOUT_FILE_PATH}"

## Corresponding sc features directory containing cell coordiantes used for patch generation
SC_FEATURES_DIR = pathlib.Path(config['paths']['sc_features_path'])
assert SC_FEATURES_DIR.exists(), f"Directory not found: {SC_FEATURES_DIR}"

## Optimization Output Saved under these directories
MLFLOW_DIR = pathlib.Path('.').absolute() / 'optuna_mlflow'
assert MLFLOW_DIR.exists(), f"Mlflow directory not found: {MLFLOW_DIR}"

OPTUNA_JOBLIB_DIR = pathlib.Path('.').absolute() / 'optuna_joblib'
assert OPTUNA_JOBLIB_DIR.exists(), f"Optuna joblib directory not found: {OPTUNA_JOBLIB_DIR}"

## Validation Output Path
VALIDATION_OUTPUT_DIR = pathlib.Path('.').absolute() / 'validation'
VALIDATION_OUTPUT_DIR.mkdir(exist_ok=True)
VALIDATION_INTERMEDIATE_DIR = VALIDATION_OUTPUT_DIR / 'intermediate'
VALIDATION_INTERMEDIATE_DIR.mkdir(exist_ok=True)
VALIDATION_PLOTS_DIR = VALIDATION_OUTPUT_DIR / 'plots'
VALIDATION_PLOTS_DIR.mkdir(exist_ok=True)
# Define the path for the master metric file
ALL_METRICS_FILE = VALIDATION_OUTPUT_DIR / 'all_metrics.csv'

## Channels for input and target are read from config
INPUT_CHANNEL_NAMES = config['data']['input_channel_keys']
TARGET_CHANNEL_NAMES = config['data']['target_channel_keys']


# In[5]:


## Patch size definition
PATCH_SIZE = 256

CONFLUENCE_GROUPS = ['high_confluence', 'low_confluence']


# ## Define Discrete Conditions to Evaluate

# In[6]:


# Relevant columns in loaddata
SITE_COLUMN = 'Metadata_Site'
WELL_COLUMN = 'Metadata_Well'
PLATE_COLUMN = 'Metadata_Plate'

PLATEMAP_COLUMN = 'platemap_file'
CELL_LINE_COLUMN = 'cell_line'
SEEDING_DENSITY_COLUMN = 'seeding_density'

# Sites are uniquely identified by the combination of these columns
UNIQUE_IDENTIFIERS = [SITE_COLUMN, WELL_COLUMN, PLATE_COLUMN]
# Conditions are uniquely identified by the combination of these columns
CONDITION_IDENTIFIERS =  [CELL_LINE_COLUMN, SEEDING_DENSITY_COLUMN, PLATEMAP_COLUMN]


# In[7]:


mlflow.set_tracking_uri(MLFLOW_DIR / 'mlruns')

mlflow_results = {}
optuna_results = defaultdict(dict)

for confluence_group_name in CONFLUENCE_GROUPS:
    ## Access relevant optimization result and logs by confluence
    experiment_name = f'FNet_optimize_{confluence_group_name}'
    experiment = mlflow.get_experiment_by_name(experiment_name)
    mlflow_results[confluence_group_name] = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    for channel_name in TARGET_CHANNEL_NAMES:
        optuna_study_path = OPTUNA_JOBLIB_DIR / f"FNet_optimize_{channel_name}_{confluence_group_name}.joblib"
        study = joblib.load(optuna_study_path)
        optuna_results[confluence_group_name][channel_name] = study

        print(f"Optuna study {channel_name} {confluence_group_name}:")
        plot_param_importances(study).show()
        plot_optimization_history(study).show()


# ## Concatenate datasplits into a single dataframe to streamline evaluation

# In[8]:


## Concat all datasplits
loaddata_df_all = pd.DataFrame()
for datasplit, file in zip(
    ['train', 'heldout'], 
    [LOADDATA_FILE_PATH, LOADDATA_HELDOUT_FILE_PATH]):
    loaddata_df = pd.read_csv(file, index_col=0)
    loaddata_df['datasplit'] = datasplit
    loaddata_df_all = pd.concat([loaddata_df_all, loaddata_df])
loaddata_df_all.head()


# ## Concatenate mlflow runs across confluence group trainings to streamline evaluation

# In[9]:


mlflow_runs = pd.DataFrame()
for confluence_group in CONFLUENCE_GROUPS:
    experiment_name = f'FNet_optimize_{confluence_group}'
    experiment = mlflow.get_experiment_by_name(experiment_name)
    mlflow_runs = pd.concat(
        [mlflow_runs, mlflow.search_runs(experiment_ids=[experiment.experiment_id])]
         )
mlflow_runs.head()


# ## Validate/Evaluate optimization trials against train and heldout dataset

# In[10]:


EVAL_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL_METRICS = [PSNR(_metric_name='psnr'), SSIM(_metric_name='ssim')]


# In[ ]:


# Load existing metrics if the file exists
if ALL_METRICS_FILE.exists():
    all_metrics_df = pd.read_csv(ALL_METRICS_FILE, dtype=str)
else:
    all_metrics_df = pd.DataFrame()

## Iterate over train and heldout
for conditions, loaddata_condition_df in loaddata_df_all.groupby(CONDITION_IDENTIFIERS + ['datasplit']):

    datasplit = conditions[-1]
    condition_dict = {condition_column_name: condition \
                      for condition_column_name, condition in \
                        zip(CONDITION_IDENTIFIERS, conditions[:-1])}
    
    condition_str = ';'.join([f"{key.replace('params.','')}={value}" for key, value in condition_dict.items()])
    print(f"Evaluating {condition_str}")

    # Check if the condition has already been evaluated
    existing_conditions = all_metrics_df.query(
        " & ".join([f"{key} == '{value}'" for key, value in condition_dict.items()])
    ) if not all_metrics_df.empty else pd.DataFrame()

    ## Collect corresponding sc features with loaddata_condition_df
    sc_features = pd.DataFrame()
    for plate in loaddata_condition_df['Metadata_Plate'].unique():
        sc_features_parquet = SC_FEATURES_DIR / f'{plate}_sc_normalized.parquet'
        if not sc_features_parquet.exists():
            print(f'{sc_features_parquet} does not exist, skipping...')
            continue 
        else:
            sc_features = pd.concat([
                sc_features, 
                pd.read_parquet(
                    sc_features_parquet,
                    columns=['Metadata_Plate', 'Metadata_Well', 'Metadata_Site', 'Metadata_Cells_Location_Center_X', 'Metadata_Cells_Location_Center_Y']
                )
            ])

    ## Load data
    pds = PatchDataset(
        _loaddata_csv=loaddata_condition_df,
        _sc_feature=sc_features,
        _input_channel_keys=INPUT_CHANNEL_NAMES,
        _target_channel_keys=TARGET_CHANNEL_NAMES,
        _input_transform=PixelDepthTransform(src_bit_depth=16, target_bit_depth=8, _always_apply=True),
        _target_transform=MinMaxNormalize(_normalization_factor=(2 ** 16) - 1, _always_apply=True),
        patch_size=PATCH_SIZE,
        verbose=False,
        patch_generation_method="random_cell",
        n_expected_patches_per_img=50,
        patch_generation_random_seed=42
    )
    
    n_patches = len(pds)
    random.seed(42)
    visualization_patch_indices = random.sample(range(n_patches), 5)

    ## Group evaluation by channel to minimize the switching between dataset target channels
    for target_channel_name, channel_mlflow_runs in mlflow_runs.groupby('params.channel_name'):
        
        pds.set_input_channel_keys(INPUT_CHANNEL_NAMES)
        pds.set_target_channel_keys([target_channel_name])
        # _, targets = next(iter(DataLoader(pds, batch_size=len(pds))))
        
        ## Iterate over models
        for _, run in channel_mlflow_runs.iterrows():

            run_id = run['run_id']
            run_name = run['tags.mlflow.runName']

            # Check if this model has already been evaluated for this condition
            if not existing_conditions.empty and run_id in existing_conditions['run_id'].values:
                print(f"Skipping evaluation for run {run_id} (already exists in master file)")
                continue
            
            ## Load model
            model_uri = run['artifact_uri']
            model_weight_path = pathlib.Path(mlflow.artifacts.download_artifacts(artifact_uri=model_uri)) /\
                'models' / 'best_model_weights.pth'
            if not model_weight_path.exists():
                # in case there are uncompleted runs
                print(f"Model weight not found for run {run_name}, skipping ...")
                continue

            model_depth = int(run['params.depth'])
            model = FNet(depth=model_depth)
            try:
                model.load_state_dict(torch.load(model_weight_path, weights_only=True))
            except:
                print(f"Fail to load model weight for run {run_id}, skipping ...")
                continue
            model.to(EVAL_DEVICE)
            
            ## Run forward pass on the dataset
            targets, predictions = predict_image(
                dataset = pds,
                model = model,
                device = EVAL_DEVICE
            )

            ## Evaluate metrics
            metrics_df = evaluate_per_image_metric(
                predictions=predictions,
                targets=targets,
                metrics=EVAL_METRICS
            )
            metrics_mean = metrics_df.mean()

            ## Concatenate metrics with condition and run information
            metrics_df['datasplit'] = datasplit
            for condition_name, value in condition_dict.items():
                metrics_df[condition_name] = value
            metrics_df['run_name'] = run_name
            metrics_df['run_id'] = run_id
            for param_name, value in run.items():
                if param_name.startswith('params.'):
                    metrics_df[param_name] = value

            ### Ensure Column Consistency Before Appending ###
            # If master file exists and has columns, ensure column match
            if not all_metrics_df.empty:
                # Add missing columns to metrics_df
                for col in all_metrics_df.columns:
                    if col not in metrics_df.columns:
                        metrics_df[col] = None  # Fill missing columns with NaN
                
                # Add new columns from metrics_df to all_metrics_df
                for col in metrics_df.columns:
                    if col not in all_metrics_df.columns:
                        all_metrics_df[col] = None  # Fill missing columns with NaN

                # Reorder metrics_df to match all_metrics_df column order
                metrics_df = metrics_df[all_metrics_df.columns]

            metrics_df.to_csv(VALIDATION_INTERMEDIATE_DIR / f'{run_name}_{condition_str}.csv', index=False)
            # Append new results to the master file
            metrics_df.to_csv(ALL_METRICS_FILE, mode='a', header=not ALL_METRICS_FILE.exists(), index=False)
            
            # Produce string representation of metrics
            metrics_mean_str = '_'.join([f"{key}={value:.2f}" for key, value in metrics_mean.items()])
            # Produce strings to identify visualization for particular model
            params_values = {key: value for key, value in run.items() if key.startswith('params.')}
            params_str = '_'.join([f"{key.replace('params.','')}={value}" for key, value in params_values.items()])
            
            plot_predictions_grid_from_model(
                model=model,
                dataset=pds,
                indices=visualization_patch_indices,
                metrics=EVAL_METRICS,
                device=EVAL_DEVICE,
                # by making the plot name start with the metrics, the files can be ordered by metrics for easy comparison
                save_path=VALIDATION_PLOTS_DIR / f'{metrics_mean_str}_{params_str}.png',
                show_plot=False
            )


# ## Preliminary visualization of Validation

# Collect the intermedaite csv outputs and concatenate into a large file

# In[ ]:





# In[ ]:


# Select only 'train' datasplit
train_df = all_metrics_df[all_metrics_df["datasplit"] == "train"].copy()
train_df["params.lr"] = pd.to_numeric(train_df["params.lr"], errors="coerce")
train_df = train_df.dropna(subset=["params.lr"])

# Define binning for params.lr only
train_df["lr_bin"] = pd.qcut(train_df["params.lr"], q=10, duplicates="drop")

# Get unique values for subplot arrangement
channel_names = train_df["params.channel_name"].unique()
confluence_levels = train_df["confluence"].unique()

# Determine subplot grid size
cols = len(channel_names)  # One column per channel_name
rows = len(confluence_levels)  # One row per confluence

fig, axes = plt.subplots(rows, cols, figsize=(10 * cols, 5 * rows), sharex=True, sharey=True)

# Ensure axes is a 2D array for consistent indexing
if rows == 1:
    axes = np.array([axes])  # Convert to 2D array with one row
if cols == 1:
    axes = np.array([[ax] for ax in axes])  # Convert to 2D array with one column

# Iterate through unique combinations of (channel_name, confluence) and plot heatmap
for i, confluence in enumerate(confluence_levels):
    for j, channel in enumerate(channel_names):
        ax = axes[i, j]

        # Filter data for this combination
        subset = train_df[(train_df["params.channel_name"] == channel) & (train_df["confluence"] == confluence)]

        # Create pivot table with integer depth values
        pivot_table = subset.pivot_table(index="params.depth", columns="lr_bin", values="SSIM", aggfunc="mean")

        # Plot heatmap
        sns.heatmap(pivot_table, cmap="viridis", annot=True, fmt=".3f", ax=ax)

        # Titles and labels
        if i == 0:
            ax.set_title(f"Channel: {channel}")
        if j == 0:
            ax.set_ylabel(f"Confluence: {confluence}")
        ax.set_xlabel("Learning Rate Bins")

plt.tight_layout()
plt.show()

