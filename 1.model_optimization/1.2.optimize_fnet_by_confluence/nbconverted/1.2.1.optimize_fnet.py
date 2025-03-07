#!/usr/bin/env python
# coding: utf-8

# # This notebook carries out hyperparameter optimization of the FNet model on two datasets (high and low confluence level) of U2-OS cell painting image data 

# In[ ]:


import pathlib
import sys
import yaml

import pandas as pd
import torch
import torch.optim as optim
import optuna
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
from virtual_stain_flow.datasets.CachedDataset import CachedDataset

## FNet training
from virtual_stain_flow.models.fnet import FNet
from virtual_stain_flow.trainers.Trainer import Trainer

from virtual_stain_flow.transforms.MinMaxNormalize import MinMaxNormalize
from virtual_stain_flow.transforms.PixelDepthTransform import PixelDepthTransform

## Metrics
from virtual_stain_flow.metrics.MetricsWrapper import MetricsWrapper
from virtual_stain_flow.metrics.PSNR import PSNR
from virtual_stain_flow.metrics.SSIM import SSIM

## callback
from virtual_stain_flow.callbacks.MlflowLogger import MlflowLogger


# ## Define paths and other train parameters

# In[4]:


## Loaddata split for train is also used for optimization
LOADDATA_FILE_PATH = pathlib.Path('.').absolute().parent.parent \
    / '0.data_preprocessing' / 'data_split_loaddata' / 'loaddata_train.csv'
assert LOADDATA_FILE_PATH.exists(), f"File not found: {LOADDATA_FILE_PATH}"

SC_FEATURES_DIR = pathlib.Path(config['paths']['sc_features_path'])
assert SC_FEATURES_DIR.exists(), f"Directory not found: {SC_FEATURES_DIR}"

## Output directories
MLFLOW_DIR = pathlib.Path('.').absolute() / 'optuna_mlflow'
MLFLOW_DIR.mkdir(parents=True, exist_ok=True)

OPTUNA_JOBLIB_DIR = pathlib.Path('.').absolute() / 'optuna_joblib'
OPTUNA_JOBLIB_DIR.mkdir(parents=True, exist_ok=True)

## Data (Patch) generation and max epoch definition
PATCH_SIZE = 256
EPOCHS = 1_000 # max number of epochs for each optimization trial

## Channels for input and target are read from config
INPUT_CHANNEL_NAMES = config['data']['input_channel_keys']
TARGET_CHANNEL_NAMES = config['data']['target_channel_keys']


# ## Defines how the train data will be divided to train models on two levels of confluence

# In[5]:


DATA_GROUPING = {
    'high_confluence': {
        'seeding_density': [12_000, 8_000]
    },
    'low_confluence': {
        'seeding_density': [4_000, 2_000, 1_000]
    }
}


# ## Define optimization objectives

# In[6]:


import gc
def free_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()

def objective(trial, dataset, channel_name, confluence_group_name):
    
    ## Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True) # adam optimizer learning rate
    beta0 = trial.suggest_float('beta0', 0.5, 0.9) # adam optimizer beta 0
    beta1 = trial.suggest_float('beta1', 0.9, 0.999) # adam optimizer beta 1
    betas = (beta0, beta1)

    batch_size = trial.suggest_int('batch_size', 16, 64, step=16)
    patience = trial.suggest_int('patience', 5, 20) # early stop patience

    conv_depth = trial.suggest_int('conv_depth', 3, 5) # convolutional depth for fnet model

    ## Setup model and optimizer
    model = FNet(depth=conv_depth)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas)
    
    ## Metrics to be computed (and logged)
    metric_fns = {
        "mse_loss": MetricsWrapper(_metric_name='mse', module=torch.nn.MSELoss()),
        "ssim_loss": SSIM(_metric_name="ssim"),
        "psnr_loss": PSNR(_metric_name="psnr"),
    }

    ## Params to log with mlflow
    params = {
            # optimizer hyperparameters
            "optimizer": "adam",
            "lr": lr,
            "beta0": beta0,
            "beta1": beta1,
            # model hyperparameter(s)
            "depth": conv_depth,
            # dataset hyperparameter(s)
            "patch_size": PATCH_SIZE,
            "channel_name": channel_name,
            "confluence": confluence_group_name,
            # data loading hyperparameter(s)
            "batch_size": batch_size,
            # training hyperparameter(s)
            "patience": patience,
            "epochs": EPOCHS,
        }

    ## mlflow logger callback
    mlflow_logger_callback = MlflowLogger(
        name='mlflow_logger',
        mlflow_uri=MLFLOW_DIR / 'mlruns',
        mlflow_experiment_name=f'FNet_optimize_{confluence_group_name}',
        mlflow_start_run_args={'run_name': f'FNet_optimize_{confluence_group_name}_{channel_name}', 'nested': True},
        mlflow_log_params_args=params
    )
    
    ## Trainer
    trainer = Trainer(
        model = model,
        optimizer = optimizer,
        backprop_loss = torch.nn.L1Loss(), # MAE loss for backpropagation
        dataset = dataset,
        batch_size = batch_size,
        epochs = EPOCHS,
        patience = patience,
        callbacks=[mlflow_logger_callback],
        metrics=metric_fns,
        device = 'cuda',
        early_termination_metric='L1Loss'
    )

    # Train the model and log validation loss
    trainer.train()
    val_loss = trainer.best_loss

    del model
    del optimizer
    del metric_fns
    del mlflow_logger_callback
    del trainer
    
    free_gpu_memory()

    return val_loss


# ## Optimize for FNet hyperparameters per Channel

# In[ ]:


N_TRIALS = 50  # Total trials

## Loaddata for optimization
loaddata_df = pd.read_csv(LOADDATA_FILE_PATH)
for confluence_group_name, conditions in DATA_GROUPING.items():

    ## Loaddata for the confluence group
    loaddata_condition_df = loaddata_df.copy()
    for condition, values in conditions.items():
        loaddata_condition_df = loaddata_condition_df[
            loaddata_condition_df[condition].isin(values)
        ]

    ## Retrieve relevant sc features by assemblying them from parquet files
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

    ## Create patch dataset
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

    for channel_name in TARGET_CHANNEL_NAMES:
        
        ##Configure dataset channel
        pds.set_input_channel_keys(INPUT_CHANNEL_NAMES)
        pds.set_target_channel_keys(channel_name)

        ## Cache dataset of channel
        cds = CachedDataset(
            dataset=pds,
            prefill_cache=True
        )

        print(f"Beginning optimization for channel: {channel_name} for {confluence_group_name}")

        # Load the existing study for the current channel
        study_path = OPTUNA_JOBLIB_DIR / f"FNet_optimize_{channel_name}_{confluence_group_name}.joblib"
        if study_path.exists():
            study = joblib.load(study_path)
        else:
            # Or create if not already existing
            study = optuna.create_study(
                direction="minimize",
                study_name=f"FNet_optimize_{channel_name}_{confluence_group_name}",
                sampler=optuna.samplers.TPESampler(seed=42)
            )
            
        # Resume optimization and run up until N_TRIALS
        while len(study.trials) < N_TRIALS:
            study.optimize(lambda trial: objective(trial, cds, channel_name, confluence_group_name), n_trials=1)
            joblib.dump(study, study_path)  # Save study after every trial
            print(f"Saved study after trial {len(study.trials)}/{N_TRIALS}")
        
        print(f"{N_TRIALS} of Hyperparameter Optimization for {channel_name}_{confluence_group_name} completed.")

        # Print best trial results        
        print(f"Best trial for channel {channel_name}:")
        print(f"  Validation Loss: {study.best_trial.value}")
        print(f"  Hyperparameters: {study.best_trial.params}")

