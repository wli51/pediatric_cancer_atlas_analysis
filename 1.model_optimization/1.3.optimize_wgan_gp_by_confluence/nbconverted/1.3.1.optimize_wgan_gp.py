#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import sys
import yaml
import random

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

## wGaN training
from virtual_stain_flow.models.unet import UNet
from virtual_stain_flow.models.discriminator import GlobalDiscriminator
from virtual_stain_flow.trainers.WGANTrainer import WGANTrainer

## wGaN losses
from virtual_stain_flow.losses.GradientPenaltyLoss import GradientPenaltyLoss
from virtual_stain_flow.losses.DiscriminatorLoss import WassersteinLoss
from virtual_stain_flow.losses.GeneratorLoss import GeneratorLoss

from virtual_stain_flow.transforms.MinMaxNormalize import MinMaxNormalize
from virtual_stain_flow.transforms.PixelDepthTransform import PixelDepthTransform

## Metrics
from virtual_stain_flow.metrics.MetricsWrapper import MetricsWrapper
from virtual_stain_flow.metrics.PSNR import PSNR
from virtual_stain_flow.metrics.SSIM import SSIM

## callback
from virtual_stain_flow.callbacks.MlflowLogger import MlflowLogger
from virtual_stain_flow.callbacks.IntermediatePlot import IntermediatePlot


# ## Define paths and other train parameters

# In[4]:


## Loaddata for train
LOADDATA_FILE_PATH = pathlib.Path('.').absolute().parent.parent \
    / '0.data_preprocessing' / 'data_split_loaddata' / 'loaddata_train.csv'
assert LOADDATA_FILE_PATH.exists()

LOADDATA_HELDOUT_FILE_PATH = pathlib.Path('.').absolute().parent.parent \
    / '0.data_preprocessing' / 'data_split_loaddata' / 'loaddata_heldout.csv'
assert LOADDATA_HELDOUT_FILE_PATH.exists(), f"Directory not found: {LOADDATA_HELDOUT_FILE_PATH}"

SC_FEATURES_DIR = pathlib.Path(config['paths']['sc_features_path'])

## Output directories
MLFLOW_DIR = pathlib.Path('.').absolute() / 'optuna_mlflow'
MLFLOW_DIR.mkdir(parents=True, exist_ok=True)

OPTUNA_JOBLIB_DIR = pathlib.Path('.').absolute() / 'optuna_joblib'
OPTUNA_JOBLIB_DIR.mkdir(parents=True, exist_ok=True)

PLOT_DIR = pathlib.Path('.').absolute() / 'optuna_plots'
PLOT_DIR.mkdir(parents=True, exist_ok=True)

## Basic data generation and max epoch definition
PATCH_SIZE = 256
EPOCHS = 1_000

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


# ## Create patched dataset from heldout data for use with plotting predictions during optimization

# In[6]:


loaddata_heldout_df = pd.read_csv(LOADDATA_HELDOUT_FILE_PATH)
## Retrieve relevant sc feature information
sc_features = pd.DataFrame()
for plate in loaddata_heldout_df['Metadata_Plate'].unique():
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

## Generate multi-channel patch dataset for plotting
pds_heldout = PatchDataset(
        _loaddata_csv=loaddata_heldout_df,
        _sc_feature=sc_features,
        _input_channel_keys=INPUT_CHANNEL_NAMES,
        _target_channel_keys=TARGET_CHANNEL_NAMES,
        _input_transform=PixelDepthTransform(src_bit_depth=16, target_bit_depth=8, _always_apply=True),
        _target_transform=MinMaxNormalize(_normalization_factor=(2 ** 16) - 1, _always_apply=True),
        patch_size=PATCH_SIZE,
        verbose=False,
        patch_generation_method="random_cell",
        n_expected_patches_per_img=5,
        patch_generation_random_seed=42
    )

## Generate list of indice to plot
n_patches = len(pds_heldout)
random.seed(42)
visualization_patch_indices = random.sample(range(n_patches), 5)


# ## Define optimization objectives

# In[ ]:


import gc
def free_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()

def objective(trial, dataset, plot_dataset, channel_name, confluence_group_name, plot_dir):

    trial_id = trial.number

    # Suggest hyperparameters
    gen_optim_lr = trial.suggest_float("gen_optim_lr", 1e-5, 1e-2, log=True)
    gen_optim_beta0 = trial.suggest_float("gen_optim_beta0", 0.5, 0.9, log=False)
    gen_optim_beta1 = trial.suggest_float('gen_optim_beta1', 0.9, 0.999, log=False)
    gen_optim_weight_decay = 0 # no weight decay for generator optimizer
    
    disc_optim_lr = trial.suggest_float("disc_optim_lr", 1e-5, 1e-2, log=True)
    disc_optim_beta0 = trial.suggest_float("disc_optim_beta0", 0.5, 0.9, log=False)
    disc_optim_beta1 = trial.suggest_float('disc_optim_beta1', 0.9, 0.999, log=False)
    disc_optim_weight_decay = trial.suggest_float('disc_optim_weight_decay', 1e-4, 1e-2, log=True)

    # convolutional depth for unet generator model
    gen_conv_depth = trial.suggest_int('gen_conv_depth', 3, 5) # convolutional depth for unet generator model
    disc_conv_depth = trial.suggest_int('disc_conv_depth', 3, 5) # convolutional depth for discriminator network

    # how often is the generator/discriminator weight updated (once every x epochs)
    gen_update_freq = trial.suggest_int('gen_update_freq', 2, 8)
    disc_update_freq = 1 # fixed for wGaN gp

    # batch size and early stopping patience
    batch_size = trial.suggest_int('batch_size', 16, 32, step=16)
    patience = trial.suggest_int('patience', 5, 20) # early stop patience

    ## Print trial information and batch information to identify 
    # batch sizes that leads to CUDA out of memory
    print(f"Trial ID: {trial_id}")
    print(f"Trial Batch Size: {batch_size}")

    ## Setup model, discriminator and optimizer
    generator = UNet(
        n_channels=1,
        n_classes=1,
        depth=gen_conv_depth,
        bilinear=False
    )
    discriminator = GlobalDiscriminator(
        n_in_channels = 2, # 1 input brightfield + 1 target fluo channel
        n_in_filters = 64,
        _conv_depth = disc_conv_depth,
        _pool_before_fc = True
    )

    generator_optimizer = optim.Adam(generator.parameters(), 
                                 lr=gen_optim_lr, 
                                 betas=(gen_optim_beta0, gen_optim_beta1),
                                 weight_decay=gen_optim_weight_decay)
    
    discriminator_optimizer = optim.Adam(discriminator.parameters(), 
                                        lr=disc_optim_lr, 
                                        betas=(disc_optim_beta0, disc_optim_beta1),
                                        weight_decay=disc_optim_weight_decay)
    
    ## Metrics to be computed (and logged)
    metric_fns = {
        "L1Loss": MetricsWrapper(_metric_name='L1Loss', module=torch.nn.L1Loss()),
        "mse_loss": MetricsWrapper(_metric_name='mse', module=torch.nn.MSELoss()),
        "ssim_loss": SSIM(_metric_name="ssim"),
        "psnr_loss": PSNR(_metric_name="psnr"),
    }

    ## Special losses

    gp_loss = GradientPenaltyLoss(
        _metric_name='gp_loss',
        discriminator=discriminator,
        weight=10.0,
    )

    gen_loss = GeneratorLoss(
        _metric_name='gen_loss'
    )

    disc_loss = WassersteinLoss(
        _metric_name='disc_loss'
    )

    ## Params to log with mlflow
    params = {
            # generation optimizer hyperparameters
            "gen_optim_lr": gen_optim_lr,
            "gen_update_freq": gen_update_freq,
            "gen_optim_beta0": gen_optim_beta0,
            "gen_optim_beta1": gen_optim_beta1,
            "gen_optim_weight_decay": gen_optim_weight_decay,
            # generator model hyperparameter(s)
            "gen_conv_depth": gen_conv_depth,
            # discriminator optimizer hyperparameters
            "disc_optim_lr": disc_optim_lr,
            "disc_update_freq": disc_update_freq,
            "disc_optim_beta0": disc_optim_beta0,
            "disc_optim_beta1": disc_optim_beta1,
            "disc_weight_decay": disc_optim_weight_decay,
            # discrminator model hyperparameter(s)
            "disc_conv_depth": disc_conv_depth,
            # dataset hyperparameters
            "patch_size": PATCH_SIZE,
            "channel_name": channel_name,
            "confluence": confluence_group_name,
            # data loader hyperparameters
            "batch_size": batch_size,
            # training hyperparameters
            "patience": patience,
            "epochs": EPOCHS,
            # optuna trial id
            "trial_id": trial_id
        }

    ## mlflow logger callback
    mlflow_logger_callback = MlflowLogger(
        name='mlflow_logger',
        mlflow_uri=MLFLOW_DIR / 'mlruns',
        mlflow_experiment_name=f'wGaN_gp_optimize_{confluence_group_name}',
        mlflow_start_run_args={'run_name': f'wGaN_gp_optimize_{confluence_group_name}_{channel_name}', 'nested': True},
        mlflow_log_params_args=params
    )
    
    trial_plot_dir = pathlib.Path(plot_dir) / str(trial_id)
    trial_plot_dir.mkdir(parents=True, exist_ok=True)
    
    plot_callback = IntermediatePlot(
            name='plotter',
            path=trial_plot_dir,
            dataset=plot_dataset,
            indices=visualization_patch_indices, # every model being trained will have the same visualization patch indices
            plot_metrics=[SSIM(_metric_name='ssim'), PSNR(_metric_name='psnr')],
            figsize=(20, 25),
            every_n_epochs=5,
            show_plot=False,
        )
    
    ## Trainer
    wgan_trainer = WGANTrainer(
        generator=generator,
        discriminator=discriminator,
        gen_optimizer=generator_optimizer,
        disc_optimizer=discriminator_optimizer,
        generator_loss_fn=gen_loss,
        discriminator_loss_fn=disc_loss,
        gradient_penalty_fn=gp_loss,
        discriminator_update_freq=disc_update_freq,
        generator_update_freq=gen_update_freq,
        dataset=dataset,
        batch_size=batch_size,
        epochs=EPOCHS,
        patience=patience,
        callbacks=[mlflow_logger_callback, plot_callback],
        metrics=metric_fns,
        device='cuda',
        early_termination_metric='L1Loss' # Early termination and optimization will be based on L1 loss
    )

    # Train the model and log validation loss
    wgan_trainer.train()
    val_loss = wgan_trainer.best_loss

    del generator
    del discriminator

    del generator_optimizer
    del discriminator_optimizer

    del gp_loss
    del gen_loss
    del disc_loss

    del wgan_trainer
    
    free_gpu_memory()

    return val_loss


# ## Optimize for wGAN GP hyperparameters per Channel

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
        patch_generation_random_seed=42
    )

    for channel_name in TARGET_CHANNEL_NAMES:

        # Create directory for plots
        plot_dir = PLOT_DIR / f"{confluence_group_name}_{channel_name}"
        plot_dir.mkdir(parents=True, exist_ok=True)

        ## Configure dataset channel
        pds.set_input_channel_keys(INPUT_CHANNEL_NAMES)
        pds.set_target_channel_keys(channel_name)

        # Configure heldout dataset channel
        pds_heldout.set_input_channel_keys(INPUT_CHANNEL_NAMES)
        pds_heldout.set_target_channel_keys(channel_name)

        ## Cache dataset of channel
        cds = CachedDataset(
            dataset=pds,
            prefill_cache=True
        )

        print(f"Beginning optimization for channel: {channel_name} for {confluence_group_name}")

        # Load the existing study for the current channel
        study_path = OPTUNA_JOBLIB_DIR / f"wGaN_gp_optimize_{channel_name}_{confluence_group_name}.joblib"
        if study_path.exists():
            study = joblib.load(study_path)
        else:
            study = optuna.create_study(
                direction="minimize",
                study_name=f"wGaN_gp_optimize_{channel_name}_{confluence_group_name}",
                sampler=optuna.samplers.TPESampler(seed=42)
            )

        # Resume optimization and run up until N_TRIALS
        while len(study.trials) < N_TRIALS:
            study.optimize(lambda trial: objective(trial, cds, pds_heldout, channel_name, confluence_group_name, plot_dir), n_trials=1)
            joblib.dump(study, study_path)  # Save study after every trial
            print(f"Saved study after trial {len(study.trials)}/{N_TRIALS}")
        
        print(f"{N_TRIALS} of Hyperparameter Optimization for {channel_name}_{confluence_group_name} completed.")

        # Print best trial results
        print(f"Best trial for channel {channel_name}:")
        print(f"  Validation Loss: {study.best_trial.value}")
        print(f"  Hyperparameters: {study.best_trial.params}")

