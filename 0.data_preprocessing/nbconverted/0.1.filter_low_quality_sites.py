#!/usr/bin/env python
# coding: utf-8

# # This notebook performs the QC step to set up the image data for use with Image2Image translation model training and Evaluation. 
# This entire data pre-processing step (and the repo in general) will be dependent on a local pediatric_cancer_atlas_profiling repo (https://github.com/WayScience/pediatric_cancer_atlas_profiling) that is ran up to 2.feature_extraction. The config.yml file will need to be configured with the correct path to the pediatric_cancer_atlas_profiling repo for this notebook to work.
# 
# This notebook relies on the whole_img_qc_output to obtain thesaturation and blur QC metrics and generates a collection of sites to be excluded from the training/evaluation. 

# In[1]:


import sys
import subprocess
import pathlib

import pandas as pd
from scipy.stats import zscore


# ## Read config

# In[2]:


def get_repo_root():
    return subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        stdout=subprocess.PIPE,
        check=True,
        text=True
    ).stdout.strip()

repo_root = get_repo_root()
sys.path.append(repo_root)

from config import (
    PROFILING_DIR
)


# ## Define paths

# In[3]:


# Directory with QC CellProfiler outputs per plate
QC_DIR = PROFILING_DIR / "1.illumination_correction" / "whole_img_qc_output"
assert QC_DIR.exists()

# Output path for plate, well and site marked for exclusion
QC_OUTPUT_DIR = pathlib.Path('.') / 'preprocessing_output'
QC_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


# ## Collect files containing plate specific QC Metrics from the profiling repo

# In[4]:


# Create an empty dictionary to store data frames for each plate
all_qc_data_frames = {}

# List all plate directories
plates = [plate.name for plate in QC_DIR.iterdir() if plate.is_dir()]

# Loop through each plate
for plate in plates:
    # Read in CSV with all image quality metrics per image for the current plate
    qc_df = pd.read_csv(QC_DIR / plate / "Image.csv")

    # Store the data frame for the current plate in the dictionary
    all_qc_data_frames[plate] = qc_df

# Print the plate names to ensure they were loaded correctly
print(all_qc_data_frames.keys())

# Select the first plate in the list
first_plate = plates[0]
print(f"Showing example for the first plate: {first_plate}")

# Access the dataframe for the first plate
example_df = all_qc_data_frames[first_plate]

# Show the shape and the first few rows of the dataframe for the first plate
print(example_df.shape)


# ## Create concatenated data frames combining blur and saturation metrics from all channels for all plates

# In[5]:


# Create an empty dictionary to store data frames for each channel
all_combined_dfs = {}

# Iterate through each channel
for channel in config['data']['target_channel_keys']: # excluding input Brightfield since the metrics are not robust to this type of channel
    # Create an empty list to store data frames for each plate
    plate_dfs = []

    # Iterate through each plate and create the specified data frame for the channel
    for plate, qc_df in all_qc_data_frames.items():
        plate_df = qc_df.filter(like="Metadata_").copy()

        # Add PowerLogLogSlope column (blur metric)
        plate_df["ImageQuality_PowerLogLogSlope"] = qc_df[
            f"ImageQuality_PowerLogLogSlope_{channel}"
        ]

        # Add PercentMaximal column (saturation metric)
        plate_df["ImageQuality_PercentMaximal"] = qc_df[
            f"ImageQuality_PercentMaximal_{channel}"
        ]

        # Add "Channel" column
        plate_df["Channel"] = channel

        # Add "Metadata_Plate" column
        plate_df["Metadata_Plate"] = plate

        # Append the data frame to the list
        plate_dfs.append(plate_df)

    # Concatenate data frames for each plate for the current channel
    all_combined_dfs[channel] = pd.concat(
        plate_dfs, keys=list(all_qc_data_frames.keys()), names=["Metadata_Plate", None]
    )

# Concatenate the channel data frames together for plotting
df = pd.concat(list(all_combined_dfs.values()), ignore_index=True)

print(df.shape)
df.head()


# ## Apply Z-scores threshold on all columns (channels) with all plates, sites with any channel that falls beyond the threshold will be marked for exclusion

# In[6]:


# Calculate Z-scores for the column with all plates
metric_z_thresh_dict = {
    "ImageQuality_PowerLogLogSlope": 2.5,
    "ImageQuality_PercentMaximal": 2,
}

total_plate_well_site = df[["Metadata_Plate", "Metadata_Well", "Metadata_Site"]].drop_duplicates()
removed_plate_well_site = pd.DataFrame()

for metric, z_thresh in metric_z_thresh_dict.items():
    z_scores = zscore(df[metric])
    outliers = df[abs(z_scores) > z_thresh]
    removed_plate_well_site = pd.concat(
        [removed_plate_well_site, outliers[["Metadata_Plate", "Metadata_Well", "Metadata_Site"]].drop_duplicates()]
    )

print(f"Out of a total of {total_plate_well_site.shape[0]} plate, well and site combos, {removed_plate_well_site.shape[0]} ({removed_plate_well_site.shape[0] * 100 / total_plate_well_site.shape[0]:.2f}%) removed due to low quality.")


# ## Export sites to be excluded as a csv

# In[7]:


removed_plate_well_site.to_csv(QC_OUTPUT_DIR / 'qc_exclusion.csv', index=False)

