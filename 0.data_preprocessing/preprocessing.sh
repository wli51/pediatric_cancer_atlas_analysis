#!/bin/bash

# Initialize Conda for Bash
conda init bash
# Activate the Conda environment (change 'your_env_name' to your environment name)
conda activate your_env_name

# Create the output directory if it does not exist
mkdir -p nbconverted

# Convert Jupyter notebooks to Python scripts in the nbconverted folder
jupyter nbconvert --to script --output-dir=nbconverted/ *.ipynb

# Run the Python scripts in order
python nbconverted/0.1.filter_low_quality_sites.py
python nbconverted/0.2.data_split.py