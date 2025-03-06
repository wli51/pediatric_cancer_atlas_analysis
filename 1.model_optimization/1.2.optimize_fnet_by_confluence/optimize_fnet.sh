#!/bin/bash

# Initialize Conda for Bash
conda init bash
# Activate the environment
conda activate alsf_analysis

# Create the output directory if it does not exist
mkdir -p nbconverted

# Convert Jupyter notebooks to Python scripts in the nbconverted folder
jupyter nbconvert --to script --output-dir=nbconverted/ *.ipynb

# Run the Python scripts in order
python nbconverted/1.2.1.optimize_fnet.py
python nbconverted/1.2.2.validate_fnet_optimization.py