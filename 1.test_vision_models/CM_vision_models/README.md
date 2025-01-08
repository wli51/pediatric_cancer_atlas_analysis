# Borrowed code Image-Image translation model training

All code that live under this directory are borrowed from [WayScience/nuclear_speckles_analysis/1.develop_vision_models](https://github.com/WayScience/nuclear_speckles_analysis).

For now this will serve as a temporary package-ish source to reference and subclass from to quickly adapt existing code for a slightly different use case. Will be changed in the future. 

## Overview
Directory structure

CM_vision_models/
├── datasets
│   ├── ImageDataset.py 
│   └── ImageMetaDataset.py
├── losses
│   ├── AbstractLoss.py
│   ├── L1Loss.py
│   ├── L2Loss.py
│   ├── PSNR.py
│   └── SSIM.py
├── models
│   └── fnet_nn_2d.py
├── transforms
│   ├── CropNPixels.py
│   └── MinMaxNormalize.py
├── ModelTrainer.py
├── README.md
└── environment.yml
