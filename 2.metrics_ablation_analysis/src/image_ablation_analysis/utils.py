"""
utils.py

Utility functions for image ablation analysis.
"""

import pathlib
from typing import Union

import numpy as np


def check_path(
    path: Union[str, pathlib.Path], 
    ensure_dir: bool = False
) -> pathlib.Path:
    """
    Ensure path is a pathlib.Path object.
    """
    if isinstance(path, str):
        path = pathlib.Path(path)
    elif not isinstance(path, pathlib.Path):
        raise TypeError(f"Expected str or pathlib.Path, got {type(path)}")

    path = path.resolve(strict=True)

    if ensure_dir and not path.is_dir():
        raise ValueError(f"Expected directory path, got {path}")
    if not ensure_dir and not path.is_file():
        raise ValueError(f"Expected file path, got {path}")
    return path    


def to_hwc(image: np.ndarray) -> np.ndarray:
    """
    Convert image to HxWxC for Albumentations.
    Accepts HxW, HxWxC, or CxHxW. 
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(image)}")
    
    if image.ndim == 2:  # first do H, W -> 1, H, W
        image = image[None, ...]
    elif image.ndim != 3:
        raise ValueError(f"Expected 2D or 3D image, got shape {image.shape}")    
    image = np.transpose(image, (1, 2, 0)) # assumes C, H, W -> H, W, C
    
    return np.ascontiguousarray(image)


def to_chw(image: np.ndarray) -> np.ndarray:
    """
    Convert back to CxHxW (for PyTorch metrics) if needed.
    Assumes input is HxWxC or HxW (where it will be expanded to HxWx1)
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(image)}")

    if image.ndim == 2:
        image = image[..., None]
    elif image.ndim != 3:
        raise ValueError(f"Expected 2D or 3D image, got shape {image.shape}")
    return np.transpose(image, (2, 0, 1))
