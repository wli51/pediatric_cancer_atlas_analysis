import pathlib
from typing import Pattern, Optional

import re
import numpy as np
import torch
from albumentations import ImageOnlyTransform
from PIL import Image
from external_code_CM_vision_models.datasets.ImageDataset import ImageDataset

class ImageDatasetMultiChannel(ImageDataset):
    """Dataset for paired Brightfield and multi-channel target images."""

    def __init__(
            self, 
            _input_dir: pathlib.Path,
            _target_dir: pathlib.Path,
            _input_channel_name: str,
            _target_channel_names: list[str],
            _channel_regex_expr: Pattern[str]=r'ch\d+',
            _input_transform: Optional[ImageOnlyTransform] = None,
            _target_transform: Optional[ImageOnlyTransform] = None,
            ):
        """Calls super class initilization and additionally stores input and target definition

        :param _input_dir: pathlib object of input directory
        :type _input_dir: pathlib.Path
        :param _target_dir: pathlib object of target directory
        :type _target_dir: pathlib.Path
        :param _input_channel_name: string pattern specifying input channel name (e.g "ch1")
        :type _input_channel_name: str
        :param _target_channel_names: list of strings specifying target channel names (e.g. ["ch1", "ch2"])
        :type _target_channel_names: list[str]
        :param _channel_regex_expr: regex expression for channel
        :type _channel_regex_expr: Pattern[str]
        :param _input_transform: input transformation
        :type _input_transform: Optional[ImageOnlyTransform]
        :param _target_transform: target transformation
        :type _target_transform: Optional[ImageOnlyTransform]
        """
        
        # superclass handles directory and transformation specification
        super().__init__(
            _input_dir=_input_dir,
            _target_dir=_target_dir,
            _input_transform=_input_transform,
            _target_transform=_target_transform,
        )

        self.__input_channel_name = _input_channel_name
        self.__target_channel_names = _target_channel_names
        self.__channel_regex_expr = _channel_regex_expr

        # ensure that only tiff extensions files are included
        self._ImageDataset__image_path = [p for p in self._ImageDataset__image_path if p.suffix == ".tiff"]
        self._ImageDataset__image_path = [
            p for p in self._ImageDataset__image_path if self._extract_channel(p) == self.__input_channel_name
        ]
    
    def _extract_channel(self, 
                         path: pathlib.Path)->str:
        
        """helper function for the extraction of channel representation

        :param path: pathlib object of (image) file 
        :raises ValueError: When no pattern matches in the filename
        :raises ValueError: When multiple patterns matche in the filename
        :return: channel name substring
        :rtype: str
        """
        
        matches = re.findall(self.__channel_regex_expr, path.name)
        
        # Ensure exactly one match
        if len(matches) == 0:
            raise ValueError(f"No matches found in file: {path}")
        elif len(matches) > 1:
            raise ValueError(f"Multiple matches found in file: {path} -> {matches}")
        
        # Return the single match
        return matches[0]
    
    @property
    def input_transform(self):
        return self.__input_transform

    @property
    def target_transform(self):
        return self.__target_transform

    @property
    def input_name(self):
        if self.__input_name is None:
            raise ValueError("The input is not yet defined, so __input_name is not defined.")
        return self.__input_name

    @property
    def target_name(self):
        if self.__target_names is None or len(self.__target_names) == 0:
            raise ValueError("The target names are not yet defined, so __target_names is not defined.")
        return self.__target_names
