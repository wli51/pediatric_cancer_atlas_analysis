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

    def __getitem__(self, _idx):
        """Retrieve input and target image

        :param _idx: index of the image
        :type _idx: int
        :return: input image, target images, and dictionary of names
        :rtype: tuple
        """

        self.__input_name = self._ImageDataset__image_path[_idx].name
        self.__target_names = [
            str(self.__input_name).replace(
                self.__input_channel_name, 
                target_channel_names) for target_channel_names in self.__target_channel_names
        ]

        input_image = np.array(
                Image.open(self._ImageDataset__input_dir / self.__input_name).convert("I;16")
        )

        target_images = np.stack([
            np.array(Image.open(self._ImageDataset__target_dir / target_name).convert("I;16"))
            for target_name in self.__target_names
        ], axis=0)  # Stacking along a channel axis

        if self._ImageDataset__input_transform:
            input_image = self._ImageDataset__input_transform(image=input_image)["image"]

            # Reshape transformed image
            input_image = torch.from_numpy(input_image).unsqueeze(0).float()

        if self._ImageDataset__target_transform:
            transformed_target_images = []
            for channel in target_images:
                transformed_channel = self._ImageDataset__target_transform(image=channel)["image"]
                transformed_target_images.append(transformed_channel)
            target_images = torch.stack([torch.from_numpy(img).float() for img in transformed_target_images], dim=0)

        return (input_image, 
                target_images,
                {"input_name": self.__input_name, "target_names": self.__target_names},
        )        