import pathlib
from typing import Pattern, Optional, Tuple

import re
import random
import numpy as np
import torch
from albumentations import ImageOnlyTransform
from PIL import Image
from external_code_CM_vision_models.datasets.ImageDataset import ImageDataset

class ImageDatasetMultiChannel(ImageDataset):
    """
    Dataset for paired Brightfield and multi-channel target images.
    Supports random patching of images and returns paired patches for training 
    if _patch_dim is specified, otherwise returns paired images.
    """

    def __init__(
            self, 
            _input_dir: pathlib.Path,
            _target_dir: pathlib.Path,
            _input_channel_name: str,
            _target_channel_names: list[str],
            _channel_regex_expr: Pattern[str]=r'ch\d+',
            _input_transform: Optional[ImageOnlyTransform] = None,
            _target_transform: Optional[ImageOnlyTransform] = None,
            _patch_dim: Optional[int] = None,
            _num_patches_per_image: Optional[int] = None,
            _patch_random_seed: Optional[int] = None
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
        :param _patch_dim: patch dimension, if unspecified, no patching is performed and __getitem__ function returns images
        :type _patch_dim: Optional[int]
        :param _num_patches_per_image: number of patches per image, only used if _patch_dim is specified
        :type _num_patches_per_image: Optional[int]
        :param _patch_random_seed: random seed for patching, only used if _patch_dim is specified
        :type _patch_random_seed: Optional[int]
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
        
        self._patch_dim = _patch_dim
        if _patch_dim is None and _num_patches_per_image is not None:
            raise ValueError("Number of patches per image should only be specified if patch dimension is specified.")
        else:
            self._num_patches_per_image = _num_patches_per_image
        
        # Store random seed for reproducibility
        if _patch_random_seed is not None:
            # randomly generate seed if not specified
            self._patch_random_seed = np.random.randint(0, 2**32 - 1)
        else:
            self._patch_random_seed = _patch_random_seed

        # Pre-generate patch coordinates for each image
        self.__precomputed_patches = {}
        if self.__patch_dim and self._num_patches_per_image:
            for idx, image_path in enumerate(self._ImageDataset__image_path):
                image_shape = np.array(Image.open(image_path).convert("I;16")).shape[:2]
                self.__precomputed_patches[idx] = self.__get_random_patches(image_shape)
        else:
            self.__precomputed_patches = None
    
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
    
    def __get_random_patches(self, image_size: Tuple[int, int])->list[Tuple[int, int]]:
        """Generate random coordinates for patches.
        
        :param image_size: size of the image
        :type image_size: Tuple[int, int]

        :return: list of patch coordinates
        :rtype: list[Tuple[int, int]]
        """
        max_x, max_y = image_size[0] - self.__patch_dim, image_size[1] - self.__patch_dim
        return [(random.randint(0, max_x), random.randint(0, max_y)) for _ in range(self._num_patches_per_image)]
    
    def __len__(self):
        # If patching is enabled, return the number of patches
        if self._patch_dim and self._num_patches_per_image:
            return len(self._ImageDataset__image_path) * self._num_patches_per_image
        else:
            return super().__len__()

    @property
    def input_name(self)->str:
        """Overridden property to return the input name

        :raises ValueError: When input is not yet defined
        :return: string representation of input name
        :rtype: str
        """
        if not self.__input_name:
            raise ValueError("The input is not yet defined, so __input_name is not defined.")
        return self.__input_name

    @property
    def target_names(self)->list[str]:
        """New property to return the target names as this is a multi-channel dataset class

        :raises ValueError: When target is not yet defined
        :return: list of string representations of target names
        :rtype: list[str]
        """
        if not self.__target_names:
            raise ValueError("The target is not yet defined, so __target_names is not defined.")
        return self.__target_names

    def __getitem__(self, _idx):
        """Retrieve input and target image

        :param _idx: index of the image
        :type _idx: int
        :return: input image, target images or input patch, target patches
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """

        if self._patch_dim and self._num_patches_per_image:

            # Compute image and patch index to access the correct patch coordinates and image names
            image_idx = _idx // self._num_patches_per_image
            patch_idx = _idx % self._num_patches_per_image
            self.__input_name = self._ImageDataset__image_path[image_idx].name
            self.__target_names = [
                str(self.__input_name).replace(self.__input_channel_name, target_channel)
                for target_channel in self.__target_channel_names
            ]

            # Load input and target images
            input_image = np.array(Image.open(self._ImageDataset__input_dir / self.__input_name).convert("I;16"))
            target_images = np.stack([
                np.array(Image.open(self._ImageDataset__target_dir / target_name).convert("I;16"))
                for target_name in self.__target_names
            ], axis=0)

            # Access precomputed patch coordinates and extract patches
            patch_coords = self.__precomputed_patches[image_idx][patch_idx]
            input_patch = input_image[patch_coords[0]:patch_coords[0] + self._patch_dim, patch_coords[1]:patch_coords[1] + self._patch_dim]
            target_patches = np.stack([
                channel[patch_coords[0]:patch_coords[0] + self._patch_dim, patch_coords[1]:patch_coords[1] + self._patch_dim]
                for channel in target_images
            ])

            # Apply transformations to patches
            if self._ImageDataset__input_transform:
                input_patch = torch.from_numpy(self._ImageDataset__input_transform(image=input_patch)["image"]).unsqueeze(0).float()
            if self._ImageDataset__target_transform:
                target_patches = torch.stack([
                    torch.from_numpy(self._ImageDataset__target_transform(image=patch)["image"]).float()
                    for patch in target_patches
                ])

            return input_patch, target_patches
        
        else:
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

        return input_image, target_images