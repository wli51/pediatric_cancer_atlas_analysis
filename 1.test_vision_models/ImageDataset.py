import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    """Dataset for paired Brightfield and multi-channel target images."""

    def __init__(self, paired_df, input_transform=None, target_transform=None):
        """
        Args:
            paired_df (pd.DataFrame): DataFrame containing paths to Brightfield and target channel images.
            input_transform (callable, optional): Transform to apply to the input image.
            target_transform (callable, optional): Transform to apply to the target images.
        """
        self.paired_df = paired_df
        self.__input_transform = input_transform
        self.__target_transform = target_transform
        self.__input_name = None  # Initialize as None
        self.__target_names = None  # Initialize as None

    def __len__(self):
        return len(self.paired_df)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the data to retrieve.

        Returns:
            tuple: (input_tensor, target_tensors)
        """
        # Load Brightfield image
        brightfield_path = self.paired_df.iloc[idx]["Brightfield"]
        brightfield_image = np.array(Image.open(brightfield_path))

        # Add channel dimension to Brightfield image
        brightfield_image = np.expand_dims(brightfield_image, axis=0)  # Shape: (1, H, W)
        self.__input_name = brightfield_path  # Update input name

        # Load target images
        target_channels = ["Alexa 488", "Alexa 647", "Alexa 568", "Alexa 488 Long (CP)", "HOECHST 33342"]
        target_images = []
        self.__target_names = []  # Update target names

        for channel in target_channels:
            target_path = self.paired_df.iloc[idx][channel]
            target_image = np.array(Image.open(target_path))
            target_image = np.expand_dims(target_image, axis=0)  # Shape: (1, H, W)
            target_images.append(target_image)
            self.__target_names.append(target_path)

        # Apply transformations
        if self.__input_transform:
            brightfield_image = self.__input_transform(image=brightfield_image)["image"]
            brightfield_image = torch.from_numpy(brightfield_image).float()
        else:
            brightfield_image = torch.from_numpy(brightfield_image).float()

        if self.__target_transform:
            target_images = [
                torch.from_numpy(self.__target_transform(image=img)["image"]).float()
                for img in target_images
            ]
        else:
            target_images = [torch.from_numpy(img).float() for img in target_images]

        # Stack target images into a single tensor (channels first)
        target_tensor = torch.cat(target_images, dim=0)  # Shape: (5, H, W)

        return brightfield_image, target_tensor

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
