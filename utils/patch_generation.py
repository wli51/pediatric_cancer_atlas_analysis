import os
import glob
import pathlib
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
import matplotlib.image as mpimg
from PIL import Image
import math
import random
from random import randint

class PatchGenerator:
    """
    """
    def __init__(self, 
                 sc_feature: Tuple[str, pathlib.Path, pd.DataFrame], 
                 loaddata_csv: Tuple[str, pathlib.Path, pd.DataFrame],
                 merge_fields: List[str]=['Metadata_Plate', 'Metadata_Well', 'Metadata_Site', 'Metadata_time_point'],
                 verbose: bool=False,
                 x_col: Optional[str]=None, 
                 y_col: Optional[str]=None, 
                 ):
        """
        """

        print("Initializing PatchGenerator ...")

        if isinstance(sc_feature, pd.DataFrame):
            self.sc_feature = sc_feature
        elif isinstance(sc_feature, (str, pathlib.Path)):
            print("Reading single cell features ...")
            try:
                self.sc_feature = pd.read_parquet(sc_feature)
            except Exception as e:
                raise ValueError(f"Error reading parquet file: {e}")
            print("Completed reading single cell features")
        else:
            raise TypeError("Invalid sc_feature type")
        print(f"Single cell features with {self.sc_feature.shape[0]} rows and {self.sc_feature.shape[1]} columns")

        if isinstance(loaddata_csv, pd.DataFrame):
            self.file_info = loaddata_csv
        elif isinstance(loaddata_csv, (str, pathlib.Path)):
            print("Reading loaddata csv ...")
            try:
                self.file_info = pd.read_csv(loaddata_csv)
            except Exception as e:
                raise ValueError(f"Error reading csv file: {e}")
            print("Completed reading loaddata csv")
        else:
            raise TypeError("Invalid loaddata_csv type")
        print(f"Loaddata csv with {self.file_info.shape[0]} rows and {self.file_info.shape[1]} columns")

        if x_col is None:
            x_col = [col for col in self.sc_feature.columns if col.lower().endswith('_x')]
        else:
            if not isinstance(x_col, str) or x_col not in self.sc_feature.columns:
                raise ValueError(f"Invalid x_col: {x_col}")
        if not x_col:
            raise ValueError("No column ending with '_X' found")
        else:
            x_col = x_col[0]
        
        if y_col is None:
            y_col = [col for col in self.sc_feature.columns if col.lower().endswith('_y')]
        else:
            if not isinstance(y_col, str) or y_col not in self.sc_feature.columns:
                raise ValueError(f"Invalid y_col: {y_col}")
        if not y_col:
            raise ValueError("No column ending with '_Y' found")
        else:
            y_col = y_col[0]
        
        self.x_col = x_col
        self.y_col = y_col
        self.merge_fields = [field for field in merge_fields if field in self.file_info.columns]
        if len(self.merge_fields) == 0:
            raise ValueError("No valid merge_fields are provided, please re-examine the data and provide the merge_fields")
        
        ## Proceed with merging 
        self.merged_data = self.sc_feature.merge(
            self.file_info, 
            on=self.merge_fields, 
            how='inner')
        print(f"Merged data on fields {self.merge_fields} resulted in {self.merged_data.shape[0]} rows and {self.merged_data.shape[1]} columns")
        print("PatchGenerator initialized successfully")
    
    def generate_patch_coords(
            self,
            patch_size: int=256,
            channel_suffixes: Optional[str]=None,
            max_attempts: int=1000,
            expected_n_tiles: Optional[int]=10,
            path_prefix: Optional[str]='PathName_',
            file_prefix: Optional[str]='FileName_',
            method: str='random',
            random_seed: int=42,
            output_dir: Optional[pathlib.Path]=None,
            verbose: bool=False,
            ):
        """
        """

        log = []
        patch_coords = {}

        if not isinstance(path_prefix, str):
            raise TypeError("path_prefix should be a string")
        if not isinstance(file_prefix, str):
            raise TypeError("file_prefix should be a string")
        if not any([col.startswith(file_prefix) for col in self.file_info.columns]):
            raise ValueError(f"Invalid file_prefix: {file_prefix}")
        if not any([col.startswith(path_prefix) for col in self.file_info.columns]):
            raise ValueError(f"Invalid path_prefix: {path_prefix}")

        if not channel_suffixes:
            channel_files = [col for col in self.file_info.columns if col.startswith(file_prefix)]
            channel_paths = [col for col in self.file_info.columns if col.startswith(path_prefix)]
            channels = [col.replace(file_prefix, '') for col in channel_files]
            channels = [col for col in channels if f'{path_prefix}{col}' in channel_paths]
        if isinstance(channel_suffixes, List) and all(f'{file_prefix}{channel_suffix}' in self.file_info.columns for channel_suffix in channel_suffixes):
            channel_files = [f'{file_prefix}{channel_suffix}' for channel_suffix in channel_suffixes]
            channel_paths = [f'{path_prefix}{channel_suffix}' for channel_suffix in channel_suffixes]
            channels = channel_suffixes

        grouping_vars = [
            f'{path_prefix}{channels[0]}',
            f'{file_prefix}{channels[0]}'
            ]
        grouped = self.merged_data.groupby(grouping_vars)

        for (image_path, image_file), group in grouped:
            
            metadata = ';'.join([str(group[field].values[0]) for field in self.merge_fields])
            log.append(f"Processing {metadata}")

            if verbose:
                print(image_path, image_file)

            image_full_path = pathlib.Path(image_path) / image_file
            if not image_full_path.exists():
                log.append(f"\tFile not found: {image_full_path}")
                continue
            try:
                image = mpimg.imread(image_full_path)
            except Exception as e:
                log.append(f"\tError reading image: {e}")
                continue
            image_size = image.shape[0]
            if verbose:
                print(image.shape)
                
            coordinates = group[[self.x_col, self.y_col]].values
            if verbose:
                print(len(coordinates))
            log.append(f"\tN cells: {len(coordinates)}")

            if method == 'random':
                _patch_coords = self._generate_random_tiles(
                    image_size=image_size,
                    tile_size=patch_size,
                    coordinates=coordinates,
                    max_attempts=max_attempts,
                    expected_n_tiles=expected_n_tiles,
                    random_seed=random_seed)
            elif method == 'gaussian':
                _patch_coords = self._generate_random_tiles_gaussian(
                    image_size=image_size,
                    tile_size=patch_size,
                    coordinates=coordinates, 
                    max_attempts=max_attempts,
                    expected_n_tiles=expected_n_tiles,
                    random_seed=random_seed)
            
            log.append(f"\tN patches: {len(_patch_coords)}")

            patch_coords[metadata] = _patch_coords

        return patch_coords, log

    def generate_patches(
        self,
        patch_coords,
        patch_size: int=256,
        channels: Optional[List[str]]=None,
        path_prefix: Optional[str]='PathName_',
        file_prefix: Optional[str]='FileName_',
        output_dir: Optional[pathlib.Path]=None,
    ):
        """
        Generate patches from the coordinates and save them to the output directory.

        Parameters:
        patch_coords (dict): Dictionary containing patch coordinates.
        output_dir (str): Directory where the patches will be saved.
        """
        if not pathlib.Path(output_dir).exists():
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        if channels is None:
            # Automatically extract all channels from the file_info
            channel_files = [col for col in self.file_info.columns if col.startswith(file_prefix)]
            channel_paths = [col for col in self.file_info.columns if col.startswith(path_prefix)]
            channels = [col.replace(file_prefix, '') for col in channel_files]
            channels = [col for col in channels if f'{path_prefix}{col}' in channel_paths]
        
        for channel in channels:
            channel_dir = pathlib.Path(output_dir) / channel
            pathlib.Path(channel_dir).mkdir(parents=True, exist_ok=True)

        for key, coords in patch_coords.items():
            key_splits = key.split(';')

            dedup_file_info = self.file_info.drop_duplicates(subset=self.merge_fields).copy()
            dedup_file_info['key'] = dedup_file_info[self.merge_fields].astype(str).agg(';'.join, axis=1)

            row = dedup_file_info[dedup_file_info['key'] == key].iloc[0]
            for channel in channels:
                image_path = pathlib.Path(row[f'{path_prefix}{channel}']) / row[f'{file_prefix}{channel}']
                image = Image.open(image_path)

                for coord in coords:
                    x, y = coord
                    patch = image.crop((x, y, x + patch_size, y + patch_size))
                    patch_filename_metadata = [f"{field}={value}" for field, value in zip(self.merge_fields, key_splits)]
                    patch_filename = f"{patch_filename_metadata}_channel={channel}_size={patch_size}_x={x}_y={y}.png"
                    patch.save(pathlib.Path(output_dir) / channel / patch_filename)

        print(f"Patches saved to {output_dir}")        

    @staticmethod
    def _generate_random_tiles(
            image_size, 
            tile_size, 
            coordinates, 
            max_attempts=1000,
            expected_n_tiles=None ,
            random_seed=None):
        """
    Generate random non-overlapping tiles and retain only those containing cells.
    
    Parameters:
        image_size (int): Size of one side of the square image.
        tile_size (int): Size of the tile in pixels.
        coordinates (list of tuples): List of (x, y) coordinates representing cells.
        max_attempts (int): Maximum attempts to generate non-overlapping tiles.
    
    Returns:
        list: List of retained tiles (top-left pixel coordinates).
    """
        # Calculate unit and grid size
        unit_size = math.gcd(image_size, tile_size)
        tile_size_units = tile_size // unit_size
        grid_size_units = image_size // unit_size

        # Map coordinates to unit grid
        cell_containing_units = {
            (x // unit_size, y // unit_size) for x, y in coordinates
        }

        placed_tiles = set()  # Store frozensets of placed tiles' unit coordinates
        retained_tiles = []  # Store pixel coordinates of retained tiles

        # Generate random tiles
        attempts = 0
        if random_seed is not None:
            random.seed(random_seed)
        n_tiles = 0
        while attempts < max_attempts:
            # Randomly select a top-left unit for the tile
            top_left_x = randint(0, grid_size_units - tile_size_units)
            top_left_y = randint(0, grid_size_units - tile_size_units)
            tile_top_left = (top_left_x, top_left_y)

            # Generate units covered by this tile
            tile_units = {
                (x, y)
                for x in range(top_left_x, top_left_x + tile_size_units)
                for y in range(top_left_y, top_left_y + tile_size_units)
            }
            
            # Check for overlap with existing tiles
            if any(tile_units & placed_tile for placed_tile in placed_tiles):
                attempts += 1
                continue

            # Check if the tile contains any cell-containing units
            if tile_units & cell_containing_units:
                # Add to retained tiles
                retained_tiles.append((top_left_x * unit_size, top_left_y * unit_size))

                # Mark the tile as placed
                placed_tiles.add(frozenset(tile_units))
                n_tiles += 1

            attempts += 1

            if expected_n_tiles is not None and n_tiles > expected_n_tiles:
                break

        return retained_tiles
    
    @staticmethod
    def _generate_random_tiles_gaussian(
            image_size, 
            tile_size,
            coordinates, 
            sigma=None, 
            max_attempts=1000, 
            expected_n_tiles=None,
            random_seed=None
    ):
        """
        Sample tiles centered around points using a Gaussian distribution.
        
        Parameters:
            image_size (int): Size of one side of the square image.
            tile_size (int): Size of the tile in pixels.
            gcf (int): Greatest common factor of image size and tile size.
            coordinates (list of tuples): List of (x, y) coordinates representing points.
            sigma (float): Standard deviation for Gaussian sampling.
            max_attempts (int): Maximum attempts to place a tile.
        
        Returns:
            list: List of retained tiles (top-left pixel coordinates).
        """

        if sigma is None:
            sigma = tile_size // 10

        # Calculate unit and grid size
        unit_size = math.gcd(image_size, tile_size)
        tile_size_units = tile_size // unit_size
        grid_size_units = image_size // unit_size

        placed_tiles = set()  # Store frozensets of placed tiles' unit coordinates
        retained_tiles = []  # Store pixel coordinates of retained tiles

        if random_seed is not None:
            random.seed(random_seed)
        for cx, cy in coordinates:
            attempts = 0
            while attempts < max_attempts:
                # Sample a tile center using Gaussian distribution
                sampled_x = int(np.clip(np.random.normal(cx, sigma), 0, image_size - 1))
                sampled_y = int(np.clip(np.random.normal(cy, sigma), 0, image_size - 1))
                
                # Align the sampled center to the top-left of the tile
                top_left_x = max(0, sampled_x - tile_size // 2)
                top_left_y = max(0, sampled_y - tile_size // 2)
                top_left_x = min(top_left_x, image_size - tile_size)  # Ensure tile is within bounds
                top_left_y = min(top_left_y, image_size - tile_size)
                
                tile_top_left = (top_left_x // unit_size, top_left_y // unit_size)

                # Generate units covered by this tile
                tile_units = {
                    (x, y)
                    for x in range(tile_top_left[0], tile_top_left[0] + tile_size_units)
                    for y in range(tile_top_left[1], tile_top_left[1] + tile_size_units)
                }
                
                # Check for overlap with existing tiles
                if any(tile_units & placed_tile for placed_tile in placed_tiles):
                    attempts += 1
                    continue

                # Add to retained tiles
                retained_tiles.append((tile_top_left[0] * unit_size, tile_top_left[1] * unit_size))

                # Mark the tile as placed
                placed_tiles.add(frozenset(tile_units))
                break

        return retained_tiles
