#!/usr/bin/env python3
"""
Augment the already-cropped nest images in-place to generate randomized samples.

This script uses Augmentor to apply random flips, rotations, brightness/contrast
adjustments, and zooms to the cropped images in:
  data/off_nest_cropped/
  data/on_nest_cropped/

All augmented images are saved directly into those same directories, without
creating any nested subfolders.

Install:
    pip install Augmentor

Run:
    python augment_cropped.py
"""
import os
from Augmentor import Pipeline

# Directories to augment in-place
CROPPED_DIRS = [
    "data/off_nest_cropped",
    "data/on_nest_cropped"
]
# How many augmented samples to generate per directory
SAMPLE_COUNT = 3000


def augment_directory(directory, sample_count):
    """Creates and runs an Augmentor pipeline for a given directory."""
    if not os.path.isdir(directory):
        print(f"Warning: directory not found: {directory}")
        return

    # Use absolute path so Augmentor writes directly into this folder
    abs_dir = os.path.abspath(directory)
    pipeline = Pipeline(source_directory=abs_dir, output_directory=abs_dir)

    # Define augmentation operations
    #pipeline.flip_left_right(probability=0.5)
    #pipeline.flip_top_bottom(probability=0.3)
    pipeline.rotate(probability=0.7, max_left_rotation=15, max_right_rotation=15)
    pipeline.random_brightness(probability=0.5, min_factor=0.7, max_factor=1.3)
    pipeline.random_contrast(probability=0.5, min_factor=0.75, max_factor=1.25)
    #pipeline.zoom_random(probability=0.3, percentage_area=0.8)

    print(f"Generating {sample_count} augmented images in {abs_dir}...")
    pipeline.sample(sample_count)


if __name__ == "__main__":
    for dir_path in CROPPED_DIRS:
        augment_directory(dir_path, SAMPLE_COUNT)
