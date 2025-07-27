#!/usr/bin/env python3
"""
Recursively batch-convert all PNGs under data/off_nest/ and data/on_nest/ to JPEG (quality 90), then remove the original PNGs.
Usage:
    pip install pillow
    python png2jpg_batch.py
"""
import os
from PIL import Image

# Directories to scan for .png images
SRC_DIRS = ["data/off_nest", "data/on_nest", "data/test_against_images"]
QUALITY  = 90  # JPEG quality (0–100)

for src_dir in SRC_DIRS:
    if not os.path.isdir(src_dir):
        print(f"Warning: directory not found: {src_dir}")
        continue

    # Walk through subfolders to catch every .png
    for root, _, files in os.walk(src_dir):
        for fname in files:
            if not fname.lower().endswith('.png'):
                continue
            png_path = os.path.join(root, fname)
            jpg_path = os.path.splitext(png_path)[0] + '.jpg'
            try:
                with Image.open(png_path) as img:
                    img.convert('RGB').save(jpg_path, quality=QUALITY)
                os.remove(png_path)
                print(f"Converted and removed: {png_path} -> {jpg_path}")
            except Exception as e:
                print(f"Failed to convert {png_path}: {e}")
