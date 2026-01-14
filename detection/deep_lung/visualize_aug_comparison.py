#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize 3D Augmentation Comparison
====================================
Side-by-Side comparsion of:
1. Original Cropped Input
2. Augmented Input (Flip, Rotate, Intensity)

This proves the augmentation logic is working.
"""

import os
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from detection.deep_lung.dataset import LungNodule3DDataset

def visualize_comparison_plotly(output_file="aug_comparison_3d.html"):
    DATA_DIR = Path(__file__).resolve().parents[2] / "cache/deep_lung_cache/train"
    
    # Load dataset with NO augmentation first to get raw method access
    # We will manually call _augment to ensure we use the same base data
    dataset = LungNodule3DDataset(DATA_DIR, split="train", augment=False)
    
    # Find positive sample
    sample_idx = 0
    for i in range(len(dataset)):
        try:
            raw = np.load(dataset.samples[i])
            if len(raw['boxes']) > 0:
                sample_idx = i
                # Check crop
                image, boxes = dataset._crop(raw['image'], raw['boxes'], dataset.crop_size)
                if len(boxes) > 0:
                    break
        except:
            continue
            
    print(f"Comparing Sample {sample_idx}...")
    
    # 1. Get Original (Cropped but not Augmented)
    raw_data = np.load(dataset.samples[sample_idx])
    img_orig, boxes_orig = dataset._crop(raw_data['image'], raw_data['boxes'], dataset.crop_size)
    
    # 2. Get Augmented
    # Loop to ensure we get a non-identity augmentation for demonstration
    max_retries = 10
    for _ in range(max_retries):
        img_aug, boxes_aug = dataset._augment(img_orig.copy(), boxes_orig.copy())
        if not np.array_equal(img_orig, img_aug):
            print("Augmentation applied successfully (Content changed).")
            # Calculate difference metric
            diff = np.abs(img_orig - img_aug).sum()
            print(f"Difference Score: {diff}")
            break
    else:
        print("Warning: Augmentation produced identical image after retries (Unlikely unless pure black/white or bug).")
    
    # Plot Side-by-Side
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=("Original (Cropped)", "Augmented (Flip/Rot/Int)")
    )
    
    def add_volume(fig, vol, boxes, row, col, name_suffix):
        # Downsample for Web Performance (Critical for Volume Rendering)
        # 128^3 = 2M voxels per plot. 2 plots = 4M. Too heavy for JS.
        # Downsample by 2: 64^3 = 262k voxels. Much better.
        
        vol_small = vol[::2, ::2, ::2]
        D_s, H_s, W_s = vol_small.shape
        
        # Grid needs to map back to original coordinates for boxes to match
        # Coordinates in vol_small are 0..63.
        # We need them to be 0..127.
        # So we scale the grid coordinates.
        
        X, Y, Z = np.mgrid[0:W_s, 0:H_s, 0:D_s]
        X = X * 2
        Y = Y * 2
        Z = Z * 2
        
        fig.add_trace(go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=vol_small.flatten(),
            isomin=0.1,
            isomax=1.0,
            opacity=0.1,
            surface_count=15,
            colorscale='Gray',
            opacityscale=[
                [0, 0],    
                [0.1, 0],
                [0.2, 0.05], # Further reduce background noise
                [0.5, 0.3], 
                [0.8, 0.8],
                [1.0, 1.0]
            ],
            caps=dict(x_show=False, y_show=False, z_show=False),
            name=f'Vol {name_suffix}'
        ), row=row, col=col)
        
        # Boxes
        for box in boxes:
            # Convert Center-Size to Corners
            zc, yc, xc, d, h, w = box
            
            x1, x2 = xc - w/2, xc + w/2
            y1, y2 = yc - h/2, yc + h/2
            z1, z2 = zc - d/2, zc + d/2
            
            # Plot box as wireframe using Scatter3d lines (Volume doesn't support lines natively)
            # Define lines: Bottom Loop, Top Loop, Vertical Pillars
            lines_x = [x1, x2, x2, x1, x1, None, x1, x2, x2, x1, x1, None, x1, x1, None, x2, x2, None, x2, x2, None, x1, x1]
            lines_y = [y1, y1, y2, y2, y1, None, y1, y1, y2, y2, y1, None, y1, y1, None, y1, y1, None, y2, y2, None, y2, y2]
            lines_z = [z1, z1, z1, z1, z1, None, z2, z2, z2, z2, z2, None, z1, z2, None, z1, z2, None, z1, z2, None, z1, z2]
            
            fig.add_trace(go.Scatter3d(
                x=lines_x, y=lines_y, z=lines_z,
                mode='lines',
                line=dict(color='red', width=5),
                showlegend=False
            ), row=row, col=col)

    add_volume(fig, img_orig, boxes_orig, 1, 1, "Orig")
    add_volume(fig, img_aug, boxes_aug, 1, 2, "Aug")
    
    fig.update_layout(title="Data Augmentation Verification")
    
    print(f"Saving comparison to {output_file}...")
    fig.write_html(output_file)
    print("Done!")

if __name__ == "__main__":
    visualize_comparison_plotly()
