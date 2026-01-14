#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize 3D Training Input (Static PNG)
========================================
Uses Matplotlib's 3D projection for a lightweight static image.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from detection.deep_lung.dataset import LungNodule3DDataset

def visualize_3d_static(output_file="aug_comparison_3d.png"):
    DATA_DIR = Path(__file__).resolve().parents[2] / "cache/deep_lung_cache/train"
    
    dataset = LungNodule3DDataset(DATA_DIR, split="train", augment=False)
    
    # Find positive sample
    sample_idx = 0
    for i in range(len(dataset)):
        try:
            raw = np.load(dataset.samples[i])
            if len(raw['boxes']) > 0:
                sample_idx = i
                image, boxes = dataset._crop(raw['image'], raw['boxes'], dataset.crop_size)
                if len(boxes) > 0:
                    break
        except:
            continue
            
    print(f"Visualizing Sample {sample_idx}...")
    
    # 1. Original
    raw_data = np.load(dataset.samples[sample_idx])
    img_orig, boxes_orig = dataset._crop(raw_data['image'], raw_data['boxes'], dataset.crop_size)
    
    # 2. Augmented - FORCE a geometric transform for demo
    # Instead of random, we explicitly apply FLIP X + ROTATE 90
    
    img_aug = np.flip(img_orig.copy(), axis=2)  # Flip X
    boxes_aug = boxes_orig.copy()
    if len(boxes_aug) > 0:
        W = img_orig.shape[2]
        boxes_aug[:, 2] = W - 1 - boxes_aug[:, 2]  # Flip X coord
        
    # Also rotate 90 for more obvious change
    img_aug = np.rot90(img_aug, 1, axes=(1, 2))  # Rotate 90 CCW in YX plane
    if len(boxes_aug) > 0:
        H = img_orig.shape[1]
        old_y = boxes_aug[:, 1].copy()
        old_x = boxes_aug[:, 2].copy()
        old_h = boxes_aug[:, 4].copy()
        old_w = boxes_aug[:, 5].copy()
        boxes_aug[:, 1] = H - 1 - old_x
        boxes_aug[:, 2] = old_y
        boxes_aug[:, 4] = old_w
        boxes_aug[:, 5] = old_h
    
    print(f"Original Box: {boxes_orig}")
    print(f"Augmented Box: {boxes_aug}")
    
    # Create figure
    fig = plt.figure(figsize=(16, 8))
    
    def plot_volume(ax, vol, boxes, title):
        # Downsample heavily for performance
        vol_s = vol[::4, ::4, ::4]
        
        # Get coordinates of high-intensity voxels (nodules/tissue)
        threshold = 0.2
        z, y, x = np.where(vol_s > threshold)
        intensity = vol_s[vol_s > threshold]
        
        # Scale back to original coordinates
        z, y, x = z * 4, y * 4, x * 4
        
        # Subsample points
        max_pts = 3000
        if len(z) > max_pts:
            idx = np.random.choice(len(z), max_pts, replace=False)
            z, y, x, intensity = z[idx], y[idx], x[idx], intensity[idx]
        
        # Scatter plot
        ax.scatter(x, y, z, c=intensity, cmap='gray', s=1, alpha=0.5)
        
        # Draw boxes
        for box in boxes:
            zc, yc, xc, d, h, w = box
            x1, x2 = xc - w/2, xc + w/2
            y1, y2 = yc - h/2, yc + h/2
            z1, z2 = zc - d/2, zc + d/2
            
            # Draw wireframe cube
            # Bottom
            ax.plot([x1, x2], [y1, y1], [z1, z1], 'r-', linewidth=2)
            ax.plot([x2, x2], [y1, y2], [z1, z1], 'r-', linewidth=2)
            ax.plot([x2, x1], [y2, y2], [z1, z1], 'r-', linewidth=2)
            ax.plot([x1, x1], [y2, y1], [z1, z1], 'r-', linewidth=2)
            # Top
            ax.plot([x1, x2], [y1, y1], [z2, z2], 'r-', linewidth=2)
            ax.plot([x2, x2], [y1, y2], [z2, z2], 'r-', linewidth=2)
            ax.plot([x2, x1], [y2, y2], [z2, z2], 'r-', linewidth=2)
            ax.plot([x1, x1], [y2, y1], [z2, z2], 'r-', linewidth=2)
            # Pillars
            ax.plot([x1, x1], [y1, y1], [z1, z2], 'r-', linewidth=2)
            ax.plot([x2, x2], [y1, y1], [z1, z2], 'r-', linewidth=2)
            ax.plot([x2, x2], [y2, y2], [z1, z2], 'r-', linewidth=2)
            ax.plot([x1, x1], [y2, y2], [z1, z2], 'r-', linewidth=2)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.set_xlim(0, 128)
        ax.set_ylim(0, 128)
        ax.set_zlim(0, 128)
    
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    plot_volume(ax1, img_orig, boxes_orig, "Original (Cropped)")
    
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    plot_volume(ax2, img_aug, boxes_aug, "Augmented (Flip/Rotate/Intensity)")
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    visualize_3d_static()
