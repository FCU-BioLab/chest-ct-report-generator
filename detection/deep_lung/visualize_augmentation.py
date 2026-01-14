#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize Data Augmentation
===========================
Demonstrates the effect of 3D data augmentation by plotting multiple views of the same sample.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from detection.deep_lung.dataset import LungNodule3DDataset

def main():
    # Config
    DATA_DIR = Path(__file__).resolve().parents[2] / "cache/deep_lung_cache/train"
    OUTPUT_FILE = "augmentation_demo.png"
    
    # Load Dataset with Augmentation
    dataset = LungNodule3DDataset(DATA_DIR, split="train", augment=True)
    
    # Pick a sample index (prefer one with nodules)
    sample_idx = 0
    for i in range(len(dataset)):
        # Inspect raw file to check for boxes without loading/augmenting yet
        try:
            raw_data = np.load(dataset.samples[i])
            if len(raw_data['boxes']) > 0:
                sample_idx = i
                break
        except:
            continue
            
    print(f"Visualizing sample {sample_idx} (loaded {len(dataset)} samples)...")
    
    # Generate N augmented versions
    num_views = 8
    cols = 4
    rows = 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 8))
    axes = axes.flatten()
    
    print("Generating augmented samples...")
    for i in range(num_views):
        # __getitem__ applies random augmentation
        image_tensor, target = dataset[sample_idx]
        
        # Image: (1, D, H, W) -> (D, H, W)
        img_vol = image_tensor[0].numpy()
        boxes = target['boxes'].numpy() # (N, 6) x1, y1, z1, x2, y2, z2
        
        # Pick center slice
        if len(boxes) > 0:
            # Center on first box
            z_center = int((boxes[0, 2] + boxes[0, 5]) / 2)
        else:
            z_center = img_vol.shape[0] // 2
            
        z_center = max(0, min(z_center, img_vol.shape[0]-1))
        
        # Plot
        ax = axes[i]
        ax.imshow(img_vol[z_center], cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"Augmentation {i+1}\nSlice Z={z_center}")
        ax.axis('off')
        
        # Draw boxes
        for box in boxes:
            x1, y1, z1, x2, y2, z2 = box
            
            # Check if box intersects this slice
            if z1 <= z_center <= z2:
                w = x2 - x1
                h = y2 - y1
                rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='lime', facecolor='none')
                ax.add_patch(rect)
                
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE)
    print(f"Saved demo to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
