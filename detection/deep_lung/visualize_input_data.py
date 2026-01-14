#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize Input 3D Data (GIF)
=============================
Generates a GIF animation of the 3D augmented input volume.
Helps verify what the model actually sees during training.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from pathlib import Path
from detection.deep_lung.dataset import LungNodule3DDataset

def visualize_input_gif(output_path="input_data_viz.gif"):
    # Config
    DATA_DIR = Path(__file__).resolve().parents[2] / "cache/deep_lung_cache/train"
    
    # Load Dataset with Augmentation
    # We want to see what the model sees: Cropped & Augmented
    dataset = LungNodule3DDataset(DATA_DIR, split="train", augment=True)
    
    # Find a sample with nodules
    sample_idx = 0
    found = False
    for i in range(len(dataset)):
        # Peek at raw data first to ensure we pick a positive sample
        try:
            raw_data = np.load(dataset.samples[i])
            if len(raw_data['boxes']) > 0:
                sample_idx = i
                found = True
                break
        except:
            continue
    
    if not found:
        print("No samples with nodules found.")
        return

    print(f"Visualizing Sample Index: {sample_idx} (augmented)...")
    
    # Get Augmented Sample
    # shape: (1, D, H, W)
    image_tensor, target = dataset[sample_idx]
    
    vol = image_tensor[0].numpy() # (D, H, W)
    boxes = target['boxes'].numpy() # (N, 6) x1, y1, z1, x2, y2, z2
    
    D, H, W = vol.shape
    print(f"Volume Shape: {vol.shape}")
    print(f"Boxes: {boxes}")
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Pre-compute frames
    frames = []
    
    def update(z):
        ax.clear()
        ax.imshow(vol[z], cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"Slice Z={z}/{D-1}")
        ax.axis('off')
        
        # Draw boxes
        for box in boxes:
            x1, y1, z1, x2, y2, z2 = box
            
            # Check overlap with this slice
            # Using simple integer inclusion
            if z1 <= z <= z2:
                w = x2 - x1
                h = y2 - y1
                # Draw
                rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='lime', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1-5, "Nodule", color='lime', fontsize=10)
                
    ani = animation.FuncAnimation(fig, update, frames=range(D), interval=100)
    
    print(f"Saving GIF to {output_path}...")
    ani.save(output_path, writer='pillow')
    print("Done!")

if __name__ == "__main__":
    visualize_input_gif()
