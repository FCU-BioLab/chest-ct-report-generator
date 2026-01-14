#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize 3D Training Input (Interactive)
=========================================
Renders the EXACT tensor data being fed to the model.
Uses Plotly to generate an interactive 3D HTML file.

Logic:
1. Loads Dataset with augment=True.
2. Extracts one sample (Tensor).
3. Thresholds the volume to find 'structure' (bones, nodules, vessels).
4. Plots these voxels as a 3D Point Cloud.
5. Draws the Ground Truth Bounding Boxes in 3D.
"""

import os
import torch
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from detection.deep_lung.dataset import LungNodule3DDataset

def visualize_3d_plotly(output_file="training_input_3d.html"):
    # Config
    DATA_DIR = Path(__file__).resolve().parents[2] / "cache/deep_lung_cache/train"
    
    # 1. Load Dataset (EXACTLY as Train Loader does)
    print("Loading Dataset (Augment=True)...")
    dataset = LungNodule3DDataset(DATA_DIR, split="train", augment=True)
    
    # Find a positive sample
    sample_idx = 0
    box_count = 0
    for i in range(len(dataset)):
        # We need to see the FINAL augmented result, so we must access dataset[i]
        # But doing this for all is slow. We'll pick one that likely has a box based on raw data.
        try:
            raw = np.load(dataset.samples[i])
            if len(raw['boxes']) > 0:
                sample_idx = i
                # Check if it STILL has boxes after crop/augment
                _, target = dataset[i]
                if len(target['boxes']) > 0:
                    box_count = len(target['boxes'])
                    break
        except:
            continue
            
    print(f"Visualizing Sample {sample_idx} (Boxes: {box_count})...")
    
    # 2. Get the Tensor (The 'Truth' of what goes into training)
    # We call it again to get a fresh random augmentation if we want, or use the one we just checked.
    # dataset[i] is deterministic if seed is fixed, but usually random.
    # Let's use the one we just pulled if we kept it? No, I didn't keep it in var.
    # Calling again gives NEW augmentation. That's fine, it's still "what goes into training".
    
    img_tensor, target = dataset[sample_idx]
    
    # Convert back to numpy for plotting
    vol = img_tensor[0].numpy() # (D, H, W)
    boxes = target['boxes'].numpy() # (N, 6)
    
    D, H, W = vol.shape
    print(f"Tensor Shape: {vol.shape}")
    print(f"Boxes: {boxes}")
    
    # 3. Create 3D Scatter Plot
    # Threshold to only show 'matter'
    # High density structures and nodules.
    # Lung window 0-1. Nodule ~ 0.5+
    
    # Let's show two layers:
    # 1. Context (Lung/Vessels) - Low opacity
    # 2. High Density (Bones/Nodules) - High opacity
    
    # Points
    z_coords, y_coords, x_coords = np.where(vol > 0.1) 
    intensity = vol[vol > 0.1]
    
    # Subsample
    max_points = 30000
    if len(z_coords) > max_points:
        idx = np.random.choice(len(z_coords), max_points, replace=False)
        z_coords = z_coords[idx]
        y_coords = y_coords[idx]
        x_coords = x_coords[idx]
        intensity = intensity[idx]
        
    fig = go.Figure()
    
    # Volume Scatter
    fig.add_trace(go.Scatter3d(
        x=x_coords, y=y_coords, z=z_coords,
        mode='markers',
        marker=dict(
            size=3,
            color=intensity,
            colorscale='Viridis',
            opacity=0.2, 
            colorbar=dict(title='Intensity')
        ),
        name='Voxels'
    ))
    
    # 4. Draw Boxes
    def get_cube_lines(box):
        x1, y1, z1, x2, y2, z2 = box
        # 8 corners
        # x1,y1,z1 -> x2,y1,z1 -> x2,y2,z1 -> x1,y2,z1 -> x1,y1,z1 (Bottom)
        # x1,y1,z2 -> x2,y1,z2 -> x2,y2,z2 -> x1,y2,z2 -> x1,y1,z2 (Top)
        # Verticals
        
        # Plotly approach: explicit line segments
        # 12 lines
        lines = []
        # Bottom face
        lines.append([x1, y1, z1, x2, y1, z1])
        lines.append([x2, y1, z1, x2, y2, z1])
        lines.append([x2, y2, z1, x1, y2, z1])
        lines.append([x1, y2, z1, x1, y1, z1])
        # Top face
        lines.append([x1, y1, z2, x2, y1, z2])
        lines.append([x2, y1, z2, x2, y2, z2])
        lines.append([x2, y2, z2, x1, y2, z2])
        lines.append([x1, y2, z2, x1, y1, z2])
        # Verticals
        lines.append([x1, y1, z1, x1, y1, z2])
        lines.append([x2, y1, z1, x2, y1, z2])
        lines.append([x2, y2, z1, x2, y2, z2])
        lines.append([x1, y2, z1, x1, y2, z2])
        return lines

    for i, box in enumerate(boxes):
        lines = get_cube_lines(box)
        for line_seg in lines:
            fig.add_trace(go.Scatter3d(
                x=[line_seg[0], line_seg[3]],
                y=[line_seg[1], line_seg[4]],
                z=[line_seg[2], line_seg[5]],
                mode='lines',
                line=dict(color='red', width=6),
                showlegend=(i==0), # Only show legend once
                name='GT Nodule'
            ))
            
    # Layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[0, W], title="X"),
            yaxis=dict(range=[0, H], title="Y"),
            zaxis=dict(range=[0, D], title="Z"),
            aspectmode='data'
        ),
        title=f"Sample {sample_idx} (Interactive 3D)<br>Augment=True"
    )
    
    print(f"Saving to {output_file}...")
    fig.write_html(output_file)
    print("Done!")

if __name__ == "__main__":
    visualize_3d_plotly()
