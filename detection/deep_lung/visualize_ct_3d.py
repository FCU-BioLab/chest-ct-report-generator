#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Slicer-style CT Volume Visualization
========================================
Shows the EXACT data that goes into training:
1. Loaded from preprocessed .npz files
2. Random Cropped to 128x128x128
3. Augmented (Flip, Rotate, Intensity Jitter)
4. Converted to Tensor

This is the SAME pipeline as train.py's DataLoader.
"""

import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from detection.deep_lung.dataset import LungNodule3DDataset

def visualize_ct_3d(output_file="ct_3d_view.html"):
    DATA_DIR = Path(__file__).resolve().parents[2] / "cache/deep_lung_cache/train"
    
    print("=" * 60)
    print("DATA PIPELINE VERIFICATION")
    print("=" * 60)
    print(f"Data Dir: {DATA_DIR}")
    print("Pipeline: .npz -> Random Crop -> Augment (Flip/Rot/Int) -> Tensor")
    print("This is IDENTICAL to what train.py uses.")
    print("=" * 60)
    
    # Load with augment=True (SAME as train.py)
    dataset = LungNodule3DDataset(DATA_DIR, split="train", augment=True)
    
    # Find positive sample
    sample_idx = 0
    for i in range(len(dataset)):
        img_tensor, target = dataset[i]
        if len(target['boxes']) > 0:
            sample_idx = i
            break
            
    print(f"\nVisualizing Sample {sample_idx}...")
    print("Calling dataset[i] applies: Crop + Augment (Random each time)")
    
    # Get augmented data
    img_tensor, target = dataset[sample_idx]
    vol = img_tensor[0].numpy()  # (D, H, W)
    boxes = target['boxes'].numpy()  # (N, 6) x1,y1,z1,x2,y2,z2 CORNER format
    
    D, H, W = vol.shape
    print(f"Tensor Shape: {vol.shape} (This is the training input shape)")
    print(f"Value Range: [{vol.min():.3f}, {vol.max():.3f}]")
    print(f"Boxes (Corner format x1,y1,z1,x2,y2,z2): {boxes}")
    
    # Less aggressive downsampling for clarity
    stride = 2
    vol_s = vol[::stride, ::stride, ::stride]
    D_s, H_s, W_s = vol_s.shape
    
    X, Y, Z = np.mgrid[0:W_s, 0:H_s, 0:D_s]
    X = X * stride
    Y = Y * stride
    Z = Z * stride
    
    fig = go.Figure()
    
    # Isosurface 1: Lung parenchyma (faint background)
    fig.add_trace(go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=vol_s.flatten(),
        isomin=0.1,
        isomax=0.3,
        surface_count=2,
        colorscale=[[0, 'rgb(30,30,50)'], [1, 'rgb(80,80,100)']],
        opacity=0.15,
        caps=dict(x_show=False, y_show=False, z_show=False),
        name='Lung'
    ))
    
    # Isosurface 2: Soft tissue / Nodule (more visible)
    fig.add_trace(go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=vol_s.flatten(),
        isomin=0.35,
        isomax=0.7,
        surface_count=5,
        colorscale=[[0, 'rgb(200,180,150)'], [1, 'rgb(255,220,180)']],
        opacity=0.5,
        caps=dict(x_show=False, y_show=False, z_show=False),
        name='Soft Tissue/Nodule'
    ))
    
    # Isosurface 3: Dense structures (bone-like, very bright)
    fig.add_trace(go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=vol_s.flatten(),
        isomin=0.7,
        isomax=1.0,
        surface_count=3,
        colorscale=[[0, 'rgb(255,255,255)'], [1, 'rgb(255,255,255)']],
        opacity=0.9,
        caps=dict(x_show=False, y_show=False, z_show=False),
        name='Dense (Bone)'
    ))
    
    # Draw bounding boxes
    for i, box in enumerate(boxes):
        x1, y1, z1, x2, y2, z2 = box
        
        edges = [
            ([x1, x2], [y1, y1], [z1, z1]), ([x2, x2], [y1, y2], [z1, z1]),
            ([x2, x1], [y2, y2], [z1, z1]), ([x1, x1], [y2, y1], [z1, z1]),
            ([x1, x2], [y1, y1], [z2, z2]), ([x2, x2], [y1, y2], [z2, z2]),
            ([x2, x1], [y2, y2], [z2, z2]), ([x1, x1], [y2, y1], [z2, z2]),
            ([x1, x1], [y1, y1], [z1, z2]), ([x2, x2], [y1, y1], [z1, z2]),
            ([x2, x2], [y2, y2], [z1, z2]), ([x1, x1], [y2, y2], [z1, z2]),
        ]
        
        for ex, ey, ez in edges:
            fig.add_trace(go.Scatter3d(
                x=ex, y=ey, z=ez,
                mode='lines',
                line=dict(color='lime', width=8),
                showlegend=(i == 0),
                name='GT BBox'
            ))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X', range=[0, W], showbackground=False),
            yaxis=dict(title='Y', range=[0, H], showbackground=False),
            zaxis=dict(title='Z', range=[0, D], showbackground=False),
            aspectmode='data',
            bgcolor='rgb(20,20,30)'
        ),
        title=f"TRAINING INPUT (Sample {sample_idx})<br>Pipeline: Crop + Augment → This Tensor",
        paper_bgcolor='rgb(20,20,30)',
        font=dict(color='white'),
        height=800
    )
    
    print(f"\nSaving to {output_file}...")
    fig.write_html(output_file)
    print("Done! Open in browser to interact.")

if __name__ == "__main__":
    visualize_ct_3d()
