#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize 3D Augmentation - Lightweight HTML
=============================================
Uses Scatter3d with minimal points to ensure browser can open.
Forces geometric augmentation to show clear difference.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from detection.deep_lung.dataset import LungNodule3DDataset

def visualize_html_light(output_file="aug_comparison_3d.html"):
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
    
    raw_data = np.load(dataset.samples[sample_idx])
    img_orig, boxes_orig = dataset._crop(raw_data['image'], raw_data['boxes'], dataset.crop_size)
    
    # Force geometric augmentation: Flip X + Rotate 90
    img_aug = np.flip(img_orig.copy(), axis=2)  # Flip X
    boxes_aug = boxes_orig.copy()
    D, H, W = img_orig.shape
    
    if len(boxes_aug) > 0:
        boxes_aug[:, 2] = W - 1 - boxes_aug[:, 2]  # Flip X coord
        
    img_aug = np.rot90(img_aug, 1, axes=(1, 2))  # Rotate 90 CCW
    if len(boxes_aug) > 0:
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
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=("Original (Cropped)", "Augmented (Flip+Rotate)")
    )
    
    def add_scatter(fig, vol, boxes, row, col, name):
        # Heavy downsample: stride 4
        vol_s = vol[::4, ::4, ::4]
        z, y, x = np.where(vol_s > 0.15)
        intensity = vol_s[vol_s > 0.15]
        
        # Scale to original coords
        z, y, x = z * 4, y * 4, x * 4
        
        # Limit points
        max_pts = 5000
        if len(z) > max_pts:
            idx = np.random.choice(len(z), max_pts, replace=False)
            z, y, x, intensity = z[idx], y[idx], x[idx], intensity[idx]
        
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=2, color=intensity, colorscale='Gray', opacity=0.3),
            name=name
        ), row=row, col=col)
        
        # Draw boxes
        for box in boxes:
            zc, yc, xc, d, h, w = box
            x1, x2 = xc - w/2, xc + w/2
            y1, y2 = yc - h/2, yc + h/2
            z1, z2 = zc - d/2, zc + d/2
            
            # Wireframe
            lines_x = [x1, x2, x2, x1, x1, None, x1, x2, x2, x1, x1, None, x1, x1, None, x2, x2, None, x2, x2, None, x1, x1]
            lines_y = [y1, y1, y2, y2, y1, None, y1, y1, y2, y2, y1, None, y1, y1, None, y1, y1, None, y2, y2, None, y2, y2]
            lines_z = [z1, z1, z1, z1, z1, None, z2, z2, z2, z2, z2, None, z1, z2, None, z1, z2, None, z1, z2, None, z1, z2]
            
            fig.add_trace(go.Scatter3d(
                x=lines_x, y=lines_y, z=lines_z,
                mode='lines', line=dict(color='red', width=5),
                showlegend=False, name='GT Box'
            ), row=row, col=col)
    
    add_scatter(fig, img_orig, boxes_orig, 1, 1, "Original")
    add_scatter(fig, img_aug, boxes_aug, 1, 2, "Augmented")
    
    fig.update_layout(
        height=600,
        title="Data Augmentation: Original vs Flip+Rotate"
    )
    
    print(f"Saving to {output_file}...")
    fig.write_html(output_file)
    print("Done!")

if __name__ == "__main__":
    visualize_html_light()
