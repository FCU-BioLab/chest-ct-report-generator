#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepLung Visualization Script
=============================
Generates 2D slice visualizations of 3D detections.
Draws Ground Truth (Green) and Predictions (Red) on Axial slices.
"""

import os
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from tqdm import tqdm

from detection.deep_lung.model import get_model
from detection.deep_lung.dataset import LungNodule3DDataset, collate_fn
from detection.deep_lung.utils import nms_3d

def visualize_sample(image, gt_boxes, pred_boxes, pred_scores, save_path, score_thresh=0.5):
    """
    Visualizes detections on central slices.
    args:
        image: (D, H, W) numpy
        gt_boxes: (N, 6) numpy (x1, y1, z1, x2, y2, z2)
        pred_boxes: (M, 6) numpy
        pred_scores: (M,) numpy
        save_path: path to save image
    """
    # Normalize image to 0-255 for plotting
    img_vol = image
    img_vol = (img_vol - img_vol.min()) / (img_vol.max() - img_vol.min() + 1e-6)
    
    # Filter predictions by score
    mask = pred_scores >= score_thresh
    pred_boxes = pred_boxes[mask]
    pred_scores = pred_scores[mask]
    
    # If no predictions, just plot GT centers
    targets_to_plot = []
    if len(pred_boxes) > 0:
        targets_to_plot.extend([(b, 'pred', s) for b, s in zip(pred_boxes, pred_scores)])
    
    if len(gt_boxes) > 0:
        targets_to_plot.extend([(b, 'gt', 1.0) for b in gt_boxes])
        
    if len(targets_to_plot) == 0:
        # Plot center slice just to show image
        targets_to_plot.append(([0,0, int(img_vol.shape[0]/2), 0,0, int(img_vol.shape[0]/2)+1], 'none', 0))
        
    # Group by Z-slice (approximate)
    # We can't plot ALL slices. Let's plot the center slice of each unique Nodule (GT or Pred).
    
    slices_to_plot = {} # z_index -> list of (box, type)
    
    for box, btype, score in targets_to_plot:
        # box: x1, y1, z1, x2, y2, z2
        z1, z2 = box[2], box[5]
        z_center = int((z1 + z2) / 2)
        z_center = max(0, min(z_center, img_vol.shape[0] - 1))
        
        if z_center not in slices_to_plot:
            slices_to_plot[z_center] = []
        slices_to_plot[z_center].append((box, btype, score))
        
    # Create Grid
    num_slices = len(slices_to_plot)
    if num_slices == 0: return

    cols = min(5, num_slices)
    rows = (num_slices + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.array(axes).reshape(-1)
    
    sorted_zs = sorted(slices_to_plot.keys())
    
    for i, ax in enumerate(axes):
        if i >= num_slices:
            ax.axis('off')
            continue
            
        z = sorted_zs[i]
        items = slices_to_plot[z]
        
        # Plot Slice
        ax.imshow(img_vol[z], cmap='gray')
        ax.set_title(f"Slice Z={z}")
        ax.axis('off')
        
        # Plot Boxes
        for box, btype, score in items:
            # box: x1, y1, z1, x2, y2, z2
            # on slice (H, W) -> (y, x)
            # x is axis 1 (columns), y is axis 0 (rows)
            # matplotlib rect: (x, y), w, h
            x1, y1, x2, y2 = box[0], box[1], box[3], box[4]
            w = x2 - x1
            h = y2 - y1
            
            # Check if this box actually intersects this slice?
            # Yes, we selected based on center. But let's verify intersection for robustness?
            # Visualizing the box on its center slice is standard.
            
            if btype == 'gt':
                color = 'lime'
                lw = 2
                label = "GT"
            elif btype == 'pred':
                color = 'red'
                lw = 2
                label = f"{score:.2f}"
            else:
                continue

            rect = patches.Rectangle((x1, y1), w, h, linewidth=lw, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1, label, color=color, fontsize=8, backgroundcolor='black')
            
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='vis_results')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--score_thresh', type=float, default=0.5)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Defaults
    if args.data_dir is None:
        args.data_dir = Path(__file__).resolve().parents[2] / "cache/deep_lung_cache/train"
        # Try test first
        test_dir = args.data_dir.parent / "test"
        if test_dir.exists():
            args.data_dir = test_dir
            
    print(f"Loading data from {args.data_dir}")
    dataset = LungNodule3DDataset(args.data_dir, split="test", augment=False)
    
    # Load Model
    model = get_model(num_classes=2)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Randomly sample Indices
    # indices = np.random.choice(len(dataset), min(len(dataset), args.num_samples), replace=False)
    # Prefer samples with GT nodules
    indices = range(min(len(dataset), args.num_samples))
    
    print("Generating visualizations...")
    with torch.no_grad():
        for i in indices:
            img, target = dataset[i]
            img_input = img.unsqueeze(0).to(device) # (1, 1, D, H, W)
            
            detections = model(img_input) # List of dicts
            det = detections[0]
            
            visualize_sample(
                image=img[0].numpy(),
                gt_boxes=target['boxes'].numpy(),
                pred_boxes=det['boxes'].cpu().numpy(),
                pred_scores=det['scores'].cpu().numpy(),
                save_path=os.path.join(args.output_dir, f"sample_{i}.png"),
                score_thresh=args.score_thresh
            )

def plot_training_curves(history, save_dir):
    """
    Plots training curves: Loss, Sensitivity, FP/Scan, AP, F1.
    history: dict of lists
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['loss']) + 1)
    
    # 1. Loss
    plt.figure()
    plt.plot(epochs, history['loss'], 'b-', label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'curve_loss.png'))
    plt.close()
    
    # 2. Sensitivity & AP & F1
    if 'sensitivity' in history and len(history['sensitivity']) > 0:
        val_epochs = range(1, len(history['sensitivity']) + 1) # Assumes calc every epoch
        
        plt.figure()
        plt.plot(val_epochs, history['sensitivity'], 'g-', label='Sensitivity (Recall)')
        if 'ap' in history:
            plt.plot(val_epochs, history['ap'], 'r-', label='AP@0.1')
        if 'f1' in history:
            plt.plot(val_epochs, history['f1'], 'm-', label='F1-Score')
            
        plt.title('Validation Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'curve_metrics.png'))
        plt.close()
        
    # 3. FP/Scan
    if 'fp_per_scan' in history and len(history['fp_per_scan']) > 0:
        val_epochs = range(1, len(history['fp_per_scan']) + 1)
        plt.figure()
        plt.plot(val_epochs, history['fp_per_scan'], 'r-', label='FP/Scan')
        plt.title('False Positives per Scan')
        plt.xlabel('Epoch')
        plt.ylabel('Count')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'curve_fp.png'))
        plt.close()

def plot_confusion_matrix(tp, fp, fn, save_path):
    """
    Plots a Detection Confusion Matrix.
    Since we don't have TN, we plot TP, FN (Missed), FP (False Alarm).
    Format:
                 Predicted
              Pos      Neg
    GT Pos   [ TP,     FN ]
    GT Neg   [ FP,     N/A]
    """
    # Create matrix data
    cm = np.array([[tp, fn], [fp, 0]])
    
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    classes = ['Nodule', 'Background']
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=['Pred Nodule', 'Pred Background'],
           yticklabels=['GT Nodule', 'GT Background'],
           title='Detection Confusion Matrix',
           ylabel='Ground Truth',
           xlabel='Prediction')
           
    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            text_str = f"{val}"
            if i == 1 and j == 1:
                text_str = "N/A"
                
            ax.text(j, i, text_str,
                    ha="center", va="center",
                    color="white" if val > thresh else "black", fontsize=14)
                    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    main()
