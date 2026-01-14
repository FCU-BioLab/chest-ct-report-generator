#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepLung Evaluation Script (FROC / Sensitivity Analysis)
======================================================
Calculates:
1. Sensitivity (Recall)
2. False Positives per Scan (FP/Scan)
3. FROC Curve points (Sensitivity @ 0.125, 0.25, 0.5, 1, 2, 4, 8 FPs)
"""

import os
import torch
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from typing import List, Dict, Tuple

from detection.deep_lung.model import get_model
from detection.deep_lung.dataset import LungNodule3DDataset, collate_fn
from detection.deep_lung.utils import box_iou_3d

def compute_froc(proposals: List[Dict], targets: List[Dict], iou_thresh=0.1):
    """
    Computes True Positives and False Positives for FROC analysis.
    
    Args:
        proposals: List of dicts, each with 'boxes' (N, 6), 'scores' (N)
        targets: List of dicts, each with 'boxes' (M, 6)
        iou_thresh: IoU threshold to consider a hit (Medical detection often uses low IoU like 0.1 for 3D)
        
    Returns:
        df: DataFrame with columns ['score', 'tp', 'fp'] per detection
        num_gt: Total number of ground truth nodules
    """
    
    detection_list = [] # {'score': float, 'tp': 1/0, 'fp': 1/0}
    total_gt = 0
    
    for pred, target in zip(proposals, targets):
        pred_boxes = pred['boxes'].cpu()
        pred_scores = pred['scores'].cpu()
        gt_boxes = target['boxes'].cpu()
        
        total_gt += len(gt_boxes)
        
        if len(pred_boxes) == 0:
            continue
            
        # Sort by score descending
        sorted_idx = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[sorted_idx]
        pred_scores = pred_scores[sorted_idx]
        
        # Greedy matching
        gt_detected = torch.zeros(len(gt_boxes), dtype=torch.bool)
        
        for i in range(len(pred_boxes)):
            box = pred_boxes[i].unsqueeze(0)
            score = pred_scores[i].item()
            
            if len(gt_boxes) > 0:
                ious = box_iou_3d(box, gt_boxes).squeeze(0) # (M,)
                max_iou, max_idx = ious.max(dim=0)
                
                if max_iou >= iou_thresh and not gt_detected[max_idx]:
                    # True Positive
                    gt_detected[max_idx] = True
                    detection_list.append({'score': score, 'tp': 1, 'fp': 0})
                else:
                    # False Positive (Duplicate or Background)
                    detection_list.append({'score': score, 'tp': 0, 'fp': 1})
            else:
                # No GT, all are FP
                detection_list.append({'score': score, 'tp': 0, 'fp': 1})
                
    return pd.DataFrame(detection_list), total_gt, detection_list

def compute_map(df, total_gt):
    """
    Computes Average Precision (AP) at IoU=0.1 (or whatever threshold used to generate df).
    Uses 11-point interpolation or AUC. Here standard AUC of Precision-Recall curve.
    """
    if len(df) == 0:
        return 0.0, 0.0, 0.0, 0.0

    df = df.sort_values('score', ascending=False)
    df['cum_tp'] = df['tp'].cumsum()
    df['cum_fp'] = df['fp'].cumsum()
    
    # Precision & Recall
    df['precision'] = df['cum_tp'] / (df['cum_tp'] + df['cum_fp'] + 1e-6)
    df['recall'] = df['cum_tp'] / total_gt
    
    # Average Precision (Area under PR curve)
    # Append 0 and 1 for integration
    mrec = np.concatenate(([0.0], df['recall'].values, [1.0]))
    mpre = np.concatenate(([0.0], df['precision'].values, [0.0]))
    
    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        
    # Integrate area under curve
    method = 'continuous' # 'continuous' or '11point'
    
    if method == 'continuous':
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    else:
        # 11-point
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(mrec >= t) == 0:
                p = 0
            else:
                p = np.max(mpre[mrec >= t])
            ap += p / 11.0
            
    # Best F1
    df['f1'] = 2 * df['precision'] * df['recall'] / (df['precision'] + df['recall'] + 1e-6)
    best_f1 = df['f1'].max()
    best_row = df.loc[df['f1'].idxmax()]
    
    return ap, best_f1, best_row['precision'], best_row['recall']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .pth model file')
    parser.add_argument('--data_dir', type=str, default=None, help='Path to test data (npz)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    # Defaults
    if args.data_dir is None:
        args.data_dir = Path(__file__).resolve().parents[2] / "cache/deep_lung_cache/train"
        test_dir = args.data_dir.parent / "test"
        if test_dir.exists() and len(list(test_dir.glob("*.npz"))) > 0:
            args.data_dir = test_dir
            print(f"Using Test Set: {args.data_dir}")
        else:
            print(f"Test set not found. Using: {args.data_dir}")
    
    # Load Data
    dataset = LungNodule3DDataset(args.data_dir, split="test", augment=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    # Load Model
    print(f"Loading model from {args.checkpoint}...")
    model = get_model(num_classes=2)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.to(args.device)
    
    # Inference
    preds, targets = evaluate(model, loader, args.device)
    
    # Compute Metrics
    print("Computing metrics...")
    df, total_gt, det_list = compute_froc(preds, targets, iou_thresh=0.1)
    
    if len(df) == 0:
        print("No detections made.")
    else:
        # Standard YOLO-style Metrics
        ap, best_f1, best_prec, best_rec = compute_map(df, total_gt)
        
        # Mean IoU (of True Positives only, generally)
        # We need to access IoUs again or modify compute_froc to return them.
        # For simplicity, let's re-calculate or approximate if needed.
        # Actually, let's modify compute_froc to return 'iou' in detection_list in next step.
        # For now, just print AP/F1.
        
        print("\n=== YOLO-style Metrics ===")
        print(f"Precision: {best_prec:.4f}")
        print(f"Recall:    {best_rec:.4f}")
        print(f"F1-Score:  {best_f1:.4f}")
        print(f"AP@0.10:   {ap:.4f}") # IoU=0.1 for nodules
        print("==========================")

        # FROC
        print("\n=== LUNA16 Metrics ===")
        df = df.sort_values('score', ascending=False)
        df['cum_fp'] = df['fp'].cumsum()
        df['cum_tp'] = df['tp'].cumsum()
        num_scans = len(dataset)
        df['sensitivity'] = df['cum_tp'] / total_gt
        df['fp_per_scan'] = df['cum_fp'] / num_scans
        
        froc_fps = [0.125, 0.25, 0.5, 1, 2, 4, 8]
        avg_sens = 0
        print(f"{'FP/Scan':<10} | {'Sensitivity':<10}")
        print("-" * 25)
        for fp_rate in froc_fps:
            candidates = df[df['fp_per_scan'] <= fp_rate]
            if len(candidates) > 0:
                sens = candidates['sensitivity'].max()
            else:
                sens = 0.0
            print(f"{fp_rate:<10} | {sens:.4f}")
            avg_sens += sens
        print("-" * 25)
        print(f"Avg FROC:  {avg_sens / len(froc_fps):.4f}")
