#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Faster R-CNN Training Script
===============================

Trains the DeepLung compatible 3D Faster R-CNN model.
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from pathlib import Path
import math
import numpy as np
import datetime

from detection.deep_lung.dataset import LungNodule3DDataset, collate_fn
from detection.deep_lung.model import get_model
from detection.deep_lung.evaluate import evaluate, compute_froc, compute_map
from detection.deep_lung.visualize import visualize_sample, plot_training_curves, plot_confusion_matrix

# ... (Config remains)

# Global History (or pass it around)
HISTORY = {'loss': [], 'sensitivity': [], 'fp_per_scan': [], 'ap': [], 'f1': []}

def run_validation(model, loader, device, epoch, output_dir):
    logger.info(f"Running Validation for Epoch {epoch}...")
    model.eval()
    
    # 1. Evaluation metrics
    preds, targets = evaluate(model, loader, device)
    df, total_gt, _ = compute_froc(preds, targets, iou_thresh=0.1)
    
    sensitivity = 0
    fp_per_scan = 0
    ap = 0
    best_f1 = 0
    best_prec = 0
    best_rec = 0
    
    if len(df) > 0:
        # Standard LUNA16
        df_sorted = df.sort_values('score', ascending=False)
        df_sorted['cum_tp'] = df_sorted['tp'].cumsum()
        num_scans = len(loader.dataset)
        sensitivity = df_sorted['cum_tp'].max() / total_gt if total_gt > 0 else 0
        fp_per_scan = df['fp'].sum() / num_scans
        
        # YOLO Metrics
        ap, best_f1, best_prec, best_rec, optimal_thresh = compute_map(df, total_gt)
        
        logger.info(f"Epoch {epoch} Eval Results:")
        logger.info(f"  [LUNA] Sensitivity: {sensitivity:.4f} | FP/Scan: {fp_per_scan:.2f}")
        logger.info(f"  [YOLO] AP@0.1: {ap:.4f} | F1: {best_f1:.4f} | Prec: {best_prec:.4f} | Rec: {best_rec:.4f}")
        logger.info(f"  [Optimal] Score Threshold: {optimal_thresh:.4f}")
        
        # Save metrics to text file
        with open(os.path.join(output_dir, "metrics.txt"), "a") as f:
            f.write(f"Epoch {epoch}: Sens={sensitivity:.4f}, FP/Scan={fp_per_scan:.4f}, AP={ap:.4f}, F1={best_f1:.4f}, OptThresh={optimal_thresh:.4f}\n")
            
        # Plot Confusion Matrix (for this epoch)
        tp_total = df['tp'].sum()
        fp_total = df['fp'].sum()
        fn_total = total_gt - tp_total
        plot_confusion_matrix(tp_total, fp_total, fn_total, os.path.join(output_dir, f"confusion_matrix_epoch_{epoch}.png"))
        
    else:
        logger.info(f"Epoch {epoch}: No detections made.")
        
    # Update History
    # NOTE: run_validation can be called periodically. We should handle HISTORY properly.
    # But since HISTORY is global or passed, we can append.
    # Wait, history lengths must match for plotting (epochs).
    # If eval is every 1 epoch, it matches loss. 
    # If eval is every 5 epochs, we handle plotting carefully (x-axis).
    # Current config is EVAL_INTERVAL=1.
    
    HISTORY['sensitivity'].append(sensitivity)
    HISTORY['fp_per_scan'].append(fp_per_scan)
    HISTORY['ap'].append(ap)
    HISTORY['f1'].append(best_f1)
    
    # Plot Curves
    plot_training_curves(HISTORY, output_dir)

    # 2. Visualization (Top 5 samples)
    vis_dir = os.path.join(output_dir, f"vis_epoch_{epoch}")
    os.makedirs(vis_dir, exist_ok=True)
    
    vis_count = 0
    with torch.no_grad():
        for i, (images, targets) in enumerate(loader):
            if vis_count >= 5: break
            
            images = torch.stack(images).to(device)
            detections = model(images)
            
            # Batch size usually 1 for val loader
            for j in range(len(images)):
                if vis_count >= 5: break
                
                img_np = images[j, 0].cpu().numpy()
                gt_boxes = targets[j]['boxes'].cpu().numpy()
                
                # Check for empty detections
                if len(detections[j]['boxes']) > 0:
                    pred_boxes = detections[j]['boxes'].cpu().numpy()
                    pred_scores = detections[j]['scores'].cpu().numpy()
                else:
                    pred_boxes = np.zeros((0, 6))
                    pred_scores = np.zeros((0,))
                
                visualize_sample(
                    img_np, gt_boxes, pred_boxes, pred_scores, 
                    save_path=os.path.join(vis_dir, f"sample_{vis_count}.png"),
                    score_thresh=0.01 # Show everything to debug
                )
                vis_count += 1

def main():
    logger.info(f"Using device: {DEVICE}")
    
    # ... (Dataset loading code identical) ...
    # Re-inserting loading code for completeness of replacement or assume safe?
    # I should be careful not to delete dataset logic.
    # The REPLACE block is targetting main() end or beginning?
    # No, I am observing `run_validation` and usage of `HISTORY`.
    
    # Let's replace just run_validation and modifying main loop.
    pass

# ... (Main loop logic needs update to append loss to history)

# Config
DATA_DIR = Path(__file__).resolve().parents[2] / "cache/deep_lung_cache/train"
VAL_DIR = Path(__file__).resolve().parents[2] / "cache/deep_lung_cache/val"
BATCH_SIZE = 2
LR = 1e-4  # Reverted from 1e-3 - too high caused no detections
EPOCHS = 100  # Increased from 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_INTERVAL = 1 # Run eval every N epochs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_one_epoch(model, optimizer, loader, device, epoch):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")
    for images, targets_list_raw in pbar: # Renamed to avoid conflict with processed targets
        # Move to device
        images = torch.stack(images).to(device)
        
        # Prepare targets (List of Dicts)
        targets = []
        for t in targets_list_raw:
            # Move all tensors in the target dictionary to the device
            t_dev = {k: v.to(device) for k, v in t.items()}
            
            # Ensure boxes have correct shape (N, 6)
            if 'boxes' in t_dev:
                boxes = t_dev['boxes']
                if boxes.ndim == 1 and boxes.numel() == 6:
                    boxes = boxes.unsqueeze(0)
                elif boxes.ndim == 1 and boxes.numel() == 0:
                    boxes = torch.empty((0, 6), device=device)
                t_dev['boxes'] = boxes
            targets.append(t_dev)
        
        optimizer.zero_grad()
        
        loss_dict = model(images, targets)
        
        # CenterNet returns 'loss' key with total, or sum all losses for older models
        if 'loss' in loss_dict:
            loss = loss_dict['loss']
        else:
            loss = sum(loss for loss in loss_dict.values())
        
        loss_val = loss.item()
        if not math.isfinite(loss_val):
            logger.warning(f"Loss is {loss_val}, skipping step.")
            optimizer.zero_grad()
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Show detailed losses in progress bar
        postfix = {'loss': f'{loss.item():.3f}'}
        if 'loss_hm' in loss_dict:
            postfix['hm'] = f'{loss_dict["loss_hm"].item():.3f}'
        pbar.set_postfix(postfix)
        

    return total_loss / len(loader)

def run_validation(model, loader, device, epoch, output_dir):
    logger.info(f"Running Validation for Epoch {epoch}...")
    model.eval()
    
    # 1. Evaluation metrics
    preds, targets = evaluate(model, loader, device)
    df, total_gt, _ = compute_froc(preds, targets, iou_thresh=0.1)
    
    if len(df) > 0:
        # Standard LUNA16
        df_sorted = df.sort_values('score', ascending=False)
        df_sorted['cum_tp'] = df_sorted['tp'].cumsum()
        num_scans = len(loader.dataset)
        sensitivity = df_sorted['cum_tp'].max() / total_gt if total_gt > 0 else 0
        fp_per_scan = df['fp'].sum() / num_scans
        
        # YOLO Metrics
        ap, best_f1, best_prec, best_rec, optimal_thresh = compute_map(df, total_gt)
        
        logger.info(f"Epoch {epoch} Eval Results:")
        logger.info(f"  [LUNA] Sensitivity: {sensitivity:.4f} | FP/Scan: {fp_per_scan:.2f}")
        logger.info(f"  [YOLO] AP@0.1: {ap:.4f} | F1: {best_f1:.4f} | Prec: {best_prec:.4f} | Rec: {best_rec:.4f}")
        
        # Save metrics to text file
        with open(os.path.join(output_dir, "metrics.txt"), "a") as f:
            f.write(f"Epoch {epoch}: Sens={sensitivity:.4f}, FP/Scan={fp_per_scan:.4f}, AP={ap:.4f}, F1={best_f1:.4f}\n")
    else:
        logger.info(f"Epoch {epoch}: No detections made.")

    # 2. Visualization (Top 5 samples)
    vis_dir = os.path.join(output_dir, f"vis_epoch_{epoch}")
    os.makedirs(vis_dir, exist_ok=True)
    
    vis_count = 0
    with torch.no_grad():
        for i, (images, targets) in enumerate(loader):
            if vis_count >= 5: break
            
            images = torch.stack(images).to(device)
            detections = model(images)
            
            # Batch size usually 1 for val loader
            for j in range(len(images)):
                if vis_count >= 5: break
                
                img_np = images[j, 0].cpu().numpy()
                gt_boxes = targets[j]['boxes'].cpu().numpy()
                pred_boxes = detections[j]['boxes'].cpu().numpy()
                pred_scores = detections[j]['scores'].cpu().numpy()
                
                visualize_sample(
                    img_np, gt_boxes, pred_boxes, pred_scores, 
                    save_path=os.path.join(vis_dir, f"sample_{vis_count}.png"),
                    score_thresh=0.01 # Show everything to debug
                )
                vis_count += 1

def main():
    logger.info(f"Using device: {DEVICE}")
    
    # 1. Dataset
    # Train
    if not os.path.exists(DATA_DIR):
        logger.warning(f"Data directory {DATA_DIR} not found. Constructing dummy dataset for code verification.")
        # Create directory
        os.makedirs(DATA_DIR, exist_ok=True)
        # Create dummy npz
        import numpy as np
        dummy_img = np.zeros((64, 64, 64), dtype=np.float32)
        dummy_box = np.array([[32, 32, 32, 10, 10, 10]]) # Center, Size
        np.savez_compressed(os.path.join(DATA_DIR, "dummy_001.npz"), image=dummy_img, boxes=dummy_box)

    train_dataset = LungNodule3DDataset(DATA_DIR, split="train", augment=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    
    # Val
    val_dataset = None
    if os.path.exists(VAL_DIR) and len(list(VAL_DIR.glob("*.npz"))) > 0:
        val_dataset = LungNodule3DDataset(VAL_DIR, split="val", augment=False)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn) # BS=1 for eval
        logger.info(f"Loaded {len(val_dataset)} validation samples.")
    else:
        logger.warning(f"No validation data found at {VAL_DIR}. Evaluation will be skipped.")
        val_loader = None
    
    # 2. Model
    model = get_model(num_classes=2)
    model.to(DEVICE)
    
    # 3. Optimizer and Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    # 4. Results Dir
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("detection/result") / f"deep_lung_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f"Saving results to {results_dir}")
    
    # Training Loop
    # os.makedirs("models", exist_ok=True) # Deprecated, saving to results_dir now
    
    for epoch in range(1, EPOCHS + 1):
        avg_loss = train_one_epoch(model, optimizer, train_loader, DEVICE, epoch)
        scheduler.step()  # Update learning rate
        current_lr = scheduler.get_last_lr()[0]
        logger.info(f"Epoch {epoch} - Avg Loss: {avg_loss:.4f} - LR: {current_lr:.2e}")
        
        HISTORY['loss'].append(avg_loss)
        
        # Validation
        if val_loader and epoch % EVAL_INTERVAL == 0:
            run_validation(model, val_loader, DEVICE, epoch, results_dir)
        
        # Save checkpoint
        if epoch % 5 == 0:
            save_path = results_dir / f"deep_lung_epoch_{epoch}.pth"
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved model to {save_path}")

if __name__ == "__main__":
    main()
