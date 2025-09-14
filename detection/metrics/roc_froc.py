#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROC和FROC曲線計算模組
包含ROC/AUC和FROC曲線的計算與可視化
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from .iou_calculations import calculate_iou_matrix

try:
    from sklearn.metrics import roc_curve, auc
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def calculate_roc_froc_curves(predictions, targets, save_dir=None):
    """計算並可視化 ROC 和 FROC 曲線"""
    if not SKLEARN_AVAILABLE:
        logging.warning("scikit-learn not available. Skipping ROC curve calculation.")
        return {
            'roc_auc': 0,
            'roc_fpr': [],
            'roc_tpr': [],
            'roc_thresholds': [],
            'froc_sensitivity': [],
            'froc_fps_per_image': [],
            'froc_thresholds': []
        }
    
    # 準備ROC數據
    y_true = []
    y_scores = []
    
    # 準備FROC數據  
    sensitivity_values = []
    fps_per_image_values = []
    thresholds = np.arange(0.1, 1.0, 0.05)
    
    # 收集所有預測分數和標籤
    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes']
        pred_scores = pred['scores']
        target_boxes = target['boxes']
        
        # 為ROC準備數據
        if len(target_boxes) > 0:
            y_true.append(1)  # 有病灶的圖像
        else:
            y_true.append(0)  # 無病灶的圖像
        
        # 使用最高置信度分數作為圖像級別的分數
        if len(pred_scores) > 0:
            y_scores.append(pred_scores.max().item())
        else:
            y_scores.append(0)
    
    # 檢查是否有足夠的類別來計算ROC曲線
    unique_labels = set(y_true)
    if len(unique_labels) < 2:
        logging.warning(f"ROC曲線計算跳過：數據集中只包含 {'正類別(有病灶)' if 1 in unique_labels else '負類別(無病灶)'} 樣本")
        logging.warning("建議：加入無病灶的影像樣本以獲得有意義的ROC/AUC指標")
        # 返回預設值
        fpr, tpr, roc_thresholds = [0, 1], [0, 1], [1, 0]
        roc_auc = float('nan')  # 明確標示為無效值
    else:
        # 計算ROC曲線
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
    
    # 計算FROC曲線
    for threshold in thresholds:
        tp, fp, fn = 0, 0, 0
        total_images = len(predictions)
        
        for pred, target in zip(predictions, targets):
            pred_boxes = pred['boxes']
            pred_scores = pred['scores']
            target_boxes = target['boxes']
            
            # 過濾低置信度預測
            valid_pred = pred_scores > threshold
            filtered_pred_boxes = pred_boxes[valid_pred]
            
            if len(filtered_pred_boxes) == 0 and len(target_boxes) == 0:
                continue
            elif len(filtered_pred_boxes) == 0:
                fn += len(target_boxes)
                continue
            elif len(target_boxes) == 0:
                fp += len(filtered_pred_boxes)
                continue
            
            # 計算IoU匹配
            iou_matrix = calculate_iou_matrix(filtered_pred_boxes, target_boxes)
            matched_targets = set()
            
            for i in range(len(filtered_pred_boxes)):
                best_iou = 0
                best_target = -1
                for j in range(len(target_boxes)):
                    if j not in matched_targets and iou_matrix[i][j] > best_iou:
                        best_iou = iou_matrix[i][j]
                        best_target = j
                
                if best_iou >= 0.5:  # IoU閾值
                    tp += 1
                    matched_targets.add(best_target)
                else:
                    fp += 1
            
            fn += len(target_boxes) - len(matched_targets)
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        fps_per_image = fp / total_images if total_images > 0 else 0
        
        sensitivity_values.append(sensitivity)
        fps_per_image_values.append(fps_per_image)
    
    # 繪製曲線
    if save_dir:
        _plot_roc_froc_curves(fpr, tpr, roc_auc, fps_per_image_values, sensitivity_values, save_dir)
    
    return {
        'roc_auc': roc_auc,
        'roc_fpr': fpr,
        'roc_tpr': tpr,
        'roc_thresholds': roc_thresholds,
        'froc_sensitivity': sensitivity_values,
        'froc_fps_per_image': fps_per_image_values,
        'froc_thresholds': thresholds
    }


def _plot_roc_froc_curves(fpr, tpr, roc_auc, fps_per_image_values, sensitivity_values, save_dir):
    """繪製並保存ROC和FROC曲線"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # ROC曲線
    if np.isnan(roc_auc):
        # 當ROC AUC為NaN時的處理
        ax1.text(0.5, 0.5, 'ROC曲線無法計算\n（數據集缺少負類別）\n請加入無病灶影像', 
                ha='center', va='center', transform=ax1.transAxes, 
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='隨機分類器')
        ax1.set_title('ROC曲線 (無法計算 - 缺少負類別)')
    else:
        ax1.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='隨機分類器')
        ax1.set_title('Receiver Operating Characteristic (ROC) Curve')
    
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate (Sensitivity)')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # FROC曲線
    ax2.plot(fps_per_image_values, sensitivity_values, color='darkgreen', lw=2, marker='o')
    ax2.set_xlabel('False Positives per Image')
    ax2.set_ylabel('Sensitivity')
    ax2.set_title('Free-response ROC (FROC) Curve')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, max(fps_per_image_values) * 1.1 if fps_per_image_values else 1])
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    # 保存圖片
    save_path = os.path.join(save_dir, 'roc_froc_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f"ROC和FROC曲線已保存到: {save_path}")
