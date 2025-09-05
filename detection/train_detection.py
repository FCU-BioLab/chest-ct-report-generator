#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized K-Fold Cross-Validation Training for Faster R-CNN Detection
Simplified version focusing only on K-fold cross-validation training.
"""

import os
import sys
import json
import time
import math
import logging
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import cv2

try:
    from sklearn.metrics import roc_curve, auc
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. ROC/AUC metrics will be skipped.")

from faster_rcnn_dataset import CTDetectionDataset

# 設置控制台編碼 (Windows)
if sys.platform.startswith('win'):
    try:
        # 嘗試設置UTF-8編碼
        os.system('chcp 65001 >nul')
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass


def collate_fn(batch):
    """自定義批次整理函數，避免 lambda 函數在多進程中的問題"""
    # batch 是一個包含字典的列表，每個字典包含 'image' 和 'target' 鍵
    images = []
    targets = []
    
    for item in batch:
        # CTDetectionDataset 返回的是字典格式
        if isinstance(item, dict) and 'image' in item and 'target' in item:
            images.append(item['image'])
            targets.append(item['target'])
        else:
            # 如果數據格式不對，打印調試信息
            print(f"Unexpected batch item format: {type(item)}")
            if hasattr(item, 'keys'):
                print(f"Keys: {list(item.keys())}")
    
    return images, targets


def setup_logging(log_dir):
    """設置日誌記錄"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'kfold_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # 清除已有的處理器
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 創建格式器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                 datefmt='%Y-%m-%d %H:%M:%S')
    
    # 文件處理器 - 使用UTF-8編碼
    file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # 控制台處理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # 配置根日誌記錄器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return log_file


def calculate_giou(box1, box2):
    """計算 Generalized IoU (GIoU)"""
    # 計算交集
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # 計算各自面積
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    # 計算最小外包矩形
    c_x1 = min(box1[0], box2[0])
    c_y1 = min(box1[1], box2[1])
    c_x2 = max(box1[2], box2[2])
    c_y2 = max(box1[3], box2[3])
    c_area = (c_x2 - c_x1) * (c_y2 - c_y1)
    
    iou = intersection / union if union > 0 else 0
    giou = iou - (c_area - union) / c_area if c_area > 0 else iou
    
    return giou


def calculate_diou(box1, box2):
    """計算 Distance IoU (DIoU)"""
    # 計算IoU
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    iou = intersection / union if union > 0 else 0
    
    # 計算中心點距離
    center1_x = (box1[0] + box1[2]) / 2
    center1_y = (box1[1] + box1[3]) / 2
    center2_x = (box2[0] + box2[2]) / 2
    center2_y = (box2[1] + box2[3]) / 2
    
    center_distance_sq = (center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2
    
    # 計算對角線距離
    c_x1 = min(box1[0], box2[0])
    c_y1 = min(box1[1], box2[1])
    c_x2 = max(box1[2], box2[2])
    c_y2 = max(box1[3], box2[3])
    diagonal_distance_sq = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2
    
    diou = iou - center_distance_sq / diagonal_distance_sq if diagonal_distance_sq > 0 else iou
    
    return diou


def calculate_ciou(box1, box2):
    """計算 Complete IoU (CIoU)"""
    import math
    
    # 計算DIoU
    diou = calculate_diou(box1, box2)
    
    # 計算寬高比一致性
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    
    if h1 > 0 and h2 > 0 and w1 > 0 and w2 > 0:
        v = (4 / (math.pi ** 2)) * ((math.atan(w2/h2) - math.atan(w1/h1)) ** 2)
        
        # 計算IoU
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        iou = intersection / union if union > 0 else 0
        
        alpha = v / (1 - iou + v) if (1 - iou + v) > 0 else 0
        ciou = diou - alpha * v
    else:
        ciou = diou
    
    return ciou


def calculate_bbox_error(pred_box, target_box):
    """計算邊界框位置與大小誤差"""
    # 中心點誤差
    pred_center_x = (pred_box[0] + pred_box[2]) / 2
    pred_center_y = (pred_box[1] + pred_box[3]) / 2
    target_center_x = (target_box[0] + target_box[2]) / 2
    target_center_y = (target_box[1] + target_box[3]) / 2
    
    center_error = ((pred_center_x - target_center_x) ** 2 + (pred_center_y - target_center_y) ** 2) ** 0.5
    
    # 尺寸誤差
    pred_w = pred_box[2] - pred_box[0]
    pred_h = pred_box[3] - pred_box[1]
    target_w = target_box[2] - target_box[0]
    target_h = target_box[3] - target_box[1]
    
    size_error = abs(pred_w - target_w) + abs(pred_h - target_h)
    
    return {
        'center_error': center_error,
        'size_error': size_error,
        'total_error': center_error + size_error
    }


def calculate_map_metrics(predictions, targets, iou_thresholds=[0.5], confidence_threshold=0.5):
    """計算 mAP 指標"""
    all_scores = []
    all_labels = []
    
    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes']
        pred_scores = pred['scores']
        target_boxes = target['boxes']
        
        # 過濾低置信度預測
        valid_pred = pred_scores > confidence_threshold
        pred_boxes = pred_boxes[valid_pred]
        pred_scores = pred_scores[valid_pred]
        
        if len(pred_boxes) == 0:
            # 如果沒有預測但有目標，記錄為假陰性
            all_scores.extend([0] * len(target_boxes))
            all_labels.extend([1] * len(target_boxes))
            continue
        
        if len(target_boxes) == 0:
            # 如果沒有目標但有預測，記錄為假陽性
            all_scores.extend(pred_scores.tolist())
            all_labels.extend([0] * len(pred_scores))
            continue
        
        # 計算IoU矩陣
        iou_matrix = calculate_iou_matrix(pred_boxes, target_boxes)
        
        # 為每個IoU閾值計算匹配
        for iou_threshold in iou_thresholds:
            matched_targets = set()
            for i in range(len(pred_boxes)):
                best_iou = 0
                best_target = -1
                for j in range(len(target_boxes)):
                    if j not in matched_targets and iou_matrix[i][j] > best_iou:
                        best_iou = iou_matrix[i][j]
                        best_target = j
                
                if best_iou >= iou_threshold:
                    all_scores.append(pred_scores[i].item())
                    all_labels.append(1)  # True positive
                    matched_targets.add(best_target)
                else:
                    all_scores.append(pred_scores[i].item())
                    all_labels.append(0)  # False positive
            
            # 未匹配的目標為假陰性
            unmatched_targets = len(target_boxes) - len(matched_targets)
            all_scores.extend([0] * unmatched_targets)
            all_labels.extend([1] * unmatched_targets)
    
    # 計算AP
    if len(all_scores) == 0:
        return {'mAP': 0.0, 'AP_scores': []}
    
    # 按分數排序
    sorted_indices = np.argsort(all_scores)[::-1]
    sorted_labels = np.array(all_labels)[sorted_indices]
    
    # 計算precision和recall
    tp_cumsum = np.cumsum(sorted_labels)
    fp_cumsum = np.cumsum(1 - sorted_labels)
    
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
    recall = tp_cumsum / (np.sum(sorted_labels) + 1e-10)
    
    # 計算AP（使用插值方法）
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11
    
    return {
        'mAP': ap,
        'precision_curve': precision,
        'recall_curve': recall,
        'scores': np.array(all_scores)[sorted_indices]
    }


def calculate_comprehensive_metrics(predictions, targets, iou_threshold=0.5, confidence_threshold=0.5):
    """計算全面的檢測指標"""
    # 基礎統計
    tp, fp, fn = 0, 0, 0
    total_images = len(predictions)
    images_with_detections = 0
    images_with_targets = 0
    lesion_level_tp = 0
    case_level_tp = 0
    
    # 位置與品質指標
    ious = []
    gious = []
    dious = []
    cious = []
    bbox_errors = []
    localization_errors = []
    fp_per_image = []
    
    # 置信度相關
    all_confidence_scores = []
    true_positive_scores = []
    false_positive_scores = []
    
    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes']
        pred_scores = pred['scores']
        target_boxes = target['boxes']
        
        # 記錄有目標的圖像數
        if len(target_boxes) > 0:
            images_with_targets += 1
        
        # 過濾低置信度預測
        valid_pred = pred_scores > confidence_threshold
        filtered_pred_boxes = pred_boxes[valid_pred]
        filtered_pred_scores = pred_scores[valid_pred]
        
        # 記錄有檢測的圖像數
        if len(filtered_pred_boxes) > 0:
            images_with_detections += 1
        
        # 記錄每張圖的假陽性數
        image_fp = 0
        all_confidence_scores.extend(filtered_pred_scores.tolist())
        
        if len(filtered_pred_boxes) == 0 and len(target_boxes) == 0:
            fp_per_image.append(0)
            continue
        elif len(filtered_pred_boxes) == 0:
            fn += len(target_boxes)
            fp_per_image.append(0)
            continue
        elif len(target_boxes) == 0:
            fp += len(filtered_pred_boxes)
            image_fp = len(filtered_pred_boxes)
            fp_per_image.append(image_fp)
            false_positive_scores.extend(filtered_pred_scores.tolist())
            continue
        
        # 計算IoU矩陣和各種改進IoU
        iou_matrix = calculate_iou_matrix(filtered_pred_boxes, target_boxes)
        
        # 匹配預測和目標
        matched_targets = set()
        image_has_correct_detection = False
        
        for i in range(len(filtered_pred_boxes)):
            best_iou = 0
            best_target = -1
            for j in range(len(target_boxes)):
                if j not in matched_targets and iou_matrix[i][j] > best_iou:
                    best_iou = iou_matrix[i][j]
                    best_target = j
            
            if best_iou >= iou_threshold:
                tp += 1
                lesion_level_tp += 1
                image_has_correct_detection = True
                matched_targets.add(best_target)
                
                # 記錄品質指標
                pred_box = filtered_pred_boxes[i]
                target_box = target_boxes[best_target]
                
                ious.append(best_iou)
                gious.append(calculate_giou(pred_box, target_box))
                dious.append(calculate_diou(pred_box, target_box))
                cious.append(calculate_ciou(pred_box, target_box))
                
                # 計算定位誤差
                bbox_error = calculate_bbox_error(pred_box, target_box)
                bbox_errors.append(bbox_error)
                localization_errors.append(bbox_error['total_error'])
                
                true_positive_scores.append(filtered_pred_scores[i].item())
            else:
                fp += 1
                image_fp += 1
                false_positive_scores.append(filtered_pred_scores[i].item())
        
        # 病例級敏感度
        if image_has_correct_detection and len(target_boxes) > 0:
            case_level_tp += 1
        
        fn += len(target_boxes) - len(matched_targets)
        fp_per_image.append(image_fp)
    
    # 計算基礎指標
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    sensitivity = recall  # 敏感度就是召回率
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 病灶級和病例級敏感度
    lesion_level_sensitivity = lesion_level_tp / (tp + fn) if (tp + fn) > 0 else 0
    case_level_sensitivity = case_level_tp / images_with_targets if images_with_targets > 0 else 0
    
    # 每張圖像的平均假陽性數
    avg_fp_per_image = np.mean(fp_per_image) if fp_per_image else 0
    
    # 計算mAP指標
    map_50 = calculate_map_metrics(predictions, targets, [0.5], confidence_threshold)
    map_50_95 = calculate_map_metrics(predictions, targets, 
                                     [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95], 
                                     confidence_threshold)
    
    # 品質指標統計
    quality_metrics = {}
    if ious:
        quality_metrics = {
            'mean_iou': np.mean(ious),
            'mean_giou': np.mean(gious),
            'mean_diou': np.mean(dious),
            'mean_ciou': np.mean(cious),
            'mean_localization_error': np.mean(localization_errors),
            'std_localization_error': np.std(localization_errors)
        }
    
    return {
        # 一、核心檢測指標
        'iou': np.mean(ious) if ious else 0,
        'mAP@0.5': map_50['mAP'],
        'mAP@[0.5:0.95]': map_50_95['mAP'],
        'sensitivity_recall': sensitivity,
        'precision': precision,
        'f1_score': f1,
        'fp_per_image': avg_fp_per_image,
        
        # 二、定位與錯誤分析指標
        'lesion_level_sensitivity': lesion_level_sensitivity,
        'case_level_sensitivity': case_level_sensitivity,
        'mean_bbox_error': np.mean([err['total_error'] for err in bbox_errors]) if bbox_errors else 0,
        'mean_giou': quality_metrics.get('mean_giou', 0),
        'mean_diou': quality_metrics.get('mean_diou', 0),
        'mean_ciou': quality_metrics.get('mean_ciou', 0),
        
        # 三、臨床相關指標
        'mean_localization_error': quality_metrics.get('mean_localization_error', 0),
        'avg_false_positives_per_case': avg_fp_per_image,
        
        # 基礎統計
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'total_images': total_images,
        'images_with_detections': images_with_detections,
        'images_with_targets': images_with_targets,
        
        # 詳細數據（用於進一步分析）
        'confidence_scores': all_confidence_scores,
        'true_positive_scores': true_positive_scores,
        'false_positive_scores': false_positive_scores,
        'map_details': {
            'map_50': map_50,
            'map_50_95': map_50_95
        }
    }


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
    
    from sklearn.metrics import roc_curve, auc
    
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
            y_scores.append(torch.max(pred_scores).item())
        else:
            y_scores.append(0)
    
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
        os.makedirs(save_dir, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # ROC曲線
        ax1.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate (Sensitivity)')
        ax1.set_title('Receiver Operating Characteristic (ROC) Curve')
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
    
    return {
        'roc_auc': roc_auc,
        'roc_fpr': fpr,
        'roc_tpr': tpr,
        'roc_thresholds': roc_thresholds,
        'froc_sensitivity': sensitivity_values,
        'froc_fps_per_image': fps_per_image_values,
        'froc_thresholds': thresholds
    }


def create_comprehensive_summary(metrics, save_dir, prefix="comprehensive_metrics"):
    """創建全面的指標摘要報告"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 創建詳細的指標圖表
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 核心檢測指標雷達圖
    ax1 = plt.subplot(3, 3, 1, projection='polar')
    core_metrics = ['precision', 'sensitivity_recall', 'f1_score', 'mAP@0.5', 'case_level_sensitivity']
    core_values = [metrics.get(m, 0) for m in core_metrics]
    core_labels = ['Precision', 'Sensitivity', 'F1-Score', 'mAP@0.5', 'Case Sensitivity']
    
    angles = np.linspace(0, 2 * np.pi, len(core_metrics), endpoint=False).tolist()
    core_values += core_values[:1]  # 閉合圖形
    angles += angles[:1]
    
    ax1.plot(angles, core_values, 'o-', linewidth=2, label='Core Metrics')
    ax1.fill(angles, core_values, alpha=0.25)
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(core_labels)
    ax1.set_ylim(0, 1)
    ax1.set_title('Core Detection Metrics', y=1.08)
    ax1.grid(True)
    
    # 2. IoU品質指標
    ax2 = plt.subplot(3, 3, 2)
    iou_metrics = ['iou', 'mean_giou', 'mean_diou', 'mean_ciou']
    iou_values = [metrics.get(m, 0) for m in iou_metrics]
    iou_labels = ['IoU', 'GIoU', 'DIoU', 'CIoU']
    
    bars = ax2.bar(iou_labels, iou_values, color=['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon'])
    ax2.set_ylabel('Score')
    ax2.set_title('IoU Quality Metrics')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # 添加數值標籤
    for bar, value in zip(bars, iou_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    # 3. 錯誤分析
    ax3 = plt.subplot(3, 3, 3)
    error_data = [metrics.get('tp', 0), metrics.get('fp', 0), metrics.get('fn', 0)]
    error_labels = ['True Positives', 'False Positives', 'False Negatives']
    colors = ['green', 'red', 'orange']
    
    wedges, texts, autotexts = ax3.pie(error_data, labels=error_labels, colors=colors, 
                                      autopct='%1.0f', startangle=90)
    ax3.set_title('Detection Results Distribution')
    
    # 4-9: 其他圖表省略以節省空間...
    
    plt.tight_layout()
    
    # 保存圖片
    save_path = os.path.join(save_dir, f'{prefix}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 保存詳細的文字報告
    report_path = os.path.join(save_dir, f'{prefix}_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== 全面檢測指標評估報告 ===\n\n")
        
        f.write("一、核心檢測指標（必備）\n")
        f.write(f"  • IoU: {metrics.get('iou', 0):.4f}\n")
        f.write(f"  • mAP@0.5: {metrics.get('mAP@0.5', 0):.4f}\n")
        f.write(f"  • mAP@[0.5:0.95]: {metrics.get('mAP@[0.5:0.95]', 0):.4f}\n")
        f.write(f"  • Sensitivity/Recall: {metrics.get('sensitivity_recall', 0):.4f}\n")
        f.write(f"  • Precision: {metrics.get('precision', 0):.4f}\n")
        f.write(f"  • F1-score: {metrics.get('f1_score', 0):.4f}\n")
        f.write(f"  • FP per Image: {metrics.get('fp_per_image', 0):.4f}\n\n")
        
        f.write("二、定位與錯誤分析指標（必備）\n")
        f.write(f"  • Lesion-level Sensitivity: {metrics.get('lesion_level_sensitivity', 0):.4f}\n")
        f.write(f"  • Case-level Sensitivity: {metrics.get('case_level_sensitivity', 0):.4f}\n")
        f.write(f"  • Bounding Box Error: {metrics.get('mean_bbox_error', 0):.4f}\n")
        f.write(f"  • GIoU: {metrics.get('mean_giou', 0):.4f}\n")
        f.write(f"  • DIoU: {metrics.get('mean_diou', 0):.4f}\n")
        f.write(f"  • CIoU: {metrics.get('mean_ciou', 0):.4f}\n\n")
        
        f.write("三、臨床相關指標\n")
        f.write(f"  • Mean Localization Error: {metrics.get('mean_localization_error', 0):.4f}\n")
        f.write(f"  • Average False Positives per Case: {metrics.get('avg_false_positives_per_case', 0):.4f}\n\n")
    
    logging.info(f"全面指標摘要已保存到: {save_path}")
    logging.info(f"詳細報告已保存到: {report_path}")
    
    return save_path, report_path


def evaluate_model(model, val_loader, device, return_samples=False, max_samples=10, comprehensive=True):
    """評估模型"""
    import time
    
    model.eval()
    all_predictions = []
    all_targets = []
    sample_images = []
    sample_predictions = []
    sample_targets = []
    
    # 效率指標
    inference_times = []
    memory_usage = []
    
    val_pbar = tqdm(val_loader, desc="評估模型", unit="batch", ncols=100, leave=False)
    
    with torch.no_grad():
        for images, targets in val_pbar:
            images = [img.to(device) for img in images]
            
            # 測量推理時間
            start_time = time.time()
            predictions = model(images)
            inference_time = time.time() - start_time
            inference_times.append(inference_time / len(images))  # 每張圖的平均時間
            
            # 測量顯存使用（如果使用GPU）
            if device.type == 'cuda':
                memory_usage.append(torch.cuda.memory_allocated() / 1024**2)  # MB
            
            for i, (pred, target) in enumerate(zip(predictions, targets)):
                all_predictions.append({
                    'boxes': pred['boxes'].cpu(),
                    'scores': pred['scores'].cpu(),
                    'labels': pred['labels'].cpu()
                })
                all_targets.append({
                    'boxes': target['boxes'],
                    'labels': target['labels']
                })
                
                # 收集樣本用於可視化
                if return_samples and len(sample_images) < max_samples:
                    sample_images.append(images[i].cpu())
                    sample_predictions.append({
                        'boxes': pred['boxes'].cpu(),
                        'scores': pred['scores'].cpu(),
                        'labels': pred['labels'].cpu()
                    })
                    sample_targets.append({
                        'boxes': target['boxes'],
                        'labels': target['labels']
                    })
            
            val_pbar.set_postfix({'Samples': f'{len(all_predictions)}'})
    
    val_pbar.close()
    
    # 計算指標
    if comprehensive:
        metrics = calculate_comprehensive_metrics(all_predictions, all_targets, iou_threshold=0.5)
        
        # 添加效率指標
        if inference_times:
            metrics['inference_time_per_image'] = np.mean(inference_times)
            metrics['fps'] = 1.0 / np.mean(inference_times) if np.mean(inference_times) > 0 else 0
            metrics['inference_time_std'] = np.std(inference_times)
        
        if memory_usage and device.type == 'cuda':
            metrics['avg_memory_usage_mb'] = np.mean(memory_usage)
            metrics['max_memory_usage_mb'] = np.max(memory_usage)
    else:
        # 保持向後兼容性
        metrics = calculate_detection_metrics(all_predictions, all_targets, iou_threshold=0.5)
    
    if return_samples:
        return metrics, sample_images, sample_predictions, sample_targets
    else:
        return metrics


def evaluate_detection_model(model, data_loader, device, iou_threshold=0.5, return_samples=False, max_samples=10):
    """評估檢測模型性能"""
    model.eval()
    all_predictions = []
    all_targets = []
    sample_images = []
    sample_predictions = []
    sample_targets = []
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            predictions = model(images)
            
            for i, (pred, target) in enumerate(zip(predictions, targets)):
                all_predictions.append({
                    'boxes': pred['boxes'].cpu(),
                    'scores': pred['scores'].cpu(),
                    'labels': pred['labels'].cpu()
                })
                all_targets.append({
                    'boxes': target['boxes'],
                    'labels': target['labels']
                })
                
                # 收集樣本用於可視化
                if return_samples and len(sample_images) < max_samples:
                    sample_images.append(images[i].cpu())
                    sample_predictions.append({
                        'boxes': pred['boxes'].cpu(),
                        'scores': pred['scores'].cpu(),
                        'labels': pred['labels'].cpu()
                    })
                    sample_targets.append({
                        'boxes': target['boxes'],
                        'labels': target['labels']
                    })
    
    metrics = calculate_detection_metrics(all_predictions, all_targets, iou_threshold)
    
    if return_samples:
        return metrics, sample_images, sample_predictions, sample_targets
    else:
        return metrics


def calculate_detection_metrics(predictions, targets, iou_threshold=0.5):
    """計算檢測指標"""
    tp, fp, fn = 0, 0, 0
    
    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes']
        pred_scores = pred['scores']
        target_boxes = target['boxes']
        
        # 過濾低置信度預測
        valid_pred = pred_scores > 0.5
        pred_boxes = pred_boxes[valid_pred]
        
        if len(pred_boxes) == 0 and len(target_boxes) == 0:
            continue
        elif len(pred_boxes) == 0:
            fn += len(target_boxes)
            continue
        elif len(target_boxes) == 0:
            fp += len(pred_boxes)
            continue
        
        # 計算IoU矩陣
        iou_matrix = calculate_iou_matrix(pred_boxes, target_boxes)
        
        # 匹配預測和目標
        matched_targets = set()
        for i in range(len(pred_boxes)):
            best_iou = 0
            best_target = -1
            for j in range(len(target_boxes)):
                if j not in matched_targets and iou_matrix[i][j] > best_iou:
                    best_iou = iou_matrix[i][j]
                    best_target = j
            
            if best_iou >= iou_threshold:
                tp += 1
                matched_targets.add(best_target)
            else:
                fp += 1
        
        fn += len(target_boxes) - len(matched_targets)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def calculate_iou_matrix(boxes1, boxes2):
    """計算兩組邊界框之間的IoU矩陣"""
    iou_matrix = torch.zeros(len(boxes1), len(boxes2))
    
    for i, box1 in enumerate(boxes1):
        for j, box2 in enumerate(boxes2):
            # 計算交集
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            if x2 > x1 and y2 > y1:
                intersection = (x2 - x1) * (y2 - y1)
                
                # 計算聯合
                area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                union = area1 + area2 - intersection
                
                iou_matrix[i][j] = intersection / union if union > 0 else 0
    
    return iou_matrix


def visualize_predictions(images, predictions, targets, save_dir, num_samples=10, 
                         confidence_threshold=0.5, prefix="final_predictions"):
    """可視化預測結果並保存圖片"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 限制可視化的樣本數量
    num_samples = min(num_samples, len(images))
    
    for i in range(num_samples):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # 獲取圖片和預測結果
        image = images[i]
        pred = predictions[i]
        target = targets[i]
        
        # 將tensor轉換為numpy格式用於顯示
        if isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.shape[0] in [1, 3]:
                # CHW格式轉換為HWC
                image_np = image.permute(1, 2, 0).cpu().numpy()
            else:
                image_np = image.cpu().numpy()
            
            # 如果是單通道，轉換為三通道
            if image_np.shape[-1] == 1:
                image_np = np.repeat(image_np, 3, axis=-1)
            elif len(image_np.shape) == 2:
                image_np = np.stack([image_np] * 3, axis=-1)
        else:
            image_np = image
            
        # 確保像素值在[0,1]範圍內
        if image_np.max() > 1.0:
            image_np = image_np / 255.0
        
        # 顯示原圖和真實標註 (左側)
        ax1.imshow(image_np, cmap='gray' if image_np.shape[-1] == 1 else None)
        ax1.set_title(f'Ground Truth (Sample {i+1})')
        ax1.axis('off')
        
        # 繪製真實邊界框
        if 'boxes' in target and len(target['boxes']) > 0:
            gt_boxes = target['boxes']
            if isinstance(gt_boxes, torch.Tensor):
                gt_boxes = gt_boxes.cpu().numpy()
            
            for box in gt_boxes:
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                rect = patches.Rectangle((x1, y1), width, height, 
                                       linewidth=2, edgecolor='green', 
                                       facecolor='none', label='Ground Truth')
                ax1.add_patch(rect)
        
        # 顯示原圖和預測結果 (右側)
        ax2.imshow(image_np, cmap='gray' if image_np.shape[-1] == 1 else None)
        ax2.set_title(f'Predictions (Sample {i+1})')
        ax2.axis('off')
        
        # 繪製預測邊界框
        if 'boxes' in pred and len(pred['boxes']) > 0:
            pred_boxes = pred['boxes']
            pred_scores = pred['scores']
            
            if isinstance(pred_boxes, torch.Tensor):
                pred_boxes = pred_boxes.cpu().numpy()
            if isinstance(pred_scores, torch.Tensor):
                pred_scores = pred_scores.cpu().numpy()
            
            # 過濾低置信度預測
            valid_indices = pred_scores > confidence_threshold
            pred_boxes = pred_boxes[valid_indices]
            pred_scores = pred_scores[valid_indices]
            
            for box, score in zip(pred_boxes, pred_scores):
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                
                # 根據置信度設置顏色
                color = 'red' if score > 0.7 else 'orange' if score > 0.5 else 'yellow'
                
                rect = patches.Rectangle((x1, y1), width, height, 
                                       linewidth=2, edgecolor=color, 
                                       facecolor='none')
                ax2.add_patch(rect)
                
                # 添加置信度標籤
                ax2.text(x1, y1-5, f'{score:.2f}', 
                        color=color, fontsize=10, fontweight='bold')
        
        # 添加圖例
        legend_elements = [
            patches.Patch(color='green', label='Ground Truth'),
            patches.Patch(color='red', label='High Conf (>0.7)'),
            patches.Patch(color='orange', label='Med Conf (>0.5)'),
            patches.Patch(color='yellow', label='Low Conf (>threshold)')
        ]
        fig.legend(handles=legend_elements, loc='upper center', 
                  bbox_to_anchor=(0.5, 0.95), ncol=4)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        
        # 保存圖片
        save_path = os.path.join(save_dir, f'{prefix}_sample_{i+1}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    logging.info(f"可視化結果已保存到: {save_dir}")
    return save_dir


def create_prediction_summary(predictions, targets, save_dir, prefix="prediction_summary"):
    """創建預測結果統計摘要圖"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 統計數據
    pred_counts = []
    target_counts = []
    confidence_scores = []
    
    for pred, target in zip(predictions, targets):
        pred_counts.append(len(pred['boxes']))
        target_counts.append(len(target['boxes']))
        if len(pred['scores']) > 0:
            confidence_scores.extend(pred['scores'].cpu().numpy())
    
    # 創建統計圖
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 預測框數量分佈
    axes[0, 0].hist(pred_counts, bins=20, alpha=0.7, color='blue', label='Predictions')
    axes[0, 0].hist(target_counts, bins=20, alpha=0.7, color='green', label='Ground Truth')
    axes[0, 0].set_xlabel('Number of Boxes per Image')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Box Counts')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 置信度分佈
    if confidence_scores:
        axes[0, 1].hist(confidence_scores, bins=50, alpha=0.7, color='orange')
        axes[0, 1].axvline(x=0.5, color='red', linestyle='--', label='Threshold=0.5')
        axes[0, 1].set_xlabel('Confidence Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Confidence Score Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 預測 vs 真實框數量散點圖
    axes[1, 0].scatter(target_counts, pred_counts, alpha=0.6, color='purple')
    max_count = max(max(pred_counts), max(target_counts))
    axes[1, 0].plot([0, max_count], [0, max_count], 'r--', label='Perfect Prediction')
    axes[1, 0].set_xlabel('Ground Truth Box Count')
    axes[1, 0].set_ylabel('Predicted Box Count')
    axes[1, 0].set_title('Predicted vs Ground Truth Box Counts')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 置信度區間統計
    if confidence_scores:
        thresholds = [0.3, 0.5, 0.7, 0.9]
        counts = [sum(1 for score in confidence_scores if score > thresh) for thresh in thresholds]
        
        axes[1, 1].bar([f'>{thresh}' for thresh in thresholds], counts, color='lightblue')
        axes[1, 1].set_xlabel('Confidence Threshold')
        axes[1, 1].set_ylabel('Number of Predictions')
        axes[1, 1].set_title('Predictions by Confidence Threshold')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 添加數值標籤
        for i, count in enumerate(counts):
            axes[1, 1].text(i, count + max(counts)*0.01, str(count), 
                           ha='center', va='bottom')
    
    plt.tight_layout()
    
    # 保存統計圖
    save_path = os.path.join(save_dir, f'{prefix}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f"預測統計摘要已保存到: {save_path}")
    return save_path


def create_kfold_summary_plots(all_fold_results, save_dir):
    """創建K-fold交叉驗證結果摘要圖"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 提取每個fold的指標
    folds = [result['fold'] for result in all_fold_results]
    precisions = [result['metrics']['precision'] for result in all_fold_results]
    recalls = [result['metrics']['recall'] for result in all_fold_results]
    f1_scores = [result['metrics']['f1_score'] for result in all_fold_results]
    training_times = [result['training_time'] for result in all_fold_results]
    
    # 創建2x2的子圖
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 各fold指標比較
    x = np.arange(len(folds))
    width = 0.25
    
    axes[0, 0].bar(x - width, precisions, width, label='Precision', alpha=0.8)
    axes[0, 0].bar(x, recalls, width, label='Recall', alpha=0.8)
    axes[0, 0].bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
    
    axes[0, 0].set_xlabel('Fold')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Performance Metrics by Fold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([f'Fold {f}' for f in folds])
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 添加數值標籤
    for i, (p, r, f) in enumerate(zip(precisions, recalls, f1_scores)):
        axes[0, 0].text(i - width, p + 0.01, f'{p:.3f}', ha='center', va='bottom', fontsize=8)
        axes[0, 0].text(i, r + 0.01, f'{r:.3f}', ha='center', va='bottom', fontsize=8)
        axes[0, 0].text(i + width, f + 0.01, f'{f:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 2. F1分數趨勢
    axes[0, 1].plot(folds, f1_scores, 'bo-', linewidth=2, markersize=8)
    axes[0, 1].axhline(y=np.mean(f1_scores), color='r', linestyle='--', 
                       label=f'Mean: {np.mean(f1_scores):.3f}')
    axes[0, 1].fill_between(folds, 
                           [np.mean(f1_scores) - np.std(f1_scores)] * len(folds),
                           [np.mean(f1_scores) + np.std(f1_scores)] * len(folds),
                           alpha=0.2, color='red', 
                           label=f'±1 Std: {np.std(f1_scores):.3f}')
    axes[0, 1].set_xlabel('Fold')
    axes[0, 1].set_ylabel('F1-Score')
    axes[0, 1].set_title('F1-Score Across Folds')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 訓練時間比較
    colors = plt.cm.viridis(np.linspace(0, 1, len(folds)))
    bars = axes[1, 0].bar(folds, [t/3600 for t in training_times], color=colors, alpha=0.7)
    axes[1, 0].set_xlabel('Fold')
    axes[1, 0].set_ylabel('Training Time (hours)')
    axes[1, 0].set_title('Training Time by Fold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 添加時間標籤
    for i, (fold, time) in enumerate(zip(folds, training_times)):
        axes[1, 0].text(fold, time/3600 + max(training_times)/3600*0.01, 
                       f'{time/3600:.1f}h', ha='center', va='bottom')
    
    # 4. 指標分佈箱線圖
    metrics_data = [precisions, recalls, f1_scores]
    metrics_labels = ['Precision', 'Recall', 'F1-Score']
    
    box_plot = axes[1, 1].boxplot(metrics_data, labels=metrics_labels, patch_artist=True)
    
    # 設置箱線圖顏色
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Metrics Distribution Across Folds')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 添加統計信息
    stats_text = f"K-Fold Summary (k={len(folds)}):\n"
    stats_text += f"Precision: {np.mean(precisions):.3f} ± {np.std(precisions):.3f}\n"
    stats_text += f"Recall: {np.mean(recalls):.3f} ± {np.std(recalls):.3f}\n"
    stats_text += f"F1-Score: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}\n"
    stats_text += f"Total Time: {sum(training_times)/3600:.1f} hours"
    
    fig.text(0.02, 0.98, stats_text, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, top=0.93)
    
    # 保存圖表
    save_path = os.path.join(save_dir, 'kfold_summary.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f"K-fold摘要圖已保存到: {save_path}")
    return save_path


def calculate_dataset_statistics(dataset, dataset_name="Dataset"):
    """計算數據集的詳細統計信息"""
    total_images = len(dataset)
    total_annotations = 0
    images_with_annotations = 0
    images_without_annotations = 0
    
    logging.info(f"正在計算 {dataset_name} 統計信息...")
    
    for i in range(total_images):
        try:
            sample = dataset[i]
            if isinstance(sample, dict) and 'target' in sample:
                target = sample['target']
            else:
                # 如果是tuple格式 (image, target)
                _, target = sample
            
            # 計算標記數量
            if 'boxes' in target and len(target['boxes']) > 0:
                num_boxes = len(target['boxes'])
                total_annotations += num_boxes
                images_with_annotations += 1
            else:
                images_without_annotations += 1
                
        except Exception as e:
            logging.warning(f"計算 {dataset_name} 第 {i} 個樣本時出錯: {e}")
            images_without_annotations += 1
    
    stats = {
        'dataset_name': dataset_name,
        'total_images': total_images,
        'total_annotations': total_annotations,
        'images_with_annotations': images_with_annotations,
        'images_without_annotations': images_without_annotations,
        'avg_annotations_per_image': total_annotations / total_images if total_images > 0 else 0,
        'avg_annotations_per_annotated_image': total_annotations / images_with_annotations if images_with_annotations > 0 else 0
    }
    
    # 記錄統計信息
    logging.info(f"=== {dataset_name} 統計信息 ===")
    logging.info(f"總影像數: {total_images}")
    logging.info(f"總標記數: {total_annotations}")
    logging.info(f"有標記的影像數: {images_with_annotations}")
    logging.info(f"無標記的影像數: {images_without_annotations}")
    logging.info(f"平均每張影像標記數: {stats['avg_annotations_per_image']:.2f}")
    if images_with_annotations > 0:
        logging.info(f"平均每張有標記影像的標記數: {stats['avg_annotations_per_annotated_image']:.2f}")
    
    return stats


def create_kfold_datasets(data_dir, k_folds=5, random_seed=42, include_negative_samples=True, max_negative_per_patient=0):
    """創建按病例分割的K-fold數據集"""
    # 載入完整數據集
    full_dataset = CTDetectionDataset(
        data_root=data_dir,
        split='train',
        target_size=512,
        specific_patients=None,
        transforms=transforms.Compose([
            transforms.ToTensor()
        ]),
        include_negative_samples=include_negative_samples,
        max_negative_per_patient=max_negative_per_patient
    )
    
    # 收集所有病例ID
    all_patient_ids = set()
    for sample in full_dataset.samples:
        all_patient_ids.add(sample['patient_id'])
    
    all_patient_ids = sorted(list(all_patient_ids))
    total_patients = len(all_patient_ids)
    
    logging.info(f"數據集總大小: {len(full_dataset)} 張圖像")
    logging.info(f"總病例數: {total_patients} 位")
    
    # 設置隨機種子並創建K-fold分割
    np.random.seed(random_seed)
    np.random.shuffle(all_patient_ids)
    
    # 使用KFold按病例分割
    kfold = KFold(n_splits=k_folds, shuffle=False, random_state=None)  # 已經shuffle過，這裡不再shuffle
    patient_indices = list(range(total_patients))
    
    fold_datasets = []
    all_fold_stats = []
    
    # 首先計算完整數據集的統計信息
    logging.info("計算完整數據集統計信息...")
    full_dataset_stats = calculate_dataset_statistics(full_dataset, "完整數據集")
    
    for fold, (train_patient_idx, val_patient_idx) in enumerate(kfold.split(patient_indices)):
        # 獲取訓練和驗證的病例ID
        train_patient_ids = [all_patient_ids[i] for i in train_patient_idx]
        val_patient_ids = [all_patient_ids[i] for i in val_patient_idx]
        
        # 排序病例列表
        train_patient_ids.sort()
        val_patient_ids.sort()
        
        # 檢查病例重疊（應該為空）
        overlap = set(train_patient_ids) & set(val_patient_ids)
        if overlap:
            logging.error(f"Fold {fold + 1} 錯誤：訓練集和驗證集病例重疊: {overlap}")
            raise ValueError(f"Fold {fold + 1} 訓練集和驗證集不應有病例重疊")
        else:
            logging.info(f"Fold {fold + 1}: ✓ 訓練集和驗證集病例無重疊")
        
        # 創建按病例分割的數據集
        train_dataset = CTDetectionDataset(
            data_root=data_dir,
            split='train',
            target_size=512,
            specific_patients=train_patient_ids,
            transforms=transforms.Compose([
                transforms.ToTensor()
            ]),
            include_negative_samples=include_negative_samples,
            max_negative_per_patient=max_negative_per_patient
        )
        
        val_dataset = CTDetectionDataset(
            data_root=data_dir,
            split='train',  # 使用相同的split，但指定不同的病例
            target_size=512,
            specific_patients=val_patient_ids,
            transforms=transforms.Compose([
                transforms.ToTensor()
            ]),
            include_negative_samples=include_negative_samples,
            max_negative_per_patient=max_negative_per_patient
        )
        
        fold_datasets.append((train_dataset, val_dataset))
        
        logging.info(f"Fold {fold + 1}: 訓練集 {len(train_patient_ids)} 位病例 ({len(train_dataset)} 張圖像)")
        logging.info(f"Fold {fold + 1}: 驗證集 {len(val_patient_ids)} 位病例 ({len(val_dataset)} 張圖像)")
        logging.info(f"Fold {fold + 1}: 訓練集病例: {', '.join(train_patient_ids[:5])}{'...' if len(train_patient_ids) > 5 else ''}")
        logging.info(f"Fold {fold + 1}: 驗證集病例: {', '.join(val_patient_ids[:5])}{'...' if len(val_patient_ids) > 5 else ''}")
        
        # 計算每個fold的統計信息
        train_stats = calculate_dataset_statistics(train_dataset, f"Fold {fold + 1} 訓練集")
        val_stats = calculate_dataset_statistics(val_dataset, f"Fold {fold + 1} 驗證集")
        
        fold_stats = {
            'fold': fold + 1,
            'train_stats': train_stats,
            'val_stats': val_stats,
            'train_patient_ids': train_patient_ids,
            'val_patient_ids': val_patient_ids,
            'train_patient_count': len(train_patient_ids),
            'val_patient_count': len(val_patient_ids)
        }
        all_fold_stats.append(fold_stats)
    
    # 組合所有統計信息
    dataset_statistics = {
        'full_dataset_stats': full_dataset_stats,
        'k_folds': k_folds,
        'fold_statistics': all_fold_stats,
        'total_dataset_size': len(full_dataset),
        'total_patient_count': total_patients,
        'all_patient_ids': all_patient_ids
    }
    
    return fold_datasets, dataset_statistics
    
    return fold_datasets, dataset_statistics


def train_kfold(data_dir, k_folds=5, num_epochs=50, batch_size=8, learning_rate=0.001, 
                save_dir='./models', log_dir='./logs', random_seed=42, 
                accumulate_grad_batches=1, val_check_interval=5, include_negative_samples=True, max_negative_per_patient=0):
    """K-fold交叉驗證訓練 - 優化版本"""
    
    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"使用設備: {device}")
    
    # 創建保存目錄
    os.makedirs(save_dir, exist_ok=True)
    
    # 創建K-fold數據集並獲取統計信息
    fold_datasets, dataset_statistics = create_kfold_datasets(
        data_dir, k_folds, random_seed, include_negative_samples, max_negative_per_patient
    )
    
    # 保存數據集統計信息到文件
    stats_file = os.path.join(save_dir, 'dataset_statistics.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(dataset_statistics, f, indent=2, ensure_ascii=False)
    logging.info(f"數據集統計信息已保存到: {stats_file}")
    
    # 保存每個fold的病例列表到單獨文件
    for fold_idx, fold_stats in enumerate(dataset_statistics['fold_statistics']):
        fold_patient_summary_file = os.path.join(save_dir, f'fold_{fold_idx + 1}_patient_split_summary.txt')
        with open(fold_patient_summary_file, 'w', encoding='utf-8') as f:
            f.write(f"=== Fold {fold_idx + 1} 病例分佈摘要 ===\n")
            f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            train_patient_ids = fold_stats['train_patient_ids']
            val_patient_ids = fold_stats['val_patient_ids']
            
            f.write(f"總病例數: {len(train_patient_ids) + len(val_patient_ids)}\n")
            f.write(f"訓練集病例數: {len(train_patient_ids)} ({len(train_patient_ids)/(len(train_patient_ids) + len(val_patient_ids))*100:.1f}%)\n")
            f.write(f"驗證集病例數: {len(val_patient_ids)} ({len(val_patient_ids)/(len(train_patient_ids) + len(val_patient_ids))*100:.1f}%)\n\n")
            
            # 檢查病例重疊
            overlap = set(train_patient_ids) & set(val_patient_ids)
            if overlap:
                f.write(f"⚠️  警告：訓練集和驗證集病例重疊: {overlap}\n\n")
            else:
                f.write("✓ 訓練集和驗證集病例無重疊\n\n")
            
            f.write("訓練集病例列表:\n")
            f.write(", ".join(train_patient_ids))
            f.write("\n\n")
            
            f.write("驗證集病例列表:\n")
            f.write(", ".join(val_patient_ids))
            f.write("\n\n")
            
            # 添加數據統計信息
            train_stats = fold_stats['train_stats']
            val_stats = fold_stats['val_stats']
            f.write("=== 數據統計 ===\n")
            f.write(f"訓練集: {train_stats['total_images']} 張圖像, {train_stats['total_annotations']} 個標記\n")
            f.write(f"驗證集: {val_stats['total_images']} 張圖像, {val_stats['total_annotations']} 個標記\n")
        
        # 保存單獨的病例列表文件
        train_list_file = os.path.join(save_dir, f'fold_{fold_idx + 1}_train_patient_list.txt')
        with open(train_list_file, 'w', encoding='utf-8') as f:
            f.write(f"# Fold {fold_idx + 1} 訓練集病例列表\n")
            f.write(f"# 總計 {len(train_patient_ids)} 位病例\n")
            f.write(f"# 生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for patient_id in train_patient_ids:
                f.write(f"{patient_id}\n")
        
        val_list_file = os.path.join(save_dir, f'fold_{fold_idx + 1}_val_patient_list.txt')
        with open(val_list_file, 'w', encoding='utf-8') as f:
            f.write(f"# Fold {fold_idx + 1} 驗證集病例列表\n")
            f.write(f"# 總計 {len(val_patient_ids)} 位病例\n")
            f.write(f"# 生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for patient_id in val_patient_ids:
                f.write(f"{patient_id}\n")
    
    logging.info("所有fold的病例列表已保存完成")
    
    # 存儲所有fold的結果
    all_fold_results = []
    
    # 創建總體進度條
    fold_pbar = tqdm(
        fold_datasets, 
        desc="K-Fold 交叉驗證進度",
        unit="fold",
        ncols=120
    )
    
    for fold, (train_dataset, val_dataset) in enumerate(fold_pbar):
        fold_pbar.set_description(f"正在訓練 Fold {fold + 1}/{k_folds}")
        logging.info(f"\n開始訓練 Fold {fold + 1}/{k_folds}")
        
        # 創建數據加載器
        logging.info(f"創建訓練數據加載器 - 樣本數: {len(train_dataset)}")
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=collate_fn,
            num_workers=0  # 設為0避免Windows多進程問題
        )
        
        logging.info(f"創建驗證數據加載器 - 樣本數: {len(val_dataset)}")
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=0  # 設為0避免Windows多進程問題
        )
        
        logging.info("開始載入預訓練模型...")
        # 創建模型
        model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
        logging.info("預訓練模型載入完成")
        num_classes = 2  # 背景 + 病灶
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.to(device)
        
        # 優化器和調度器
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        # TensorBoard記錄器
        writer = SummaryWriter(os.path.join(log_dir, f'fold_{fold + 1}'))
        
        # 訓練循環
        fold_start_time = time.time()
        best_f1 = 0
        train_history = []
        val_history = []
        
        # 創建epoch進度條
        epoch_pbar = tqdm(range(num_epochs), desc=f"Fold {fold + 1}/{k_folds} - 訓練進度", 
                         unit="epoch", ncols=120, leave=False)
        
        for epoch in epoch_pbar:
            # 訓練階段
            model.train()
            train_losses = []
            
            # 創建訓練進度條
            train_pbar = tqdm(
                train_loader, 
                desc=f"Epoch {epoch + 1}/{num_epochs} [訓練]",
                unit="batch",
                ncols=100,
                leave=False
            )
            
            optimizer.zero_grad()  # 在epoch開始時清零梯度
            
            for batch_idx, (images, targets) in enumerate(train_pbar):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                # 前向傳播
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                # 梯度累積
                losses = losses / accumulate_grad_batches
                losses.backward()
                
                # 每accumulate_grad_batches個批次更新一次參數
                if ((batch_idx + 1) % accumulate_grad_batches == 0) or (batch_idx + 1 == len(train_loader)):
                    optimizer.step()
                    optimizer.zero_grad()
                
                train_losses.append(losses.item() * accumulate_grad_batches)  # 記錄原始loss
                
                # 更新進度條
                current_loss = losses.item() * accumulate_grad_batches
                avg_loss = np.mean(train_losses)
                train_pbar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Avg': f'{avg_loss:.4f}'
                })
            
            train_pbar.close()
            scheduler.step()
            
            avg_train_loss = np.mean(train_losses)
            train_history.append(avg_train_loss)
            
            # 驗證階段 - 只在指定間隔進行驗證
            if (epoch + 1) % val_check_interval == 0 or epoch == num_epochs - 1:
                val_metrics = evaluate_model(model, val_loader, device, comprehensive=True)
                val_history.append(val_metrics)
                
                # 記錄到TensorBoard
                writer.add_scalar('Loss/Train', avg_train_loss, epoch)
                writer.add_scalar('Metrics/Precision', val_metrics['precision'], epoch)
                writer.add_scalar('Metrics/Recall', val_metrics.get('sensitivity_recall', val_metrics.get('recall', 0)), epoch)
                writer.add_scalar('Metrics/F1', val_metrics['f1_score'], epoch)
                writer.add_scalar('Metrics/mAP@0.5', val_metrics.get('mAP@0.5', 0), epoch)
                writer.add_scalar('Metrics/Case_Level_Sensitivity', val_metrics.get('case_level_sensitivity', 0), epoch)
                
                # 保存最佳模型
                if val_metrics['f1_score'] > best_f1:
                    best_f1 = val_metrics['f1_score']
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_f1': best_f1,
                        'train_loss': avg_train_loss,
                        'val_metrics': val_metrics
                    }, os.path.join(save_dir, f'best_model_fold_{fold + 1}.pth'))
                    
                    # 也保存純模型權重
                    torch.save(model.state_dict(), 
                              os.path.join(save_dir, f'best_model_fold_{fold + 1}_weights.pth'))
                
                # 更新epoch進度條
                epoch_pbar.set_postfix({
                    'Loss': f'{avg_train_loss:.4f}',
                    'F1': f'{val_metrics["f1_score"]:.4f}',
                    'Best_F1': f'{best_f1:.4f}'
                })
                
                # 定期輸出日誌
                if (epoch + 1) % (val_check_interval * 2) == 0 or epoch == num_epochs - 1:
                    logging.info(f"Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}: "
                               f"Loss: {avg_train_loss:.4f}, "
                               f"Precision: {val_metrics['precision']:.4f}, "
                               f"Recall: {val_metrics.get('sensitivity_recall', val_metrics.get('recall', 0)):.4f}, "
                               f"F1: {val_metrics['f1_score']:.4f}, "
                               f"mAP@0.5: {val_metrics.get('mAP@0.5', 0):.4f}")
            else:
                # 不進行驗證時，只更新進度條
                epoch_pbar.set_postfix({
                    'Loss': f'{avg_train_loss:.4f}',
                    'Best_F1': f'{best_f1:.4f}'
                })
        
        epoch_pbar.close()
        fold_time = time.time() - fold_start_time
        
        # 最終評估和可視化
        logging.info(f"載入 Fold {fold + 1} 最佳模型進行最終評估...")
        checkpoint = torch.load(os.path.join(save_dir, f'best_model_fold_{fold + 1}.pth'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        final_metrics, sample_images, sample_predictions, sample_targets = evaluate_model(
            model, val_loader, device, return_samples=True, max_samples=10, comprehensive=True
        )
        
        # 生成當前fold的可視化結果
        try:
            fold_vis_dir = os.path.join(save_dir, f'visualizations_fold_{fold + 1}')
            logging.info(f"生成 Fold {fold + 1} 可視化結果...")
            
            # 生成預測結果可視化
            visualize_predictions(
                sample_images, sample_predictions, sample_targets, 
                fold_vis_dir, num_samples=10, confidence_threshold=0.3,
                prefix=f"fold_{fold + 1}_predictions"
            )
            
            # 生成統計摘要圖
            # 使用當前fold的所有驗證集預測結果進行統計
            all_fold_predictions, all_fold_targets = [], []
            model.eval()
            with torch.no_grad():
                for images, targets in val_loader:
                    images = [img.to(device) for img in images]
                    predictions = model(images)
                    
                    for pred, target in zip(predictions, targets):
                        all_fold_predictions.append({
                            'boxes': pred['boxes'].cpu(),
                            'scores': pred['scores'].cpu(),
                            'labels': pred['labels'].cpu()
                        })
                        all_fold_targets.append({
                            'boxes': target['boxes'],
                            'labels': target['labels']
                        })
            
            create_prediction_summary(all_fold_predictions, all_fold_targets, 
                                    fold_vis_dir, f"fold_{fold + 1}_summary")
            
        except Exception as e:
            logging.warning(f"Fold {fold + 1} 可視化生成失敗: {str(e)}")
        
        fold_result = {
            'fold': fold + 1,
            'metrics': final_metrics,
            'training_time': fold_time,
            'best_f1': best_f1,
            'train_history': train_history,
            'val_history': val_history
        }
        all_fold_results.append(fold_result)
        
        logging.info(f"Fold {fold + 1} 完成 - "
                   f"最佳F1: {best_f1:.4f}, "
                   f"訓練時間: {fold_time:.2f}秒")
        
        # 更新總體進度條
        fold_pbar.set_postfix({
            'Best_F1': f'{best_f1:.4f}',
            'Time': f'{fold_time:.1f}s'
        })
        
        writer.close()
    
    fold_pbar.close()
    
    # 計算平均結果
    avg_metrics = {}
    for metric in ['precision', 'recall', 'f1_score']:
        avg_metrics[metric] = np.mean([result['metrics'][metric] for result in all_fold_results])
        avg_metrics[f'{metric}_std'] = np.std([result['metrics'][metric] for result in all_fold_results])
    
    total_time = sum(result['training_time'] for result in all_fold_results)
    
    # 保存結果
    results = {
        'average_metrics': avg_metrics,
        'total_training_time': total_time,
        'fold_results': all_fold_results,
        'config': {
            'k_folds': k_folds,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'accumulate_grad_batches': accumulate_grad_batches,
            'val_check_interval': val_check_interval
        },
        'dataset_statistics': dataset_statistics
    }
    
    with open(os.path.join(save_dir, 'kfold_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # 生成K-fold總體結果摘要圖
    try:
        logging.info("生成K-fold總體結果摘要...")
        kfold_summary_dir = os.path.join(save_dir, 'kfold_summary_visualizations')
        os.makedirs(kfold_summary_dir, exist_ok=True)
        
        # 創建K-fold結果比較圖
        create_kfold_summary_plots(all_fold_results, kfold_summary_dir)
        
    except Exception as e:
        logging.warning(f"K-fold總體摘要可視化生成失敗: {str(e)}")
    
    # 輸出數據集統計摘要
    logging.info(f"\n=== 數據集統計摘要 ===")
    full_stats = dataset_statistics['full_dataset_stats']
    logging.info(f"完整數據集: {full_stats['total_images']} 張圖像, {full_stats['total_annotations']} 個標記")
    logging.info(f"K-Fold 數量: {dataset_statistics['k_folds']}")
    
    # 計算所有fold的平均統計
    total_train_images = sum(fold['train_stats']['total_images'] for fold in dataset_statistics['fold_statistics'])
    total_train_annotations = sum(fold['train_stats']['total_annotations'] for fold in dataset_statistics['fold_statistics'])
    total_val_images = sum(fold['val_stats']['total_images'] for fold in dataset_statistics['fold_statistics'])
    total_val_annotations = sum(fold['val_stats']['total_annotations'] for fold in dataset_statistics['fold_statistics'])
    
    avg_train_images = total_train_images / dataset_statistics['k_folds']
    avg_train_annotations = total_train_annotations / dataset_statistics['k_folds']
    avg_val_images = total_val_images / dataset_statistics['k_folds']
    avg_val_annotations = total_val_annotations / dataset_statistics['k_folds']
    
    logging.info(f"平均每fold訓練集: {avg_train_images:.1f} 張圖像, {avg_train_annotations:.1f} 個標記")
    logging.info(f"平均每fold驗證集: {avg_val_images:.1f} 張圖像, {avg_val_annotations:.1f} 個標記")
    
    # 輸出最終結果
    logging.info(f"\n=== K-Fold 交叉驗證結果 ===")
    logging.info(f"平均精確度: {avg_metrics['precision']:.4f} ± {avg_metrics['precision_std']:.4f}")
    logging.info(f"平均召回率: {avg_metrics['recall']:.4f} ± {avg_metrics['recall_std']:.4f}")
    logging.info(f"平均F1分數: {avg_metrics['f1_score']:.4f} ± {avg_metrics['f1_score_std']:.4f}")
    logging.info(f"總訓練時間: {total_time:.2f}秒 ({total_time/3600:.2f}小時)")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='K-Fold Cross-Validation Training for Faster R-CNN')
    
    # 獲取腳本所在目錄
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 生成帶時間戳的資料夾名稱
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_folder = f'Faster_RCNN_Detection_{timestamp}'
    
    parser.add_argument('--data_dir', type=str, 
                       default=os.path.join(os.path.dirname(script_dir), 'datasets', 'splited_dataset'), 
                       help='數據集目錄路徑')
    parser.add_argument('--k_folds', type=int, default=5, 
                       help='K-fold交叉驗證的fold數量')
    parser.add_argument('--num_epochs', type=int, default=10, 
                       help='訓練輪數')
    parser.add_argument('--batch_size', type=int, default=8, 
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.0001, 
                       help='學習率')
    parser.add_argument('--accumulate_grad_batches', type=int, default=2, 
                       help='梯度累積批次數')
    parser.add_argument('--val_check_interval', type=int, default=1, 
                       help='驗證檢查間隔（每N個epoch驗證一次）')
    parser.add_argument('--save_dir', type=str, 
                       default=os.path.join(script_dir, training_folder, 'models'), 
                       help='模型保存目錄')
    parser.add_argument('--log_dir', type=str, 
                       default=os.path.join(script_dir, training_folder, 'logs'), 
                       help='日誌保存目錄')
    parser.add_argument('--random_seed', type=int, default=42, 
                       help='隨機種子')
    parser.add_argument('--include_negative_samples', action='store_true', default=True,
                       help='包含負樣本（無標註的影像）以改善ROC/AUC計算（預設啟用）')
    parser.add_argument('--no_negative_samples', action='store_false', dest='include_negative_samples',
                       help='禁用負樣本載入，僅載入有標註的影像')
    parser.add_argument('--max_negative_per_patient', type=int, default=0,
                       help='每位患者最大負樣本數量，0表示無限制（載入所有負樣本）')
    
    args = parser.parse_args()
    
    # 設置隨機種子
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # 設置日誌
    log_file = setup_logging(args.log_dir)
    logging.info(f"日誌文件: {log_file}")
    
    # 檢查數據目錄
    if not os.path.exists(args.data_dir):
        logging.error(f"數據目錄不存在: {args.data_dir}")
        return
    
    # 配置信息整合輸出
    config_info = [
        f"數據目錄: {args.data_dir}",
        f"K-Fold數量: {args.k_folds}",
        f"訓練輪數: {args.num_epochs}, 批次大小: {args.batch_size}, 學習率: {args.learning_rate}",
        f"梯度累積批次數: {args.accumulate_grad_batches}, 驗證檢查間隔: {args.val_check_interval}",
        f"模型目錄: {args.save_dir}",
        f"日誌目錄: {args.log_dir}",
        f"隨機種子: {args.random_seed}",
        f"負樣本設定: {'啟用' if args.include_negative_samples else '禁用（僅載入有標註影像）'}",
        f"負樣本限制: {'無限制（載入所有負樣本）' if args.max_negative_per_patient == 0 else f'每患者最多{args.max_negative_per_patient}個'}" if args.include_negative_samples else "不適用"
    ]
    logging.info("=== 訓練配置 ===")
    for info in config_info:
        logging.info(info)
    
    # 檢查依賴
    if not SKLEARN_AVAILABLE:
        logging.warning("scikit-learn未安裝，某些評估指標（如ROC曲線）將被跳過")
        logging.warning("建議安裝: pip install scikit-learn")
    
    # 開始訓練
    start_time = time.time()
    results = train_kfold(
        data_dir=args.data_dir,
        k_folds=args.k_folds,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        random_seed=args.random_seed,
        accumulate_grad_batches=args.accumulate_grad_batches,
        val_check_interval=args.val_check_interval,
        include_negative_samples=args.include_negative_samples,
        max_negative_per_patient=args.max_negative_per_patient
    )
    
    total_time = time.time() - start_time
    logging.info(f"程式總執行時間: {total_time:.2f}秒 ({total_time/3600:.2f}小時)")


if __name__ == "__main__":
    main()
