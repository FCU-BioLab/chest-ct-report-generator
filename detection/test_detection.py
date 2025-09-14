#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Detection Model - 测试训练好的 Faster R-CNN 检测模型
用于测试训练结果，评估模型性能和可视化预测结果
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
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


# =============================================================================
# 配置常量 - 可根據需要修改
# =============================================================================

# 預定義的模型路徑搜索順序 (相對於detection目錄)
DEFAULT_MODEL_SEARCH_PATHS = [
    'Simple_Training_Refactored_20250905_142020/models/best_model.pth',
]

# 默認測試配置
DEFAULT_CONFIDENCE_THRESHOLDS = [0.3, 0.5, 0.7]
DEFAULT_IOU_THRESHOLDS = [0.3, 0.5, 0.7]
DEFAULT_BATCH_SIZE = 8
DEFAULT_VISUALIZE_SAMPLES = 15

# =============================================================================


def collate_fn(batch):
    """自定義批次整理函數，避免 lambda 函數在多進程中的問題"""
    images = []
    targets = []
    
    for item in batch:
        if isinstance(item, dict) and 'image' in item and 'target' in item:
            images.append(item['image'])
            targets.append(item['target'])
        else:
            print(f"Unexpected batch item format: {type(item)}")
            if hasattr(item, 'keys'):
                print(f"Keys: {list(item.keys())}")
    
    return images, targets


def setup_logging(log_dir):
    """設置日誌記錄"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'test_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
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
    
    # 2-9: 其他圖表省略以節省空間...
    
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


def calculate_iou(box1, box2):
    """計算兩個邊界框的IoU"""
    # 計算交集
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # 計算聯合
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def calculate_iou_matrix(boxes1, boxes2):
    """計算兩組邊界框之間的IoU矩陣"""
    iou_matrix = torch.zeros(len(boxes1), len(boxes2))
    
    for i, box1 in enumerate(boxes1):
        for j, box2 in enumerate(boxes2):
            iou_matrix[i][j] = calculate_iou(box1, box2)
    
    return iou_matrix


def calculate_detection_metrics(predictions, targets, iou_threshold=0.5, confidence_threshold=0.5):
    """計算檢測指標 - 使用全面評估"""
    return calculate_comprehensive_metrics(predictions, targets, iou_threshold, confidence_threshold)


def calculate_ap(predictions, targets, iou_threshold=0.5):
    """計算平均精度 (Average Precision)"""
    all_scores = []
    all_tp = []
    total_positives = 0
    
    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes']
        pred_scores = pred['scores']
        target_boxes = target['boxes']
        
        total_positives += len(target_boxes)
        
        if len(pred_boxes) == 0:
            continue
            
        if len(target_boxes) == 0:
            # 所有預測都是假陽性
            for score in pred_scores:
                all_scores.append(float(score))
                all_tp.append(0)
            continue
        
        # 計算IoU矩陣
        iou_matrix = calculate_iou_matrix(pred_boxes, target_boxes)
        
        # 為每個預測分配最佳匹配
        for i, score in enumerate(pred_scores):
            all_scores.append(float(score))
            
            # 找到最佳匹配
            best_iou = 0
            for j in range(len(target_boxes)):
                if iou_matrix[i][j] > best_iou:
                    best_iou = iou_matrix[i][j]
            
            all_tp.append(1 if best_iou >= iou_threshold else 0)
    
    if len(all_scores) == 0:
        return 0.0
    
    # 按分數排序
    sorted_indices = np.argsort(all_scores)[::-1]
    sorted_tp = np.array(all_tp)[sorted_indices]
    
    # 計算累積精度和召回率
    cumulative_tp = np.cumsum(sorted_tp)
    cumulative_fp = np.cumsum(1 - sorted_tp)
    
    precisions = cumulative_tp / (cumulative_tp + cumulative_fp)
    recalls = cumulative_tp / total_positives if total_positives > 0 else np.zeros_like(cumulative_tp)
    
    # 計算AP（11點插值）
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        mask = recalls >= t
        if np.any(mask):
            ap += np.max(precisions[mask])
    
    return ap / 11.0


def visualize_predictions(images, predictions, targets, save_dir, num_samples=10, 
                         confidence_threshold=0.5, prefix="test_predictions"):
    """可視化預測結果並保存圖片"""
    os.makedirs(save_dir, exist_ok=True)
    
    num_samples = min(num_samples, len(images))
    
    for i in range(num_samples):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # 獲取圖片和預測結果
        image = images[i]
        pred = predictions[i]
        target = targets[i]
        
        # 將tensor轉換為numpy格式用於顯示
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image_np = image.permute(1, 2, 0).cpu().numpy()
            else:
                image_np = image.cpu().numpy()
        else:
            image_np = np.array(image)
            
        # 確保像素值在[0,1]範圍內
        if image_np.max() > 1.0:
            image_np = image_np / 255.0
        
        # 處理灰度圖像
        if image_np.ndim == 3 and image_np.shape[2] == 1:
            image_np = image_np.squeeze(2)
        elif image_np.ndim == 3 and image_np.shape[0] == 1:
            image_np = image_np.squeeze(0)
        
        # 顯示原圖和真實標註 (左側)
        ax1.imshow(image_np, cmap='gray' if image_np.ndim == 2 else None)
        ax1.set_title(f'Ground Truth (Sample {i+1})')
        ax1.axis('off')
        
        # 繪製真實邊界框
        if 'boxes' in target and len(target['boxes']) > 0:
            for box in target['boxes']:
                rect = patches.Rectangle(
                    (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                    linewidth=2, edgecolor='green', facecolor='none'
                )
                ax1.add_patch(rect)
        
        # 顯示原圖和預測結果 (右側)
        ax2.imshow(image_np, cmap='gray' if image_np.ndim == 2 else None)
        ax2.set_title(f'Predictions (Sample {i+1})')
        ax2.axis('off')
        
        # 繪製預測邊界框
        if 'boxes' in pred and len(pred['boxes']) > 0:
            boxes = pred['boxes']
            scores = pred['scores']
            
            for box, score in zip(boxes, scores):
                if score >= confidence_threshold:
                    # 根據置信度選擇顏色
                    if score > 0.7:
                        color = 'red'
                    elif score > 0.5:
                        color = 'orange'
                    else:
                        color = 'yellow'
                    
                    rect = patches.Rectangle(
                        (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                        linewidth=2, edgecolor=color, facecolor='none'
                    )
                    ax2.add_patch(rect)
                    
                    # 添加置信度標籤
                    ax2.text(box[0], box[1] - 5, f'{score:.2f}', 
                            fontsize=8, color=color, weight='bold')
        
        # 添加圖例
        legend_elements = [
            patches.Patch(color='green', label='Ground Truth'),
            patches.Patch(color='red', label='High Conf (>0.7)'),
            patches.Patch(color='orange', label='Med Conf (>0.5)'),
            patches.Patch(color='yellow', label=f'Low Conf (>{confidence_threshold})')
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


def create_confidence_analysis(predictions, save_dir, prefix="confidence_analysis"):
    """創建置信度分析圖"""
    os.makedirs(save_dir, exist_ok=True)
    
    all_scores = []
    for pred in predictions:
        if 'scores' in pred and len(pred['scores']) > 0:
            all_scores.extend(pred['scores'].cpu().numpy())
    
    if not all_scores:
        logging.warning("沒有預測分數可以分析")
        return None
    
    # 創建置信度分析圖
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 置信度分佈直方圖
    axes[0, 0].hist(all_scores, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_xlabel('Confidence Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Confidence Scores')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(x=0.5, color='red', linestyle='--', label='Threshold 0.5')
    axes[0, 0].legend()
    
    # 2. 累積分佈
    sorted_scores = np.sort(all_scores)
    cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    axes[0, 1].plot(sorted_scores, cumulative, color='green', linewidth=2)
    axes[0, 1].set_xlabel('Confidence Score')
    axes[0, 1].set_ylabel('Cumulative Probability')
    axes[0, 1].set_title('Cumulative Distribution of Confidence Scores')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(x=0.5, color='red', linestyle='--', label='Threshold 0.5')
    axes[0, 1].legend()
    
    # 3. 不同閾值下的預測數量
    thresholds = np.arange(0.1, 1.0, 0.05)
    counts = [np.sum(np.array(all_scores) >= t) for t in thresholds]
    axes[1, 0].plot(thresholds, counts, marker='o', color='purple')
    axes[1, 0].set_xlabel('Confidence Threshold')
    axes[1, 0].set_ylabel('Number of Predictions')
    axes[1, 0].set_title('Predictions Count vs Confidence Threshold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 置信度統計信息
    axes[1, 1].text(0.1, 0.9, f'Total Predictions: {len(all_scores)}', fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.8, f'Mean Score: {np.mean(all_scores):.3f}', fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.7, f'Std Score: {np.std(all_scores):.3f}', fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.6, f'Min Score: {np.min(all_scores):.3f}', fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.5, f'Max Score: {np.max(all_scores):.3f}', fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.4, f'Median Score: {np.median(all_scores):.3f}', fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.3, f'Predictions > 0.5: {np.sum(np.array(all_scores) > 0.5)}', fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.2, f'Predictions > 0.7: {np.sum(np.array(all_scores) > 0.7)}', fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.1, f'Predictions > 0.9: {np.sum(np.array(all_scores) > 0.9)}', fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].set_title('Confidence Statistics')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # 保存圖片
    save_path = os.path.join(save_dir, f'{prefix}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f"置信度分析圖已保存到: {save_path}")
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


def load_model(model_path, device):
    """載入訓練好的模型"""
    logging.info(f"載入模型: {model_path}")
    
    # 創建模型架構
    model = fasterrcnn_resnet50_fpn(weights=None)
    num_classes = 2  # 背景 + 病灶
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # 載入權重
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logging.info("模型載入成功")
    else:
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    model.to(device)
    model.eval()
    return model


def save_test_patient_lists(save_dir, dataset, split):
    """保存測試集的病例列表"""
    
    # 提取測試集病例列表
    patient_ids = []
    for i in range(len(dataset)):
        try:
            if hasattr(dataset, 'samples'):
                sample = dataset.samples[i]
                patient_id = sample['patient_id']
                if patient_id not in patient_ids:
                    patient_ids.append(patient_id)
        except:
            continue
    
    # 排序病例列表
    patient_ids.sort()
    
    if len(patient_ids) == 0:
        logging.warning("無法提取病例列表，可能是數據集結構問題")
        return {}
    
    # 保存病例列表
    patients_file = os.path.join(save_dir, f'{split}_patient_list.txt')
    with open(patients_file, 'w', encoding='utf-8') as f:
        f.write(f"# {split.upper()} 數據集病例列表\n")
        f.write(f"# 總計 {len(patient_ids)} 位病例\n")
        f.write(f"# 生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for patient_id in patient_ids:
            f.write(f"{patient_id}\n")
    
    # 保存詳細的病例分佈摘要
    summary_file = os.path.join(save_dir, f'{split}_patient_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"=== {split.upper()} 數據集病例摘要 ===\n")
        f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"總病例數: {len(patient_ids)}\n")
        f.write(f"總樣本數: {len(dataset)}\n")
        f.write(f"平均每位病例樣本數: {len(dataset)/len(patient_ids):.1f}\n\n")
        
        f.write("病例列表:\n")
        f.write(", ".join(patient_ids))
        f.write("\n")
    
    logging.info(f"{split.upper()} 數據集病例列表已保存到:")
    logging.info(f"  - 病例列表: {patients_file}")
    logging.info(f"  - 摘要: {summary_file}")
    
    return {
        'patient_ids': patient_ids,
        'patient_count': len(patient_ids),
        'samples_per_patient': len(dataset) / len(patient_ids) if len(patient_ids) > 0 else 0
    }


def validate_dataset_split(data_dir, split):
    """验证数据集分割是否存在"""
    split_dir = os.path.join(data_dir, split)
    patients_file = os.path.join(data_dir, f'{split}_patients.txt')
    
    if not os.path.exists(split_dir):
        logging.error(f"數據集分割目錄不存在: {split_dir}")
        
        # 列出可用的分割
        available_splits = []
        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                available_splits.append(item)
        
        if available_splits:
            logging.error(f"可用的數據集分割: {', '.join(available_splits)}")
            logging.error(f"請使用 --split 參數指定正確的分割，例如: --split {available_splits[0]}")
        
        return False
    
    if not os.path.exists(patients_file):
        logging.warning(f"患者列表文件不存在: {patients_file}")
        logging.info(f"將直接從目錄載入數據: {split_dir}")
    
    return True


def create_test_dataset(data_dir, split='test', target_size=512, include_negative_samples=True, max_negative_per_patient=0):
    """創建測試數據集並計算統計信息"""
    logging.info(f"正在創建 {split} 數據集...")
    logging.info(f"數據目錄: {data_dir}")
    logging.info(f"目標圖像大小: {target_size}")
    if include_negative_samples:
        if max_negative_per_patient == 0:
            logging.info(f"負樣本設定: 啟用（無限制，載入所有負樣本）")
        else:
            logging.info(f"負樣本設定: 啟用（每患者最多{max_negative_per_patient}個負樣本）")
    else:
        logging.info(f"負樣本設定: 禁用（僅載入有標註影像）")
    
    test_dataset = CTDetectionDataset(
        data_root=data_dir,
        split=split,
        target_size=target_size,
        specific_patients=None,
        transforms=transforms.Compose([
            transforms.ToTensor()
        ]),
        include_negative_samples=include_negative_samples,
        max_negative_per_patient=max_negative_per_patient
    )
    
    dataset_size = len(test_dataset)
    logging.info(f"測試數據集大小: {dataset_size}")
    
    if dataset_size == 0:
        logging.warning("⚠️  數據集為空！")
        logging.warning("可能的原因:")
        logging.warning("  1. 數據集分割目錄為空")
        logging.warning("  2. 沒有找到有效的XML標註文件")
        logging.warning("  3. 數據路徑配置不正確")
        
        # 显示目录内容以帮助诊断
        split_dir = os.path.join(data_dir, split)
        if os.path.exists(split_dir):
            try:
                items = os.listdir(split_dir)
                logging.info(f"📁 {split} 目錄內容 ({len(items)} 項目):")
                for i, item in enumerate(items[:10], 1):  # 只显示前10个
                    item_path = os.path.join(split_dir, item)
                    if os.path.isdir(item_path):
                        logging.info(f"  {i}. 📁 {item}/")
                    else:
                        logging.info(f"  {i}. 📄 {item}")
                if len(items) > 10:
                    logging.info(f"  ... (還有 {len(items) - 10} 個項目)")
            except Exception as e:
                logging.error(f"無法讀取目錄內容: {e}")
        
        return test_dataset, None
    
    # 提取病例列表
    patient_ids = []
    for i in range(len(test_dataset)):
        try:
            if hasattr(test_dataset, 'samples'):
                sample = test_dataset.samples[i]
                patient_id = sample['patient_id']
                if patient_id not in patient_ids:
                    patient_ids.append(patient_id)
        except:
            continue
    
    # 排序病例列表
    patient_ids.sort()
    
    logging.info(f"{split.upper()} 數據集包含 {len(patient_ids)} 位病例")
    logging.info(f"{split.upper()} 數據集病例: {', '.join(patient_ids[:10])}{'...' if len(patient_ids) > 10 else ''}")
    
    # 計算數據集統計信息
    dataset_stats = calculate_dataset_statistics(test_dataset, f"{split.upper()} 數據集")
    
    # 添加病例信息到統計中
    if dataset_stats:
        dataset_stats['patient_ids'] = patient_ids
        dataset_stats['patient_count'] = len(patient_ids)
        dataset_stats['samples_per_patient'] = len(test_dataset) / len(patient_ids) if len(patient_ids) > 0 else 0
    
    return test_dataset, dataset_stats


def evaluate_model_comprehensive(model, test_loader, device, confidence_thresholds=[0.3, 0.5, 0.7], 
                                iou_thresholds=[0.3, 0.5, 0.7]):
    """綜合評估模型性能"""
    import time
    
    model.eval()
    all_predictions = []
    all_targets = []
    all_images = []
    
    # 效率指標
    inference_times = []
    memory_usage = []
    
    logging.info("開始模型評估...")
    test_pbar = tqdm(test_loader, desc="評估進度", unit="batch", ncols=100)
    
    with torch.no_grad():
        for images, targets in test_pbar:
            images = [img.to(device) for img in images]
            
            # 測量推理時間
            start_time = time.time()
            predictions = model(images)
            inference_time = time.time() - start_time
            inference_times.append(inference_time / len(images))  # 每張圖的平均時間
            
            # 測量顯存使用（如果使用GPU）
            if device.type == 'cuda':
                memory_usage.append(torch.cuda.memory_allocated() / 1024**2)  # MB
            
            # 將結果移到CPU
            predictions = [{k: v.cpu() for k, v in pred.items()} for pred in predictions]
            targets = [{k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in target.items()} for target in targets]
            
            all_predictions.extend(predictions)
            all_targets.extend(targets)
            all_images.extend([img.cpu() for img in images])
    
    test_pbar.close()
    
    # 計算不同閾值下的指標
    results = {}
    
    for conf_thresh in confidence_thresholds:
        for iou_thresh in iou_thresholds:
            key = f"conf_{conf_thresh}_iou_{iou_thresh}"
            metrics = calculate_comprehensive_metrics(
                all_predictions, all_targets, 
                iou_threshold=iou_thresh, 
                confidence_threshold=conf_thresh
            )
            
            # 添加效率指標
            if inference_times:
                metrics['inference_time_per_image'] = np.mean(inference_times)
                metrics['fps'] = 1.0 / np.mean(inference_times) if np.mean(inference_times) > 0 else 0
                metrics['inference_time_std'] = np.std(inference_times)
            
            if memory_usage and device.type == 'cuda':
                metrics['avg_memory_usage_mb'] = np.mean(memory_usage)
                metrics['max_memory_usage_mb'] = np.max(memory_usage)
            
            results[key] = metrics
            
            logging.info(f"置信度閾值 {conf_thresh}, IoU閾值 {iou_thresh}:")
            logging.info(f"  精確度: {metrics['precision']:.4f}")
            logging.info(f"  召回率: {metrics['sensitivity_recall']:.4f}")
            logging.info(f"  F1分數: {metrics['f1_score']:.4f}")
            logging.info(f"  mAP@0.5: {metrics['mAP@0.5']:.4f}")
            logging.info(f"  mAP@[0.5:0.95]: {metrics['mAP@[0.5:0.95]']:.4f}")
            logging.info(f"  病例級敏感度: {metrics['case_level_sensitivity']:.4f}")
    
    # 使用標準設置計算綜合結果
    comprehensive_metrics = calculate_comprehensive_metrics(
        all_predictions, all_targets, 
        iou_threshold=0.5, 
        confidence_threshold=0.5
    )
    
    # 添加效率指標到綜合結果
    if inference_times:
        comprehensive_metrics['inference_time_per_image'] = np.mean(inference_times)
        comprehensive_metrics['fps'] = 1.0 / np.mean(inference_times) if np.mean(inference_times) > 0 else 0
        comprehensive_metrics['inference_time_std'] = np.std(inference_times)
    
    if memory_usage and device.type == 'cuda':
        comprehensive_metrics['avg_memory_usage_mb'] = np.mean(memory_usage)
        comprehensive_metrics['max_memory_usage_mb'] = np.max(memory_usage)
    
    results['comprehensive'] = comprehensive_metrics
    
    # 生成ROC和FROC曲線
    try:
        roc_froc_results = calculate_roc_froc_curves(all_predictions, all_targets)
        results['roc_froc'] = roc_froc_results
        logging.info(f"ROC AUC: {roc_froc_results['roc_auc']:.4f}")
    except Exception as e:
        logging.warning(f"ROC/FROC計算失敗: {str(e)}")
    
    return results, all_predictions, all_targets, all_images


def test_detection_model(model_path, data_dir, batch_size=8, save_dir='./test_results', 
                        confidence_thresholds=[0.3, 0.5, 0.7], iou_thresholds=[0.3, 0.5, 0.7],
                        visualize_samples=15, split='val', include_negative_samples=True, max_negative_per_patient=0,
                        extract_deep_features=True):
    """測試檢測模型的主要函數"""
    
    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"使用設備: {device}")
    
    # 創建保存目錄
    os.makedirs(save_dir, exist_ok=True)
    
    # 載入模型
    model = load_model(model_path, device)
    
    # 創建測試數據集並獲取統計信息
    test_dataset, dataset_stats = create_test_dataset(
        data_dir, split=split, include_negative_samples=include_negative_samples, max_negative_per_patient=max_negative_per_patient
    )
    
    # 保存數據集統計信息到文件
    if dataset_stats:
        stats_file = os.path.join(save_dir, f'{split}_dataset_statistics.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_stats, f, indent=2, ensure_ascii=False)
        logging.info(f"數據集統計信息已保存到: {stats_file}")
        
        # 保存病例列表到單獨文件
        save_test_patient_lists(save_dir, test_dataset, split)
        logging.info(f"數據集統計信息已保存到: {stats_file}")
    
    # 創建數據加載器
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # 綜合評估模型
    start_time = time.time()
    results, predictions, targets, images = evaluate_model_comprehensive(
        model, test_loader, device, confidence_thresholds, iou_thresholds
    )
    evaluation_time = time.time() - start_time
    
    # 可視化預測結果
    if visualize_samples > 0:
        logging.info("生成可視化結果...")
        vis_dir = os.path.join(save_dir, 'visualizations')
        
        try:
            visualize_predictions(
                images[:visualize_samples], 
                predictions[:visualize_samples], 
                targets[:visualize_samples],
                vis_dir, 
                num_samples=visualize_samples,
                confidence_threshold=0.5,
                prefix="test_predictions"
            )
        except Exception as e:
            logging.error(f"可視化生成失敗: {str(e)}")
    
    # 提取深層特徵（預設啟用）
    if extract_deep_features:
        logging.info("開始提取深層特徵（預設啟用）...")
        features_dir = os.path.join(save_dir, 'deep_features')
        
        try:
            # 導入特徵提取模塊
            from deep_feature_extractor import extract_features_from_dataset
            
            # 提取特徵，使用標準置信度閾值
            extract_features_from_dataset(
                model_path=model_path,
                data_dir=data_dir,
                save_dir=features_dir,
                split=split,
                confidence_threshold=0.5,  # 使用標準閾值
                device=device.type
            )
            
            logging.info(f"深層特徵已保存到: {features_dir}")
            
            # 生成特徵摘要報告
            try:
                from feature_loader import FeatureLoader, FeatureVisualizer
                
                feature_loader = FeatureLoader(features_dir)
                visualizer = FeatureVisualizer(feature_loader)
                
                # 生成數據集摘要
                summary_path = os.path.join(features_dir, 'feature_summary_report.md')
                visualizer.create_dataset_summary(summary_path)
                logging.info(f"特徵摘要報告已保存到: {summary_path}")
                
                # 為前幾個病例生成詳細報告
                patient_ids = feature_loader.get_all_patient_ids()[:5]
                for patient_id in patient_ids:
                    try:
                        # 將病例報告保存在對應的病例資料夾中
                        patient_dir = os.path.join(features_dir, patient_id)
                        os.makedirs(patient_dir, exist_ok=True)
                        report_path = os.path.join(patient_dir, f"{patient_id}_feature_report.md")
                        visualizer.create_patient_report(patient_id, report_path)
                    except Exception as e:
                        logging.warning(f"生成病例 {patient_id} 特徵報告失敗: {str(e)}")
                
                # 輸出統計信息到日誌
                stats = feature_loader.get_statistical_summary()
                logging.info("=== 深層特徵提取統計 ===")
                logging.info(f"處理病例數: {stats.get('total_patients', 0)}")
                logging.info(f"含病灶病例: {stats.get('patients_with_lesions', 0)}")
                logging.info(f"病灶檢出率: {stats.get('lesion_detection_rate', 0):.1%}")
                logging.info(f"平均病灶數/病例: {stats.get('avg_detections_per_patient', 0):.1f}")
                
            except Exception as e:
                logging.warning(f"生成特徵摘要報告失敗: {str(e)}")
                
        except Exception as e:
            logging.error(f"深層特徵提取失敗: {str(e)}")
            # 繼續執行，不中斷主要測試流程
    else:
        logging.info("深層特徵提取已禁用（使用 --no_extract_features 禁用）")
    
    # 生成置信度分析
    try:
        logging.info("生成置信度分析...")
        conf_analysis_dir = os.path.join(save_dir, 'analysis')
        create_confidence_analysis(predictions, conf_analysis_dir)
    except Exception as e:
        logging.error(f"置信度分析生成失敗: {str(e)}")
    
    # 生成全面的評估指標報告
    try:
        if 'comprehensive' in results:
            logging.info("生成全面評估指標報告...")
            summary_dir = os.path.join(save_dir, 'comprehensive_summary')
            create_comprehensive_summary(results['comprehensive'], summary_dir, "test_comprehensive_metrics")
        
        # 生成ROC和FROC曲線可視化
        if 'roc_froc' in results:
            logging.info("生成ROC和FROC曲線...")
            roc_froc_dir = os.path.join(save_dir, 'roc_froc')
            calculate_roc_froc_curves(predictions, targets, roc_froc_dir)
    except Exception as e:
        logging.error(f"評估報告生成失敗: {str(e)}")
    
    # 保存測試結果
    test_summary = {
        'model_path': model_path,
        'data_dir': data_dir,
        'test_samples': len(test_dataset),
        'evaluation_time': evaluation_time,
        'device': str(device),
        'results': results,
        'test_config': {
            'batch_size': batch_size,
            'confidence_thresholds': confidence_thresholds,
            'iou_thresholds': iou_thresholds,
            'split': split
        },
        'dataset_statistics': dataset_stats
    }
    
    # 保存結果到JSON文件
    results_file = os.path.join(save_dir, 'test_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(test_summary, f, indent=2, ensure_ascii=False, default=str)
    
    # 輸出最終結果摘要
    logging.info("\n=== 測試結果摘要 ===")
    logging.info(f"測試樣本數: {len(test_dataset)}")
    logging.info(f"評估時間: {evaluation_time:.2f}秒")
    
    # 輸出數據集統計
    if 'dataset_statistics' in test_summary:
        stats = test_summary['dataset_statistics']
        logging.info("\n=== 數據集統計 ===")
        logging.info(f"測試影像數: {stats.get('total_images', 0)}")
        logging.info(f"測試標記數: {stats.get('total_annotations', 0)}")
        if stats.get('total_images', 0) > 0:
            logging.info(f"平均每圖標記數: {stats.get('total_annotations', 0) / stats.get('total_images', 1):.2f}")
        
        # 輸出病例統計
        if 'patient_ids' in stats:
            patient_ids = stats['patient_ids']
            logging.info(f"測試病例數: {len(patient_ids)}")
            logging.info(f"平均每位病例樣本數: {stats.get('samples_per_patient', 0):.1f}")
            logging.info(f"測試病例: {', '.join(patient_ids[:10])}{'...' if len(patient_ids) > 10 else ''}")
    
    # 輸出全面評估結果
    if 'comprehensive' in results:
        comp_metrics = results['comprehensive']
        logging.info(f"\n=== 全面評估結果 (標準設置: 置信度0.5, IoU0.5) ===")
        logging.info(f"精確度 (Precision): {comp_metrics.get('precision', 0):.4f}")
        logging.info(f"召回率/敏感度 (Recall/Sensitivity): {comp_metrics.get('sensitivity_recall', 0):.4f}")
        logging.info(f"F1分數 (F1-Score): {comp_metrics.get('f1_score', 0):.4f}")
        logging.info(f"mAP@0.5: {comp_metrics.get('mAP@0.5', 0):.4f}")
        logging.info(f"mAP@[0.5:0.95]: {comp_metrics.get('mAP@[0.5:0.95]', 0):.4f}")
        logging.info(f"病灶級敏感度: {comp_metrics.get('lesion_level_sensitivity', 0):.4f}")
        logging.info(f"病例級敏感度: {comp_metrics.get('case_level_sensitivity', 0):.4f}")
        logging.info(f"每圖平均假陽性數: {comp_metrics.get('fp_per_image', 0):.4f}")
        logging.info(f"IoU: {comp_metrics.get('iou', 0):.4f}")
        logging.info(f"GIoU: {comp_metrics.get('mean_giou', 0):.4f}")
        logging.info(f"DIoU: {comp_metrics.get('mean_diou', 0):.4f}")
        logging.info(f"CIoU: {comp_metrics.get('mean_ciou', 0):.4f}")
        
        if 'inference_time_per_image' in comp_metrics:
            logging.info(f"每圖推理時間: {comp_metrics['inference_time_per_image']*1000:.2f} ms")
            logging.info(f"FPS: {comp_metrics.get('fps', 0):.2f}")
        
        if 'avg_memory_usage_mb' in comp_metrics:
            logging.info(f"平均顯存使用: {comp_metrics['avg_memory_usage_mb']:.2f} MB")
            logging.info(f"峰值顯存使用: {comp_metrics['max_memory_usage_mb']:.2f} MB")
    
    # 輸出ROC/FROC結果
    if 'roc_froc' in results:
        roc_froc = results['roc_froc']
        logging.info(f"ROC AUC: {roc_froc.get('roc_auc', 0):.4f}")
    
    # 輸出最佳性能指標
    best_f1 = 0
    best_config = ""
    for key, metrics in results.items():
        if key not in ['comprehensive', 'roc_froc'] and isinstance(metrics, dict) and 'f1_score' in metrics:
            if metrics['f1_score'] > best_f1:
                best_f1 = metrics['f1_score']
                best_config = key
    
    if best_config:
        best_metrics = results[best_config]
        logging.info(f"\n最佳F1性能配置: {best_config}")
        logging.info(f"  精確度: {best_metrics.get('precision', 0):.4f}")
        logging.info(f"  召回率: {best_metrics.get('sensitivity_recall', best_metrics.get('recall', 0)):.4f}")
        logging.info(f"  F1分數: {best_metrics['f1_score']:.4f}")
        logging.info(f"  TP: {best_metrics.get('tp', 0)}, FP: {best_metrics.get('fp', 0)}, FN: {best_metrics.get('fn', 0)}")
    
    logging.info(f"\n測試結果已保存到: {save_dir}")
    logging.info(f"詳細結果文件: {results_file}")
    
    return test_summary


def main():
    parser = argparse.ArgumentParser(description='Test Faster R-CNN Detection Model')
    
    # 獲取腳本所在目錄
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 生成帶時間戳的資料夾名稱
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_folder = f'test_results_{timestamp}'
    log_folder = f'test_logs_{timestamp}'
    
    # 構建預定義的模型路徑 (按優先級排序)
    default_model_paths = [
        os.path.join(script_dir, path) for path in DEFAULT_MODEL_SEARCH_PATHS
    ]
    
    # 尋找第一個存在的模型文件作為默認值
    default_model_path = None
    for model_path in default_model_paths:
        if os.path.exists(model_path):
            default_model_path = model_path
            break
    
    # 模型相關參數
    parser.add_argument('--model_path', type=str, default=default_model_path,
                       help='訓練好的模型權重文件路徑 (.pth)，如果不指定會自動尋找默認模型')
    parser.add_argument('--list_models', action='store_true',
                       help='列出所有可用的模型路徑並退出')
    parser.add_argument('--check_dataset', action='store_true',
                       help='檢查數據集狀態並退出')
    parser.add_argument('--data_dir', type=str, 
                       default=os.path.join(os.path.dirname(script_dir), 'datasets', 'splited_dataset'), 
                       help='測試數據集目錄路徑')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                       help='使用的數據集分割 (train/val/test)，默認使用test分割')
    
    # 測試參數
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, 
                       help='批次大小')
    parser.add_argument('--confidence_thresholds', type=float, nargs='+', default=DEFAULT_CONFIDENCE_THRESHOLDS,
                       help='置信度閾值列表')
    parser.add_argument('--iou_thresholds', type=float, nargs='+', default=DEFAULT_IOU_THRESHOLDS,
                       help='IoU閾值列表')
    parser.add_argument('--visualize_samples', type=int, default=DEFAULT_VISUALIZE_SAMPLES,
                       help='可視化樣本數量 (0表示不生成可視化)')
    parser.add_argument('--include_negative_samples', action='store_true', default=True,
                       help='包含負樣本（無標註的影像）以改善ROC/AUC計算（預設啟用）')
    parser.add_argument('--no_negative_samples', action='store_false', dest='include_negative_samples',
                       help='禁用負樣本載入，僅載入有標註的影像')
    parser.add_argument('--max_negative_per_patient', type=int, default=20,
                       help='每位患者最大負樣本數量，0表示無限制（載入所有負樣本）')
    
    # 特徵提取參數
    parser.add_argument('--extract_features', action='store_true', default=True,
                       help='提取深層特徵供LLM生成報告使用（預設啟用）')
    parser.add_argument('--no_extract_features', action='store_false', dest='extract_features',
                       help='禁用深層特徵提取')
    
    # 輸出參數
    parser.add_argument('--save_dir', type=str, 
                       default=os.path.join(script_dir, test_folder), 
                       help='測試結果保存目錄')
    parser.add_argument('--log_dir', type=str, 
                       default=os.path.join(script_dir, log_folder), 
                       help='日誌保存目錄')
    
    args = parser.parse_args()
    
    # 如果用戶要求列出模型，顯示後退出
    if args.list_models:
        print("=== 可用的模型路徑 ===")
        found_models = []
        for i, model_path in enumerate(default_model_paths, 1):
            exists = os.path.exists(model_path)
            status = "存在" if exists else "不存在"
            marker = "[存在]" if exists else "[不存在]"
            print(f"  {i}. {marker} {model_path} ({status})")
            if exists:
                found_models.append(model_path)
        
        print(f"\n找到 {len(found_models)} 個可用的模型文件")
        if found_models:
            print(f"默認會使用: {found_models[0]}")
        else:
            print("未找到任何模型文件，請確保至少有一個模型文件存在")
        return
    
    # 如果用戶要求檢查數據集，顯示後退出
    if args.check_dataset:
        print("=== 數據集狀態檢查 ===")
        print(f"數據目錄: {args.data_dir}")
        
        if not os.path.exists(args.data_dir):
            print(f"❌ 數據目錄不存在: {args.data_dir}")
            return
        
        print("✅ 數據目錄存在")
        
        # 检查可用的分割
        available_splits = []
        for item in os.listdir(args.data_dir):
            item_path = os.path.join(args.data_dir, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                available_splits.append(item)
        
        print(f"📁 可用的數據集分割: {', '.join(available_splits) if available_splits else '無'}")
        
        # 检查每个分割的详细信息
        for split in available_splits:
            split_dir = os.path.join(args.data_dir, split)
            patients_file = os.path.join(args.data_dir, f'{split}_patients.txt')
            
            print(f"\n--- {split.upper()} 分割 ---")
            print(f"分割目錄: {split_dir}")
            print(f"患者列表: {patients_file} {'✅' if os.path.exists(patients_file) else '❌'}")
            
            try:
                items = os.listdir(split_dir)
                dirs = [item for item in items if os.path.isdir(os.path.join(split_dir, item))]
                files = [item for item in items if os.path.isfile(os.path.join(split_dir, item))]
                
                print(f"患者目錄數量: {len(dirs)}")
                print(f"文件數量: {len(files)}")
                
                if dirs:
                    print(f"患者目錄示例: {', '.join(dirs[:5])}")
                    if len(dirs) > 5:
                        print(f"  ... (還有 {len(dirs) - 5} 個)")
                        
            except Exception as e:
                print(f"❌ 無法讀取目錄: {e}")
        
        print(f"\n建議使用的分割: {available_splits[0] if available_splits else 'train'}")
        return
    
    # 設置日誌
    log_file = setup_logging(args.log_dir)
    logging.info(f"日誌文件: {log_file}")
    
    # 檢查模型文件
    if args.model_path is None:
        logging.error("未找到任何可用的模型文件！")
        logging.error("請確保以下路徑之一存在模型文件:")
        for model_path in default_model_paths:
            logging.error(f"  - {model_path}")
        logging.error("或者使用 --model_path 參數指定模型文件路徑")
        return
    
    if not os.path.exists(args.model_path):
        logging.error(f"模型文件不存在: {args.model_path}")
        return
    
    logging.info(f"使用模型文件: {args.model_path}")
    
    # 檢查數據目錄
    if not os.path.exists(args.data_dir):
        logging.error(f"數據目錄不存在: {args.data_dir}")
        return
    
    # 验证数据集分割
    if not validate_dataset_split(args.data_dir, args.split):
        return
    
    # 輸出配置信息
    logging.info("=== 測試配置 ===")
    logging.info(f"模型路徑: {args.model_path}")
    
    # 顯示所有檢查過的模型路徑
    logging.info("檢查的模型路徑:")
    for i, model_path in enumerate(default_model_paths, 1):
        exists = "[存在]" if os.path.exists(model_path) else "[不存在]"
        status = "(使用中)" if model_path == args.model_path else ""
        logging.info(f"  {i}. {exists} {model_path} {status}")
    
    logging.info(f"數據目錄: {args.data_dir}")
    logging.info(f"數據分割: {args.split}")
    logging.info(f"批次大小: {args.batch_size}")
    logging.info(f"置信度閾值: {args.confidence_thresholds}")
    logging.info(f"IoU閾值: {args.iou_thresholds}")
    logging.info(f"可視化樣本數: {args.visualize_samples}")
    logging.info(f"結果保存目錄: {args.save_dir}")
    if args.include_negative_samples:
        if args.max_negative_per_patient == 0:
            logging.info(f"負樣本設定: 啟用（無限制，載入所有負樣本）")
        else:
            logging.info(f"負樣本設定: 啟用（每患者最多{args.max_negative_per_patient}個負樣本）")
    else:
        logging.info(f"負樣本設定: 禁用（僅載入有標註影像）")
    logging.info(f"深層特徵提取: {'啟用（預設）' if args.extract_features else '禁用'}")
    
    # 開始測試
    start_time = time.time()
    try:
        test_summary = test_detection_model(
            model_path=args.model_path,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            save_dir=args.save_dir,
            confidence_thresholds=args.confidence_thresholds,
            iou_thresholds=args.iou_thresholds,
            visualize_samples=args.visualize_samples,
            split=args.split,
            include_negative_samples=args.include_negative_samples,
            max_negative_per_patient=args.max_negative_per_patient,
            extract_deep_features=args.extract_features
        )
        
        total_time = time.time() - start_time
        logging.info(f"\n測試完成！總耗時: {total_time:.2f}秒 ({total_time/60:.2f}分鐘)")
        
    except Exception as e:
        logging.error(f"測試過程中發生錯誤: {str(e)}")
        raise


if __name__ == "__main__":
    main()
