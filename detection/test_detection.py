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

# 設置控制台編碼 (Windows)
if sys.platform.startswith('win'):
    import locale
    try:
        # 嘗試設置UTF-8編碼
        os.system('chcp 65001 >nul')
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

import torch
import torch.nn as nn
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

from faster_rcnn_dataset import CTDetectionDataset


# =============================================================================
# 配置常量 - 可根據需要修改
# =============================================================================

# 預定義的模型路徑搜索順序 (相對於detection目錄)
DEFAULT_MODEL_SEARCH_PATHS = [
    'Simple_Training/models/best_model.pth',
    'Faster_RCNN_Detection/models/best_model_fold_1.pth',
    'Faster_RCNN_Detection/models/best_model_fold_2.pth',
    'Faster_RCNN_Detection/models/best_model_fold_3.pth',
    'Faster_RCNN_Detection/models/best_model_fold_4.pth',
    'Faster_RCNN_Detection/models/best_model_fold_5.pth',
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
    """計算檢測指標"""
    tp, fp, fn = 0, 0, 0
    total_predictions = 0
    total_targets = 0
    
    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes']
        pred_scores = pred['scores']
        target_boxes = target['boxes']
        
        # 過濾低置信度預測
        valid_pred = pred_scores > confidence_threshold
        pred_boxes = pred_boxes[valid_pred]
        pred_scores = pred_scores[valid_pred]
        
        total_predictions += len(pred_boxes)
        total_targets += len(target_boxes)
        
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
        
        # 匹配預測和目標（貪心匹配）
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
        'fn': fn,
        'total_predictions': total_predictions,
        'total_targets': total_targets
    }


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


def create_test_dataset(data_dir, split='test', target_size=512):
    """創建測試數據集"""
    logging.info(f"正在創建 {split} 數據集...")
    logging.info(f"數據目錄: {data_dir}")
    logging.info(f"目標圖像大小: {target_size}")
    
    test_dataset = CTDetectionDataset(
        data_root=data_dir,
        split=split,
        target_size=target_size,
        specific_patients=None,
        transforms=transforms.Compose([
            transforms.ToTensor()
        ])
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
    
    return test_dataset


def evaluate_model_comprehensive(model, test_loader, device, confidence_thresholds=[0.3, 0.5, 0.7], 
                                iou_thresholds=[0.3, 0.5, 0.7]):
    """綜合評估模型性能"""
    model.eval()
    all_predictions = []
    all_targets = []
    all_images = []
    
    logging.info("開始模型評估...")
    test_pbar = tqdm(test_loader, desc="評估進度", unit="batch", ncols=100)
    
    with torch.no_grad():
        for images, targets in test_pbar:
            images = [img.to(device) for img in images]
            
            # 模型推理
            predictions = model(images)
            
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
            metrics = calculate_detection_metrics(
                all_predictions, all_targets, 
                iou_threshold=iou_thresh, 
                confidence_threshold=conf_thresh
            )
            results[key] = metrics
            
            logging.info(f"置信度閾值 {conf_thresh}, IoU閾值 {iou_thresh}:")
            logging.info(f"  精確度: {metrics['precision']:.4f}")
            logging.info(f"  召回率: {metrics['recall']:.4f}")
            logging.info(f"  F1分數: {metrics['f1_score']:.4f}")
    
    # 計算mAP
    logging.info("計算平均精度 (AP)...")
    ap_scores = {}
    for iou_thresh in iou_thresholds:
        ap = calculate_ap(all_predictions, all_targets, iou_threshold=iou_thresh)
        ap_scores[f"AP_IoU_{iou_thresh}"] = ap
        logging.info(f"AP @ IoU {iou_thresh}: {ap:.4f}")
    
    results['ap_scores'] = ap_scores
    
    return results, all_predictions, all_targets, all_images


def test_detection_model(model_path, data_dir, batch_size=8, save_dir='./test_results', 
                        confidence_thresholds=[0.3, 0.5, 0.7], iou_thresholds=[0.3, 0.5, 0.7],
                        visualize_samples=15, split='val'):
    """測試檢測模型的主要函數"""
    
    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"使用設備: {device}")
    
    # 創建保存目錄
    os.makedirs(save_dir, exist_ok=True)
    
    # 載入模型
    model = load_model(model_path, device)
    
    # 創建測試數據集
    test_dataset = create_test_dataset(data_dir, split=split)
    
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
    
    # 生成置信度分析
    try:
        logging.info("生成置信度分析...")
        conf_analysis_dir = os.path.join(save_dir, 'analysis')
        create_confidence_analysis(predictions, conf_analysis_dir)
    except Exception as e:
        logging.error(f"置信度分析生成失敗: {str(e)}")
    
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
        }
    }
    
    # 保存結果到JSON文件
    results_file = os.path.join(save_dir, 'test_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(test_summary, f, indent=2, ensure_ascii=False)
    
    # 輸出最終結果摘要
    logging.info("\n=== 測試結果摘要 ===")
    logging.info(f"測試樣本數: {len(test_dataset)}")
    logging.info(f"評估時間: {evaluation_time:.2f}秒")
    
    # 輸出最佳性能指標
    best_f1 = 0
    best_config = ""
    for key, metrics in results.items():
        if key != 'ap_scores' and 'f1_score' in metrics:
            if metrics['f1_score'] > best_f1:
                best_f1 = metrics['f1_score']
                best_config = key
    
    if best_config:
        best_metrics = results[best_config]
        logging.info(f"\n最佳性能配置: {best_config}")
        logging.info(f"  精確度: {best_metrics['precision']:.4f}")
        logging.info(f"  召回率: {best_metrics['recall']:.4f}")
        logging.info(f"  F1分數: {best_metrics['f1_score']:.4f}")
        logging.info(f"  TP: {best_metrics['tp']}, FP: {best_metrics['fp']}, FN: {best_metrics['fn']}")
    
    # 輸出AP分數
    if 'ap_scores' in results:
        logging.info("\n平均精度 (AP) 分數:")
        for ap_key, ap_value in results['ap_scores'].items():
            logging.info(f"  {ap_key}: {ap_value:.4f}")
    
    logging.info(f"\n測試結果已保存到: {save_dir}")
    logging.info(f"詳細結果文件: {results_file}")
    
    return test_summary


def main():
    parser = argparse.ArgumentParser(description='Test Faster R-CNN Detection Model')
    
    # 獲取腳本所在目錄
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
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
    
    # 輸出參數
    parser.add_argument('--save_dir', type=str, 
                       default=os.path.join(script_dir, 'test_results'), 
                       help='測試結果保存目錄')
    parser.add_argument('--log_dir', type=str, 
                       default=os.path.join(script_dir, 'test_logs'), 
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
            split=args.split
        )
        
        total_time = time.time() - start_time
        logging.info(f"\n測試完成！總耗時: {total_time:.2f}秒 ({total_time/60:.2f}分鐘)")
        
    except Exception as e:
        logging.error(f"測試過程中發生錯誤: {str(e)}")
        raise


if __name__ == "__main__":
    main()
