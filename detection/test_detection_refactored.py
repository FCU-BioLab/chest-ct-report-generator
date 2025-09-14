#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Detection Model - 測試訓練好的 Faster R-CNN 檢測模型 - 重構版本
用於測試訓練結果，評估模型性能和可視化預測結果
使用模組化計算函數提高代碼可維護性
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

from faster_rcnn_dataset import CTDetectionDataset

# 設置控制台編碼 (Windows)
if sys.platform.startswith('win'):
    try:
        os.system('chcp 65001 >nul')
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

# 嘗試導入模組化的計算函數，如果失敗則使用內聯版本
try:
    from metrics.detection_metrics import calculate_comprehensive_metrics, calculate_detection_metrics
    from metrics.roc_froc import calculate_roc_froc_curves
    from metrics.iou_calculations import calculate_giou, calculate_diou, calculate_ciou, calculate_bbox_error, calculate_iou_matrix
    from visualization import visualize_predictions, create_prediction_summary, create_comprehensive_summary
    from evaluation import evaluate_model
    from utils import collate_fn, setup_logging
    MODULES_IMPORTED = True
    logging.info("已成功導入模組化計算函數")
except ImportError as e:
    logging.warning(f"模組化導入失敗: {e}")
    logging.warning("將使用內聯函數作為備選方案")
    MODULES_IMPORTED = False

try:
    from sklearn.metrics import roc_curve, auc
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. ROC/AUC metrics will be skipped.")


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


def collate_fn_fallback(batch):
    """自定義批次整理函數的備選版本"""
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


def setup_logging_fallback(log_dir):
    """設置日誌記錄的備選版本"""
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


def simple_iou_calculation(box1, box2):
    """簡化版本的IoU計算 - 備選方案"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def simple_comprehensive_metrics(predictions, targets, iou_threshold=0.5, confidence_threshold=0.5):
    """簡化版本的全面檢測指標計算 - 備選方案"""
    tp, fp, fn = 0, 0, 0
    ious = []
    confidence_scores = []
    
    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes']
        pred_scores = pred['scores']
        target_boxes = target['boxes']
        
        # 過濾低置信度預測
        valid_pred = pred_scores > confidence_threshold
        filtered_pred_boxes = pred_boxes[valid_pred]
        filtered_pred_scores = pred_scores[valid_pred]
        
        confidence_scores.extend(filtered_pred_scores.tolist())
        
        if len(filtered_pred_boxes) == 0 and len(target_boxes) == 0:
            continue
        elif len(filtered_pred_boxes) == 0:
            fn += len(target_boxes)
            continue
        elif len(target_boxes) == 0:
            fp += len(filtered_pred_boxes)
            continue
        
        # 計算IoU和匹配
        matched_targets = set()
        for i, pred_box in enumerate(filtered_pred_boxes):
            best_iou = 0
            best_target = -1
            for j, target_box in enumerate(target_boxes):
                if j not in matched_targets:
                    iou = simple_iou_calculation(pred_box, target_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_target = j
            
            if best_iou >= iou_threshold:
                tp += 1
                matched_targets.add(best_target)
                ious.append(best_iou)
            else:
                fp += 1
        
        fn += len(target_boxes) - len(matched_targets)
    
    # 計算基礎指標
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'iou': np.mean(ious) if ious else 0,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'mean_confidence': np.mean(confidence_scores) if confidence_scores else 0
    }


def simple_visualize_predictions(images, predictions, targets, save_dir, num_samples=10, 
                                confidence_threshold=0.5, prefix="predictions"):
    """簡化版本的預測結果可視化 - 備選方案"""
    os.makedirs(save_dir, exist_ok=True)
    
    num_samples = min(num_samples, len(images))
    
    for i in range(num_samples):
        image = images[i]
        pred = predictions[i]
        target = targets[i]
        
        # 轉換張量為numpy array
        if isinstance(image, torch.Tensor):
            image_np = image.permute(1, 2, 0).cpu().numpy()
        else:
            image_np = image
        
        # 創建圖像
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image_np, cmap='gray')
        
        # 繪製真實邊界框（綠色）
        if 'boxes' in target and len(target['boxes']) > 0:
            for box in target['boxes']:
                rect = patches.Rectangle(
                    (box[0], box[1]), box[2]-box[0], box[3]-box[1],
                    linewidth=2, edgecolor='green', facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(box[0], box[1], 'GT', fontsize=12, color='green', 
                       bbox=dict(facecolor='white', alpha=0.8))
        
        # 繪製預測邊界框（紅色）
        if 'boxes' in pred and len(pred['boxes']) > 0:
            valid_pred = pred['scores'] > confidence_threshold
            pred_boxes = pred['boxes'][valid_pred]
            pred_scores = pred['scores'][valid_pred]
            
            for box, score in zip(pred_boxes, pred_scores):
                rect = patches.Rectangle(
                    (box[0], box[1]), box[2]-box[0], box[3]-box[1],
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(box[0], box[1]-10, f'Pred: {score:.2f}', fontsize=10, color='red',
                       bbox=dict(facecolor='white', alpha=0.8))
        
        ax.set_title(f'Sample {i+1} - Confidence Threshold: {confidence_threshold}')
        ax.axis('off')
        
        # 保存圖像
        save_path = os.path.join(save_dir, f'{prefix}_sample_{i+1}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    logging.info(f"簡化可視化結果已保存到: {save_dir}")
    return save_dir


def find_model_file(detection_dir):
    """智能搜索模型文件"""
    import glob
    
    for search_pattern in DEFAULT_MODEL_SEARCH_PATHS:
        search_path = os.path.join(detection_dir, search_pattern)
        
        # 處理通配符模式
        if '*' in search_pattern:
            matches = glob.glob(search_path)
            if matches:
                # 選擇最新的文件
                latest_file = max(matches, key=os.path.getctime)
                if os.path.exists(latest_file):
                    logging.info(f"找到模型文件（通配符匹配）: {latest_file}")
                    return latest_file
        else:
            if os.path.exists(search_path):
                logging.info(f"找到模型文件: {search_path}")
                return search_path
    
    logging.warning("未找到預定義的模型文件")
    return None


def load_model(model_path, device):
    """載入訓練好的模型"""
    # 創建模型架構
    model = fasterrcnn_resnet50_fpn(weights=None)
    num_classes = 2  # 背景 + 病灶
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # 載入權重
    if model_path.endswith('.pth'):
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                logging.info(f"成功載入模型檢查點: {model_path}")
                if 'best_f1' in checkpoint:
                    logging.info(f"模型最佳F1分數: {checkpoint['best_f1']:.4f}")
            else:
                model.load_state_dict(checkpoint)
                logging.info(f"成功載入模型權重: {model_path}")
        except Exception as e:
            logging.error(f"載入模型失敗: {e}")
            return None
    else:
        logging.error(f"不支援的模型文件格式: {model_path}")
        return None
    
    model.to(device)
    model.eval()
    return model


def evaluate_model_comprehensive(model, test_loader, device, confidence_thresholds=[0.3, 0.5, 0.7], 
                                iou_thresholds=[0.3, 0.5, 0.7]):
    """綜合評估模型性能 - 與原始版本一致的完整評估"""
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
            
            if MODULES_IMPORTED:
                try:
                    metrics = calculate_comprehensive_metrics(
                        all_predictions, all_targets, 
                        iou_threshold=iou_thresh, 
                        confidence_threshold=conf_thresh
                    )
                except Exception as e:
                    logging.warning(f"使用模組化指標計算失敗: {e}")
                    metrics = simple_comprehensive_metrics(
                        all_predictions, all_targets, iou_thresh, conf_thresh
                    )
            else:
                metrics = simple_comprehensive_metrics(
                    all_predictions, all_targets, iou_thresh, conf_thresh
                )
            
            results[key] = {
                'confidence_threshold': conf_thresh,
                'iou_threshold': iou_thresh,
                'metrics': metrics
            }
    
    # 使用標準設置計算綜合結果
    if MODULES_IMPORTED:
        try:
            comprehensive_metrics = calculate_comprehensive_metrics(
                all_predictions, all_targets, 
                iou_threshold=0.5, 
                confidence_threshold=0.5
            )
        except Exception as e:
            logging.warning(f"使用模組化指標計算失敗: {e}")
            comprehensive_metrics = simple_comprehensive_metrics(
                all_predictions, all_targets, 0.5, 0.5
            )
    else:
        comprehensive_metrics = simple_comprehensive_metrics(
            all_predictions, all_targets, 0.5, 0.5
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
    if MODULES_IMPORTED and SKLEARN_AVAILABLE:
        try:
            roc_froc_results = calculate_roc_froc_curves(all_predictions, all_targets)
            results['roc_froc'] = roc_froc_results
            logging.info(f"ROC AUC: {roc_froc_results['roc_auc']:.4f}")
        except Exception as e:
            logging.warning(f"ROC/FROC計算失敗: {str(e)}")
    
    return results, all_predictions, all_targets, all_images


def create_test_dataset(data_dir, split='test', target_size=512, include_negative_samples=True, max_negative_per_patient=0):
    """創建測試數據集並計算統計信息 - 與原始版本一致"""
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
            sample = test_dataset[i]
            if isinstance(sample, dict) and 'image_path' in sample:
                # 從圖像路徑提取病例ID
                image_path = sample['image_path']
                if isinstance(image_path, str):
                    # 假設路徑格式為 .../patient_id/...
                    path_parts = image_path.replace('\\', '/').split('/')
                    for part in path_parts:
                        if part.startswith('A') and len(part) >= 4:  # 假設病例ID格式為A0001, A0002等
                            if part not in patient_ids:
                                patient_ids.append(part)
                            break
        except Exception as e:
            logging.warning(f"無法提取第{i}個樣本的病例ID: {e}")
            continue
    
    # 排序病例列表
    patient_ids.sort()
    
    logging.info(f"{split.upper()} 數據集包含 {len(patient_ids)} 位病例")
    logging.info(f"{split.upper()} 數據集病例: {', '.join(patient_ids[:10])}{'...' if len(patient_ids) > 10 else ''}")
    
    # 計算數據集統計信息
    if MODULES_IMPORTED:
        try:
            from metrics.dataset_statistics import calculate_dataset_statistics
            dataset_stats = calculate_dataset_statistics(test_dataset, f"{split.upper()} 數據集")
        except Exception as e:
            logging.warning(f"使用模組化統計計算失敗: {e}")
            dataset_stats = calculate_simple_dataset_statistics(test_dataset, f"{split.upper()} 數據集")
    else:
        dataset_stats = calculate_simple_dataset_statistics(test_dataset, f"{split.upper()} 數據集")
    
    # 添加病例信息到統計中
    if dataset_stats:
        dataset_stats['patient_ids'] = patient_ids
        dataset_stats['patient_count'] = len(patient_ids)
        dataset_stats['samples_per_patient'] = len(test_dataset) / len(patient_ids) if len(patient_ids) > 0 else 0
    
    return test_dataset, dataset_stats


def calculate_simple_dataset_statistics(dataset, dataset_name="Dataset"):
    """簡化版本的數據集統計信息計算 - 備選方案"""
    total_images = len(dataset)
    total_annotations = 0
    images_with_annotations = 0
    images_without_annotations = 0
    
    logging.info(f"正在計算 {dataset_name} 統計信息...")
    
    for i in range(min(total_images, 1000)):  # 限制樣本數量以提高效率
        try:
            sample = dataset[i]
            if isinstance(sample, dict) and 'target' in sample:
                target = sample['target']
                if 'boxes' in target and len(target['boxes']) > 0:
                    total_annotations += len(target['boxes'])
                    images_with_annotations += 1
                else:
                    images_without_annotations += 1
            else:
                images_without_annotations += 1
        except Exception as e:
            logging.warning(f"計算統計時跳過第{i}個樣本: {e}")
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


def save_test_patient_lists(save_dir, dataset, split):
    """保存測試集的病例列表 - 與原始版本一致"""
    # 提取測試集病例列表
    patient_ids = []
    for i in range(len(dataset)):
        try:
            sample = dataset[i]
            if isinstance(sample, dict) and 'image_path' in sample:
                # 從圖像路徑提取病例ID
                image_path = sample['image_path']
                if isinstance(image_path, str):
                    # 假設路徑格式為 .../patient_id/...
                    path_parts = image_path.replace('\\', '/').split('/')
                    for part in path_parts:
                        if part.startswith('A') and len(part) >= 4:  # 假設病例ID格式為A0001, A0002等
                            if part not in patient_ids:
                                patient_ids.append(part)
                            break
        except Exception as e:
            continue
    
    # 排序病例列表
    patient_ids.sort()
    
    if len(patient_ids) == 0:
        logging.warning(f"未能從{split}數據集中提取到任何病例ID")
        return {'patient_ids': [], 'patient_count': 0, 'samples_per_patient': 0}
    
    # 保存病例列表
    patients_file = os.path.join(save_dir, f'{split}_patient_list.txt')
    with open(patients_file, 'w', encoding='utf-8') as f:
        f.write(f"{split.upper()} 數據集病例列表\n")
        f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"總病例數: {len(patient_ids)}\n")
        f.write("=" * 50 + "\n")
        for i, patient_id in enumerate(patient_ids, 1):
            f.write(f"{i:3d}. {patient_id}\n")
    
    # 保存詳細的病例分佈摘要
    summary_file = os.path.join(save_dir, f'{split}_patient_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"{split.upper()} 數據集病例分佈摘要\n")
        f.write("=" * 50 + "\n")
        f.write(f"數據集分割: {split}\n")
        f.write(f"總樣本數: {len(dataset)}\n")
        f.write(f"病例數: {len(patient_ids)}\n")
        f.write(f"平均每位病例樣本數: {len(dataset) / len(patient_ids):.2f}\n")
        f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n詳細病例列表:\n")
        for i, patient_id in enumerate(patient_ids, 1):
            f.write(f"{i:3d}. {patient_id}\n")
    
    logging.info(f"{split.upper()} 數據集病例列表已保存到:")
    logging.info(f"  - 病例列表: {patients_file}")
    logging.info(f"  - 摘要: {summary_file}")
    
    return {
        'patient_ids': patient_ids,
        'patient_count': len(patient_ids),
        'samples_per_patient': len(dataset) / len(patient_ids) if len(patient_ids) > 0 else 0
    }


def test_detection_model(model_path, data_dir, batch_size=8, save_dir='./test_results', 
                        confidence_thresholds=[0.3, 0.5, 0.7], iou_thresholds=[0.3, 0.5, 0.7],
                        visualize_samples=15, split='test', include_negative_samples=True, max_negative_per_patient=0,
                        extract_deep_features=True):
    """測試檢測模型的主要函數 - 與原始版本一致"""
    
    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"使用設備: {device}")
    
    # 創建保存目錄
    os.makedirs(save_dir, exist_ok=True)
    
    # 載入模型
    model = load_model(model_path, device)
    if model is None:
        logging.error("模型載入失敗")
        return None
    
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
    
    # 選擇適當的collate函數
    collate_func = collate_fn if MODULES_IMPORTED else collate_fn_fallback
    
    # 創建數據加載器
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_func,
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
        
        if MODULES_IMPORTED:
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
                logging.warning(f"使用模組化可視化失敗: {e}")
                simple_visualize_predictions(
                    images[:visualize_samples], 
                    predictions[:visualize_samples], 
                    targets[:visualize_samples],
                    vis_dir, 
                    num_samples=visualize_samples,
                    confidence_threshold=0.5,
                    prefix="test_predictions"
                )
        else:
            simple_visualize_predictions(
                images[:visualize_samples], 
                predictions[:visualize_samples], 
                targets[:visualize_samples],
                vis_dir, 
                num_samples=visualize_samples,
                confidence_threshold=0.5,
                prefix="test_predictions"
            )
    
    # 提取深層特徵（預設啟用）
    if extract_deep_features:
        logging.info("開始提取深層特徵（預設啟用）...")
        features_dir = os.path.join(save_dir, 'deep_features')
        try:
            if MODULES_IMPORTED:
                from deep_feature_extractor import extract_deep_features_batch
                extract_deep_features_batch(model, test_loader, device, features_dir)
            else:
                logging.warning("深層特徵提取需要模組化功能，已跳過")
        except Exception as e:
            logging.warning(f"深層特徵提取失敗: {e}")
    else:
        logging.info("跳過深層特徵提取")
    
    # 生成置信度分析
    try:
        if MODULES_IMPORTED:
            from visualization import create_confidence_analysis
            create_confidence_analysis(predictions, save_dir, "confidence_analysis")
        else:
            logging.info("置信度分析需要模組化功能，使用簡化版本")
            create_simple_confidence_analysis(predictions, save_dir)
    except Exception as e:
        logging.warning(f"置信度分析失敗: {e}")
    
    # 生成全面的評估指標報告
    try:
        if MODULES_IMPORTED:
            from visualization import create_comprehensive_summary
            create_comprehensive_summary(results['comprehensive'], save_dir, "comprehensive_metrics")
        else:
            logging.info("綜合報告需要模組化功能，使用簡化版本")
            create_simple_comprehensive_summary(results, save_dir)
    except Exception as e:
        logging.warning(f"綜合報告生成失敗: {e}")
    
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
            'split': split,
            'include_negative_samples': include_negative_samples,
            'max_negative_per_patient': max_negative_per_patient
        },
        'dataset_statistics': dataset_stats
    }
    
    # 保存結果到JSON文件
    results_file = os.path.join(save_dir, 'test_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(test_summary, f, indent=2, default=str, ensure_ascii=False)
    
    # 輸出最終結果摘要
    logging.info("\n=== 測試結果摘要 ===")
    logging.info(f"測試樣本數: {len(test_dataset)}")
    logging.info(f"評估時間: {evaluation_time:.2f}秒")
    
    # 輸出數據集統計
    if 'dataset_statistics' in test_summary and test_summary['dataset_statistics']:
        stats = test_summary['dataset_statistics']
        logging.info(f"數據集統計:")
        logging.info(f"  總影像數: {stats.get('total_images', 0)}")
        logging.info(f"  總標記數: {stats.get('total_annotations', 0)}")
        logging.info(f"  病例數: {stats.get('patient_count', 0)}")
    
    # 輸出全面評估結果
    if 'comprehensive' in results:
        comp_metrics = results['comprehensive']
        logging.info(f"全面評估結果:")
        logging.info(f"  精確度: {comp_metrics.get('precision', 0):.4f}")
        logging.info(f"  召回率: {comp_metrics.get('recall', comp_metrics.get('sensitivity_recall', 0)):.4f}")
        logging.info(f"  F1分數: {comp_metrics.get('f1_score', 0):.4f}")
        logging.info(f"  平均IoU: {comp_metrics.get('iou', 0):.4f}")
        
        if 'mAP@0.5' in comp_metrics:
            logging.info(f"  mAP@0.5: {comp_metrics['mAP@0.5']:.4f}")
        if 'mAP@[0.5:0.95]' in comp_metrics:
            logging.info(f"  mAP@[0.5:0.95]: {comp_metrics['mAP@[0.5:0.95]']:.4f}")
    
    # 輸出ROC/FROC結果
    if 'roc_froc' in results:
        roc_results = results['roc_froc']
        logging.info(f"ROC/FROC分析:")
        logging.info(f"  ROC AUC: {roc_results.get('roc_auc', 0):.4f}")
    
    # 輸出最佳性能指標
    best_f1 = 0
    best_config = ""
    for key, metrics in results.items():
        if key not in ['comprehensive', 'roc_froc'] and 'metrics' in metrics:
            f1 = metrics['metrics'].get('f1_score', 0)
            if f1 > best_f1:
                best_f1 = f1
                best_config = f"置信度={metrics['confidence_threshold']}, IoU={metrics['iou_threshold']}"
    
    if best_config:
        logging.info(f"最佳配置: {best_config}")
        logging.info(f"最佳F1分數: {best_f1:.4f}")
    
    logging.info(f"\n測試結果已保存到: {save_dir}")
    logging.info(f"詳細結果文件: {results_file}")
    
    return test_summary


def create_simple_confidence_analysis(predictions, save_dir):
    """簡化版本的置信度分析 - 備選方案"""
    os.makedirs(save_dir, exist_ok=True)
    
    all_scores = []
    for pred in predictions:
        if 'scores' in pred:
            all_scores.extend(pred['scores'].tolist())
    
    if not all_scores:
        logging.warning("沒有置信度分數可供分析")
        return
    
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
    save_path = os.path.join(save_dir, 'confidence_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f"置信度分析圖已保存到: {save_path}")


def create_simple_comprehensive_summary(results, save_dir):
    """簡化版本的綜合摘要 - 備選方案"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 提取指標
    configs = []
    f1_scores = []
    precisions = []
    recalls = []
    ious = []
    
    for key, result in results.items():
        if key not in ['comprehensive', 'roc_froc'] and 'metrics' in result:
            configs.append(f"C{result['confidence_threshold']}_I{result['iou_threshold']}")
            metrics = result['metrics']
            f1_scores.append(metrics.get('f1_score', 0))
            precisions.append(metrics.get('precision', 0))
            recalls.append(metrics.get('recall', 0))
            ious.append(metrics.get('iou', 0))
    
    if not configs:
        logging.warning("沒有有效的測試結果用於綜合摘要")
        return
    
    # 創建綜合摘要圖表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. F1分數比較
    axes[0, 0].bar(range(len(configs)), f1_scores, alpha=0.7, color='skyblue')
    axes[0, 0].set_xlabel('Configuration')
    axes[0, 0].set_ylabel('F1-Score')
    axes[0, 0].set_title('F1-Score by Configuration')
    axes[0, 0].set_xticks(range(len(configs)))
    axes[0, 0].set_xticklabels(configs, rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 精確度vs召回率散點圖
    axes[0, 1].scatter(recalls, precisions, s=100, alpha=0.7, c=f1_scores, cmap='viridis')
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision vs Recall')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. IoU分佈
    axes[1, 0].bar(range(len(configs)), ious, alpha=0.7, color='lightgreen')
    axes[1, 0].set_xlabel('Configuration')
    axes[1, 0].set_ylabel('Mean IoU')
    axes[1, 0].set_title('Mean IoU by Configuration')
    axes[1, 0].set_xticks(range(len(configs)))
    axes[1, 0].set_xticklabels(configs, rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 統計信息
    if f1_scores:
        best_idx = np.argmax(f1_scores)
        axes[1, 1].text(0.1, 0.9, f'Best Config: {configs[best_idx]}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.8, f'Best F1: {f1_scores[best_idx]:.4f}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.7, f'Best Precision: {precisions[best_idx]:.4f}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.6, f'Best Recall: {recalls[best_idx]:.4f}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.5, f'Best IoU: {ious[best_idx]:.4f}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.3, f'Mean F1: {np.mean(f1_scores):.4f}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.2, f'Std F1: {np.std(f1_scores):.4f}', fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].set_title('Performance Summary')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # 保存圖片
    save_path = os.path.join(save_dir, 'comprehensive_summary.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f"綜合摘要圖已保存到: {save_path}")


def create_results_summary(results, save_dir):
    """創建測試結果摘要報告"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 提取不同配置的指標
    configs = []
    f1_scores = []
    precisions = []
    recalls = []
    ious = []
    
    for key, result in results.items():
        if key not in ['comprehensive', 'roc_froc'] and 'metrics' in result:
            configs.append(f"C{result['confidence_threshold']}_I{result['iou_threshold']}")
            metrics = result['metrics']
            f1_scores.append(metrics.get('f1_score', 0))
            precisions.append(metrics.get('precision', 0))
            recalls.append(metrics.get('recall', 0))
            ious.append(metrics.get('iou', 0))
    
    if not configs:
        logging.warning("沒有有效的測試結果用於摘要")
        return
    
    # 創建摘要圖表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. F1分數比較
    axes[0, 0].bar(range(len(configs)), f1_scores, alpha=0.7, color='skyblue')
    axes[0, 0].set_xlabel('Configuration')
    axes[0, 0].set_ylabel('F1-Score')
    axes[0, 0].set_title('F1-Score by Configuration')
    axes[0, 0].set_xticks(range(len(configs)))
    axes[0, 0].set_xticklabels(configs, rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 添加數值標籤
    for i, v in enumerate(f1_scores):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # 2. 精確度vs召回率散點圖
    axes[0, 1].scatter(recalls, precisions, s=100, alpha=0.7, c=f1_scores, cmap='viridis')
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision vs Recall')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 添加配置標籤
    for i, config in enumerate(configs):
        axes[0, 1].annotate(config, (recalls[i], precisions[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 3. IoU分佈
    axes[1, 0].bar(range(len(configs)), ious, alpha=0.7, color='lightgreen')
    axes[1, 0].set_xlabel('Configuration')
    axes[1, 0].set_ylabel('Mean IoU')
    axes[1, 0].set_title('Mean IoU by Configuration')
    axes[1, 0].set_xticks(range(len(configs)))
    axes[1, 0].set_xticklabels(configs, rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 綜合指標雷達圖
    if len(configs) > 0:
        # 選擇最佳F1分數的配置
        best_idx = np.argmax(f1_scores)
        metrics_values = [precisions[best_idx], recalls[best_idx], f1_scores[best_idx], ious[best_idx]]
        metrics_labels = ['Precision', 'Recall', 'F1-Score', 'IoU']
        
        angles = np.linspace(0, 2 * np.pi, len(metrics_labels), endpoint=False).tolist()
        metrics_values += metrics_values[:1]  # 閉合圖形
        angles += angles[:1]
        
        ax_radar = plt.subplot(2, 2, 4, projection='polar')
        ax_radar.plot(angles, metrics_values, 'o-', linewidth=2, label=f'Best Config: {configs[best_idx]}')
        ax_radar.fill(angles, metrics_values, alpha=0.25)
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(metrics_labels)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('Best Configuration Metrics')
        ax_radar.grid(True)
        ax_radar.legend()
    
    plt.tight_layout()
    
    # 保存圖片
    save_path = os.path.join(save_dir, 'test_results_summary.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 創建文字報告
    report_path = os.path.join(save_dir, 'test_summary_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== 模型測試結果摘要 ===\n\n")
        
        if configs:
            best_idx = np.argmax(f1_scores)
            f.write(f"最佳配置: {configs[best_idx]}\n")
            f.write(f"  F1分數: {f1_scores[best_idx]:.4f}\n")
            f.write(f"  精確度: {precisions[best_idx]:.4f}\n")
            f.write(f"  召回率: {recalls[best_idx]:.4f}\n")
            f.write(f"  平均IoU: {ious[best_idx]:.4f}\n\n")
        
        f.write("所有配置結果:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'配置':<15} {'F1分數':<10} {'精確度':<10} {'召回率':<10} {'IoU':<10}\n")
        f.write("-" * 80 + "\n")
        
        for i, config in enumerate(configs):
            f.write(f"{config:<15} {f1_scores[i]:<10.4f} {precisions[i]:<10.4f} "
                   f"{recalls[i]:<10.4f} {ious[i]:<10.4f}\n")
    
    logging.info(f"測試結果摘要已保存到: {save_path}")
    logging.info(f"詳細報告已保存到: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Test Detection Model - 重構版本（完整評估）')
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    parser.add_argument('--model_path', type=str, default=None,
                       help='模型文件路徑，如果未指定則自動搜索')
    parser.add_argument('--data_dir', type=str, 
                       default=os.path.join(os.path.dirname(script_dir), 'datasets', 'splited_dataset'), 
                       help='測試數據集目錄路徑')
    parser.add_argument('--save_dir', type=str, 
                       default=os.path.join(script_dir, f'test_results_refactored_{timestamp}'), 
                       help='結果保存目錄')
    parser.add_argument('--log_dir', type=str, 
                       default=os.path.join(script_dir, f'test_logs_refactored_{timestamp}'), 
                       help='日誌保存目錄')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='批次大小')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val', 'train'],
                       help='要測試的數據集分割')
    parser.add_argument('--confidence_thresholds', nargs='+', type=float, 
                       default=DEFAULT_CONFIDENCE_THRESHOLDS,
                       help='置信度閾值列表')
    parser.add_argument('--iou_thresholds', nargs='+', type=float, 
                       default=DEFAULT_IOU_THRESHOLDS,
                       help='IoU閾值列表')
    parser.add_argument('--visualize_samples', type=int, default=DEFAULT_VISUALIZE_SAMPLES,
                       help='可視化樣本數量')
    parser.add_argument('--include_negative_samples', action='store_true', default=True,
                       help='包含負樣本（無標註的影像）以改善ROC/AUC計算（預設啟用）')
    parser.add_argument('--no_negative_samples', action='store_false', dest='include_negative_samples',
                       help='禁用負樣本載入，僅載入有標註的影像')
    parser.add_argument('--max_negative_per_patient', type=int, default=0,
                       help='每位患者最大負樣本數量，0表示無限制（載入所有負樣本）')
    parser.add_argument('--extract_features', action='store_true', default=True,
                       help='提取深層特徵供LLM生成報告使用（預設啟用）')
    parser.add_argument('--no_extract_features', action='store_false', dest='extract_features',
                       help='禁用深層特徵提取')
    
    args = parser.parse_args()
    
    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 設置日誌
    setup_func = setup_logging if MODULES_IMPORTED else setup_logging_fallback
    log_file = setup_func(args.log_dir)
    logging.info(f"日誌文件: {log_file}")
    
    # 配置信息
    logging.info("=== 模型測試配置（完整評估版本）===")
    logging.info(f"模組化狀態: {'啟用' if MODULES_IMPORTED else '備選模式'}")
    logging.info(f"使用設備: {device}")
    logging.info(f"數據目錄: {args.data_dir}")
    logging.info(f"測試分割: {args.split}")
    logging.info(f"批次大小: {args.batch_size}")
    logging.info(f"置信度閾值: {args.confidence_thresholds}")
    logging.info(f"IoU閾值: {args.iou_thresholds}")
    logging.info(f"可視化樣本數: {args.visualize_samples}")
    logging.info(f"負樣本設定: {'啟用' if args.include_negative_samples else '禁用'}")
    if args.include_negative_samples:
        if args.max_negative_per_patient == 0:
            logging.info(f"負樣本限制: 無限制")
        else:
            logging.info(f"負樣本限制: 每患者最多{args.max_negative_per_patient}個")
    logging.info(f"深層特徵提取: {'啟用' if args.extract_features else '禁用'}")
    
    # 檢查數據目錄
    if not os.path.exists(args.data_dir):
        logging.error(f"數據目錄不存在: {args.data_dir}")
        return
    
    # 尋找模型文件
    if args.model_path is None:
        model_path = find_model_file(script_dir)
        if model_path is None:
            logging.error("無法找到模型文件，請使用 --model_path 指定")
            return
    else:
        model_path = args.model_path
        if not os.path.exists(model_path):
            logging.error(f"指定的模型文件不存在: {model_path}")
            return
    
    logging.info(f"使用模型: {model_path}")
    
    # 開始測試
    start_time = time.time()
    
    logging.info("開始完整模型測試...")
    test_summary = test_detection_model(
        model_path=model_path,
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
    
    if test_summary is None:
        logging.error("測試失敗")
        return
    
    # 創建結果摘要
    create_results_summary(test_summary['results'], args.save_dir)
    
    total_time = time.time() - start_time
    
    logging.info(f"\n=== 完整測試完成 ===")
    logging.info(f"程式總執行時間: {total_time:.2f}秒 ({total_time/3600:.2f}小時)")
    logging.info(f"結果保存在: {args.save_dir}")


if __name__ == "__main__":
    main()
