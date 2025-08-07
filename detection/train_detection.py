#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Faster R-CNN 目標檢測訓練腳本
胸部CT影像病灶檢測模型（背景 vs 病灶）

🚀 快速使用:
   python train_detection.py --mode traditional
   python train_detection.py --mode kfold

作者: GitHub Copilot
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import KFold

# 添加src路徑
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from faster_rcnn_model import FasterRCNN, create_faster_rcnn_model, create_faster_rcnn_from_classification
from faster_rcnn_dataset import CTDetectionDataset, create_detection_dataloaders, collate_fn
from gpu_dataloader_optimizer import create_optimized_dataloader, optimize_gpu_memory

def setup_logging(log_dir):
    """設置日誌"""
    os.makedirs(log_dir, exist_ok=True)
    
    # 創建formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 創建logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # 清除現有的handlers
    logger.handlers.clear()
    
    # 檔案handler（支援UTF-8編碼）
    file_handler = logging.FileHandler(
        os.path.join(log_dir, 'training.log'), 
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 控制台handler（避免Unicode問題）
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logging.getLogger(__name__)

def evaluate_detection_model(model, val_loader, device, logger):
    """評估Faster R-CNN檢測模型"""
    model.eval()
    
    total_loss = 0
    num_batches = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="評估中"):
            # 將圖片移到設備上（使用非阻塞傳輸）
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
            
            # 在評估模式下，Faster R-CNN返回預測結果
            predictions = model(images)
            
            # 收集預測和目標用於計算指標
            all_predictions.extend(predictions)
            all_targets.extend(targets)
            
            num_batches += 1
    
    # 計算檢測指標
    metrics = calculate_detection_metrics(all_predictions, all_targets, logger)
    
    return metrics

def calculate_detection_metrics(predictions, targets, logger):
    """計算檢測指標"""
    total_tp = 0  # True Positives
    total_fp = 0  # False Positives
    total_fn = 0  # False Negatives
    total_detections = 0
    total_ground_truth = 0
    
    iou_threshold = 0.5
    score_threshold = 0.5
    
    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes'].cpu()
        pred_labels = pred['labels'].cpu()
        pred_scores = pred['scores'].cpu()
        
        target_boxes = target['boxes'].cpu()
        target_labels = target['labels'].cpu()
        
        # 過濾高置信度預測
        high_conf_mask = pred_scores > score_threshold
        pred_boxes = pred_boxes[high_conf_mask]
        pred_labels = pred_labels[high_conf_mask]
        pred_scores = pred_scores[high_conf_mask]
        
        total_detections += len(pred_boxes)
        total_ground_truth += len(target_boxes)
        
        # 計算IoU矩陣
        if len(pred_boxes) > 0 and len(target_boxes) > 0:
            ious = calculate_iou_matrix(pred_boxes, target_boxes)
            
            # 匹配預測和目標
            matched_targets = set()
            
            for i, (pred_box, pred_label, pred_score) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
                best_iou = 0
                best_target_idx = -1
                
                for j, target_label in enumerate(target_labels):
                    if j not in matched_targets and pred_label == target_label:
                        iou = ious[i, j]
                        if iou > best_iou:
                            best_iou = iou
                            best_target_idx = j
                
                if best_iou >= iou_threshold:
                    total_tp += 1
                    matched_targets.add(best_target_idx)
                else:
                    total_fp += 1
            
            total_fn += len(target_boxes) - len(matched_targets)
        else:
            total_fp += len(pred_boxes)
            total_fn += len(target_boxes)
    
    # 計算精度、召回率和F1分數
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'avg_detections_per_image': total_detections / len(predictions) if predictions else 0,
        'avg_gt_per_image': total_ground_truth / len(targets) if targets else 0
    }
    
    logger.info(f"檢測評估結果:")
    logger.info(f"  精度 (Precision): {precision:.4f}")
    logger.info(f"  召回率 (Recall): {recall:.4f}")
    logger.info(f"  F1分數: {f1_score:.4f}")
    logger.info(f"  True Positives: {total_tp}")
    logger.info(f"  False Positives: {total_fp}")
    logger.info(f"  False Negatives: {total_fn}")
    
    return metrics

def calculate_iou_matrix(boxes1, boxes2):
    """計算兩組邊界框的IoU矩陣"""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # 計算交集
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom
    
    wh = (rb - lt).clamp(min=0)  # width-height
    inter = wh[:, :, 0] * wh[:, :, 1]  # intersection
    
    # 計算聯集
    union = area1[:, None] + area2 - inter
    
    # 計算IoU
    iou = inter / union
    return iou

def visualize_predictions(model, dataset, device, save_path, num_samples=5):
    """視覺化Faster R-CNN預測結果"""
    model.eval()
    
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    if num_samples == 1:
        axes = axes.reshape(2, 1)
    
    class_names = {0: 'Background', 1: 'Lesion'}
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        
        # 準備輸入
        image = sample['image'].unsqueeze(0).to(device, non_blocking=True)
        target = sample['target']
        
        # 預測
        with torch.no_grad():
            predictions = model([image.squeeze(0)])
            pred = predictions[0]
        
        # 獲取影像（轉換為可視化格式）
        img_np = sample['image'].squeeze().cpu().numpy()
        
        # 顯示原始標註
        axes[0, i].imshow(img_np, cmap='gray')
        axes[0, i].set_title(f'真實標註 - {sample["patient_id"]}')
        axes[0, i].axis('off')
        
        # 繪製真實邊界框
        true_boxes = target['boxes'].cpu().numpy()
        true_labels = target['labels'].cpu().numpy()
        
        for box, label in zip(true_boxes, true_labels):
            if label > 0:  # 忽略背景
                x1, y1, x2, y2 = box
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   fill=False, color='green', linewidth=2)
                axes[0, i].add_patch(rect)
                axes[0, i].text(x1, y1-5, f'{class_names[label]}', 
                               color='green', fontsize=8)
        
        # 顯示預測結果
        axes[1, i].imshow(img_np, cmap='gray')
        axes[1, i].set_title(f'預測結果')
        axes[1, i].axis('off')
        
        # 繪製預測邊界框（只顯示高置信度的）
        pred_boxes = pred['boxes'].cpu().numpy()
        pred_labels = pred['labels'].cpu().numpy()
        pred_scores = pred['scores'].cpu().numpy()
        
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            if score > 0.5 and label > 0:  # 置信度閾值和忽略背景
                x1, y1, x2, y2 = box
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   fill=False, color='red', linewidth=2)
                axes[1, i].add_patch(rect)
                axes[1, i].text(x1, y1-5, f'{class_names[label]} ({score:.2f})', 
                               color='red', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def create_kfold_datasets(data_root, k_folds=5, random_seed=42):
    """創建K-Fold交叉驗證資料集"""
    
    # 讀取所有患者數據
    train_patients_file = os.path.join(data_root, 'train_patients.txt')
    if not os.path.exists(train_patients_file):
        raise FileNotFoundError(f"找不到患者列表檔案: {train_patients_file}")
    
    with open(train_patients_file, 'r') as f:
        all_patients = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"總共找到 {len(all_patients)} 位患者")
    
    # 創建K-Fold分割器
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
    
    fold_splits = []
    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(all_patients)):
        train_patients = [all_patients[i] for i in train_indices]
        val_patients = [all_patients[i] for i in val_indices]
        
        fold_splits.append({
            'fold': fold_idx + 1,
            'train_patients': train_patients,
            'val_patients': val_patients,
            'train_size': len(train_patients),
            'val_size': len(val_patients)
        })
        
        print(f"Fold {fold_idx + 1}: 訓練={len(train_patients)}, 驗證={len(val_patients)}")
    
    return fold_splits

def train_detection_model_kfold(args):
    """K-Fold交叉驗證訓練"""
    
    # 創建K-Fold分割
    fold_splits = create_kfold_datasets(args.data_root, args.k_folds, args.random_seed)
    
    # 記錄所有fold的結果
    all_fold_results = []
    
    for fold_data in fold_splits:
        fold_num = fold_data['fold']
        print(f"\n{'='*50}")
        print(f"開始訓練 Fold {fold_num}/{args.k_folds}")
        print(f"{'='*50}")
        
        # 為每個fold創建獨立的輸出目錄
        fold_output_dir = os.path.join(args.output_dir, f'fold_{fold_num}')
        os.makedirs(fold_output_dir, exist_ok=True)
        
        # 設置日誌
        log_dir = os.path.join(fold_output_dir, 'logs')
        logger = setup_logging(log_dir)
        
        # 設備設定 - 強制使用GPU
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            # 清空GPU快取
            torch.cuda.empty_cache()
        else:
            device = torch.device('cpu')
            logger.warning("GPU不可用，使用CPU訓練")
        
        logger.info(f"使用設備: {device}")
        
        # 創建模型
        if args.classification_model_path and os.path.exists(args.classification_model_path):
            logger.info(f"注意：Faster R-CNN暫不支援從分類模型載入，將創建新模型")
            model = create_faster_rcnn_model(
                num_classes=args.num_classes
            )
        else:
            logger.info("創建新的Faster R-CNN檢測模型")
            model = create_faster_rcnn_model(
                num_classes=args.num_classes
            )
        
        model = model.to(device)
        
        # 檢查模型是否成功移到GPU
        if device.type == 'cuda':
            logger.info(f"模型已移至GPU")
            logger.info(f"當前GPU記憶體使用: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            
            # 優化GPU記憶體設定
            optimize_gpu_memory()
        
        # 創建當前fold的資料載入器
        logger.info("載入資料集...")
        train_dataset = CTDetectionDataset(
            data_root=args.data_root,
            split='train',
            target_size=args.image_size,
            specific_patients=fold_data['train_patients']
        )
        
        val_dataset = CTDetectionDataset(
            data_root=args.data_root,
            split='train',  # 從train資料夾中選取驗證患者
            target_size=args.image_size,
            specific_patients=fold_data['val_patients']
        )
        
        # 使用GPU優化的DataLoader
        logger.info("創建GPU優化的資料載入器...")
        train_loader = create_optimized_dataloader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        
        val_loader = create_optimized_dataloader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        logger.info(f"Fold {fold_num} - 訓練集大小: {len(train_dataset)}")
        logger.info(f"Fold {fold_num} - 驗證集大小: {len(val_dataset)}")
        
        # 優化器和學習率調度器
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
        
        # TensorBoard
        writer = SummaryWriter(log_dir)
        
        # 訓練循環
        best_val_f1 = 0.0
        fold_history = []
        
        for epoch in range(args.num_epochs):
            logger.info(f"\nFold {fold_num} - Epoch {epoch+1}/{args.num_epochs}")
            
            # 訓練階段
            model.train()
            train_losses = []
            
            progress_bar = tqdm(train_loader, desc=f"Fold {fold_num} - Epoch {epoch+1}")
            for images, targets in progress_bar:
                # 將數據移到設備上（使用非阻塞傳輸）
                images = [img.to(device, non_blocking=True) for img in images]
                targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
                
                optimizer.zero_grad()
                
                # Faster R-CNN在訓練模式下返回損失字典
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                losses.backward()
                optimizer.step()
                
                train_losses.append(losses.item())
                progress_bar.set_postfix({'loss': f'{losses.item():.4f}'})
            
            avg_train_loss = np.mean(train_losses)
            
            # GPU記憶體監控
            if device.type == 'cuda':
                gpu_memory = torch.cuda.memory_allocated()/1024**3
                logger.info(f"Epoch {epoch+1} - GPU記憶體使用: {gpu_memory:.2f} GB")
            
            # 驗證階段
            logger.info("開始驗證...")
            val_metrics = evaluate_detection_model(model, val_loader, device, logger)
            
            # 學習率調整
            scheduler.step()
            
            # 記錄到TensorBoard
            writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            writer.add_scalar('Detection/Precision', val_metrics['precision'], epoch)
            writer.add_scalar('Detection/Recall', val_metrics['recall'], epoch)
            writer.add_scalar('Detection/F1_Score', val_metrics['f1_score'], epoch)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
            # 記錄歷史
            fold_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'precision': val_metrics['precision'],
                'recall': val_metrics['recall'],
                'f1_score': val_metrics['f1_score']
            })
            
            # 保存最佳模型（基於F1分數）
            if val_metrics['f1_score'] > best_val_f1:
                best_val_f1 = val_metrics['f1_score']
                
                model_save_path = os.path.join(fold_output_dir, 'best_detection_model.pth')
                torch.save({
                    'fold': fold_num,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_f1_score': best_val_f1,
                    'val_metrics': val_metrics
                }, model_save_path)
                
                logger.info(f"保存最佳模型到: {model_save_path}")
            
            # 視覺化預測結果（每10個epoch一次）
            if (epoch + 1) % 10 == 0:
                vis_path = os.path.join(fold_output_dir, f'predictions_epoch_{epoch+1}.png')
                visualize_predictions(model, val_dataset, device, vis_path)
                logger.info(f"保存預測視覺化到: {vis_path}")
        
        writer.close()
        
        # 記錄當前fold的結果
        fold_result = {
            'fold': fold_num,
            'best_val_f1_score': best_val_f1,
            'train_patients': fold_data['train_patients'],
            'val_patients': fold_data['val_patients'],
            'history': fold_history
        }
        all_fold_results.append(fold_result)
        
        logger.info(f"Fold {fold_num} 完成！最佳F1分數: {best_val_f1:.4f}")
        
        # 保存當前fold的結果
        fold_result_path = os.path.join(fold_output_dir, 'fold_results.json')
        with open(fold_result_path, 'w', encoding='utf-8') as f:
            json.dump(fold_result, f, indent=2, ensure_ascii=False)
    
    # 計算和保存總體結果
    f1_scores = [result['best_val_f1_score'] for result in all_fold_results]
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    
    final_results = {
        'k_folds': args.k_folds,
        'mean_f1_score': mean_f1,
        'std_f1_score': std_f1,
        'individual_f1_scores': f1_scores,
        'all_fold_results': all_fold_results
    }
    
    # 保存總體結果
    final_results_path = os.path.join(args.output_dir, 'kfold_final_results.json')
    with open(final_results_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    # 打印總體結果
    print(f"\n{'='*60}")
    print(f"K-Fold 交叉驗證結果總結")
    print(f"{'='*60}")
    print(f"平均F1分數: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"各Fold F1分數: {[f'{f1:.4f}' for f1 in f1_scores]}")
    print(f"最佳Fold: {np.argmax(f1_scores) + 1} (F1分數: {np.max(f1_scores):.4f})")
    print(f"最差Fold: {np.argmin(f1_scores) + 1} (F1分數: {np.min(f1_scores):.4f})")
    print(f"結果已保存到: {final_results_path}")
    
    return final_results

def train_detection_model(args):
    """訓練檢測模型"""
    
    # 設置日誌
    log_dir = os.path.join(args.output_dir, 'logs')
    logger = setup_logging(log_dir)
    
    # 設備設定 - 強制使用GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        # 清空GPU快取
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
        logger.warning("GPU不可用，使用CPU訓練")
    
    logger.info(f"使用設備: {device}")
    
    # 創建模型
    if args.classification_model_path and os.path.exists(args.classification_model_path):
        logger.info(f"注意：Faster R-CNN暫不支援從分類模型載入，將創建新模型")
        model = create_faster_rcnn_model(
            num_classes=args.num_classes
        )
    else:
        logger.info("創建新的Faster R-CNN檢測模型")
        model = create_faster_rcnn_model(
            num_classes=args.num_classes
        )
    
    model = model.to(device)
    
    # 檢查模型是否成功移到GPU
    if device.type == 'cuda':
        logger.info(f"模型已移至GPU")
        logger.info(f"當前GPU記憶體使用: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        
        # 優化GPU記憶體設定
        optimize_gpu_memory()
    
    # 創建資料載入器 - 從train資料中分割訓練和驗證集
    logger.info("載入資料集...")
    
    # 讀取所有訓練患者
    train_patients_file = os.path.join(args.data_root, 'train_patients.txt')
    if not os.path.exists(train_patients_file):
        raise FileNotFoundError(f"找不到患者列表檔案: {train_patients_file}")
    
    with open(train_patients_file, 'r') as f:
        all_patients = [line.strip() for line in f.readlines() if line.strip()]
    
    logger.info(f"總共找到 {len(all_patients)} 位患者")
    
    # 分割訓練和驗證患者 (80% 訓練, 20% 驗證)
    np.random.seed(args.random_seed)
    shuffled_patients = np.random.permutation(all_patients)
    
    val_ratio = getattr(args, 'val_ratio', 0.2)  # 默認 20% 作為驗證集
    val_size = int(len(shuffled_patients) * val_ratio)
    
    train_patients = shuffled_patients[val_size:].tolist()
    val_patients = shuffled_patients[:val_size].tolist()
    
    logger.info(f"訓練患者: {len(train_patients)} 位")
    logger.info(f"驗證患者: {len(val_patients)} 位")
    logger.info(f"驗證比例: {val_ratio*100:.1f}%")
    
    # 創建資料集
    train_dataset = CTDetectionDataset(
        data_root=args.data_root,
        split='train',
        target_size=args.image_size,
        specific_patients=train_patients
    )
    
    val_dataset = CTDetectionDataset(
        data_root=args.data_root,
        split='train',  # 從train資料夾中選取驗證患者
        target_size=args.image_size,
        specific_patients=val_patients
    )
    
    # 使用GPU優化的DataLoader
    logger.info("創建GPU優化的資料載入器...")
    train_loader = create_optimized_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = create_optimized_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    logger.info(f"訓練集大小: {len(train_dataset)}")
    logger.info(f"驗證集大小: {len(val_dataset)}")
    
    # 優化器和學習率調度器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    # TensorBoard
    writer = SummaryWriter(log_dir)
    
    # 訓練循環
    best_val_f1 = 0.0
    for epoch in range(args.num_epochs):
        logger.info(f"\n開始訓練 Epoch {epoch+1}/{args.num_epochs}")
        
        # 訓練階段
        model.train()
        train_losses = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for images, targets in progress_bar:
            # 將數據移到設備上（使用非阻塞傳輸）
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            
            # Faster R-CNN在訓練模式下返回損失字典
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            losses.backward()
            optimizer.step()
            
            train_losses.append(losses.item())
            progress_bar.set_postfix({'loss': f'{losses.item():.4f}'})
        
        avg_train_loss = np.mean(train_losses)
        
        # GPU記憶體監控
        if device.type == 'cuda':
            gpu_memory = torch.cuda.memory_allocated()/1024**3
            logger.info(f"Epoch {epoch+1} - GPU記憶體使用: {gpu_memory:.2f} GB")
        
        # 驗證階段
        logger.info("開始驗證...")
        val_metrics = evaluate_detection_model(model, val_loader, device, logger)
        
        # 學習率調整
        scheduler.step()
        
        # 記錄到TensorBoard
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Detection/Precision', val_metrics['precision'], epoch)
        writer.add_scalar('Detection/Recall', val_metrics['recall'], epoch)
        writer.add_scalar('Detection/F1_Score', val_metrics['f1_score'], epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # 保存最佳模型（基於F1分數）
        if val_metrics['f1_score'] > best_val_f1:
            best_val_f1 = val_metrics['f1_score']
            
            model_save_path = os.path.join(args.output_dir, 'best_detection_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1_score': best_val_f1,
                'val_metrics': val_metrics
            }, model_save_path)
            
            logger.info(f"保存最佳模型到: {model_save_path}")
        
        # 視覺化預測結果（每5個epoch一次）
        if (epoch + 1) % 5 == 0:
            vis_path = os.path.join(args.output_dir, f'predictions_epoch_{epoch+1}.png')
            visualize_predictions(model, val_dataset, device, vis_path)
            logger.info(f"保存預測視覺化到: {vis_path}")
    
    writer.close()
    logger.info(f"訓練完成！最佳F1分數: {best_val_f1:.4f}")

def print_usage_examples():
    """打印使用範例"""
    print("\n" + "="*70)
    print("📚 Faster R-CNN 目標檢測訓練腳本使用指南")
    print("="*70)
    print("\n🚀 快速開始:")
    print("1. 傳統訓練模式（推薦用於最終模型訓練）:")
    print("   python train_detection.py --mode traditional")
    print()
    print("2. K-Fold 交叉驗證模式（推薦用於模型評估）:")
    print("   python train_detection.py --mode kfold")
    print()
    print("⚙️  進階使用:")
    print("3. 自訂傳統訓練參數:")
    print("   python train_detection.py --mode custom --val_ratio 0.15 --num_epochs 100")
    print()
    print("4. 自訂K-Fold參數:")
    print("   python train_detection.py --mode custom --use_kfold --k_folds 10 --num_epochs 30")
    print()
    print("5. 完全自訂:")
    print("   python train_detection.py --mode custom --batch_size 32 --learning_rate 5e-5")
    print()
    print("📊 結果分析（僅K-Fold模式）:")
    print("   python analyze_kfold_results.py --results_dir Faster_RCNN_Detection")
    print()
    print("📋 參數說明:")
    print("- --mode: 執行模式 [traditional/kfold/custom]")
    print("- --val_ratio: 驗證集比例 (傳統模式，預設0.2)")
    print("- --k_folds: Fold數量 (K-Fold模式，預設5)")
    print("- --num_epochs: 訓練輪數 (預設100)")
    print("- --batch_size: 批次大小 (預設12)")
    print("- --learning_rate: 學習率 (預設1e-4)")
    print("- --num_classes: 類別數量 (預設2：背景+病灶)")
    print("="*70)

def main():
    parser = argparse.ArgumentParser(description='Faster R-CNN目標檢測訓練')
    
    # 添加執行模式選擇
    parser.add_argument('--mode', type=str, choices=['traditional', 'kfold', 'custom'], 
                      default='custom',
                      help='執行模式: traditional(傳統訓練), kfold(K-Fold交叉驗證), custom(自訂參數)')
    
    # 添加幫助選項
    parser.add_argument('--help-examples', action='store_true',
                      help='顯示使用範例')
    
    # 默認資料路徑設為splited_dataset
    default_data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets', 'splited_dataset')
    parser.add_argument('--data_root', type=str, default=default_data_root,
                      help='資料根目錄')
    
    # 默認分類模型路徑（Faster R-CNN暫時沒有預訓練分類模型）
    default_classification_model = ""  # 留空，因為Faster R-CNN不支援從分類模型轉移
    parser.add_argument('--classification_model_path', type=str, default=default_classification_model,
                      help='預訓練分類模型路徑（Faster R-CNN暫不支援）')
    
    # 默認輸出目錄
    default_output_dir = os.path.join(os.path.dirname(__file__), 'Faster_RCNN_Detection')
    parser.add_argument('--output_dir', type=str, default=default_output_dir,
                      help='輸出目錄')
    parser.add_argument('--num_classes', type=int, default=2,
                      help='類別數量（預設2：背景+病灶）')
    parser.add_argument('--image_size', type=int, default=512,
                      help='影像尺寸')
    parser.add_argument('--batch_size', type=int, default=12,
                      help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=100,
                      help='訓練輪數')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='學習率')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                      help='驗證集比例（在非K-Fold模式下使用）')
    
    # K-Fold 相關參數
    parser.add_argument('--use_kfold', action='store_true',
                      help='是否使用K-Fold交叉驗證')
    parser.add_argument('--k_folds', type=int, default=5,
                      help='K-Fold的fold數量')
    parser.add_argument('--random_seed', type=int, default=42,
                      help='隨機種子')
    
    args = parser.parse_args()
    
    # 如果要求顯示範例，顯示後退出
    if args.help_examples:
        print_usage_examples()
        return
    
    # 根據模式設定參數
    if args.mode == 'traditional':
        print("="*60)
        print("🚀 傳統訓練模式（從train資料中分割驗證集）")
        print("="*60)
        print("參數設定:")
        print(f"- 檢測類別: 2類（背景 + 病灶）")
        print(f"- 驗證集比例: {args.val_ratio*100:.1f}%")
        print(f"- 訓練輪數: {args.num_epochs}")
        print(f"- 批次大小: {args.batch_size}")
        print(f"- 學習率: {args.learning_rate}")
        print(f"- 隨機種子: {args.random_seed}")
        print("說明:")
        print("- 使用Faster R-CNN進行二分類目標檢測")
        print("- 將從train_patients.txt中的患者隨機分割")
        print(f"- {(1-args.val_ratio)*100:.0f}%用於訓練，{args.val_ratio*100:.0f}%用於驗證")
        print("- 使用固定隨機種子確保可重現性")
        print("="*60)
        
        # 傳統訓練模式不使用K-Fold
        args.use_kfold = False
        
    elif args.mode == 'kfold':
        print("="*60)
        print("🔄 K-Fold 交叉驗證模式")
        print("="*60)
        print("參數設定:")
        print(f"- 檢測類別: 2類（背景 + 病灶）")
        print(f"- K-Fold數量: {args.k_folds}")
        print(f"- 訓練輪數: {args.num_epochs}")
        print(f"- 批次大小: {args.batch_size}")
        print(f"- 學習率: {args.learning_rate}")
        print(f"- 隨機種子: {args.random_seed}")
        print("說明:")
        print("- 使用Faster R-CNN進行二分類目標檢測")
        print("- 使用K-Fold交叉驗證評估模型性能")
        print("- 每個fold都會產生獨立的模型和評估結果")
        print("- 最終產生平均性能統計")
        print("="*60)
        
        # K-Fold模式強制啟用K-Fold
        args.use_kfold = True
        
    else:  # custom mode
        print("="*60)
        print("⚙️  自訂參數模式")
        print("="*60)
        print("參數設定:")
        print(f"- 檢測類別: {args.num_classes}類")
        print(f"- 使用K-Fold: {'是' if args.use_kfold else '否'}")
        if args.use_kfold:
            print(f"- K-Fold數量: {args.k_folds}")
        else:
            print(f"- 驗證集比例: {args.val_ratio*100:.1f}%")
        print(f"- 訓練輪數: {args.num_epochs}")
        print(f"- 批次大小: {args.batch_size}")
        print(f"- 學習率: {args.learning_rate}")
        print(f"- 隨機種子: {args.random_seed}")
        print("說明:")
        print("- 使用Faster R-CNN進行目標檢測")
        print("="*60)
    
    # 創建輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存參數
    with open(os.path.join(args.output_dir, 'training_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    try:
        # 根據參數選擇訓練方式
        if args.use_kfold:
            print(f"開始執行 {args.k_folds}-Fold 交叉驗證訓練...")
            train_detection_model_kfold(args)
            print("="*60)
            print("✅ K-Fold 交叉驗證訓練完成！")
            print("📊 請使用 analyze_kfold_results.py 分析結果")
        else:
            print("開始執行傳統訓練...")
            train_detection_model(args)
            print("="*60)
            print("✅ 傳統訓練完成！")
            
    except KeyboardInterrupt:
        print("\n❌ 訓練被用戶中斷")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 訓練過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
