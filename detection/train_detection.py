#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FPS-Former 目標檢測訓練腳本
基於Feature Pyramid Swin Transformer的目標檢測模型

🚀 快速使用方式:

1. 傳統訓練模式（推薦用於最終模型）:
   python train_detection.py --mode traditional

2. K-Fold 交叉驗證模式（推薦用於模型評估）:
   python train_detection.py --mode kfold

3. 自訂參數模式:
   python train_detection.py --mode custom --use_kfold --k_folds 5
   python train_detection.py --mode custom --val_ratio 0.15 --num_epochs 100

📋 主要功能：
1. 基於FPS-Former的目標檢測模型
2. 支援傳統訓練和K-Fold交叉驗證兩種模式
3. 自動從train資料分割驗證集
4. 訓練目標檢測模型（分類+邊界框回歸）
5. 評估檢測性能並視覺化結果
6. 完整的日誌記錄和TensorBoard監控

作者: GitHub Copilot
日期: 2025-08-02（更新為FPS-Former）
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

from fps_former_model import FPSFormerForDetection, create_fps_former_detection_model, create_fps_former_from_classification
from detection_dataset import CTDetectionDataset, create_detection_dataloaders

def setup_logging(log_dir):
    """設置日誌"""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def evaluate_detection_model(model, val_loader, device, logger):
    """評估檢測模型"""
    model.eval()
    
    total_loss = 0
    correct_classifications = 0
    total_samples = 0
    
    bbox_errors = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="評估中"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            bbox_targets = batch['bbox_targets'].to(device)
            
            outputs = model(pixel_values, labels, bbox_targets)
            
            # 累計損失
            total_loss += outputs['loss'].item()
            
            # 分類準確率
            class_preds = torch.argmax(outputs['class_logits'], dim=1)
            correct_classifications += (class_preds == labels).sum().item()
            total_samples += labels.size(0)
            
            # 邊界框誤差（只對有目標的樣本計算）
            positive_mask = labels > 0
            if positive_mask.sum() > 0:
                bbox_pred = outputs['bbox_pred'][positive_mask]
                bbox_target = bbox_targets[positive_mask]
                
                # 計算L1誤差
                bbox_error = torch.abs(bbox_pred - bbox_target).mean(dim=1)
                bbox_errors.extend(bbox_error.cpu().numpy())
    
    # 計算平均指標
    avg_loss = total_loss / len(val_loader)
    classification_acc = correct_classifications / total_samples
    avg_bbox_error = np.mean(bbox_errors) if bbox_errors else 0.0
    
    metrics = {
        'val_loss': avg_loss,
        'classification_accuracy': classification_acc,
        'bbox_l1_error': avg_bbox_error
    }
    
    logger.info(f"驗證結果:")
    logger.info(f"  損失: {avg_loss:.4f}")
    logger.info(f"  分類準確率: {classification_acc:.4f}")
    logger.info(f"  邊界框L1誤差: {avg_bbox_error:.4f}")
    
    return metrics

def visualize_predictions(model, dataset, device, save_path, num_samples=5):
    """視覺化預測結果"""
    model.eval()
    
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    if num_samples == 1:
        axes = axes.reshape(2, 1)
    
    class_names = {0: 'Background', 1: 'A/Adenocarcinoma', 2: 'B', 3: 'E', 4: 'G'}
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        
        # 準備輸入
        pixel_values = sample['pixel_values'].unsqueeze(0).to(device)
        true_label = sample['labels'].item()
        true_bbox = sample['bbox_targets'].numpy()
        
        # 預測
        with torch.no_grad():
            outputs = model(pixel_values)
            pred_class = torch.argmax(outputs['class_logits'], dim=1).item()
            pred_bbox = outputs['bbox_pred'][0].cpu().numpy()
            objectness = torch.sigmoid(outputs['objectness_logits'][0]).item()
        
        # 獲取影像
        image = sample['pixel_values'].permute(1, 2, 0).numpy()[:, :, 0]
        
        # 顯示原始標註
        axes[0, i].imshow(image, cmap='gray')
        axes[0, i].set_title(f'真實: {class_names[true_label]}')
        axes[0, i].axis('off')
        
        # 繪製真實邊界框
        if true_bbox[2] > 0 and true_bbox[3] > 0:
            x_center, y_center, width, height = true_bbox
            x1 = (x_center - width/2) * image.shape[1]
            y1 = (y_center - height/2) * image.shape[0]
            x2 = (x_center + width/2) * image.shape[1]
            y2 = (y_center + height/2) * image.shape[0]
            
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, color='green', linewidth=2)
            axes[0, i].add_patch(rect)
        
        # 顯示預測結果
        axes[1, i].imshow(image, cmap='gray')
        axes[1, i].set_title(f'預測: {class_names[pred_class]} ({objectness:.2f})')
        axes[1, i].axis('off')
        
        # 繪製預測邊界框
        if objectness > 0.5:  # 只顯示高置信度的預測
            x_center, y_center, width, height = pred_bbox
            x1 = (x_center - width/2) * image.shape[1]
            y1 = (y_center - height/2) * image.shape[0]
            x2 = (x_center + width/2) * image.shape[1]
            y2 = (y_center + height/2) * image.shape[0]
            
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, color='red', linewidth=2)
            axes[1, i].add_patch(rect)
    
    plt.tight_layout()
    plt.savefig(save_path)
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
        
        # 設備
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用設備: {device}")
        
        # 創建模型
        if args.classification_model_path and os.path.exists(args.classification_model_path):
            logger.info(f"嘗試從分類模型載入預訓練權重: {args.classification_model_path}")
            model = create_fps_former_from_classification(
                args.classification_model_path, 
                num_classes=args.num_classes
            )
        else:
            logger.info("創建新的FPS-Former檢測模型")
            model = create_fps_former_detection_model(
                num_classes=args.num_classes, 
                image_size=args.image_size
            )
        
        model = model.to(device)
        
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
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        logger.info(f"Fold {fold_num} - 訓練集大小: {len(train_dataset)}")
        logger.info(f"Fold {fold_num} - 驗證集大小: {len(val_dataset)}")
        
        # 優化器和學習率調度器
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
        
        # TensorBoard
        writer = SummaryWriter(log_dir)
        
        # 訓練循環
        best_val_acc = 0.0
        fold_history = []
        
        for epoch in range(args.num_epochs):
            logger.info(f"\nFold {fold_num} - Epoch {epoch+1}/{args.num_epochs}")
            
            # 訓練階段
            model.train()
            train_loss = 0
            train_steps = 0
            
            progress_bar = tqdm(train_loader, desc=f"Fold {fold_num} - Epoch {epoch+1}")
            for batch in progress_bar:
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                bbox_targets = batch['bbox_targets'].to(device)
                
                optimizer.zero_grad()
                
                outputs = model(pixel_values, labels, bbox_targets)
                loss = outputs['loss']
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_steps += 1
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_train_loss = train_loss / train_steps
            
            # 驗證階段
            logger.info("開始驗證...")
            val_metrics = evaluate_detection_model(model, val_loader, device, logger)
            
            # 學習率調整
            scheduler.step()
            
            # 記錄到TensorBoard
            writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            writer.add_scalar('Loss/Val', val_metrics['val_loss'], epoch)
            writer.add_scalar('Accuracy/Val', val_metrics['classification_accuracy'], epoch)
            writer.add_scalar('BBox_Error/Val', val_metrics['bbox_l1_error'], epoch)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
            # 記錄歷史
            fold_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': val_metrics['val_loss'],
                'val_accuracy': val_metrics['classification_accuracy'],
                'bbox_error': val_metrics['bbox_l1_error']
            })
            
            # 保存最佳模型
            if val_metrics['classification_accuracy'] > best_val_acc:
                best_val_acc = val_metrics['classification_accuracy']
                
                model_save_path = os.path.join(fold_output_dir, 'best_detection_model.pth')
                torch.save({
                    'fold': fold_num,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': best_val_acc,
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
            'best_val_accuracy': best_val_acc,
            'train_patients': fold_data['train_patients'],
            'val_patients': fold_data['val_patients'],
            'history': fold_history
        }
        all_fold_results.append(fold_result)
        
        logger.info(f"Fold {fold_num} 完成！最佳驗證準確率: {best_val_acc:.4f}")
        
        # 保存當前fold的結果
        fold_result_path = os.path.join(fold_output_dir, 'fold_results.json')
        with open(fold_result_path, 'w', encoding='utf-8') as f:
            json.dump(fold_result, f, indent=2, ensure_ascii=False)
    
    # 計算和保存總體結果
    accuracies = [result['best_val_accuracy'] for result in all_fold_results]
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    final_results = {
        'k_folds': args.k_folds,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'individual_accuracies': accuracies,
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
    print(f"平均準確率: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"各Fold準確率: {[f'{acc:.4f}' for acc in accuracies]}")
    print(f"最佳Fold: {np.argmax(accuracies) + 1} (準確率: {np.max(accuracies):.4f})")
    print(f"最差Fold: {np.argmin(accuracies) + 1} (準確率: {np.min(accuracies):.4f})")
    print(f"結果已保存到: {final_results_path}")
    
    return final_results

def train_detection_model(args):
    """訓練檢測模型"""
    
    # 設置日誌
    log_dir = os.path.join(args.output_dir, 'logs')
    logger = setup_logging(log_dir)
    
    # 設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用設備: {device}")
    
    # 創建模型
    if args.classification_model_path and os.path.exists(args.classification_model_path):
        logger.info(f"嘗試從分類模型載入預訓練權重: {args.classification_model_path}")
        model = create_fps_former_from_classification(
            args.classification_model_path, 
            num_classes=args.num_classes
        )
    else:
        logger.info("創建新的FPS-Former檢測模型")
        model = create_fps_former_detection_model(
            num_classes=args.num_classes, 
            image_size=args.image_size
        )
    
    model = model.to(device)
    
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
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f"訓練集大小: {len(train_dataset)}")
    logger.info(f"驗證集大小: {len(val_dataset)}")
    
    # 優化器和學習率調度器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    # TensorBoard
    writer = SummaryWriter(log_dir)
    
    # 訓練循環
    best_val_acc = 0.0
    for epoch in range(args.num_epochs):
        logger.info(f"\n開始訓練 Epoch {epoch+1}/{args.num_epochs}")
        
        # 訓練階段
        model.train()
        train_loss = 0
        train_steps = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in progress_bar:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            bbox_targets = batch['bbox_targets'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(pixel_values, labels, bbox_targets)
            loss = outputs['loss']
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / train_steps
        
        # 驗證階段
        logger.info("開始驗證...")
        val_metrics = evaluate_detection_model(model, val_loader, device, logger)
        
        # 學習率調整
        scheduler.step()
        
        # 記錄到TensorBoard
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val', val_metrics['val_loss'], epoch)
        writer.add_scalar('Accuracy/Val', val_metrics['classification_accuracy'], epoch)
        writer.add_scalar('BBox_Error/Val', val_metrics['bbox_l1_error'], epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # 保存最佳模型
        if val_metrics['classification_accuracy'] > best_val_acc:
            best_val_acc = val_metrics['classification_accuracy']
            
            model_save_path = os.path.join(args.output_dir, 'best_detection_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': best_val_acc,
                'val_metrics': val_metrics
            }, model_save_path)
            
            logger.info(f"保存最佳模型到: {model_save_path}")
        
        # 視覺化預測結果（每5個epoch一次）
        if (epoch + 1) % 5 == 0:
            vis_path = os.path.join(args.output_dir, f'predictions_epoch_{epoch+1}.png')
            visualize_predictions(model, val_dataset, device, vis_path)
            logger.info(f"保存預測視覺化到: {vis_path}")
    
    writer.close()
    logger.info(f"訓練完成！最佳驗證準確率: {best_val_acc:.4f}")

def print_usage_examples():
    """打印使用範例"""
    print("\n" + "="*70)
    print("📚 FPS-Former 目標檢測訓練腳本使用指南")
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
    print("   python train_detection.py --mode custom --batch_size 16 --learning_rate 5e-5")
    print()
    print("📊 結果分析（僅K-Fold模式）:")
    print("   python analyze_kfold_results.py --results_dir FPS_Former_Detection")
    print()
    print("📋 參數說明:")
    print("- --mode: 執行模式 [traditional/kfold/custom]")
    print("- --val_ratio: 驗證集比例 (傳統模式，預設0.2)")
    print("- --k_folds: Fold數量 (K-Fold模式，預設5)")
    print("- --num_epochs: 訓練輪數 (預設50)")
    print("- --batch_size: 批次大小 (預設8)")
    print("- --learning_rate: 學習率 (預設1e-4)")
    print("="*70)

def main():
    parser = argparse.ArgumentParser(description='FPS-Former目標檢測訓練')
    
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
    
    # 默認分類模型路徑（FPS-Former暫時沒有預訓練分類模型）
    default_classification_model = ""  # 留空，因為FPS-Former是新模型
    parser.add_argument('--classification_model_path', type=str, default=default_classification_model,
                      help='預訓練分類模型路徑（FPS-Former暫無預訓練模型）')
    
    # 默認輸出目錄
    default_output_dir = os.path.join(os.path.dirname(__file__), 'FPS_Former_Detection')
    parser.add_argument('--output_dir', type=str, default=default_output_dir,
                      help='輸出目錄')
    parser.add_argument('--num_classes', type=int, default=5,
                      help='類別數量（包含背景）')
    parser.add_argument('--image_size', type=int, default=224,
                      help='影像尺寸')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=50,
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
        print(f"- 驗證集比例: {args.val_ratio*100:.1f}%")
        print(f"- 訓練輪數: {args.num_epochs}")
        print(f"- 批次大小: {args.batch_size}")
        print(f"- 學習率: {args.learning_rate}")
        print(f"- 隨機種子: {args.random_seed}")
        print("說明:")
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
        print(f"- K-Fold數量: {args.k_folds}")
        print(f"- 訓練輪數: {args.num_epochs}")
        print(f"- 批次大小: {args.batch_size}")
        print(f"- 學習率: {args.learning_rate}")
        print(f"- 隨機種子: {args.random_seed}")
        print("說明:")
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
        print(f"- 使用K-Fold: {'是' if args.use_kfold else '否'}")
        if args.use_kfold:
            print(f"- K-Fold數量: {args.k_folds}")
        else:
            print(f"- 驗證集比例: {args.val_ratio*100:.1f}%")
        print(f"- 訓練輪數: {args.num_epochs}")
        print(f"- 批次大小: {args.batch_size}")
        print(f"- 學習率: {args.learning_rate}")
        print(f"- 隨機種子: {args.random_seed}")
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
