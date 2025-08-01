#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CT-ViT 目標檢測訓練腳本
將現有的分類模型升級為目標檢測模型

使用方式:
python train_detection.py --data_root matched_data_by_patient --classification_model_path CT_ViT/models/best_model.pth

主要功能：
1. 從現有分類模型載入預訓練權重
2. 訓練目標檢測模型
3. 評估檢測性能
4. 視覺化檢測結果

作者: GitHub Copilot
日期: 2025-07-25
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

# 添加src路徑
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from detection_model import CTViTForDetection, create_detection_model_from_classification
from detection_dataset import CTDetectionDataset, create_detection_dataloaders
from transformers import ViTConfig

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
        logger.info(f"從分類模型載入預訓練權重: {args.classification_model_path}")
        model = create_detection_model_from_classification(
            args.classification_model_path, 
            num_classes=args.num_classes
        )
    else:
        logger.info("創建新的檢測模型")
        config = ViTConfig(
            image_size=args.image_size,
            patch_size=16,
            num_channels=3,  # RGB格式
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            num_labels=args.num_classes,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        model = CTViTForDetection(config)
    
    model = model.to(device)
    
    # 創建資料載入器
    logger.info("載入資料集...")
    train_dataset = CTDetectionDataset(
        data_root=args.data_root,
        split='train',
        target_size=args.image_size
    )
    
    val_dataset = CTDetectionDataset(
        data_root=args.data_root,
        split='val',
        target_size=args.image_size
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

def main():
    parser = argparse.ArgumentParser(description='CT-ViT目標檢測訓練')
    
    # 默認資料路徑設為上層目錄的matched_data_by_patient
    default_data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'matched_data_by_patient')
    parser.add_argument('--data_root', type=str, default=default_data_root,
                      help='資料根目錄')
    
    # 默認分類模型路徑
    default_classification_model = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'CT_ViT', 'models', 'best_model.pth')
    parser.add_argument('--classification_model_path', type=str, default=default_classification_model,
                      help='預訓練分類模型路徑')
    
    # 默認輸出目錄
    default_output_dir = os.path.join(os.path.dirname(__file__), 'CT_ViT_Detection')
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
    
    args = parser.parse_args()
    
    # 創建輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存參數
    with open(os.path.join(args.output_dir, 'training_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # 開始訓練
    train_detection_model(args)

if __name__ == "__main__":
    main()
