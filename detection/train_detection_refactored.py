#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
簡化版本的 Faster R-CNN 訓練腳本 - 重構版本
使用訓練/驗證分割（不使用 K-Fold）
將計算模組化以提高代碼可維護性
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
from tqdm import tqdm

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
    from metrics.dataset_statistics import calculate_dataset_statistics, save_patient_lists
    from visualization import visualize_predictions, create_prediction_summary, create_comprehensive_summary
    from data_processing import create_train_val_datasets
    from evaluation import evaluate_model
    from utils import collate_fn, setup_logging
    MODULES_IMPORTED = True
    logging.info("已成功導入模組化計算函數")
except ImportError as e:
    logging.warning(f"模組化導入失敗: {e}")
    logging.warning("將使用內聯函數作為備選方案")
    MODULES_IMPORTED = False


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
    log_file = os.path.join(log_dir, f'simple_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
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


def simple_detection_metrics(predictions, targets, iou_threshold=0.5):
    """簡化版本的檢測指標計算 - 備選方案"""
    tp, fp, fn = 0, 0, 0
    
    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes']
        pred_scores = pred['scores']
        target_boxes = target['boxes']
        
        # 過濾低置信度預測
        valid_pred = pred_scores > 0.5
        filtered_pred_boxes = pred_boxes[valid_pred]
        
        if len(filtered_pred_boxes) == 0 and len(target_boxes) == 0:
            continue
        elif len(filtered_pred_boxes) == 0:
            fn += len(target_boxes)
            continue
        elif len(target_boxes) == 0:
            fp += len(filtered_pred_boxes)
            continue
        
        # 簡化的IoU計算和匹配
        matched_targets = set()
        for i, pred_box in enumerate(filtered_pred_boxes):
            best_iou = 0
            best_target = -1
            for j, target_box in enumerate(target_boxes):
                if j not in matched_targets:
                    # 簡化的IoU計算
                    x1 = max(pred_box[0], target_box[0])
                    y1 = max(pred_box[1], target_box[1])
                    x2 = min(pred_box[2], target_box[2])
                    y2 = min(pred_box[3], target_box[3])
                    
                    intersection = max(0, x2 - x1) * max(0, y2 - y1)
                    area1 = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
                    area2 = (target_box[2] - target_box[0]) * (target_box[3] - target_box[1])
                    union = area1 + area2 - intersection
                    
                    iou = intersection / union if union > 0 else 0
                    
                    if iou > best_iou:
                        best_iou = iou
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


def evaluate_model_fallback(model, val_loader, device):
    """簡化版本的模型評估 - 備選方案"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="評估模型"):
            images = [img.to(device) for img in images]
            predictions = model(images)
            
            for pred, target in zip(predictions, targets):
                all_predictions.append({
                    'boxes': pred['boxes'].cpu(),
                    'scores': pred['scores'].cpu(),
                    'labels': pred['labels'].cpu()
                })
                all_targets.append({
                    'boxes': target['boxes'],
                    'labels': target['labels']
                })
    
    metrics = simple_detection_metrics(all_predictions, all_targets)
    return metrics


def create_simple_datasets(data_dir, val_split=0.2, random_seed=42):
    """簡化版本的數據集創建 - 備選方案"""
    # 載入完整數據集
    full_dataset = CTDetectionDataset(
        data_root=data_dir,
        split='train',
        target_size=512,
        transforms=transforms.Compose([transforms.ToTensor()])
    )
    
    # 簡單的隨機分割
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    torch.manual_seed(random_seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    return train_dataset, val_dataset


def train_simple(data_dir, num_epochs=50, batch_size=8, learning_rate=0.001, 
                 val_split=0.2, save_dir='./models', log_dir='./logs',
                 include_negative_samples=True, max_negative_per_patient=0):
    """簡化的訓練/驗證分割訓練"""
    
    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"使用設備: {device}")
    
    # 創建保存目錄
    os.makedirs(save_dir, exist_ok=True)
    
    # 創建數據集
    if MODULES_IMPORTED:
        try:
            train_dataset, val_dataset, dataset_stats = create_train_val_datasets(
                data_dir, val_split, random_seed=42, 
                include_negative_samples=include_negative_samples,
                max_negative_per_patient=max_negative_per_patient
            )
            
            # 保存病例列表
            save_patient_lists(save_dir, dataset_stats)
            
        except Exception as e:
            logging.warning(f"使用模組化數據集創建失敗: {e}")
            logging.warning("切換到簡化版本")
            train_dataset, val_dataset = create_simple_datasets(data_dir, val_split)
    else:
        train_dataset, val_dataset = create_simple_datasets(data_dir, val_split)
    
    # 選擇適當的collate函數
    collate_func = collate_fn if MODULES_IMPORTED else collate_fn_fallback
    
    # 創建數據加載器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_func,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_func,
        num_workers=0
    )
    
    # 創建模型
    logging.info("載入預訓練模型...")
    model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
    num_classes = 2  # 背景 + 病灶
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)
    logging.info("模型載入完成")
    
    # 優化器和調度器
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # TensorBoard記錄器
    writer = SummaryWriter(log_dir)
    
    # 訓練循環
    start_time = time.time()
    best_f1 = 0
    train_history = []
    val_history = []
    
    # 選擇適當的評估函數
    eval_func = evaluate_model if MODULES_IMPORTED else evaluate_model_fallback
    
    for epoch in range(num_epochs):
        # 訓練階段
        model.train()
        train_losses = []
        
        train_pbar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch + 1}/{num_epochs} [訓練]",
            unit="batch",
            ncols=100
        )
        
        for batch_idx, (images, targets) in enumerate(train_pbar):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            losses.backward()
            optimizer.step()
            
            train_losses.append(losses.item())
            
            # 更新進度條
            current_loss = losses.item()
            avg_loss = np.mean(train_losses)
            train_pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Avg': f'{avg_loss:.4f}'
            })
        
        train_pbar.close()
        scheduler.step()
        
        # 驗證階段
        val_metrics = eval_func(model, val_loader, device)
        avg_train_loss = np.mean(train_losses)
        
        # Debug: 打印可用的metrics鍵
        if epoch == 0:  # 只在第一個epoch打印
            logging.info(f"可用的metrics鍵: {list(val_metrics.keys())}")
        
        # 記錄歷史
        train_history.append(avg_train_loss)
        val_history.append(val_metrics)
        
        # 記錄到TensorBoard
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Metrics/Precision', val_metrics['precision'], epoch)
        
        # 處理不同的recall鍵名
        recall_value = val_metrics.get('recall', val_metrics.get('sensitivity_recall', 0))
        writer.add_scalar('Metrics/Recall', recall_value, epoch)
        writer.add_scalar('Metrics/F1', val_metrics['f1_score'], epoch)
        
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
            }, os.path.join(save_dir, 'best_model.pth'))
            
            # 也保存純模型權重
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model_weights.pth'))
        
        # 定期保存檢查點
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_metrics': val_metrics
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # 每個 epoch 記錄結果
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            # 處理不同的recall鍵名
            recall_value = val_metrics.get('recall', val_metrics.get('sensitivity_recall', 0))
            
            logging.info(f"Epoch {epoch + 1}/{num_epochs}: "
                       f"Loss: {avg_train_loss:.4f}, "
                       f"Precision: {val_metrics['precision']:.4f}, "
                       f"Recall: {recall_value:.4f}, "
                       f"F1: {val_metrics['f1_score']:.4f}")
    
    total_time = time.time() - start_time
    
    # 載入最佳模型進行最終評估
    logging.info("載入最佳模型進行最終評估...")
    checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 最終評估
    final_metrics = eval_func(model, val_loader, device)
    
    # 保存訓練結果
    results = {
        'final_metrics': final_metrics,
        'best_f1': best_f1,
        'total_training_time': total_time,
        'train_history': train_history,
        'val_history': val_history,
        'config': {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'val_split': val_split,
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset)
        }
    }
    
    with open(os.path.join(save_dir, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # 輸出最終結果
    logging.info(f"\n=== 訓練完成 ===")
    logging.info(f"最佳 F1 分數: {best_f1:.4f}")
    logging.info(f"最終評估結果:")
    logging.info(f"  精確度: {final_metrics['precision']:.4f}")
    
    # 處理不同的recall鍵名
    final_recall = final_metrics.get('recall', final_metrics.get('sensitivity_recall', 0))
    logging.info(f"  召回率: {final_recall:.4f}")
    logging.info(f"  F1分數: {final_metrics['f1_score']:.4f}")
    logging.info(f"總訓練時間: {total_time:.2f}秒")
    
    writer.close()
    return results


def main():
    parser = argparse.ArgumentParser(description='簡化版本的 Faster R-CNN 訓練腳本 - 重構版本')
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_folder = f'Simple_Training_Refactored_{timestamp}'
    
    parser.add_argument('--data_dir', type=str, 
                       default=os.path.join(os.path.dirname(script_dir), 'datasets', 'splited_dataset'), 
                       help='數據集目錄路徑')
    parser.add_argument('--num_epochs', type=int, default=5, help='訓練輪數')
    parser.add_argument('--batch_size', type=int, default=12, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='學習率')
    parser.add_argument('--val_split', type=float, default=0.2, help='驗證集比例 (0.0-1.0)')
    parser.add_argument('--save_dir', type=str, 
                       default=os.path.join(script_dir, training_folder, 'models'), 
                       help='模型保存目錄')
    parser.add_argument('--log_dir', type=str, 
                       default=os.path.join(script_dir, training_folder, 'logs'), 
                       help='日誌保存目錄')
    parser.add_argument('--random_seed', type=int, default=42, help='隨機種子')
    parser.add_argument('--include_negative_samples', action='store_true', default=True,
                       help='包含負樣本（無標註的影像）')
    parser.add_argument('--max_negative_per_patient', type=int, default=10,
                       help='每位患者最大負樣本數量，0表示無限制')
    
    args = parser.parse_args()
    
    # 設置隨機種子
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # 設置日誌
    setup_func = setup_logging if MODULES_IMPORTED else setup_logging_fallback
    log_file = setup_func(args.log_dir)
    logging.info(f"日誌文件: {log_file}")
    
    # 檢查數據目錄
    if not os.path.exists(args.data_dir):
        logging.error(f"數據目錄不存在: {args.data_dir}")
        return
    
    # 配置信息
    logging.info("=== 訓練配置 ===")
    logging.info(f"模組化狀態: {'啟用' if MODULES_IMPORTED else '備選模式'}")
    logging.info(f"數據目錄: {args.data_dir}")
    logging.info(f"訓練設定: {args.num_epochs} epochs, batch={args.batch_size}, lr={args.learning_rate}")
    logging.info(f"驗證集比例: {args.val_split:.1%}, 隨機種子: {args.random_seed}")
    
    # 開始訓練
    start_time = time.time()
    results = train_simple(
        data_dir=args.data_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_split=args.val_split,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        include_negative_samples=args.include_negative_samples,
        max_negative_per_patient=args.max_negative_per_patient
    )
    
    total_time = time.time() - start_time
    logging.info(f"程式總執行時間: {total_time:.2f}秒 ({total_time/3600:.2f}小時)")


if __name__ == "__main__":
    main()
