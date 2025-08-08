"""
Simple Train/Validation Split Training for Faster R-CNN Detection
不使用 K-Fold，直接進行訓練/驗證分割的版本
"""

import os
import json
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
from tqdm import tqdm

from faster_rcnn_dataset import CTDetectionDataset


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
    log_file = os.path.join(log_dir, f'simple_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file


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


def create_train_val_datasets(data_dir, val_split=0.2, random_seed=42):
    """創建訓練/驗證數據集分割"""
    # 載入完整數據集
    dataset = CTDetectionDataset(
        data_root=data_dir,
        split='train',
        target_size=512,
        specific_patients=None,
        transforms=transforms.Compose([
            transforms.ToTensor()
        ])
    )
    
    # 計算分割大小
    dataset_size = len(dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    # 設置隨機種子以確保可重現性
    torch.manual_seed(random_seed)
    
    # 隨機分割數據集
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    logging.info(f"數據集總大小: {dataset_size}")
    logging.info(f"訓練集大小: {train_size} ({(1-val_split)*100:.1f}%)")
    logging.info(f"驗證集大小: {val_size} ({val_split*100:.1f}%)")
    
    return train_dataset, val_dataset


def evaluate_model(model, val_loader, device):
    """評估模型"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    val_pbar = tqdm(val_loader, desc="評估模型", unit="batch", ncols=100)
    
    with torch.no_grad():
        for images, targets in val_pbar:
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
            
            val_pbar.set_postfix({'Samples': f'{len(all_predictions)}'})
    
    val_pbar.close()
    
    # 計算指標
    metrics = calculate_detection_metrics(all_predictions, all_targets, iou_threshold=0.5)
    return metrics


def train_simple(data_dir, num_epochs=50, batch_size=8, learning_rate=0.001, 
                 val_split=0.2, save_dir='./models', log_dir='./logs'):
    """簡單的訓練/驗證分割訓練"""
    
    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"使用設備: {device}")
    
    # 創建保存目錄
    os.makedirs(save_dir, exist_ok=True)
    
    # 創建訓練/驗證數據集
    train_dataset, val_dataset = create_train_val_datasets(data_dir, val_split)
    
    # 創建數據加載器
    logging.info("創建數據加載器...")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0  # 設為0避免Windows多進程問題
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0  # 設為0避免Windows多進程問題
    )
    
    # 創建模型
    logging.info("載入預訓練模型...")
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 2  # 背景 + 病灶
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)
    logging.info("模型載入完成")
    
    # 優化器和調度器
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # TensorBoard記錄器
    writer = SummaryWriter(log_dir)
    
    # 訓練循環
    start_time = time.time()
    best_f1 = 0
    train_history = []
    val_history = []
    
    # 創建總體訓練進度條
    epoch_pbar = tqdm(range(num_epochs), desc="訓練進度", unit="epoch", ncols=120)
    
    for epoch in epoch_pbar:
        # 訓練階段
        model.train()
        train_losses = []
        
        # 訓練進度條
        train_pbar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch + 1}/{num_epochs} [訓練]",
            unit="batch",
            ncols=100,
            leave=False
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
        val_metrics = evaluate_model(model, val_loader, device)
        avg_train_loss = np.mean(train_losses)
        
        # 記錄歷史
        train_history.append(avg_train_loss)
        val_history.append(val_metrics)
        
        # 記錄到TensorBoard
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Metrics/Precision', val_metrics['precision'], epoch)
        writer.add_scalar('Metrics/Recall', val_metrics['recall'], epoch)
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
            
            # 也保存純模型權重，方便推理使用
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
        
        # 更新總體進度條
        epoch_pbar.set_postfix({
            'Loss': f'{avg_train_loss:.4f}',
            'F1': f'{val_metrics["f1_score"]:.4f}',
            'Best_F1': f'{best_f1:.4f}'
        })
        
        # 每個 epoch 記錄結果
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            logging.info(f"Epoch {epoch + 1}/{num_epochs}: "
                       f"Loss: {avg_train_loss:.4f}, "
                       f"Precision: {val_metrics['precision']:.4f}, "
                       f"Recall: {val_metrics['recall']:.4f}, "
                       f"F1: {val_metrics['f1_score']:.4f}")
    
    epoch_pbar.close()
    total_time = time.time() - start_time
    
    # 載入最佳模型進行最終評估
    logging.info("載入最佳模型進行最終評估...")
    checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    final_metrics = evaluate_model(model, val_loader, device)
    
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
    logging.info(f"最終精確度: {final_metrics['precision']:.4f}")
    logging.info(f"最終召回率: {final_metrics['recall']:.4f}")
    logging.info(f"最終F1分數: {final_metrics['f1_score']:.4f}")
    logging.info(f"總訓練時間: {total_time:.2f}秒 ({total_time/3600:.2f}小時)")
    
    writer.close()
    return results


def main():
    parser = argparse.ArgumentParser(description='Simple Train/Validation Split Training for Faster R-CNN')
    
    # 獲取腳本所在目錄
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser.add_argument('--data_dir', type=str, 
                       default=os.path.join(os.path.dirname(script_dir), 'datasets', 'splited_dataset'), 
                       help='數據集目錄路徑')
    parser.add_argument('--num_epochs', type=int, default=50, 
                       help='訓練輪數')
    parser.add_argument('--batch_size', type=int, default=16, 
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.0001, 
                       help='學習率')
    parser.add_argument('--val_split', type=float, default=0.2, 
                       help='驗證集比例 (0.0-1.0)')
    parser.add_argument('--save_dir', type=str, 
                       default=os.path.join(script_dir, 'Simple_Training', 'models'), 
                       help='模型保存目錄')
    parser.add_argument('--log_dir', type=str, 
                       default=os.path.join(script_dir, 'Simple_Training', 'logs'), 
                       help='日誌保存目錄')
    parser.add_argument('--random_seed', type=int, default=42, 
                       help='隨機種子')
    
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
    
    # 輸出配置信息
    logging.info("=== 訓練配置 ===")
    logging.info(f"數據目錄: {args.data_dir}")
    logging.info(f"訓練輪數: {args.num_epochs}")
    logging.info(f"批次大小: {args.batch_size}")
    logging.info(f"學習率: {args.learning_rate}")
    logging.info(f"驗證集比例: {args.val_split}")
    logging.info(f"模型保存目錄: {args.save_dir}")
    logging.info(f"日誌目錄: {args.log_dir}")
    logging.info(f"隨機種子: {args.random_seed}")
    
    # 開始訓練
    start_time = time.time()
    results = train_simple(
        data_dir=args.data_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_split=args.val_split,
        save_dir=args.save_dir,
        log_dir=args.log_dir
    )
    
    total_time = time.time() - start_time
    logging.info(f"程式總執行時間: {total_time:.2f}秒 ({total_time/3600:.2f}小時)")


if __name__ == "__main__":
    main()
