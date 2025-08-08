"""
Optimized K-Fold Cross-Validation Training for Faster R-CNN Detection
Simplified version focusing only on K-fold cross-validation training.
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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from sklearn.model_selection import KFold
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
    log_file = os.path.join(log_dir, f'kfold_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file


def evaluate_detection_model(model, data_loader, device, iou_threshold=0.5):
    """評估檢測模型性能"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in data_loader:
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
    
    metrics = calculate_detection_metrics(all_predictions, all_targets, iou_threshold)
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


def create_kfold_datasets(data_dir, k_folds=5):
    """創建K-fold數據集分割"""
    dataset = CTDetectionDataset(
        data_root=data_dir,
        split='train',
        target_size=512,
        specific_patients=None,
        transforms=transforms.Compose([
            transforms.ToTensor()
        ])
    )
    
    indices = list(range(len(dataset)))
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    fold_datasets = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)
        fold_datasets.append((train_dataset, val_dataset))
        
        logging.info(f"Fold {fold + 1}: Training samples: {len(train_idx)}, Validation samples: {len(val_idx)}")
    
    return fold_datasets


def train_kfold(data_dir, k_folds=5, num_epochs=50, batch_size=8, learning_rate=0.001, 
                save_dir='./models', log_dir='./logs'):
    """K-fold交叉驗證訓練"""
    
    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"使用設備: {device}")
    
    # 創建保存目錄
    os.makedirs(save_dir, exist_ok=True)
    
    # 創建K-fold數據集
    fold_datasets = create_kfold_datasets(data_dir, k_folds)
    
    # 存儲所有fold的結果
    all_fold_results = []
    
    # 創建總體進度條
    fold_pbar = tqdm(
        fold_datasets, 
        desc="K-Fold 交叉驗證進度",
        unit="fold",
        ncols=100
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
        model = fasterrcnn_resnet50_fpn(pretrained=True)
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
        
        for epoch in range(num_epochs):
            # 訓練
            logging.info(f"開始 Epoch {epoch + 1}/{num_epochs} 訓練...")
            model.train()
            train_losses = []
            
            # 創建訓練進度條
            train_pbar = tqdm(
                train_loader, 
                desc=f"Fold {fold + 1}/{k_folds}, Epoch {epoch + 1}/{num_epochs} [訓練]",
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
                
                # 更新進度條顯示當前loss
                current_loss = losses.item()
                train_pbar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Avg_Loss': f'{np.mean(train_losses):.4f}'
                })
            
            train_pbar.close()
            scheduler.step()
            
            # 驗證
            logging.info(f"開始 Epoch {epoch + 1} 驗證...")
            
            # 創建驗證進度條
            val_pbar = tqdm(
                val_loader,
                desc=f"Fold {fold + 1}/{k_folds}, Epoch {epoch + 1}/{num_epochs} [驗證]",
                unit="batch",
                ncols=100
            )
            
            model.eval()
            all_predictions = []
            all_targets = []
            
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
                    
                    # 更新驗證進度條
                    val_pbar.set_postfix({
                        'Samples': f'{len(all_predictions)}'
                    })
            
            val_pbar.close()
            
            # 計算驗證指標
            val_metrics = calculate_detection_metrics(all_predictions, all_targets, iou_threshold=0.5)
            avg_train_loss = np.mean(train_losses)
            
            # 記錄到TensorBoard
            writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            writer.add_scalar('Metrics/Precision', val_metrics['precision'], epoch)
            writer.add_scalar('Metrics/Recall', val_metrics['recall'], epoch)
            writer.add_scalar('Metrics/F1', val_metrics['f1_score'], epoch)
            
            # 保存最佳模型
            if val_metrics['f1_score'] > best_f1:
                best_f1 = val_metrics['f1_score']
                torch.save(model.state_dict(), 
                          os.path.join(save_dir, f'best_model_fold_{fold + 1}.pth'))
            
            if epoch % 10 == 0:
                logging.info(f"Fold {fold + 1}, Epoch {epoch}/{num_epochs}: "
                           f"Loss: {avg_train_loss:.4f}, "
                           f"Precision: {val_metrics['precision']:.4f}, "
                           f"Recall: {val_metrics['recall']:.4f}, "
                           f"F1: {val_metrics['f1_score']:.4f}")
        
        fold_time = time.time() - fold_start_time
        
        # 最終評估
        model.load_state_dict(torch.load(os.path.join(save_dir, f'best_model_fold_{fold + 1}.pth')))
        final_metrics = evaluate_detection_model(model, val_loader, device)
        
        fold_result = {
            'fold': fold + 1,
            'metrics': final_metrics,
            'training_time': fold_time,
            'best_f1': best_f1
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
            'learning_rate': learning_rate
        }
    }
    
    with open(os.path.join(save_dir, 'kfold_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
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
    
    parser.add_argument('--data_dir', type=str, default=os.path.join(os.path.dirname(script_dir), 'datasets', 'splited_dataset'), 
                       help='數據集目錄路徑')
    parser.add_argument('--k_folds', type=int, default=2, 
                       help='K-fold交叉驗證的fold數量')
    parser.add_argument('--num_epochs', type=int, default=50, 
                       help='每個fold的訓練輪數')
    parser.add_argument('--batch_size', type=int, default=16, 
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.0001, 
                       help='學習率')
    parser.add_argument('--save_dir', type=str, default=os.path.join(script_dir, 'Faster_RCNN_Detection', 'models'), 
                       help='模型保存目錄')
    parser.add_argument('--log_dir', type=str, default=os.path.join(script_dir, 'Faster_RCNN_Detection', 'logs'), 
                       help='日誌保存目錄')
    
    args = parser.parse_args()
    
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
    logging.info(f"K-Fold數量: {args.k_folds}")
    logging.info(f"訓練輪數: {args.num_epochs}")
    logging.info(f"批次大小: {args.batch_size}")
    logging.info(f"學習率: {args.learning_rate}")
    logging.info(f"模型保存目錄: {args.save_dir}")
    logging.info(f"日誌目錄: {args.log_dir}")
    
    # 開始訓練
    start_time = time.time()
    results = train_kfold(
        data_dir=args.data_dir,
        k_folds=args.k_folds,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_dir=args.save_dir,
        log_dir=args.log_dir
    )
    
    total_time = time.time() - start_time
    logging.info(f"總執行時間: {total_time:.2f}秒 ({total_time/3600:.2f}小時)")


if __name__ == "__main__":
    main()
