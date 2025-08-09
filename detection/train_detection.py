"""
Optimized K-Fold Cross-Validation Training for Faster R-CNN Detection
Simplified version focusing only on K-fold cross-validation training.
"""

import os
import json
import time
import logging
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


def evaluate_model(model, val_loader, device, return_samples=False, max_samples=10):
    """評估模型"""
    model.eval()
    all_predictions = []
    all_targets = []
    sample_images = []
    sample_predictions = []
    sample_targets = []
    
    val_pbar = tqdm(val_loader, desc="評估模型", unit="batch", ncols=100, leave=False)
    
    with torch.no_grad():
        for images, targets in val_pbar:
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
            
            val_pbar.set_postfix({'Samples': f'{len(all_predictions)}'})
    
    val_pbar.close()
    
    # 計算指標
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


def create_kfold_datasets(data_dir, k_folds=5, random_seed=42):
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
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
    
    fold_datasets = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)
        fold_datasets.append((train_dataset, val_dataset))
        
        logging.info(f"Fold {fold + 1}: Training samples: {len(train_idx)}, Validation samples: {len(val_idx)}")
    
    return fold_datasets


def train_kfold(data_dir, k_folds=5, num_epochs=50, batch_size=8, learning_rate=0.001, 
                save_dir='./models', log_dir='./logs', random_seed=42, 
                accumulate_grad_batches=1, val_check_interval=5):
    """K-fold交叉驗證訓練 - 優化版本"""
    
    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"使用設備: {device}")
    
    # 創建保存目錄
    os.makedirs(save_dir, exist_ok=True)
    
    # 創建K-fold數據集
    fold_datasets = create_kfold_datasets(data_dir, k_folds, random_seed)
    
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
                val_metrics = evaluate_model(model, val_loader, device)
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
                               f"Recall: {val_metrics['recall']:.4f}, "
                               f"F1: {val_metrics['f1_score']:.4f}")
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
            model, val_loader, device, return_samples=True, max_samples=10
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
        }
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
    
    # 輸出最終結果
    logging.info(f"\n=== K-Fold 交叉驗證結果 ===")
    logging.info(f"平均精確度: {avg_metrics['precision']:.4f} ± {avg_metrics['precision_std']:.4f}")
    logging.info(f"平均召回率: {avg_metrics['recall']:.4f} ± {avg_metrics['recall_std']:.4f}")
    logging.info(f"平均F1分數: {avg_metrics['f1_score']:.4f} ± {avg_metrics['f1_score_std']:.4f}")
    logging.info(f"總訓練時間: {total_time:.2f}秒 ({total_time/3600:.2f}小時)")
    
    return results


def main():
    # 獲取腳本所在目錄
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 直接使用預設參數，不需要命令行解析
    class Args:
        def __init__(self):
            self.data_dir = os.path.join(os.path.dirname(script_dir), 'datasets', 'splited_dataset')
            self.k_folds = 5  # 減少fold數量以加快測試
            self.num_epochs = 10  # 減少epoch數量以加快測試
            self.batch_size = 8  # 減少批次大小以降低內存使用
            self.learning_rate = 0.0001
            self.accumulate_grad_batches = 2  # 增加梯度累積以補償較小的批次大小
            self.val_check_interval = 1  # 每個epoch都驗證以便觀察進度
            self.save_dir = os.path.join(script_dir, 'Faster_RCNN_Detection', 'models')
            self.log_dir = os.path.join(script_dir, 'Faster_RCNN_Detection', 'logs')
            self.random_seed = 42
    
    args = Args()
    
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
    logging.info(f"K-Fold數量: {args.k_folds}")
    logging.info(f"訓練輪數: {args.num_epochs}")
    logging.info(f"批次大小: {args.batch_size}")
    logging.info(f"學習率: {args.learning_rate}")
    logging.info(f"梯度累積批次數: {args.accumulate_grad_batches}")
    logging.info(f"驗證檢查間隔: {args.val_check_interval}")
    logging.info(f"模型保存目錄: {args.save_dir}")
    logging.info(f"日誌目錄: {args.log_dir}")
    logging.info(f"隨機種子: {args.random_seed}")
    
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
        val_check_interval=args.val_check_interval
    )
    
    total_time = time.time() - start_time
    logging.info(f"總執行時間: {total_time:.2f}秒 ({total_time/3600:.2f}小時)")


if __name__ == "__main__":
    main()
