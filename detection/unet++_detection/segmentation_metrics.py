#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UNet++ Segmentation Metrics
UNet++ 分割評估指標模組

該模組提供：
1. 分割任務的各種評估指標
2. IoU, Dice, Precision, Recall, F1-score 計算
3. 多類別和二元分割支持
4. 像素級和區域級指標
5. 統計報告生成

作者: GitHub Copilot
日期: 2025-09-19
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import logging
from sklearn.metrics import confusion_matrix, classification_report
from scipy import ndimage
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class SegmentationMetrics:
    """分割評估指標計算器"""
    
    def __init__(self, num_classes: int = 2, ignore_background: bool = True):
        """
        初始化指標計算器
        
        Args:
            num_classes: 類別數量
            ignore_background: 是否忽略背景類別
        """
        self.num_classes = num_classes
        self.ignore_background = ignore_background
        self.reset()
        
    def reset(self):
        """重置指標累積器"""
        self.total_samples = 0
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.pixel_accuracy_sum = 0.0
        self.mean_iou_sum = 0.0
        self.dice_scores = []
        self.hausdorff_distances = []
        
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        更新指標
        
        Args:
            predictions: 預測結果 (B, C, H, W) 或 (B, H, W)
            targets: 真實標籤 (B, H, W)
        """
        # 確保輸入是numpy數組
        if torch.is_tensor(predictions):
            if predictions.dim() == 4:  # (B, C, H, W)
                predictions = torch.argmax(predictions, dim=1)  # (B, H, W)
            predictions = predictions.cpu().numpy()
            
        if torch.is_tensor(targets):
            targets = targets.cpu().numpy()
        
        batch_size = predictions.shape[0]
        self.total_samples += batch_size
        
        for i in range(batch_size):
            pred_i = predictions[i].flatten()
            target_i = targets[i].flatten()
            
            # 更新混淆矩陣
            cm = confusion_matrix(target_i, pred_i, labels=list(range(self.num_classes)))
            self.confusion_matrix += cm
            
            # 計算像素準確率
            pixel_acc = np.sum(pred_i == target_i) / len(pred_i)
            self.pixel_accuracy_sum += pixel_acc
            
            # 計算IoU
            iou_scores = self._calculate_iou_per_class(pred_i, target_i)
            if self.ignore_background and len(iou_scores) > 1:
                mean_iou = np.mean(iou_scores[1:])  # 忽略背景類別
            else:
                mean_iou = np.mean(iou_scores)
            self.mean_iou_sum += mean_iou
            
            # 計算Dice分數
            dice_scores = self._calculate_dice_per_class(predictions[i], targets[i])
            self.dice_scores.append(dice_scores)
            
    def _calculate_iou_per_class(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """計算每個類別的IoU"""
        iou_scores = []
        
        for class_id in range(self.num_classes):
            pred_mask = (predictions == class_id)
            target_mask = (targets == class_id)
            
            intersection = np.sum(pred_mask & target_mask)
            union = np.sum(pred_mask | target_mask)
            
            if union == 0:
                iou = 1.0 if intersection == 0 else 0.0
            else:
                iou = intersection / union
                
            iou_scores.append(iou)
            
        return np.array(iou_scores)
        
    def _calculate_dice_per_class(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """計算每個類別的Dice分數"""
        dice_scores = []
        
        for class_id in range(self.num_classes):
            pred_mask = (predictions == class_id)
            target_mask = (targets == class_id)
            
            intersection = np.sum(pred_mask & target_mask)
            total = np.sum(pred_mask) + np.sum(target_mask)
            
            if total == 0:
                dice = 1.0 if intersection == 0 else 0.0
            else:
                dice = 2.0 * intersection / total
                
            dice_scores.append(dice)
            
        return np.array(dice_scores)
        
    def _calculate_hausdorff_distance(self, pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
        """計算Hausdorff距離"""
        try:
            from scipy.spatial.distance import directed_hausdorff
            
            # 提取邊界點
            pred_points = np.column_stack(np.where(pred_mask))
            target_points = np.column_stack(np.where(target_mask))
            
            if len(pred_points) == 0 or len(target_points) == 0:
                return float('inf')
                
            # 計算雙向Hausdorff距離
            h1 = directed_hausdorff(pred_points, target_points)[0]
            h2 = directed_hausdorff(target_points, pred_points)[0]
            
            return max(h1, h2)
            
        except ImportError:
            logger.warning("scipy 不可用，無法計算Hausdorff距離")
            return 0.0
        except Exception as e:
            logger.warning(f"計算Hausdorff距離時出錯: {e}")
            return 0.0
            
    def compute_metrics(self) -> Dict[str, float]:
        """計算最終指標"""
        if self.total_samples == 0:
            logger.warning("沒有樣本用於計算指標")
            return {}
            
        metrics = {}
        
        # 像素準確率
        metrics['pixel_accuracy'] = self.pixel_accuracy_sum / self.total_samples
        
        # 平均IoU
        metrics['mean_iou'] = self.mean_iou_sum / self.total_samples
        
        # 從混淆矩陣計算指標
        cm = self.confusion_matrix
        
        # 每個類別的精確率、召回率、F1分數
        class_metrics = {}
        for class_id in range(self.num_classes):
            tp = cm[class_id, class_id]
            fp = np.sum(cm[:, class_id]) - tp
            fn = np.sum(cm[class_id, :]) - tp
            tn = np.sum(cm) - tp - fp - fn
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            class_metrics[f'class_{class_id}_precision'] = precision
            class_metrics[f'class_{class_id}_recall'] = recall
            class_metrics[f'class_{class_id}_f1_score'] = f1_score
            class_metrics[f'class_{class_id}_iou'] = self._calculate_class_iou_from_cm(cm, class_id)
            
        metrics.update(class_metrics)
        
        # 宏平均指標
        if self.ignore_background and self.num_classes > 1:
            start_class = 1
        else:
            start_class = 0
            
        precisions = [class_metrics[f'class_{i}_precision'] for i in range(start_class, self.num_classes)]
        recalls = [class_metrics[f'class_{i}_recall'] for i in range(start_class, self.num_classes)]
        f1_scores = [class_metrics[f'class_{i}_f1_score'] for i in range(start_class, self.num_classes)]
        
        metrics['macro_precision'] = np.mean(precisions)
        metrics['macro_recall'] = np.mean(recalls)
        metrics['macro_f1_score'] = np.mean(f1_scores)
        
        # Dice分數統計
        if self.dice_scores:
            dice_array = np.array(self.dice_scores)
            for class_id in range(self.num_classes):
                metrics[f'class_{class_id}_dice'] = np.mean(dice_array[:, class_id])
                
            if self.ignore_background and self.num_classes > 1:
                metrics['mean_dice'] = np.mean(dice_array[:, 1:])
            else:
                metrics['mean_dice'] = np.mean(dice_array)
        
        return metrics
        
    def _calculate_class_iou_from_cm(self, cm: np.ndarray, class_id: int) -> float:
        """從混淆矩陣計算特定類別的IoU"""
        tp = cm[class_id, class_id]
        fp = np.sum(cm[:, class_id]) - tp
        fn = np.sum(cm[class_id, :]) - tp
        
        union = tp + fp + fn
        if union == 0:
            return 1.0 if tp == 0 else 0.0
        else:
            return tp / union
            
    def get_confusion_matrix(self) -> np.ndarray:
        """獲取混淆矩陣"""
        return self.confusion_matrix.copy()
        
    def plot_confusion_matrix(self, save_path: Optional[str] = None, 
                            class_names: Optional[List[str]] = None):
        """繪製混淆矩陣"""
        if class_names is None:
            class_names = [f'Class {i}' for i in range(self.num_classes)]
            
        # 正規化混淆矩陣
        cm_normalized = self.confusion_matrix.astype('float') / self.confusion_matrix.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Normalized Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"混淆矩陣已保存至: {save_path}")
        
        plt.show()


def calculate_advanced_segmentation_metrics(predictions: torch.Tensor, 
                                          targets: torch.Tensor,
                                          num_classes: int = 2) -> Dict[str, float]:
    """
    計算高級分割指標
    
    Args:
        predictions: 預測結果 (B, C, H, W)
        targets: 真實標籤 (B, H, W)
        num_classes: 類別數量
        
    Returns:
        指標字典
    """
    device = predictions.device
    batch_size = predictions.shape[0]
    
    # 轉換為概率
    if predictions.dim() == 4 and predictions.shape[1] > 1:
        pred_probs = F.softmax(predictions, dim=1)
        pred_labels = torch.argmax(pred_probs, dim=1)
    else:
        pred_labels = predictions.squeeze(1) if predictions.dim() == 4 else predictions
        pred_probs = predictions
    
    metrics = {}
    
    # 基本指標
    correct_pixels = (pred_labels == targets).float()
    metrics['pixel_accuracy'] = correct_pixels.mean().item()
    
    # 每個類別的指標
    class_metrics = {f'class_{i}': {'iou': 0.0, 'dice': 0.0, 'precision': 0.0, 'recall': 0.0} 
                    for i in range(num_classes)}
    
    total_iou = 0.0
    total_dice = 0.0
    
    for class_id in range(num_classes):
        class_pred = (pred_labels == class_id).float()
        class_target = (targets == class_id).float()
        
        # IoU
        intersection = (class_pred * class_target).sum(dim=[1, 2])
        union = (class_pred + class_target - class_pred * class_target).sum(dim=[1, 2])
        iou = (intersection / (union + 1e-8)).mean().item()
        
        # Dice
        dice = (2 * intersection / (class_pred.sum(dim=[1, 2]) + class_target.sum(dim=[1, 2]) + 1e-8)).mean().item()
        
        # Precision & Recall
        tp = intersection.sum().item()
        fp = (class_pred.sum() - intersection.sum()).item()
        fn = (class_target.sum() - intersection.sum()).item()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        
        class_metrics[f'class_{class_id}']['iou'] = iou
        class_metrics[f'class_{class_id}']['dice'] = dice
        class_metrics[f'class_{class_id}']['precision'] = precision
        class_metrics[f'class_{class_id}']['recall'] = recall
        
        total_iou += iou
        total_dice += dice
    
    # 平均指標
    metrics['mean_iou'] = total_iou / num_classes
    metrics['mean_dice'] = total_dice / num_classes
    
    # 如果忽略背景，重新計算平均指標
    if num_classes > 1:
        foreground_iou = sum(class_metrics[f'class_{i}']['iou'] for i in range(1, num_classes)) / (num_classes - 1)
        foreground_dice = sum(class_metrics[f'class_{i}']['dice'] for i in range(1, num_classes)) / (num_classes - 1)
        metrics['foreground_mean_iou'] = foreground_iou
        metrics['foreground_mean_dice'] = foreground_dice
    
    # 展平類別指標
    for class_id in range(num_classes):
        for metric_name, value in class_metrics[f'class_{class_id}'].items():
            metrics[f'class_{class_id}_{metric_name}'] = value
    
    return metrics


def evaluate_segmentation_model(model, dataloader, device, num_classes: int = 2,
                               save_dir: Optional[str] = None) -> Dict[str, float]:
    """
    評估分割模型
    
    Args:
        model: 訓練好的模型
        dataloader: 數據載入器
        device: 計算設備
        num_classes: 類別數量
        save_dir: 結果保存目錄
        
    Returns:
        評估指標字典
    """
    model.eval()
    
    metrics_calculator = SegmentationMetrics(num_classes=num_classes)
    all_predictions = []
    all_targets = []
    
    logger.info("開始評估分割模型...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, dict):
                images = batch['images'].to(device)
                targets = batch['targets']
                # 提取分割目標
                seg_targets = torch.stack([t['segmentation'] for t in targets]).to(device)
            else:
                images, seg_targets = batch
                images = images.to(device)
                seg_targets = seg_targets.to(device)
            
            # 前向傳播
            outputs = model(images)
            
            # 提取分割預測
            if isinstance(outputs, dict) and 'segmentation' in outputs:
                seg_predictions = outputs['segmentation']
            else:
                seg_predictions = outputs
            
            # 更新指標
            metrics_calculator.update(seg_predictions, seg_targets)
            
            # 保存預測結果用於可視化
            if len(all_predictions) < 10:  # 只保存前10個批次用於可視化
                pred_labels = torch.argmax(seg_predictions, dim=1)
                all_predictions.append(pred_labels.cpu())
                all_targets.append(seg_targets.cpu())
    
    # 計算最終指標
    final_metrics = metrics_calculator.compute_metrics()
    
    logger.info("模型評估完成")
    logger.info(f"像素準確率: {final_metrics.get('pixel_accuracy', 0):.4f}")
    logger.info(f"平均IoU: {final_metrics.get('mean_iou', 0):.4f}")
    logger.info(f"平均Dice: {final_metrics.get('mean_dice', 0):.4f}")
    
    # 保存結果
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存指標JSON
        with open(save_path / 'segmentation_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(final_metrics, f, ensure_ascii=False, indent=2)
        
        # 保存混淆矩陣
        metrics_calculator.plot_confusion_matrix(
            save_path / 'confusion_matrix.png',
            class_names=['Background', 'Lesion'] if num_classes == 2 else None
        )
        
        logger.info(f"評估結果已保存至: {save_dir}")
    
    return final_metrics


if __name__ == "__main__":
    # 測試代碼
    logging.basicConfig(level=logging.INFO)
    
    # 創建模擬數據
    batch_size, height, width = 4, 256, 256
    num_classes = 2
    
    # 模擬預測和目標
    predictions = torch.randn(batch_size, num_classes, height, width)
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    
    # 計算指標
    metrics = calculate_advanced_segmentation_metrics(predictions, targets, num_classes)
    
    print("分割指標:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # 測試指標計算器
    metrics_calculator = SegmentationMetrics(num_classes=num_classes)
    metrics_calculator.update(predictions, targets)
    final_metrics = metrics_calculator.compute_metrics()
    
    print("\n使用指標計算器:")
    for key, value in final_metrics.items():
        print(f"{key}: {value:.4f}")