#!/usr/bin/env python3
"""
UNet++ 肺結節分割訓練 - 訓練器模組
===================================

提供完整的訓練流程：
1. 訓練迴圈
2. 驗證評估
3. 多指標計算（ROI Dice, Lesion-wise）
4. Early Stopping
5. 模型保存與載入
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import json
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from skimage import measure

import sys

# 支援直接執行和模組執行
try:
    from .config import Config
    from .model import get_model, count_parameters
    from .losses import get_loss_function
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from train_unetpp.config import Config
    from train_unetpp.model import get_model, count_parameters
    from train_unetpp.losses import get_loss_function


logger = logging.getLogger(__name__)


def _convert_to_json_serializable(obj):
    """將 numpy 類型轉換為 JSON 可序列化的 Python 類型"""
    if isinstance(obj, dict):
        return {k: _convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_json_serializable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


class EarlyStopping:
    """Early Stopping 機制"""
    
    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 0.001,
        mode: str = "max"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class MetricsCalculator:
    """評估指標計算器"""
    
    def __init__(self, threshold: float = 0.5, target_threshold: float = 0.0):
        """
        Args:
            threshold: 預測輸出的二值化閾值
            target_threshold: 目標遮罩的二值化閾值（對於軟共識遮罩，使用 0 可捕獲任何標註）
        """
        self.threshold = threshold
        self.target_threshold = target_threshold
    
    def dice_score(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        smooth: float = 1e-6
    ) -> float:
        """計算 Dice Score"""
        pred_binary = (pred > self.threshold).astype(np.float32)
        target_binary = (target > self.target_threshold).astype(np.float32)
        
        intersection = (pred_binary * target_binary).sum()
        union = pred_binary.sum() + target_binary.sum()
        
        if union == 0:
            return 1.0 if target_binary.sum() == 0 else 0.0
        
        return (2 * intersection + smooth) / (union + smooth)
    
    def iou_score(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        smooth: float = 1e-6
    ) -> float:
        """計算 IoU (Jaccard Index)"""
        pred_binary = (pred > self.threshold).astype(np.float32)
        target_binary = (target > self.target_threshold).astype(np.float32)
        
        intersection = (pred_binary * target_binary).sum()
        union = pred_binary.sum() + target_binary.sum() - intersection
        
        if union == 0:
            return 1.0 if target_binary.sum() == 0 else 0.0
        
        return (intersection + smooth) / (union + smooth)
    
    def precision_recall(
        self,
        pred: np.ndarray,
        target: np.ndarray
    ) -> Tuple[float, float]:
        """計算 Precision 和 Recall"""
        pred_binary = (pred > self.threshold).astype(np.float32)
        target_binary = (target > self.target_threshold).astype(np.float32)
        
        tp = (pred_binary * target_binary).sum()
        fp = (pred_binary * (1 - target_binary)).sum()
        fn = ((1 - pred_binary) * target_binary).sum()
        
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        
        return precision, recall
    
    def lesion_wise_metrics(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        iou_threshold: float = 0.1
    ) -> Dict[str, float]:
        """
        計算結節級別指標
        
        Args:
            pred: 預測遮罩
            target: 目標遮罩
            iou_threshold: 判定為 TP 的 IoU 閾值
            
        Returns:
            結節級別指標
        """
        pred_binary = (pred > self.threshold).astype(np.int32)
        target_binary = (target > self.target_threshold).astype(np.int32)
        
        pred_labels = measure.label(pred_binary)
        target_labels = measure.label(target_binary)
        
        pred_regions = measure.regionprops(pred_labels)
        target_regions = measure.regionprops(target_labels)
        
        # 計算 TP, FP, FN
        tp = 0
        matched_targets = set()
        
        for pred_region in pred_regions:
            pred_mask = (pred_labels == pred_region.label)
            best_iou = 0
            best_target_id = None
            
            for target_region in target_regions:
                if target_region.label in matched_targets:
                    continue
                
                target_mask = (target_labels == target_region.label)
                intersection = (pred_mask & target_mask).sum()
                union = (pred_mask | target_mask).sum()
                iou = intersection / (union + 1e-6)
                
                if iou > best_iou:
                    best_iou = iou
                    best_target_id = target_region.label
            
            if best_iou >= iou_threshold and best_target_id is not None:
                tp += 1
                matched_targets.add(best_target_id)
        
        fp = len(pred_regions) - tp
        fn = len(target_regions) - len(matched_targets)
        
        sensitivity = tp / (tp + fn + 1e-6)
        precision = tp / (tp + fp + 1e-6)
        f1 = 2 * precision * sensitivity / (precision + sensitivity + 1e-6)
        
        return {
            'lesion_sensitivity': sensitivity,
            'lesion_precision': precision,
            'lesion_f1': f1,
            'fp_count': fp,
            'fn_count': fn,
            'tp_count': tp
        }


class UNetPPTrainer:
    """UNet++ 訓練器"""
    
    def __init__(
        self,
        config: Config,
        model: Optional[nn.Module] = None,
        device: Optional[str] = None,
        data_split: Optional[Dict] = None,
        output_dir: Optional[Path] = None
    ):
        """
        初始化訓練器
        
        Args:
            config: 配置物件
            model: 模型（若為 None 則根據配置創建）
            device: 設備
            data_split: 資料分割資訊 {'train_ids': [...], 'val_ids': [...], 'test_ids': [...]}
            output_dir: 輸出目錄（若為 None 則自動創建）
        """
        self.config = config
        self.device = device or config.device
        self.data_split = data_split or {}
        
        # 創建模型
        if model is None:
            self.model = get_model(config)
        else:
            self.model = model
        
        self.model = self.model.to(self.device)
        logger.info(f"模型參數數量: {count_parameters(self.model):,}")
        
        # 損失函數
        self.criterion = get_loss_function(config)
        
        # 優化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # 學習率調度器
        if config.training.scheduler == "cosine":
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=2,
                eta_min=config.training.min_lr
            )
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5
            )
        
        # Early Stopping
        self.early_stopping = EarlyStopping(
            patience=config.training.early_stopping_patience,
            min_delta=config.training.early_stopping_min_delta
        )
        
        # 指標計算器
        self.metrics_calc = MetricsCalculator(
            threshold=config.inference.prediction_threshold
        )
        
        # 混合精度訓練
        self.use_amp = config.training.use_amp
        self.scaler = GradScaler() if self.use_amp else None
        
        # 訓練記錄
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_dice': [],
            'val_iou': [],
            'val_precision': [],
            'val_recall': [],
            'val_lesion_f1': [],
            'val_lesion_sensitivity': [],
            'val_lesion_precision': [],
            'lr': []
        }
        
        # 最佳指標記錄
        self.best_metrics = {
            'val_dice': 0.0,
            'val_iou': 0.0,
            'val_precision': 0.0,
            'val_recall': 0.0,
            'val_lesion_f1': 0.0,
            'val_lesion_sensitivity': 0.0,
            'val_lesion_precision': 0.0,
            'epoch': 0
        }
        
        # 輸出目錄（使用傳入的或自動創建）
        if output_dir is not None:
            self.output_dir = Path(output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path(config.data.output_dir) / f"{config.experiment_name}_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        config.save(str(self.output_dir / "config.json"))
        
        # 保存 data split
        if self.data_split:
            self._save_data_split()
    
    def _save_data_split(self):
        """保存 data split 到 JSON"""
        split_path = self.output_dir / "data_split.json"
        split_data = {
            'train_ids': self.data_split.get('train_ids', []),
            'val_ids': self.data_split.get('val_ids', []),
            'test_ids': self.data_split.get('test_ids', []),
            'num_train': len(self.data_split.get('train_ids', [])),
            'num_val': len(self.data_split.get('val_ids', [])),
            'num_test': len(self.data_split.get('test_ids', []))
        }
        with open(split_path, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Data split saved: {split_path}")
    
    def _plot_training_curves(self, epoch: int):
        """繪製即時訓練曲線"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        epochs_range = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        ax = axes[0, 0]
        ax.plot(epochs_range, self.history['train_loss'], 'b-', label='Train Loss')
        ax.plot(epochs_range, self.history['val_loss'], 'r-', label='Val Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True)
        
        # Dice
        ax = axes[0, 1]
        ax.plot(epochs_range, self.history['val_dice'], 'g-', label='Val Dice')
        ax.axhline(y=self.best_metrics['val_dice'], color='r', linestyle='--', 
                   label=f"Best: {self.best_metrics['val_dice']:.4f}")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Dice Score')
        ax.set_title('Validation Dice Score')
        ax.legend()
        ax.grid(True)
        
        # IoU
        ax = axes[1, 0]
        ax.plot(epochs_range, self.history['val_iou'], 'purple', label='Val IoU')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('IoU')
        ax.set_title('Validation IoU')
        ax.legend()
        ax.grid(True)
        
        # Learning Rate
        ax = axes[1, 1]
        ax.plot(epochs_range, self.history['lr'], 'orange', label='LR')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.legend()
        ax.grid(True)
        ax.set_yscale('log')
        
        plt.suptitle(f'Training Progress - Epoch {epoch + 1}', fontsize=14)
        plt.tight_layout()
        
        save_path = self.output_dir / "training_curves.png"
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    def _save_best_metrics(self):
        """保存最佳指標"""
        metrics_path = self.output_dir / "best_metrics.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(_convert_to_json_serializable(self.best_metrics), f, indent=2, ensure_ascii=False)
        logger.info(f"Best metrics saved: {metrics_path}")

    def train_epoch(self, dataloader: DataLoader) -> float:
        """訓練一個 epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc="Training", leave=False)
        for batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader, epoch: int = None, save_samples: bool = True) -> Dict[str, float]:
        """驗證（含視覺化輸出）"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_preds = []
        all_targets = []
        all_images = []
        sample_patient_ids = []
        
        pbar = tqdm(dataloader, desc="Validating", leave=False)
        for batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
            
            total_loss += loss.item()
            num_batches += 1
            
            # 收集預測和目標
            preds = torch.sigmoid(outputs).cpu().numpy()
            targets = masks.cpu().numpy()
            imgs = images.cpu().numpy()
            
            all_preds.append(preds)
            all_targets.append(targets)
            all_images.append(imgs)
            sample_patient_ids.extend(batch['patient_id'])
        
        # 計算指標
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_images = np.concatenate(all_images, axis=0)
        
        # 像素級指標
        dice = self.metrics_calc.dice_score(all_preds, all_targets)
        iou = self.metrics_calc.iou_score(all_preds, all_targets)
        precision, recall = self.metrics_calc.precision_recall(all_preds, all_targets)
        
        # 結節級別指標（對每個樣本計算並平均）
        lesion_metrics = self.metrics_calc.lesion_wise_metrics(all_preds, all_targets)
        
        metrics = {
            'loss': total_loss / num_batches,
            'dice': dice,
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'lesion_f1': lesion_metrics['lesion_f1'],
            'lesion_sensitivity': lesion_metrics['lesion_sensitivity'],
            'lesion_precision': lesion_metrics['lesion_precision'],
            'tp_count': lesion_metrics['tp_count'],
            'fp_count': lesion_metrics['fp_count'],
            'fn_count': lesion_metrics['fn_count']
        }
        
        # 保存視覺化樣本
        if save_samples and epoch is not None:
            self._save_validation_samples(
                all_images, all_targets, all_preds, 
                sample_patient_ids, epoch, num_samples=4
            )
        
        return metrics
    
    def _save_validation_samples(
        self, 
        images: np.ndarray, 
        targets: np.ndarray, 
        preds: np.ndarray,
        patient_ids: list,
        epoch: int,
        num_samples: int = 4
    ):
        """保存驗證樣本視覺化"""
        import matplotlib.pyplot as plt
        
        # 確保輸出目錄存在
        vis_dir = self.output_dir / "validation_samples"
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # 選擇有正樣本的樣本優先
        positive_indices = [i for i in range(len(targets)) if targets[i].sum() > 0]
        if len(positive_indices) >= num_samples:
            selected_indices = positive_indices[:num_samples]
        else:
            selected_indices = list(range(min(num_samples, len(images))))
        
        fig, axes = plt.subplots(len(selected_indices), 4, figsize=(16, 4 * len(selected_indices)))
        if len(selected_indices) == 1:
            axes = axes.reshape(1, -1)
        
        for row, idx in enumerate(selected_indices):
            img = images[idx]
            gt = targets[idx]
            pred = preds[idx]
            patient_id = patient_ids[idx] if idx < len(patient_ids) else f"sample_{idx}"
            
            # 取中間通道或唯一通道作為顯示影像
            if img.ndim == 3:
                if img.shape[0] == 1:
                    display_img = img[0]  # 2D 模式：單通道
                elif img.shape[0] == 3:
                    display_img = img[1]  # 2.5D 模式：中間切片
                else:
                    display_img = img[0]
            else:
                display_img = img
            
            # GT 和 Pred 取第一個通道
            gt_2d = gt[0] if gt.ndim == 3 else gt
            pred_2d = pred[0] if pred.ndim == 3 else pred
            
            # 1. 原始影像
            axes[row, 0].imshow(display_img, cmap='gray')
            axes[row, 0].set_title(f'{patient_id}\nInput')
            axes[row, 0].axis('off')
            
            # 2. Ground Truth
            axes[row, 1].imshow(gt_2d, cmap='gray', vmin=0, vmax=1)
            axes[row, 1].set_title('Ground Truth')
            axes[row, 1].axis('off')
            
            # 3. Prediction
            axes[row, 2].imshow(pred_2d, cmap='gray', vmin=0, vmax=1)
            axes[row, 2].set_title(f'Prediction\n(max={pred_2d.max():.2f})')
            axes[row, 2].axis('off')
            
            # 4. Overlay (GT=Green, Pred=Red)
            overlay = np.stack([display_img] * 3, axis=-1)
            overlay = (overlay - overlay.min()) / (overlay.max() - overlay.min() + 1e-8)
            overlay[gt_2d > 0, 1] = 1.0  # GT = Green (使用 > 0 因為軟共識遮罩值可能 < 0.5)
            overlay[pred_2d > 0.5, 0] = 1.0  # Pred = Red
            axes[row, 3].imshow(overlay)
            axes[row, 3].set_title('Overlay\n(GT=Green, Pred=Red)')
            axes[row, 3].axis('off')
        
        plt.suptitle(f'Validation Samples - Epoch {epoch + 1}', fontsize=14)
        plt.tight_layout()
        
        save_path = vis_dir / f"epoch_{epoch + 1:03d}.png"
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Validation samples saved: {save_path}")
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Dict[str, List[float]]:
        """
        訓練模型
        
        Args:
            train_loader: 訓練資料載入器
            val_loader: 驗證資料載入器
            
        Returns:
            訓練歷史
        """
        best_dice = 0.0
        
        logger.info(f"開始訓練，共 {self.config.training.epochs} 個 epoch")
        logger.info(f"輸出目錄: {self.output_dir}")
        
        for epoch in range(self.config.training.epochs):
            epoch_start = time.time()
            
            # 訓練
            train_loss = self.train_epoch(train_loader)
            
            # 驗證（每 5 個 epoch 保存視覺化樣本）
            save_vis = (epoch % 5 == 0) or (epoch == self.config.training.epochs - 1)
            val_metrics = self.validate(val_loader, epoch=epoch, save_samples=save_vis)
            
            # 更新學習率
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metrics['dice'])
            else:
                self.scheduler.step()
            
            # 取得 step 後的真實 LR
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 記錄（包含 lesion-wise 指標）
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_dice'].append(val_metrics['dice'])
            self.history['val_iou'].append(val_metrics['iou'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['val_lesion_f1'].append(val_metrics['lesion_f1'])
            self.history['val_lesion_sensitivity'].append(val_metrics['lesion_sensitivity'])
            self.history['val_lesion_precision'].append(val_metrics['lesion_precision'])
            self.history['lr'].append(current_lr)
            
            # 保存最佳模型
            if val_metrics['dice'] > best_dice:
                best_dice = val_metrics['dice']
                # 更新最佳指標（包含 lesion-wise）
                self.best_metrics = {
                    'val_dice': val_metrics['dice'],
                    'val_iou': val_metrics['iou'],
                    'val_precision': val_metrics['precision'],
                    'val_recall': val_metrics['recall'],
                    'val_lesion_f1': val_metrics['lesion_f1'],
                    'val_lesion_sensitivity': val_metrics['lesion_sensitivity'],
                    'val_lesion_precision': val_metrics['lesion_precision'],
                    'val_loss': val_metrics['loss'],
                    'epoch': epoch + 1
                }
                self.save_checkpoint(
                    str(self.output_dir / "best_model.pth"),
                    epoch,
                    val_metrics
                )
                self._save_best_metrics()
            
            # Early Stopping
            if self.early_stopping(val_metrics['dice']):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
            
            # 日誌
            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch + 1}/{self.config.training.epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Dice: {val_metrics['dice']:.4f}, "
                f"Val IoU: {val_metrics['iou']:.4f}, "
                f"Lesion F1: {val_metrics['lesion_f1']:.4f}, "
                f"LR: {current_lr:.2e}, "
                f"Time: {epoch_time:.1f}s"
            )
            
            # 每 epoch 繪製訓練曲線
            self._plot_training_curves(epoch)
        
        # 保存最終模型
        self.save_checkpoint(
            str(self.output_dir / "final_model.pth"),
            epoch,
            val_metrics
        )
        
        # 保存訓練歷史
        with open(self.output_dir / "history.json", 'w', encoding='utf-8') as f:
            json.dump(_convert_to_json_serializable(self.history), f, indent=2)
        
        # 保存最佳指標
        self._save_best_metrics()
        
        logger.info(f"訓練完成！最佳 Dice: {best_dice:.4f} (Epoch {self.best_metrics['epoch']})")
        
        return self.history
    
    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        metrics: Dict[str, float]
    ):
        """保存檢查點"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': {
                'model': self.config.model.__dict__,
                'training': self.config.training.__dict__
            }
        }
        torch.save(checkpoint, path)
        logger.info(f"模型已保存: {path}")
    
    def load_checkpoint(self, path: str):
        """載入檢查點"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"模型已載入: {path}")
        logger.info(f"Epoch: {checkpoint['epoch']}, Metrics: {checkpoint['metrics']}")
        
        return checkpoint


if __name__ == "__main__":
    # 測試訓練器
    try:
        from .config import get_default_config
    except ImportError:
        from train_unetpp.config import get_default_config
    
    config = get_default_config()
    config.training.epochs = 2
    config.training.batch_size = 2
    
    trainer = UNetPPTrainer(config)
    
    print(f"Trainer initialized")
    print(f"Output dir: {trainer.output_dir}")
