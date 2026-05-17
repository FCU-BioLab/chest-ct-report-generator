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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from skimage import measure
from scipy.ndimage import binary_dilation

import sys

# 支援直接執行和模組執行
try:
    from .config import Config
    from .model import get_model, count_parameters
    from .losses import get_loss_function, BCEDiceLoss
    from .postprocess import PatchPostProcessor, PostProcessConfig
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from train_unetpp.config import Config
    from train_unetpp.model import get_model, count_parameters
    from train_unetpp.losses import get_loss_function, BCEDiceLoss
    from train_unetpp.postprocess import PatchPostProcessor, PostProcessConfig


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
        patience: int = 0,
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
    
    def __init__(self, threshold: float = 0.5, target_threshold: float = 0.0, min_area_px: int = 5):
        """
        Args:
            threshold: 預測輸出的二值化閾值
            target_threshold: 目標遮罩的二值化閾值（對於軟共識遮罩，使用 0 可捕獲任何標註）
            min_area_px: 2D lesion-wise 計算時，過濾掉面積小於此值的碎片（僅影響指標，不影響訓練）
        """
        self.threshold = threshold
        self.target_threshold = target_threshold
        self.min_area_px = min_area_px
    
    def dice_score(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        smooth: float = 1e-6
    ) -> float:
        """
        計算 Global Dice Score（聚合所有 pixels 後計算）
        
        Evaluation follows CSEA-Net paper.
        """
        pred_binary = (pred > self.threshold).astype(np.float32)
        target_binary = (target > self.target_threshold).astype(np.float32)
        
        # Global aggregation: sum all pixels first
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
    
    def _boundary_band(self, mask2d: np.ndarray, d: int) -> np.ndarray:
        """生成邊界帶（外側 ring）"""
        if d <= 0:
            return mask2d.astype(bool)
        dil = binary_dilation(mask2d, iterations=d)
        band = np.logical_xor(dil, mask2d)
        return band
    
    def boundary_iou(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        d: int = 2
    ) -> float:
        """
        計算 Boundary IoU（CSEA-Net 論文版本）
        
        Args:
            pred: 預測遮罩 (N, C, H, W) 或 (N, H, W) 或 (H, W)
            target: 目標遮罩
            d: 邊界寬度（膨脹迭代次數）
            
        Returns:
            平均 Boundary IoU
        """
        pred_b = (pred > self.threshold)
        tar_b = (target > self.target_threshold)
        
        # squeeze 到 (N, H, W)
        if pred_b.ndim == 4:
            pred_b = pred_b[:, 0]
            tar_b = tar_b[:, 0]
        if pred_b.ndim == 2:
            pred_b = pred_b[None, ...]
            tar_b = tar_b[None, ...]
        
        ious = []
        for i in range(pred_b.shape[0]):
            bp = self._boundary_band(pred_b[i], d)
            bt = self._boundary_band(tar_b[i], d)
            inter = np.logical_and(bp, bt).sum()
            union = np.logical_or(bp, bt).sum()
            if union == 0:
                ious.append(1.0)  # 都沒有邊界視為完美
            else:
                ious.append(inter / (union + 1e-6))
        return float(np.mean(ious))
    
    def lesion_wise_metrics(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        iou_threshold: float = 0.1
    ) -> Dict[str, float]:
        """
        計算結節級別指標（對所有 2D 切片聚合）
        
        Args:
            pred: 預測遮罩 (N, C, H, W) 或 (N, H, W) 或 (H, W)
            target: 目標遮罩
            iou_threshold: 判定為 TP 的 IoU 閾值
            
        Returns:
            結節級別指標
        """
        pred_binary = (pred > self.threshold).astype(np.int32)
        target_binary = (target > self.target_threshold).astype(np.int32)
        
        # 處理不同維度：擠壓到 2D 或 3D
        if pred_binary.ndim == 4:
            # (N, C, H, W) -> (N, H, W) 取第一個 channel
            pred_binary = pred_binary[:, 0, :, :]
            target_binary = target_binary[:, 0, :, :]
        elif pred_binary.ndim == 3 and pred_binary.shape[0] == 1:
            # (1, H, W) -> (H, W)
            pred_binary = pred_binary[0]
            target_binary = target_binary[0]
        
        # 對每個切片計算 lesion metrics 並聚合
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        if pred_binary.ndim == 3:
            # (N, H, W) - 多個切片
            for i in range(pred_binary.shape[0]):
                slice_result = self._lesion_metrics_2d(
                    pred_binary[i], target_binary[i], iou_threshold
                )
                total_tp += slice_result['tp']
                total_fp += slice_result['fp']
                total_fn += slice_result['fn']
        else:
            # (H, W) - 單個切片
            slice_result = self._lesion_metrics_2d(
                pred_binary, target_binary, iou_threshold
            )
            total_tp = slice_result['tp']
            total_fp = slice_result['fp']
            total_fn = slice_result['fn']
        
        sensitivity = total_tp / (total_tp + total_fn + 1e-6)
        precision = total_tp / (total_tp + total_fp + 1e-6)
        f1 = 2 * precision * sensitivity / (precision + sensitivity + 1e-6)
        
        return {
            'lesion_sensitivity': sensitivity,
            'lesion_precision': precision,
            'lesion_f1': f1,
            'fp_count': total_fp,
            'fn_count': total_fn,
            'tp_count': total_tp,
            'lesion_unit': 'per_slice'
        }
    
    def _lesion_metrics_2d(
        self,
        pred_2d: np.ndarray,
        target_2d: np.ndarray,
        iou_threshold: float
    ) -> Dict[str, int]:
        """計算單一 2D 切片的 lesion metrics（會過濾掉面積 < min_area_px 的碎片）"""
        pred_labels = measure.label(pred_2d)
        target_labels = measure.label(target_2d)
        
        pred_regions = measure.regionprops(pred_labels)
        target_regions = measure.regionprops(target_labels)
        
        # 過濾掉面積太小的 pred regions（減少 FP 碎片）
        pred_regions = [r for r in pred_regions if r.area >= self.min_area_px]
        
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
        
        return {'tp': tp, 'fp': fp, 'fn': fn}


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
        
        # 訓練損失函數（AdaptiveLoss 用於 training）
        self.criterion = get_loss_function(config)
        
        # 驗證損失函數（BCE+Dice 用於 val/test 評估）
        self.val_criterion = BCEDiceLoss(dice_weight=config.training.dice_weight, bce_weight=1.0)
        
        # 優化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # 學習率調度器
        if config.training.scheduler == "cosine":
            # 使用 CosineAnnealingLR（不 restart），避免震盪
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.training.epochs,
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
            threshold=config.inference.prediction_threshold,
            target_threshold=config.inference.target_threshold,
            min_area_px=config.inference.min_area_px
        )
        
        # 後處理器（用於驗證時計算後處理後的 metrics）
        pp_config = PostProcessConfig(
            threshold=config.inference.prediction_threshold,
            use_lung_mask=False,  # patch 級別無需 lung mask
            min_size_mm3=config.inference.patch_min_size_mm3,
            max_size_mm3=config.inference.patch_max_size_mm3,
            closing_radius_mm=config.inference.patch_closing_radius_mm,
            fill_holes=config.inference.patch_fill_holes,
            fill_holes_3d=False
        )
        # 從 config 讀取 patch spacing
        patch_spacing = config.inference.patch_spacing_mm
        self.postprocessor = PatchPostProcessor(pp_config, spacing=np.array([patch_spacing, patch_spacing]))
        
        # 混合精度訓練
        self.use_amp = config.training.use_amp
        self.scaler = GradScaler(device='cuda') if self.use_amp else None
        
        # 訓練記錄
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_dice': [],
            'val_iou': [],
            'val_precision': [],
            'val_recall': [],
            # Post-processed metrics
            'val_pp_dice': [],
            'val_pp_iou': [],
            'val_pp_precision': [],
            'val_pp_recall': [],
            'lr': []
        }
        
        # 最佳指標記錄
        self.best_metrics = {
            'val_dice': 0.0,
            'val_iou': 0.0,
            'val_precision': 0.0,
            'val_recall': 0.0,
            # Post-processed
            'val_pp_dice': 0.0,
            'val_pp_iou': 0.0,
            'val_pp_precision': 0.0,
            'val_pp_recall': 0.0,
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

    def _compute_loss(self, outputs, targets, criterion):
        """計算損失（支援 Deep Supervision）"""
        if isinstance(outputs, (list, tuple)):
            loss = 0
            # Weighted sum for deep supervision
            # Weights: [1.0, 0.5, 0.25, 0.125] or just average?
            # MrGiovanni usually sums them up. Average keeps scale similar.
            for output in outputs:
                loss += criterion(output, targets)
            loss /= len(outputs)
            return loss
        else:
            return criterion(outputs, targets)

    def train_epoch(self, dataloader: DataLoader, log_first_batch: bool = False) -> float:
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
                with autocast('cuda'):
                    outputs = self.model(images)
                    loss = self._compute_loss(outputs, masks, self.criterion)

                if not torch.isfinite(loss):
                    logger.warning("Skipping batch with non-finite loss: %s", loss.item())
                    continue
                
                self.scaler.scale(loss).backward()
                
                # Gradient Clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self._compute_loss(outputs, masks, self.criterion)

                if not torch.isfinite(loss):
                    logger.warning("Skipping batch with non-finite loss: %s", loss.item())
                    continue

                loss.backward()
                
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip)
                
                self.optimizer.step()
            
            # === Debug 記錄：首批次 shape ===
            if log_first_batch and num_batches == 0:
                out_shape = outputs[0].shape if isinstance(outputs, (list, tuple)) else outputs.shape
                logger.info(f"[DEBUG] First batch - input shape: {images.shape}, output shape: {out_shape} (List len: {len(outputs) if isinstance(outputs, (list, tuple)) else 0}), mask shape: {masks.shape}")
                logger.info(f"[DEBUG] Input channels: {images.shape[1]} ({'2D' if images.shape[1] == 1 else '2.5D' if images.shape[1] == 3 else 'other'})")
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches if num_batches > 0 else float("nan")
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader, epoch: int = None, save_samples: bool = True) -> Dict[str, float]:
        """
        驗證（patch-level 評估，與 train 相同格式）
        """
        self.model.eval()
        
        all_preds = []
        all_targets = []
        all_images = []
        sample_info = []
        
        # 累積 val loss
        val_loss_sum = 0.0
        val_loss_n = 0
        
        pbar = tqdm(dataloader, desc="Validating", leave=False)
        for batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            if self.use_amp:
                with autocast('cuda'):
                    outputs = self.model(images)
                    # For validation, use only the final output (first element in our NestedUNet list)
                    val_output = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
                    loss = self.val_criterion(val_output, masks)
            else:
                outputs = self.model(images)
                val_output = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
                loss = self.val_criterion(val_output, masks)
            
            val_loss_sum += loss.item() * images.shape[0]
            val_loss_n += images.shape[0]
            
            # 收集預測結果
            preds = torch.sigmoid(val_output).cpu().numpy()
            targets = masks.cpu().numpy()
            # 動態計算中心 channel（支援 3 或 5 channel）
            c_mid = images.shape[1] // 2
            imgs = images[:, c_mid, :, :].cpu().numpy()
            
            for i in range(images.shape[0]):
                all_preds.append(preds[i])
                all_targets.append(targets[i])
                all_images.append(imgs[i])
                
                # 支援 patient_id (LNDb) 或 case_id (MSD)
                id_key = 'patient_id' if 'patient_id' in batch else 'case_id'
                sample_info.append({
                    'id': batch[id_key][i] if isinstance(batch[id_key], list) else str(batch[id_key][i]),
                    'slice_idx': batch['slice_idx'][i] if isinstance(batch['slice_idx'], list) else int(batch['slice_idx'][i]),
                    'patch_idx': int(batch['patch_idx'][i]) if 'patch_idx' in batch else -1
                })
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # === 計算 Metrics ===
        total_intersection = 0.0
        total_pred_sum = 0.0
        total_target_sum = 0.0
        total_union = 0.0
        total_patches = len(all_preds)
        
        for pred, target in zip(all_preds, all_targets):
            pred_binary = (pred > self.metrics_calc.threshold).astype(np.float32)
            target_binary = (target > self.metrics_calc.target_threshold).astype(np.float32)
            
            intersection = (pred_binary * target_binary).sum()
            total_intersection += intersection
            total_pred_sum += pred_binary.sum()
            total_target_sum += target_binary.sum()
            total_union += pred_binary.sum() + target_binary.sum() - intersection
        
        # Pixel-level Segmentation Metrics (raw threshold)
        smooth = 1e-6
        dice = (2 * total_intersection + smooth) / (total_pred_sum + total_target_sum + smooth)
        iou = (total_intersection + smooth) / (total_union + smooth)
        precision = total_intersection / (total_pred_sum + smooth)
        recall = total_intersection / (total_target_sum + smooth)
        
        # === Post-processed Metrics ===
        # 後處理流程: threshold → CC filter → closing → fill holes
        pp_intersection = 0.0
        pp_pred_sum = 0.0
        pp_target_sum = 0.0
        pp_union = 0.0
        
        for pred, target in zip(all_preds, all_targets):
            # 後處理預測結果
            pred_2d = pred[0] if len(pred.shape) == 3 else pred
            pp_pred = self.postprocessor.process_patch(pred_2d).astype(np.float32)
            
            target_binary = (target > self.metrics_calc.target_threshold).astype(np.float32)
            target_2d = target_binary[0] if len(target_binary.shape) == 3 else target_binary
            
            intersection = (pp_pred * target_2d).sum()
            pp_intersection += intersection
            pp_pred_sum += pp_pred.sum()
            pp_target_sum += target_2d.sum()
            pp_union += pp_pred.sum() + target_2d.sum() - intersection
        
        pp_dice = (2 * pp_intersection + smooth) / (pp_pred_sum + pp_target_sum + smooth)
        pp_iou = (pp_intersection + smooth) / (pp_union + smooth)
        pp_precision = pp_intersection / (pp_pred_sum + smooth)
        pp_recall = pp_intersection / (pp_target_sum + smooth)
        
        # === Positive-only Metrics ===
        # 只在有結節的 patch 上計算，避免被大量負樣本稀釋
        pos_intersection = 0.0
        pos_pred_sum = 0.0
        pos_target_sum = 0.0
        pos_union = 0.0
        pos_patch_count = 0
        
        for pred, target in zip(all_preds, all_targets):
            target_binary = (target > self.metrics_calc.target_threshold).astype(np.float32)
            # 只計算 GT > 0 的 patch
            if target_binary.sum() > 0:
                pos_patch_count += 1
                pred_binary = (pred > self.metrics_calc.threshold).astype(np.float32)
                intersection = (pred_binary * target_binary).sum()
                pos_intersection += intersection
                pos_pred_sum += pred_binary.sum()
                pos_target_sum += target_binary.sum()
                pos_union += pred_binary.sum() + target_binary.sum() - intersection
        
        if pos_patch_count > 0:
            dice_pos = (2 * pos_intersection + smooth) / (pos_pred_sum + pos_target_sum + smooth)
            iou_pos = (pos_intersection + smooth) / (pos_union + smooth)
            recall_pos = pos_intersection / (pos_target_sum + smooth)
        else:
            dice_pos = 0.0
            iou_pos = 0.0
            recall_pos = 0.0
        
        metrics = {
            'loss': val_loss_sum / max(val_loss_n, 1),
            # Raw metrics (threshold only)
            'dice': dice,
            'iou': iou,
            'precision': precision,
            'recall': recall,
            # Positive-only metrics (更貼近結節分割)
            'dice_pos': dice_pos,
            'iou_pos': iou_pos,
            'recall_pos': recall_pos,
            'pos_patch_count': pos_patch_count,
            # Post-processed metrics
            'pp_dice': pp_dice,
            'pp_iou': pp_iou,
            'pp_precision': pp_precision,
            'pp_recall': pp_recall,
            # Stats
            'total_patches': total_patches,
            'avg_pred_area': float(total_pred_sum / total_patches) if total_patches > 0 else 0,
            'avg_gt_area': float(total_target_sum / total_patches) if total_patches > 0 else 0
        }
        
        # 保存視覺化樣本
        if save_samples and epoch is not None and len(all_preds) > 0:
            # 選擇有 GT 的樣本
            positive_indices = [i for i, t in enumerate(all_targets) if t.sum() > 0]
            if len(positive_indices) >= 4:
                selected = positive_indices[:4]
            else:
                selected = list(range(min(4, len(all_preds))))
            
            self._save_validation_samples(
                [all_images[i] for i in selected],
                [all_targets[i] for i in selected],
                [all_preds[i] for i in selected],
                [sample_info[i]['id'] for i in selected],
                epoch
            )
        
        return metrics
    
    def _save_validation_samples(self, images, targets, preds, patient_ids, epoch):
        """視覺化保存"""
        import matplotlib.pyplot as plt
        
        vis_dir = self.output_dir / "validation_samples"
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        n_samples = min(4, len(images))
        if n_samples == 0:
            return
        
        fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_samples):
            img = images[i]  # (H, W) - patch
            target = targets[i][0] if len(targets[i].shape) == 3 else targets[i]
            pred = preds[i][0] if len(preds[i].shape) == 3 else preds[i]
            pred_binary = (pred > 0.5).astype(np.float32)
            
            # Image (patch)
            axes[i, 0].imshow(img, cmap='gray')
            axes[i, 0].set_title(f'{patient_ids[i]} - Patch')
            axes[i, 0].axis('off')
            
            # GT (full slice)
            axes[i, 1].imshow(target, cmap='Reds', vmin=0, vmax=1)
            axes[i, 1].set_title(f'GT (area={target.sum():.0f})')
            axes[i, 1].axis('off')
            
            # Pred (full slice)
            axes[i, 2].imshow(pred, cmap='Blues', vmin=0, vmax=1)
            axes[i, 2].set_title(f'Pred (area={pred_binary.sum():.0f})')
            axes[i, 2].axis('off')
            
            # Overlay
            overlay = np.zeros((*target.shape, 3))
            overlay[:, :, 0] = target  # GT 紅色
            overlay[:, :, 2] = pred_binary  # Pred 藍色
            axes[i, 3].imshow(np.clip(overlay, 0, 1))
            axes[i, 3].set_title('Overlay (R=GT, B=Pred)')
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig(vis_dir / f'epoch_{epoch:03d}.png', dpi=100)
        plt.close()
    
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
        first_batch_logged = False
        
        logger.info(f"開始訓練，共 {self.config.training.epochs} 個 epoch")
        logger.info(f"輸出目錄: {self.output_dir}")
        
        # === Debug 記錄：閾值設定 ===
        logger.info(f"[DEBUG] MetricsCalculator - pred_threshold: {self.metrics_calc.threshold}, target_threshold: {self.metrics_calc.target_threshold}, min_area_px: {self.metrics_calc.min_area_px}")
        logger.info("[CSEA-Net Alignment] Evaluation: Binary GT (threshold=0.5), Global Dice, Boundary IoU (d=2)")
        
        # === Sanity Check: Patch-level 驗證 ===
        logger.info(f"[Patch-level] Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
        val_first_batch = next(iter(val_loader))
        if 'image' in val_first_batch and 'mask' in val_first_batch:
            logger.info(f"[Patch-level Sanity] Val batch OK: image={val_first_batch['image'].shape}, mask={val_first_batch['mask'].shape}")
        else:
            logger.error(f"[Patch-level Sanity] Unexpected val batch keys: {list(val_first_batch.keys())}")
        
        for epoch in range(self.config.training.epochs):
            epoch_start = time.time()
            
            # 訓練（第一個 epoch 記錄首批次 shape）
            train_loss = self.train_epoch(train_loader, log_first_batch=(epoch == 0))
            
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
            
            # 記錄
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_dice'].append(val_metrics['dice'])
            self.history['val_iou'].append(val_metrics['iou'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            # Post-processed metrics
            self.history['val_pp_dice'].append(val_metrics['pp_dice'])
            self.history['val_pp_iou'].append(val_metrics['pp_iou'])
            self.history['val_pp_precision'].append(val_metrics['pp_precision'])
            self.history['val_pp_recall'].append(val_metrics['pp_recall'])
            self.history['lr'].append(current_lr)
            
            # 保存最佳模型（使用後處理後的 Dice 作為 model selection 指標）
            pp_dice = val_metrics['pp_dice']
            if pp_dice > best_dice:
                best_dice = pp_dice
                # 更新最佳指標
                self.best_metrics = {
                    'val_dice': val_metrics['dice'],
                    'val_iou': val_metrics['iou'],
                    'val_precision': val_metrics['precision'],
                    'val_recall': val_metrics['recall'],
                    'val_pp_dice': val_metrics['pp_dice'],
                    'val_pp_iou': val_metrics['pp_iou'],
                    'val_pp_precision': val_metrics['pp_precision'],
                    'val_pp_recall': val_metrics['pp_recall'],
                    'val_loss': val_metrics['loss'],
                    'epoch': epoch + 1
                }
                self.save_checkpoint(
                    str(self.output_dir / "best_model.pth"),
                    epoch,
                    val_metrics
                )
                self._save_best_metrics()
                logger.info(f"New best model saved! PP Dice: {pp_dice:.4f} (Raw Dice: {val_metrics['dice']:.4f})")
            
            # Early Stopping (使用後處理後的 Dice)
            if self.early_stopping(pp_dice):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
            
            # 日誌
            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch + 1}/{self.config.training.epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}, "
                f"LR: {current_lr:.2e}, Time: {epoch_time:.1f}s"
            )
            logger.info(
                f"  Raw  - Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}, "
                f"Prec: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}"
            )
            logger.info(
                f"  PP   - Dice: {val_metrics['pp_dice']:.4f}, IoU: {val_metrics['pp_iou']:.4f}, "
                f"Prec: {val_metrics['pp_precision']:.4f}, Recall: {val_metrics['pp_recall']:.4f}"
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
        
        logger.info(f"訓練完成！最佳 PP Dice: {best_dice:.4f} (Epoch {self.best_metrics['epoch']})")
        
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
