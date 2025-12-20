#!/usr/bin/env python3
"""
UNet++ 訓練器模組
提供 UNet++ 模型的訓練與評估功能
"""

import logging
from pathlib import Path
from typing import Dict, Optional, List
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR, ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt

from .losses import CombinedLoss, DeepSupervisionLoss, get_loss_function
from .utils import compute_all_metrics, EarlyStopping, PatientMetricsTracker


class SegmentationVisualizer:
    """
    分割結果可視化工具
    """
    
    def __init__(self, output_dir: str, dpi: int = 150):
        self.output_dir = Path(output_dir)
        self.dpi = dpi
        self.vis_dir = self.output_dir / "visualizations"
        self.vis_dir.mkdir(parents=True, exist_ok=True)
    
    def save_slice_comparison(
        self,
        image: np.ndarray,
        gt_mask: np.ndarray,
        pred_mask: np.ndarray,
        patient_id: str,
        slice_idx: int,
        dice_score: float = None,
        iou_score: float = None
    ):
        """保存切片對比圖"""
        import matplotlib
        matplotlib.use('Agg')
        
        safe_patient_id = str(patient_id).replace('.', '_').replace('/', '_')[:50]
        patient_vis_dir = self.vis_dir / safe_patient_id
        patient_vis_dir.mkdir(parents=True, exist_ok=True)
        
        # 處理影像格式
        if len(image.shape) == 3:
            if image.shape[0] == 3:
                image = image[0]
            elif image.shape[2] == 3:
                image = image[:, :, 0]
        
        gt_binary = (gt_mask > 0.5).astype(np.float32)
        pred_binary = (pred_mask > 0.5).astype(np.float32)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原始影像
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title(f'CT Image\nPatient: {str(patient_id)[:20]}...')
        axes[0].axis('off')
        
        # Ground Truth
        axes[1].imshow(image, cmap='gray')
        gt_overlay = np.zeros((*gt_binary.shape, 4))
        gt_overlay[gt_binary > 0] = [0, 1, 0, 0.4]
        axes[1].imshow(gt_overlay)
        axes[1].contour(gt_binary, levels=[0.5], colors=['lime'], linewidths=2)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Prediction
        axes[2].imshow(image, cmap='gray')
        pred_overlay = np.zeros((*pred_binary.shape, 4))
        pred_overlay[pred_binary > 0] = [1, 0, 0, 0.4]
        axes[2].imshow(pred_overlay)
        axes[2].contour(pred_binary, levels=[0.5], colors=['red'], linewidths=2)
        title = 'Prediction'
        if dice_score is not None:
            title += f'\nDice: {dice_score:.4f}'
        if iou_score is not None:
            title += f', IoU: {iou_score:.4f}'
        axes[2].set_title(title)
        axes[2].axis('off')
        
        plt.tight_layout()
        save_path = patient_vis_dir / f"slice_{slice_idx:04d}_comparison.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)


class UNetPPTrainer:
    """
    UNet++ 訓練器
    
    提供完整的訓練、驗證和評估功能
    
    Args:
        model: UNet++ 模型
        train_loader: 訓練資料載入器
        val_loader: 驗證資料載入器
        test_loader: 測試資料載入器 (可選)
        criterion: 損失函數
        optimizer: 優化器
        scheduler: 學習率調度器
        device: 計算設備
        output_dir: 輸出目錄
        use_amp: 是否使用混合精度訓練
        accumulation_steps: 梯度累積步數
        deep_supervision: 是否使用深度監督
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        criterion: nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler = None,
        device: torch.device = None,
        output_dir: str = "result",
        use_amp: bool = True,
        accumulation_steps: int = 1,
        deep_supervision: bool = True,
        save_visualizations: bool = True,
        empty_gt_handling: str = 'skip',  # 'skip', 'dice_1', 'dice_0'
        threshold: float = 0.5  # Binary threshold for predictions
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion if criterion else CombinedLoss()
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_amp = use_amp and torch.cuda.is_available()
        self.accumulation_steps = accumulation_steps
        self.deep_supervision = deep_supervision
        self.save_visualizations = save_visualizations
        self.threshold = threshold
        
        # Empty GT handling strategy
        assert empty_gt_handling in ('skip', 'dice_1', 'dice_0'), \
            f"empty_gt_handling must be 'skip', 'dice_1', or 'dice_0', got: {empty_gt_handling}"
        self.empty_gt_handling = empty_gt_handling
        
        # 設定優化器
        if optimizer is None:
            self.optimizer = AdamW(
                model.parameters(), 
                lr=1e-4, 
                weight_decay=1e-4
            )
        else:
            self.optimizer = optimizer
        
        # 設定調度器
        self.scheduler = scheduler
        
        # 混合精度 - 使用新 API (torch.amp.GradScaler)
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
        
        # 日誌
        self.logger = logging.getLogger(__name__)
        
        # 指標追蹤
        self.train_losses = []
        self.val_losses = []
        self.val_dices = []
        self.best_dice = 0.0
        self.best_epoch = 0
        
        # 可視化工具
        if save_visualizations:
            self.visualizer = SegmentationVisualizer(str(self.output_dir))
        else:
            self.visualizer = None
        
        # 患者指標追蹤 (only for positive GT samples)
        self.patient_tracker = PatientMetricsTracker()
        
        # 深度監督損失
        if deep_supervision:
            self.ds_criterion = DeepSupervisionLoss(base_loss=self.criterion)
        
        # 移動模型到設備
        self.model.to(self.device)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train one epoch with pos-GT-only Dice calculation.
        
        Metrics:
        - dice: Average Dice over positive-GT samples only
        - loss: Average loss over all batches
        - positive_gt_samples: Number of samples with lesions
        - pred_positive_ratio: Ratio of samples where model predicts any positive
        - empty_fp_rate: FP rate on empty GT samples
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Per-epoch counters (reset at epoch start)
        total_samples = 0
        positive_gt_samples = 0
        empty_gt_samples = 0
        pred_positive_count = 0  # Samples with any positive prediction
        empty_fp_count = 0       # Empty GT samples with FP
        empty_fp_pixels = 0      # Total FP pixels on empty GT
        total_dice_sum = 0.0     # Sum of per-sample Dice (pos-GT only)
        total_pixel_pos_ratio = 0.0
        num_batches_for_pixel = 0
        
        self.optimizer.zero_grad()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward pass with AMP (using new API)
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    if self.deep_supervision and hasattr(self.model, 'get_deep_supervision_outputs'):
                        outputs = self.model.get_deep_supervision_outputs(images)
                        loss = self.ds_criterion(outputs, masks)
                        pred = outputs[-1]
                    else:
                        pred = self.model(images)
                        loss = self.criterion(pred, masks)
                    
                    loss = loss / self.accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                if self.deep_supervision and hasattr(self.model, 'get_deep_supervision_outputs'):
                    outputs = self.model.get_deep_supervision_outputs(images)
                    loss = self.ds_criterion(outputs, masks)
                    pred = outputs[-1]
                else:
                    pred = self.model(images)
                    loss = self.criterion(pred, masks)
                
                loss = loss / self.accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Metrics calculation (optimized: batch-level where possible)
            with torch.no_grad():
                pred_sigmoid = torch.sigmoid(pred)
                pred_binary = (pred_sigmoid > self.threshold).float()
                masks_binary = (masks > 0.5).float()
                
                # Batch-level pixel positive ratio
                batch_pixel_pos = pred_binary.mean().item()
                total_pixel_pos_ratio += batch_pixel_pos
                num_batches_for_pixel += 1
                
                # Per-sample metrics (compute gt_sum/pred_sum on GPU)
                B = masks.shape[0]
                gt_sums = masks_binary.view(B, -1).sum(dim=1)   # [B]
                pred_sums = pred_binary.view(B, -1).sum(dim=1)  # [B]
                
                batch_dice_sum = 0.0
                batch_pos_gt_count = 0
                
                for i in range(B):
                    total_samples += 1
                    gt_sum = gt_sums[i].item()
                    pred_sum = pred_sums[i].item()
                    
                    if pred_sum > 0:
                        pred_positive_count += 1
                    
                    if gt_sum == 0:
                        # Empty GT sample
                        empty_gt_samples += 1
                        if pred_sum > 0:
                            empty_fp_count += 1
                            empty_fp_pixels += pred_sum
                        # Skip for Dice calculation (pos-GT-only)
                        continue
                    else:
                        # Positive GT sample - compute Dice
                        positive_gt_samples += 1
                        batch_pos_gt_count += 1
                        intersection = (pred_binary[i] * masks_binary[i]).sum().item()
                        dice = (2.0 * intersection + 1e-6) / (pred_sum + gt_sum + 1e-6)
                        batch_dice_sum += dice
                        total_dice_sum += dice
                
                # Batch-level average Dice (pos-GT only)
                batch_dice = batch_dice_sum / max(batch_pos_gt_count, 1)
            
            total_loss += loss.item() * self.accumulation_steps
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item() * self.accumulation_steps:.4f}',
                'dice(posGT)': f'{batch_dice:.4f}'
            })
        
        # Epoch averages
        avg_loss = total_loss / max(num_batches, 1)
        avg_dice = total_dice_sum / max(positive_gt_samples, 1)  # pos-GT-only average
        avg_pixel_pos_ratio = total_pixel_pos_ratio / max(num_batches_for_pixel, 1)
        pred_positive_ratio = pred_positive_count / max(total_samples, 1)
        empty_fp_rate = empty_fp_count / max(empty_gt_samples, 1) if empty_gt_samples > 0 else 0
        
        # Logging
        if total_samples > 0:
            gt_stats_msg = (f"  📊 [Train] Epoch {epoch}: "
                           f"pos_GT={positive_gt_samples}/{total_samples} ({positive_gt_samples/total_samples:.1%}), "
                           f"empty_GT={empty_gt_samples}, threshold={self.threshold}")
            print(gt_stats_msg)
            self.logger.info(gt_stats_msg)
            
            fp_msg = (f"  📈 [Train] pred_positive_ratio={pred_positive_ratio:.1%}, "
                     f"empty_FP_rate={empty_fp_rate:.1%} ({empty_fp_count}/{empty_gt_samples}), "
                     f"FP_pixels={empty_fp_pixels:.0f}")
            print(fp_msg)
            self.logger.info(fp_msg)
            
            pixel_msg = f"  🔬 [Train] pixel_positive_ratio={avg_pixel_pos_ratio:.4%}"
            print(pixel_msg)
            self.logger.info(pixel_msg)
        
        return {
            'loss': avg_loss,
            'dice': avg_dice,  # pos-GT-only
            'positive_gt_samples': positive_gt_samples,
            'empty_gt_samples': empty_gt_samples,
            'total_samples': total_samples,
            'pred_positive_ratio': pred_positive_ratio,
            'empty_fp_rate': empty_fp_rate,
            'empty_fp_pixels': empty_fp_pixels
        }
    
    @torch.no_grad()
    def validate(self, loader: DataLoader = None, save_vis: bool = False) -> Dict[str, float]:
        """
        Validate model with pos-GT-only metrics.
        
        Dice/IoU are calculated ONLY over positive-GT samples.
        Empty GT samples contribute only to auxiliary metrics.
        PatientMetricsTracker receives only positive-GT samples.
        
        Args:
            loader: DataLoader (defaults to val_loader)
            save_vis: Whether to save visualizations
        
        Returns:
            Dictionary of validation metrics
        """
        if loader is None:
            loader = self.val_loader
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # Metrics lists (pos-GT only)
        all_dices = []
        all_ious = []
        
        # Counters
        total_samples = 0
        positive_gt_samples = 0
        empty_gt_samples = 0
        pred_positive_count = 0
        empty_fp_count = 0
        empty_fp_pixels = 0
        
        self.patient_tracker.clear()
        
        pbar = tqdm(loader, desc="Validation")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            patient_ids = batch['patient_id']
            slice_indices = batch['slice_idx']
            
            # Forward pass
            pred = self.model(images)
            loss = self.criterion(pred, masks)
            
            # Binary predictions using unified threshold
            pred_sigmoid = torch.sigmoid(pred)
            pred_binary = (pred_sigmoid > self.threshold).float()
            masks_binary = (masks > 0.5).float()
            
            # Pre-compute sums on GPU for efficiency
            B = masks.shape[0]
            gt_sums = masks_binary.view(B, -1).sum(dim=1)   # [B]
            pred_sums = pred_binary.view(B, -1).sum(dim=1)  # [B]
            
            for i in range(B):
                total_samples += 1
                gt_sum = gt_sums[i].item()
                pred_sum = pred_sums[i].item()
                
                if pred_sum > 0:
                    pred_positive_count += 1
                
                if gt_sum == 0:
                    # Empty GT - only update auxiliary metrics
                    empty_gt_samples += 1
                    if pred_sum > 0:
                        empty_fp_count += 1
                        empty_fp_pixels += pred_sum
                    # DO NOT add to all_dices/all_ious/patient_tracker
                    continue
                else:
                    # Positive GT - compute real metrics
                    positive_gt_samples += 1
                    
                    # Convert to numpy for compute_all_metrics
                    pred_np = pred_binary[i, 0].cpu().numpy()
                    mask_np = masks_binary[i, 0].cpu().numpy()
                    
                    metrics = compute_all_metrics(pred_np, mask_np)
                    all_dices.append(metrics['dice'])
                    all_ious.append(metrics['iou'])
                    
                    # Only add positive GT samples to tracker
                    self.patient_tracker.add_sample(
                        patient_ids[i],
                        slice_indices[i],
                        metrics
                    )
                    
                    # Save visualization for positive GT samples
                    if save_vis and self.visualizer and batch_idx < 10:
                        image_np = images[i].cpu().numpy()
                        self.visualizer.save_slice_comparison(
                            image_np, mask_np, pred_np,
                            patient_ids[i], slice_indices[i],
                            metrics['dice'], metrics['iou']
                        )
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice(posGT)': f'{np.mean(all_dices) if all_dices else 0:.4f}'
            })
        
        # Calculate averages (pos-GT-only)
        avg_loss = total_loss / max(num_batches, 1)
        avg_dice = np.mean(all_dices) if all_dices else 0.0
        avg_iou = np.mean(all_ious) if all_ious else 0.0
        std_dice = np.std(all_dices) if all_dices else 0.0
        pred_positive_ratio = pred_positive_count / max(total_samples, 1)
        empty_fp_rate = empty_fp_count / max(empty_gt_samples, 1) if empty_gt_samples > 0 else 0
        
        # Logging
        if total_samples > 0:
            val_stats_msg = (f"  📊 [Val] pos_GT={positive_gt_samples}/{total_samples} "
                            f"({positive_gt_samples/total_samples:.1%}), "
                            f"empty_GT={empty_gt_samples}, threshold={self.threshold}")
            print(val_stats_msg)
            self.logger.info(val_stats_msg)
            
            fp_msg = (f"  📈 [Val] pred_positive_ratio={pred_positive_ratio:.1%}, "
                     f"empty_FP_rate={empty_fp_rate:.1%} ({empty_fp_count}/{empty_gt_samples}), "
                     f"FP_pixels={empty_fp_pixels:.0f}")
            print(fp_msg)
            self.logger.info(fp_msg)
            
            if positive_gt_samples == 0:
                warn_msg = "  ⚠️ [Val] No positive GT samples! Dice/IoU are undefined (returning 0)."
                print(warn_msg)
                self.logger.warning(warn_msg)
        
        # Get patient-level metrics (only from positive GT samples)
        overall_metrics = self.patient_tracker.get_overall_metrics()
        
        return {
            'loss': avg_loss,
            'dice': avg_dice,  # pos-GT-only
            'iou': avg_iou,    # pos-GT-only (different from dice!)
            'std_dice': std_dice,
            'positive_gt_samples': positive_gt_samples,
            'empty_gt_samples': empty_gt_samples,
            'total_samples': total_samples,
            'pred_positive_ratio': pred_positive_ratio,
            'empty_fp_rate': empty_fp_rate,
            'empty_fp_pixels': empty_fp_pixels,
            **overall_metrics
        }
    
    def train(
        self,
        epochs: int,
        early_stopping_patience: int = 20,
        save_freq: int = 10
    ) -> Dict:
        """
        完整訓練流程
        
        Args:
            epochs: 訓練 epoch 數
            early_stopping_patience: 早停耐心值
            save_freq: 儲存頻率
        
        Returns:
            訓練歷史記錄
        """
        self.logger.info(f"Starting training for {epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self.logger.info(f"Threshold: {self.threshold} | Metrics: pos-GT-only")
        
        early_stopping = EarlyStopping(patience=early_stopping_patience, mode='max')
        history = {
            'train_loss': [],
            'train_dice': [],
            'val_loss': [],
            'val_dice': [],
            'val_iou': [],
            'lr': []
        }
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            # 訓練
            train_metrics = self.train_epoch(epoch)
            
            # 驗證
            val_metrics = self.validate(save_vis=(epoch % save_freq == 0))
            
            # 更新學習率
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['dice'])
                else:
                    self.scheduler.step()
            
            # 記錄歷史
            history['train_loss'].append(train_metrics['loss'])
            history['train_dice'].append(train_metrics['dice'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_dice'].append(val_metrics['dice'])
            history['val_iou'].append(val_metrics['iou'])
            history['lr'].append(current_lr)
            
            epoch_time = time.time() - epoch_start
            
            # 日誌
            self.logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Dice: {train_metrics['dice']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Dice: {val_metrics['dice']:.4f} | "
                f"Val IoU: {val_metrics['iou']:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )
            
            # 儲存最佳模型
            if val_metrics['dice'] > self.best_dice:
                self.best_dice = val_metrics['dice']
                self.best_epoch = epoch
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                self.logger.info(f"New best model! Dice: {self.best_dice:.4f}")
            
            # 定期儲存
            if epoch % save_freq == 0:
                self.save_checkpoint(epoch, val_metrics)
            
            # 早停檢查
            if early_stopping(val_metrics['dice']):
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time/3600:.2f} hours")
        self.logger.info(f"Best Dice: {self.best_dice:.4f} at epoch {self.best_epoch}")
        
        # 儲存訓練歷史
        self.save_training_history(history)
        self.plot_training_curves(history)
        
        return history
    
    def save_checkpoint(
        self, 
        epoch: int, 
        metrics: Dict, 
        is_best: bool = False
    ):
        """儲存模型 checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'best_dice': self.best_dice,
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if is_best:
            path = self.output_dir / 'best_model.pth'
        else:
            path = self.output_dir / f'checkpoint_epoch_{epoch}.pth'
        
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """載入模型 checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_dice = checkpoint.get('best_dice', 0.0)
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        self.logger.info(f"Best Dice: {self.best_dice:.4f}")
        
        return checkpoint.get('epoch', 0)
    
    def save_training_history(self, history: Dict):
        """儲存訓練歷史"""
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
    
    def plot_training_curves(self, history: Dict):
        """繪製訓練曲線"""
        import matplotlib
        matplotlib.use('Agg')
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss
        axes[0, 0].plot(history['train_loss'], label='Train')
        axes[0, 0].plot(history['val_loss'], label='Val')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Curve')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Dice
        axes[0, 1].plot(history['train_dice'], label='Train')
        axes[0, 1].plot(history['val_dice'], label='Val')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice')
        axes[0, 1].set_title('Dice Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # IoU
        axes[1, 0].plot(history['val_iou'], label='Val IoU')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('IoU')
        axes[1, 0].set_title('IoU Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning Rate
        axes[1, 1].plot(history['lr'])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=150)
        plt.close()
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader = None) -> Dict:
        """
        在測試集上評估模型
        
        Args:
            loader: 測試資料載入器
        
        Returns:
            評估指標
        """
        if loader is None:
            loader = self.test_loader if self.test_loader else self.val_loader
        
        self.logger.info("Evaluating model on test set...")
        
        metrics = self.validate(loader, save_vis=True)
        
        # 儲存評估結果
        results_path = self.output_dir / 'test_results.json'
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"Test Results: Dice={metrics['dice']:.4f}, IoU={metrics['iou']:.4f}")
        
        return metrics
    
    @torch.no_grad()
    def extract_llm_features(
        self, 
        loader: DataLoader = None, 
        output_dir: str = None
    ) -> Dict:
        """
        提取 LLM 特徵用於報告生成
        
        從 encoder 提取深層特徵 (512-dim)，並從分割結果提取形態學特徵
        
        Args:
            loader: 資料載入器 (預設使用 test_loader)
            output_dir: 輸出目錄 (預設使用 self.output_dir)
        
        Returns:
            包含所有樣本特徵的字典
        """
        if loader is None:
            loader = self.test_loader if self.test_loader else self.val_loader
        if output_dir is None:
            output_dir = self.output_dir
        
        output_dir = Path(output_dir)
        llm_features_dir = output_dir / "llm_features"
        llm_features_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.eval()
        all_features = {
            'extraction_info': {
                'total_samples': 0,
                'deep_feature_dim': None,
                'morphological_features': [
                    'area_pixels', 'area_ratio', 'circularity', 
                    'centroid_x', 'centroid_y', 'bbox_area_ratio'
                ]
            },
            'samples': []
        }
        
        self.logger.info("Extracting LLM features from model...")
        
        pbar = tqdm(loader, desc="Extracting LLM features")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            patient_ids = batch['patient_id']
            slice_indices = batch['slice_idx']
            
            # 1. 提取 encoder 深層特徵
            # 檢查模型是否有 encoder 屬性 (SMP models)
            if hasattr(self.model, 'encoder'):
                encoder_features = self.model.encoder(images)
                # encoder_features[-1] 是 bottleneck
                bottleneck = encoder_features[-1]  # [B, C, H, W]
                # Global Average Pooling -> [B, C]
                deep_features = F.adaptive_avg_pool2d(bottleneck, 1).flatten(1)
            else:
                # 自定義模型：使用 get_encoder_features
                if hasattr(self.model, 'get_encoder_features'):
                    encoder_features = self.model.get_encoder_features(images)
                    bottleneck = encoder_features[-1]
                    deep_features = F.adaptive_avg_pool2d(bottleneck, 1).flatten(1)
                else:
                    # 無法提取特徵，使用空向量
                    deep_features = torch.zeros(images.shape[0], 512, device=self.device)
            
            # 記錄特徵維度
            if all_features['extraction_info']['deep_feature_dim'] is None:
                all_features['extraction_info']['deep_feature_dim'] = deep_features.shape[1]
            
            # 2. 獲取分割預測
            outputs = self.model(images)
            pred_probs = torch.sigmoid(outputs)
            pred_masks = (pred_probs > 0.5).float()
            
            # 3. 為每個樣本提取特徵
            for i in range(images.shape[0]):
                patient_id = patient_ids[i]
                slice_idx = slice_indices[i] if isinstance(slice_indices[i], int) else slice_indices[i].item()
                
                # 深層特徵
                deep_feat = deep_features[i].cpu().numpy().tolist()
                
                # 形態學特徵
                pred_mask_np = pred_masks[i, 0].cpu().numpy()
                gt_mask_np = masks[i, 0].cpu().numpy()
                
                morph_features = self._extract_morphological_features(pred_mask_np)
                gt_morph_features = self._extract_morphological_features(gt_mask_np)
                
                # Dice score
                intersection = (pred_mask_np * gt_mask_np).sum()
                dice = (2.0 * intersection + 1e-6) / (pred_mask_np.sum() + gt_mask_np.sum() + 1e-6)
                
                sample_data = {
                    'patient_id': str(patient_id),
                    'slice_idx': int(slice_idx),
                    'deep_features': deep_feat,
                    'predicted_morphological': morph_features,
                    'ground_truth_morphological': gt_morph_features,
                    'dice_score': float(dice),
                    'prediction_confidence': float(pred_probs[i].mean().cpu().numpy())
                }
                
                all_features['samples'].append(sample_data)
        
        all_features['extraction_info']['total_samples'] = len(all_features['samples'])
        
        # 保存完整特徵 JSON
        json_path = llm_features_dir / "llm_features.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_features, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"✅ LLM features saved to: {json_path}")
        self.logger.info(f"   Total samples: {all_features['extraction_info']['total_samples']}")
        self.logger.info(f"   Deep feature dim: {all_features['extraction_info']['deep_feature_dim']}")
        
        # 生成摘要文件 (更易讀的格式)
        summary_path = llm_features_dir / "features_summary.txt"
        self._save_llm_features_summary(all_features, summary_path)
        
        return all_features
    
    def _extract_morphological_features(self, mask: np.ndarray) -> Dict:
        """
        從分割遮罩提取形態學特徵
        
        Args:
            mask: 二值遮罩 [H, W]
        
        Returns:
            形態學特徵字典
        """
        features = {
            'has_lesion': False,
            'area_pixels': 0,
            'area_ratio': 0.0,
            'circularity': 0.0,
            'centroid_x': 0.0,
            'centroid_y': 0.0,
            'bbox': [0, 0, 0, 0],
            'bbox_area_ratio': 0.0
        }
        
        mask_binary = (mask > 0.5).astype(np.uint8)
        area = mask_binary.sum()
        
        if area == 0:
            return features
        
        features['has_lesion'] = True
        features['area_pixels'] = int(area)
        features['area_ratio'] = float(area / mask_binary.size)
        
        # 找到連通區域
        try:
            from skimage import measure
            labeled = measure.label(mask_binary)
            props = measure.regionprops(labeled)
            
            if props:
                # 取最大的區域
                largest = max(props, key=lambda x: x.area)
                
                # Circularity
                perimeter = largest.perimeter
                if perimeter > 0:
                    features['circularity'] = float(4 * np.pi * largest.area / (perimeter ** 2))
                
                # Centroid (normalized)
                cy, cx = largest.centroid
                features['centroid_x'] = float(cx / mask.shape[1])
                features['centroid_y'] = float(cy / mask.shape[0])
                
                # Bounding box (minr, minc, maxr, maxc)
                minr, minc, maxr, maxc = largest.bbox
                features['bbox'] = [int(minc), int(minr), int(maxc), int(maxr)]
                
                # Bbox area ratio
                bbox_area = (maxr - minr) * (maxc - minc)
                if bbox_area > 0:
                    features['bbox_area_ratio'] = float(largest.area / bbox_area)
                    
        except ImportError:
            # 簡化版本（無 skimage）
            ys, xs = np.where(mask_binary > 0)
            if len(xs) > 0:
                features['centroid_x'] = float(np.mean(xs) / mask.shape[1])
                features['centroid_y'] = float(np.mean(ys) / mask.shape[0])
                features['bbox'] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        
        return features
    
    def _save_llm_features_summary(self, features: Dict, output_path: Path):
        """保存 LLM 特徵摘要為可讀文本"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("LLM Feature Extraction Summary\n")
            f.write("=" * 80 + "\n\n")
            
            info = features['extraction_info']
            f.write(f"Total Samples: {info['total_samples']}\n")
            f.write(f"Deep Feature Dimension: {info['deep_feature_dim']}\n")
            f.write(f"Morphological Features: {', '.join(info['morphological_features'])}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("Sample Details (showing first 10)\n")
            f.write("-" * 80 + "\n\n")
            
            for sample in features['samples'][:10]:
                f.write(f"Patient: {sample['patient_id']}, Slice: {sample['slice_idx']}\n")
                f.write(f"  Dice Score: {sample['dice_score']:.4f}\n")
                f.write(f"  Prediction Confidence: {sample['prediction_confidence']:.4f}\n")
                
                morph = sample['predicted_morphological']
                if morph['has_lesion']:
                    f.write(f"  Lesion Area: {morph['area_pixels']} pixels ({morph['area_ratio']*100:.2f}%)\n")
                    f.write(f"  Circularity: {morph['circularity']:.3f}\n")
                    f.write(f"  Centroid: ({morph['centroid_x']:.2f}, {morph['centroid_y']:.2f})\n")
                else:
                    f.write(f"  No lesion detected\n")
                f.write("\n")
            
            if len(features['samples']) > 10:
                f.write(f"... and {len(features['samples']) - 10} more samples\n")
        
        self.logger.info(f"✅ Features summary saved to: {output_path}")


def create_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader] = None,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    loss_type: str = "combined",
    epochs: int = 100,
    warmup_epochs: int = 5,
    output_dir: str = "result",
    device: torch.device = None,
    **kwargs
) -> UNetPPTrainer:
    """
    工廠函數: 創建訓練器
    
    Args:
        model: UNet++ 模型
        train_loader: 訓練資料載入器
        val_loader: 驗證資料載入器
        test_loader: 測試資料載入器
        lr: 學習率
        weight_decay: 權重衰減
        loss_type: 損失函數類型
        epochs: 訓練 epoch 數
        warmup_epochs: warmup epoch 數
        output_dir: 輸出目錄
        device: 計算設備
        **kwargs: 其他參數
    
    Returns:
        UNetPPTrainer 實例
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 損失函數
    criterion = get_loss_function(loss_type)
    
    # 優化器
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 調度器: Warmup + Cosine Annealing
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=0.1, 
        end_factor=1.0, 
        total_iters=warmup_epochs
    )
    main_scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=max(1, epochs - warmup_epochs),  # 確保 T_max >= 1 避免除零錯誤
        eta_min=1e-7
    )
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs]
    )
    
    trainer = UNetPPTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir,
        **kwargs
    )
    
    return trainer


if __name__ == "__main__":
    # 測試訓練器
    print("Testing UNet++ trainer...")
    
    from .model import UNetPlusPlus
    
    # 創建模型
    model = UNetPlusPlus(in_channels=3, out_channels=1)
    
    # 創建假資料
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 100
        
        def __getitem__(self, idx):
            return {
                'image': torch.randn(3, 256, 256),
                'mask': (torch.rand(1, 256, 256) > 0.5).float(),
                'bbox': torch.tensor([50, 50, 200, 200]).float(),
                'patient_id': f'patient_{idx}',
                'slice_idx': idx
            }
    
    dataset = DummyDataset()
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 創建訓練器
    trainer = create_trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        epochs=2,
        output_dir='test_output'
    )
    
    # 訓練一個 epoch
    train_metrics = trainer.train_epoch(1)
    print(f"Train metrics: {train_metrics}")
    
    # 驗證
    val_metrics = trainer.validate()
    print(f"Val metrics: {val_metrics}")
    
    print("\nTrainer tests passed!")
