#!/usr/bin/env python3
"""
MedSAM2 視頻模式訓練器
======================

利用 MedSAM2 的視頻預測能力，將 CT 切片序列當作視頻進行訓練。

使用 SAM2VideoTrainer (sam2/sam2_video_trainer.py) 作為核心訓練模組，
該模組支援梯度計算，適合 fine-tuning。

核心訓練策略：
1. 給定第一幀的 prompt (bbox)
2. 讓模型預測該幀的分割
3. 透過 memory 機制預測後續幀
4. 計算所有幀的損失並反向傳播

這種方法可以學習到：
- 病灶的空間連續性
- 跨切片的形態變化
- 時序一致的分割
"""

import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import json
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib.pyplot as plt

from .config import VideoConfig
from .video_dataset import VideoLesionDataset, collate_video_batch

logger = logging.getLogger(__name__)


class SegmentationMetrics:
    """
    分割評估指標計算器
    
    支援的指標:
    - Dice Score (F1)
    - IoU (Jaccard Index)
    - Precision
    - Recall (Sensitivity)
    - Specificity
    - Hausdorff Distance 95 (HD95)
    - Average Surface Distance (ASD)
    """
    
    def __init__(self, smooth: float = 1e-6):
        self.smooth = smooth
    
    def compute_all(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> Dict[str, float]:
        """
        計算所有指標
        
        Args:
            pred: 預測 mask (binary, 0/1)
            target: GT mask (binary, 0/1)
            spacing: 體素間距 (z, y, x) in mm，用於距離計算
            
        Returns:
            包含所有指標的字典
        """
        pred = pred.float()
        target = target.float()
        
        # 展平
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # 基本統計量
        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        fn = ((1 - pred_flat) * target_flat).sum()
        tn = ((1 - pred_flat) * (1 - target_flat)).sum()
        
        # Dice Score
        dice = (2.0 * tp + self.smooth) / (2.0 * tp + fp + fn + self.smooth)
        
        # IoU (Jaccard)
        iou = (tp + self.smooth) / (tp + fp + fn + self.smooth)
        
        # Precision
        precision = (tp + self.smooth) / (tp + fp + self.smooth)
        
        # Recall (Sensitivity)
        recall = (tp + self.smooth) / (tp + fn + self.smooth)
        
        # Specificity
        specificity = (tn + self.smooth) / (tn + fp + self.smooth)
        
        # F2 Score (更重視 Recall)
        f2 = (5.0 * tp + self.smooth) / (5.0 * tp + 4.0 * fn + fp + self.smooth)
        
        metrics = {
            'dice': dice.item(),
            'iou': iou.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'specificity': specificity.item(),
            'f2': f2.item(),
        }
        
        # 計算距離指標（需要 numpy，較慢）
        try:
            pred_np = pred.cpu().numpy()
            target_np = target.cpu().numpy()
            
            if pred_np.sum() > 0 and target_np.sum() > 0:
                hd95, asd = self._compute_surface_distances(pred_np, target_np, spacing)
                metrics['hd95'] = hd95
                metrics['asd'] = asd
            else:
                # 無法計算距離（其中一個為空）
                # 預測為空但GT有值 → 完全漏檢 → inf
                # 預測有值但GT為空 → 完全誤報 → inf  
                # 都為空 → 無法計算 → nan
                if pred_np.sum() == 0 and target_np.sum() == 0:
                    metrics['hd95'] = float('nan')
                    metrics['asd'] = float('nan')
                else:
                    metrics['hd95'] = float('inf')
                    metrics['asd'] = float('inf')
        except Exception as e:
            metrics['hd95'] = float('nan')
            metrics['asd'] = float('nan')
        
        return metrics
    
    def _compute_surface_distances(
        self, 
        pred: np.ndarray, 
        target: np.ndarray,
        spacing: Tuple[float, float, float]
    ) -> Tuple[float, float]:
        """
        計算表面距離指標 (HD95, ASD)
        
        使用 scipy.ndimage 進行距離變換
        """
        from scipy import ndimage
        
        # 獲取表面（邊界）
        pred_border = self._get_surface(pred)
        target_border = self._get_surface(target)
        
        if pred_border.sum() == 0 or target_border.sum() == 0:
            return float('inf'), float('inf')
        
        # 距離變換
        dt_pred = ndimage.distance_transform_edt(~pred.astype(bool), sampling=spacing)
        dt_target = ndimage.distance_transform_edt(~target.astype(bool), sampling=spacing)
        
        # 表面點到另一表面的距離
        dist_pred_to_target = dt_target[pred_border > 0]
        dist_target_to_pred = dt_pred[target_border > 0]
        
        # 合併所有表面距離
        all_distances = np.concatenate([dist_pred_to_target, dist_target_to_pred])
        
        # HD95
        hd95 = np.percentile(all_distances, 95) if len(all_distances) > 0 else float('inf')
        
        # ASD (Average Surface Distance)
        asd = np.mean(all_distances) if len(all_distances) > 0 else float('inf')
        
        return float(hd95), float(asd)
    
    def _get_surface(self, mask: np.ndarray) -> np.ndarray:
        """獲取 mask 的表面（邊界像素）"""
        from scipy import ndimage
        
        # 腐蝕操作
        eroded = ndimage.binary_erosion(mask)
        # 表面 = 原始 - 腐蝕
        surface = mask.astype(np.uint8) - eroded.astype(np.uint8)
        return surface


class DiceLoss(nn.Module):
    """Dice Loss"""
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        計算 Dice Loss
        
        Args:
            pred: 預測 logits (未經 sigmoid)
            target: Ground truth masks
        """
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1).float()
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance (AMP-safe version)"""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: logits (before sigmoid)
            target: binary target
        """
        target = target.float()
        
        # Use BCE with logits for numerical stability and AMP compatibility
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # Calculate pt for focal weight
        pred_prob = torch.sigmoid(pred)
        pt = torch.where(target == 1, pred_prob, 1 - pred_prob)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        return (focal_weight * bce).mean()


class TemporalConsistencyLoss(nn.Module):
    """
    時序一致性損失
    
    確保相鄰幀之間的預測具有平滑過渡
    """
    def __init__(self):
        super().__init__()
    
    def forward(
        self, 
        predictions: List[torch.Tensor],
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            predictions: List of (B, 1, H, W) predictions for each frame
            masks: (B, T, H, W) ground truth
        """
        if len(predictions) < 2:
            return torch.tensor(0.0, device=predictions[0].device)
        
        consistency_loss = 0.0
        count = 0
        
        for i in range(len(predictions) - 1):
            # 獲取相鄰幀的預測
            pred_curr = torch.sigmoid(predictions[i])  # (B, 1, H, W)
            pred_next = torch.sigmoid(predictions[i + 1])
            
            # 計算預測差異
            pred_diff = (pred_curr - pred_next).abs().mean()
            
            # 計算 GT mask 差異
            mask_curr = masks[:, i:i+1].float()
            mask_next = masks[:, i+1:i+2].float()
            mask_diff = (mask_curr - mask_next).abs().mean()
            
            # 預測差異應該與 mask 差異相近
            # 如果 mask 沒變，預測也不應該變太多
            consistency_loss += F.mse_loss(pred_diff, mask_diff)
            count += 1
        
        return consistency_loss / max(count, 1)


class MedSAM2VideoTrainer:
    """
    MedSAM2 視頻模式訓練器
    
    使用 SAM2VideoTrainer 進行視頻分割訓練。
    """
    
    def __init__(self, config: VideoConfig):
        """
        初始化訓練器
        
        Args:
            config: 視頻訓練配置
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # 設定輸出目錄
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        
        # 設定日誌
        self._setup_logging()
        
        # 載入模型
        self._load_model()
        
        # 損失函數
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.bce_loss = nn.BCEWithLogitsLoss()  # 新增 BCE Loss
        self.temporal_loss = TemporalConsistencyLoss()
        
        # 評估指標計算器
        self.metrics_calculator = SegmentationMetrics()
        
        # 訓練歷史 - 包含所有評估指標
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_dice': [],
            'val_iou': [],
            'val_precision': [],
            'val_recall': [],
            'val_specificity': [],
            'val_f2': [],
            'val_hd95': [],
            'val_asd': [],
            'learning_rate': [],
        }
        
        self.best_val_dice = 0.0
        self.best_epoch = 0
        
        logger.info("🎬 MedSAM2 視頻訓練器初始化完成")
    
    def _setup_logging(self):
        """設定日誌"""
        log_file = self.output_dir / "training.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        )
        
        logger.addHandler(file_handler)
    
    def _load_model(self):
        """載入 SAM2VideoTrainer 模型"""
        import sys
        import os
        
        # 添加 MedSAM2 路徑
        medsam2_path = Path(__file__).parent.parent / "MedSAM2"
        if str(medsam2_path) not in sys.path:
            sys.path.insert(0, str(medsam2_path))
        
        # 初始化 Hydra 配置（必須在 import SAM2VideoTrainer 之前）
        from hydra import initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        
        # 清除已存在的 Hydra 實例
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        
        # 設定配置目錄
        config_dir = os.path.abspath(str(medsam2_path / "sam2" / "configs"))
        initialize_config_dir(config_dir=config_dir, version_base="1.2")
        logger.info(f"📁 Hydra 配置目錄: {config_dir}")
        
        from sam2.sam2_video_trainer import SAM2VideoTrainer
        
        # 模型配置（不需要 .yaml 後綴）
        model_cfg = self.config.model.config.replace('.yaml', '')
        checkpoint_path = Path(__file__).parent.parent / self.config.model.checkpoint
        
        logger.info(f"📦 載入模型: {model_cfg}")
        logger.info(f"📦 Checkpoint: {checkpoint_path}")
        
        # 建立視頻訓練器
        self.model = SAM2VideoTrainer(
            model_cfg=model_cfg,
            sam2_checkpoint=str(checkpoint_path),
            device=self.device,
            memory_size=7,
            mask_threshold=0.5,
            use_mask_threshold=False,  # 訓練時不使用 threshold
        )
        
        # 凍結 image encoder（可選）
        if hasattr(self.model.model, 'image_encoder'):
            for param in self.model.model.image_encoder.parameters():
                param.requires_grad = False
            logger.info("🔒 Image Encoder 已凍結")
        
        # 凍結 prompt encoder（可選，減少過擬合）
        freeze_prompt = getattr(self.config.training, 'freeze_prompt_encoder', False)
        if freeze_prompt and hasattr(self.model.model, 'sam_prompt_encoder'):
            for param in self.model.model.sam_prompt_encoder.parameters():
                param.requires_grad = False
            logger.info("🔒 Prompt Encoder 已凍結")
        
        # 凍結 memory encoder（可選，進一步減少過擬合）
        freeze_memory = getattr(self.config.training, 'freeze_memory_encoder', False)
        if freeze_memory and hasattr(self.model.model, 'memory_encoder'):
            for param in self.model.model.memory_encoder.parameters():
                param.requires_grad = False
            logger.info("🔒 Memory Encoder 已凍結")
        
        # 取得可訓練參數
        self.trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        total_params = sum(p.numel() for p in self.trainable_params)
        logger.info(f"🔧 可訓練參數: {total_params:,}")
    
    def _create_optimizer(self):
        """建立優化器和調度器 - 使用分層學習率"""
        # 分層學習率：mask decoder 和 memory attention 使用更高學習率
        param_groups = []
        decoder_params = []
        memory_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'mask_decoder' in name or 'sam_mask_decoder' in name:
                decoder_params.append(param)
            elif 'memory' in name:
                memory_params.append(param)
            else:
                other_params.append(param)
        
        base_lr = self.config.training.learning_rate
        decoder_multiplier = getattr(self.config.training, 'decoder_lr_multiplier', 10.0)
        
        if decoder_params:
            param_groups.append({
                'params': decoder_params,
                'lr': base_lr * decoder_multiplier,
                'name': 'mask_decoder'
            })
            logger.info(f"  - Mask Decoder: {len(decoder_params)} params @ lr={base_lr * decoder_multiplier:.2e}")
        
        if memory_params:
            memory_multiplier = 2.0  # memory 模組用較低倍率確保穩定
            param_groups.append({
                'params': memory_params,
                'lr': base_lr * memory_multiplier,
                'name': 'memory'
            })
            logger.info(f"  - Memory: {len(memory_params)} params @ lr={base_lr * memory_multiplier:.2e}")
        
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': base_lr,
                'name': 'other'
            })
            logger.info(f"  - Other: {len(other_params)} params @ lr={base_lr:.2e}")
        
        self.optimizer = AdamW(
            param_groups if param_groups else self.trainable_params,
            lr=base_lr,
            weight_decay=self.config.training.weight_decay,
        )
        
        # Scheduler 在 train() 中創建，因為需要知道實際的 steps 數量
        self.scheduler = None
        self._param_groups = param_groups
        self._max_lr_cap = getattr(self.config.training, 'max_lr', 3e-4)
        
        # AMP Scaler
        self.scaler = GradScaler() if self.config.training.use_amp else None
        
        logger.info(f"⚙️ 優化器: AdamW (lr={self.config.training.learning_rate})")
    
    def _create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """建立資料載入器"""
        train_dataset = VideoLesionDataset(
            npz_dir=self.config.data.npz_dir,
            split="train",
            image_size=self.config.model.image_size,
            max_video_length=self.config.data.max_video_length,
            augmentation=True,
        )
        
        val_dataset = VideoLesionDataset(
            npz_dir=self.config.data.npz_dir,
            split="val",
            image_size=self.config.model.image_size,
            max_video_length=self.config.data.max_video_length,
            augmentation=False,
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=collate_video_batch,
            pin_memory=True,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collate_video_batch,
            pin_memory=True,
        )
        
        logger.info(f"📊 訓練集: {len(train_dataset)} 樣本")
        logger.info(f"📊 驗證集: {len(val_dataset)} 樣本")
        
        return train_loader, val_loader
    
    def _create_scheduler(self, num_training_steps: int):
        """建立學習率調度器"""
        max_lr_cap = self._max_lr_cap
        param_groups = self._param_groups
        
        # 使用 OneCycleLR 配合平緩的 warmup
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=[min(pg['lr'] * 2, max_lr_cap) for pg in param_groups] if param_groups else max_lr_cap,
            total_steps=num_training_steps,
            pct_start=0.15,  # warmup 佔 15%
            anneal_strategy='cos',
            div_factor=10.0,  # 起始 lr = max_lr / 10
            final_div_factor=100.0,  # 結束 lr = max_lr / 100
        )
        logger.info(f"📈 學習率調度: OneCycleLR (max_lr={max_lr_cap:.2e}, total_steps={num_training_steps})")
    
    def train(self):
        """執行訓練"""
        logger.info("=" * 60)
        logger.info("🚀 開始視頻模式訓練")
        logger.info("=" * 60)
        
        # 建立資料載入器
        train_loader, val_loader = self._create_dataloaders()
        
        if len(train_loader) == 0:
            logger.error("❌ 訓練集為空，請先執行 NPZ 轉換")
            return
        
        # 建立優化器
        self._create_optimizer()
        
        # 計算總訓練步數並建立 scheduler
        num_training_steps = len(train_loader) * self.config.training.epochs
        self._create_scheduler(num_training_steps)
        
        # 保存配置
        self.config.save(self.output_dir / "config.json")
        
        # 訓練迴圈
        patience_counter = 0
        
        for epoch in range(1, self.config.training.epochs + 1):
            epoch_start = time.time()
            
            # 記錄當前 epoch（用於課程學習策略）
            self._current_epoch = epoch
            
            # 訓練
            train_loss = self._train_epoch(train_loader, epoch)
            
            # 驗證 - 現在返回所有評估指標
            val_loss, val_metrics = self._validate(val_loader, epoch)
            
            # 記錄當前學習率
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 記錄歷史 - 所有指標
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_dice'].append(val_metrics['dice'])
            self.history['val_iou'].append(val_metrics['iou'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['val_specificity'].append(val_metrics['specificity'])
            self.history['val_f2'].append(val_metrics['f2'])
            self.history['val_hd95'].append(val_metrics.get('hd95', float('nan')))
            self.history['val_asd'].append(val_metrics.get('asd', float('nan')))
            self.history['learning_rate'].append(current_lr)
            
            epoch_time = time.time() - epoch_start
            
            # 日誌 - 詳細指標
            logger.info(
                f"Epoch {epoch:3d}/{self.config.training.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )
            logger.info(
                f"  📊 Dice: {val_metrics['dice']:.4f} | "
                f"IoU: {val_metrics['iou']:.4f} | "
                f"Prec: {val_metrics['precision']:.4f} | "
                f"Recall: {val_metrics['recall']:.4f} | "
                f"F2: {val_metrics['f2']:.4f}"
            )
            hd95_val = val_metrics.get('hd95', float('nan'))
            asd_val = val_metrics.get('asd', float('nan'))
            if np.isfinite(hd95_val) and hd95_val < 1000:
                logger.info(
                    f"  📏 HD95: {hd95_val:.2f}mm | "
                    f"ASD: {asd_val:.2f}mm"
                )
            elif np.isinf(hd95_val):
                logger.info(
                    f"  📏 HD95: inf (預測為空) | "
                    f"ASD: inf"
                )
            else:
                logger.info(
                    f"  📏 HD95: N/A | ASD: N/A"
                )
            
            val_dice = val_metrics['dice']
            
            # 保存最佳模型
            if val_dice > self.best_val_dice:
                self.best_val_dice = val_dice
                self.best_epoch = epoch
                self._save_checkpoint(epoch, is_best=True)
                patience_counter = 0
                logger.info(f"  ⭐ 新的最佳模型! Dice: {val_dice:.4f}")
            else:
                patience_counter += 1
            
            # 定期保存
            if epoch % 10 == 0:
                self._save_checkpoint(epoch)
            
            # 早停
            if patience_counter >= self.config.training.early_stopping_patience:
                logger.info(f"Early stopping triggered (patience={self.config.training.early_stopping_patience})")
                break
        
        # 訓練完成
        self._save_checkpoint(epoch, is_final=True)
        self._save_history()
        self._plot_history()
        
        # 最終評估報告
        self._log_final_report()
        
        logger.info("=" * 60)
        logger.info(f"✅ 訓練完成!")
        logger.info(f"  - 最佳 Dice: {self.best_val_dice:.4f} (Epoch {self.best_epoch})")
        logger.info("=" * 60)
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """訓練一個 epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            loss = self._train_step(batch)
            
            if loss is not None:
                total_loss += loss
                num_batches += 1
                pbar.set_postfix({'loss': f'{loss:.4f}'})
                
                # OneCycleLR 需要每個 batch 更新
                if self.scheduler is not None:
                    self.scheduler.step()
        
        return total_loss / max(num_batches, 1)
    
    def _train_step(self, batch: Dict) -> Optional[float]:
        """單一訓練步驟"""
        try:
            # 獲取資料
            # batch['frames']: (B, D, 3, H, W) - 視頻幀
            # batch['masks']: (B, D, H, W) - GT masks
            # batch['bbox']: (B, 4) - 第一幀的 bounding box
            
            frames = batch['frames'].to(self.device)  # (B, D, 3, H, W)
            masks = batch['masks'].to(self.device)    # (B, D, H, W)
            bbox = batch['bbox'].to(self.device)      # (B, 4)
            center_idx = batch['center_idx'][0]       # 中心幀索引
            
            B, D, C, H, W = frames.shape
            
            # 獲取 prompt 類型設定
            prompt_type = getattr(self.config.training, 'prompt_type', 'bbox')
            
            # 準備 prompt
            if prompt_type == 'point':
                # Point prompt: (point_coords, point_labels)
                center_point = batch['center_point'].to(self.device)  # (B, 2)
                # 擴展為 (B, 1, 2) - 一個前景點
                point_coords = center_point.unsqueeze(1)  # (B, 1, 2)
                point_labels = torch.ones(B, 1, dtype=torch.int32, device=self.device)  # 1 = foreground
                points = (point_coords, point_labels)
                bboxes_for_model = None
            else:
                # Bbox prompt
                points = None
                bboxes_for_model = bbox
            
            with autocast(enabled=self.config.training.use_amp):
                # ============================================================
                # 改進的訓練策略：混合 GT 引導和自主預測
                # ============================================================
                # 策略：
                # 1. 只給中心幀（有標註的那幀）提供 GT mask 作為條件
                # 2. 其他幀讓模型自己傳播預測
                # 3. 這樣訓練和驗證更一致，模型學會真正的時序傳播
                # ============================================================
                
                # 準備 GT masks - 只在中心幀提供指導
                # 其餘幀使用 None 或零 mask，讓模型自己傳播
                gt_masks_for_memory = torch.zeros(B, D, 1, H, W, device=self.device)
                gt_masks_for_memory[:, center_idx, 0] = masks[:, center_idx].float()
                
                # 使用課程學習策略：隨機決定是否使用 GT 引導
                # 訓練初期多用 GT，後期減少依賴
                use_gt_guidance = getattr(self, '_current_epoch', 1) <= self.config.training.warmup_epochs
                
                if use_gt_guidance:
                    # Warmup 階段：使用完整 GT 指導
                    full_gt_masks = masks.unsqueeze(2).float()  # (B, T, 1, H, W)
                    all_masks, all_logits, all_ious = self.model(
                        videos=frames,
                        bboxes=bboxes_for_model,
                        points=points,
                        labels=full_gt_masks,
                    )
                else:
                    # 正常訓練：只用中心幀 GT，其他幀自主傳播
                    all_masks, all_logits, all_ious = self.model(
                        videos=frames,
                        bboxes=bboxes_for_model,
                        points=points,
                        labels=gt_masks_for_memory,  # 只有中心幀有 GT
                    )
                
                # 計算損失
                # all_masks: List of (B, 1, H, W)
                # all_logits: List of (B, 1, H, W)
                
                total_loss = torch.tensor(0.0, device=self.device)
                bce_weight = getattr(self.config.training, 'bce_weight', 0.5)
                
                for t, (pred_logit, pred_mask) in enumerate(zip(all_logits, all_masks)):
                    target = masks[:, t].unsqueeze(1).float()  # (B, 1, H, W)
                    
                    # 確保大小匹配
                    if pred_logit.shape[-2:] != target.shape[-2:]:
                        pred_logit = F.interpolate(
                            pred_logit, size=target.shape[-2:],
                            mode='bilinear', align_corners=False
                        )
                    
                    # Dice + Focal + BCE Loss
                    dice_l = self.dice_loss(pred_logit, target)
                    focal_l = self.focal_loss(pred_logit, target)
                    bce_l = self.bce_loss(pred_logit, target)
                    
                    # 對中心幀給予更高權重（這是有標註的幀）
                    frame_weight = 2.0 if t == center_idx else 1.0
                    
                    total_loss += frame_weight * (
                        self.config.training.dice_weight * dice_l +
                        self.config.training.focal_weight * focal_l +
                        bce_weight * bce_l
                    )
                
                # 時序一致性損失
                if len(all_logits) > 1:
                    temporal_l = self.temporal_loss(all_logits, masks)
                    total_loss += self.config.training.propagation_weight * temporal_l
                
                # 平均
                total_loss = total_loss / D
            
            # 反向傳播
            self.optimizer.zero_grad()
            
            if self.scaler:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.trainable_params, 
                    self.config.training.grad_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.trainable_params,
                    self.config.training.grad_clip
                )
                self.optimizer.step()
            
            return total_loss.item()
            
        except Exception as e:
            logger.warning(f"⚠️ 訓練步驟失敗: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    @torch.no_grad()
    def _validate(self, val_loader: DataLoader, epoch: int) -> Tuple[float, Dict[str, float]]:
        """驗證並計算所有評估指標"""
        self.model.eval()
        
        total_loss = 0.0
        num_samples = 0
        
        # 累積所有指標
        metrics_sum = {
            'dice': 0.0,
            'iou': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'specificity': 0.0,
            'f2': 0.0,
            'hd95': 0.0,
            'asd': 0.0,
        }
        hd95_count = 0  # HD95 有效樣本數（排除 inf）
        
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
        
        for batch in pbar:
            loss, metrics = self._validate_step(batch)
            
            if loss is not None and metrics is not None:
                total_loss += loss
                num_samples += 1
                
                for key in metrics_sum:
                    if key in ['hd95', 'asd']:
                        # 距離指標需要特殊處理（排除 inf 和 nan）
                        val = metrics.get(key, float('inf'))
                        if np.isfinite(val):
                            metrics_sum[key] += val
                            if key == 'hd95':
                                hd95_count += 1
                    else:
                        metrics_sum[key] += metrics.get(key, 0.0)
                
                pbar.set_postfix({
                    'dice': f"{metrics['dice']:.4f}",
                    'iou': f"{metrics['iou']:.4f}"
                })
        
        # 計算平均值
        avg_loss = total_loss / max(num_samples, 1)
        avg_metrics = {}
        for key in metrics_sum:
            if key in ['hd95', 'asd']:
                # 距離指標：若無有效樣本則返回 inf
                if hd95_count > 0:
                    avg_metrics[key] = metrics_sum[key] / hd95_count
                else:
                    avg_metrics[key] = float('inf')
            else:
                avg_metrics[key] = metrics_sum[key] / max(num_samples, 1)
        
        return avg_loss, avg_metrics
    
    def _validate_step(self, batch: Dict) -> Tuple[Optional[float], Optional[Dict[str, float]]]:
        """
        單一驗證步驟 - 計算所有評估指標
        
        驗證策略：
        - 給模型中心幀的 GT mask 作為初始條件（模擬真實使用場景）
        - 評估模型將這個 mask 傳播到其他幀的能力
        - 這與推論時用戶提供第一幀標註的場景一致
        """
        try:
            frames = batch['frames'].to(self.device)
            masks = batch['masks'].to(self.device)
            bbox = batch['bbox'].to(self.device)
            center_idx = batch['center_idx'][0]  # 中心幀索引
            
            B, D, C, H, W = frames.shape
            
            # 獲取 prompt 類型設定
            prompt_type = getattr(self.config.training, 'prompt_type', 'bbox')
            
            # 準備 prompt
            if prompt_type == 'point':
                center_point = batch['center_point'].to(self.device)  # (B, 2)
                point_coords = center_point.unsqueeze(1)  # (B, 1, 2)
                point_labels = torch.ones(B, 1, dtype=torch.int32, device=self.device)
                points = (point_coords, point_labels)
                bboxes_for_model = None
            else:
                points = None
                bboxes_for_model = bbox
            
            # 準備初始條件：只給中心幀的 GT mask
            # 這模擬真實推論場景：用戶在一幀上標註，模型傳播到其他幀
            init_labels = torch.zeros(B, D, 1, H, W, device=self.device)
            init_labels[:, center_idx, 0] = masks[:, center_idx].float()
            
            # Forward - 使用中心幀 GT 作為初始條件
            all_masks, all_logits, all_ious = self.model(
                videos=frames,
                bboxes=bboxes_for_model,
                points=points,
                labels=init_labels,  # 只有中心幀有 GT
            )
            
            # 收集所有預測
            all_preds = []
            all_targets = []
            
            for t, pred_mask in enumerate(all_masks):
                target = masks[:, t].unsqueeze(1).float()
                
                if pred_mask.shape[-2:] != target.shape[-2:]:
                    pred_mask = F.interpolate(
                        pred_mask, size=target.shape[-2:],
                        mode='bilinear', align_corners=False
                    )
                
                all_preds.append(pred_mask)
                all_targets.append(target)
            
            # Stack predictions and targets
            preds = torch.cat(all_preds, dim=1)  # (B, D, H, W)
            targets = torch.cat(all_targets, dim=1)
            
            # Binary predictions
            preds_binary = (preds > 0.5).float()
            
            # 計算所有評估指標
            metrics = self.metrics_calculator.compute_all(
                pred=preds_binary,
                target=targets,
                spacing=(1.0, 1.0, 1.0)  # 可根據實際 spacing 調整
            )
            
            # Loss = 1 - Dice
            loss = 1.0 - metrics['dice']
            
            return loss, metrics
            
        except Exception as e:
            logger.warning(f"⚠️ 驗證步驟失敗: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False, is_final: bool = False):
        """保存 checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_dice': self.best_val_dice,
            'history': self.history,
            'config': {
                'model': self.config.model.__dict__,
                'training': self.config.training.__dict__,
            }
        }
        
        if is_best:
            path = self.output_dir / "checkpoints" / "best_model.pt"
        elif is_final:
            path = self.output_dir / "checkpoints" / "final_model.pt"
        else:
            path = self.output_dir / "checkpoints" / f"epoch_{epoch:03d}.pt"
        
        torch.save(checkpoint, path)
        logger.info(f"💾 Checkpoint 保存: {path.name}")
    
    def _save_history(self):
        """保存訓練歷史"""
        history_path = self.output_dir / "history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def _plot_history(self):
        """繪製訓練曲線 - 包含所有評估指標"""
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # 1. Loss Curve
        axes[0, 0].plot(epochs, self.history['train_loss'], label='Train', color='blue')
        axes[0, 0].plot(epochs, self.history['val_loss'], label='Val', color='orange')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Curve')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. Dice Score
        axes[0, 1].plot(epochs, self.history['val_dice'], label='Val Dice', color='green')
        axes[0, 1].axhline(y=self.best_val_dice, color='r', linestyle='--', 
                          label=f'Best: {self.best_val_dice:.4f}')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Score')
        axes[0, 1].set_title('Validation Dice')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        axes[0, 1].set_ylim([0, 1])
        
        # 3. IoU
        axes[0, 2].plot(epochs, self.history['val_iou'], label='Val IoU', color='purple')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('IoU')
        axes[0, 2].set_title('Validation IoU (Jaccard)')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        axes[0, 2].set_ylim([0, 1])
        
        # 4. Precision & Recall
        axes[1, 0].plot(epochs, self.history['val_precision'], label='Precision', color='blue')
        axes[1, 0].plot(epochs, self.history['val_recall'], label='Recall', color='red')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Precision & Recall')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        axes[1, 0].set_ylim([0, 1])
        
        # 5. Specificity & F2
        axes[1, 1].plot(epochs, self.history['val_specificity'], label='Specificity', color='cyan')
        axes[1, 1].plot(epochs, self.history['val_f2'], label='F2 Score', color='magenta')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Specificity & F2 Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_ylim([0, 1])
        
        # 6. HD95 (Hausdorff Distance 95)
        hd95_values = [v if np.isfinite(v) else np.nan for v in self.history['val_hd95']]
        axes[1, 2].plot(epochs, hd95_values, label='HD95', color='brown', marker='.')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('HD95 (mm)')
        axes[1, 2].set_title('Hausdorff Distance 95')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        # 7. ASD (Average Surface Distance)
        asd_values = [v if np.isfinite(v) else np.nan for v in self.history['val_asd']]
        axes[2, 0].plot(epochs, asd_values, label='ASD', color='olive', marker='.')
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('ASD (mm)')
        axes[2, 0].set_title('Average Surface Distance')
        axes[2, 0].legend()
        axes[2, 0].grid(True)
        
        # 8. Learning Rate
        axes[2, 1].plot(epochs, self.history['learning_rate'], color='gray')
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('Learning Rate')
        axes[2, 1].set_title('Learning Rate Schedule')
        axes[2, 1].set_yscale('log')
        axes[2, 1].grid(True)
        
        # 9. Summary
        axes[2, 2].axis('off')
        
        # 獲取最佳 epoch 的所有指標
        best_idx = self.best_epoch - 1 if self.best_epoch > 0 else len(self.history['val_dice']) - 1
        summary_text = f"""
╔══════════════════════════════════════╗
║         Training Summary             ║
╠══════════════════════════════════════╣
║ Best Epoch: {self.best_epoch:4d}                     ║
║ Total Epochs: {len(self.history['train_loss']):4d}                   ║
╠══════════════════════════════════════╣
║ Best Metrics (Epoch {self.best_epoch}):             ║
║   Dice:        {self.history['val_dice'][best_idx]:.4f}               ║
║   IoU:         {self.history['val_iou'][best_idx]:.4f}               ║
║   Precision:   {self.history['val_precision'][best_idx]:.4f}               ║
║   Recall:      {self.history['val_recall'][best_idx]:.4f}               ║
║   F2 Score:    {self.history['val_f2'][best_idx]:.4f}               ║
╠══════════════════════════════════════╣
║ Final Train Loss: {self.history['train_loss'][-1]:.4f}            ║
║ Final Val Loss:   {self.history['val_loss'][-1]:.4f}            ║
╚══════════════════════════════════════╝
        """
        axes[2, 2].text(0.05, 0.5, summary_text, fontsize=10, family='monospace',
                       verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_curves.png", dpi=150)
        plt.close()
        
        # 額外保存詳細指標圖
        self._plot_detailed_metrics()
        
        logger.info(f"📊 訓練曲線保存: training_curves.png")
    
    def _plot_detailed_metrics(self):
        """繪製詳細指標比較圖"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # 繪製所有主要指標
        ax.plot(epochs, self.history['val_dice'], label='Dice', linewidth=2)
        ax.plot(epochs, self.history['val_iou'], label='IoU', linewidth=2)
        ax.plot(epochs, self.history['val_precision'], label='Precision', linewidth=1.5, linestyle='--')
        ax.plot(epochs, self.history['val_recall'], label='Recall', linewidth=1.5, linestyle='--')
        ax.plot(epochs, self.history['val_f2'], label='F2', linewidth=1.5, linestyle=':')
        
        ax.axvline(x=self.best_epoch, color='red', linestyle='--', alpha=0.5, label=f'Best Epoch ({self.best_epoch})')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('All Validation Metrics Over Training', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "detailed_metrics.png", dpi=150)
        plt.close()
    
    def _log_final_report(self):
        """輸出最終評估報告"""
        best_idx = self.best_epoch - 1 if self.best_epoch > 0 else len(self.history['val_dice']) - 1
        
        logger.info("=" * 60)
        logger.info("📋 最終評估報告")
        logger.info("=" * 60)
        logger.info(f"  最佳 Epoch: {self.best_epoch}")
        logger.info("-" * 40)
        logger.info("  [分割品質指標]")
        logger.info(f"    Dice Score:    {self.history['val_dice'][best_idx]:.4f}")
        logger.info(f"    IoU (Jaccard): {self.history['val_iou'][best_idx]:.4f}")
        logger.info("-" * 40)
        logger.info("  [分類指標]")
        logger.info(f"    Precision:     {self.history['val_precision'][best_idx]:.4f}")
        logger.info(f"    Recall:        {self.history['val_recall'][best_idx]:.4f}")
        logger.info(f"    Specificity:   {self.history['val_specificity'][best_idx]:.4f}")
        logger.info(f"    F2 Score:      {self.history['val_f2'][best_idx]:.4f}")
        logger.info("-" * 40)
        logger.info("  [距離指標]")
        hd95 = self.history['val_hd95'][best_idx]
        asd = self.history['val_asd'][best_idx]
        if np.isfinite(hd95):
            logger.info(f"    HD95:          {hd95:.2f} mm")
            logger.info(f"    ASD:           {asd:.2f} mm")
        else:
            logger.info(f"    HD95:          N/A")
            logger.info(f"    ASD:           N/A")
        logger.info("=" * 60)
    
    def load_checkpoint(self, checkpoint_path: str):
        """載入 checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_dice = checkpoint.get('best_val_dice', 0.0)
        self.history = checkpoint.get('history', self.history)
        
        logger.info(f"📦 載入 checkpoint: {checkpoint_path}")
        logger.info(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
        logger.info(f"  - Best Dice: {self.best_val_dice:.4f}")
