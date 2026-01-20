#!/usr/bin/env python3
"""
3D U-Net Video Trainer
======================

Trainer logic for 3D U-Net.
"""

import logging
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from skimage import measure, morphology
from typing import Dict, Tuple, Optional

from .dataset import VolumetricDataset, collate_video_batch
from .model import get_model
from .config import Config

logger = logging.getLogger(__name__)

# ============ Postprocess Functions ============

def generate_lung_mask(volume: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """
    從 CT volume 生成簡易肺部遮罩
    使用 Otsu-like thresholding 和形態學操作
    
    Args:
        volume: (D, H, W) normalized to [0, 1]
        threshold: 肺部閾值 (通常肺部較暗)
        
    Returns:
        lung_mask: (D, H, W) binary mask
    """
    # 肺部通常比較暗（在 CT 中 HU 較低）
    # 假設 volume 已經正規化到 [0, 1]，肺部大約在 0.2-0.4 範圍
    binary = (volume < threshold).astype(np.uint8)
    
    # 形態學操作：閉運算填補小孔
    struct = morphology.ball(2) if binary.ndim == 3 else morphology.disk(2)
    binary = morphology.binary_closing(binary, struct)
    
    # 移除小的連通區域，保留最大的兩個（左右肺）
    labels = measure.label(binary)
    if labels.max() > 0:
        regions = measure.regionprops(labels)
        # 按面積排序，保留最大的兩個
        regions = sorted(regions, key=lambda x: x.area, reverse=True)
        lung_mask = np.zeros_like(binary)
        for region in regions[:2]:  # 最多保留兩個（左右肺）
            if region.area > 1000:  # 過濾太小的區域
                lung_mask[labels == region.label] = 1
        return lung_mask.astype(np.uint8)
    
    return binary


def postprocess_prediction(
    prob_mask: np.ndarray,
    lung_mask: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    min_size_voxels: int = 10,
    apply_closing: bool = True
) -> np.ndarray:
    """
    後處理預測結果
    
    Args:
        prob_mask: (D, H, W) 機率圖 [0, 1]
        lung_mask: (D, H, W) 肺部遮罩（可選）
        threshold: 二值化閾值
        min_size_voxels: 最小連通區域體素數
        apply_closing: 是否應用閉運算
        
    Returns:
        binary_mask: 後處理後的二值遮罩
    """
    # Step 1: Threshold
    binary = (prob_mask > threshold).astype(np.uint8)
    
    # Step 2: Apply lung mask
    if lung_mask is not None:
        binary = binary * (lung_mask > 0).astype(np.uint8)
    
    # Step 3: Connected component filtering
    if binary.sum() > 0:
        labels = measure.label(binary)
        filtered = np.zeros_like(binary)
        for region in measure.regionprops(labels):
            if region.area >= min_size_voxels:
                filtered[labels == region.label] = 1
        binary = filtered
    
    # Step 4: Morphological closing
    if apply_closing and binary.sum() > 0:
        struct = morphology.ball(1) if binary.ndim == 3 else morphology.disk(1)
        binary = morphology.binary_closing(binary, struct).astype(np.uint8)
    
    return binary.astype(np.uint8)


def calc_detection_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    iou_threshold: float = 0.1
) -> Dict[str, float]:
    """
    計算結節檢測指標 (基於連通區域)
    
    Args:
        pred: (D, H, W) 預測的二值遮罩
        gt: (D, H, W) Ground truth 二值遮罩
        iou_threshold: 判定為 TP 的 IoU 閾值
        
    Returns:
        metrics: Dict with TP, FP, FN, Precision, Recall, F1
    """
    # 標記連通區域
    pred_labels = measure.label(pred)
    gt_labels = measure.label(gt)
    
    pred_regions = measure.regionprops(pred_labels)
    gt_regions = measure.regionprops(gt_labels)
    
    n_pred = len(pred_regions)
    n_gt = len(gt_regions)
    
    # 特殊情況處理
    if n_gt == 0 and n_pred == 0:
        return {'TP': 0, 'FP': 0, 'FN': 0, 'Precision': 1.0, 'Recall': 1.0, 'F1': 1.0}
    if n_gt == 0:
        return {'TP': 0, 'FP': n_pred, 'FN': 0, 'Precision': 0.0, 'Recall': 1.0, 'F1': 0.0}
    if n_pred == 0:
        return {'TP': 0, 'FP': 0, 'FN': n_gt, 'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0}
    
    # 計算每個 GT 區域是否被偵測到
    gt_matched = [False] * n_gt
    pred_matched = [False] * n_pred
    
    for i, gt_region in enumerate(gt_regions):
        gt_mask = (gt_labels == gt_region.label)
        best_iou = 0
        best_pred_idx = -1
        
        for j, pred_region in enumerate(pred_regions):
            if pred_matched[j]:
                continue
            pred_mask = (pred_labels == pred_region.label)
            
            # 計算 IoU
            inter = np.logical_and(gt_mask, pred_mask).sum()
            union = np.logical_or(gt_mask, pred_mask).sum()
            iou = inter / (union + 1e-6)
            
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = j
        
        if best_iou >= iou_threshold and best_pred_idx >= 0:
            gt_matched[i] = True
            pred_matched[best_pred_idx] = True
    
    TP = sum(gt_matched)
    FN = n_gt - TP
    FP = sum([not m for m in pred_matched])
    
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    return {
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }


def calc_dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """計算 Dice Score"""
    inter = np.logical_and(pred > 0, gt > 0).sum()
    union = (pred > 0).sum() + (gt > 0).sum()
    return (2.0 * inter) / (union + 1e-6)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):  # 降低 smooth，讓稀少正樣本的梯度更強
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice

def calc_detection_rate(logits_batch, masks_batch, threshold=0.1):
    """
    Calculate Nodule Detection Rate (Sensitivity)
    Hit if IoU > threshold
    """
    batch_size = logits_batch.shape[0]
    hits = 0
    
    pred_batch = (torch.sigmoid(logits_batch) > 0.5).float()
    
    for i in range(batch_size):
        pred = pred_batch[i]
        mask = masks_batch[i]
        
        inter = (pred * mask).sum()
        union = pred.sum() + mask.sum() - inter
        
        if union > 0:
            iou = inter / union
            if iou > threshold:
                hits += 1
        elif mask.sum() == 0:
            # GT is empty
            if pred.sum() == 0:
                hits += 1 # Correctly predicted empty
        else:
             pass 

    return hits

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.95, gamma=2.0, reduction='mean'):  # alpha 提高到 0.95 增強正樣本權重
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        pt = torch.exp(-bce_loss)
        
        # Calculate alpha weighting
        # alpha_t = alpha where target=1, (1-alpha) where target=0
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class UNet3DTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.ckpt_dir.mkdir(exist_ok=True)
        
        # Model
        self.model = get_model(config).to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Losses
        # 1. 帶權重的 BCE Loss：降低 pos_weight 減少過度預測
        self.bce_weighted = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([150.0]).to(self.device)  # 從 500 降到 150
        )
        # 2. Focal Loss：alpha=0.90 稍微降低，平衡召回與精確
        self.focal = FocalLoss(alpha=0.90, gamma=2.0)
        # 3. Dice Loss：smooth=1e-5 讓稀少正樣本有更強梯度
        self.dice = DiceLoss(smooth=1e-5)
        
        # Datasets
        self.train_loader = self._create_loader("train", shuffle=True)
        self.val_loader = self._create_loader("val", shuffle=False)
        
        # Scheduler: 進一步降低 max_lr，穩定訓練
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.training.learning_rate * 3,  # 從 5x 降到 3x (max_lr = 3e-4)
            epochs=config.training.epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.1,  # 更快達到峰值，更多時間下降
            div_factor=3,   # 初始 lr = max_lr / 3
            final_div_factor=30
        )
        
        # Add File Handler to Logger
        file_handler = logging.FileHandler(self.output_dir / "training.log", encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        logger.addHandler(file_handler)
        logger.info(f"📝 Logging to file: {self.output_dir / 'training.log'}")
        
        self.best_val_score = 0.0
        
    def _create_loader(self, split: str, shuffle: bool):
        dataset = VolumetricDataset(
            npz_dir=self.config.data.npz_dir,
            split=split,
            image_size=self.config.model.image_size,
            max_depth=self.config.data.max_depth,
            augmentation=(split == 'train')
        )
        return DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            collate_fn=collate_video_batch,
            pin_memory=True
        )

    def train(self):
        logger.info(f"🚀 Starting training for {self.config.training.epochs} epochs")
        history = {'train_loss': [], 'val_dice': [], 'val_det_rate': []}
        best_dice = 0.0
        
        for epoch in range(1, self.config.training.epochs + 1):
            train_loss = self.train_epoch(epoch)
            # Validation
            val_dice, val_det_rate = self.validate(epoch)
            
            history['train_loss'].append(train_loss)
            history['val_dice'].append(val_dice)
            history['val_det_rate'].append(val_det_rate)
            
            logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Dice={val_dice:.4f}")
            
            # Plot metrics
            self.plot_metrics(history['train_loss'], history['val_dice'], history['val_det_rate'])
            
            # Save Best
            if val_dice > best_dice:
                best_dice = val_dice
                logger.info(f"🆕 New best model saved (Dice={best_dice:.4f})")
                torch.save(self.model.state_dict(), self.output_dir / "best_model.pth")
            
            # Save periodic
            if epoch % 5 == 0:
                self.save_checkpoint(epoch, val_dice)
                
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Train Ep {epoch}")
        
        for batch in pbar:
            images = batch['image'].to(self.device).float()
            masks = batch['mask'].to(self.device).float()
            
            self.optimizer.zero_grad()
            
            logits = self.model(images)
            if isinstance(logits, list):
                # Deep supervision
                loss = 0
                weights = [1.0, 0.5, 0.25, 0.125]  # Example weights
                for i, logit in enumerate(logits[:len(weights)]):
                    # Resize mask to logit size if needed
                    if logit.shape != masks.shape:
                         target = F.interpolate(masks, size=logit.shape[2:], mode='nearest')
                    else:
                        target = masks
                    
                    l_focal = self.focal(logit, target)
                    l_dice = self.dice(logit, target)
                    loss += weights[i] * (l_focal + 2.0 * l_dice)
            else:
                # 組合 Loss: BCE(weighted) + Focal + 2*Dice
                loss_bce = self.bce_weighted(logits, masks)
                loss_focal = self.focal(logits, masks)
                loss_dice = self.dice(logits, masks)
                # 提高 Dice 權重，讓模型更重視精確度而非只是召回
                loss = loss_bce + 0.5 * loss_focal + 2.0 * loss_dice
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'lr': self.scheduler.get_last_lr()[0]})
            
        return total_loss / len(self.train_loader)

    def validate(self, epoch: int) -> float:
        self.model.eval()
        total_dice = 0
        count = 0
        total_hits = 0
        total_nodules = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Val Ep {epoch}"):
                images = batch['image'].to(self.device).float()
                masks = batch['mask'].to(self.device).float()
                
                logits = self.model(images)
                if isinstance(logits, list):
                    logits = logits[0]
                
                # Dice score calculation
                pred = (torch.sigmoid(logits) > 0.5).float()
                
                if count == 0 and epoch % 1 == 0:
                    # Debug logging for first batch of epoch
                    logger.info(f"🔍 DEBUG [Val Ep {epoch}]")
                    logger.info(f"  Img: min={images.min():.4f}, max={images.max():.4f}, mean={images.mean():.4f}")
                    logger.info(f"  Msk: sum={masks.sum()}, unique={torch.unique(masks)}")
                    logger.info(f"  Logits: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")
                    logger.info(f"  Pred: sum={pred.sum()}, unique={torch.unique(pred)}")
                    
                    # Save debug image
                    try:
                        import matplotlib.pyplot as plt
                        import numpy as np
                        
                        # Take first sample
                        img_t = images[0, 0].cpu().numpy() # (D, H, W)
                        msk_t = masks[0, 0].cpu().numpy()
                        prob_t = torch.sigmoid(logits[0, 0]).cpu().numpy() # Probability map
                        
                        # Find slice with largest mask, or center
                        if msk_t.sum() > 0:
                            z_idx = np.argmax(msk_t.sum(axis=(1, 2)))
                        else:
                            z_idx = msk_t.shape[0] // 2
                            
                        plt.figure(figsize=(15, 5))
                        plt.subplot(1, 4, 1); plt.imshow(img_t[z_idx], cmap='gray'); plt.title(f'Image (z={z_idx})')
                        plt.subplot(1, 4, 2); plt.imshow(msk_t[z_idx], cmap='gray'); plt.title('GT Mask')
                        plt.subplot(1, 4, 3); plt.imshow(prob_t[z_idx], cmap='jet', vmin=0, vmax=1); plt.title(f'Prob Map') # Heatmap
                        plt.subplot(1, 4, 4); plt.hist(prob_t.flatten(), bins=50); plt.title('Prob Dist')
                        
                        plt.savefig(self.output_dir / f"debug_ep{epoch:03d}.png")
                        plt.close()
                        logger.info(f"  🖼️ Saved debug image to {self.output_dir / f'debug_ep{epoch:03d}.png'}")
                    except Exception as e:
                        logger.warning(f"  ❌ Failed to save debug image: {e}")

                inter = (pred * masks).sum()
                union = pred.sum() + masks.sum()
                dice = (2. * inter) / (union + 1e-6)
                
                total_dice += dice.item()
                
                # Detection Rate
                # 3D connected components
                batch_hits = calc_detection_rate(logits, masks)
                total_hits += batch_hits
                # Count samples that actually have nodules (gt > 0)
                # masks: (B, 1, D, H, W) -> Check if max value in each sample is 1
                total_nodules += (masks.amax(dim=(1,2,3,4)).sum().item())
                
                count += 1
        
        avg_dice = total_dice / count if count > 0 else 0.0
        det_rate = total_hits / (total_nodules + 1e-6) if total_nodules > 0 else 0.0
        
        logger.info(f"📊 Val Metrics: Dice={avg_dice:.4f}, Det_Rate={det_rate*100:.2f}%")
        return avg_dice, det_rate
            
    def plot_metrics(self, train_losses, val_dices, val_det_rates):
        try:
            import matplotlib.pyplot as plt
            epochs = range(1, len(train_losses) + 1)
            
            plt.figure(figsize=(15, 5))
            
            # Loss
            plt.subplot(1, 3, 1)
            plt.plot(epochs, train_losses, 'b-', label='Train Loss')
            plt.title('Training Loss')
            plt.xlabel('Epochs'); plt.ylabel('Loss')
            plt.grid(True)
            
            # Dice
            plt.subplot(1, 3, 2)
            plt.plot(epochs, val_dices, 'g-', label='Val Dice')
            plt.title('Validation Dice Score')
            plt.xlabel('Epochs'); plt.ylabel('Dice')
            plt.grid(True)
            
            # Det Rate
            plt.subplot(1, 3, 3)
            plt.plot(epochs, [d * 100 for d in val_det_rates], 'r-', label='Det Rate (%)')
            plt.title('Nodule Detection Rate')
            plt.xlabel('Epochs'); plt.ylabel('Rate (%)')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "metrics.png")
            plt.close()
        except Exception as e:
            logger.warning(f"Failed to plot metrics: {e}")

    def save_checkpoint(self, epoch: int, score: float, is_best: bool = False):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'score': score,
            'config': self.config.save(str(self.ckpt_dir / "config.json")) # save config is weird here, returns None
        }
        
        filename = f"checkpoint_ep{epoch}.pt"
        torch.save(state, self.ckpt_dir / filename)
        
        if is_best:
            torch.save(state, self.ckpt_dir / "best_model.pt")
            logger.info(f"🆕 New best model saved (Dice={score:.4f})")

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint"""
        logger.info(f"📥 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"✅ Checkpoint loaded (Epoch {checkpoint.get('epoch', '?')}, Score {checkpoint.get('score', 0.0):.4f})")

    def evaluate(self, split: str = 'test', use_postprocess: bool = True):
        """
        Evaluate on specific split with postprocessing and detection metrics
        
        Args:
            split: 'test', 'val', or 'train'
            use_postprocess: Whether to apply lung mask and postprocessing
        """
        logger.info(f"📊 Evaluating on {split} set (postprocess={use_postprocess})...")
        loader = self._create_loader(split, shuffle=False)
        self.model.eval()
        
        # Accumulators
        all_dice_raw = []
        all_dice_post = []
        total_tp, total_fp, total_fn = 0, 0, 0
        sample_results = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Eval {split}"):
                images = batch['image'].to(self.device).float()
                masks = batch['mask'].to(self.device).float()
                
                logits = self.model(images)
                if isinstance(logits, list):
                    logits = logits[0]
                
                probs = torch.sigmoid(logits)
                
                # Process each sample in batch
                batch_size = images.shape[0]
                for i in range(batch_size):
                    # Get numpy arrays
                    img_np = images[i, 0].cpu().numpy()  # (D, H, W)
                    mask_np = masks[i, 0].cpu().numpy()  # (D, H, W)
                    prob_np = probs[i, 0].cpu().numpy()  # (D, H, W)
                    
                    # Raw prediction (no postprocess)
                    pred_raw = (prob_np > 0.5).astype(np.uint8)
                    dice_raw = calc_dice_score(pred_raw, mask_np)
                    all_dice_raw.append(dice_raw)
                    
                    if use_postprocess:
                        # Generate lung mask from image
                        lung_mask = generate_lung_mask(img_np, threshold=0.35)
                        
                        # Apply postprocessing
                        pred_post = postprocess_prediction(
                            prob_np,
                            lung_mask=lung_mask,
                            threshold=0.5,
                            min_size_voxels=10,
                            apply_closing=True
                        )
                        
                        dice_post = calc_dice_score(pred_post, mask_np)
                        all_dice_post.append(dice_post)
                        
                        # Calculate detection metrics
                        metrics = calc_detection_metrics(pred_post, mask_np, iou_threshold=0.1)
                    else:
                        pred_post = pred_raw
                        all_dice_post.append(dice_raw)
                        metrics = calc_detection_metrics(pred_raw, mask_np, iou_threshold=0.1)
                    
                    total_tp += metrics['TP']
                    total_fp += metrics['FP']
                    total_fn += metrics['FN']
                    
                    sample_results.append({
                        'npz_path': batch['npz_path'][i] if 'npz_path' in batch else f'sample_{len(sample_results)}',
                        'dice_raw': dice_raw,
                        'dice_post': all_dice_post[-1],
                        'TP': metrics['TP'],
                        'FP': metrics['FP'],
                        'FN': metrics['FN']
                    })
        
        # Compute overall metrics
        avg_dice_raw = np.mean(all_dice_raw) if all_dice_raw else 0.0
        avg_dice_post = np.mean(all_dice_post) if all_dice_post else 0.0
        
        overall_precision = total_tp / (total_tp + total_fp + 1e-6)
        overall_recall = total_tp / (total_tp + total_fn + 1e-6)
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall + 1e-6)
        
        # Log results
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 {split.upper()} SET EVALUATION RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"📈 Segmentation Metrics:")
        logger.info(f"   • Dice (Raw):         {avg_dice_raw:.4f}")
        logger.info(f"   • Dice (Postprocess): {avg_dice_post:.4f}")
        logger.info(f"\n🎯 Detection Metrics (IoU≥0.1):")
        logger.info(f"   • True Positives:  {total_tp}")
        logger.info(f"   • False Positives: {total_fp}")
        logger.info(f"   • False Negatives: {total_fn}")
        logger.info(f"   • Precision:       {overall_precision:.4f}")
        logger.info(f"   • Recall:          {overall_recall:.4f}")
        logger.info(f"   • F1 Score:        {overall_f1:.4f}")
        logger.info(f"{'='*60}\n")
        
        # Return comprehensive results
        return {
            'dice_raw': avg_dice_raw,
            'dice_post': avg_dice_post,
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1,
            'TP': total_tp,
            'FP': total_fp,
            'FN': total_fn,
            'sample_results': sample_results
        }
