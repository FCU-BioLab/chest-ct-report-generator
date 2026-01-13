#!/usr/bin/env python3
"""
UNet++ 肺結節分割訓練 - 損失函數模組
===================================

提供損失函數：
1. BCEDiceLoss - Val/Test 評估用 (BCE + Dice)
2. DiceFocalLoss - Training 用 (Dice + Focal, MedSAM2 風格)
3. AdaptiveLoss - 舊版 Training 用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    """
    BCE + Dice Loss
    
    用於 Validation/Testing 的 patch-level 和 lesion-level loss
    """
    
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 1.0, eps: float = 1e-5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.eps = eps
    
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # BCE
        bce_loss = F.binary_cross_entropy_with_logits(logits, target)
        
        # Dice
        probs = torch.sigmoid(logits)
        probs_flat = probs.contiguous().view(probs.size(0), -1)
        target_flat = target.contiguous().view(target.size(0), -1)
        
        inter = (probs_flat * target_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + target_flat.sum(dim=1)
        dice = (2 * inter + self.eps) / (union + self.eps)
        dice_loss = 1 - dice.mean()
        
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


class DiceFocalLoss(nn.Module):
    """
    Dice + Focal Loss (MedSAM2 風格)
    
    結合 Dice Loss (優化重疊度) 和 Focal Loss (解決類別不平衡)
    比 Tversky 更適合無 Prompt 的自動分割任務
    """
    
    def __init__(
        self,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        smooth: float = 1e-5
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.smooth = smooth
    
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # === Dice Loss ===
        probs = torch.sigmoid(logits)
        probs_flat = probs.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        
        intersection = (probs_flat * target_flat).sum()
        union = probs_flat.sum() + target_flat.sum()
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score
        
        # === Focal Loss (Binary) ===
        # 使用 BCEWithLogitsLoss 的 reduction='none' 來手動計算 focal term
        bce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        pt = torch.exp(-bce_loss)  # pt is the probability of being classified correctly
        focal_loss = (self.focal_alpha * (1-pt)**self.focal_gamma * bce_loss).mean()
        
        return self.dice_weight * dice_loss + self.focal_weight * focal_loss


class AdaptiveLoss(nn.Module):
    """
    自適應損失函數（Training 用）
    
    根據 GT area 選擇不同的 loss：
    - GT area == 0 (負樣本): Focal BCE（避免背景主導）
    - GT area > 0 (正樣本): BCE + Dice（優化分割品質）
    """
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        eps: float = 1e-6
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.eps = eps
    
    def _focal_bce(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Focal BCE for negative samples"""
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        pred_prob = torch.sigmoid(logits)
        p_t = target * pred_prob + (1 - target) * (1 - pred_prob)
        focal_weight = (1 - p_t) ** self.focal_gamma
        alpha_t = target * self.focal_alpha + (1 - target) * (1 - self.focal_alpha)
        return (alpha_t * focal_weight * bce).mean()
    
    def _bce_dice(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """BCE + Dice for positive samples"""
        bce_loss = F.binary_cross_entropy_with_logits(logits, target)
        
        probs = torch.sigmoid(logits)
        probs_flat = probs.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        
        inter = (probs_flat * target_flat).sum()
        union = probs_flat.sum() + target_flat.sum()
        dice = (2 * inter + self.eps) / (union + self.eps)
        dice_loss = 1 - dice
        
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss
    
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size = logits.size(0)
        total_loss = 0.0
        
        for i in range(batch_size):
            gt_area = target[i].sum()
            
            if gt_area == 0:
                loss_i = self._focal_bce(logits[i:i+1], target[i:i+1])
            else:
                loss_i = self._bce_dice(logits[i:i+1], target[i:i+1])
            
            total_loss += loss_i
        
        return total_loss / batch_size


class TverskyLoss(nn.Module):
    """
    Tversky Loss - 可調控 FP 和 FN 的權重
    """
    
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        
        TP = (probs_flat * target_flat).sum()
        FP = ((1 - target_flat) * probs_flat).sum()
        FN = (target_flat * (1 - probs_flat)).sum()
        
        tversky = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth)
        
        return 1 - tversky


def get_loss_function(config) -> nn.Module:
    """
    根據配置獲取損失函數
    """
    loss_type = config.training.loss_type
    dice_weight = config.training.dice_weight
    
    if loss_type == "dice_focal":
        return DiceFocalLoss(dice_weight=1.0, focal_weight=1.0)
    elif loss_type == "adaptive":
        return AdaptiveLoss(dice_weight=dice_weight)
    elif loss_type == "bce_dice":
        return BCEDiceLoss(dice_weight=dice_weight, bce_weight=1.0)
    elif loss_type == "tversky":
        return TverskyLoss(alpha=0.7, beta=0.3)
    else:
        # Default fallback
        return DiceFocalLoss(dice_weight=1.0, focal_weight=1.0)


if __name__ == "__main__":
    # 測試損失函數
    pred = torch.randn(4, 1, 160, 160)
    target = torch.randint(0, 2, (4, 1, 160, 160)).float()
    
    print("損失函數測試:")
    
    dice_focal = DiceFocalLoss()
    print(f"DiceFocalLoss: {dice_focal(pred, target):.4f}")
    
    bce_dice = BCEDiceLoss()
    print(f"BCEDiceLoss: {bce_dice(pred, target):.4f}")
