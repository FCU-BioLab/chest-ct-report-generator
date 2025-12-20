#!/usr/bin/env python3
"""
損失函數模組
提供 Dice Loss、BCE Loss 和組合損失函數
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class DiceLoss(nn.Module):
    """
    Dice Loss 用於分割任務
    
    Args:
        smooth: 平滑項，避免除零
    """
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        計算 Dice Loss
        
        Args:
            pred: 預測值 (sigmoid後的機率值)
            target: 目標遮罩 (0或1)
            
        Returns:
            Dice Loss (1 - Dice係數)
        """
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """
    結合 Dice Loss 和 Binary Cross Entropy Loss
    
    用於穩定地分割小型目標物件
    
    公式: L = dice_weight * L_Dice + bce_weight * L_BCE
    
    Args:
        dice_weight: Dice Loss 的權重 (預設 0.5)
        bce_weight: BCE Loss 的權重 (預設 1.0)
    """
    
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        計算組合損失
        
        Args:
            pred: 模型輸出的 logits (未經 sigmoid)
            target: 目標遮罩 (0或1)
            
        Returns:
            加權組合的總損失
        """
        pred_sigmoid = torch.sigmoid(pred)
        
        dice = self.dice_loss(pred_sigmoid, target)
        bce = self.bce_loss(pred, target)
        
        total_loss = self.dice_weight * dice + self.bce_weight * bce
        
        return total_loss


class StableCombinedLoss(nn.Module):
    """
    [Anti-Collapse] Stable Combined Loss with configurable weights and pos_weight.
    
    Key features for preventing model collapse:
    1. BCE with pos_weight: Upweight positive class to combat imbalance
    2. Higher Dice weight: Default 0.7 Dice + 0.3 BCE prevents BCE domination
    3. Per-sample Dice: More stable gradient flow
    
    Args:
        bce_weight: BCE Loss weight (default 0.3)
        dice_weight: Dice Loss weight (default 0.7)
        pos_weight: Weight for positive class in BCE (default 10.0)
        eps: Numerical stability term
    """
    
    def __init__(
        self, 
        bce_weight: float = 0.3,   # Reduced from 1.0
        dice_weight: float = 0.7,  # Increased from 1.0
        pos_weight: float = 10.0,  # NEW: upweight positive pixels
        eps: float = 1e-6
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight = pos_weight
        self.eps = eps
        
        # Weighted BCE - pos_weight as tensor (will be moved to device in forward)
        self._pos_weight_value = pos_weight
        self.bce = None  # Created lazily with correct device
    
    def _get_bce(self, device: torch.device) -> nn.BCEWithLogitsLoss:
        """Lazy creation of BCE with pos_weight on correct device."""
        if self.bce is None or self.bce.pos_weight.device != device:
            pos_weight_tensor = torch.tensor([self._pos_weight_value], device=device)
            self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        return self.bce
    
    def soft_dice_loss(self, prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Per-sample Soft Dice Loss.
        
        Args:
            prob: sigmoid(logits), shape (B, 1, H, W)
            target: binary mask, shape (B, 1, H, W)
        """
        prob = prob.float()
        target = target.float()
        
        # Per-sample Dice (sum over H, W dimensions)
        dims = (2, 3)  # (B, C, H, W) -> sum over H, W
        intersection = (prob * target).sum(dims)
        union = prob.sum(dims) + target.sum(dims)
        dice = (2 * intersection + self.eps) / (union + self.eps)
        
        return 1 - dice.mean()
    
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model output (pre-sigmoid), shape (B, 1, H, W)
            target: Binary mask (0/1), shape (B, 1, H, W)
        """
        target = target.float()
        
        # Weighted BCE: logits directly, pos_weight upweights positive pixels
        bce_fn = self._get_bce(logits.device)
        bce_loss = bce_fn(logits, target)
        
        # Dice: requires sigmoid probabilities
        prob = torch.sigmoid(logits)
        dice_loss = self.soft_dice_loss(prob, target)
        
        total = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        
        return total


class FocalLoss(nn.Module):
    """
    Focal Loss - 解決極端類別不平衡問題
    
    Args:
        alpha: 類別權重
        gamma: 聚焦參數 (預設 2.0)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pred_prob = torch.sigmoid(pred)
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss
        
        return focal_loss.mean()


class TverskyLoss(nn.Module):
    """
    Tversky Loss - 可調控 FP 和 FN 的權重
    
    當 alpha=beta=0.5 時等同於 Dice Loss
    alpha > beta 時更注重減少 False Negatives (提高 Recall)
    alpha < beta 時更注重減少 False Positives (提高 Precision)
    
    Args:
        alpha: FN 權重 (預設 0.7，強調減少漏檢)
        beta: FP 權重 (預設 0.3)
        smooth: 平滑項
    """
    
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        TP = (pred * target).sum()
        FP = ((1 - target) * pred).sum()
        FN = (target * (1 - pred)).sum()
        
        tversky = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth)
        
        return 1 - tversky


class BoundaryLoss(nn.Module):
    """
    Boundary Loss - 改善分割邊界
    
    通過計算邊界區域的損失來提高邊緣準確度
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_sigmoid = torch.sigmoid(pred)
        
        # 確保正確的維度
        if len(pred.shape) == 3:
            pred_sigmoid = pred_sigmoid.unsqueeze(1)
            target = target.unsqueeze(1)
        
        # 計算邊界
        pred_boundary = self._get_boundary(pred_sigmoid)
        target_boundary = self._get_boundary(target)
        
        # 邊界區域的 BCE Loss
        boundary_weight = target_boundary + 1.0
        weighted_bce = F.binary_cross_entropy(
            pred_sigmoid.view_as(target), 
            target, 
            reduction='none'
        )
        weighted_bce = (weighted_bce * boundary_weight).mean()
        
        return weighted_bce
    
    def _get_boundary(self, mask: torch.Tensor) -> torch.Tensor:
        """使用最大池化來找邊界"""
        dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
        boundary = dilated - eroded
        return boundary


class EnhancedCombinedLoss(nn.Module):
    """
    增強版組合損失
    
    結合 Dice、Focal、Tversky 和 Boundary Loss
    專為提升 DSC 到 0.9+ 設計
    
    Args:
        dice_weight: Dice Loss 權重
        focal_weight: Focal Loss 權重
        tversky_weight: Tversky Loss 權重
        boundary_weight: Boundary Loss 權重
    """
    
    def __init__(
        self, 
        dice_weight: float = 0.5,
        focal_weight: float = 0.2,
        tversky_weight: float = 0.2,
        boundary_weight: float = 0.1
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        self.boundary_weight = boundary_weight
        
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.tversky_loss = TverskyLoss(alpha=0.7, beta=0.3)
        self.boundary_loss = BoundaryLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_sigmoid = torch.sigmoid(pred)
        
        dice = self.dice_loss(pred_sigmoid, target)
        focal = self.focal_loss(pred, target)
        tversky = self.tversky_loss(pred, target)
        boundary = self.boundary_loss(pred, target)
        
        total = (self.dice_weight * dice + 
                 self.focal_weight * focal + 
                 self.tversky_weight * tversky +
                 self.boundary_weight * boundary)
        
        return total


class DeepSupervisionLoss(nn.Module):
    """
    深度監督損失
    
    用於 UNet++ 的深度監督訓練，對多個輸出層施加損失
    
    Args:
        base_loss: 基礎損失函數
        weights: 各層的損失權重
    """
    
    def __init__(
        self, 
        base_loss: nn.Module = None,
        weights: Optional[List[float]] = None
    ):
        super().__init__()
        self.base_loss = base_loss if base_loss else CombinedLoss()
        self.weights = weights if weights else [0.5, 0.5, 0.75, 1.0]
    
    def forward(
        self, 
        outputs: List[torch.Tensor], 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        計算深度監督損失
        
        Args:
            outputs: 模型多層輸出的列表
            target: 目標遮罩
        
        Returns:
            加權總損失
        """
        total_loss = 0.0
        
        for i, (output, weight) in enumerate(zip(outputs, self.weights)):
            # 確保輸出和目標尺寸相同
            if output.shape[2:] != target.shape[2:]:
                output = F.interpolate(
                    output, 
                    size=target.shape[2:], 
                    mode='bilinear', 
                    align_corners=True
                )
            
            loss = self.base_loss(output, target)
            total_loss += weight * loss
        
        return total_loss / sum(self.weights)


class FocalDiceLoss(nn.Module):
    """
    [Anti-Collapse] Focal + Dice Loss - Best for extreme class imbalance.
    
    Focal Loss reduces contribution from easy negatives (background pixels),
    while Dice Loss directly optimizes the overlap metric.
    
    This combination is highly effective for:
    - Small lesion segmentation
    - High background-to-foreground ratio
    - Preventing model collapse to all-background prediction
    
    Args:
        focal_weight: Focal Loss weight (default 0.5)
        dice_weight: Dice Loss weight (default 0.5)
        alpha: Focal Loss class weight (default 0.75, favors positive class)
        gamma: Focal Loss focusing parameter (default 2.0)
        eps: Numerical stability term
    """
    
    def __init__(
        self,
        focal_weight: float = 0.5,
        dice_weight: float = 0.5,
        alpha: float = 0.75,  # Higher alpha favors positive class
        gamma: float = 2.0,
        eps: float = 1e-6
    ):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
    
    def focal_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Focal Loss with per-pixel weighting."""
        bce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        prob = torch.sigmoid(logits)
        
        # p_t: probability of correct class
        p_t = prob * target + (1 - prob) * (1 - target)
        
        # alpha_t: class weight
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        # Focal weight: (1 - p_t)^gamma reduces easy examples
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        loss = focal_weight * bce_loss
        return loss.mean()
    
    def soft_dice_loss(self, prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-sample Soft Dice Loss."""
        prob = prob.float()
        target = target.float()
        
        # Sum over H, W
        dims = (2, 3)
        intersection = (prob * target).sum(dims)
        union = prob.sum(dims) + target.sum(dims)
        dice = (2 * intersection + self.eps) / (union + self.eps)
        
        return 1 - dice.mean()
    
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model output (pre-sigmoid), shape (B, 1, H, W)
            target: Binary mask (0/1), shape (B, 1, H, W)
        """
        target = target.float()
        
        focal = self.focal_loss(logits, target)
        prob = torch.sigmoid(logits)
        dice = self.soft_dice_loss(prob, target)
        
        total = self.focal_weight * focal + self.dice_weight * dice
        
        return total


def get_loss_function(
    loss_type: str = "stable",
    **kwargs
) -> nn.Module:
    """
    Factory function for creating loss functions.
    
    Available types:
        - "dice": Simple Dice Loss
        - "combined": BCE + Dice (old defaults)
        - "stable": [Anti-Collapse] Weighted BCE (pos_weight=10) + Dice (0.3 + 0.7)
        - "focal_dice": [Anti-Collapse] Focal + Dice (best for extreme imbalance)
        - "enhanced": Dice + Focal + Tversky + Boundary
        - "tversky": Tversky Loss (FP/FN tradeoff)
        - "focal": Focal Loss only
        - "deep_supervision": Wrapper for multi-output models
    
    Args:
        loss_type: Loss function type
        **kwargs: Loss function parameters
    
    Returns:
        Loss function instance
    """
    if loss_type == "dice":
        return DiceLoss(**kwargs)
    elif loss_type == "combined":
        return CombinedLoss(**kwargs)
    elif loss_type == "stable":
        # [Anti-Collapse] Weighted BCE + Dice with better defaults
        return StableCombinedLoss(**kwargs)
    elif loss_type == "focal_dice":
        # [Anti-Collapse] Focal + Dice - best for extreme imbalance
        return FocalDiceLoss(**kwargs)
    elif loss_type == "enhanced":
        return EnhancedCombinedLoss(**kwargs)
    elif loss_type == "tversky":
        return TverskyLoss(**kwargs)
    elif loss_type == "focal":
        return FocalLoss(**kwargs)
    elif loss_type == "deep_supervision":
        base_loss = kwargs.pop('base_loss', FocalDiceLoss())  # Better default
        return DeepSupervisionLoss(base_loss=base_loss, **kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. "
                        f"Available: dice, combined, stable, focal_dice, enhanced, tversky, focal, deep_supervision")


if __name__ == "__main__":
    # 測試損失函數
    print("Testing loss functions...")
    
    pred = torch.randn(4, 1, 256, 256)
    target = (torch.rand(4, 1, 256, 256) > 0.5).float()
    
    # 測試各種損失
    losses = {
        "Dice": DiceLoss(),
        "Combined": CombinedLoss(),
        "Enhanced": EnhancedCombinedLoss(),
        "Focal": FocalLoss(),
        "Tversky": TverskyLoss(),
    }
    
    for name, loss_fn in losses.items():
        loss = loss_fn(pred, target)
        print(f"{name} Loss: {loss.item():.4f}")
    
    # 測試深度監督損失
    outputs = [torch.randn(4, 1, 256, 256) for _ in range(4)]
    ds_loss = DeepSupervisionLoss()
    loss = ds_loss(outputs, target)
    print(f"Deep Supervision Loss: {loss.item():.4f}")
    
    print("\nAll loss function tests passed!")
