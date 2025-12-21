#!/usr/bin/env python3
"""
UNet++ 肺結節分割訓練 - 損失函數模組
===================================

提供多種損失函數：
1. Dice Loss（基本分割損失）
2. Focal BCE（解決類別不平衡）
3. Tversky Loss（可調 FN/FP 權重）
4. Soft Dice Loss（支援軟標籤）
5. Combined Loss（組合損失）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss"""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        計算 Dice Loss
        
        Args:
            pred: 預測值（sigmoid 後的機率）
            target: 目標遮罩
            
        Returns:
            Dice Loss (1 - Dice Score)
        """
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        
        return 1 - dice


class SoftDiceLoss(nn.Module):
    """
    Soft Dice Loss - 支援軟標籤（0-1 連續值）
    
    用於多醫師共識標註
    """
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        計算 Soft Dice Loss
        
        Args:
            pred: 預測值 logits
            target: 軟標籤（0-1 連續值）
        """
        pred = torch.sigmoid(pred)
        
        # 處理每個樣本
        batch_size = pred.size(0)
        
        pred_flat = pred.view(batch_size, -1)
        target_flat = target.view(batch_size, -1)
        
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss - 解決極端類別不平衡
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        計算 Focal Loss
        
        Args:
            pred: 預測值 logits
            target: 目標遮罩
        """
        # BCE with logits
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # 計算 p_t
        pred_prob = torch.sigmoid(pred)
        p_t = target * pred_prob + (1 - target) * (1 - pred_prob)
        
        # 計算 focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # 計算 alpha weight
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
        
        focal_loss = alpha_t * focal_weight * bce
        
        return focal_loss.mean()


class TverskyLoss(nn.Module):
    """
    Tversky Loss - 可調控 FP/FN 權重
    
    當 α = β = 0.5 時等同於 Dice Loss
    α > β 時更注重減少 FN（提高 Recall）
    """
    
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha  # FN 權重
        self.beta = beta    # FP 權重
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        計算 Tversky Loss
        """
        pred = torch.sigmoid(pred)
        
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # True Positive
        tp = (pred_flat * target_flat).sum()
        # False Positive
        fp = ((1 - target_flat) * pred_flat).sum()
        # False Negative
        fn = (target_flat * (1 - pred_flat)).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        
        return 1 - tversky


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss
    
    結合 Focal 和 Tversky 的優點，對小目標更有效
    """
    
    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 0.75,
        smooth: float = 1.0
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        tp = (pred_flat * target_flat).sum()
        fp = ((1 - target_flat) * pred_flat).sum()
        fn = (target_flat * (1 - pred_flat)).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        
        # Focal 調製
        focal_tversky = (1 - tversky) ** self.gamma
        
        return focal_tversky


class BoundaryLoss(nn.Module):
    """
    Boundary Loss - 改善分割邊界
    """
    
    def __init__(self, theta0: float = 3, theta: float = 5):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        
        # 計算邊界
        pred_boundary = self._get_boundary(pred)
        target_boundary = self._get_boundary(target)
        
        # 邊界損失
        loss = F.binary_cross_entropy(pred_boundary, target_boundary)
        
        return loss
    
    def _get_boundary(self, mask: torch.Tensor) -> torch.Tensor:
        """使用最大池化找邊界"""
        # 膨脹
        dilated = F.max_pool2d(
            mask,
            kernel_size=self.theta,
            stride=1,
            padding=self.theta // 2
        )
        # 腐蝕
        eroded = -F.max_pool2d(
            -mask,
            kernel_size=self.theta0,
            stride=1,
            padding=self.theta0 // 2
        )
        
        boundary = dilated - eroded
        return boundary


class CombinedLoss(nn.Module):
    """
    組合損失函數
    
    預設組合：Dice + Focal BCE
    """
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_weight: float = 0.3,
        tversky_weight: float = 0.2,
        boundary_weight: float = 0.0,
        use_soft_dice: bool = True,
        tversky_alpha: float = 0.7,
        tversky_beta: float = 0.3
    ):
        super().__init__()
        
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        self.boundary_weight = boundary_weight
        
        if use_soft_dice:
            self.dice_loss = SoftDiceLoss()
        else:
            self.dice_loss = DiceLoss()
        
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.tversky_loss = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        
        if boundary_weight > 0:
            self.boundary_loss = BoundaryLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        計算組合損失
        """
        total_loss = 0.0
        
        if self.dice_weight > 0:
            total_loss += self.dice_weight * self.dice_loss(pred, target)
        
        if self.focal_weight > 0:
            total_loss += self.focal_weight * self.focal_loss(pred, target)
        
        if self.tversky_weight > 0:
            total_loss += self.tversky_weight * self.tversky_loss(pred, target)
        
        if self.boundary_weight > 0:
            total_loss += self.boundary_weight * self.boundary_loss(pred, target)
        
        return total_loss


def get_loss_function(config) -> nn.Module:
    """
    根據配置獲取損失函數
    
    Args:
        config: 配置物件
        
    Returns:
        損失函數
    """
    loss_type = config.training.loss_type
    
    if loss_type == "dice":
        return SoftDiceLoss()
    elif loss_type == "focal":
        return FocalLoss()
    elif loss_type == "tversky":
        return TverskyLoss(
            alpha=config.training.tversky_alpha,
            beta=config.training.tversky_beta
        )
    elif loss_type == "combined":
        return CombinedLoss(
            dice_weight=config.training.dice_weight,
            focal_weight=config.training.focal_weight,
            tversky_weight=config.training.tversky_weight,
            tversky_alpha=config.training.tversky_alpha,
            tversky_beta=config.training.tversky_beta,
            use_soft_dice=True
        )
    else:
        raise ValueError(f"未知的損失函數類型: {loss_type}")


if __name__ == "__main__":
    # 測試損失函數
    pred = torch.randn(4, 1, 160, 160)
    target = torch.randint(0, 2, (4, 1, 160, 160)).float()
    
    # 軟標籤測試
    soft_target = torch.rand(4, 1, 160, 160)
    
    print("損失函數測試:")
    
    dice_loss = DiceLoss()
    print(f"Dice Loss: {dice_loss(pred, target):.4f}")
    
    soft_dice_loss = SoftDiceLoss()
    print(f"Soft Dice Loss: {soft_dice_loss(pred, soft_target):.4f}")
    
    focal_loss = FocalLoss()
    print(f"Focal Loss: {focal_loss(pred, target):.4f}")
    
    tversky_loss = TverskyLoss()
    print(f"Tversky Loss: {tversky_loss(pred, target):.4f}")
    
    combined_loss = CombinedLoss()
    print(f"Combined Loss: {combined_loss(pred, target):.4f}")
