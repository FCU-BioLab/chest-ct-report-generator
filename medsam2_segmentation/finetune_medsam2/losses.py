#!/usr/bin/env python3
"""
損失函數模組
提供 Dice Loss 和組合損失函數
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    
    適用於醫學影像分割，特別是類別不平衡的情況
    
    Args:
        dice_weight: Dice Loss 的權重 (預設 0.8，強調重疊度)
        bce_weight: BCE Loss 的權重 (預設 0.2，輔助訓練)
    """
    
    def __init__(self, dice_weight: float = 0.8, bce_weight: float = 0.2):
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
        # 應用 sigmoid 到預測值用於 Dice Loss
        pred_sigmoid = torch.sigmoid(pred)
        
        # 計算兩種損失
        dice = self.dice_loss(pred_sigmoid, target)
        bce = self.bce_loss(pred, target)  # BCE Loss 內部會處理 sigmoid
        
        # 加權組合
        total_loss = self.dice_weight * dice + self.bce_weight * bce
        
        return total_loss


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
        """
        計算 Focal Loss
        
        Args:
            pred: 模型輸出的 logits
            target: 目標遮罩
            
        Returns:
            Focal Loss
        """
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pred_prob = torch.sigmoid(pred)
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss
        
        return focal_loss.mean()
