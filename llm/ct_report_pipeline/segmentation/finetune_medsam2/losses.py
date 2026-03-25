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
    
    用於穩定地分割小型目標物件
    
    公式: L = 0.5 * L_Dice + L_BCE
    
    - Dice Loss: 解決類別不平衡問題，直接優化分割重疊度
    - BCE Loss: 提供像素級別的穩定梯度，幫助小目標學習
    
    參考: 結合兩種損失可以在保持 Dice 優化的同時，
         利用 BCE 的穩定梯度來改善小型結節的分割效果
    
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
        計算組合損失: L = 0.5 * L_Dice + L_BCE
        
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
        
        # 加權組合: L = 0.5 * L_Dice + L_BCE
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
    
    def __init__(self, theta0: float = 3, theta: float = 5):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_sigmoid = torch.sigmoid(pred)
        
        # 計算邊界區域
        # 使用形態學操作找到邊界
        n, c, h, w = pred.shape if len(pred.shape) == 4 else (1, 1, pred.shape[0], pred.shape[1])
        
        # 簡化版：使用梯度來找邊界
        pred_view = pred_sigmoid.view(n, 1, h, w) if len(pred.shape) < 4 else pred_sigmoid
        target_view = target.view(n, 1, h, w) if len(target.shape) < 4 else target
        
        # Sobel-like 邊界偵測
        pred_boundary = self._get_boundary(pred_view)
        target_boundary = self._get_boundary(target_view)
        
        # 邊界區域的 BCE Loss
        boundary_weight = target_boundary + 1.0  # 邊界區域權重更高
        weighted_bce = F.binary_cross_entropy(pred_sigmoid, target, reduction='none')
        weighted_bce = (weighted_bce * boundary_weight.view_as(weighted_bce)).mean()
        
        return weighted_bce
    
    def _get_boundary(self, mask: torch.Tensor) -> torch.Tensor:
        """使用最大池化來找邊界"""
        # 膨脹
        dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
        # 腐蝕
        eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
        # 邊界 = 膨脹 - 腐蝕
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
