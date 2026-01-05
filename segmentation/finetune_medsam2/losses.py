#!/usr/bin/env python3
"""
損失函數模組
========================================

✅ 重構：優先使用 MedSAM2 原生損失函數

MedSAM2 原生函數 (from training.loss_fns):
- dice_loss: Dice Loss，類似 generalized IOU for masks
- sigmoid_focal_loss: Focal Loss，處理類別不平衡
- iou_loss: IoU prediction loss

本模組保留自訂擴充:
- TverskyLoss: 可調控 FP/FN 權重
- BoundaryLoss: 改善分割邊界
- EnhancedCombinedLoss: 結合多種損失
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# 導入 MedSAM2 原生損失函數
# ============================================================================

import os
import warnings
import multiprocessing

# 添加 MedSAM2 路徑
_medsam2_path = Path(__file__).parent.parent / "MedSAM2"
if str(_medsam2_path) not in sys.path:
    sys.path.insert(0, str(_medsam2_path))

def _is_main_process() -> bool:
    """檢查是否在主進程中（非 DataLoader worker）"""
    try:
        # 如果當前進程名稱包含 'SpawnProcess' 或 'ForkProcess'，則是 worker
        current = multiprocessing.current_process()
        return current.name == 'MainProcess'
    except:
        return True

try:
    # ✅ 使用 MedSAM2 原生損失函數
    from training.loss_fns import dice_loss as medsam2_dice_loss
    from training.loss_fns import sigmoid_focal_loss as medsam2_focal_loss
    from training.loss_fns import iou_loss as medsam2_iou_loss
    NATIVE_LOSS_AVAILABLE = True
except ImportError as e:
    NATIVE_LOSS_AVAILABLE = False
    medsam2_dice_loss = None
    medsam2_focal_loss = None
    medsam2_iou_loss = None
    # 只在主進程顯示警告
    if _is_main_process():
        warnings.warn(
            "MedSAM2 原生損失函數不可用（缺少依賴），使用自訂實作。效能無影響。",
            UserWarning,
            stacklevel=1
        )

class DiceLoss(nn.Module):
    """
    Dice Loss 用於分割任務
    
    ✅ 重構：內部使用 MedSAM2 原生 dice_loss（如果可用）
    
    Args:
        smooth: 平滑項，避免除零（僅用於 fallback 模式）
        use_native: 是否使用 MedSAM2 原生實作（預設 True）
    """
    
    def __init__(self, smooth: float = 1.0, use_native: bool = True):
        super().__init__()
        self.smooth = smooth
        self.use_native = use_native and NATIVE_LOSS_AVAILABLE
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        計算 Dice Loss
        
        Args:
            pred: 預測值 (sigmoid 前的 logits 或 sigmoid 後的機率值)
            target: 目標遮罩 (0或1)
            
        Returns:
            Dice Loss (1 - Dice 係數)
        """
        if self.use_native:
            # ✅ 使用 MedSAM2 原生 dice_loss
            # 注意：MedSAM2 原生函數有 bug - 只 flatten inputs 不 flatten targets
            # 所以我們需要預先 flatten 兩者到相同形狀
            
            # 確保至少 2D: [N, H*W]
            if pred.dim() == 2:  # [H, W]
                pred = pred.flatten().unsqueeze(0)  # [1, H*W]
                target = target.flatten().unsqueeze(0)  # [1, H*W]
            elif pred.dim() == 3:  # [N, H, W] or [1, H, W]
                N = pred.shape[0]
                pred = pred.view(N, -1)  # [N, H*W]
                target = target.view(N, -1)  # [N, H*W]
            elif pred.dim() == 4:  # [N, C, H, W]
                N = pred.shape[0]
                pred = pred.view(N, -1)  # [N, C*H*W]
                target = target.view(N, -1)  # [N, C*H*W]
            
            # 如果已經是 sigmoid 後的值，需要轉回 logits
            if pred.min() >= 0 and pred.max() <= 1:
                pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
                pred = torch.log(pred / (1 - pred))
            
            # 現在 pred 和 target 都是 [N, D]，原生函數會做：
            # inputs = inputs.sigmoid()  # [N, D]
            # inputs = inputs.flatten(1)  # [N, D] -> [N, D] (no change)
            # (inputs * targets).sum(1)  # now works!
            return medsam2_dice_loss(pred, target.float(), num_objects=1, loss_on_multimask=False)
        else:
            # Fallback: 自訂實作
            if pred.min() < 0 or pred.max() > 1:
                pred = torch.sigmoid(pred)
            
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
    
    - Dice Loss: 解決類別不平衡問題，直接優化分割重疊度
    - BCE Loss: 提供像素級別的穩定梯度，幫助小目標學習
    
    ✅ 重構：DiceLoss 內部使用 MedSAM2 原生實作
    
    Args:
        dice_weight: Dice Loss 的權重 (預設 0.5)
        bce_weight: BCE Loss 的權重 (預設 1.0)
    """
    
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(use_native=True)  # ✅ 使用原生
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        計算組合損失: L = dice_weight * L_Dice + bce_weight * L_BCE
        
        Args:
            pred: 模型輸出的 logits (未經 sigmoid)
            target: 目標遮罩 (0或1)
            
        Returns:
            加權組合的總損失
        """
        # DiceLoss 內部會處理 sigmoid
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        
        total_loss = self.dice_weight * dice + self.bce_weight * bce
        
        return total_loss


class FocalLoss(nn.Module):
    """
    Focal Loss - 解決極端類別不平衡問題
    
    ✅ 重構：內部使用 MedSAM2 原生 sigmoid_focal_loss（如果可用）
    
    Args:
        alpha: 類別權重
        gamma: 聚焦參數 (預設 2.0)
        use_native: 是否使用 MedSAM2 原生實作（預設 True）
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, use_native: bool = True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.use_native = use_native and NATIVE_LOSS_AVAILABLE
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        計算 Focal Loss
        
        Args:
            pred: 模型輸出的 logits
            target: 目標遮罩
            
        Returns:
            Focal Loss
        """
        if self.use_native:
            # ✅ 使用 MedSAM2 原生 sigmoid_focal_loss
            # 確保維度正確：[N, 1, H, W]
            if pred.dim() == 2:
                pred = pred.unsqueeze(0).unsqueeze(0)
                target = target.unsqueeze(0).unsqueeze(0)
            elif pred.dim() == 3:
                pred = pred.unsqueeze(1)
                target = target.unsqueeze(1)
            
            return medsam2_focal_loss(
                pred, target.float(), num_objects=1,
                alpha=self.alpha, gamma=self.gamma,
                loss_on_multimask=False
            )
        else:
            # Fallback: 自訂實作
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
    
    注意：MedSAM2 原生沒有 Tversky Loss，保留自訂實作
    
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
        """
        計算 Tversky Loss
        
        Args:
            pred: 模型輸出的 logits (未經 sigmoid)
            target: 目標遮罩 (0或1)
            
        Returns:
            Tversky Loss
        """
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
    
    ✅ 優化：調整權重以改善小型結節分割
    - 降低 dice_weight，增加 tversky_weight 以減少漏檢
    - 適當增加 boundary_weight 改善邊緣準確度
    
    Args:
        dice_weight: Dice Loss 權重
        focal_weight: Focal Loss 權重
        tversky_weight: Tversky Loss 權重
        boundary_weight: Boundary Loss 權重
    """
    
    def __init__(
        self, 
        dice_weight: float = 0.4,     # ✅ 從 0.5 降低到 0.4
        focal_weight: float = 0.2,
        tversky_weight: float = 0.3,  # ✅ 從 0.2 提高到 0.3（減少漏檢）
        boundary_weight: float = 0.1
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        self.boundary_weight = boundary_weight
        
        # ✅ 使用 MedSAM2 原生損失（如果可用）
        self.dice_loss = DiceLoss(use_native=True)
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0, use_native=True)
        self.tversky_loss = TverskyLoss(alpha=0.7, beta=0.3)
        self.boundary_loss = BoundaryLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        計算增強版組合損失
        
        ✅ 重構：Dice 和 Focal 使用 MedSAM2 原生實作
        
        Args:
            pred: 模型輸出的 logits (未經 sigmoid)
            target: 目標遮罩 (0或1)
            
        Returns:
            加權組合的總損失
        """
        # DiceLoss 和 FocalLoss 現在內部會處理 sigmoid
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        tversky = self.tversky_loss(pred, target)
        boundary = self.boundary_loss(pred, target)
        
        total = (self.dice_weight * dice + 
                 self.focal_weight * focal + 
                 self.tversky_weight * tversky +
                 self.boundary_weight * boundary)
        
        return total


# ============================================================================
# MedSAM2 原生損失類別包裝器（供進階使用）
# ============================================================================

class MedSAM2NativeLoss(nn.Module):
    """
    完全使用 MedSAM2 原生損失函數的包裝類別
    
    結合 dice_loss + sigmoid_focal_loss，與 MedSAM2 原生訓練保持一致
    
    Args:
        dice_weight: Dice Loss 權重
        focal_weight: Focal Loss 權重
        focal_alpha: Focal Loss alpha 參數
        focal_gamma: Focal Loss gamma 參數
    """
    
    def __init__(
        self,
        dice_weight: float = 1.0,
        focal_weight: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        if not NATIVE_LOSS_AVAILABLE:
            raise RuntimeError(
                "MedSAM2 原生損失函數不可用。"
                "請確保 MedSAM2 安裝正確且 training.loss_fns 可導入。"
            )
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        計算 MedSAM2 原生風格損失
        
        Args:
            pred: 模型輸出的 logits [N, 1, H, W] 或 [H, W]
            target: 目標遮罩 [N, 1, H, W] 或 [H, W]
            
        Returns:
            加權組合損失
        """
        # 確保維度正確
        if pred.dim() == 2:
            pred = pred.unsqueeze(0).unsqueeze(0)
            target = target.unsqueeze(0).unsqueeze(0)
        elif pred.dim() == 3:
            pred = pred.unsqueeze(1)
            target = target.unsqueeze(1)
        
        target = target.float()
        
        dice = medsam2_dice_loss(pred, target, num_objects=1, loss_on_multimask=False)
        focal = medsam2_focal_loss(
            pred, target, num_objects=1,
            alpha=self.focal_alpha, gamma=self.focal_gamma,
            loss_on_multimask=False
        )
        
        return self.dice_weight * dice + self.focal_weight * focal
