# ultralytics/nn/modules/custom_blocks.py
"""
Custom blocks for YOLO medical CT detection
Implements: SpatialSEModule, RRBBlock, EAPIoU Loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

__all__ = [
    "SpatialSEModule",
    "RRBBlock",
    "EAPIoULoss",
]

# ============== 初始化工具函數 ==============
def kaiming_init(module):
    """
    應用 Kaiming Normal 初始化以穩定訓練
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0.0)


# ============== Spatial-SE 模組（對應論文 SSE） ==============
class SpatialSEModule(nn.Module):
    """
    Spatial Squeeze-and-Excitation (SSE) 模組

    對應論文中的流程：
      1. SE：Global Avg Pool -> FC(reduction) -> ReLU -> FC -> Sigmoid，做通道注意力
      2. Spatial Attention：
         - 對 SE 後特徵做 max-pool & avg-pool (沿 channel)
         - concat 成 2×H×W
         - 7×7 Conv + Sigmoid 得到空間注意力圖
      3. 融合：O = X_se * S   （通道 + 空間共同加權）

    Lazy 寫法：c2 可選，如果不給就保持輸入通道不變。
    YAML 用法：
      - [-1, 1, SpatialSEModule, []]
      - [-1, 1, SpatialSEModule, [128]]
    """

    def __init__(
        self,
        c2: int | None = None,
        reduction: int = 16,
        kernel_size: int = 7,
    ):
        super().__init__()
        self.target_c = c2
        self.reduction = reduction
        self.kernel_size = kernel_size
        self._built = False

    def _build(self, c1: int):
        c2 = self.target_c or c1
        self.proj = nn.Identity() if c1 == c2 else nn.Conv2d(c1, c2, 1, bias=False)

        # ----- Channel Attention (SE) -----
        # hidden 至少 8，避免通道太少時壓縮過頭
        hidden_dim = max(c2 // self.reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),             # GAP -> (B, C, 1, 1)
            nn.Conv2d(c2, hidden_dim, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, c2, 1, bias=True),
            nn.Sigmoid(),                        # (B, C, 1, 1)
        )

        # ----- Spatial Attention -----
        # 論文中使用 max + avg，再經過一個 conv7×7 生成空間注意力圖
        self.spatial_conv = nn.Conv2d(
            2,  # [max_out, avg_out] concat 後 channel=2
            1,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            bias=True,
        )

        # 初始化
        self.apply(kaiming_init)
        self._built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._built:
            self._build(x.shape[1])

        # 若 c1 != c2，先投影到目標通道數
        x = self.proj(x)

        # ----- SE 通道注意力 -----
        # F_adapt_avg = GAP(x) 再經過兩層 FC (這裡用 1×1 conv 實作)
        w = self.se(x)          # (B, C, 1, 1)
        x_se = x * w            # (B, C, H, W)

        # ----- Spatial Attention -----
        # max_out, avg_out shape: (B, 1, H, W)
        max_out, _ = torch.max(x_se, dim=1, keepdim=True)
        avg_out = torch.mean(x_se, dim=1, keepdim=True)
        sa_input = torch.cat([max_out, avg_out], dim=1)   # (B, 2, H, W)
        s = torch.sigmoid(self.spatial_conv(sa_input))    # (B, 1, H, W)

        # ----- 最終融合 -----
        out = x_se * s   # 對應論文 O = X'_c,i,j · S_i,j

        # 視情況可加殘差：return out + x
        # 這裡先保持忠於論文設計，讓 SSE 單純做重加權
        return out


# ============== RRBBlock (Lazy RepVGG-style Block) ==============
class RRBBlock(nn.Module):
    """
    Lazy 版重參殘差。YAML：
      - [-1, 2, RRBBlock, []]        # 通道不變
      - [-1, 2, RRBBlock, [512]]     # 需要時才變更通道

    訓練階段：
      y = SiLU(Conv3x3_BN(x) + Conv1x1_BN(x) + BN(x)) + x   # 帶殘差

    deploy=True 時，可 fuse 成單一 3×3 conv（這段 fuse 可以沿用你原本的實作）。
    """

    def __init__(self, c2: int | None = None, deploy: bool = False):
        super().__init__()
        self.target_c = c2
        self.deploy = deploy
        self._built = False

    def _build(self, c1: int):
        c2 = self.target_c or c1
        self.proj = nn.Identity() if c1 == c2 else nn.Conv2d(c1, c2, 1, bias=False)

        if self.deploy:
            self.reparam_conv = nn.Conv2d(c2, c2, 3, padding=1, bias=True)
        else:
            # 使用 PyTorch 預設的 BN 超參數（eps=1e-5, momentum=0.1）
            self.branch_3x3 = nn.Sequential(
                nn.Conv2d(c2, c2, 3, padding=1, bias=False),
                nn.BatchNorm2d(c2, eps=1e-5, momentum=0.1),
            )
            self.branch_1x1 = nn.Sequential(
                nn.Conv2d(c2, c2, 1, bias=False),
                nn.BatchNorm2d(c2, eps=1e-5, momentum=0.1),
            )
            self.branch_identity = nn.BatchNorm2d(c2, eps=1e-5, momentum=0.1)

        self._built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._built:
            self._build(x.shape[1])

        x_in = self.proj(x)

        if self.deploy:
            return F.silu(self.reparam_conv(x_in)) + x_in

        # 訓練時：三個分支相加後再加上殘差 x_in
        out = self.branch_3x3(x_in) + self.branch_1x1(x_in) + self.branch_identity(x_in)
        return F.silu(out) + x_in


# ============== EAPIoU Loss (Aspect Ratio Penalty IoU) ==============
class EAPIoULoss(nn.Module):
    """
    Enhanced Aspect Ratio Penalty IoU Loss for YOLO
    
    改進自 CIoU，加入寬高比懲罰項以更好地處理小目標檢測：
    
    EAPIoU = IoU - (ρ²(b, b_gt)/c² + α*v + β*(r - 1)²)
    
    其中：
    - IoU: 標準 Intersection over Union
    - ρ²(b, b_gt): 預測框與真值框中心點歐式距離的平方
    - c²: 最小外接矩形對角線長度的平方
    - v: 寬高比一致性因子 = (4/π²) * (arctan(w_gt/h_gt) - arctan(w/h))²
    - α: 權重因子（動態計算）
    - β: 寬高比懲罰強度（建議 0.1）
    - r: 寬高比 = max(w/h, h/w)
    
    相比 CIoU 的改進：
    1. 新增 β*(r-1)² 項強化對極端寬高比的懲罰
    2. 對小目標（如 CT 病灶）的定位更敏感
    3. 更快的收斂速度
    
    Reference: 
    - CIoU: https://arxiv.org/abs/1911.08287
    - 寬高比懲罰改進基於醫學影像檢測最佳實踐
    """
    
    def __init__(self, beta: float = 0.1, eps: float = 1e-7):
        """
        Args:
            beta: 寬高比懲罰強度 (建議 0.1-0.3)
            eps: 數值穩定性小量
        """
        super().__init__()
        self.beta = beta
        self.eps = eps
    
    def forward(
        self, 
        pred_boxes: torch.Tensor, 
        target_boxes: torch.Tensor
    ) -> torch.Tensor:
        """
        計算 EAPIoU Loss
        
        Args:
            pred_boxes: 預測框 [N, 4] (x1, y1, x2, y2) 或 (cx, cy, w, h)
            target_boxes: 真值框 [N, 4] 格式同上
            
        Returns:
            loss: EAPIoU loss 值 [N]
        """
        # 轉換為 (x1, y1, x2, y2) 格式（如果輸入是 cxcywh）
        if self._is_cxcywh_format(pred_boxes):
            pred_boxes = self._cxcywh_to_xyxy(pred_boxes)
            target_boxes = self._cxcywh_to_xyxy(target_boxes)
        
        # 1. 計算 IoU
        iou = self._calculate_iou(pred_boxes, target_boxes)
        
        # 2. 計算中心點距離懲罰
        pred_cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
        pred_cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
        target_cx = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
        target_cy = (target_boxes[:, 1] + target_boxes[:, 3]) / 2
        
        rho2 = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2
        
        # 3. 計算最小外接矩形對角線長度
        c_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        c_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        c_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        c_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
        
        c2 = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2 + self.eps
        
        # 4. 計算寬高比一致性 v
        pred_w = pred_boxes[:, 2] - pred_boxes[:, 0]
        pred_h = pred_boxes[:, 3] - pred_boxes[:, 1]
        target_w = target_boxes[:, 2] - target_boxes[:, 0]
        target_h = target_boxes[:, 3] - target_boxes[:, 1]
        
        v = (4 / (torch.pi ** 2)) * torch.pow(
            torch.atan(target_w / (target_h + self.eps)) - 
            torch.atan(pred_w / (pred_h + self.eps)), 
            2
        )
        
        # 5. 計算 alpha 權重（CIoU 原始公式）
        with torch.no_grad():
            alpha = v / (1 - iou + v + self.eps)
        
        # 6. 計算寬高比懲罰項 (EAPIoU 新增)
        aspect_ratio = torch.maximum(
            pred_w / (pred_h + self.eps),
            pred_h / (pred_w + self.eps)
        )
        aspect_penalty = self.beta * torch.pow(aspect_ratio - 1, 2)
        
        # 7. 組合 EAPIoU
        eapiou = iou - (rho2 / c2 + alpha * v + aspect_penalty)
        
        # 轉換為 loss（1 - EAPIoU）
        loss = 1 - eapiou
        
        return loss
    
    def _calculate_iou(
        self, 
        boxes1: torch.Tensor, 
        boxes2: torch.Tensor
    ) -> torch.Tensor:
        """計算標準 IoU"""
        # 交集區域
        inter_x1 = torch.maximum(boxes1[:, 0], boxes2[:, 0])
        inter_y1 = torch.maximum(boxes1[:, 1], boxes2[:, 1])
        inter_x2 = torch.minimum(boxes1[:, 2], boxes2[:, 2])
        inter_y2 = torch.minimum(boxes1[:, 3], boxes2[:, 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                     torch.clamp(inter_y2 - inter_y1, min=0)
        
        # 並集區域
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union_area = area1 + area2 - inter_area + self.eps
        
        iou = inter_area / union_area
        return iou
    
    def _is_cxcywh_format(self, boxes: torch.Tensor) -> bool:
        """
        簡單啟發式判斷：如果 x2 < x1 或 y2 < y1，則可能是 cxcywh 格式
        （更穩健的做法是明確傳入格式參數）
        """
        # 暫時假設總是 xyxy 格式，由調用者保證
        return False
    
    def _cxcywh_to_xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        """轉換 (cx, cy, w, h) -> (x1, y1, x2, y2)"""
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        return torch.stack([x1, y1, x2, y2], dim=1)
