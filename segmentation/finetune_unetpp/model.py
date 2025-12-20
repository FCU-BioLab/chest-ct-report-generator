#!/usr/bin/env python3
"""
UNet++ 模型架構
===============

實現 UNet++ (Nested U-Net) 用於醫學影像分割

特點:
1. 嵌套的 skip connections (dense skip pathways)
2. 深度監督 (deep supervision)
3. 支援多種 backbone (ResNet, EfficientNet, etc.)
4. 可選的注意力機制

參考論文:
- Zhou et al., "UNet++: A Nested U-Net Architecture for Medical Image Segmentation"
- https://arxiv.org/abs/1807.10165
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class ConvBlock(nn.Module):
    """
    基礎卷積塊: Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d -> ReLU
    
    Args:
        in_channels: 輸入通道數
        out_channels: 輸出通道數
        use_dropout: 是否使用 dropout
        dropout_rate: dropout 比率
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        use_dropout: bool = False,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        
        if use_dropout:
            layers.append(nn.Dropout2d(dropout_rate))
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AttentionGate(nn.Module):
    """
    注意力閘門 (Attention Gate)
    
    用於增強模型對重要特徵的關注
    
    Args:
        F_g: 來自更深層的特徵通道數
        F_l: 來自 skip connection 的特徵通道數
        F_int: 中間層的通道數
    """
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g: 來自更深層的特徵 (gating signal)
            x: 來自 skip connection 的特徵
        
        Returns:
            注意力加權後的特徵
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # 確保尺寸匹配
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class SqueezeExcite(nn.Module):
    """
    Squeeze-and-Excitation (SE) 模組
    
    通道注意力機制，用於強調重要的特徵通道
    
    Args:
        in_channels: 輸入通道數
        reduction: 降維比率
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Squeeze: Global Average Pooling
        w = F.adaptive_avg_pool2d(x, 1)
        # Excitation: FC -> ReLU -> FC -> Sigmoid
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        # Scale
        return x * w


class UNetPlusPlus(nn.Module):
    """
    UNet++ (Nested U-Net) 模型
    
    採用嵌套的 skip connections 設計，可以捕捉多尺度特徵
    
    Args:
        in_channels: 輸入通道數 (預設 3 for RGB, 1 for grayscale)
        out_channels: 輸出通道數 (預設 1 for binary segmentation)
        features: 各層的特徵通道數
        use_attention: 是否使用注意力機制
        use_se: 是否使用 Squeeze-and-Excitation
        deep_supervision: 是否使用深度監督
        dropout_rate: Dropout 比率
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: List[int] = [64, 128, 256, 512, 1024],
        use_attention: bool = True,
        use_se: bool = True,
        deep_supervision: bool = True,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.use_attention = use_attention
        self.use_se = use_se
        self.deep_supervision = deep_supervision
        self.n_levels = len(features)
        
        # 下採樣
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 上採樣
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # ========== Encoder (下採樣路徑) ==========
        # X_0,0 -> X_1,0 -> X_2,0 -> X_3,0 -> X_4,0
        self.conv0_0 = ConvBlock(in_channels, features[0], dropout_rate=dropout_rate)
        self.conv1_0 = ConvBlock(features[0], features[1], dropout_rate=dropout_rate)
        self.conv2_0 = ConvBlock(features[1], features[2], dropout_rate=dropout_rate)
        self.conv3_0 = ConvBlock(features[2], features[3], dropout_rate=dropout_rate)
        self.conv4_0 = ConvBlock(features[3], features[4], use_dropout=True, dropout_rate=dropout_rate)
        
        # ========== Dense Skip Connections ==========
        # Level 1: X_0,1
        self.conv0_1 = ConvBlock(features[0] + features[1], features[0])
        
        # Level 2: X_1,1, X_0,2
        self.conv1_1 = ConvBlock(features[1] + features[2], features[1])
        self.conv0_2 = ConvBlock(features[0] * 2 + features[1], features[0])
        
        # Level 3: X_2,1, X_1,2, X_0,3
        self.conv2_1 = ConvBlock(features[2] + features[3], features[2])
        self.conv1_2 = ConvBlock(features[1] * 2 + features[2], features[1])
        self.conv0_3 = ConvBlock(features[0] * 3 + features[1], features[0])
        
        # Level 4: X_3,1, X_2,2, X_1,3, X_0,4
        self.conv3_1 = ConvBlock(features[3] + features[4], features[3])
        self.conv2_2 = ConvBlock(features[2] * 2 + features[3], features[2])
        self.conv1_3 = ConvBlock(features[1] * 3 + features[2], features[1])
        self.conv0_4 = ConvBlock(features[0] * 4 + features[1], features[0])
        
        # ========== Attention Gates ==========
        if use_attention:
            self.att0_1 = AttentionGate(features[1], features[0], features[0] // 2)
            self.att0_2 = AttentionGate(features[1], features[0], features[0] // 2)
            self.att0_3 = AttentionGate(features[1], features[0], features[0] // 2)
            self.att0_4 = AttentionGate(features[1], features[0], features[0] // 2)
            
            self.att1_1 = AttentionGate(features[2], features[1], features[1] // 2)
            self.att1_2 = AttentionGate(features[2], features[1], features[1] // 2)
            self.att1_3 = AttentionGate(features[2], features[1], features[1] // 2)
            
            self.att2_1 = AttentionGate(features[3], features[2], features[2] // 2)
            self.att2_2 = AttentionGate(features[3], features[2], features[2] // 2)
            
            self.att3_1 = AttentionGate(features[4], features[3], features[3] // 2)
        
        # ========== SE Blocks ==========
        if use_se:
            self.se0_0 = SqueezeExcite(features[0])
            self.se1_0 = SqueezeExcite(features[1])
            self.se2_0 = SqueezeExcite(features[2])
            self.se3_0 = SqueezeExcite(features[3])
            self.se4_0 = SqueezeExcite(features[4])
        
        # ========== Output Layers ==========
        if deep_supervision:
            self.final1 = nn.Conv2d(features[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(features[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(features[0], out_channels, kernel_size=1)
            self.final4 = nn.Conv2d(features[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        # 權重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型權重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        Args:
            x: 輸入影像 [B, C, H, W]
        
        Returns:
            分割預測 [B, out_channels, H, W]
            如果 deep_supervision=True，返回多個輸出的平均
        """
        # ========== Encoder ==========
        x0_0 = self.conv0_0(x)
        if self.use_se:
            x0_0 = self.se0_0(x0_0)
        
        x1_0 = self.conv1_0(self.pool(x0_0))
        if self.use_se:
            x1_0 = self.se1_0(x1_0)
        
        x2_0 = self.conv2_0(self.pool(x1_0))
        if self.use_se:
            x2_0 = self.se2_0(x2_0)
        
        x3_0 = self.conv3_0(self.pool(x2_0))
        if self.use_se:
            x3_0 = self.se3_0(x3_0)
        
        x4_0 = self.conv4_0(self.pool(x3_0))
        if self.use_se:
            x4_0 = self.se4_0(x4_0)
        
        # ========== Decoder Level 1 ==========
        x0_0_att = self.att0_1(self.up(x1_0), x0_0) if self.use_attention else x0_0
        x0_1 = self.conv0_1(torch.cat([x0_0_att, self.up(x1_0)], dim=1))
        
        # ========== Decoder Level 2 ==========
        x1_0_att = self.att1_1(self.up(x2_0), x1_0) if self.use_attention else x1_0
        x1_1 = self.conv1_1(torch.cat([x1_0_att, self.up(x2_0)], dim=1))
        
        x0_0_att2 = self.att0_2(self.up(x1_1), x0_0) if self.use_attention else x0_0
        x0_2 = self.conv0_2(torch.cat([x0_0_att2, x0_1, self.up(x1_1)], dim=1))
        
        # ========== Decoder Level 3 ==========
        x2_0_att = self.att2_1(self.up(x3_0), x2_0) if self.use_attention else x2_0
        x2_1 = self.conv2_1(torch.cat([x2_0_att, self.up(x3_0)], dim=1))
        
        x1_0_att2 = self.att1_2(self.up(x2_1), x1_0) if self.use_attention else x1_0
        x1_2 = self.conv1_2(torch.cat([x1_0_att2, x1_1, self.up(x2_1)], dim=1))
        
        x0_0_att3 = self.att0_3(self.up(x1_2), x0_0) if self.use_attention else x0_0
        x0_3 = self.conv0_3(torch.cat([x0_0_att3, x0_1, x0_2, self.up(x1_2)], dim=1))
        
        # ========== Decoder Level 4 ==========
        x3_0_att = self.att3_1(self.up(x4_0), x3_0) if self.use_attention else x3_0
        x3_1 = self.conv3_1(torch.cat([x3_0_att, self.up(x4_0)], dim=1))
        
        x2_0_att2 = self.att2_2(self.up(x3_1), x2_0) if self.use_attention else x2_0
        x2_2 = self.conv2_2(torch.cat([x2_0_att2, x2_1, self.up(x3_1)], dim=1))
        
        x1_0_att3 = self.att1_3(self.up(x2_2), x1_0) if self.use_attention else x1_0
        x1_3 = self.conv1_3(torch.cat([x1_0_att3, x1_1, x1_2, self.up(x2_2)], dim=1))
        
        x0_0_att4 = self.att0_4(self.up(x1_3), x0_0) if self.use_attention else x0_0
        x0_4 = self.conv0_4(torch.cat([x0_0_att4, x0_1, x0_2, x0_3, self.up(x1_3)], dim=1))
        
        # ========== Output ==========
        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            
            # 訓練時返回所有輸出，推論時返回平均
            if self.training:
                return (output1 + output2 + output3 + output4) / 4
            else:
                return output4  # 推論時只返回最終輸出
        else:
            return self.final(x0_4)
    
    def get_deep_supervision_outputs(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        獲取所有深度監督輸出（用於計算深度監督損失）
        
        Args:
            x: 輸入影像 [B, C, H, W]
        
        Returns:
            List of outputs from different levels
        """
        # ========== Encoder ==========
        x0_0 = self.conv0_0(x)
        if self.use_se:
            x0_0 = self.se0_0(x0_0)
        
        x1_0 = self.conv1_0(self.pool(x0_0))
        if self.use_se:
            x1_0 = self.se1_0(x1_0)
        
        x2_0 = self.conv2_0(self.pool(x1_0))
        if self.use_se:
            x2_0 = self.se2_0(x2_0)
        
        x3_0 = self.conv3_0(self.pool(x2_0))
        if self.use_se:
            x3_0 = self.se3_0(x3_0)
        
        x4_0 = self.conv4_0(self.pool(x3_0))
        if self.use_se:
            x4_0 = self.se4_0(x4_0)
        
        # ========== Decoder ==========
        x0_0_att = self.att0_1(self.up(x1_0), x0_0) if self.use_attention else x0_0
        x0_1 = self.conv0_1(torch.cat([x0_0_att, self.up(x1_0)], dim=1))
        
        x1_0_att = self.att1_1(self.up(x2_0), x1_0) if self.use_attention else x1_0
        x1_1 = self.conv1_1(torch.cat([x1_0_att, self.up(x2_0)], dim=1))
        
        x0_0_att2 = self.att0_2(self.up(x1_1), x0_0) if self.use_attention else x0_0
        x0_2 = self.conv0_2(torch.cat([x0_0_att2, x0_1, self.up(x1_1)], dim=1))
        
        x2_0_att = self.att2_1(self.up(x3_0), x2_0) if self.use_attention else x2_0
        x2_1 = self.conv2_1(torch.cat([x2_0_att, self.up(x3_0)], dim=1))
        
        x1_0_att2 = self.att1_2(self.up(x2_1), x1_0) if self.use_attention else x1_0
        x1_2 = self.conv1_2(torch.cat([x1_0_att2, x1_1, self.up(x2_1)], dim=1))
        
        x0_0_att3 = self.att0_3(self.up(x1_2), x0_0) if self.use_attention else x0_0
        x0_3 = self.conv0_3(torch.cat([x0_0_att3, x0_1, x0_2, self.up(x1_2)], dim=1))
        
        x3_0_att = self.att3_1(self.up(x4_0), x3_0) if self.use_attention else x3_0
        x3_1 = self.conv3_1(torch.cat([x3_0_att, self.up(x4_0)], dim=1))
        
        x2_0_att2 = self.att2_2(self.up(x3_1), x2_0) if self.use_attention else x2_0
        x2_2 = self.conv2_2(torch.cat([x2_0_att2, x2_1, self.up(x3_1)], dim=1))
        
        x1_0_att3 = self.att1_3(self.up(x2_2), x1_0) if self.use_attention else x1_0
        x1_3 = self.conv1_3(torch.cat([x1_0_att3, x1_1, x1_2, self.up(x2_2)], dim=1))
        
        x0_0_att4 = self.att0_4(self.up(x1_3), x0_0) if self.use_attention else x0_0
        x0_4 = self.conv0_4(torch.cat([x0_0_att4, x0_1, x0_2, x0_3, self.up(x1_3)], dim=1))
        
        # 輸出
        output1 = self.final1(x0_1)
        output2 = self.final2(x0_2)
        output3 = self.final3(x0_3)
        output4 = self.final4(x0_4)
        
        return [output1, output2, output3, output4]
    
    def get_encoder_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        獲取 Encoder 各層的特徵（用於特徵提取）
        
        Args:
            x: 輸入影像 [B, C, H, W]
        
        Returns:
            List of encoder features
        """
        features = []
        
        x0_0 = self.conv0_0(x)
        if self.use_se:
            x0_0 = self.se0_0(x0_0)
        features.append(x0_0)
        
        x1_0 = self.conv1_0(self.pool(x0_0))
        if self.use_se:
            x1_0 = self.se1_0(x1_0)
        features.append(x1_0)
        
        x2_0 = self.conv2_0(self.pool(x1_0))
        if self.use_se:
            x2_0 = self.se2_0(x2_0)
        features.append(x2_0)
        
        x3_0 = self.conv3_0(self.pool(x2_0))
        if self.use_se:
            x3_0 = self.se3_0(x3_0)
        features.append(x3_0)
        
        x4_0 = self.conv4_0(self.pool(x3_0))
        if self.use_se:
            x4_0 = self.se4_0(x4_0)
        features.append(x4_0)
        
        return features


class UNetPlusPlusLite(nn.Module):
    """
    輕量版 UNet++ (適用於小型資料集或有限計算資源)
    
    相比標準 UNet++:
    - 減少特徵通道數
    - 減少網路深度 (4 levels instead of 5)
    - 沒有 SE blocks
    
    Args:
        in_channels: 輸入通道數
        out_channels: 輸出通道數
        features: 各層特徵通道數 (預設 [32, 64, 128, 256])
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: List[int] = [32, 64, 128, 256]
    ):
        super().__init__()
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Encoder
        self.conv0_0 = ConvBlock(in_channels, features[0])
        self.conv1_0 = ConvBlock(features[0], features[1])
        self.conv2_0 = ConvBlock(features[1], features[2])
        self.conv3_0 = ConvBlock(features[2], features[3])
        
        # Decoder
        self.conv0_1 = ConvBlock(features[0] + features[1], features[0])
        self.conv1_1 = ConvBlock(features[1] + features[2], features[1])
        self.conv0_2 = ConvBlock(features[0] * 2 + features[1], features[0])
        self.conv2_1 = ConvBlock(features[2] + features[3], features[2])
        self.conv1_2 = ConvBlock(features[1] * 2 + features[2], features[1])
        self.conv0_3 = ConvBlock(features[0] * 3 + features[1], features[0])
        
        # Output
        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1))
        
        return self.final(x0_3)


def get_unetpp_model(
    model_type: str = "standard",
    in_channels: int = 3,
    out_channels: int = 1,
    pretrained: bool = False,
    pretrained_path: Optional[str] = None,
    **kwargs
) -> nn.Module:
    """
    工廠函數: 創建 UNet++ 模型
    
    Args:
        model_type: 模型類型 ("standard", "lite", "attention", "no_attention")
        in_channels: 輸入通道數
        out_channels: 輸出通道數
        pretrained: 是否載入預訓練權重
        pretrained_path: 預訓練權重路徑
        **kwargs: 其他模型參數 (features, deep_supervision 等)
    
    Returns:
        UNet++ 模型實例
    """
    # 從 kwargs 中提取 deep_supervision，預設為 True
    deep_supervision = kwargs.pop('deep_supervision', True)
    
    if model_type == "standard":
        model = UNetPlusPlus(
            in_channels=in_channels,
            out_channels=out_channels,
            use_attention=True,
            use_se=True,
            deep_supervision=deep_supervision,
            **kwargs
        )
    elif model_type == "lite":
        model = UNetPlusPlusLite(
            in_channels=in_channels,
            out_channels=out_channels,
            **kwargs
        )
    elif model_type == "attention":
        model = UNetPlusPlus(
            in_channels=in_channels,
            out_channels=out_channels,
            use_attention=True,
            use_se=True,
            deep_supervision=deep_supervision,
            **kwargs
        )
    elif model_type == "no_attention":
        model = UNetPlusPlus(
            in_channels=in_channels,
            out_channels=out_channels,
            use_attention=False,
            use_se=False,
            deep_supervision=deep_supervision,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if pretrained and pretrained_path:
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded pretrained weights from {pretrained_path}")
    
    return model


def get_smp_unetpp_model(
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
    encoder_depth: int = 5,
    decoder_use_batchnorm: bool = True,
    decoder_attention_type: Optional[str] = None,
) -> nn.Module:
    """
    使用 segmentation_models.pytorch 創建 UNet++ 模型
    
    Args:
        encoder_name: Encoder 名稱，例如:
            - "resnet34" (推薦，平衡性能與速度)
            - "resnet50", "resnet101" (更大容量)
            - "efficientnet-b0" ~ "efficientnet-b7" (高效)
            - "mobilenet_v2" (輕量)
            - "timm-resnest50d" (最新架構)
        encoder_weights: 預訓練權重
            - "imagenet": 使用 ImageNet 預訓練權重
            - None: 隨機初始化
        in_channels: 輸入通道數 (1 for grayscale, 3 for RGB)
        classes: 輸出類別數 (1 for binary segmentation)
        encoder_depth: Encoder 深度 (3-5)
        decoder_use_batchnorm: 是否在 decoder 使用 BatchNorm
        decoder_attention_type: 注意力機制類型
            - None: 無注意力
            - "scse": 使用 Squeeze-and-Excitation 注意力
    
    Returns:
        SMP UNet++ 模型實例
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        raise ImportError(
            "segmentation_models_pytorch is required. "
            "Install with: pip install segmentation-models-pytorch"
        )
    
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        encoder_depth=encoder_depth,
        decoder_use_batchnorm=decoder_use_batchnorm,
        decoder_attention_type=decoder_attention_type,
    )
    
    return model


def count_parameters(model: nn.Module) -> int:
    """計算模型參數數量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 測試模型
    print("Testing UNet++ models...")
    
    # 測試標準 UNet++
    model_standard = UNetPlusPlus(in_channels=3, out_channels=1)
    x = torch.randn(2, 3, 256, 256)
    out = model_standard(x)
    print(f"Standard UNet++: input={x.shape}, output={out.shape}")
    print(f"Parameters: {count_parameters(model_standard):,}")
    
    # 測試輕量版
    model_lite = UNetPlusPlusLite(in_channels=3, out_channels=1)
    out_lite = model_lite(x)
    print(f"UNet++ Lite: input={x.shape}, output={out_lite.shape}")
    print(f"Parameters: {count_parameters(model_lite):,}")
    
    # 測試深度監督輸出
    model_standard.train()
    outputs = model_standard.get_deep_supervision_outputs(x)
    print(f"Deep supervision outputs: {[o.shape for o in outputs]}")
    
    print("\nAll tests passed!")
