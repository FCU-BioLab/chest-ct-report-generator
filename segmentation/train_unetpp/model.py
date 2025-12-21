#!/usr/bin/env python3
"""
UNet++ 肺結節分割訓練 - 模型模組
===================================

使用 segmentation-models-pytorch 建立 UNet++ 模型
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    logging.warning("segmentation-models-pytorch 未安裝，部分功能不可用")


logger = logging.getLogger(__name__)


def create_unetpp_model(
    encoder_name: str = "efficientnet-b4",
    encoder_weights: Optional[str] = "imagenet",
    in_channels: int = 3,
    num_classes: int = 1,
    deep_supervision: bool = False
) -> nn.Module:
    """
    創建 UNet++ 模型
    
    Args:
        encoder_name: 編碼器名稱
        encoder_weights: 預訓練權重
        in_channels: 輸入通道數（2.5D = 3）
        num_classes: 輸出類別數
        deep_supervision: 是否使用深度監督
        
    Returns:
        UNet++ 模型
    """
    if not SMP_AVAILABLE:
        raise ImportError("請安裝 segmentation-models-pytorch: pip install segmentation-models-pytorch")
    
    # 創建 UNet++ 模型
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None  # 輸出 logits，在 loss 中處理
    )
    
    logger.info(f"創建 UNet++ 模型: encoder={encoder_name}, in_channels={in_channels}")
    
    if deep_supervision:
        # 將模型包裝為深度監督版本
        model = DeepSupervisionWrapper(model, num_classes)
    
    return model


class DeepSupervisionWrapper(nn.Module):
    """深度監督包裝器"""
    
    def __init__(self, model: nn.Module, num_classes: int = 1):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        
        # 獲取解碼器輸出通道數
        # UNet++ 有多個解碼輸出
        self.deep_heads = nn.ModuleList()
        
        # 假設解碼器有 5 個階段
        # 這裡簡化處理，只使用最終輸出
        
    def forward(self, x):
        # 返回主輸出
        return self.model(x)


class UNetPP(nn.Module):
    """
    簡化版 UNet++（不使用 smp 時的備選方案）
    
    基本的 UNet++ 實現，用於測試和調試
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        features: list = [64, 128, 256, 512, 1024]
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # 編碼器
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        prev_channels = in_channels
        for feature in features:
            self.encoders.append(self._double_conv(prev_channels, feature))
            self.pools.append(nn.MaxPool2d(2, 2))
            prev_channels = feature
        
        # 解碼器（簡化版 - 實際 UNet++ 有更多跳躍連接）
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        for i in range(len(features) - 1, 0, -1):
            self.upsamples.append(
                nn.ConvTranspose2d(features[i], features[i-1], 2, 2)
            )
            self.decoders.append(
                self._double_conv(features[i-1] * 2, features[i-1])
            )
        
        # 輸出層
        self.final_conv = nn.Conv2d(features[0], num_classes, 1)
    
    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 編碼
        encoder_features = []
        for encoder, pool in zip(self.encoders[:-1], self.pools[:-1]):
            x = encoder(x)
            encoder_features.append(x)
            x = pool(x)
        
        x = self.encoders[-1](x)
        
        # 解碼
        for i, (upsample, decoder) in enumerate(zip(self.upsamples, self.decoders)):
            x = upsample(x)
            skip = encoder_features[-(i+1)]
            
            # 處理尺寸不匹配
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
        
        return self.final_conv(x)


def get_model(config) -> nn.Module:
    """
    根據配置獲取模型
    
    Args:
        config: 配置物件
        
    Returns:
        模型
    """
    if SMP_AVAILABLE:
        return create_unetpp_model(
            encoder_name=config.model.encoder_name,
            encoder_weights=config.model.encoder_weights,
            in_channels=config.model.in_channels,
            num_classes=config.model.num_classes,
            deep_supervision=config.model.deep_supervision
        )
    else:
        logger.warning("使用簡化版 UNet++")
        return UNetPP(
            in_channels=config.model.in_channels,
            num_classes=config.model.num_classes
        )


def count_parameters(model: nn.Module) -> int:
    """計算模型參數數量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 測試模型
    from .config import get_default_config
    
    config = get_default_config()
    model = get_model(config)
    
    print(f"Model: {type(model).__name__}")
    print(f"Parameters: {count_parameters(model):,}")
    
    # 測試前向傳播
    x = torch.randn(2, 3, 160, 160)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
