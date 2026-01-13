#!/usr/bin/env python3
"""
UNet++ 肺結節分割訓練 - 模型模組
===================================

Custom implementation of UNet++ (NestedUNet) based on MrGiovanni's official repository.
Supports proper Deep Supervision with 4 outputs.
Reference: https://github.com/MrGiovanni/UNetPlusPlus
"""

import logging
from typing import Optional, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    logging.warning("segmentation-models-pytorch 未安裝，只能使用基本功能")


logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """Standard Conv-BN-ReLU block"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class NestedUNet(nn.Module):
    """
    UNet++ (Nested U-Net) with Deep Supervision
    
    Architecture:
        x0_0 - x0_1 - x0_2 - x0_3 - x0_4 (Output)
          |      |      |      |
        x1_0 - x1_1 - x1_2 - x1_3
          |      |      |
        x2_0 - x2_1 - x2_2
          |      |
        x3_0 - x3_1
          |
        x4_0 (Bottleneck)
        
    Deep Supervision Outputs: [x0_1, x0_2, x0_3, x0_4]
    """
    
    def __init__(
        self,
        encoder_name: str = "efficientnet-b0",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        num_classes: int = 1,
        deep_supervision: bool = True
    ):
        super().__init__()
        
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes
        
        # --- Encoder (Backbone) ---
        # Use SMP to get pretrained backbone features comfortably
        if not SMP_AVAILABLE:
            raise ImportError("segmentation-models-pytorch is required for backbone.")
            
        self.encoder = smp.encoders.get_encoder(
            encoder_name, 
            in_channels=in_channels, 
            weights=encoder_weights
        )
        
        # Get channel counts from encoder
        # EfficientNet usually returns 6 stages [in, s1, s2, s3, s4, s5]
        # We need 5 stages for UNet++ standard 5-level depth
        encoder_channels = self.encoder.out_channels
        # Select indices: 0 (input resol), 1 (x2), 2 (x4), 3 (x8), 4 (x16) usually
        # But for EfficientNet it's often [3, 32, 24, 40, 112, 320] (example)
        # We pick 5 stages. x0_0 is usually full res or 1/2. 
        # SMP UNet uses stages (0), 1, 2, 3, 4. Let's align with that.
        
        # Filters for decoder stages
        # We'll project encoder channels to these standard filter counts or keep them?
        # MrGiovanni uses [32, 64, 128, 256, 512] for VGG
        # Let's use a fixed filter schedule for the decoder path to keep it clean.
        filters = [32, 64, 128, 256, 512]
        
        # 1x1 Convs to adapt backbone channels to skip connection channels
        self.adapt_enc = nn.ModuleList([
            nn.Conv2d(c, f, 1) for c, f in zip(encoder_channels[:5], filters)
        ])
        
        # --- Decoder Nodes ---
        # Row 0
        self.conv0_0 = ConvBlock(filters[0], filters[0]) # actually this is adapt_enc[0] + processing if needed, but let's assume adapt_enc does the job of x0_0 linear transform or we process it.
        # Actually, x0_0, x1_0 etc ARE the encoder features (adapted).
        
        # Up + Concat + Conv
        # x0_1: cat(x0_0, up(x1_0)) -> In: f0 + f0 (up from f1? no typically upsample is just resizing)
        # Standard UNet++: skip from same level, up from lower.
        # Up(x1_0) has channels f1. We constrain f1 to match f0 spatial? yes.
        # But we force channel reduction on Up? Usually ConvTranspose or Bilinear + Conv.
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # x0_1: Input = x0_0 (f0) + up(x1_0) (f1->f0?). 
        # MrGiovanni impl: torch.cat([x0_0, self.up(x1_0)], 1). In_ch = f0 + f1 (if not reduced).
        # Let's stick to filters list for "node Output channels".
        
        # Node definitions
        # Row 0
        self.conv0_1 = ConvBlock(filters[0] + filters[1], filters[0])
        self.conv0_2 = ConvBlock(filters[0]*2 + filters[1], filters[0])
        self.conv0_3 = ConvBlock(filters[0]*3 + filters[1], filters[0])
        self.conv0_4 = ConvBlock(filters[0]*4 + filters[1], filters[0])
        
        # Row 1
        self.conv1_1 = ConvBlock(filters[1] + filters[2], filters[1])
        self.conv1_2 = ConvBlock(filters[1]*2 + filters[2], filters[1])
        self.conv1_3 = ConvBlock(filters[1]*3 + filters[2], filters[1])
        
        # Row 2
        self.conv2_1 = ConvBlock(filters[2] + filters[3], filters[2])
        self.conv2_2 = ConvBlock(filters[2]*2 + filters[3], filters[2])
        
        # Row 3
        self.conv3_1 = ConvBlock(filters[3] + filters[4], filters[3])
        
        # Prediction Heads (1x1 Conv)
        self.final0_1 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
        self.final0_2 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
        self.final0_3 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
        self.final0_4 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
        
    def forward(self, x):
        # Backbone features
        features = self.encoder(x)
        # Adapt features to fixed filter sizes [f0, f1, f2, f3, f4]
        # features[0] is usually input resolution (or stride 1)
        x0_0 = self.adapt_enc[0](features[0])
        x1_0 = self.adapt_enc[1](features[1])
        x2_0 = self.adapt_enc[2](features[2])
        x3_0 = self.adapt_enc[3](features[3])
        x4_0 = self.adapt_enc[4](features[4])
        
        # Column 1
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        
        # Column 2
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        
        # Column 3
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        
        # Column 4
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        
        # Output
        if self.deep_supervision:
            out1 = self.final0_1(x0_1)
            out2 = self.final0_2(x0_2)
            out3 = self.final0_3(x0_3)
            out4 = self.final0_4(x0_4) # Final best output
            return [out4, out3, out2, out1] # Ordered by importance usually, or spatial? standard is [final, ...]
        else:
            return self.final0_4(x0_4)


def get_model(config) -> nn.Module:
    """
    根據配置獲取模型
    """
    return NestedUNet(
        encoder_name=config.model.encoder_name,
        encoder_weights=config.model.encoder_weights,
        in_channels=config.model.in_channels,
        num_classes=config.model.num_classes,
        deep_supervision=config.model.deep_supervision
    )


def count_parameters(model: nn.Module) -> int:
    """計算模型參數數量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test
    from .config import get_default_config
    config = get_default_config()
    model = get_model(config)
    print(f"Model created: {type(model).__name__}")
    print(f"Parameters: {count_parameters(model):,}")
    
    # Test forward
    x = torch.randn(2, 5, 224, 224) # 5 channels
    # x = torch.randn(2, 3, 224, 224) # 3 channels test? NO, config says 5
    y = model(x)
    if isinstance(y, list):
        print(f"Deep supervision outputs: {len(y)}")
        print(f"Shapes: {[o.shape for o in y]}")
    else:
        print(f"Output shape: {y.shape}")
