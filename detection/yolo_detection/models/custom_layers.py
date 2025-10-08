#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Medical Image Detection Custom Modules for YOLOv7

Includes:
- CBAM (Convolutional Block Attention Module)
- SimAM (Simple, Parameter-Free Attention Module)
- Swin Transformer Block (Lightweight version)
- BiFPN (Bidirectional Feature Pyramid Network)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ==================== CBAM Attention ====================
class ChannelAttention(nn.Module):
    """Channel Attention Module for CBAM"""
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module for CBAM"""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    Paper: CBAM: Convolutional Block Attention Module (ECCV 2018)
    """
    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        x = x * self.channel_attention(x)
        # Spatial attention
        x = x * self.spatial_attention(x)
        return x


# ==================== SimAM Attention ====================
class SimAM(nn.Module):
    """
    Simple, Parameter-Free Attention Module
    Paper: SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks
    """
    def __init__(self, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        n = w * h - 1
        
        # Calculate mean and variance
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        
        return x * torch.sigmoid(y)


# ==================== Swin Transformer Block ====================
class WindowAttention(nn.Module):
    """
    Window-based Multi-head Self Attention (W-MSA) module
    Simplified version for medical imaging
    """
    def __init__(self, dim: int, window_size: int, num_heads: int = 8, qkv_bias: bool = True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input features with shape (num_windows*B, N, C)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block (Lightweight version for detection)
    Paper: Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    """
    def __init__(self, dim: int, num_heads: int = 8, window_size: int = 7, mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature (B, C, H, W)
        Returns:
            Output feature (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Reshape to (B, H*W, C)
        x_flat = x.flatten(2).transpose(1, 2)
        
        # Window partition
        shortcut = x_flat
        x_flat = self.norm1(x_flat)
        
        # Window attention
        x_windows = self.attn(x_flat)
        
        # Residual connection
        x_flat = shortcut + x_windows
        
        # FFN
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        
        # Reshape back to (B, C, H, W)
        x = x_flat.transpose(1, 2).reshape(B, C, H, W)
        return x


# ==================== BiFPN ====================
class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class BiFPNLayer(nn.Module):
    """
    Single BiFPN Layer with learnable weights
    Bidirectional Feature Pyramid Network
    """
    def __init__(self, channels: int, epsilon: float = 1e-4):
        super().__init__()
        self.epsilon = epsilon
        
        # Top-down pathway weights
        self.w1 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.w2 = nn.Parameter(torch.ones(3, dtype=torch.float32))
        
        # Convolutions for feature fusion
        self.conv_td = nn.ModuleList([
            DepthwiseSeparableConv(channels, channels)
            for _ in range(2)
        ])
        
        self.conv_out = nn.ModuleList([
            DepthwiseSeparableConv(channels, channels)
            for _ in range(3)
        ])
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: List of feature maps [P3, P4, P5] from backbone
        Returns:
            List of fused feature maps
        """
        p3, p4, p5 = features
        
        # Top-down pathway
        # P4_td = Conv(w1_0 * P4 + w1_1 * Resize(P5))
        w1 = F.relu(self.w1)
        w1 = w1 / (torch.sum(w1, dim=0) + self.epsilon)
        p4_td = self.conv_td[0](
            w1[0] * p4 + w1[1] * F.interpolate(p5, size=p4.shape[-2:], mode='nearest')
        )
        
        # P3_td = Conv(w1_0 * P3 + w1_1 * Resize(P4_td))
        p3_td = self.conv_td[1](
            w1[0] * p3 + w1[1] * F.interpolate(p4_td, size=p3.shape[-2:], mode='nearest')
        )
        
        # Bottom-up pathway
        # P4_out = Conv(w2_0 * P4 + w2_1 * P4_td + w2_2 * Resize(P3_td))
        w2 = F.relu(self.w2)
        w2 = w2 / (torch.sum(w2, dim=0) + self.epsilon)
        p4_out = self.conv_out[1](
            w2[0] * p4 + w2[1] * p4_td + w2[2] * F.interpolate(p3_td, scale_factor=0.5, mode='nearest')
        )
        
        # P5_out = Conv(w2_0 * P5 + w2_1 * Resize(P4_out))
        p5_out = self.conv_out[2](
            w2[0] * p5 + w2[1] * F.interpolate(p4_out, scale_factor=0.5, mode='nearest')
        )
        
        # P3_out
        p3_out = self.conv_out[0](p3_td)
        
        return [p3_out, p4_out, p5_out]


class BiFPN(nn.Module):
    """
    Bidirectional Feature Pyramid Network
    Paper: EfficientDet: Scalable and Efficient Object Detection
    """
    def __init__(self, channels: int, num_layers: int = 3):
        super().__init__()
        self.num_layers = num_layers
        
        # Stack multiple BiFPN layers
        self.bifpn_layers = nn.ModuleList([
            BiFPNLayer(channels)
            for _ in range(num_layers)
        ])
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: List of feature maps from backbone [P3, P4, P5]
        Returns:
            List of fused feature maps
        """
        for bifpn_layer in self.bifpn_layers:
            features = bifpn_layer(features)
        return features


# ==================== Module Registry ====================
def create_cbam(in_channels: int, reduction: int = 16, **kwargs) -> nn.Module:
    """Factory function for CBAM"""
    return CBAM(in_channels, reduction)


def create_simam(**kwargs) -> nn.Module:
    """Factory function for SimAM"""
    return SimAM()


def create_swin_block(dim: int, num_heads: int = 8, window_size: int = 7, **kwargs) -> nn.Module:
    """Factory function for Swin Transformer Block"""
    return SwinTransformerBlock(dim, num_heads, window_size)


def create_bifpn(channels: int, num_layers: int = 3, **kwargs) -> nn.Module:
    """Factory function for BiFPN"""
    return BiFPN(channels, num_layers)


# ==================== Module Testing ====================
if __name__ == "__main__":
    # Test CBAM
    print("Testing CBAM...")
    cbam = CBAM(in_channels=256)
    x = torch.randn(2, 256, 32, 32)
    out = cbam(x)
    print(f"CBAM input: {x.shape}, output: {out.shape}")
    
    # Test SimAM
    print("\nTesting SimAM...")
    simam = SimAM()
    x = torch.randn(2, 256, 32, 32)
    out = simam(x)
    print(f"SimAM input: {x.shape}, output: {out.shape}")
    
    # Test Swin Transformer Block
    print("\nTesting Swin Transformer Block...")
    swin = SwinTransformerBlock(dim=256, num_heads=8, window_size=7)
    x = torch.randn(2, 256, 28, 28)
    out = swin(x)
    print(f"Swin input: {x.shape}, output: {out.shape}")
    
    # Test BiFPN
    print("\nTesting BiFPN...")
    bifpn = BiFPN(channels=256, num_layers=3)
    features = [
        torch.randn(2, 256, 80, 80),  # P3
        torch.randn(2, 256, 40, 40),  # P4
        torch.randn(2, 256, 20, 20),  # P5
    ]
    out = bifpn(features)
    print(f"BiFPN input: {[f.shape for f in features]}")
    print(f"BiFPN output: {[f.shape for f in out]}")
    
    print("\nAll modules tested successfully!")
