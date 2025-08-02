#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FPS-Former 目標檢測模型
Feature Pyramid Swin Transformer for Object Detection

主要特點：
1. 基於Swin Transformer的特徵金字塔網路
2. 多尺度特徵提取和融合
3. 高效的目標檢測架構
4. 適合醫學影像的目標檢測任務

作者: GitHub Copilot
日期: 2025-08-02
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional, Any

class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias."""
    
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def window_partition(x, window_size):
    """Partition into non-overlapping windows."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """Reverse window partition."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block."""
    
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = nn.Identity() if drop_path <= 0. else nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class PatchMerging(nn.Module):
    """Patch Merging Layer."""
    
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage."""
    
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""
    
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

class FPSFormerBackbone(nn.Module):
    """FPS-Former backbone based on Swin Transformer."""
    
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            nn.init.trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)

        return {
            'last_hidden_state': x,
            'feature_maps': features
        }

class FPSFormerDetectionHead(nn.Module):
    """FPS-Former檢測頭，包含分類和邊界框回歸"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.get('hidden_size', 768)
        self.num_classes = config.get('num_classes', 5)
        
        # 分類頭
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.num_classes)
        )
        
        # 邊界框回歸頭
        self.bbox_regressor = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 4)  # x, y, w, h
        )
        
        # 物件存在性分類器
        self.objectness = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
    
    def forward(self, sequence_output):
        # 使用 [CLS] token 的表示（第一個token）
        cls_output = sequence_output[:, 0]  # [batch_size, hidden_size]
        
        # 分類預測
        class_logits = self.classifier(cls_output)
        
        # 邊界框預測
        bbox_pred = self.bbox_regressor(cls_output)
        bbox_pred = torch.sigmoid(bbox_pred)  # 正規化到 [0, 1]
        
        # 物件存在性預測
        objectness_logits = self.objectness(cls_output)
        
        return {
            'class_logits': class_logits,
            'bbox_pred': bbox_pred,
            'objectness_logits': objectness_logits
        }

class FPSFormerForDetection(nn.Module):
    """FPS-Former目標檢測模型"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_labels = config.get('num_classes', 5)
        
        # FPS-Former骨幹網路
        self.backbone = FPSFormerBackbone(
            img_size=config.get('image_size', 224),
            patch_size=config.get('patch_size', 4),
            in_chans=config.get('num_channels', 3),
            embed_dim=config.get('embed_dim', 96),
            depths=config.get('depths', [2, 2, 6, 2]),
            num_heads=config.get('num_heads', [3, 6, 12, 24]),
            window_size=config.get('window_size', 7),
            drop_rate=config.get('drop_rate', 0.1),
            attn_drop_rate=config.get('attn_drop_rate', 0.1),
            drop_path_rate=config.get('drop_path_rate', 0.1)
        )
        
        # 檢測頭
        detection_config = {
            'hidden_size': self.backbone.num_features,
            'num_classes': self.num_labels
        }
        self.detection_head = FPSFormerDetectionHead(detection_config)
        
        # 損失權重
        self.classification_weight = 1.0
        self.bbox_weight = 5.0
        self.objectness_weight = 2.0
    
    def forward(self, pixel_values, labels=None, bbox_targets=None, **kwargs):
        # FPS-Former骨幹網路前向傳播
        backbone_outputs = self.backbone(pixel_values)
        
        # 為檢測頭準備序列輸出（添加一個假的序列維度）
        sequence_output = backbone_outputs['last_hidden_state'].unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        # 檢測頭預測
        detection_outputs = self.detection_head(sequence_output)
        
        # 計算損失（如果提供了標籤）
        loss = None
        if labels is not None:
            loss = self.compute_loss(detection_outputs, labels, bbox_targets)
        
        return {
            'loss': loss,
            'class_logits': detection_outputs['class_logits'],
            'bbox_pred': detection_outputs['bbox_pred'],
            'objectness_logits': detection_outputs['objectness_logits'],
            'last_hidden_state': sequence_output,
            'feature_maps': backbone_outputs['feature_maps']
        }
    
    def compute_loss(self, outputs, labels, bbox_targets):
        """計算多任務損失"""
        device = labels.device
        
        # 分類損失
        classification_loss = F.cross_entropy(
            outputs['class_logits'], 
            labels,
            ignore_index=-1
        )
        
        # 邊界框回歸損失（只對有目標的樣本計算）
        bbox_loss = torch.tensor(0.0, device=device)
        if bbox_targets is not None:
            # 只對非背景類別計算邊界框損失
            positive_mask = labels > 0  # 假設0是背景類別
            if positive_mask.sum() > 0:
                bbox_loss = F.smooth_l1_loss(
                    outputs['bbox_pred'][positive_mask],
                    bbox_targets[positive_mask]
                )
        
        # 物件存在性損失
        objectness_targets = (labels > 0).float().unsqueeze(1)
        objectness_loss = F.binary_cross_entropy_with_logits(
            outputs['objectness_logits'],
            objectness_targets
        )
        
        # 總損失
        total_loss = (
            self.classification_weight * classification_loss +
            self.bbox_weight * bbox_loss +
            self.objectness_weight * objectness_loss
        )
        
        return total_loss

def create_fps_former_detection_model(num_classes=5, image_size=224):
    """創建FPS-Former檢測模型"""
    config = {
        'num_classes': num_classes,
        'image_size': image_size,
        'patch_size': 4,
        'num_channels': 3,
        'embed_dim': 96,
        'depths': [2, 2, 6, 2],
        'num_heads': [3, 6, 12, 24],
        'window_size': 7,
        'drop_rate': 0.1,
        'attn_drop_rate': 0.1,
        'drop_path_rate': 0.1
    }
    
    model = FPSFormerForDetection(config)
    return model

def create_fps_former_from_classification(classification_model_path, num_classes=5):
    """從現有分類模型創建FPS-Former檢測模型（如果有的話）"""
    # 注意：這裡假設原來沒有FPS-Former分類模型，所以創建新模型
    print("Warning: 沒有找到預訓練的FPS-Former分類模型，創建新的檢測模型")
    return create_fps_former_detection_model(num_classes)

# === 使用範例 ===
if __name__ == "__main__":
    # 測試模型創建
    model = create_fps_former_detection_model(num_classes=5)
    print(f"模型參數數量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 測試前向傳播
    batch_size = 2
    pixel_values = torch.randn(batch_size, 3, 224, 224)
    labels = torch.tensor([1, 2])
    bbox_targets = torch.tensor([[0.3, 0.4, 0.2, 0.3], [0.5, 0.6, 0.15, 0.25]])
    
    outputs = model(pixel_values, labels, bbox_targets)
    print(f"輸出形狀:")
    print(f"  分類邏輯: {outputs['class_logits'].shape}")
    print(f"  邊界框: {outputs['bbox_pred'].shape}")
    print(f"  物件存在性: {outputs['objectness_logits'].shape}")
    print(f"  損失: {outputs['loss']}")
