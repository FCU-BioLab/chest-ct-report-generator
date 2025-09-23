#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UNet++ Detection Model
基於 UNet++ 的醫學影像病灶檢測模型

UNet++ 特點：
1. 嵌套跳躍連接 (Nested Skip Connections)
2. 深度監督 (Deep Supervision)
3. 語義分割與目標檢測的結合
4. 多尺度特徵融合

實現功能：
- 語義分割：生成病灶分割遮罩
- 目標檢測：預測邊界框和類別
- 端到端訓練

作者: GitHub Copilot
日期: 2025-09-18
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging


class ConvBlock(nn.Module):
    """基礎卷積塊"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, use_batchnorm: bool = True):
        super(ConvBlock, self).__init__()
        
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_batchnorm))
        
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        layers.append(nn.ReLU(inplace=True))
        
        self.conv_block = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.conv_block(x)


class DoubleConv(nn.Module):
    """雙卷積層"""
    
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True):
        super(DoubleConv, self).__init__()
        
        self.conv1 = ConvBlock(in_channels, out_channels, use_batchnorm=use_batchnorm)
        self.conv2 = ConvBlock(out_channels, out_channels, use_batchnorm=use_batchnorm)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNetPlusPlus(nn.Module):
    """
    UNet++ 網路架構
    
    Args:
        in_channels: 輸入通道數
        n_classes: 分割類別數
        feature_scale: 特徵縮放因子
        is_deconv: 是否使用反卷積
        use_batchnorm: 是否使用批次歸一化
        deep_supervision: 是否使用深度監督
    """
    
    def __init__(self, in_channels: int = 1, n_classes: int = 1, feature_scale: int = 4,
                 is_deconv: bool = True, use_batchnorm: bool = True, deep_supervision: bool = True):
        super(UNetPlusPlus, self).__init__()
        
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.use_batchnorm = use_batchnorm
        self.deep_supervision = deep_supervision
        
        # 計算每層的通道數
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]
        
        # 編碼器 (Encoder)
        self.conv00 = DoubleConv(self.in_channels, filters[0], self.use_batchnorm)
        self.conv10 = DoubleConv(filters[0], filters[1], self.use_batchnorm)
        self.conv20 = DoubleConv(filters[1], filters[2], self.use_batchnorm)
        self.conv30 = DoubleConv(filters[2], filters[3], self.use_batchnorm)
        self.conv40 = DoubleConv(filters[3], filters[4], self.use_batchnorm)
        
        # 下採樣
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        # 嵌套跳躍連接
        # Level 1: X_i,1 = H([X_i,0, Up(X_{i+1,0})])
        self.conv01 = DoubleConv(filters[0] + filters[0], filters[0], self.use_batchnorm)  # X00 + Up(X10)
        self.conv11 = DoubleConv(filters[1] + filters[1], filters[1], self.use_batchnorm)  # X10 + Up(X20)  
        self.conv21 = DoubleConv(filters[2] + filters[2], filters[2], self.use_batchnorm)  # X20 + Up(X30)
        self.conv31 = DoubleConv(filters[3] + filters[3], filters[3], self.use_batchnorm)  # X30 + Up(X40)
        
        # Level 2: X_i,2 = H([X_i,0, X_i,1, Up(X_{i+1,1})])
        self.conv02 = DoubleConv(filters[0] * 2 + filters[0], filters[0], self.use_batchnorm)  # X00 + X01 + Up(X11)
        self.conv12 = DoubleConv(filters[1] * 2 + filters[1], filters[1], self.use_batchnorm)  # X10 + X11 + Up(X21)
        self.conv22 = DoubleConv(filters[2] * 2 + filters[2], filters[2], self.use_batchnorm)  # X20 + X21 + Up(X31)
        
        # Level 3: X_i,3 = H([X_i,0, X_i,1, X_i,2, Up(X_{i+1,2})])
        self.conv03 = DoubleConv(filters[0] * 3 + filters[0], filters[0], self.use_batchnorm)  # X00 + X01 + X02 + Up(X12)
        self.conv13 = DoubleConv(filters[1] * 3 + filters[1], filters[1], self.use_batchnorm)  # X10 + X11 + X12 + Up(X22)
        
        # Level 4: X_0,4 = H([X_0,0, X_0,1, X_0,2, X_0,3, Up(X_1,3)])
        self.conv04 = DoubleConv(filters[0] * 4 + filters[0], filters[0], self.use_batchnorm)  # X00 + X01 + X02 + X03 + Up(X13)
        
        # 上採樣層（包含通道數調整）
        if self.is_deconv:
            # Level 1 上採樣 - 調整通道數以匹配目標層
            self.up_concat01 = nn.Sequential(
                self._make_deconv_layer(filters[1], filters[0]),
                nn.Conv2d(filters[0], filters[0], 1)  # 通道數調整
            )
            self.up_concat11 = nn.Sequential(
                self._make_deconv_layer(filters[2], filters[1]),
                nn.Conv2d(filters[1], filters[1], 1)
            )
            self.up_concat21 = nn.Sequential(
                self._make_deconv_layer(filters[3], filters[2]),
                nn.Conv2d(filters[2], filters[2], 1)
            )
            self.up_concat31 = nn.Sequential(
                self._make_deconv_layer(filters[4], filters[3]),
                nn.Conv2d(filters[3], filters[3], 1)
            )
            
            # Level 2 上採樣
            self.up_concat02 = nn.Sequential(
                self._make_deconv_layer(filters[1], filters[0]),
                nn.Conv2d(filters[0], filters[0], 1)
            )
            self.up_concat12 = nn.Sequential(
                self._make_deconv_layer(filters[2], filters[1]),
                nn.Conv2d(filters[1], filters[1], 1)
            )
            self.up_concat22 = nn.Sequential(
                self._make_deconv_layer(filters[3], filters[2]),
                nn.Conv2d(filters[2], filters[2], 1)
            )
            
            # Level 3 上採樣
            self.up_concat03 = nn.Sequential(
                self._make_deconv_layer(filters[1], filters[0]),
                nn.Conv2d(filters[0], filters[0], 1)
            )
            self.up_concat13 = nn.Sequential(
                self._make_deconv_layer(filters[2], filters[1]),
                nn.Conv2d(filters[1], filters[1], 1)
            )
            
            # Level 4 上採樣
            self.up_concat04 = nn.Sequential(
                self._make_deconv_layer(filters[1], filters[0]),
                nn.Conv2d(filters[0], filters[0], 1)
            )
        else:
            # 使用插值上採樣 + 通道調整
            self.up_concat01 = nn.Conv2d(filters[1], filters[0], 1)
            self.up_concat11 = nn.Conv2d(filters[2], filters[1], 1)
            self.up_concat21 = nn.Conv2d(filters[3], filters[2], 1)
            self.up_concat31 = nn.Conv2d(filters[4], filters[3], 1)
            
            self.up_concat02 = nn.Conv2d(filters[1], filters[0], 1)
            self.up_concat12 = nn.Conv2d(filters[2], filters[1], 1)
            self.up_concat22 = nn.Conv2d(filters[3], filters[2], 1)
            
            self.up_concat03 = nn.Conv2d(filters[1], filters[0], 1)
            self.up_concat13 = nn.Conv2d(filters[2], filters[1], 1)
            
            self.up_concat04 = nn.Conv2d(filters[1], filters[0], 1)
        
        # 最終分割輸出層
        if self.deep_supervision:
            self.final1 = nn.Conv2d(filters[0], self.n_classes, 1)
            self.final2 = nn.Conv2d(filters[0], self.n_classes, 1)
            self.final3 = nn.Conv2d(filters[0], self.n_classes, 1)
            self.final4 = nn.Conv2d(filters[0], self.n_classes, 1)
        else:
            self.final = nn.Conv2d(filters[0], self.n_classes, 1)
        
        # 初始化權重
        self._initialize_weights()
    
    def _make_deconv_layer(self, in_channels: int, out_channels: int):
        """創建反卷積層"""
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def _initialize_weights(self):
        """初始化網路權重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def forward(self, inputs):
        # 編碼器路徑（下採樣）
        X00 = self.conv00(inputs)  # [B, filters[0], H, W]
        maxpool0 = self.maxpool(X00)
        
        X10 = self.conv10(maxpool0)  # [B, filters[1], H/2, W/2]
        maxpool1 = self.maxpool(X10)
        
        X20 = self.conv20(maxpool1)  # [B, filters[2], H/4, W/4]
        maxpool2 = self.maxpool(X20)
        
        X30 = self.conv30(maxpool2)  # [B, filters[3], H/8, W/8]
        maxpool3 = self.maxpool(X30)
        
        X40 = self.conv40(maxpool3)  # [B, filters[4], H/16, W/16]
        
        # 嵌套跳躍連接 - Level 1
        # X01: 連接 X00 和 上採樣的 X10
        if self.is_deconv:
            up_X10 = self.up_concat01(X10)  # 直接使用包含通道調整的上採樣
        else:
            up_X10 = F.interpolate(X10, size=X00.shape[2:], mode='bilinear', align_corners=True)
            up_X10 = self.up_concat01(up_X10)  # 通道調整
        X01 = self.conv01(torch.cat([X00, up_X10], 1))  # [filters[0] + filters[0]] -> filters[0]
        
        # X11: 連接 X10 和 上採樣的 X20
        if self.is_deconv:
            up_X20 = self.up_concat11(X20)
        else:
            up_X20 = F.interpolate(X20, size=X10.shape[2:], mode='bilinear', align_corners=True)
            up_X20 = self.up_concat11(up_X20)
        X11 = self.conv11(torch.cat([X10, up_X20], 1))  # [filters[1] + filters[1]] -> filters[1]
        
        # X21: 連接 X20 和 上採樣的 X30
        if self.is_deconv:
            up_X30 = self.up_concat21(X30)
        else:
            up_X30 = F.interpolate(X30, size=X20.shape[2:], mode='bilinear', align_corners=True)
            up_X30 = self.up_concat21(up_X30)
        X21 = self.conv21(torch.cat([X20, up_X30], 1))  # [filters[2] + filters[2]] -> filters[2]
        
        # X31: 連接 X30 和 上採樣的 X40
        if self.is_deconv:
            up_X40 = self.up_concat31(X40)
        else:
            up_X40 = F.interpolate(X40, size=X30.shape[2:], mode='bilinear', align_corners=True)
            up_X40 = self.up_concat31(up_X40)
        X31 = self.conv31(torch.cat([X30, up_X40], 1))  # [filters[3] + filters[3]] -> filters[3]
        
        # 嵌套跳躍連接 - Level 2
        # X02: 連接 X00, X01 和 上採樣的 X11
        if self.is_deconv:
            up_X11 = self.up_concat02(X11)
        else:
            up_X11 = F.interpolate(X11, size=X00.shape[2:], mode='bilinear', align_corners=True)
            up_X11 = self.up_concat02(up_X11)
        X02 = self.conv02(torch.cat([X00, X01, up_X11], 1))  # [filters[0]*2 + filters[0]] -> filters[0]
        
        # X12: 連接 X10, X11 和 上採樣的 X21
        if self.is_deconv:
            up_X21 = self.up_concat12(X21)
        else:
            up_X21 = F.interpolate(X21, size=X10.shape[2:], mode='bilinear', align_corners=True)
            up_X21 = self.up_concat12(up_X21)
        X12 = self.conv12(torch.cat([X10, X11, up_X21], 1))  # [filters[1]*2 + filters[1]] -> filters[1]
        
        # X22: 連接 X20, X21 和 上採樣的 X31
        if self.is_deconv:
            up_X31 = self.up_concat22(X31)
        else:
            up_X31 = F.interpolate(X31, size=X20.shape[2:], mode='bilinear', align_corners=True)
            up_X31 = self.up_concat22(up_X31)
        X22 = self.conv22(torch.cat([X20, X21, up_X31], 1))  # [filters[2]*2 + filters[2]] -> filters[2]
        
        # 嵌套跳躍連接 - Level 3
        # X03: 連接 X00, X01, X02 和 上採樣的 X12
        if self.is_deconv:
            up_X12 = self.up_concat03(X12)
        else:
            up_X12 = F.interpolate(X12, size=X00.shape[2:], mode='bilinear', align_corners=True)
            up_X12 = self.up_concat03(up_X12)
        X03 = self.conv03(torch.cat([X00, X01, X02, up_X12], 1))  # [filters[0]*3 + filters[0]] -> filters[0]
        
        # X13: 連接 X10, X11, X12 和 上採樣的 X22
        if self.is_deconv:
            up_X22 = self.up_concat13(X22)
        else:
            up_X22 = F.interpolate(X22, size=X10.shape[2:], mode='bilinear', align_corners=True)
            up_X22 = self.up_concat13(up_X22)
        X13 = self.conv13(torch.cat([X10, X11, X12, up_X22], 1))  # [filters[1]*3 + filters[1]] -> filters[1]
        
        # 嵌套跳躍連接 - Level 4 (最終輸出)
        # X04: 連接 X00, X01, X02, X03 和 上採樣的 X13
        if self.is_deconv:
            up_X13 = self.up_concat04(X13)
        else:
            up_X13 = F.interpolate(X13, size=X00.shape[2:], mode='bilinear', align_corners=True)
            up_X13 = self.up_concat04(up_X13)
        X04 = self.conv04(torch.cat([X00, X01, X02, X03, up_X13], 1))  # [filters[0]*4 + filters[0]] -> filters[0]
        
        # 輸出
        if self.deep_supervision:
            output1 = self.final1(X01)
            output2 = self.final2(X02)
            output3 = self.final3(X03)
            output4 = self.final4(X04)
            
            return [output1, output2, output3, output4]
        else:
            output = self.final(X04)
            return output


class BBoxRegressor(nn.Module):
    """邊界框回歸頭"""
    
    def __init__(self, in_channels: int, num_anchors: int = 9):
        super(BBoxRegressor, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 128, 3, padding=1)
        self.bbox_pred = nn.Conv2d(128, num_anchors * 4, 1)  # 4個座標 (x1, y1, x2, y2)
        
        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        bbox_pred = self.bbox_pred(x)
        return bbox_pred


class ClassificationHead(nn.Module):
    """分類頭"""
    
    def __init__(self, in_channels: int, num_classes: int = 2, num_anchors: int = 9):
        super(ClassificationHead, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 128, 3, padding=1)
        self.cls_pred = nn.Conv2d(128, num_anchors * num_classes, 1)
        
        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        cls_pred = self.cls_pred(x)
        return cls_pred


class UNetPPDetector(nn.Module):
    """
    UNet++ 檢測器
    結合分割和檢測的端到端模型
    
    Args:
        in_channels: 輸入通道數
        num_classes: 檢測類別數（包含背景）
        segmentation_classes: 分割類別數
        feature_scale: 特徵縮放因子
        num_anchors: 錨點數量
    """
    
    def __init__(self, in_channels: int = 1, num_classes: int = 2, 
                 segmentation_classes: int = 1, feature_scale: int = 4, 
                 num_anchors: int = 9):
        super(UNetPPDetector, self).__init__()
        
        self.num_classes = num_classes
        self.segmentation_classes = segmentation_classes
        self.num_anchors = num_anchors
        
        # UNet++ backbone (修改為不使用深度監督以便獲取中間特徵)
        self.backbone = UNetPlusPlus(
            in_channels=in_channels,
            n_classes=segmentation_classes,
            feature_scale=feature_scale,
            deep_supervision=True  # 保持深度監督用於分割
        )
        
        # 計算特徵通道數
        backbone_channels = int(64 / feature_scale)
        
        # 檢測頭 - 使用正確的通道數
        self.bbox_regressor = BBoxRegressor(backbone_channels, num_anchors)
        self.classification_head = ClassificationHead(backbone_channels, num_classes, num_anchors)
        
        # 添加特徵提取層來從 backbone 獲取檢測特徵
        self.detection_feature_extractor = nn.Sequential(
            nn.Conv2d(backbone_channels, backbone_channels, 3, padding=1),
            nn.BatchNorm2d(backbone_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(backbone_channels, backbone_channels, 3, padding=1),
            nn.BatchNorm2d(backbone_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # 儲存原始輸入形狀
        if isinstance(x, list):
            # 如果輸入是列表，堆疊成張量
            x = torch.stack(x)
            
        original_shape = x.shape
        
        # 獲取 UNet++ 的分割輸出
        seg_outputs = self.backbone(x)
        
        # 從 backbone 提取檢測特徵
        # 我們需要修改 backbone 來返回中間特徵，這裡使用一個簡化方法
        detection_features = self._extract_detection_features_from_backbone(x)
        
        # 檢測分支
        bbox_pred = self.bbox_regressor(detection_features)
        cls_pred = self.classification_head(detection_features)
        
        # 確保分割輸出的格式正確
        if isinstance(seg_outputs, list):
            # 深度監督輸出，使用最後一個輸出作為主要分割結果
            main_seg_output = seg_outputs[-1]
            # 保持所有輸出用於損失計算
            segmentation_outputs = seg_outputs
        else:
            # 單一輸出
            main_seg_output = seg_outputs
            segmentation_outputs = seg_outputs
        
        return {
            'segmentation': segmentation_outputs,
            'bbox_pred': bbox_pred,
            'cls_pred': cls_pred
        }
    
    def _extract_detection_features_from_backbone(self, x):
        """
        從 backbone 提取檢測特徵
        這是一個簡化版本，實際應用中應該修改 backbone 來直接返回特徵
        """
        # 重新進行前向傳播獲取中間特徵
        # 這不是最效率的方法，但可以工作
        
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.backbone.feature_scale) for x in filters]
        
        # 編碼器第一層
        x00 = self.backbone.conv00(x)  # 這會給我們需要的特徵
        
        # 使用特徵提取器進一步處理
        detection_features = self.detection_feature_extractor(x00)
        
        return detection_features


def test_unet_plus_plus():
    """測試 UNet++ 模型"""
    # 創建測試數據
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 測試分割模型
    model_seg = UNetPlusPlus(in_channels=1, n_classes=1, deep_supervision=True)
    model_seg = model_seg.to(device)
    
    # 測試檢測模型
    model_det = UNetPPDetector(in_channels=1, num_classes=2)
    model_det = model_det.to(device)
    
    # 測試輸入
    test_input = torch.randn(2, 1, 512, 512).to(device)
    
    print("測試 UNet++ 分割模型...")
    with torch.no_grad():
        seg_output = model_seg(test_input)
        if isinstance(seg_output, list):
            print(f"深度監督輸出: {len(seg_output)} 個輸出")
            for i, out in enumerate(seg_output):
                print(f"  輸出 {i+1}: {out.shape}")
        else:
            print(f"分割輸出形狀: {seg_output.shape}")
    
    print("\n測試 UNet++ 檢測模型...")
    with torch.no_grad():
        det_output = model_det(test_input)
        print(f"分割輸出形狀: {det_output['segmentation'][-1].shape}")
        print(f"邊界框預測形狀: {det_output['bbox_pred'].shape}")
        print(f"分類預測形狀: {det_output['cls_pred'].shape}")
    
    print("\n模型測試完成！")


if __name__ == "__main__":
    test_unet_plus_plus()