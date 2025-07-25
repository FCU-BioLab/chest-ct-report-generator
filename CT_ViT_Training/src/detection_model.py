#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CT-ViT 目標檢測升級版本
將原本的分類模型升級為同時支援分類和邊界框回歸的目標檢測模型

主要改進：
1. 從純分類升級到分類+邊界框檢測
2. 支援多個目標檢測
3. 整合原有的Vision Transformer架構
4. 保持與現有資料集的相容性

作者: GitHub Copilot  
日期: 2025-07-25
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from transformers import ViTModel, ViTConfig
from transformers.modeling_outputs import BaseModelOutput

class CTViTDetectionHead(nn.Module):
    """CT-ViT檢測頭，包含分類和邊界框回歸"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_classes = config.num_labels
        
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
            nn.Linear(256, 1)  # 物件/背景
        )
        
    def forward(self, sequence_output):
        # 使用[CLS] token的輸出
        cls_output = sequence_output[:, 0]  # [batch_size, hidden_size]
        
        # 分類預測
        class_logits = self.classifier(cls_output)
        
        # 邊界框預測 (正規化座標 0-1)
        bbox_pred = torch.sigmoid(self.bbox_regressor(cls_output))
        
        # 物件存在性預測
        objectness_logits = self.objectness(cls_output)
        
        return {
            'class_logits': class_logits,
            'bbox_pred': bbox_pred,
            'objectness_logits': objectness_logits
        }

class CTViTForDetection(nn.Module):
    """升級版CT-ViT，支援目標檢測"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels
        
        # Vision Transformer骨幹網路（保持原有架構）
        self.vit = ViTModel(config)
        
        # 檢測頭
        self.detection_head = CTViTDetectionHead(config)
        
        # 損失權重
        self.classification_weight = 1.0
        self.bbox_weight = 5.0
        self.objectness_weight = 2.0
        
    def forward(self, pixel_values, labels=None, bbox_targets=None, **kwargs):
        # Vision Transformer前向傳播
        outputs = self.vit(pixel_values=pixel_values)
        sequence_output = outputs.last_hidden_state
        
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
            'last_hidden_state': sequence_output
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

class CTViTDetectionTrainer:
    """CT-ViT檢測模型訓練器"""
    
    def __init__(self, model, config, logger=None):
        self.model = model
        self.config = config
        self.logger = logger
        
    def train_step(self, batch):
        """單步訓練"""
        self.model.train()
        
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        bbox_targets = batch.get('bbox_targets', None)
        
        outputs = self.model(
            pixel_values=pixel_values,
            labels=labels,
            bbox_targets=bbox_targets
        )
        
        return outputs['loss']
    
    def predict(self, pixel_values, confidence_threshold=0.5):
        """預測函式"""
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
            
            # 處理分類預測
            class_probs = F.softmax(outputs['class_logits'], dim=-1)
            class_pred = torch.argmax(class_probs, dim=-1)
            
            # 處理物件存在性預測
            objectness_probs = torch.sigmoid(outputs['objectness_logits'])
            
            # 處理邊界框預測
            bbox_pred = outputs['bbox_pred']
            
            # 基於置信度過濾
            confident_mask = objectness_probs.squeeze() > confidence_threshold
            
            results = []
            for i, is_confident in enumerate(confident_mask):
                if is_confident:
                    results.append({
                        'class_id': class_pred[i].item(),
                        'class_prob': class_probs[i].max().item(),
                        'bbox': bbox_pred[i].cpu().numpy(),
                        'objectness': objectness_probs[i].item()
                    })
            
            return results

def create_detection_model_from_classification(classification_model_path, num_classes=4):
    """從現有分類模型創建檢測模型"""
    
    # 載入原始分類模型的權重
    checkpoint = torch.load(classification_model_path, map_location='cpu')
    
    # 創建新的配置
    config = ViTConfig(
        image_size=224,
        patch_size=16,
        num_channels=1,  # CT影像單通道
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        num_labels=num_classes,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1
    )
    
    # 創建檢測模型
    detection_model = CTViTForDetection(config)
    
    # 載入ViT骨幹網路的權重
    if 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
    else:
        model_state_dict = checkpoint
    
    # 過濾出ViT相關的權重
    vit_state_dict = {}
    for key, value in model_state_dict.items():
        if key.startswith('vit.') or key.startswith('embeddings.') or key.startswith('encoder.'):
            vit_state_dict[key] = value
    
    # 載入權重（忽略檢測頭的權重）
    detection_model.load_state_dict(vit_state_dict, strict=False)
    
    print(f"成功從分類模型載入ViT骨幹網路權重")
    print(f"檢測頭將重新訓練")
    
    return detection_model

# === 使用範例 ===
if __name__ == "__main__":
    # 測試模型創建
    config = ViTConfig(
        image_size=224,
        patch_size=16,
        num_channels=1,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        num_labels=4,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1
    )
    
    model = CTViTForDetection(config)
    print(f"模型參數數量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 測試前向傳播
    batch_size = 2
    pixel_values = torch.randn(batch_size, 1, 224, 224)
    labels = torch.tensor([1, 2])
    bbox_targets = torch.tensor([[0.3, 0.4, 0.2, 0.3], [0.5, 0.6, 0.15, 0.25]])
    
    outputs = model(pixel_values, labels, bbox_targets)
    print(f"輸出形狀:")
    print(f"  分類邏輯: {outputs['class_logits'].shape}")
    print(f"  邊界框: {outputs['bbox_pred'].shape}")
    print(f"  物件存在性: {outputs['objectness_logits'].shape}")
    print(f"  損失: {outputs['loss']}")
