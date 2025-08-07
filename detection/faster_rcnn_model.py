#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Faster R-CNN 目標檢測模型
基於torchvision的Faster R-CNN實現，適用於胸部CT影像病灶檢測

主要特點：
1. 使用ResNet50作為backbone
2. 二分類：背景(0) vs 病灶(1)
3. 預測邊界框回歸
4. 適合醫學影像的目標檢測任務

作者: GitHub Copilot
日期: 2025-08-06
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from typing import Dict, List, Tuple, Optional, Any

class FasterRCNN(nn.Module):
    """
    Faster R-CNN模型，用於胸部CT影像病灶檢測
    
    Args:
        num_classes: 類別數量（包含背景，默認2：背景+病灶）
        pretrained: 是否使用預訓練權重
        min_size: 輸入影像的最小尺寸
        max_size: 輸入影像的最大尺寸
    """
    
    def __init__(self, num_classes=2, pretrained=True, min_size=800, max_size=1333):
        super(FasterRCNN, self).__init__()
        
        self.num_classes = num_classes
        
        # 載入預訓練的Faster R-CNN模型
        self.model = fasterrcnn_resnet50_fpn(
            pretrained=pretrained,
            min_size=min_size,
            max_size=max_size,
            rpn_pre_nms_top_n_train=2000,
            rpn_pre_nms_top_n_test=1000,
            rpn_post_nms_top_n_train=2000,
            rpn_post_nms_top_n_test=1000,
            rpn_nms_thresh=0.7,
            rpn_fg_iou_thresh=0.7,
            rpn_bg_iou_thresh=0.3,
            rpn_batch_size_per_image=256,
            rpn_positive_fraction=0.5,
            box_score_thresh=0.05,
            box_nms_thresh=0.5,
            box_detections_per_img=100,
            box_fg_iou_thresh=0.5,
            box_bg_iou_thresh=0.5,
            box_batch_size_per_image=512,
            box_positive_fraction=0.25
        )
        
        # 修改分類器頭部以適應我們的類別數
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # 不修改第一層，而是在前向傳播時處理通道問題
        
    def forward(self, images, targets=None):
        """
        前向傳播
        
        Args:
            images: 輸入影像 [batch_size, 1, H, W] 或 [1, H, W] 或影像列表
            targets: 訓練時的標註資訊（推論時為None）
                    格式：[{'boxes': tensor, 'labels': tensor}, ...]
        
        Returns:
            訓練時返回損失字典，推論時返回檢測結果
        """
        # 處理不同格式的輸入
        if isinstance(images, torch.Tensor):
            if images.dim() == 2:
                # 2D 影像，添加通道維度
                images = images.unsqueeze(0).unsqueeze(0)  # 變成 [1, 1, H, W]
            elif images.dim() == 3:
                images = images.unsqueeze(0)  # 添加batch維度
            
            # 將張量轉換為影像列表（Faster R-CNN期望的格式）
            if images.size(1) == 1:
                # 灰階影像，複製為三通道 [3, H, W]
                image_list = [images[i, 0].repeat(3, 1, 1) for i in range(images.size(0))]
            else:
                image_list = [images[i] for i in range(images.size(0))]
        elif isinstance(images, list):
            # 已經是列表格式
            image_list = []
            for img in images:
                if img.dim() == 2:
                    # 2D影像，添加通道維度並複製為三通道 [3, H, W]
                    image_list.append(img.unsqueeze(0).repeat(3, 1, 1))
                elif img.dim() == 3 and img.size(0) == 1:
                    # 單通道，複製為三通道
                    image_list.append(img.repeat(3, 1, 1))
                elif img.dim() == 3 and img.size(0) == 3:
                    # 已經是三通道，直接使用
                    image_list.append(img)
                else:
                    raise ValueError(f"不支援的影像維度: {img.shape}")
        else:
            raise ValueError(f"不支援的輸入格式: {type(images)}")
        
        return self.model(image_list, targets)
    
    def predict(self, images, score_threshold=0.5):
        """
        推論模式，返回過濾後的檢測結果
        
        Args:
            images: 輸入影像
            score_threshold: 置信度閾值
            
        Returns:
            檢測結果列表
        """
        self.eval()
        
        with torch.no_grad():
            predictions = self.forward(images)
        
        # 過濾低置信度的檢測結果
        filtered_predictions = []
        for pred in predictions:
            # 獲取高置信度的檢測結果
            high_conf_mask = pred['scores'] > score_threshold
            
            filtered_pred = {
                'boxes': pred['boxes'][high_conf_mask],
                'labels': pred['labels'][high_conf_mask],
                'scores': pred['scores'][high_conf_mask]
            }
            filtered_predictions.append(filtered_pred)
        
        return filtered_predictions

def create_faster_rcnn_model(num_classes=2, pretrained=True, **kwargs):
    """
    創建Faster R-CNN模型
    
    Args:
        num_classes: 類別數量（默認2：背景+病灶）
        pretrained: 是否使用預訓練權重
        **kwargs: 其他參數
        
    Returns:
        Faster R-CNN模型
    """
    model = FasterRCNN(num_classes=num_classes, pretrained=pretrained, **kwargs)
    return model

def create_faster_rcnn_from_classification(classification_model_path, num_classes=2):
    """
    從分類模型創建檢測模型（如果有預訓練的分類模型）
    
    Args:
        classification_model_path: 分類模型路徑
        num_classes: 檢測類別數量
        
    Returns:
        Faster R-CNN模型
    """
    # 目前直接創建新模型，因為Faster R-CNN和分類模型架構差異較大
    # 如果需要遷移學習，可以在這裡實現特徵提取器的權重轉移
    print(f"注意：Faster R-CNN暫不支援從分類模型轉移，將創建新模型")
    return create_faster_rcnn_model(num_classes=num_classes, pretrained=True)

class FasterRCNNLoss:
    """
    Faster R-CNN損失函數封裝
    """
    
    def __init__(self):
        pass
    
    def __call__(self, predictions, targets):
        """
        計算Faster R-CNN損失
        
        Args:
            predictions: 模型預測結果（包含損失字典）
            targets: 真實標註
            
        Returns:
            總損失
        """
        # Faster R-CNN在訓練模式下直接返回損失字典
        if isinstance(predictions, dict) and 'loss_classifier' in predictions:
            total_loss = (
                predictions['loss_classifier'] +
                predictions['loss_box_reg'] +
                predictions['loss_objectness'] +
                predictions['loss_rpn_box_reg']
            )
            return total_loss
        else:
            raise ValueError("預期在訓練模式下獲得損失字典")

def collate_fn(batch):
    """
    自定義的批次整理函數，用於處理不同尺寸的影像和標註
    
    Args:
        batch: 批次數據
        
    Returns:
        整理後的批次數據
    """
    images = []
    targets = []
    
    for sample in batch:
        images.append(sample['image'])
        
        # 構建Faster R-CNN期望的target格式
        target = {
            'boxes': sample['boxes'],  # [N, 4] 格式: [x1, y1, x2, y2]
            'labels': sample['labels']  # [N] 類別標籤
        }
        targets.append(target)
    
    return images, targets

# 測試程式碼
if __name__ == "__main__":
    # 測試模型創建
    model = create_faster_rcnn_model(num_classes=2)
    print(f"模型創建成功，類別數：{model.num_classes}")
    
    # 測試前向傳播（推論模式）
    model.eval()
    dummy_input = torch.randn(512, 512)  # 灰階影像，2D
    
    with torch.no_grad():
        outputs = model([dummy_input])  # 傳入影像列表
        print(f"推論輸出格式：{type(outputs)}")
        if outputs:
            print(f"檢測數量：{len(outputs[0]['boxes'])}")
    
    # 測試訓練模式
    model.train()
    dummy_targets = [{
        'boxes': torch.tensor([[100, 100, 200, 200]], dtype=torch.float32),
        'labels': torch.tensor([1], dtype=torch.int64),
        'image_id': torch.tensor([0])
    }]
    
    try:
        loss_dict = model([dummy_input], dummy_targets)  # 傳入影像列表
        print(f"訓練損失鍵：{list(loss_dict.keys())}")
        total_loss = sum(loss for loss in loss_dict.values())
        print(f"總損失：{total_loss:.4f}")
    except Exception as e:
        print(f"訓練模式測試錯誤：{e}")
