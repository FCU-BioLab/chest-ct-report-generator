#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FPS-Former 模型測試腳本
測試FPS-Former模型的基本功能

作者: GitHub Copilot
日期: 2025-08-02
"""

import torch
import numpy as np
from fps_former_model import FPSFormerForDetection, create_fps_former_detection_model

def test_fps_former_model():
    """測試FPS-Former模型"""
    print("="*60)
    print("🧪 FPS-Former 模型測試")
    print("="*60)
    
    # 檢查設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    try:
        # 創建模型
        print("\n1. 創建FPS-Former模型...")
        model = create_fps_former_detection_model(num_classes=5, image_size=224)
        model = model.to(device)
        
        # 計算參數數量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   ✅ 模型創建成功")
        print(f"   📊 總參數數量: {total_params:,}")
        print(f"   🎯 可訓練參數: {trainable_params:,}")
        print(f"   💾 模型大小約: {total_params * 4 / 1024 / 1024:.1f} MB")
        
        # 測試前向傳播
        print("\n2. 測試前向傳播...")
        batch_size = 2
        pixel_values = torch.randn(batch_size, 3, 224, 224).to(device)
        labels = torch.tensor([1, 2]).to(device)
        bbox_targets = torch.tensor([[0.3, 0.4, 0.2, 0.3], [0.5, 0.6, 0.15, 0.25]]).to(device)
        
        # 前向傳播
        with torch.no_grad():
            outputs = model(pixel_values, labels, bbox_targets)
        
        print(f"   ✅ 前向傳播成功")
        print(f"   📏 輸出形狀:")
        print(f"      - 分類邏輯: {outputs['class_logits'].shape}")
        print(f"      - 邊界框預測: {outputs['bbox_pred'].shape}")
        print(f"      - 物件存在性: {outputs['objectness_logits'].shape}")
        print(f"      - 損失值: {outputs['loss']:.4f}")
        
        # 測試推理模式
        print("\n3. 測試推理模式...")
        model.eval()
        with torch.no_grad():
            inference_outputs = model(pixel_values)
        
        # 檢查輸出範圍
        class_probs = torch.softmax(inference_outputs['class_logits'], dim=-1)
        bbox_pred = inference_outputs['bbox_pred']
        objectness_prob = torch.sigmoid(inference_outputs['objectness_logits'])
        
        print(f"   ✅ 推理模式成功")
        print(f"   📊 輸出檢查:")
        print(f"      - 分類概率範圍: [{class_probs.min():.3f}, {class_probs.max():.3f}]")
        print(f"      - 邊界框範圍: [{bbox_pred.min():.3f}, {bbox_pred.max():.3f}]")
        print(f"      - 物件概率範圍: [{objectness_prob.min():.3f}, {objectness_prob.max():.3f}]")
        
        # 測試記憶體使用
        print("\n4. 記憶體使用測試...")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_used = torch.cuda.memory_allocated() / 1024 / 1024
            print(f"   📊 GPU記憶體使用: {memory_used:.1f} MB")
        
        print("\n✅ 所有測試通過！")
        print("🎉 FPS-Former模型運行正常")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_with_random_baseline():
    """與隨機基線比較"""
    print("\n" + "="*60)
    print("📊 與隨機基線比較")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 創建模型
    model = create_fps_former_detection_model(num_classes=5)
    model = model.to(device)
    model.eval()
    
    # 創建測試數據
    batch_size = 10
    pixel_values = torch.randn(batch_size, 3, 224, 224).to(device)
    
    with torch.no_grad():
        outputs = model(pixel_values)
        class_probs = torch.softmax(outputs['class_logits'], dim=-1)
        
        # 計算預測分佈（移到CPU進行計算）
        predicted_classes = torch.argmax(class_probs, dim=-1).cpu()
        class_distribution = torch.bincount(predicted_classes, minlength=5).float()
        class_distribution = class_distribution / class_distribution.sum()
        
        print("預測類別分佈:")
        for i, prob in enumerate(class_distribution):
            print(f"   類別 {i}: {prob:.3f}")
        
        # 隨機基線 (均勻分佈) - 確保在CPU上
        random_baseline = torch.tensor([0.2] * 5)
        
        # 計算KL散度
        kl_div = torch.nn.functional.kl_div(
            torch.log(class_distribution + 1e-8),
            random_baseline,
            reduction='sum'
        )
        
        print(f"\n與隨機基線的KL散度: {kl_div:.4f}")
        print("（較低的值表示模型還需要更多訓練）")

if __name__ == "__main__":
    # 運行測試
    success = test_fps_former_model()
    
    if success:
        compare_with_random_baseline()
        
        print("\n" + "="*60)
        print("🎯 下一步建議:")
        print("1. 運行訓練腳本: python train_detection.py --mode traditional")
        print("2. 或進行K-Fold評估: python train_detection.py --mode kfold")
        print("3. 調整超參數進行優化")
        print("="*60)
    else:
        print("\n❌ 請檢查模型實現或依賴包安裝")
