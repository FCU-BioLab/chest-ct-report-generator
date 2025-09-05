#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
簡化的特徵提取測試
只測試核心功能，避免複雜的數據加載

作者: GitHub Copilot
日期: 2025-09-03
"""

import os
import sys
import torch
import logging
import numpy as np

# 添加當前目錄到路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def test_simple_feature_extraction():
    """簡化的特徵提取測試"""
    
    # 設置日誌
    logging.basicConfig(level=logging.INFO)
    
    print("=== 簡化特徵提取測試 ===")
    
    # 檢查GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用設備: {device}")
    
    # 尋找模型
    model_paths = [
        'Simple_Training_20250901_120146/models/best_model.pth',
        'Simple_Training_20250830_100356/models/best_model.pth',
    ]
    
    model_path = None
    for path in model_paths:
        full_path = os.path.join(current_dir, path)
        if os.path.exists(full_path):
            model_path = full_path
            print(f"✅ 找到模型: {path}")
            break
    
    if not model_path:
        print("❌ 未找到模型文件")
        return False
    
    try:
        # 導入特徵提取器
        from deep_feature_extractor import DeepFeatureExtractor
        
        # 初始化提取器
        print("\n1. 初始化特徵提取器...")
        extractor = DeepFeatureExtractor(model_path, device)
        print("✅ 特徵提取器初始化成功")
        
        # 創建測試圖像
        print("\n2. 創建測試圖像...")
        test_image = torch.randn(1, 512, 512)  # 模擬CT切片
        print(f"✅ 測試圖像創建: {test_image.shape}")
        
        # 測試全局特徵提取
        print("\n3. 測試全局特徵提取...")
        global_features = extractor.extract_global_features(test_image)
        print(f"✅ 全局特徵提取成功")
        print(f"   特徵類型: {list(global_features.keys())}")
        
        for feat_name, feat_value in global_features.items():
            if isinstance(feat_value, dict):
                print(f"   {feat_name}: 字典類型，包含 {list(feat_value.keys())}")
            elif isinstance(feat_value, torch.Tensor):
                print(f"   {feat_name}: 張量形狀 {feat_value.shape}")
            else:
                print(f"   {feat_name}: {type(feat_value)}")
        
        # 測試檢測特徵提取
        print("\n4. 測試檢測特徵提取...")
        detection_features = extractor.extract_detection_features(test_image)
        print(f"✅ 檢測特徵提取成功")
        print(f"   檢測到病灶數: {detection_features['num_detections']}")
        
        # 測試病例級特徵提取
        print("\n5. 測試病例級特徵提取...")
        test_images = [test_image for _ in range(3)]  # 模擬3張切片
        patient_features = extractor.extract_patient_features(test_images, "TEST_PATIENT", 0.5)
        print(f"✅ 病例級特徵提取成功")
        print(f"   病例ID: {patient_features['patient_id']}")
        print(f"   切片數: {patient_features['num_slices']}")
        print(f"   全局特徵類型: {list(patient_features['global_features'].keys())}")
        
        # 檢查特徵結構
        print("\n6. 檢查特徵結構...")
        for feat_name, feat_data in patient_features['global_features'].items():
            if isinstance(feat_data, dict):
                if feat_name == 'fpn':
                    print(f"   {feat_name}: FPN特徵，包含 {list(feat_data.keys())}")
                    for fpn_key, fpn_stats in feat_data.items():
                        if isinstance(fpn_stats, dict) and 'mean' in fpn_stats:
                            print(f"     {fpn_key}: 統計特徵 (mean維度: {fpn_stats['mean'].shape})")
                else:
                    if 'mean' in feat_data:
                        print(f"   {feat_name}: 統計特徵 (mean維度: {feat_data['mean'].shape})")
                    else:
                        print(f"   {feat_name}: 其他字典類型")
            else:
                print(f"   {feat_name}: {type(feat_data)}")
        
        print("\n✅ 所有測試通過！")
        return True
        
    except Exception as e:
        print(f"\n❌ 測試失敗: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函數"""
    print("簡化特徵提取測試")
    print("=" * 30)
    
    success = test_simple_feature_extraction()
    
    if success:
        print("\n🎉 測試成功！特徵提取功能正常工作")
        print("\n現在可以運行完整測試:")
        print("python detection\\test_detection.py --split test")
    else:
        print("\n💥 測試失敗，請檢查錯誤信息")
    
    return 0 if success else 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
