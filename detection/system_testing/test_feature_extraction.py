#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試深層特徵提取功能
簡單的驗證腳本，確保特徵提取流程正常工作

作者: GitHub Copilot
日期: 2025-09-03
"""

import os
import sys
import torch
import logging
from pathlib import Path

# 添加當前目錄到路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def test_feature_extraction():
    """測試特徵提取功能"""
    
    # 設置日誌
    logging.basicConfig(level=logging.INFO)
    
    print("=== 測試深層特徵提取功能 ===")
    
    # 檢查GPU可用性
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用設備: {device}")
    
    # 尋找模型文件
    print("\n1. 尋找模型文件...")
    model_paths = [
        'Simple_Training_20250901_120146/models/best_model.pth',
        'Simple_Training_20250830_100356/models/best_model.pth',
        'Faster_RCNN_Detection/models/best_model_fold_1.pth',
    ]
    
    model_path = None
    for path in model_paths:
        full_path = os.path.join(current_dir, path)
        if os.path.exists(full_path):
            model_path = full_path
            print(f"✅ 找到模型: {path}")
            break
        else:
            print(f"❌ 模型不存在: {path}")
    
    if not model_path:
        print("❌ 未找到任何模型文件，無法繼續測試")
        return False
    
    # 尋找數據目錄
    print("\n2. 尋找數據目錄...")
    project_root = os.path.dirname(current_dir)
    data_paths = [
        os.path.join(project_root, 'datasets', 'splited_dataset'),
        os.path.join(project_root, 'datasets', 'all_patient_data'),
        os.path.join(project_root, 'datasets'),
    ]
    
    data_dir = None
    for path in data_paths:
        if os.path.exists(path):
            data_dir = path
            print(f"✅ 找到數據目錄: {path}")
            break
        else:
            print(f"❌ 數據目錄不存在: {path}")
    
    if not data_dir:
        print("❌ 未找到數據目錄，無法繼續測試")
        return False
    
    # 測試特徵提取器導入
    print("\n3. 測試模塊導入...")
    try:
        from deep_feature_extractor import DeepFeatureExtractor, extract_features_from_dataset
        print("✅ deep_feature_extractor 導入成功")
    except Exception as e:
        print(f"❌ deep_feature_extractor 導入失敗: {str(e)}")
        return False
    
    try:
        from feature_loader import FeatureLoader, FeatureVisualizer
        print("✅ feature_loader 導入成功")
    except Exception as e:
        print(f"❌ feature_loader 導入失敗: {str(e)}")
        return False
    
    # 測試模型加載
    print("\n4. 測試模型加載...")
    try:
        extractor = DeepFeatureExtractor(model_path, device)
        print("✅ 模型加載成功")
    except Exception as e:
        print(f"❌ 模型加載失敗: {str(e)}")
        return False
    
    # 測試數據集加載
    print("\n5. 測試數據集加載...")
    try:
        from faster_rcnn_dataset import CTDetectionDataset
        
        # 檢查可用的數據集分割
        data_dir_abs = os.path.abspath(data_dir)
        available_splits = []
        
        if os.path.exists(data_dir_abs):
            for item in os.listdir(data_dir_abs):
                if item.endswith('_patients.txt'):
                    split_name = item.replace('_patients.txt', '')
                    available_splits.append(split_name)
                elif os.path.isdir(os.path.join(data_dir_abs, item)) and item in ['train', 'test', 'val']:
                    available_splits.append(item)
        
        print(f"🔍 可用分割: {available_splits}")
        
        # 選擇一個可用的分割進行測試
        test_split = None
        for split in ['test', 'train', 'val']:
            if split in available_splits:
                test_split = split
                break
        
        if not test_split:
            print("❌ 未找到可用的數據集分割")
            return False
            
        print(f"   使用分割: {test_split}")
        
        dataset = CTDetectionDataset(
            data_root=data_dir,
            split=test_split,
            transforms=None,
            include_negative_samples=True,
            max_negative_per_patient=2  # 限制樣本數進行快速測試
        )
        print(f"✅ 數據集加載成功，共 {len(dataset)} 個樣本")
        
        if len(dataset) == 0:
            print("❌ 數據集為空，無法繼續測試")
            return False
            
    except Exception as e:
        print(f"❌ 數據集加載失敗: {str(e)}")
        return False
    
    # 測試單個樣本的特徵提取
    print("\n6. 測試單個樣本特徵提取...")
    try:
        sample = dataset[0]
        image = sample['image']
        patient_id = sample['patient_id']
        
        print(f"測試病例: {patient_id}")
        print(f"圖像形狀: {image.shape}")
        
        # 提取全局特徵
        global_features = extractor.extract_global_features(image)
        print(f"✅ 全局特徵提取成功，特徵類型: {list(global_features.keys())}")
        
        # 提取檢測特徵
        detection_features = extractor.extract_detection_features(image)
        print(f"✅ 檢測特徵提取成功，檢測到 {detection_features['num_detections']} 個病灶")
        
    except Exception as e:
        print(f"❌ 單個樣本特徵提取失敗: {str(e)}")
        return False
    
    # 測試小規模特徵提取
    print("\n7. 測試小規模特徵提取（前3個病例）...")
    try:
        test_save_dir = os.path.join(current_dir, 'test_features')
        
        # 組織測試數據（取前幾個病例）
        patient_data = {}
        max_patients = 3
        max_slices_per_patient = 5
        
        for i in range(min(len(dataset), max_patients * max_slices_per_patient)):
            sample = dataset[i]
            patient_id = sample['patient_id']
            
            if patient_id not in patient_data:
                patient_data[patient_id] = []
            
            if len(patient_data[patient_id]) < max_slices_per_patient:
                patient_data[patient_id].append(sample['image'])
            
            # 限制病例數
            if len(patient_data) >= max_patients:
                break
        
        print(f"測試病例: {list(patient_data.keys())}")
        
        # 提取病例特徵
        for patient_id, images in patient_data.items():
            print(f"處理病例 {patient_id} ({len(images)} 張切片)...")
            patient_features = extractor.extract_patient_features(images, patient_id)
            
            # 簡單驗證特徵結構
            assert 'patient_id' in patient_features
            assert 'num_slices' in patient_features
            assert 'global_features' in patient_features
            assert 'detection_features' in patient_features
            print(f"✅ 病例 {patient_id} 特徵提取成功")
        
        print("✅ 小規模特徵提取測試通過")
        
    except Exception as e:
        print(f"❌ 小規模特徵提取失敗: {str(e)}")
        return False
    
    # 測試特徵加載
    print("\n8. 測試特徵保存和加載...")
    try:
        import pickle
        import json
        
        # 保存測試特徵
        os.makedirs(test_save_dir, exist_ok=True)
        test_patient_id = list(patient_data.keys())[0]
        test_features = extractor.extract_patient_features(
            patient_data[test_patient_id], test_patient_id
        )
        
        # 創建病例資料夾
        patient_test_dir = os.path.join(test_save_dir, test_patient_id)
        os.makedirs(patient_test_dir, exist_ok=True)
        
        # 保存pkl格式
        pkl_path = os.path.join(patient_test_dir, f"{test_patient_id}_features.pkl")
        with open(pkl_path, 'wb') as f:
            pickle.dump(test_features, f)
        print(f"✅ 特徵保存成功: {pkl_path}")
        
        # 測試加載
        with open(pkl_path, 'rb') as f:
            loaded_features = pickle.load(f)
        
        assert loaded_features['patient_id'] == test_patient_id
        print("✅ 特徵加載驗證成功")
        
        # 測試FeatureLoader
        loader = FeatureLoader(test_save_dir)
        patient_features = loader.get_patient_features(test_patient_id)
        assert patient_features is not None
        
        # 測試LLM特徵生成
        llm_prompt = loader.generate_llm_prompt_features(test_patient_id)
        assert len(llm_prompt) > 0
        print("✅ LLM特徵生成成功")
        print(f"LLM特徵長度: {len(llm_prompt)} 字符")
        
    except Exception as e:
        print(f"❌ 特徵保存/加載測試失敗: {str(e)}")
        return False
    
    print("\n🎉 所有測試通過！深層特徵提取功能正常工作")
    
    # 清理測試文件
    try:
        import shutil
        if os.path.exists(test_save_dir):
            shutil.rmtree(test_save_dir)
        print("✅ 測試文件清理完成")
    except:
        print("⚠️ 測試文件清理失敗，請手動刪除 test_features 目錄")
    
    return True

def main():
    """主函數"""
    print("深層特徵提取功能測試")
    print("=" * 50)
    
    success = test_feature_extraction()
    
    if success:
        print("\n✅ 測試結果: 通過")
        print("\n接下來你可以：")
        print("1. 運行 python extract_features.py 進行交互式特徵提取")
        print("2. 運行 python test_detection.py --extract_features 進行完整測試+特徵提取")
        print("3. 查看 README_FEATURES.md 了解詳細用法")
        return 0
    else:
        print("\n❌ 測試結果: 失敗")
        print("\n請檢查：")
        print("1. 模型文件是否存在")
        print("2. 數據集是否正確配置")
        print("3. 相關依賴是否安裝")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
