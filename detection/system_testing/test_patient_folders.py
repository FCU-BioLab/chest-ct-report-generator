#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試新的病例資料夾結構
驗證深層特徵是否正確保存在病例資料夾中

作者: GitHub Copilot
日期: 2025-09-03
"""

import os
import sys
import torch
import logging
from datetime import datetime

# 添加當前目錄到路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def test_patient_folder_structure():
    """測試病例資料夾結構"""
    
    # 設置日誌
    logging.basicConfig(level=logging.INFO)
    
    print("=== 測試病例資料夾結構 ===")
    print()
    
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
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        extractor = DeepFeatureExtractor(model_path, device)
        print("✅ 特徵提取器初始化成功")
        
        # 創建測試目錄
        test_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_save_dir = os.path.join(current_dir, f'test_patient_folders_{test_time}')
        
        # 模擬病例數據
        print("\n2. 模擬病例數據...")
        test_patients = {
            'TEST_PATIENT_A': [torch.randn(1, 512, 512) for _ in range(3)],
            'TEST_PATIENT_B': [torch.randn(1, 512, 512) for _ in range(2)],
        }
        
        print(f"模擬病例: {list(test_patients.keys())}")
        
        # 提取並保存特徵
        print("\n3. 提取並保存特徵...")
        for patient_id, images in test_patients.items():
            print(f"\n處理病例: {patient_id} ({len(images)} 張切片)")
            
            # 提取特徵
            patient_features = extractor.extract_patient_features(images, patient_id)
            
            # 保存到病例資料夾
            patient_dir = os.path.join(test_save_dir, patient_id)
            os.makedirs(patient_dir, exist_ok=True)
            
            # 保存pkl文件
            import pickle
            pkl_path = os.path.join(patient_dir, f"{patient_id}_features.pkl")
            with open(pkl_path, 'wb') as f:
                pickle.dump(patient_features, f)
            
            # 保存json文件
            import json
            from deep_feature_extractor import convert_features_to_json
            json_features = convert_features_to_json(patient_features)
            json_path = os.path.join(patient_dir, f"{patient_id}_features.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_features, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 保存成功:")
            print(f"   PKL: {pkl_path}")
            print(f"   JSON: {json_path}")
        
        # 驗證資料夾結構
        print("\n4. 驗證資料夾結構...")
        print(f"測試目錄: {test_save_dir}")
        
        for patient_id in test_patients.keys():
            patient_dir = os.path.join(test_save_dir, patient_id)
            pkl_file = os.path.join(patient_dir, f"{patient_id}_features.pkl")
            json_file = os.path.join(patient_dir, f"{patient_id}_features.json")
            
            print(f"\n病例 {patient_id}:")
            print(f"  資料夾: {'存在' if os.path.exists(patient_dir) else '不存在'}")
            print(f"  PKL文件: {'存在' if os.path.exists(pkl_file) else '不存在'}")
            print(f"  JSON文件: {'存在' if os.path.exists(json_file) else '不存在'}")
            
            if os.path.exists(pkl_file):
                file_size = os.path.getsize(pkl_file)
                print(f"  PKL文件大小: {file_size} bytes")
        
        # 測試加載器
        print("\n5. 測試特徵加載器...")
        from feature_loader import FeatureLoader
        
        loader = FeatureLoader(test_save_dir)
        loaded_patients = loader.get_all_patient_ids()
        
        print(f"加載器找到的病例: {loaded_patients}")
        
        for patient_id in loaded_patients:
            features = loader.get_patient_features(patient_id)
            if features:
                print(f"✅ 成功加載病例 {patient_id} 的特徵")
                print(f"   切片數: {features.get('num_slices', 0)}")
                print(f"   全局特徵類型: {list(features.get('global_features', {}).keys())}")
            else:
                print(f"❌ 加載病例 {patient_id} 失敗")
        
        # 顯示最終的目錄結構
        print(f"\n6. 最終目錄結構:")
        print(f"{test_save_dir}/")
        for item in os.listdir(test_save_dir):
            item_path = os.path.join(test_save_dir, item)
            if os.path.isdir(item_path):
                print(f"├── {item}/")
                for subitem in os.listdir(item_path):
                    print(f"│   ├── {subitem}")
            else:
                print(f"├── {item}")
        
        print("\n✅ 病例資料夾結構測試成功！")
        
        # 清理測試文件
        try:
            import shutil
            shutil.rmtree(test_save_dir)
            print("✅ 測試文件清理完成")
        except:
            print(f"⚠️ 請手動刪除測試目錄: {test_save_dir}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 測試失敗: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函數"""
    print("病例資料夾結構測試")
    print("=" * 30)
    
    success = test_patient_folder_structure()
    
    if success:
        print(f"\n🎉 測試成功！")
        print(f"\n新的資料夾結構:")
        print(f"deep_features/")
        print(f"├── A0001/")
        print(f"│   ├── A0001_features.pkl")
        print(f"│   ├── A0001_features.json")
        print(f"│   └── A0001_feature_report.md")
        print(f"├── A0002/")
        print(f"│   ├── A0002_features.pkl")
        print(f"│   ├── A0002_features.json")
        print(f"│   └── A0002_feature_report.md")
        print(f"└── feature_summary_report.md")
        
        print(f"\n現在可以運行:")
        print(f"python detection\\faster_rcnn_detection\\test_detection.py --split test")
    else:
        print(f"\n💥 測試失敗")
    
    return 0 if success else 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
