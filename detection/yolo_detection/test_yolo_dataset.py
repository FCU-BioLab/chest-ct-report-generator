#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試修改後的YOLOv11訓練腳本是否正常工作
"""

import sys
from pathlib import Path

# 添加父目錄到路徑
sys.path.append(str(Path(__file__).resolve().parent.parent))

def test_imports():
    """測試模組導入"""
    print("🔍 測試模組導入...")
    
    try:
        from train_yolov11 import YOLOv11CTDataset, TrainingConfig
        print("✅ YOLOv11CTDataset 導入成功")
        print("✅ TrainingConfig 導入成功")
        return True
    except ImportError as e:
        print(f"❌ 導入失敗: {e}")
        return False

def test_dataset_creation():
    """測試數據集創建"""
    print("\n🔍 測試數據集創建...")
    
    try:
        from train_yolov11 import YOLOv11CTDataset
        
        # 使用相對路徑指向數據集
        data_dir = "../../datasets/splited_dataset"
        
        # 創建一個小的測試數據集
        dataset = YOLOv11CTDataset(
            data_dir=data_dir,
            split="train",
            include_negative_samples=False,  # 只載入正樣本以加快測試
            max_negative_per_patient=0,
            patient_ids=["A0001", "A0002"],  # 只測試兩個患者
            image_size=512
        )
        
        print(f"✅ 數據集創建成功")
        print(f"   數據集大小: {len(dataset.samples)}")
        
        if len(dataset.samples) > 0:
            # 測試單個樣本
            sample = dataset.rcnn_dataset[0]
            print(f"   樣本測試:")
            print(f"   - 影像形狀: {sample['image'].shape}")
            print(f"   - 目標格式: {type(sample['target'])}")
            print(f"   - 患者ID: {sample['patient_id']}")
            
            # 檢查YOLO格式
            target = sample['target']
            if 'boxes' in target and len(target['boxes']) > 0:
                print(f"   - YOLO框格式: {target['boxes'].shape}")
                print(f"   - 標籤: {target['labels']}")
            else:
                print(f"   - 無標註（負樣本）")
        
        return True
        
    except Exception as e:
        print(f"❌ 數據集創建失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_creation():
    """測試配置創建"""
    print("\n🔍 測試配置創建...")
    
    try:
        from train_yolov11 import TrainingConfig
        
        config = TrainingConfig(
            data_dir="../../datasets/splited_dataset",
            num_epochs=1,  # 短訓練用於測試
            batch_size=2,
            val_ratio=0.2
        )
        
        print("✅ 配置創建成功")
        print(f"   數據目錄: {config.data_dir}")
        print(f"   訓練輪數: {config.num_epochs}")
        print(f"   批次大小: {config.batch_size}")
        print(f"   驗證比例: {config.val_ratio}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置創建失敗: {e}")
        return False

def main():
    """主測試函數"""
    print("=" * 60)
    print("YOLOv11 訓練腳本修改測試")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_dataset_creation,
        test_config_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"測試結果: {passed}/{total} 通過")
    
    if passed == total:
        print("🎉 所有測試通過！YOLOv11 訓練腳本修改成功")
    else:
        print("⚠️ 部分測試失敗，需要進一步檢查")
    
    print("=" * 60)

if __name__ == "__main__":
    main()