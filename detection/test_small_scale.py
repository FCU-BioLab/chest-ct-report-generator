#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小規模測試腳本
用於驗證移除抽樣後的程式是否正常運行
"""

import os
import sys

# 添加src路徑
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_small_scale():
    """小規模測試"""
    from faster_rcnn_dataset import CTDetectionDataset
    
    print("開始小規模測試...")
    
    data_root = os.path.join('..', 'datasets', 'splited_dataset')
    
    # 讀取前10位患者進行測試
    train_patients_file = os.path.join(data_root, 'train_patients.txt')
    with open(train_patients_file, 'r') as f:
        all_patients = [line.strip() for line in f.readlines() if line.strip()]
    
    test_patients = all_patients[:10]  # 只測試前10位患者
    print(f"測試患者: {test_patients}")
    
    try:
        # 創建小規模資料集
        print("創建資料集...")
        dataset = CTDetectionDataset(
            data_root=data_root,
            split='train',
            target_size=512,
            specific_patients=test_patients
        )
        
        print(f"✅ 成功載入 {len(dataset)} 個樣本")
        
        # 測試第一個樣本
        if len(dataset) > 0:
            print("測試第一個樣本...")
            sample = dataset[0]
            print(f"✅ 樣本載入成功")
            print(f"  影像形狀: {sample['image'].shape}")
            print(f"  目標鍵值: {list(sample['target'].keys())}")
            print(f"  患者ID: {sample['patient_id']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_small_scale()
    if success:
        print("\n✅ 小規模測試通過！可以嘗試完整訓練。")
    else:
        print("\n❌ 小規模測試失敗！需要進一步調試。")
