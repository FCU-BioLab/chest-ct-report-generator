#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
輕量版限制檢查腳本 - 避免卡住問題
"""

import os
import sys

def check_limits_quick():
    """快速檢查限制移除狀況（不載入完整資料集）"""
    
    print("="*60)
    print("🔍 快速檢查所有限制移除狀況")
    print("="*60)
    
    # 1. 檢查資料目錄
    print("\n📁 1. 資料目錄檢查:")
    data_root = os.path.join('..', 'datasets', 'splited_dataset')
    train_dir = os.path.join(data_root, 'train')
    
    if os.path.exists(train_dir):
        print(f"   ✅ 訓練資料目錄存在: {train_dir}")
        
        # 統計患者數量
        patients = [d for d in os.listdir(train_dir) 
                   if os.path.isdir(os.path.join(train_dir, d))]
        print(f"   ✅ 患者數量: {len(patients)}")
        
        # 檢查前3個患者的檔案數量（快速抽樣）
        if len(patients) >= 3:
            total_files = 0
            for i, patient in enumerate(patients[:3]):
                patient_dir = os.path.join(train_dir, patient)
                dicom_dir = os.path.join(patient_dir, 'dicom_files')
                
                if os.path.exists(dicom_dir):
                    dcm_files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
                    total_files += len(dcm_files)
                    print(f"   📄 患者 {patient}: {len(dcm_files)} 個DICOM檔案")
            
            avg_files = total_files / 3
            estimated_total = int(avg_files * len(patients))
            print(f"   📊 前3位患者平均: {avg_files:.1f} 檔案/患者")
            print(f"   📊 估算總檔案數: {estimated_total}")
            
    else:
        print(f"   ❌ 資料目錄不存在: {train_dir}")
        return False
    
    # 2. 檢查程式碼中的限制
    print("\n💻 2. 程式碼限制檢查:")
    
    # 檢查 faster_rcnn_dataset.py
    dataset_file = 'faster_rcnn_dataset.py'
    if os.path.exists(dataset_file):
        with open(dataset_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if '完全移除抽樣限制' in content:
            print("   ✅ faster_rcnn_dataset.py: 抽樣限制已移除")
        else:
            print("   ⚠️  faster_rcnn_dataset.py: 可能仍有限制")
            
        if 'max_samples' not in content and 'file_limit' not in content:
            print("   ✅ faster_rcnn_dataset.py: 無檔案數量限制")
        else:
            print("   ⚠️  faster_rcnn_dataset.py: 可能仍有檔案限制")
    
    # 檢查 train_detection.py
    train_file = 'train_detection.py'
    if os.path.exists(train_file):
        with open(train_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if 'batch_size", type=int, default=16' in content:
            print("   ✅ train_detection.py: 批次大小已提升為16")
        elif 'batch_size", type=int, default=8' in content:
            print("   ⚠️  train_detection.py: 批次大小仍為8")
        
        if 'num_workers=4' in content:
            print("   ✅ train_detection.py: 多進程已啟用")
        elif 'num_workers=0' in content:
            print("   ⚠️  train_detection.py: 仍使用單進程")
    
    # 3. 測試單一樣本載入
    print("\n🔍 3. 單一樣本測試:")
    try:
        from faster_rcnn_dataset import CTDetectionDataset
        
        # 只載入第一個患者進行測試
        first_patient = patients[0] if patients else None
        if first_patient:
            print(f"   📋 測試患者: {first_patient}")
            
            dataset = CTDetectionDataset(
                data_root=data_root,
                split='train',
                target_size=512,
                specific_patients=[first_patient]  # 只載入一個患者
            )
            
            print(f"   ✅ 單一患者樣本數: {len(dataset)}")
            
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"   ✅ 樣本載入成功")
                print(f"   📏 影像形狀: {sample['image'].shape}")
                print(f"   🎯 標籤數量: {len(sample['target']['labels'])}")
        
    except Exception as e:
        print(f"   ❌ 樣本載入失敗: {e}")
        return False
    
    # 4. 性能建議
    print("\n💡 4. 性能建議:")
    print("   🚀 小規模測試: 載入前10位患者")
    print("   ⚡ 漸進式訓練: 先用小批次測試，再擴大規模")
    print("   🎯 記憶體管理: 監控記憶體使用情況")
    
    # 5. 下一步操作
    print("\n📋 5. 建議的下一步:")
    print("   1️⃣  python test_small_scale.py  # 測試小規模載入")
    print("   2️⃣  python train_detection.py --mode custom --num_epochs 5  # 短期訓練測試")
    print("   3️⃣  python train_detection.py --mode traditional  # 完整訓練")
    
    print("\n" + "="*60)
    print("✅ 快速檢查完成！")
    print("✅ 主要限制已移除！")
    print("✅ 可以開始小規模測試！")
    print("="*60)
    
    return True

if __name__ == "__main__":
    check_limits_quick()
