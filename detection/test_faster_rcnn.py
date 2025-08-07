#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試 Faster R-CNN 模型和資料集的簡單腳本

用於驗證修改後的程式碼是否正常工作
"""

import sys
import os
import torch

# 添加路徑
sys.path.append(os.path.dirname(__file__))

def test_faster_rcnn_model():
    """測試Faster R-CNN模型"""
    print("="*50)
    print("測試 Faster R-CNN 模型")
    print("="*50)
    
    try:
        from faster_rcnn_model import create_faster_rcnn_model
        
        # 創建模型
        model = create_faster_rcnn_model(num_classes=2)
        print(f"✅ 模型創建成功，類別數：{model.num_classes}")
        
        # 測試推論模式
        model.eval()
        dummy_input = torch.randn(1, 512, 512)  # 灰階影像，單通道
        
        with torch.no_grad():
            outputs = model([dummy_input])  # 傳入影像列表
            print(f"✅ 推論模式測試成功")
            if outputs:
                print(f"   檢測結果數量：{len(outputs[0]['boxes'])}")
        
        # 測試訓練模式
        model.train()
        dummy_targets = [{
            'boxes': torch.tensor([[100, 100, 200, 200]], dtype=torch.float32),
            'labels': torch.tensor([1], dtype=torch.int64),
            'image_id': torch.tensor([0])
        }]
        
        loss_dict = model([dummy_input], dummy_targets)  # 傳入影像列表
        print(f"✅ 訓練模式測試成功")
        print(f"   損失項目：{list(loss_dict.keys())}")
        total_loss = sum(loss for loss in loss_dict.values())
        print(f"   總損失：{total_loss:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型測試失敗：{e}")
        return False

def test_faster_rcnn_dataset():
    """測試Faster R-CNN資料集"""
    print("\n" + "="*50)
    print("測試 Faster R-CNN 資料集")
    print("="*50)
    
    try:
        from faster_rcnn_dataset import CTDetectionDataset, collate_fn
        
        # 測試資料集創建（使用假資料路徑）
        data_root = "../../datasets/splited_dataset"
        
        if not os.path.exists(data_root):
            print(f"⚠️  資料路徑不存在：{data_root}")
            print("   創建空資料集測試...")
            
            # 創建資料集（即使路徑不存在也應該能初始化）
            dataset = CTDetectionDataset(
                data_root=data_root,
                split='train',
                target_size=512
            )
            print(f"✅ 資料集創建成功，樣本數：{len(dataset)}")
            
        else:
            dataset = CTDetectionDataset(
                data_root=data_root,
                split='train',
                target_size=512
            )
            print(f"✅ 資料集載入成功，樣本數：{len(dataset)}")
            
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"   樣本格式檢查：")
                print(f"   - 影像形狀：{sample['image'].shape}")
                print(f"   - 邊界框數量：{len(sample['target']['boxes'])}")
                print(f"   - 標籤：{sample['target']['labels']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 資料集測試失敗：{e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_script():
    """測試訓練腳本導入"""
    print("\n" + "="*50)
    print("測試訓練腳本導入")
    print("="*50)
    
    try:
        # 測試主要函數導入
        from train_detection import (
            evaluate_detection_model,
            visualize_predictions,
            calculate_detection_metrics,
            create_kfold_datasets
        )
        print("✅ 訓練腳本函數導入成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 訓練腳本測試失敗：{e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主測試函數"""
    print("🧪 開始測試 Faster R-CNN 實現")
    print("=" * 70)
    
    results = []
    
    # 測試模型
    results.append(test_faster_rcnn_model())
    
    # 測試資料集
    results.append(test_faster_rcnn_dataset())
    
    # 測試訓練腳本
    results.append(test_training_script())
    
    # 總結
    print("\n" + "="*70)
    print("🏁 測試結果總結")
    print("="*70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"通過測試：{passed}/{total}")
    
    if passed == total:
        print("✅ 所有測試通過！程式碼修改成功")
        print("\n📋 主要變更：")
        print("1. ✅ 模型架構：CT-ViT → Faster R-CNN")
        print("2. ✅ 分類任務：多分類 → 二分類（背景 vs 病灶）")
        print("3. ✅ 評估指標：準確率 → 精度/召回率/F1分數")
        print("4. ✅ 資料格式：自訂格式 → Faster R-CNN標準格式")
        print("5. ✅ 邊界框：中心點格式 → 左上右下格式")
        
        print("\n🚀 建議使用方式：")
        print("   python train_detection.py --mode traditional --num_epochs 20")
        print("   python train_detection.py --mode kfold --k_folds 3")
    else:
        print("❌ 部分測試失敗，請檢查錯誤訊息")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
