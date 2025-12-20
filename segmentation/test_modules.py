#!/usr/bin/env python3
"""
快速測試腳本 - 驗證模組是否正常工作
"""

import sys
from pathlib import Path

def test_imports():
    """測試所有模組是否能正常匯入"""
    print("🧪 測試模組匯入...")
    
    try:
        from finetune_medsam2 import (
            ChestTumorDataset,
            DataAugmentation,
            MedSAM2Trainer,
            DiceLoss,
            CombinedLoss,
            setup_logging,
            split_dataset,
            compute_all_metrics
        )
        print("✅ 所有模組匯入成功！")
        return True
    except ImportError as e:
        print(f"❌ 模組匯入失敗: {e}")
        return False


def test_loss_functions():
    """測試損失函數"""
    print("\n🧪 測試損失函數...")
    
    try:
        import torch
        from finetune_medsam2.losses import DiceLoss, CombinedLoss
        
        # 創建測試資料
        pred = torch.randn(1, 256, 256)
        target = torch.randint(0, 2, (1, 256, 256)).float()
        
        # 測試 Dice Loss
        dice_loss = DiceLoss()
        loss_value = dice_loss(torch.sigmoid(pred), target)
        print(f"  Dice Loss: {loss_value.item():.4f}")
        
        # 測試 Combined Loss
        combined_loss = CombinedLoss()
        loss_value = combined_loss(pred, target)
        print(f"  Combined Loss: {loss_value.item():.4f}")
        
        print("✅ 損失函數測試通過！")
        return True
    except Exception as e:
        print(f"❌ 損失函數測試失敗: {e}")
        return False


def test_metrics():
    """測試評估指標"""
    print("\n🧪 測試評估指標...")
    
    try:
        import torch
        from finetune_medsam2.utils import compute_all_metrics
        
        # 創建測試資料
        pred = torch.randn(256, 256)
        target = torch.randint(0, 2, (256, 256)).float()
        
        # 計算指標
        metrics = compute_all_metrics(pred, target)
        
        print("  評估指標:")
        for key, value in metrics.items():
            print(f"    {key}: {value:.4f}")
        
        print("✅ 評估指標測試通過！")
        return True
    except Exception as e:
        print(f"❌ 評估指標測試失敗: {e}")
        return False


def test_dataset_split():
    """測試資料集分割"""
    print("\n🧪 測試資料集分割...")
    
    try:
        from finetune_medsam2.utils import split_dataset
        
        # 創建測試患者 ID
        patient_ids = [f"patient_{i:03d}" for i in range(100)]
        
        # 分割資料集
        train_ids, val_ids, test_ids = split_dataset(
            patient_ids, 
            train_ratio=0.7, 
            val_ratio=0.15, 
            test_ratio=0.15,
            seed=42
        )
        
        print(f"  Train: {len(train_ids)} 個")
        print(f"  Val: {len(val_ids)} 個")
        print(f"  Test: {len(test_ids)} 個")
        print(f"  總計: {len(train_ids) + len(val_ids) + len(test_ids)} 個")
        
        # 驗證沒有重疊
        all_ids = set(train_ids) | set(val_ids) | set(test_ids)
        assert len(all_ids) == len(patient_ids), "資料集分割有重疊！"
        
        print("✅ 資料集分割測試通過！")
        return True
    except Exception as e:
        print(f"❌ 資料集分割測試失敗: {e}")
        return False


def test_logging():
    """測試日誌系統"""
    print("\n🧪 測試日誌系統...")
    
    try:
        from finetune_medsam2.utils import setup_logging
        import tempfile
        import logging
        
        # 使用臨時目錄
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_logging(log_dir=tmpdir)
            logger.info("測試日誌訊息")
            
            # ✅ 修正：在檢查檔案前先關閉 handlers
            # 清除所有 handlers 以釋放檔案鎖定
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
            
            # 檢查是否創建 log 檔案
            log_files = list(Path(tmpdir).glob("*.log"))
            assert len(log_files) == 1, f"應該只有 1 個 log 檔案，實際有 {len(log_files)} 個"
            
            print(f"  Log 檔案: {log_files[0].name}")
            print(f"  Log 檔案大小: {log_files[0].stat().st_size} bytes")
        
        print("✅ 日誌系統測試通過！")
        return True
    except Exception as e:
        print(f"❌ 日誌系統測試失敗: {e}")
        return False


def main():
    """執行所有測試"""
    print("="*80)
    print("MedSAM2 Fine-tuning 模組測試")
    print("="*80)
    
    results = []
    
    # 執行測試
    results.append(("模組匯入", test_imports()))
    results.append(("損失函數", test_loss_functions()))
    results.append(("評估指標", test_metrics()))
    results.append(("資料集分割", test_dataset_split()))
    results.append(("日誌系統", test_logging()))
    
    # 輸出總結
    print("\n" + "="*80)
    print("測試總結")
    print("="*80)
    
    for name, passed in results:
        status = "✅ 通過" if passed else "❌ 失敗"
        print(f"{name:15s}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("="*80)
    if all_passed:
        print("🎉 所有測試通過！系統已準備就緒。")
        return 0
    else:
        print("⚠️ 部分測試失敗，請檢查錯誤訊息。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
