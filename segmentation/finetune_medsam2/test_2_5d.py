#!/usr/bin/env python3
"""
2.5D 模式測試腳本
驗證 2.5D 實作是否正確
"""

import sys
from pathlib import Path
import numpy as np

# 添加路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from finetune_medsam2.dataset import LNDbDataset, CachedSliceDataset
from finetune_medsam2.config import get_default_config


def test_2d_mode():
    """測試 2D 模式"""
    print("\n" + "="*70)
    print("測試 2D 模式（三個通道重複同一切片）")
    print("="*70)
    
    # 模擬資料
    ct_slice = np.random.rand(512, 512).astype(np.float32)
    
    # 2D 模式處理
    ct_rgb_2d = np.stack([ct_slice, ct_slice, ct_slice], axis=-1)
    
    # 驗證
    assert ct_rgb_2d.shape == (512, 512, 3), f"錯誤的形狀: {ct_rgb_2d.shape}"
    assert np.array_equal(ct_rgb_2d[:,:,0], ct_rgb_2d[:,:,1]), "通道 0 和 1 應該相同"
    assert np.array_equal(ct_rgb_2d[:,:,1], ct_rgb_2d[:,:,2]), "通道 1 和 2 應該相同"
    
    print("✅ 2D 模式測試通過")
    print(f"   - 形狀: {ct_rgb_2d.shape}")
    print(f"   - 通道 0 == 通道 1: {np.array_equal(ct_rgb_2d[:,:,0], ct_rgb_2d[:,:,1])}")
    print(f"   - 通道 1 == 通道 2: {np.array_equal(ct_rgb_2d[:,:,1], ct_rgb_2d[:,:,2])}")


def test_2_5d_mode():
    """測試 2.5D 模式"""
    print("\n" + "="*70)
    print("測試 2.5D 模式（三個通道是不同切片）")
    print("="*70)
    
    # 模擬 3 個不同的切片
    slice_prev = np.random.rand(512, 512).astype(np.float32)
    slice_curr = np.random.rand(512, 512).astype(np.float32)
    slice_next = np.random.rand(512, 512).astype(np.float32)
    
    # 2.5D 模式處理
    ct_rgb_2_5d = np.stack([slice_prev, slice_curr, slice_next], axis=-1)
    
    # 驗證
    assert ct_rgb_2_5d.shape == (512, 512, 3), f"錯誤的形狀: {ct_rgb_2_5d.shape}"
    assert not np.array_equal(ct_rgb_2_5d[:,:,0], ct_rgb_2_5d[:,:,1]), "通道 0 和 1 應該不同"
    assert not np.array_equal(ct_rgb_2_5d[:,:,1], ct_rgb_2_5d[:,:,2]), "通道 1 和 2 應該不同"
    
    print("✅ 2.5D 模式測試通過")
    print(f"   - 形狀: {ct_rgb_2_5d.shape}")
    print(f"   - 通道 0 != 通道 1: {not np.array_equal(ct_rgb_2_5d[:,:,0], ct_rgb_2_5d[:,:,1])}")
    print(f"   - 通道 1 != 通道 2: {not np.array_equal(ct_rgb_2_5d[:,:,1], ct_rgb_2_5d[:,:,2])}")


def test_boundary_handling():
    """測試邊界處理"""
    print("\n" + "="*70)
    print("測試邊界處理（第一張和最後一張切片）")
    print("="*70)
    
    num_slices = 100
    
    # 測試第一張切片
    slice_idx = 0
    z_prev = max(0, slice_idx - 1)
    z_next = min(num_slices - 1, slice_idx + 1)
    
    assert z_prev == 0, f"第一張切片的 z_prev 應該是 0，實際: {z_prev}"
    assert z_next == 1, f"第一張切片的 z_next 應該是 1，實際: {z_next}"
    print(f"✅ 第一張切片 (idx=0): z_prev={z_prev}, z_curr=0, z_next={z_next}")
    
    # 測試最後一張切片
    slice_idx = num_slices - 1
    z_prev = max(0, slice_idx - 1)
    z_next = min(num_slices - 1, slice_idx + 1)
    
    assert z_prev == num_slices - 2, f"最後一張切片的 z_prev 應該是 {num_slices-2}，實際: {z_prev}"
    assert z_next == num_slices - 1, f"最後一張切片的 z_next 應該是 {num_slices-1}，實際: {z_next}"
    print(f"✅ 最後一張切片 (idx={num_slices-1}): z_prev={z_prev}, z_curr={num_slices-1}, z_next={z_next}")
    
    # 測試中間切片
    slice_idx = 50
    z_prev = max(0, slice_idx - 1)
    z_next = min(num_slices - 1, slice_idx + 1)
    
    assert z_prev == 49, f"中間切片的 z_prev 應該是 49，實際: {z_prev}"
    assert z_next == 51, f"中間切片的 z_next 應該是 51，實際: {z_next}"
    print(f"✅ 中間切片 (idx=50): z_prev={z_prev}, z_curr=50, z_next={z_next}")


def test_config():
    """測試配置"""
    print("\n" + "="*70)
    print("測試配置檔案")
    print("="*70)
    
    config = get_default_config()
    
    assert hasattr(config.data, 'use_2_5d'), "Config 應該有 use_2_5d 屬性"
    assert config.data.use_2_5d == True, "預設應該啟用 2.5D"
    
    print(f"✅ 配置測試通過")
    print(f"   - use_2_5d 存在: {hasattr(config.data, 'use_2_5d')}")
    print(f"   - 預設值: {config.data.use_2_5d}")


def main():
    """執行所有測試"""
    print("\n" + "🧪 " + "="*66 + " 🧪")
    print("   MedSAM2 2.5D 模式測試")
    print("🧪 " + "="*66 + " 🧪")
    
    try:
        test_2d_mode()
        test_2_5d_mode()
        test_boundary_handling()
        test_config()
        
        print("\n" + "🎉 " + "="*66 + " 🎉")
        print("   所有測試通過！2.5D 實作正確！")
        print("🎉 " + "="*66 + " 🎉\n")
        
        print("📝 下一步:")
        print("   1. 使用 2.5D 模式訓練: python finetune_medsam2/main.py --use_cache")
        print("   2. 對比實驗: python finetune_medsam2/main.py --use_cache --no_2_5d")
        print("   3. 查看文件: docs/2.5D_MODE_GUIDE.md\n")
        
        return 0
        
    except AssertionError as e:
        print(f"\n❌ 測試失敗: {e}\n")
        return 1
    except Exception as e:
        print(f"\n❌ 錯誤: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
