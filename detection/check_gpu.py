#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU檢測和設定腳本
確保PyTorch能正確使用GPU進行訓練
"""

import torch
import sys

def check_gpu_setup():
    """檢查GPU設定和可用性"""
    
    print("="*60)
    print("🔍 GPU檢測和設定檢查")
    print("="*60)
    
    # 1. 基本PyTorch信息
    print(f"\n📋 PyTorch版本: {torch.__version__}")
    print(f"📋 CUDA版本: {torch.version.cuda if torch.version.cuda else '未安裝'}")
    print(f"📋 cuDNN版本: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else '未安裝'}")
    
    # 2. GPU可用性檢查
    print(f"\n🔍 GPU檢查:")
    if torch.cuda.is_available():
        print("✅ CUDA 可用")
        print(f"✅ GPU數量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"✅ GPU {i}: {gpu_name}")
            print(f"   記憶體: {gpu_memory:.1f} GB")
            
        # 3. 設定預設GPU
        device = torch.device('cuda:0')
        print(f"✅ 預設GPU設備: {device}")
        
        # 4. 測試GPU功能
        print(f"\n🧪 GPU功能測試:")
        try:
            # 創建測試張量
            test_tensor = torch.randn(1000, 1000).to(device)
            result = torch.mm(test_tensor, test_tensor.t())
            
            # 檢查記憶體使用
            memory_used = torch.cuda.memory_allocated() / 1024**2
            print(f"✅ GPU運算測試成功")
            print(f"✅ 記憶體使用: {memory_used:.1f} MB")
            
            # 清理
            del test_tensor, result
            torch.cuda.empty_cache()
            print(f"✅ GPU記憶體已清理")
            
        except Exception as e:
            print(f"❌ GPU測試失敗: {e}")
            return False
            
    else:
        print("❌ CUDA 不可用")
        print("原因可能是:")
        print("1. 沒有安裝CUDA")
        print("2. PyTorch沒有CUDA支援")
        print("3. GPU驅動程式問題")
        return False
    
    # 5. 建議的訓練設定
    print(f"\n🚀 建議的訓練設定:")
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        if gpu_memory >= 8:
            batch_size = 16
        elif gpu_memory >= 6:
            batch_size = 12
        elif gpu_memory >= 4:
            batch_size = 8
        else:
            batch_size = 4
            
        print(f"✅ 建議批次大小: {batch_size}")
        print(f"✅ 建議num_workers: 4")
        print(f"✅ 建議image_size: 512")
        
        print(f"\n🎯 推薦訓練指令:")
        print(f"python train_detection.py --mode traditional --batch_size {batch_size}")
    else:
        print("❌ 無GPU可用，將使用CPU訓練（速度較慢）")
        print("python train_detection.py --mode traditional --batch_size 4")
    
    print("\n" + "="*60)
    
    return torch.cuda.is_available()

if __name__ == "__main__":
    success = check_gpu_setup()
    
    if success:
        print("✅ GPU設定正常，可以開始訓練！")
        sys.exit(0)
    else:
        print("❌ GPU設定有問題，請檢查CUDA安裝")
        sys.exit(1)
