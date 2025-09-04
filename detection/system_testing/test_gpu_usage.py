#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU使用驗證腳本
確認PyTorch是否正確使用GPU進行計算
"""

import torch
import time
import numpy as np

def test_gpu_utilization():
    """測試GPU利用率"""
    print("🔥 開始GPU使用率測試...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    device = torch.device('cuda')
    print(f"✅ 使用設備: {device}")
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    
    # 創建大型張量進行GPU計算
    print("🚀 開始GPU密集計算...")
    
    for i in range(10):
        # 創建大型隨機張量
        a = torch.randn(2000, 2000, device=device)
        b = torch.randn(2000, 2000, device=device)
        
        start_time = time.time()
        
        # 執行矩陣運算
        c = torch.mm(a, b)
        c = torch.relu(c)
        c = torch.sum(c)
        
        # 同步GPU操作
        torch.cuda.synchronize()
        
        end_time = time.time()
        
        memory_used = torch.cuda.memory_allocated() / (1024**3)
        print(f"  迭代 {i+1}: 時間={end_time-start_time:.3f}s, GPU記憶體={memory_used:.2f}GB")
        
        time.sleep(1)  # 等待1秒讓GPU使用率可以被觀察到
    
    print("✅ GPU測試完成！")
    return True

def test_model_on_gpu():
    """測試模型在GPU上的運行"""
    print("\n🔥 測試模型GPU使用...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    device = torch.device('cuda')
    
    # 創建簡單的CNN模型
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 128, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(128, 10)
    ).to(device)
    
    print(f"✅ 模型已移至GPU")
    
    # 創建批次數據
    batch_size = 32
    
    for i in range(5):
        # 創建隨機輸入
        inputs = torch.randn(batch_size, 3, 224, 224, device=device)
        targets = torch.randint(0, 10, (batch_size,), device=device)
        
        start_time = time.time()
        
        # 前向傳播
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        
        # 反向傳播
        loss.backward()
        
        # 同步GPU操作
        torch.cuda.synchronize()
        
        end_time = time.time()
        
        memory_used = torch.cuda.memory_allocated() / (1024**3)
        print(f"  批次 {i+1}: 時間={end_time-start_time:.3f}s, 損失={loss.item():.4f}, GPU記憶體={memory_used:.2f}GB")
        
        time.sleep(1)
    
    print("✅ 模型GPU測試完成！")
    return True

if __name__ == "__main__":
    print("🎯 GPU使用驗證測試")
    print("="*50)
    print("請在另一個終端運行 'nvidia-smi' 命令來監控GPU使用率")
    print("="*50)
    
    # 測試基本GPU計算
    success1 = test_gpu_utilization()
    
    if success1:
        # 測試模型GPU使用
        success2 = test_model_on_gpu()
        
        if success2:
            print("\n🎉 所有GPU測試通過！")
            print("如果nvidia-smi顯示GPU使用率很低，可能是:")
            print("1. 計算任務太簡單")
            print("2. 系統GPU監控有延遲")
            print("3. 需要更大的批次大小或模型")
        else:
            print("\n❌ 模型GPU測試失敗")
    else:
        print("\n❌ 基本GPU測試失敗")
