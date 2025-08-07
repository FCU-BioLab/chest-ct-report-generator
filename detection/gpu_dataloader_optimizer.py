#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU加速資料載入工具
優化PyTorch DataLoader的GPU性能
"""

import torch
import psutil
import os

def get_optimal_num_workers(batch_size=12):
    """根據系統配置自動計算最佳的num_workers數量"""
    
    # 獲取CPU核心數
    cpu_cores = psutil.cpu_count(logical=False)  # 物理核心
    logical_cores = psutil.cpu_count(logical=True)  # 邏輯核心
    
    # 獲取可用記憶體（GB）
    available_memory = psutil.virtual_memory().available / (1024**3)
    
    print(f"🔍 系統資源檢測:")
    print(f"  CPU物理核心: {cpu_cores}")
    print(f"  CPU邏輯核心: {logical_cores}")
    print(f"  可用記憶體: {available_memory:.1f} GB")
    print(f"  當前批次大小: {batch_size}")
    
    # 根據經驗公式計算最佳num_workers
    # 考慮因素：CPU核心數、批次大小、記憶體
    
    if available_memory < 8:
        # 記憶體不足時，減少worker數量
        optimal_workers = min(2, cpu_cores)
        print(f"⚠️  記憶體較少，建議使用較少的workers")
    elif available_memory < 16:
        # 中等記憶體
        optimal_workers = min(4, cpu_cores)
    else:
        # 充足記憶體
        optimal_workers = min(8, cpu_cores)
    
    # 根據批次大小調整
    if batch_size >= 16:
        optimal_workers = min(optimal_workers, 6)
    elif batch_size >= 32:
        optimal_workers = min(optimal_workers, 4)
    
    print(f"🚀 建議的num_workers: {optimal_workers}")
    
    return optimal_workers

def get_gpu_memory_optimization_settings():
    """獲取GPU記憶體優化設定"""
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return {}
    
    # 獲取GPU信息
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    print(f"🎮 GPU信息:")
    print(f"  名稱: {gpu_name}")
    print(f"  記憶體: {gpu_memory:.1f} GB")
    
    settings = {
        'pin_memory': True,  # 基本設定
        'persistent_workers': True,  # 持久化workers（PyTorch 1.7+）
    }
    
    # 根據GPU記憶體調整設定
    if gpu_memory >= 8:
        settings['prefetch_factor'] = 4  # 預取更多批次
    elif gpu_memory >= 6:
        settings['prefetch_factor'] = 3
    else:
        settings['prefetch_factor'] = 2
    
    print(f"🔧 GPU優化設定:")
    for key, value in settings.items():
        print(f"  {key}: {value}")
    print(f"  注意: non_blocking將在.to()方法中使用")
    
    return settings

def create_optimized_dataloader(dataset, batch_size, shuffle=True, 
                              num_workers=None, collate_fn=None, **kwargs):
    """創建GPU優化的DataLoader"""
    
    print(f"\n{'='*50}")
    print(f"🚀 創建優化的DataLoader")
    print(f"{'='*50}")
    
    # 自動計算最佳num_workers
    if num_workers is None:
        num_workers = get_optimal_num_workers(batch_size)
    
    # 獲取GPU優化設定
    gpu_settings = get_gpu_memory_optimization_settings()
    
    # 合併設定
    dataloader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'collate_fn': collate_fn,
        **gpu_settings,
        **kwargs  # 允許覆蓋設定
    }
    
    print(f"\n📋 最終DataLoader設定:")
    for key, value in dataloader_kwargs.items():
        if key != 'collate_fn':  # 不顯示函數對象
            print(f"  {key}: {value}")
    
    try:
        dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
        print(f"✅ DataLoader創建成功！")
        return dataloader
    except Exception as e:
        print(f"❌ DataLoader創建失敗: {e}")
        # 回退到基本設定
        print(f"🔄 使用基本設定重試...")
        basic_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': min(2, num_workers),  # 減少workers
            'pin_memory': True,
            'collate_fn': collate_fn
        }
        return torch.utils.data.DataLoader(dataset, **basic_kwargs)

def optimize_gpu_memory():
    """優化GPU記憶體設定"""
    
    if not torch.cuda.is_available():
        return
    
    print(f"\n🔧 GPU記憶體優化:")
    
    # 啟用cudnn benchmark（對固定輸入大小有效）
    torch.backends.cudnn.benchmark = True
    print(f"  ✅ 啟用cuDNN benchmark")
    
    # 設定記憶體分配策略
    try:
        # 避免記憶體碎片化
        torch.cuda.empty_cache()
        print(f"  ✅ 清理GPU記憶體快取")
    except:
        pass
    
    # 顯示當前記憶體使用
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        cached = torch.cuda.memory_reserved() / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        print(f"  📊 GPU記憶體使用:")
        print(f"    已分配: {allocated:.2f} GB")
        print(f"    已快取: {cached:.2f} GB") 
        print(f"    總計: {total:.2f} GB")
        print(f"    使用率: {(allocated/total)*100:.1f}%")

if __name__ == "__main__":
    print("🎯 GPU加速資料載入工具測試")
    
    # 測試系統配置
    optimal_workers = get_optimal_num_workers(batch_size=12)
    
    # 測試GPU設定
    gpu_settings = get_gpu_memory_optimization_settings()
    
    # 測試記憶體優化
    optimize_gpu_memory()
