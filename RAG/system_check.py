#!/usr/bin/env python3
"""
系統資源檢測腳本
檢測GPU、記憶體等資源，建議適合的模型
"""

import torch
import psutil
import sys
import os
from pathlib import Path

def check_system_resources():
    """檢測系統資源"""
    print("=== 系統資源檢測 ===")
    
    # 檢測GPU
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 可用: {cuda_available}")
    
    if cuda_available:
        gpu_count = torch.cuda.device_count()
        print(f"GPU 數量: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    
    # 檢測CPU和記憶體
    cpu_count = psutil.cpu_count()
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)
    memory_available_gb = memory.available / (1024**3)
    
    print(f"CPU 核心數: {cpu_count}")
    print(f"總記憶體: {memory_gb:.1f}GB")
    print(f"可用記憶體: {memory_available_gb:.1f}GB")
    
    return {
        'cuda_available': cuda_available,
        'gpu_count': gpu_count if cuda_available else 0,
        'memory_gb': memory_gb,
        'memory_available_gb': memory_available_gb,
        'cpu_count': cpu_count
    }

def recommend_model(system_info):
    """根據系統資源建議適合的模型"""
    print("\n=== 模型建議 ===")
    
    if system_info['cuda_available']:
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        if gpu_memory >= 16:
            print("✅ 建議使用: google/gemma-3-4b-it")
            print("   GPU記憶體充足，可以運行完整的Gemma 3 4B模型")
        elif gpu_memory >= 8:
            print("⚠️  建議使用: google/gemma-2b-it")
            print("   GPU記憶體中等，建議使用較小的Gemma 2B模型")
        else:
            print("⚠️  建議使用: distilgpt2")
            print("   GPU記憶體不足，建議使用輕量級模型")
    else:
        if system_info['memory_available_gb'] >= 16:
            print("⚠️  建議使用: google/gemma-2b-it")
            print("   CPU模式，記憶體充足，可嘗試Gemma 2B模型")
        elif system_info['memory_available_gb'] >= 8:
            print("✅ 建議使用: distilgpt2")
            print("   CPU模式，記憶體中等，建議使用DistilGPT-2")
        else:
            print("⚠️  建議使用: 模板生成")
            print("   記憶體不足，建議使用模板生成功能")
    
    print("\n=== 優化建議 ===")
    if not system_info['cuda_available']:
        print("🔧 考慮安裝NVIDIA GPU和CUDA來提升性能")
    
    if system_info['memory_available_gb'] < 8:
        print("🔧 關閉其他應用程式以釋放更多記憶體")
    
    print("🔧 如果記憶體不足，可以使用量化技術(8-bit)")
    print("🔧 考慮使用雲端GPU服務(Google Colab, AWS, etc.)")

def check_huggingface_login():
    """檢查Hugging Face登入狀態"""
    print("\n=== Hugging Face 狀態 ===")
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"✅ 已登入 Hugging Face: {user_info['name']}")
        return True
    except ImportError:
        print("❌ huggingface_hub 未安裝")
        print("   請執行: pip install huggingface_hub")
        return False
    except Exception as e:
        print("❌ 未登入 Hugging Face")
        print("   請執行: huggingface-cli login")
        print(f"   錯誤: {str(e)}")
        return False

def main():
    print("胸部CT報告生成器 - 系統檢測")
    print("=" * 50)
    
    # 檢測系統資源
    system_info = check_system_resources()
    
    # 建議模型
    recommend_model(system_info)
    
    # 檢查Hugging Face登入
    hf_status = check_huggingface_login()
    
    # 檢查本地模型
    check_local_models()
    
    print("\n" + "=" * 50)
    print("檢測完成！")
    
    # 提供下一步建議
    print("\n📋 建議的下一步:")
    if not hf_status:
        print("1. 先登入 Hugging Face: huggingface-cli login")
    
    print("2. 下載模型:")
    print("   python download_model_simple.py all")
    print("   或: python download_model_simple.py --model all")
    print("3. 運行程序:")
    if system_info['cuda_available']:
        print("   python Gemma3_GUI.py  # GPU版本")
    else:
        print("   python Gemma3_GUI_CPU.py  # CPU優化版")

def check_local_models():
    """檢查本地已下載的模型"""
    print("\n=== 本地模型狀態 ===")
    
    model_dir = Path("model")
    if not model_dir.exists():
        print("❌ model/ 目錄不存在")
        print("   請執行下載腳本: python download_model_simple.py all")
        return False
    
    models_found = False
    
    # 檢查 sentence transformer
    st_path = model_dir / "sentence_transformer"
    if st_path.exists():
        size = sum(f.stat().st_size for f in st_path.rglob('*') if f.is_file())
        size_mb = size / (1024 * 1024)
        print(f"✅ Sentence Transformer: {size_mb:.1f} MB")
        models_found = True
    else:
        print("❌ Sentence Transformer 未下載")
    
    # 檢查 Gemma 3
    gemma_path = model_dir / "gemma-3-4b-it"
    if gemma_path.exists():
        size = sum(f.stat().st_size for f in gemma_path.rglob('*') if f.is_file())
        size_gb = size / (1024 * 1024 * 1024)
        print(f"✅ Gemma 3 4B: {size_gb:.1f} GB")
        models_found = True
    else:
        print("❌ Gemma 3 4B 未下載")
    
    if not models_found:
        print("\n💡 建議執行: python download_model_simple.py all")
    
    return models_found

if __name__ == "__main__":
    main()
