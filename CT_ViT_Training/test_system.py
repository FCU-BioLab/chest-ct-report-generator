#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CT-ViT 系統快速測試腳本
驗證所有模組是否可以正常導入和運行

作者: GitHub Copilot
日期: 2025-07-22
"""

import sys
import os
import traceback
from pathlib import Path

def test_imports():
    """測試模組導入"""
    print("🔍 測試模組導入...")
    
    # 添加src路徑
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    tests = [
        ("config", "CTViTConfig"),
        ("data_processing", "DICOMProcessor, CTDataset"),
        ("model", "CTViTTrainer"),
        ("utils", "setup_logging, get_device")
    ]
    
    for module_name, components in tests:
        try:
            module = __import__(module_name)
            print(f"   ✅ {module_name}: {components}")
        except Exception as e:
            print(f"   ❌ {module_name}: {e}")

def test_external_dependencies():
    """測試外部依賴"""
    print("\n🔍 測試外部依賴...")
    
    dependencies = [
        "torch",
        "transformers", 
        "PIL",
        "numpy",
        "yaml",
        "sklearn"
    ]
    
    missing_deps = []
    
    for dep in dependencies:
        try:
            if dep == "PIL":
                __import__("PIL")
            elif dep == "sklearn":
                __import__("sklearn")
            else:
                __import__(dep)
            print(f"   ✅ {dep}")
        except ImportError:
            print(f"   ❌ {dep} (未安裝)")
            missing_deps.append(dep)
        except Exception as e:
            print(f"   ⚠️  {dep} (警告: {e})")
    
    return missing_deps

def test_optional_dependencies():
    """測試可選依賴"""
    print("\n🔍 測試可選依賴...")
    
    optional_deps = [
        ("cv2", "OpenCV"),
        ("pydicom", "DICOM處理"),
        ("matplotlib", "可視化"),
        ("seaborn", "統計圖表"),
        ("tensorboard", "訓練監控")
    ]
    
    for dep, description in optional_deps:
        try:
            __import__(dep)
            print(f"   ✅ {dep} ({description})")
        except ImportError:
            print(f"   ❌ {dep} ({description}) - 未安裝")

def test_config_loading():
    """測試配置文件載入"""
    print("\n🔍 測試配置文件...")
    
    config_path = Path("configs/default_config.yaml")
    if config_path.exists():
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"   ✅ 配置文件載入成功")
            print(f"   📊 配置項數量: {len(config)}")
        except Exception as e:
            print(f"   ❌ 配置文件載入失敗: {e}")
    else:
        print(f"   ❌ 配置文件不存在: {config_path}")

def test_gpu_availability():
    """測試GPU可用性"""
    print("\n🔍 測試GPU環境...")
    
    try:
        import torch
        print(f"   📦 PyTorch版本: {torch.__version__}")
        print(f"   🖥️  CUDA可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"   🎮 GPU數量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"      GPU {i}: {name} ({memory:.1f}GB)")
        else:
            print("   ⚠️  將使用CPU進行訓練")
            
    except Exception as e:
        print(f"   ❌ GPU檢測失敗: {e}")

def test_file_structure():
    """測試文件結構"""
    print("\n🔍 測試文件結構...")
    
    required_files = [
        "train.py",
        "inference.py", 
        "requirements.txt",
        "README.md"
    ]
    
    required_dirs = [
        "src",
        "configs",
        "scripts",
        "docs",
        "legacy"
    ]
    
    for file in required_files:
        if Path(file).exists():
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} (缺失)")
    
    for dir in required_dirs:
        if Path(dir).exists():
            file_count = len(list(Path(dir).glob("*")))
            print(f"   ✅ {dir}/ ({file_count} 個文件)")
        else:
            print(f"   ❌ {dir}/ (缺失)")

def main():
    """主測試函數"""
    print("🚀 CT-ViT 系統健康檢查")
    print("=" * 50)
    
    # 執行各項測試
    test_file_structure()
    test_imports()
    missing_deps = test_external_dependencies()
    test_optional_dependencies()
    test_config_loading()
    test_gpu_availability()
    
    print("\n" + "=" * 50)
    print("📋 檢查結果摘要:")
    
    if missing_deps:
        print(f"⚠️  缺少依賴套件: {', '.join(missing_deps)}")
        print("💡 請運行以下命令安裝:")
        print(f"   pip install {' '.join(missing_deps)}")
        print("   或使用: pip install -r requirements.txt")
    else:
        print("✅ 所有核心依賴都已安裝")
    
    print("\n🎯 下一步操作:")
    print("1. 如有缺失依賴，請先安裝")
    print("2. 準備您的資料集")
    print("3. 運行 scripts/run_ct_vit.bat 開始使用")
    print("4. 或直接運行: python train.py --config configs/default_config.yaml")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 測試過程中發生錯誤:")
        print(f"   {e}")
        print("\n🔧 詳細錯誤信息:")
        traceback.print_exc()
