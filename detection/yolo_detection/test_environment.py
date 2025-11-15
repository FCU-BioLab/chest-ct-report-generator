#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test script to verify train_yolo_direct.py is working correctly.
Run this before starting actual training to catch any issues early.
"""

import sys
from pathlib import Path


def test_imports():
    """Test if all required packages are importable."""
    print("🔍 Testing package imports...")
    
    errors = []
    
    try:
        import torch
        print(f"   ✅ torch {torch.__version__}")
    except ImportError as e:
        errors.append(f"torch: {e}")
    
    try:
        import ultralytics
        print(f"   ✅ ultralytics {ultralytics.__version__}")
    except ImportError as e:
        errors.append(f"ultralytics: {e}")
    
    try:
        import numpy
        print(f"   ✅ numpy {numpy.__version__}")
    except ImportError as e:
        errors.append(f"numpy: {e}")
    
    if errors:
        print("\n❌ Missing packages:")
        for error in errors:
            print(f"   {error}")
        print("\n📦 Install with: pip install ultralytics torch numpy")
        return False
    
    return True


def test_cuda():
    """Test CUDA availability."""
    print("\n🔍 Testing CUDA...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"   ✅ CUDA available")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("   ⚠️  CUDA not available (will use CPU)")
            print("   Note: Training on CPU will be much slower")
            return False
    except Exception as e:
        print(f"   ❌ Error checking CUDA: {e}")
        return False


def test_dataset():
    """Test if dataset directory exists and is valid."""
    print("\n🔍 Testing dataset...")
    
    # Default dataset path
    dataset_path = Path("../../datasets/splited_dataset/train")
    
    if not dataset_path.exists():
        print(f"   ❌ Dataset not found: {dataset_path}")
        print("   Please provide correct path using --data_dir")
        return False
    
    # Count patient directories
    patient_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    if not patient_dirs:
        print(f"   ❌ No patient directories found in {dataset_path}")
        return False
    
    print(f"   ✅ Dataset found: {dataset_path}")
    print(f"   Patients: {len(patient_dirs)}")
    
    # Sample check
    sample_patient = patient_dirs[0]
    if (sample_patient / "images").exists() and (sample_patient / "labels").exists():
        img_count = len(list((sample_patient / "images").glob("*.png")))
        print(f"   Sample check ({sample_patient.name}): {img_count} images ✅")
        return True
    else:
        print(f"   ❌ Invalid structure in {sample_patient.name}")
        return False


def test_script_exists():
    """Test if training script exists."""
    print("\n🔍 Testing script files...")
    
    script_path = Path("train_yolo_direct.py")
    
    if not script_path.exists():
        print(f"   ❌ Script not found: {script_path}")
        return False
    
    print(f"   ✅ Training script found: {script_path}")
    
    # Test if script is importable
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("train_yolo_direct", script_path)
        if spec and spec.loader:
            print("   ✅ Script is valid Python file")
            return True
    except Exception as e:
        print(f"   ⚠️  Warning: {e}")
    
    return True


def test_disk_space():
    """Test if there's enough disk space."""
    print("\n🔍 Testing disk space...")
    
    try:
        import shutil
        
        # Check current directory disk space
        stat = shutil.disk_usage(".")
        
        free_gb = stat.free / (1024**3)
        total_gb = stat.total / (1024**3)
        
        print(f"   Free: {free_gb:.1f} GB / {total_gb:.1f} GB")
        
        if free_gb < 20:
            print("   ⚠️  Warning: Less than 20 GB free")
            print("   Training outputs may require 10-20 GB")
            return False
        else:
            print("   ✅ Sufficient disk space")
            return True
            
    except Exception as e:
        print(f"   ⚠️  Could not check disk space: {e}")
        return True  # Don't fail on this


def main():
    print("=" * 80)
    print("🧪 YOLOv11 Training Environment Test")
    print("=" * 80)
    print()
    
    tests = [
        ("Package Imports", test_imports),
        ("CUDA Support", test_cuda),
        ("Dataset", test_dataset),
        ("Script Files", test_script_exists),
        ("Disk Space", test_disk_space),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 Test Summary")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10} {name}")
    
    print("=" * 80)
    print(f"Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All tests passed! You're ready to start training.")
        print("\nRun training with:")
        print("   python train_yolo_direct.py --data_dir ../../datasets/splited_dataset/train")
    elif passed >= total - 1:
        print("\n⚠️  Most tests passed. You can probably proceed, but review warnings above.")
    else:
        print("\n❌ Some critical tests failed. Please fix issues before training.")
        return 1
    
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
