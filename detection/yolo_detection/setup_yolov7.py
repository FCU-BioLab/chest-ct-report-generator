#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup and Verification Script for YOLOv7 Medical Training

Checks dependencies, validates model configs, and tests modules
"""

import sys
import subprocess
from pathlib import Path

def print_banner(text):
    """Print a formatted banner"""
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80 + "\n")


def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected")
        print("⚠️  Python 3.8+ required")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_dependencies():
    """Check required dependencies"""
    print("\nChecking dependencies...")
    
    required = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy',
        'cv2': 'OpenCV (opencv-python)',
        'yaml': 'PyYAML',
        'tqdm': 'tqdm',
    }
    
    optional = {
        'pydicom': 'PyDICOM (for DICOM support)',
        'SimpleITK': 'SimpleITK (for medical imaging)',
    }
    
    missing_required = []
    missing_optional = []
    
    # Check required
    for module, name in required.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"❌ {name} - MISSING")
            missing_required.append(name)
    
    # Check optional
    print("\nOptional dependencies:")
    for module, name in optional.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"⚠️  {name} - not installed")
            missing_optional.append(name)
    
    if missing_required:
        print("\n❌ Missing required dependencies:")
        for dep in missing_required:
            print(f"   - {dep}")
        print("\nInstall with:")
        print("   pip install -r requirements_yolov7.txt")
        return False
    
    return True


def check_cuda():
    """Check CUDA availability"""
    print("\nChecking CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available")
            print(f"  Devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
        else:
            print("⚠️  CUDA not available - training will use CPU")
    except ImportError:
        print("❌ PyTorch not installed")


def check_model_configs():
    """Check model configuration files"""
    print("\nChecking model configurations...")
    
    configs = [
        "models/yolov7_medical.yaml",
        "models/yolov7_baseline.yaml",
    ]
    
    script_dir = Path(__file__).parent
    
    for config in configs:
        config_path = script_dir / config
        if config_path.exists():
            print(f"✓ {config}")
        else:
            print(f"❌ {config} - NOT FOUND")
            return False
    
    return True


def check_modules():
    """Check custom modules"""
    print("\nChecking custom modules...")
    
    modules = [
        "models/custom_layers.py",
        "yolov7_model.py",
        "yolov7_dataset.py",
        "yolov7_utils.py",
        "train_yolov7_medical.py",
    ]
    
    script_dir = Path(__file__).parent
    
    for module in modules:
        module_path = script_dir / module
        if module_path.exists():
            print(f"✓ {module}")
        else:
            print(f"❌ {module} - NOT FOUND")
            return False
    
    return True


def test_custom_layers():
    """Test custom layer imports"""
    print("\nTesting custom layers...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent / "models"))
        from custom_layers import CBAM, SimAM, SwinTransformerBlock, BiFPN
        
        print("✓ CBAM")
        print("✓ SimAM")
        print("✓ SwinTransformerBlock")
        print("✓ BiFPN")
        
        # Quick instantiation test
        import torch
        
        print("\nTesting module instantiation...")
        cbam = CBAM(in_channels=256)
        print("✓ CBAM instantiation")
        
        simam = SimAM()
        print("✓ SimAM instantiation")
        
        swin = SwinTransformerBlock(dim=256, num_heads=8)
        print("✓ Swin Transformer instantiation")
        
        bifpn = BiFPN(channels=256, num_layers=3)
        print("✓ BiFPN instantiation")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing custom layers: {e}")
        return False


def test_model_loading():
    """Test model loading"""
    print("\nTesting model loading...")
    
    try:
        from yolov7_model import load_yolov7_model
        
        # Test with medical modules
        script_dir = Path(__file__).parent
        config_path = script_dir / "models" / "yolov7_medical.yaml"
        
        if not config_path.exists():
            print(f"❌ Config not found: {config_path}")
            return False
        
        print("Loading YOLOv7 with medical modules...")
        model = load_yolov7_model(
            cfg_path=str(config_path),
            use_medical_modules=True,
            device='cpu'
        )
        
        print("✓ Model loaded successfully")
        
        # Test forward pass
        import torch
        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            y = model(x)
        
        print("✓ Forward pass successful")
        
        # Print parameters
        params = model.count_parameters()
        print(f"\nModel statistics:")
        print(f"  Total parameters: {params['total']:,}")
        print(f"  Trainable parameters: {params['trainable']:,}")
        print(f"  Medical module parameters: {params['medical_modules']:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main setup verification"""
    print_banner("YOLOv7 Medical Training Setup Verification")
    
    all_passed = True
    
    # Check Python version
    if not check_python_version():
        all_passed = False
    
    # Check dependencies
    if not check_dependencies():
        all_passed = False
        print("\n❌ Setup incomplete - install dependencies first")
        return
    
    # Check CUDA
    check_cuda()
    
    # Check files
    if not check_model_configs():
        all_passed = False
    
    if not check_modules():
        all_passed = False
    
    # Test modules
    if not test_custom_layers():
        all_passed = False
    
    if not test_model_loading():
        all_passed = False
    
    # Summary
    print_banner("Setup Verification Complete")
    
    if all_passed:
        print("✓ All checks passed!")
        print("\nYou can now start training with:")
        print("  python train_yolov7_medical.py --data_dir <your_data_dir> --epochs 120")
        print("\nFor more options:")
        print("  python train_yolov7_medical.py --help")
    else:
        print("❌ Some checks failed")
        print("Please fix the issues above before training")
    
    print()


if __name__ == "__main__":
    main()
