#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick GPU Check Script
"""

import torch
import sys

def check_gpu():
    print("=" * 60)
    print("GPU Configuration Check")
    print("=" * 60)
    
    print(f"\n✓ PyTorch Version: {torch.__version__}")
    print(f"✓ CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA Version: {torch.version.cuda}")
        print(f"✓ GPU Count: {torch.cuda.device_count()}")
        print(f"✓ Current Device: {torch.cuda.current_device()}")
        print(f"✓ Device Name: {torch.cuda.get_device_name(0)}")
        
        props = torch.cuda.get_device_properties(0)
        print(f"✓ GPU Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"✓ CUDA Capability: {props.major}.{props.minor}")
        
        # Test tensor creation
        try:
            x = torch.randn(1000, 1000, device='cuda')
            print(f"\n✓ Test tensor created on GPU: {x.device}")
            print(f"✓ GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            print(f"✓ GPU Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
            del x
            torch.cuda.empty_cache()
            print("✓ GPU tensor test PASSED")
        except Exception as e:
            print(f"✗ GPU tensor test FAILED: {e}")
            return False
        
        print("\n" + "=" * 60)
        print("GPU is ready for training! 🚀")
        print("=" * 60)
        return True
    else:
        print("\n" + "=" * 60)
        print("⚠ CUDA is not available!")
        print("Training will use CPU (very slow)")
        print("=" * 60)
        return False

if __name__ == "__main__":
    success = check_gpu()
    sys.exit(0 if success else 1)
