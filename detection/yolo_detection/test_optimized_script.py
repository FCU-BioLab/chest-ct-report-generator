#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for YOLOv11 Simple Training
Tests the basic functionality and dependency checking
"""

import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(__file__))

def test_imports():
    """Test if the module can be imported and basic functions work"""
    print("Testing imports...")
    
    try:
        from train_yolov11_simple import (
            check_dependencies, 
            validate_requirements,
            ULTRALYTICS_AVAILABLE,
            MATPLOTLIB_AVAILABLE,
            YOLO_MODULES_AVAILABLE
        )
        print("✓ Module imports successful")
        
        # Test dependency checking
        print("\nChecking dependencies...")
        deps = check_dependencies()
        print(f"Dependencies: {deps}")
        
        # Test requirement validation
        valid = validate_requirements()
        print(f"Requirements valid: {valid}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_directory_setup():
    """Test directory setup functionality"""
    print("\nTesting directory setup...")
    
    try:
        from train_yolov11_simple import setup_yolo_directories
        
        base_dir = setup_yolo_directories()
        print(f"✓ Directories set up at: {base_dir}")
        return True
        
    except Exception as e:
        print(f"✗ Directory setup error: {e}")
        return False

def main():
    """Main test function"""
    print("YOLOv11 Simple Training - Test Script")
    print("=" * 50)
    
    # Test imports
    import_success = test_imports()
    
    if import_success:
        # Test directory setup
        dir_success = test_directory_setup()
        
        if dir_success:
            print("\n✓ All basic tests passed!")
            print("\nTo run dependency check only:")
            print("python train_yolov11_simple.py --check_deps")
        else:
            print("\n✗ Directory setup test failed")
    else:
        print("\n✗ Import test failed")
        print("Please check if all required files are present.")

if __name__ == "__main__":
    main()
