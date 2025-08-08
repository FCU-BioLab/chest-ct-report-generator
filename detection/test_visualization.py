"""
測試更新後的訓練腳本中的可視化功能
"""

import sys
import os

# 添加項目根目錄到Python路徑
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def test_imports():
    """測試是否能正確導入所需的包"""
    try:
        import matplotlib.pyplot as plt
        print("✅ matplotlib.pyplot 導入成功")
    except ImportError as e:
        print(f"❌ matplotlib.pyplot 導入失敗: {e}")
    
    try:
        import matplotlib.patches as patches
        print("✅ matplotlib.patches 導入成功")
    except ImportError as e:
        print(f"❌ matplotlib.patches 導入失敗: {e}")
    
    try:
        import cv2
        print("✅ opencv-python 導入成功")
    except ImportError as e:
        print(f"❌ opencv-python 導入失敗: {e}")
    
    try:
        import torch
        print("✅ torch 導入成功")
    except ImportError as e:
        print(f"❌ torch 導入失敗: {e}")
    
    try:
        import numpy as np
        print("✅ numpy 導入成功")
    except ImportError as e:
        print(f"❌ numpy 導入失敗: {e}")

def test_visualization_functions():
    """測試可視化函數是否能正確定義"""
    try:
        # 嘗試導入更新後的訓練腳本
        from detection.train_detection_simple import visualize_predictions, create_prediction_summary
        print("✅ train_detection_simple 可視化函數導入成功")
    except ImportError as e:
        print(f"❌ train_detection_simple 可視化函數導入失敗: {e}")
    
    try:
        from detection.train_detection import visualize_predictions, create_prediction_summary, create_kfold_summary_plots
        print("✅ train_detection 可視化函數導入成功")
    except ImportError as e:
        print(f"❌ train_detection 可視化函數導入失敗: {e}")

if __name__ == "__main__":
    print("=== 測試可視化功能依賴 ===")
    test_imports()
    print("\n=== 測試可視化函數定義 ===")
    test_visualization_functions()
    print("\n=== 測試完成 ===")
