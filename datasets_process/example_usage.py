#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用範例腳本
展示如何使用 split_dataset.py 和 inference_ct_vit.py

作者: GitHub Copilot
日期: 2025-07-24
"""

import os
import subprocess
from pathlib import Path

def run_dataset_split():
    """執行資料集劃分範例"""
    print("🚀 執行資料集劃分...")
    
    cmd = [
        "python", "split_dataset.py",
        "--source_dir", "d:/GitHub/chest-ct-report-generator/matched_data_by_patient",
        "--output_dir", "d:/GitHub/chest-ct-report-generator/dataset_splits",
        "--train_ratio", "0.7",
        "--val_ratio", "0.15", 
        "--test_ratio", "0.15",
        "--random_seed", "42"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ 資料集劃分完成")
            print(result.stdout)
        else:
            print("❌ 資料集劃分失敗")
            print(result.stderr)
    except Exception as e:
        print(f"❌ 執行錯誤: {str(e)}")

def run_inference_example():
    """執行推理範例"""
    print("🚀 執行推理範例...")
    
    # 假設模型路徑和測試文件
    model_path = "d:/GitHub/chest-ct-report-generator/CT_ViT_Training/models/best_model"
    test_dir = "d:/GitHub/chest-ct-report-generator/dataset_splits/test"
    output_dir = "d:/GitHub/chest-ct-report-generator/datasets_process/inference_results"
    
    # 檢查路徑是否存在
    if not Path(model_path).exists():
        print(f"❌ 模型路徑不存在: {model_path}")
        print("請先訓練模型或提供正確的模型路徑")
        return
    
    if not Path(test_dir).exists():
        print(f"❌ 測試目錄不存在: {test_dir}")
        print("請先執行資料集劃分")
        return
    
    # 使用現有的 CT_ViT_Training/inference.py
    inference_script = "d:/GitHub/chest-ct-report-generator/CT_ViT_Training/inference.py"
    
    # 批量推理範例
    cmd = [
        "python", inference_script,
        "--model_path", model_path,
        "--mode", "batch",
        "--input", test_dir,
        "--output", output_dir
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="d:/GitHub/chest-ct-report-generator/CT_ViT_Training")
        if result.returncode == 0:
            print("✅ 推理完成")
            print(result.stdout)
        else:
            print("❌ 推理失敗")
            print(result.stderr)
    except Exception as e:
        print(f"❌ 執行錯誤: {str(e)}")

def run_evaluation_example():
    """執行評估範例"""
    print("🚀 執行評估範例...")
    
    model_path = "d:/GitHub/chest-ct-report-generator/CT_ViT_Training/models/best_model"
    test_dir = "d:/GitHub/chest-ct-report-generator/dataset_splits/test"
    output_dir = "d:/GitHub/chest-ct-report-generator/datasets_process/evaluation_results"
    
    # 使用現有的 CT_ViT_Training/inference.py
    inference_script = "d:/GitHub/chest-ct-report-generator/CT_ViT_Training/inference.py"
    
    # 評估範例
    cmd = [
        "python", inference_script,
        "--model_path", model_path,
        "--mode", "evaluate",
        "--input", test_dir,
        "--output", output_dir
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="d:/GitHub/chest-ct-report-generator/CT_ViT_Training")
        if result.returncode == 0:
            print("✅ 評估完成")
            print(result.stdout)
        else:
            print("❌ 評估失敗")
            print(result.stderr)
    except Exception as e:
        print(f"❌ 執行錯誤: {str(e)}")

def main():
    """主函數"""
    print("=== CT-ViT 工具使用範例 ===\n")
    
    while True:
        print("\n請選擇要執行的操作:")
        print("1. 資料集劃分")
        print("2. 批量推理")
        print("3. 模型評估")
        print("4. 退出")
        
        choice = input("\n請輸入選項 (1-4): ").strip()
        
        if choice == '1':
            run_dataset_split()
        elif choice == '2':
            run_inference_example()
        elif choice == '3':
            run_evaluation_example()
        elif choice == '4':
            print("👋 再見!")
            break
        else:
            print("❌ 無效選項，請重新選擇")

if __name__ == "__main__":
    main()
