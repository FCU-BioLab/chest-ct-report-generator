#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取深層特徵的便捷腳本
用於批量提取病例的深層特徵供LLM生成報告使用

使用方法:
1. 使用預設模型和數據路徑
2. 自定義路徑
3. 批量處理多個數據集

作者: GitHub Copilot
日期: 2025-09-03
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# 添加detection目錄到路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from deep_feature_extractor import extract_features_from_dataset

def setup_logging(save_dir):
    """設置日誌記錄"""
    log_dir = os.path.join(save_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'extract_features_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # 清除已有的處理器
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 創建格式器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                 datefmt='%Y-%m-%d %H:%M:%S')
    
    # 文件處理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # 控制台處理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # 配置根日誌記錄器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return log_file

def find_best_model():
    """尋找最佳的訓練模型"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 預定義的模型搜索路徑
    search_paths = [
        'Simple_Training_20250901_120146/models/best_model.pth',
        'Simple_Training_20250830_100356/models/best_model.pth',
        'Faster_RCNN_Detection/models/best_model_fold_1.pth',
        'Faster_RCNN_Detection/models/best_model_fold_2.pth',
        'Faster_RCNN_Detection/models/best_model_fold_3.pth',
        'Faster_RCNN_Detection/models/best_model_fold_4.pth',
        'Faster_RCNN_Detection/models/best_model_fold_5.pth',
    ]
    
    for path in search_paths:
        full_path = os.path.join(script_dir, path)
        if os.path.exists(full_path):
            return full_path
    
    return None

def find_data_directory():
    """尋找數據集目錄"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # 可能的數據目錄路徑
    data_paths = [
        os.path.join(project_root, 'datasets', 'splited_dataset'),
        os.path.join(project_root, 'datasets', 'all_patient_data'),
        os.path.join(project_root, 'datasets'),
    ]
    
    for path in data_paths:
        if os.path.exists(path):
            return path
    
    return None

def extract_features_interactive():
    """互動式特徵提取"""
    print("=== 深層特徵提取工具 ===")
    print()
    
    # 尋找模型
    default_model = find_best_model()
    if default_model:
        print(f"找到默認模型: {default_model}")
        use_default = input("是否使用默認模型? (y/n): ").lower().strip()
        if use_default in ['y', 'yes', '']:
            model_path = default_model
        else:
            model_path = input("請輸入模型路徑: ").strip()
    else:
        print("未找到默認模型")
        model_path = input("請輸入模型路徑: ").strip()
    
    if not os.path.exists(model_path):
        print(f"錯誤: 模型文件不存在 {model_path}")
        return
    
    # 尋找數據目錄
    default_data = find_data_directory()
    if default_data:
        print(f"找到默認數據目錄: {default_data}")
        use_default = input("是否使用默認數據目錄? (y/n): ").lower().strip()
        if use_default in ['y', 'yes', '']:
            data_dir = default_data
        else:
            data_dir = input("請輸入數據目錄路徑: ").strip()
    else:
        print("未找到默認數據目錄")
        data_dir = input("請輸入數據目錄路徑: ").strip()
    
    if not os.path.exists(data_dir):
        print(f"錯誤: 數據目錄不存在 {data_dir}")
        return
    
    # 選擇數據集分割
    print("\n選擇數據集分割:")
    print("1. val (驗證集)")
    print("2. test (測試集)")
    print("3. train (訓練集)")
    choice = input("請選擇 (1-3, 默認為1): ").strip()
    
    split_map = {'1': 'val', '2': 'test', '3': 'train', '': 'val'}
    split = split_map.get(choice, 'val')
    
    # 設置置信度閾值
    conf_input = input("請輸入檢測置信度閾值 (0.0-1.0, 默認0.5): ").strip()
    try:
        confidence_threshold = float(conf_input) if conf_input else 0.5
        confidence_threshold = max(0.0, min(1.0, confidence_threshold))
    except ValueError:
        confidence_threshold = 0.5
    
    # 設置保存目錄
    default_save_dir = f"./deep_features_{split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir_input = input(f"請輸入特徵保存目錄 (默認 {default_save_dir}): ").strip()
    save_dir = save_dir_input if save_dir_input else default_save_dir
    
    print(f"\n=== 配置確認 ===")
    print(f"模型路徑: {model_path}")
    print(f"數據目錄: {data_dir}")
    print(f"數據集分割: {split}")
    print(f"置信度閾值: {confidence_threshold}")
    print(f"保存目錄: {save_dir}")
    print()
    
    confirm = input("確認開始特徵提取? (y/n): ").lower().strip()
    if confirm not in ['y', 'yes']:
        print("取消特徵提取")
        return
    
    # 設置日誌
    log_file = setup_logging(save_dir)
    print(f"日誌文件: {log_file}")
    
    # 開始特徵提取
    try:
        extract_features_from_dataset(
            model_path=model_path,
            data_dir=data_dir,
            save_dir=save_dir,
            split=split,
            confidence_threshold=confidence_threshold
        )
        print(f"\n✅ 特徵提取完成！結果保存在: {save_dir}")
    except Exception as e:
        print(f"\n❌ 特徵提取失敗: {str(e)}")
        logging.error(f"特徵提取失敗: {str(e)}")

def extract_features_batch():
    """批量提取多個數據集的特徵"""
    print("=== 批量特徵提取 ===")
    
    # 找到模型
    model_path = find_best_model()
    if not model_path:
        print("未找到默認模型，請先使用互動模式設置")
        return
    
    # 找到數據目錄
    data_dir = find_data_directory()
    if not data_dir:
        print("未找到默認數據目錄，請先使用互動模式設置")
        return
    
    print(f"使用模型: {model_path}")
    print(f"使用數據目錄: {data_dir}")
    
    # 批量處理不同分割
    splits = ['val', 'test']
    confidence_thresholds = [0.3, 0.5, 0.7]
    
    base_save_dir = f"./deep_features_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    for split in splits:
        for conf_thresh in confidence_thresholds:
            save_dir = os.path.join(base_save_dir, f"{split}_conf{conf_thresh}")
            
            print(f"\n處理: {split} 數據集，置信度閾值 {conf_thresh}")
            
            # 設置日誌
            log_file = setup_logging(save_dir)
            
            try:
                extract_features_from_dataset(
                    model_path=model_path,
                    data_dir=data_dir,
                    save_dir=save_dir,
                    split=split,
                    confidence_threshold=conf_thresh
                )
                print(f"✅ 完成: {split} 數據集 (置信度 {conf_thresh})")
            except Exception as e:
                print(f"❌ 失敗: {split} 數據集 (置信度 {conf_thresh}): {str(e)}")
                logging.error(f"特徵提取失敗: {str(e)}")
    
    print(f"\n🎉 批量特徵提取完成！結果保存在: {base_save_dir}")

def main():
    """主函數"""
    print("深層特徵提取工具")
    print("================")
    print()
    print("選擇運行模式:")
    print("1. 互動式特徵提取")
    print("2. 批量特徵提取")
    print("3. 退出")
    
    while True:
        choice = input("\n請選擇 (1-3): ").strip()
        
        if choice == '1':
            extract_features_interactive()
            break
        elif choice == '2':
            extract_features_batch()
            break
        elif choice == '3':
            print("再見！")
            break
        else:
            print("無效選擇，請輸入 1-3")

if __name__ == "__main__":
    main()
