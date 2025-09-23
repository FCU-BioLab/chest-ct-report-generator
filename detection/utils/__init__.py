#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用工具模組
包含常用的工具函數
"""

import os
import sys
import logging
from datetime import datetime


def collate_fn(batch):
    """自定義批次整理函數，避免 lambda 函數在多進程中的問題"""
    # batch 是一個包含字典的列表，每個字典包含 'image' 和 'target' 鍵
    images = []
    targets = []
    
    for item in batch:
        # CTDetectionDataset 返回的是字典格式
        if isinstance(item, dict) and 'image' in item and 'target' in item:
            images.append(item['image'])
            targets.append(item['target'])
        else:
            # 如果數據格式不對，打印調試信息
            print(f"Unexpected batch item format: {type(item)}")
            if hasattr(item, 'keys'):
                print(f"Keys: {list(item.keys())}")
    
    return images, targets


def setup_logging(log_dir):
    """設置日誌記錄"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'simple_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # 清除已有的處理器
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 創建格式器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                 datefmt='%Y-%m-%d %H:%M:%S')
    
    # 文件處理器 - 使用UTF-8編碼
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


def setup_console_encoding():
    """設置控制台編碼 (Windows)"""
    if sys.platform.startswith('win'):
        try:
            os.system('chcp 65001 >nul')
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except:
            pass
