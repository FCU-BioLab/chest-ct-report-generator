#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download YOLO Models - 下载YOLO模型脚本
用于预先下载训练所需的YOLO预训练模型
"""

import os
import sys

def download_yolo_models():
    """下载YOLO模型"""
    print("开始下载YOLO预训练模型...")
    
    try:
        from ultralytics import YOLO
        print("✅ ultralytics 导入成功")
    except ImportError:
        print("❌ ultralytics 未安装，请运行: pip install ultralytics")
        return False
    
    # 要下载的模型列表
    models_to_download = [
        'yolo11n.pt',
        'yolo11s.pt', 
        'yolo11m.pt',
        'yolov8n.pt',  # 备选模型
        'yolov8s.pt',  # 备选模型
    ]
    
    successful_downloads = []
    failed_downloads = []
    
    for model_name in models_to_download:
        try:
            print(f"\n正在下载 {model_name}...")
            model = YOLO(model_name)
            print(f"✅ {model_name} 下载成功")
            successful_downloads.append(model_name)
        except Exception as e:
            print(f"❌ {model_name} 下载失败: {e}")
            failed_downloads.append(model_name)
    
    print("\n" + "=" * 50)
    print("下载结果摘要:")
    print("=" * 50)
    
    if successful_downloads:
        print(f"✅ 成功下载 {len(successful_downloads)} 个模型:")
        for model in successful_downloads:
            print(f"   - {model}")
    
    if failed_downloads:
        print(f"\n❌ 失败 {len(failed_downloads)} 个模型:")
        for model in failed_downloads:
            print(f"   - {model}")
    
    if successful_downloads:
        print(f"\n🎉 至少有 {len(successful_downloads)} 个模型可用于训练！")
        return True
    else:
        print("\n😞 没有成功下载任何模型")
        return False

def main():
    """主函数"""
    print("YOLO模型下载工具")
    print("此脚本将下载训练所需的YOLO预训练模型")
    print()
    
    success = download_yolo_models()
    
    if success:
        print("\n现在您可以开始训练:")
        print("python detection/train_yolov11_simple.py --data_dir datasets/all_patient_data")
        print("或运行环境测试:")
        print("python detection/test_yolov11_setup.py")
    else:
        print("\n请检查网络连接或安装最新版本的ultralytics")

if __name__ == "__main__":
    main()
