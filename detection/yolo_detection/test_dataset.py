#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試資料集載入
"""

import sys
sys.path.append('../..')
from faster_rcnn_detection.faster_rcnn_dataset import CTDetectionDataset

# 測試載入訓練集
try:
    print("測試載入訓練集中的特定患者...")
    dataset = CTDetectionDataset(
        data_root='../../datasets/splited_dataset',
        split='train',
        target_size=640,
        specific_patients=['A0001', 'A0002'],  # 測試前兩個患者
        include_negative_samples=True,
        max_negative_per_patient=0
    )
    print(f'成功載入 {len(dataset.samples)} 個樣本')
    if len(dataset.samples) > 0:
        print(f'第一個樣本: {dataset.samples[0]}')
    else:
        print("沒有載入任何樣本")
        
except Exception as e:
    print(f'錯誤: {e}')
    import traceback
    traceback.print_exc()