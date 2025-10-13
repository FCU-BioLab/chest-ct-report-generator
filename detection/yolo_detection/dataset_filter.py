#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
資料集過濾工具 - 限制負樣本數量

此模組提供函數來過濾預處理資料集中的負樣本，
模仿 YOLOv11 的策略：每個患者最多保留 N 個負樣本

使用方式:
    from dataset_filter import filter_negative_samples
    
    filtered_dataset = filter_negative_samples(
        dataset, 
        max_negative_per_patient=20
    )
"""

import logging
from pathlib import Path
from torch.utils.data import Dataset, Subset
import re
from collections import defaultdict

LOGGER = logging.getLogger(__name__)


def extract_patient_id(file_path: Path) -> str:
    """
    從檔案路徑中提取患者 ID
    
    支援的格式：
    - A0001_slice_0001.png
    - patient_A0001/images/slice_0001.png
    
    Args:
        file_path: 圖像檔案路徑
    
    Returns:
        患者 ID (例如: 'A0001')
    """
    # 嘗試從檔名提取
    filename = file_path.stem  # 去除副檔名
    
    # 模式 1: A0001_slice_0001
    match = re.match(r'(A\d{4})', filename)
    if match:
        return match.group(1)
    
    # 模式 2: 從父目錄提取
    for parent in file_path.parents:
        match = re.match(r'(A\d{4})', parent.name)
        if match:
            return match.group(1)
    
    # 如果無法提取，使用檔名的前綴
    return filename.split('_')[0] if '_' in filename else 'unknown'


def filter_negative_samples(
    dataset: Dataset,
    max_negative_per_patient: int = 20,
) -> Subset:
    """
    過濾資料集中的負樣本
    
    保留所有正樣本，但限制每個患者的負樣本數量，
    以減少資料不平衡和訓練時間
    
    Args:
        dataset: PreprocessedYOLODataset 實例
        max_negative_per_patient: 每個患者最多保留的負樣本數量
    
    Returns:
        Subset: 過濾後的資料集子集
    """
    LOGGER.info("=" * 80)
    LOGGER.info("開始過濾負樣本...")
    LOGGER.info(f"原始樣本數: {len(dataset)}")
    LOGGER.info(f"每患者最多負樣本: {max_negative_per_patient}")
    
    # 確保 dataset 有必要的屬性
    if not hasattr(dataset, 'positive_indices') or not hasattr(dataset, 'negative_indices'):
        LOGGER.warning("資料集沒有 positive_indices/negative_indices，無法過濾")
        return dataset
    
    # 按患者分組負樣本
    patient_negatives = defaultdict(list)
    
    for idx in dataset.negative_indices:
        # 獲取圖像檔案路徑
        if hasattr(dataset, 'image_files'):
            img_path = dataset.image_files[idx]
        else:
            LOGGER.warning("無法訪問 image_files，跳過過濾")
            return dataset
        
        # 提取患者 ID
        patient_id = extract_patient_id(img_path)
        patient_negatives[patient_id].append(idx)
    
    LOGGER.info(f"檢測到 {len(patient_negatives)} 位患者的負樣本")
    
    # 限制每個患者的負樣本數量
    selected_negative_indices = []
    total_removed = 0
    
    for patient_id, neg_indices in patient_negatives.items():
        original_count = len(neg_indices)
        
        if original_count <= max_negative_per_patient:
            # 負樣本數量在限制內，全部保留
            selected_negative_indices.extend(neg_indices)
        else:
            # 隨機選擇指定數量的負樣本
            import random
            random.seed(42)  # 確保可重現
            selected = random.sample(neg_indices, max_negative_per_patient)
            selected_negative_indices.extend(selected)
            removed = original_count - max_negative_per_patient
            total_removed += removed
            
            if original_count > max_negative_per_patient * 2:  # 只記錄大幅刪減的患者
                LOGGER.debug(f"患者 {patient_id}: 保留 {max_negative_per_patient}/{original_count} 負樣本")
    
    # 合併正樣本和篩選後的負樣本
    selected_indices = sorted(list(dataset.positive_indices) + selected_negative_indices)
    
    LOGGER.info("=" * 80)
    LOGGER.info("過濾完成！")
    LOGGER.info(f"正樣本數: {len(dataset.positive_indices)} (全部保留)")
    LOGGER.info(f"原始負樣本數: {len(dataset.negative_indices)}")
    LOGGER.info(f"過濾後負樣本數: {len(selected_negative_indices)}")
    LOGGER.info(f"移除負樣本數: {total_removed}")
    LOGGER.info(f"最終樣本數: {len(selected_indices)} (原始: {len(dataset)})")
    LOGGER.info(f"壓縮率: {len(selected_indices)/len(dataset)*100:.1f}%")
    LOGGER.info("=" * 80)
    
    # 返回 Subset
    return Subset(dataset, selected_indices)


def get_dataset_statistics(dataset: Dataset) -> dict:
    """
    獲取資料集統計信息
    
    Args:
        dataset: 資料集實例
    
    Returns:
        統計信息字典
    """
    stats = {
        'total': len(dataset),
        'positive': 0,
        'negative': 0,
    }
    
    if hasattr(dataset, 'positive_indices'):
        stats['positive'] = len(dataset.positive_indices)
    
    if hasattr(dataset, 'negative_indices'):
        stats['negative'] = len(dataset.negative_indices)
    
    # 如果是 Subset，嘗試從原始資料集獲取
    if hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
        original_dataset = dataset.dataset
        if hasattr(original_dataset, 'positive_indices'):
            # 計算 subset 中有多少正樣本
            subset_indices_set = set(dataset.indices)
            positive_indices_set = set(original_dataset.positive_indices)
            negative_indices_set = set(original_dataset.negative_indices)
            
            stats['positive'] = len(subset_indices_set & positive_indices_set)
            stats['negative'] = len(subset_indices_set & negative_indices_set)
    
    return stats
