#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
數據集處理模組
包含數據集創建、分割和管理功能
"""

import os
import logging
import numpy as np
import torchvision.transforms as transforms
from faster_rcnn_dataset import CTDetectionDataset
from metrics.dataset_statistics import calculate_dataset_statistics, save_patient_lists


def create_train_val_datasets(data_dir, val_split=0.2, random_seed=42, 
                              include_negative_samples=True, max_negative_per_patient=0):
    """創建按病例分割的訓練/驗證數據集"""
    # 載入完整數據集
    full_dataset = CTDetectionDataset(
        data_root=data_dir,
        split='train',
        target_size=512,
        specific_patients=None,
        transforms=transforms.Compose([
            transforms.ToTensor()
        ]),
        include_negative_samples=include_negative_samples,
        max_negative_per_patient=max_negative_per_patient
    )
    
    # 收集所有病例ID
    all_patient_ids = set()
    for sample in full_dataset.samples:
        all_patient_ids.add(sample['patient_id'])
    
    all_patient_ids = sorted(list(all_patient_ids))
    total_patients = len(all_patient_ids)
    
    logging.info(f"數據集總大小: {len(full_dataset)} 張圖像")
    logging.info(f"總病例數: {total_patients} 位")
    
    # 設置隨機種子並按病例分割
    np.random.seed(random_seed)
    np.random.shuffle(all_patient_ids)
    
    # 計算分割點
    val_patient_count = int(val_split * total_patients)
    train_patient_count = total_patients - val_patient_count
    
    # 分割病例
    train_patient_ids = all_patient_ids[:train_patient_count]
    val_patient_ids = all_patient_ids[train_patient_count:]
    
    # 排序病例列表以便於閱讀和記錄
    train_patient_ids.sort()
    val_patient_ids.sort()
    
    logging.info(f"訓練集病例數: {len(train_patient_ids)} 位 ({len(train_patient_ids)/total_patients*100:.1f}%)")
    logging.info(f"驗證集病例數: {len(val_patient_ids)} 位 ({len(val_patient_ids)/total_patients*100:.1f}%)")
    logging.info(f"訓練集病例列表: {', '.join(train_patient_ids[:10])}{'...' if len(train_patient_ids) > 10 else ''}")
    logging.info(f"驗證集病例列表: {', '.join(val_patient_ids[:10])}{'...' if len(val_patient_ids) > 10 else ''}")
    
    # 檢查病例重疊（應該為空）
    overlap = set(train_patient_ids) & set(val_patient_ids)
    if overlap:
        logging.error(f"錯誤：訓練集和驗證集病例重疊: {overlap}")
        raise ValueError("訓練集和驗證集不應有病例重疊")
    else:
        logging.info("✓ 訓練集和驗證集病例無重疊")
    
    # 創建按病例分割的數據集
    train_dataset = CTDetectionDataset(
        data_root=data_dir,
        split='train',
        target_size=512,
        specific_patients=train_patient_ids,
        transforms=transforms.Compose([
            transforms.ToTensor()
        ]),
        include_negative_samples=include_negative_samples,
        max_negative_per_patient=max_negative_per_patient
    )
    
    val_dataset = CTDetectionDataset(
        data_root=data_dir,
        split='train',  # 使用相同的split，但指定不同的病例
        target_size=512,
        specific_patients=val_patient_ids,
        transforms=transforms.Compose([
            transforms.ToTensor()
        ]),
        include_negative_samples=include_negative_samples,
        max_negative_per_patient=max_negative_per_patient
    )
    
    logging.info(f"訓練集圖像數: {len(train_dataset)} 張")
    logging.info(f"驗證集圖像數: {len(val_dataset)} 張")
    
    # 計算詳細統計信息
    train_stats = calculate_dataset_statistics(train_dataset, "訓練集")
    val_stats = calculate_dataset_statistics(val_dataset, "驗證集")
    
    # 合併統計信息，包含病例列表
    dataset_stats = {
        'total_dataset_size': len(full_dataset),
        'train_stats': train_stats,
        'val_stats': val_stats,
        'split_ratio': {'train': len(train_patient_ids)/total_patients, 'val': len(val_patient_ids)/total_patients},
        'train_patient_ids': train_patient_ids,
        'val_patient_ids': val_patient_ids,
        'train_patient_count': len(train_patient_ids),
        'val_patient_count': len(val_patient_ids),
        'total_patient_count': total_patients
    }
    
    return train_dataset, val_dataset, dataset_stats


def create_kfold_datasets(data_dir, n_folds=5, random_seed=42, 
                         include_negative_samples=True, max_negative_per_patient=0):
    """Create K-fold cross-validation datasets split by patient"""
    logging.info(f"Creating {n_folds}-fold cross-validation datasets...")
    
    # Load complete dataset
    full_dataset = CTDetectionDataset(
        data_root=data_dir,
        split='train',
        target_size=640,
        specific_patients=None,
        transforms=transforms.Compose([transforms.ToTensor()]),
        include_negative_samples=include_negative_samples,
        max_negative_per_patient=max_negative_per_patient
    )
    
    # Collect all patient IDs
    all_patient_ids = set()
    for sample in full_dataset.samples:
        all_patient_ids.add(sample['patient_id'])
    
    all_patient_ids = sorted(list(all_patient_ids))
    total_patients = len(all_patient_ids)
    
    logging.info(f"Total patients: {total_patients}")
    logging.info(f"Total dataset size: {len(full_dataset)} images")
    
    # Set random seed and shuffle patients
    np.random.seed(random_seed)
    np.random.shuffle(all_patient_ids)
    
    # Create fold splits
    fold_size = total_patients // n_folds
    folds = []
    
    for fold in range(n_folds):
        start_idx = fold * fold_size
        if fold == n_folds - 1:  # Last fold gets remaining patients
            end_idx = total_patients
        else:
            end_idx = start_idx + fold_size
        
        fold_patient_ids = all_patient_ids[start_idx:end_idx]
        folds.append(fold_patient_ids)
        logging.info(f"Fold {fold+1}: {len(fold_patient_ids)} patients")
    
    return folds, full_dataset
