#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
數據集統計模組
包含數據集統計分析功能
"""

import logging
from datetime import datetime


def calculate_dataset_statistics(dataset, dataset_name="Dataset"):
    """計算數據集的詳細統計信息"""
    total_images = len(dataset)
    total_annotations = 0
    images_with_annotations = 0
    images_without_annotations = 0
    
    logging.info(f"正在計算 {dataset_name} 統計信息...")
    
    for i in range(total_images):
        try:
            sample = dataset[i]
            if isinstance(sample, dict) and 'target' in sample:
                target = sample['target']
            else:
                # 如果是tuple格式 (image, target)
                _, target = sample
            
            # 計算標記數量
            if 'boxes' in target and len(target['boxes']) > 0:
                num_boxes = len(target['boxes'])
                total_annotations += num_boxes
                images_with_annotations += 1
            else:
                images_without_annotations += 1
                
        except Exception as e:
            logging.warning(f"計算 {dataset_name} 第 {i} 個樣本時出錯: {e}")
            images_without_annotations += 1
    
    stats = {
        'dataset_name': dataset_name,
        'total_images': total_images,
        'total_annotations': total_annotations,
        'images_with_annotations': images_with_annotations,
        'images_without_annotations': images_without_annotations,
        'avg_annotations_per_image': total_annotations / total_images if total_images > 0 else 0,
        'avg_annotations_per_annotated_image': total_annotations / images_with_annotations if images_with_annotations > 0 else 0
    }
    
    # 記錄統計信息
    logging.info(f"=== {dataset_name} 統計信息 ===")
    logging.info(f"總影像數: {total_images}")
    logging.info(f"總標記數: {total_annotations}")
    logging.info(f"有標記的影像數: {images_with_annotations}")
    logging.info(f"無標記的影像數: {images_without_annotations}")
    logging.info(f"平均每張影像標記數: {stats['avg_annotations_per_image']:.2f}")
    if images_with_annotations > 0:
        logging.info(f"平均每張有標記影像的標記數: {stats['avg_annotations_per_annotated_image']:.2f}")
    
    # 檢查負類別（無病灶影像）的存在
    if images_without_annotations == 0:
        logging.warning(f"⚠️  {dataset_name} 中沒有無標記的影像（負類別）")
        logging.warning("   這會影響ROC/AUC等指標的計算")
        logging.warning("   建議：加入無病灶的影像以獲得更全面的評估指標")
    else:
        logging.info(f"✓ {dataset_name} 包含 {images_without_annotations} 張無標記影像（負類別）")
    
    return stats


def save_patient_lists(save_dir, dataset_stats):
    """保存訓練集和驗證集的病例列表到單獨文件"""
    import os
    
    # 保存訓練集病例列表
    train_patients_file = os.path.join(save_dir, 'train_patient_list.txt')
    with open(train_patients_file, 'w', encoding='utf-8') as f:
        f.write("# 訓練集病例列表\n")
        f.write(f"# 總計 {len(dataset_stats['train_patient_ids'])} 位病例\n")
        f.write(f"# 生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for patient_id in dataset_stats['train_patient_ids']:
            f.write(f"{patient_id}\n")
    
    # 保存驗證集病例列表
    val_patients_file = os.path.join(save_dir, 'val_patient_list.txt')
    with open(val_patients_file, 'w', encoding='utf-8') as f:
        f.write("# 驗證集病例列表\n")
        f.write(f"# 總計 {len(dataset_stats['val_patient_ids'])} 位病例\n")
        f.write(f"# 生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for patient_id in dataset_stats['val_patient_ids']:
            f.write(f"{patient_id}\n")
    
    # 保存詳細的病例分佈摘要
    summary_file = os.path.join(save_dir, 'patient_split_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=== 病例分佈摘要 ===\n")
        f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        total_patients = dataset_stats.get('total_patient_count', len(dataset_stats['train_patient_ids']) + len(dataset_stats['val_patient_ids']))
        f.write(f"總病例數: {total_patients}\n")
        f.write(f"訓練集病例數: {len(dataset_stats['train_patient_ids'])} ({len(dataset_stats['train_patient_ids'])/total_patients*100:.1f}%)\n")
        f.write(f"驗證集病例數: {len(dataset_stats['val_patient_ids'])} ({len(dataset_stats['val_patient_ids'])/total_patients*100:.1f}%)\n\n")
        
        # 檢查病例重疊
        overlap = set(dataset_stats['train_patient_ids']) & set(dataset_stats['val_patient_ids'])
        if overlap:
            f.write(f"⚠️  警告：訓練集和驗證集病例重疊: {overlap}\n\n")
        else:
            f.write("✓ 訓練集和驗證集病例無重疊\n\n")
        
        f.write("訓練集病例列表:\n")
        f.write(", ".join(dataset_stats['train_patient_ids']))
        f.write("\n\n")
        
        f.write("驗證集病例列表:\n")
        f.write(", ".join(dataset_stats['val_patient_ids']))
        f.write("\n\n")
        
        # 添加數據統計信息
        train_stats = dataset_stats['train_stats']
        val_stats = dataset_stats['val_stats']
        f.write("=== 數據統計 ===\n")
        f.write(f"訓練集: {train_stats['total_images']} 張圖像, {train_stats['total_annotations']} 個標記\n")
        f.write(f"驗證集: {val_stats['total_images']} 張圖像, {val_stats['total_annotations']} 個標記\n")
        f.write(f"總計: {train_stats['total_images'] + val_stats['total_images']} 張圖像, {train_stats['total_annotations'] + val_stats['total_annotations']} 個標記\n")
    
    logging.info(f"病例列表已保存到:")
    logging.info(f"  - 訓練集: {train_patients_file}")
    logging.info(f"  - 驗證集: {val_patients_file}")
    logging.info(f"  - 摘要: {summary_file}")
