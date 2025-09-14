#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型評估模組
包含模型評估和推理時間測量功能
"""

import time
import logging
import torch
import numpy as np
from tqdm import tqdm
from metrics.detection_metrics import calculate_comprehensive_metrics, calculate_detection_metrics


def evaluate_model(model, val_loader, device, return_samples=False, max_samples=10, comprehensive=True):
    """評估模型"""
    model.eval()
    all_predictions = []
    all_targets = []
    sample_images = []
    sample_predictions = []
    sample_targets = []
    
    # 效率指標
    inference_times = []
    memory_usage = []
    
    val_pbar = tqdm(val_loader, desc="評估模型", unit="batch", ncols=100)
    
    with torch.no_grad():
        for images, targets in val_pbar:
            images = [img.to(device) for img in images]
            
            # 測量推理時間
            start_time = time.time()
            predictions = model(images)
            inference_time = time.time() - start_time
            inference_times.append(inference_time / len(images))  # 每張圖的平均時間
            
            # 測量顯存使用（如果使用GPU）
            if device.type == 'cuda':
                memory_usage.append(torch.cuda.memory_allocated() / 1024**2)  # MB
            
            for i, (pred, target) in enumerate(zip(predictions, targets)):
                all_predictions.append({
                    'boxes': pred['boxes'].cpu(),
                    'scores': pred['scores'].cpu(),
                    'labels': pred['labels'].cpu()
                })
                all_targets.append({
                    'boxes': target['boxes'],
                    'labels': target['labels']
                })
                
                # 收集樣本用於可視化
                if return_samples and len(sample_images) < max_samples:
                    sample_images.append(images[i].cpu())
                    sample_predictions.append({
                        'boxes': pred['boxes'].cpu(),
                        'scores': pred['scores'].cpu(),
                        'labels': pred['labels'].cpu()
                    })
                    sample_targets.append({
                        'boxes': target['boxes'],
                        'labels': target['labels']
                    })
            
            val_pbar.set_postfix({'Samples': f'{len(all_predictions)}'})
    
    val_pbar.close()
    
    # 計算指標
    if comprehensive:
        metrics = calculate_comprehensive_metrics(all_predictions, all_targets, iou_threshold=0.5)
        
        # 添加效率指標
        if inference_times:
            metrics['inference_time_per_image'] = np.mean(inference_times)
            metrics['fps'] = 1.0 / np.mean(inference_times) if np.mean(inference_times) > 0 else 0
            metrics['inference_time_std'] = np.std(inference_times)
        
        if memory_usage and device.type == 'cuda':
            metrics['avg_memory_usage_mb'] = np.mean(memory_usage)
            metrics['max_memory_usage_mb'] = np.max(memory_usage)
    else:
        # 保持向後兼容性
        metrics = calculate_detection_metrics(all_predictions, all_targets, iou_threshold=0.5)
    
    if return_samples:
        return metrics, sample_images, sample_predictions, sample_targets
    else:
        return metrics
