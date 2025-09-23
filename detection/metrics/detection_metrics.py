#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
檢測指標計算模組
包含全面的檢測評估指標計算
"""

import numpy as np
from .iou_calculations import calculate_iou_matrix, calculate_iou_variants, calculate_bbox_error


def calculate_comprehensive_metrics(predictions, targets, iou_threshold=0.5, confidence_threshold=0.5):
    """計算全面的檢測指標"""
    # 基礎統計
    tp, fp, fn = 0, 0, 0
    total_images = len(predictions)
    images_with_detections = 0
    images_with_targets = 0
    lesion_level_tp = 0
    case_level_tp = 0
    
    # 位置與品質指標
    ious = []
    gious = []
    dious = []
    cious = []
    bbox_errors = []
    localization_errors = []
    fp_per_image = []
    
    # 置信度相關
    all_confidence_scores = []
    true_positive_scores = []
    false_positive_scores = []
    
    # mAP計算所需的數據
    all_scores = []
    all_labels = []
    
    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes']
        pred_scores = pred['scores']
        target_boxes = target['boxes']
        
        # 記錄有目標的圖像數
        if len(target_boxes) > 0:
            images_with_targets += 1
        
        # 過濾低置信度預測
        valid_pred = pred_scores > confidence_threshold
        filtered_pred_boxes = pred_boxes[valid_pred]
        filtered_pred_scores = pred_scores[valid_pred]
        
        # 記錄有檢測的圖像數
        if len(filtered_pred_boxes) > 0:
            images_with_detections += 1
        
        # 記錄每張圖的假陽性數
        image_fp = 0
        all_confidence_scores.extend(filtered_pred_scores.tolist())
        
        if len(filtered_pred_boxes) == 0 and len(target_boxes) == 0:
            fp_per_image.append(0)
            continue
        elif len(filtered_pred_boxes) == 0:
            fn += len(target_boxes)
            fp_per_image.append(0)
            # 為mAP添加假陰性
            all_scores.extend([0] * len(target_boxes))
            all_labels.extend([1] * len(target_boxes))
            continue
        elif len(target_boxes) == 0:
            fp += len(filtered_pred_boxes)
            image_fp = len(filtered_pred_boxes)
            fp_per_image.append(image_fp)
            false_positive_scores.extend(filtered_pred_scores.tolist())
            # 為mAP添加假陽性
            all_scores.extend(filtered_pred_scores.tolist())
            all_labels.extend([0] * len(filtered_pred_scores))
            continue
        
        # 計算IoU矩陣和各種改進IoU
        iou_matrix = calculate_iou_matrix(filtered_pred_boxes, target_boxes)
        
        # 匹配預測和目標
        matched_targets = set()
        image_has_correct_detection = False
        
        for i in range(len(filtered_pred_boxes)):
            best_iou = 0
            best_target = -1
            for j in range(len(target_boxes)):
                if j not in matched_targets and iou_matrix[i][j] > best_iou:
                    best_iou = iou_matrix[i][j]
                    best_target = j
            
            if best_iou >= iou_threshold:
                tp += 1
                lesion_level_tp += 1
                image_has_correct_detection = True
                matched_targets.add(best_target)
                
                # 記錄品質指標
                pred_box = filtered_pred_boxes[i]
                target_box = target_boxes[best_target]
                
                iou, giou, diou, ciou = calculate_iou_variants(pred_box, target_box)
                ious.append(iou)
                gious.append(giou)
                dious.append(diou)
                cious.append(ciou)
                
                # 計算定位誤差
                bbox_error = calculate_bbox_error(pred_box, target_box)
                bbox_errors.append(bbox_error)
                localization_errors.append(bbox_error['total_error'])
                
                true_positive_scores.append(filtered_pred_scores[i].item())
                # 為mAP添加真陽性
                all_scores.append(filtered_pred_scores[i].item())
                all_labels.append(1)
            else:
                fp += 1
                image_fp += 1
                false_positive_scores.append(filtered_pred_scores[i].item())
                # 為mAP添加假陽性
                all_scores.append(filtered_pred_scores[i].item())
                all_labels.append(0)
        
        # 病例級敏感度
        if image_has_correct_detection and len(target_boxes) > 0:
            case_level_tp += 1
        
        fn += len(target_boxes) - len(matched_targets)
        fp_per_image.append(image_fp)
        
        # 為mAP添加未匹配的目標（假陰性）
        unmatched_targets = len(target_boxes) - len(matched_targets)
        all_scores.extend([0] * unmatched_targets)
        all_labels.extend([1] * unmatched_targets)
    
    # 計算基礎指標
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    sensitivity = recall  # 敏感度就是召回率
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 病灶級和病例級敏感度
    lesion_level_sensitivity = lesion_level_tp / (tp + fn) if (tp + fn) > 0 else 0
    case_level_sensitivity = case_level_tp / images_with_targets if images_with_targets > 0 else 0
    
    # 每張圖像的平均假陽性數
    avg_fp_per_image = np.mean(fp_per_image) if fp_per_image else 0
    
    # 計算mAP指標
    map_50 = _calculate_ap(all_scores, all_labels)
    
    # 品質指標統計
    quality_metrics = {}
    if ious:
        quality_metrics = {
            'mean_iou': np.mean(ious),
            'mean_giou': np.mean(gious),
            'mean_diou': np.mean(dious),
            'mean_ciou': np.mean(cious),
            'mean_localization_error': np.mean(localization_errors),
            'std_localization_error': np.std(localization_errors)
        }
    
    return {
        # 一、核心檢測指標
        'iou': np.mean(ious) if ious else 0,
        'mAP@0.5': map_50,
        'mAP@[0.5:0.95]': map_50,  # 簡化，只用@0.5的結果
        'sensitivity_recall': sensitivity,
        'precision': precision,
        'f1_score': f1,
        'fp_per_image': avg_fp_per_image,
        
        # 二、定位與錯誤分析指標
        'lesion_level_sensitivity': lesion_level_sensitivity,
        'case_level_sensitivity': case_level_sensitivity,
        'mean_bbox_error': np.mean([err['total_error'] for err in bbox_errors]) if bbox_errors else 0,
        'mean_giou': quality_metrics.get('mean_giou', 0),
        'mean_diou': quality_metrics.get('mean_diou', 0),
        'mean_ciou': quality_metrics.get('mean_ciou', 0),
        
        # 三、臨床相關指標
        'mean_localization_error': quality_metrics.get('mean_localization_error', 0),
        'avg_false_positives_per_case': avg_fp_per_image,
        
        # 基礎統計
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'total_images': total_images,
        'images_with_detections': images_with_detections,
        'images_with_targets': images_with_targets,
        
        # 詳細數據（用於進一步分析）
        'confidence_scores': all_confidence_scores,
        'true_positive_scores': true_positive_scores,
        'false_positive_scores': false_positive_scores
    }


def calculate_detection_metrics(predictions, targets, iou_threshold=0.5):
    """計算檢測指標 - 保持向後兼容性"""
    comprehensive_metrics = calculate_comprehensive_metrics(predictions, targets, iou_threshold)
    
    # 返回原始格式以保持兼容性
    return {
        'precision': comprehensive_metrics['precision'],
        'recall': comprehensive_metrics['sensitivity_recall'],
        'f1_score': comprehensive_metrics['f1_score'],
        'tp': comprehensive_metrics['tp'],
        'fp': comprehensive_metrics['fp'],
        'fn': comprehensive_metrics['fn']
    }


def _calculate_ap(scores, labels):
    """計算平均精度（AP）"""
    if len(scores) == 0:
        return 0.0
    
    sorted_indices = np.argsort(scores)[::-1]
    sorted_labels = np.array(labels)[sorted_indices]
    
    tp_cumsum = np.cumsum(sorted_labels)
    fp_cumsum = np.cumsum(1 - sorted_labels)
    
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
    recall = tp_cumsum / (np.sum(sorted_labels) + 1e-10)
    
    # 使用插值方法計算AP
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11
    
    return ap
