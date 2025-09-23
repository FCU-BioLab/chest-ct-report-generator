#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IoU計算相關函數模組
包含各種IoU變體的計算方法
"""

import math
import torch
import numpy as np


def calculate_iou_variants(box1, box2):
    """一次性計算所有IoU變體 - 優化版本"""
    # 基本IoU計算
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    iou = intersection / union if union > 0 else 0
    
    # 計算最小外包矩形（GIoU用）
    c_x1 = min(box1[0], box2[0])
    c_y1 = min(box1[1], box2[1])
    c_x2 = max(box1[2], box2[2])
    c_y2 = max(box1[3], box2[3])
    c_area = (c_x2 - c_x1) * (c_y2 - c_y1)
    
    # GIoU
    giou = iou - (c_area - union) / c_area if c_area > 0 else iou
    
    # 中心點距離（DIoU和CIoU用）
    center1_x = (box1[0] + box1[2]) / 2
    center1_y = (box1[1] + box1[3]) / 2
    center2_x = (box2[0] + box2[2]) / 2
    center2_y = (box2[1] + box2[3]) / 2
    
    center_distance_sq = (center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2
    diagonal_distance_sq = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2
    
    # DIoU
    diou = iou - center_distance_sq / diagonal_distance_sq if diagonal_distance_sq > 0 else iou
    
    # CIoU
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    
    if h1 > 0 and h2 > 0 and w1 > 0 and w2 > 0:
        v = (4 / (math.pi ** 2)) * ((math.atan(w2/h2) - math.atan(w1/h1)) ** 2)
        alpha = v / (1 - iou + v) if (1 - iou + v) > 0 else 0
        ciou = diou - alpha * v
    else:
        ciou = diou
    
    return iou, giou, diou, ciou


def calculate_iou_matrix(boxes1, boxes2):
    """計算兩組邊界框之間的IoU矩陣"""
    iou_matrix = torch.zeros(len(boxes1), len(boxes2))
    
    for i, box1 in enumerate(boxes1):
        for j, box2 in enumerate(boxes2):
            iou, _, _, _ = calculate_iou_variants(box1, box2)
            iou_matrix[i][j] = iou
    
    return iou_matrix


def calculate_bbox_error(pred_box, target_box):
    """計算邊界框位置與大小誤差"""
    # 中心點誤差
    pred_center_x = (pred_box[0] + pred_box[2]) / 2
    pred_center_y = (pred_box[1] + pred_box[3]) / 2
    target_center_x = (target_box[0] + target_box[2]) / 2
    target_center_y = (target_box[1] + target_box[3]) / 2
    
    center_error = ((pred_center_x - target_center_x) ** 2 + (pred_center_y - target_center_y) ** 2) ** 0.5
    
    # 尺寸誤差
    pred_w = pred_box[2] - pred_box[0]
    pred_h = pred_box[3] - pred_box[1]
    target_w = target_box[2] - target_box[0]
    target_h = target_box[3] - target_box[1]
    
    size_error = abs(pred_w - target_w) + abs(pred_h - target_h)
    
    return {
        'center_error': center_error,
        'size_error': size_error,
        'total_error': center_error + size_error
    }


def calculate_giou(box1, box2):
    """Calculate Generalized IoU (GIoU) between two bounding boxes
    
    Args:
        box1: [x1, y1, x2, y2] format
        box2: [x1, y1, x2, y2] format
    
    Returns:
        float: GIoU value
    """
    iou, giou, _, _ = calculate_iou_variants(box1, box2)
    return giou


def calculate_diou(box1, box2):
    """Calculate Distance IoU (DIoU) between two bounding boxes
    
    Args:
        box1: [x1, y1, x2, y2] format
        box2: [x1, y1, x2, y2] format
    
    Returns:
        float: DIoU value
    """
    _, _, diou, _ = calculate_iou_variants(box1, box2)
    return diou


def calculate_ciou(box1, box2):
    """Calculate Complete IoU (CIoU) between two bounding boxes
    
    Args:
        box1: [x1, y1, x2, y2] format
        box2: [x1, y1, x2, y2] format
    
    Returns:
        float: CIoU value
    """
    _, _, _, ciou = calculate_iou_variants(box1, box2)
    return ciou
