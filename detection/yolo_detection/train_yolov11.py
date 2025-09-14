#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv11 K-Fold Cross-Validation Training for CT Detection
基于原有 Faster R-CNN 训练逻辑，适配 YOLOv11 目标检测模型

主要功能：
1. K-fold交叉验证训练
2. 综合指标评估 (mAP, IoU variants, FROC curves)
3. 自动模型保存和结果可视化
4. 支持负样本包含和数据增强

作者: Based on Faster R-CNN training logic
日期: 2025-09-06
"""

import os
import sys
import json
import time
import logging
import math
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import cv2

try:
    from sklearn.metrics import roc_curve, auc
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. ROC/AUC metrics will be skipped.")

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logging.error("ultralytics not available. Please install: pip install ultralytics")
    sys.exit(1)

# 导入自定义模块
from faster_rcnn_dataset import CTDetectionDataset

# 設置控制台編碼 (Windows)
if sys.platform.startswith('win'):
    try:
        os.system('chcp 65001 >nul')
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

# 尝试导入模块化计算函数
try:
    from metrics.detection_metrics import calculate_comprehensive_metrics, calculate_detection_metrics
    from metrics.roc_froc import calculate_roc_froc_curves
    from metrics.dataset_statistics import calculate_dataset_statistics, save_patient_lists
    from metrics.iou_calculations import calculate_giou, calculate_diou, calculate_ciou, calculate_bbox_error, calculate_iou_matrix
    from visualization import visualize_predictions, create_prediction_summary, create_comprehensive_summary, create_kfold_summary_plots
    from data_processing import create_kfold_datasets
    from evaluation import evaluate_model
    from utils import collate_fn, setup_logging
    MODULES_IMPORTED = True
    logging.info("Successfully imported modular computation functions")
except ImportError as e:
    logging.warning(f"Modular import failed: {e}")
    logging.warning("Using inline functions as fallback")
    MODULES_IMPORTED = False


class YOLOv11CTDataset:
    """YOLOv11格式的CT检测数据集适配器"""
    
    def __init__(self, data_dir: str, split: str = 'train', include_negative_samples: bool = True, 
                 max_negative_per_patient: int = 0, patient_ids: Optional[List[str]] = None):
        """
        初始化YOLOv11数据集适配器
        
        Args:
            data_dir: 数据根目录
            split: 数据集分割类型 ('train', 'val', 'test')
            include_negative_samples: 是否包含负样本
            max_negative_per_patient: 每个病例最大负样本数
            patient_ids: 指定使用的病例ID列表
        """
        self.data_dir = data_dir
        self.split = split
        self.include_negative_samples = include_negative_samples
        self.max_negative_per_patient = max_negative_per_patient
        self.patient_ids = patient_ids
        
        # 创建Faster R-CNN数据集实例来获取数据
        # When patient_ids are specified, use 'all' split to access the flat directory structure
        actual_split = 'all' if patient_ids is not None else split
        self.rcnn_dataset = CTDetectionDataset(
            data_root=data_dir,
            split=actual_split,
            target_size=640,  # YOLOv11推荐尺寸
            specific_patients=patient_ids,
            transforms=transforms.Compose([transforms.ToTensor()]),
            include_negative_samples=include_negative_samples,
            max_negative_per_patient=max_negative_per_patient
        )
        
        self.samples = self.rcnn_dataset.samples
        
    def prepare_yolo_format(self, output_dir: str) -> str:
        """
        将数据转换为YOLOv11格式并保存
        
        Args:
            output_dir: 输出目录
            
        Returns:
            str: YOLO数据集配置文件路径
        """
        # 创建YOLO格式目录结构
        yolo_dir = os.path.join(output_dir, f'yolo_dataset_{self.split}')
        images_dir = os.path.join(yolo_dir, 'images', self.split)
        labels_dir = os.path.join(yolo_dir, 'labels', self.split)
        
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        logging.info(f"Preparing YOLOv11 format dataset to: {yolo_dir}")
        
        # Process each sample
        for idx, sample in enumerate(tqdm(self.samples, desc=f"Converting {self.split} data")):
            # 获取图像和标注
            item = self.rcnn_dataset[idx]
            image = item['image']
            target = item['target']
            
            # 保存图像
            # Use available sample information to create filename
            patient_id = sample.get('patient_id', 'unknown')
            file_name = sample.get('file_name', 'unknown')
            sop_instance_uid = sample.get('sop_instance_uid', str(idx))
            
            # Create a simplified filename using available information
            image_filename = f"{patient_id}_{sop_instance_uid}_{idx:06d}.png"
            image_path = os.path.join(images_dir, image_filename)
            
            # 转换图像格式并保存
            if isinstance(image, torch.Tensor):
                # 从tensor转换为numpy
                if image.dim() == 3 and image.shape[0] == 1:
                    image_np = image.squeeze(0).numpy()
                elif image.dim() == 3 and image.shape[0] == 3:
                    image_np = image.permute(1, 2, 0).numpy()
                else:
                    image_np = image.numpy()
                
                # 归一化到0-255
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
                
                # 保存为PNG
                if len(image_np.shape) == 2:
                    cv2.imwrite(image_path, image_np)
                else:
                    cv2.imwrite(image_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            
            # 转换标注为YOLO格式
            # Create label filename to match image filename
            label_filename = f"{patient_id}_{sop_instance_uid}_{idx:06d}.txt"
            label_path = os.path.join(labels_dir, label_filename)
            
            # 获取图像尺寸
            img_height, img_width = 640, 640  # YOLOv11输入尺寸
            
            with open(label_path, 'w') as f:
                if 'boxes' in target and len(target['boxes']) > 0:
                    for box, label in zip(target['boxes'], target['labels']):
                        # 转换边界框格式：从[x1,y1,x2,y2]到[center_x, center_y, width, height]
                        x1, y1, x2, y2 = box.tolist()
                        
                        # 计算中心点和宽高（归一化）
                        center_x = (x1 + x2) / 2.0 / img_width
                        center_y = (y1 + y2) / 2.0 / img_height
                        width = (x2 - x1) / img_width
                        height = (y2 - y1) / img_height
                        
                        # YOLO类别ID（减1，因为我们的标签从1开始，YOLO从0开始）
                        class_id = int(label.item()) - 1 if label.item() > 0 else 0
                        
                        # 写入YOLO格式标注
                        f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
        
        # 创建数据集配置文件
        dataset_config = {
            'path': yolo_dir,
            'train': f'images/{self.split}' if self.split == 'train' else None,
            'val': f'images/{self.split}' if self.split == 'val' else None,
            'test': f'images/{self.split}' if self.split == 'test' else None,
            'nc': 1,  # 类别数量（只有病灶一类）
            'names': ['lesion']  # 类别名称
        }
        
        config_path = os.path.join(yolo_dir, 'dataset.yaml')
        with open(config_path, 'w') as f:
            yaml_content = f"""path: {yolo_dir}
train: images/{self.split if self.split == 'train' else 'train'}
val: images/{self.split if self.split == 'val' else 'val'}

nc: 1
names: ['lesion']
"""
            f.write(yaml_content)
        
        logging.info(f"YOLOv11 dataset configuration saved to: {config_path}")
        return config_path


def setup_logging(log_dir: str) -> str:
    """设置日志记录"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'yolov11_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # 清除已有的处理器
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 创建格式器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                 datefmt='%Y-%m-%d %H:%M:%S')
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # 配置根日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return log_file


def calculate_yolo_metrics(results, confidence_threshold: float = 0.5) -> Dict[str, float]:
    """
    计算YOLOv11训练结果的指标
    
    Args:
        results: YOLOv11训练结果对象
        confidence_threshold: 置信度阈值
        
    Returns:
        Dict[str, float]: 计算得到的指标字典
    """
    if results is None or not hasattr(results, 'results'):
        return {}
    
    # 从YOLOv11结果中提取指标
    metrics = {}
    
    try:
        # 获取最后一个epoch的指标
        if hasattr(results, 'results') and results.results:
            last_result = results.results[-1]
            
            # 提取关键指标
            if hasattr(last_result, 'box'):
                box_metrics = last_result.box
                metrics.update({
                    'precision': float(getattr(box_metrics, 'p', 0)),
                    'recall': float(getattr(box_metrics, 'r', 0)),
                    'mAP@0.5': float(getattr(box_metrics, 'map50', 0)),
                    'mAP@[0.5:0.95]': float(getattr(box_metrics, 'map', 0)),
                })
                
                # 计算F1分数
                p, r = metrics.get('precision', 0), metrics.get('recall', 0)
                metrics['f1_score'] = 2 * p * r / (p + r) if (p + r) > 0 else 0
        
        # 如果有验证结果，也提取相关指标
        if hasattr(results, 'val') and results.val:
            val_metrics = results.val
            if hasattr(val_metrics, 'box'):
                val_box = val_metrics.box
                metrics.update({
                    'val_precision': float(getattr(val_box, 'p', 0)),
                    'val_recall': float(getattr(val_box, 'r', 0)),
                    'val_mAP@0.5': float(getattr(val_box, 'map50', 0)),
                    'val_mAP@[0.5:0.95]': float(getattr(val_box, 'map', 0)),
                })
    
    except Exception as e:
        logging.warning(f"提取YOLOv11指标时出错: {e}")
    
    return metrics


def evaluate_yolo_model(model_path: str, dataset_config: str, device: str = 'auto') -> Dict[str, Any]:
    """
    评估训练好的YOLOv11模型
    
    Args:
        model_path: 模型路径
        dataset_config: 数据集配置文件路径
        device: 设备类型
        
    Returns:
        Dict[str, Any]: 评估结果
    """
    try:
        # 加载模型
        model = YOLO(model_path)
        
        # 在验证集上评估
        val_results = model.val(data=dataset_config, device=device)
        
        # 提取指标
        metrics = calculate_yolo_metrics(val_results)
        
        return {
            'metrics': metrics,
            'val_results': val_results,
            'model_path': model_path
        }
    
    except Exception as e:
        logging.error(f"评估YOLOv11模型时出错: {e}")
        return {'metrics': {}, 'error': str(e)}


def create_kfold_datasets(data_dir: str, k_folds: int = 5, random_seed: int = 42, 
                         include_negative_samples: bool = True, max_negative_per_patient: int = 0) -> Tuple[List, Dict]:
    """
    创建按病例分割的K-fold数据集
    
    Args:
        data_dir: 数据目录
        k_folds: K折数量
        random_seed: 随机种子
        include_negative_samples: 是否包含负样本
        max_negative_per_patient: 每个病例最大负样本数
        
    Returns:
        Tuple[List, Dict]: (fold数据集列表, 数据集统计信息)
    """
    # 载入完整数据集
    full_dataset = CTDetectionDataset(
        data_root=data_dir,
        split='train',
        target_size=640,
        specific_patients=None,
        transforms=transforms.Compose([transforms.ToTensor()]),
        include_negative_samples=include_negative_samples,
        max_negative_per_patient=max_negative_per_patient
    )
    
    # 收集所有病例ID
    all_patient_ids = set()
    for sample in full_dataset.samples:
        all_patient_ids.add(sample['patient_id'])
    
    all_patient_ids = sorted(list(all_patient_ids))
    total_patients = len(all_patient_ids)
    
    logging.info(f"数据集总大小: {len(full_dataset)} 张图像")
    logging.info(f"总病例数: {total_patients} 位")
    
    # 设置随机种子并创建K-fold分割
    np.random.seed(random_seed)
    np.random.shuffle(all_patient_ids)
    
    # 使用KFold按病例分割
    kfold = KFold(n_splits=k_folds, shuffle=False, random_state=None)
    patient_indices = list(range(total_patients))
    
    fold_datasets = []
    all_fold_stats = []
    
    # 计算完整数据集的统计信息
    logging.info("计算完整数据集统计信息...")
    full_dataset_stats = calculate_dataset_statistics(full_dataset, "完整数据集")
    
    for fold, (train_patient_idx, val_patient_idx) in enumerate(kfold.split(patient_indices)):
        logging.info(f"\n=== 准备 Fold {fold + 1}/{k_folds} ===")
        
        # 获取训练和验证病例ID
        train_patient_ids = [all_patient_ids[i] for i in train_patient_idx]
        val_patient_ids = [all_patient_ids[i] for i in val_patient_idx]
        
        logging.info(f"训练病例数: {len(train_patient_ids)}")
        logging.info(f"验证病例数: {len(val_patient_ids)}")
        
        # 创建YOLOv11数据集适配器
        train_yolo_dataset = YOLOv11CTDataset(
            data_dir=data_dir,
            split='train',
            include_negative_samples=include_negative_samples,
            max_negative_per_patient=max_negative_per_patient,
            patient_ids=train_patient_ids
        )
        
        val_yolo_dataset = YOLOv11CTDataset(
            data_dir=data_dir,
            split='val',
            include_negative_samples=include_negative_samples,
            max_negative_per_patient=max_negative_per_patient,
            patient_ids=val_patient_ids
        )
        
        # 计算训练和验证集统计信息
        train_stats = calculate_dataset_statistics(train_yolo_dataset.rcnn_dataset, f"Fold {fold + 1} 训练集")
        val_stats = calculate_dataset_statistics(val_yolo_dataset.rcnn_dataset, f"Fold {fold + 1} 验证集")
        
        fold_datasets.append((train_yolo_dataset, val_yolo_dataset))
        all_fold_stats.append({
            'fold': fold + 1,
            'train_patients': train_patient_ids,
            'val_patients': val_patient_ids,
            'train_stats': train_stats,
            'val_stats': val_stats
        })
    
    # 组合所有统计信息
    dataset_statistics = {
        'full_dataset_stats': full_dataset_stats,
        'k_folds': k_folds,
        'fold_statistics': all_fold_stats,
        'total_dataset_size': len(full_dataset),
        'total_patient_count': total_patients,
        'all_patient_ids': all_patient_ids
    }
    
    return fold_datasets, dataset_statistics


def calculate_dataset_statistics(dataset, dataset_name: str = "Dataset") -> Dict[str, Any]:
    """计算数据集的详细统计信息"""
    total_images = len(dataset)
    total_annotations = 0
    images_with_annotations = 0
    images_without_annotations = 0
    
    logging.info(f"正在计算 {dataset_name} 统计信息...")
    
    for i in range(total_images):
        try:
            item = dataset[i]
            target = item['target']
            
            if 'boxes' in target and len(target['boxes']) > 0:
                num_boxes = len(target['boxes'])
                total_annotations += num_boxes
                images_with_annotations += 1
            else:
                images_without_annotations += 1
        except Exception as e:
            logging.warning(f"处理第{i}个样本时出错: {e}")
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
    
    # 记录统计信息
    logging.info(f"=== {dataset_name} 统计信息 ===")
    logging.info(f"总影像数: {total_images}")
    logging.info(f"总标记数: {total_annotations}")
    logging.info(f"有标记的影像数: {images_with_annotations}")
    logging.info(f"无标记的影像数: {images_without_annotations}")
    logging.info(f"平均每张影像标记数: {stats['avg_annotations_per_image']:.2f}")
    if images_with_annotations > 0:
        logging.info(f"平均每张有标记影像的标记数: {stats['avg_annotations_per_annotated_image']:.2f}")
    
    return stats


def train_yolov11_kfold(data_dir: str, k_folds: int = 5, num_epochs: int = 100, 
                        batch_size: int = 16, learning_rate: float = 0.01,
                        save_dir: str = './yolov11_models', log_dir: str = './yolov11_logs',
                        random_seed: int = 42, include_negative_samples: bool = True,
                        max_negative_per_patient: int = 0, imgsz: int = 640,
                        model_size: str = 'n') -> Dict[str, Any]:
    """
    YOLOv11 K-fold交叉验证训练
    
    Args:
        data_dir: 数据目录
        k_folds: K折数量
        num_epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        save_dir: 模型保存目录
        log_dir: 日志目录
        random_seed: 随机种子
        include_negative_samples: 是否包含负样本
        max_negative_per_patient: 每个病例最大负样本数
        imgsz: 图像尺寸
        model_size: 模型大小 ('n', 's', 'm', 'l', 'x')
        
    Returns:
        Dict[str, Any]: 训练结果
    """
    # 设置日志
    log_file = setup_logging(log_dir)
    logging.info("开始YOLOv11 K-fold交叉验证训练")
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"使用设备: {device}")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建K-fold数据集
    fold_datasets, dataset_statistics = create_kfold_datasets(
        data_dir=data_dir,
        k_folds=k_folds,
        random_seed=random_seed,
        include_negative_samples=include_negative_samples,
        max_negative_per_patient=max_negative_per_patient
    )
    
    # 存储所有fold的结果
    all_fold_results = []
    
    # 创建总体进度条
    fold_pbar = tqdm(
        fold_datasets,
        desc="YOLOv11 K-Fold 交叉验证进度",
        unit="fold",
        ncols=120
    )
    
    for fold, (train_dataset, val_dataset) in enumerate(fold_pbar):
        fold_start_time = time.time()
        
        logging.info(f"\n{'='*60}")
        logging.info(f"开始训练 Fold {fold + 1}/{k_folds}")
        logging.info(f"{'='*60}")
        
        # 为当前fold创建数据集配置
        fold_save_dir = os.path.join(save_dir, f'fold_{fold + 1}')
        os.makedirs(fold_save_dir, exist_ok=True)
        
        # 准备YOLOv11格式数据
        train_config_path = train_dataset.prepare_yolo_format(fold_save_dir)
        val_config_path = val_dataset.prepare_yolo_format(fold_save_dir)
        
        # 合并训练和验证数据集配置
        combined_config_path = os.path.join(fold_save_dir, 'combined_dataset.yaml')
        with open(combined_config_path, 'w') as f:
            yaml_content = f"""path: {fold_save_dir}
train: yolo_dataset_train/images/train
val: yolo_dataset_val/images/val

nc: 1
names: ['lesion']
"""
            f.write(yaml_content)
        
        try:
            # 初始化YOLOv11模型
            model_name = f'yolov11{model_size}.pt'
            model = YOLO(model_name)
            
            logging.info(f"使用模型: {model_name}")
            logging.info(f"训练配置文件: {combined_config_path}")
            
            # 训练参数
            train_args = {
                'data': combined_config_path,
                'epochs': num_epochs,
                'batch': batch_size,
                'lr0': learning_rate,
                'imgsz': imgsz,
                'device': device,
                'project': fold_save_dir,
                'name': f'fold_{fold + 1}_training',
                'save': True,
                'save_period': 10,  # 每10个epoch保存一次
                'val': True,
                'plots': True,
                'verbose': True,
                'patience': 50,  # 早停耐心值
                'workers': 4,  # 数据加载进程数
                'seed': random_seed,
            }
            
            # 开始训练
            logging.info(f"开始训练 Fold {fold + 1}")
            results = model.train(**train_args)
            
            # 计算训练时间
            fold_time = time.time() - fold_start_time
            
            # 评估模型
            best_model_path = os.path.join(fold_save_dir, f'fold_{fold + 1}_training', 'weights', 'best.pt')
            eval_results = evaluate_yolo_model(best_model_path, combined_config_path, device)
            
            # 保存fold结果
            fold_result = {
                'fold': fold + 1,
                'training_time': fold_time,
                'model_path': best_model_path,
                'config_path': combined_config_path,
                'metrics': eval_results.get('metrics', {}),
                'train_args': train_args,
                'results': results
            }
            
            all_fold_results.append(fold_result)
            
            # 更新进度条描述
            metrics = eval_results.get('metrics', {})
            f1 = metrics.get('f1_score', 0)
            map50 = metrics.get('mAP@0.5', 0)
            fold_pbar.set_postfix({
                'F1': f'{f1:.3f}', 
                'mAP@0.5': f'{map50:.3f}',
                'Time': f'{fold_time/3600:.1f}h'
            })
            
            logging.info(f"Fold {fold + 1} 完成 - F1: {f1:.3f}, mAP@0.5: {map50:.3f}, 用时: {fold_time/3600:.1f}小时")
            
        except Exception as e:
            logging.error(f"Fold {fold + 1} 训练失败: {e}")
            fold_result = {
                'fold': fold + 1,
                'training_time': time.time() - fold_start_time,
                'error': str(e),
                'metrics': {}
            }
            all_fold_results.append(fold_result)
            continue
    
    fold_pbar.close()
    
    # 计算平均结果
    valid_results = [r for r in all_fold_results if 'error' not in r]
    
    if valid_results:
        avg_metrics = {}
        for metric in ['precision', 'recall', 'f1_score', 'mAP@0.5', 'mAP@[0.5:0.95]']:
            values = [r['metrics'].get(metric, 0) for r in valid_results]
            avg_metrics[metric] = np.mean(values) if values else 0
            avg_metrics[f'{metric}_std'] = np.std(values) if values else 0
    else:
        avg_metrics = {}
    
    total_time = sum(result['training_time'] for result in all_fold_results)
    
    # 保存结果
    results_summary = {
        'average_metrics': avg_metrics,
        'total_training_time': total_time,
        'successful_folds': len(valid_results),
        'total_folds': k_folds,
        'fold_results': all_fold_results,
        'config': {
            'k_folds': k_folds,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'imgsz': imgsz,
            'model_size': model_size,
            'include_negative_samples': include_negative_samples,
            'max_negative_per_patient': max_negative_per_patient
        },
        'dataset_statistics': dataset_statistics
    }
    
    # 保存结果到JSON文件
    results_file = os.path.join(save_dir, 'yolov11_kfold_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False, default=str)
    
    # 输出最终结果摘要
    logging.info(f"\n{'='*60}")
    logging.info("YOLOv11 K-Fold 交叉验证完成")
    logging.info(f"{'='*60}")
    logging.info(f"成功完成的fold数: {len(valid_results)}/{k_folds}")
    logging.info(f"总训练时间: {total_time/3600:.1f} 小时")
    
    if avg_metrics:
        logging.info(f"平均指标:")
        for metric, value in avg_metrics.items():
            if not metric.endswith('_std'):
                std_value = avg_metrics.get(f'{metric}_std', 0)
                logging.info(f"  {metric}: {value:.3f} ± {std_value:.3f}")
    
    logging.info(f"结果已保存到: {results_file}")
    
    return results_summary


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv11 K-fold Cross-Validation Training for CT Detection')
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录路径')
    parser.add_argument('--k_folds', type=int, default=5, help='K折数量')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--save_dir', type=str, default='./yolov11_models', help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='./yolov11_logs', help='日志目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--include_negative', action='store_true', help='是否包含负样本')
    parser.add_argument('--max_negative', type=int, default=0, help='每个病例最大负样本数')
    parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸')
    parser.add_argument('--model_size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'], help='模型大小')
    
    args = parser.parse_args()
    
    # 检查ultralytics是否可用
    if not ULTRALYTICS_AVAILABLE:
        print("错误: 请安装ultralytics库")
        print("安装命令: pip install ultralytics")
        return
    
    # 开始训练
    results = train_yolov11_kfold(
        data_dir=args.data_dir,
        k_folds=args.k_folds,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        random_seed=args.seed,
        include_negative_samples=args.include_negative,
        max_negative_per_patient=args.max_negative,
        imgsz=args.imgsz,
        model_size=args.model_size
    )
    
    print(f"\n训练完成! 结果保存在: {args.save_dir}")


if __name__ == "__main__":
    main()
