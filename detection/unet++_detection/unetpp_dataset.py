#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UNet++ Detection Dataset
UNet++ 檢測數據集處理模組

該模組處理醫學影像數據，同時支援分割和檢測任務：
1. 解析 XML 標註文件生成分割遮罩
2. 提取邊界框信息用於檢測
3. 數據增強和預處理
4. 支援多標籤分割和檢測

作者: GitHub Copilot
日期: 2025-09-18
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
try:
    import pydicom
except ImportError:
    pydicom = None
    print("警告: pydicom 未安裝，DICOM 載入功能將不可用")

from PIL import Image, ImageDraw
import cv2
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
import json
from pathlib import Path
import albumentations as A
try:
    from albumentations.pytorch import ToTensorV2
except ImportError:
    print("警告: albumentations 未安裝，將使用基本的 torchvision transforms")
    ToTensorV2 = None


class XMLAnnotationParser:
    """XML標註檔案解析器 - UNet++ 版本"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        
        # 病灶類別映射
        self.class_mapping = {
            'A': 1,  # 結節
            'B': 2,  # 腫塊  
            'E': 3,  # 其他病變
            'G': 4,  # 鈣化
            'background': 0  # 背景
        }
        
        # 反向映射
        self.id_to_class = {v: k for k, v in self.class_mapping.items()}
        
    def parse_xml(self, xml_path: str) -> Dict[str, Any]:
        """解析單個XML標註檔案，生成分割和檢測標註"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # 獲取影像尺寸
            size_elem = root.find('size')
            if size_elem is not None:
                width = int(size_elem.find('width').text)
                height = int(size_elem.find('height').text)
            else:
                width, height = 512, 512  # 預設尺寸
            
            annotations = []
            bboxes = []
            labels = []
            
            # 解析所有物件
            for obj in root.findall('object'):
                name_elem = obj.find('name')
                bbox_elem = obj.find('bndbox')
                
                if name_elem is not None and bbox_elem is not None:
                    class_name = name_elem.text
                    class_id = self.class_mapping.get(class_name, 1)  # 默認為結節
                    
                    # 提取邊界框座標
                    xmin = int(bbox_elem.find('xmin').text)
                    ymin = int(bbox_elem.find('ymin').text)
                    xmax = int(bbox_elem.find('xmax').text)
                    ymax = int(bbox_elem.find('ymax').text)
                    
                    # 確保座標有效
                    xmin = max(0, min(xmin, width - 1))
                    ymin = max(0, min(ymin, height - 1))
                    xmax = max(xmin + 1, min(xmax, width))
                    ymax = max(ymin + 1, min(ymax, height))
                    
                    # 只有當邊界框有效時才添加
                    if xmax > xmin and ymax > ymin:
                        annotations.append({
                            'class_name': class_name,
                            'class_id': class_id,
                            'bbox': [xmin, ymin, xmax, ymax],
                        })
                        
                        bboxes.append([xmin, ymin, xmax, ymax])
                        labels.append(class_id)
            
            return {
                'image_size': (width, height),
                'annotations': annotations,
                'bboxes': bboxes,
                'labels': labels,
                'num_objects': len(annotations)
            }
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"解析XML失敗 {xml_path}: {e}")
            return {
                'image_size': (512, 512),
                'annotations': [],
                'bboxes': [],
                'labels': [],
                'num_objects': 0
            }
    
    def create_segmentation_mask(self, annotations: List[Dict], image_size: Tuple[int, int], 
                               multi_class: bool = True) -> np.ndarray:
        """根據標註創建分割遮罩"""
        width, height = image_size
        
        if multi_class:
            # 多類別分割遮罩 - 使用 int64 以確保與 CrossEntropyLoss 兼容
            mask = np.zeros((height, width), dtype=np.int64)
            
            for ann in annotations:
                bbox = ann['bbox']
                class_id = ann['class_id']
                
                # 確保 class_id 在有效範圍內 (假設最多5個類別: 0背景, 1-4病灶類型)
                if class_id < 0:
                    class_id = 0
                elif class_id > 4:
                    class_id = 1  # 超出範圍的類別統一歸為病灶類型1
                
                xmin, ymin, xmax, ymax = bbox
                # 創建矩形遮罩（可以改為更精確的形狀）
                mask[ymin:ymax, xmin:xmax] = class_id
        else:
            # 二元分割遮罩（病灶 vs 背景）- 使用 int64 以確保與 CrossEntropyLoss 兼容
            mask = np.zeros((height, width), dtype=np.int64)
            
            for ann in annotations:
                bbox = ann['bbox']
                xmin, ymin, xmax, ymax = bbox
                mask[ymin:ymax, xmin:xmax] = 1  # 所有病灶都標記為1
        
        return mask


class UNetPPDetectionDataset(Dataset):
    """
    UNet++ 檢測數據集
    
    同時提供分割和檢測標註，支援端到端訓練
    
    Args:
        data_dir: 數據目錄路徑
        xml_dir: XML標註目錄路徑
        transform: 數據增強變換
        multi_class_segmentation: 是否使用多類別分割
        min_size: 最小影像尺寸
        max_size: 最大影像尺寸
    """
    
    def __init__(self, data_dir: str, xml_dir: str, transform=None, 
                 multi_class_segmentation: bool = False, min_size: int = 512, 
                 max_size: int = 1024, logger=None):
        
        self.data_dir = Path(data_dir)
        self.xml_dir = Path(xml_dir)
        self.transform = transform
        self.multi_class_segmentation = multi_class_segmentation
        self.min_size = min_size
        self.max_size = max_size
        self.logger = logger or logging.getLogger(__name__)
        
        # 初始化解析器
        self.xml_parser = XMLAnnotationParser(logger=self.logger)
        
        # 載入數據列表
        self.data_list = self._load_data_list()
        
        # 預設變換
        if self.transform is None:
            self.transform = self._get_default_transform()
        
        if self.logger:
            self.logger.info(f"載入 {len(self.data_list)} 個樣本")
    
    def _load_data_list(self) -> List[Dict[str, str]]:
        """載入數據列表"""
        data_list = []
        
        # 遍歷XML目錄找到所有標註文件
        xml_files = list(self.xml_dir.glob("*.xml"))
        
        for xml_file in xml_files:
            # 假設DICOM文件與XML文件同名但不同擴展名
            patient_id = xml_file.stem
            
            # 尋找對應的DICOM或影像文件
            possible_extensions = ['.dcm', '.dicom', '.jpg', '.png', '.tiff', '.tif']
            dicom_path = None
            
            for ext in possible_extensions:
                candidate_path = self.data_dir / f"{patient_id}{ext}"
                if candidate_path.exists():
                    dicom_path = candidate_path
                    break
            
            if dicom_path and dicom_path.exists():
                data_list.append({
                    'patient_id': patient_id,
                    'dicom_path': str(dicom_path),
                    'xml_path': str(xml_file)
                })
            else:
                if self.logger:
                    self.logger.warning(f"找不到對應的影像文件: {patient_id}")
        
        return data_list
    
    def _get_default_transform(self):
        """獲取預設的數據變換"""
        return A.Compose([
            A.Resize(height=512, width=512),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.GaussNoise(var_limit=(0.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(1, 3), p=0.3),
            A.Normalize(mean=[0.485], std=[0.229]),  # 單通道歸一化
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    
    def _load_dicom_image(self, dicom_path: str) -> np.ndarray:
        """載入DICOM影像"""
        try:
            if pydicom is not None and dicom_path.endswith(('.dcm', '.dicom')):
                # 載入DICOM文件
                dicom = pydicom.dcmread(dicom_path)
                image = dicom.pixel_array.astype(np.float32)
                
                # 歸一化到0-255範圍
                image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            else:
                # 載入一般影像文件
                image = cv2.imread(dicom_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise ValueError(f"無法載入影像: {dicom_path}")
            
            return image
            
        except Exception as e:
            self.logger.error(f"載入DICOM失敗 {dicom_path}: {e}")
            # 返回空白影像
            return np.zeros((512, 512), dtype=np.uint8)
    
    def _resize_image_and_annotations(self, image: np.ndarray, annotations: Dict, 
                                    target_size: Tuple[int, int]) -> Tuple[np.ndarray, Dict]:
        """調整影像和標註尺寸"""
        orig_h, orig_w = image.shape[:2]
        target_h, target_w = target_size
        
        # 計算縮放比例
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        
        # 調整影像尺寸
        resized_image = cv2.resize(image, (target_w, target_h))
        
        # 調整邊界框
        resized_bboxes = []
        for bbox in annotations['bboxes']:
            xmin, ymin, xmax, ymax = bbox
            new_bbox = [
                int(xmin * scale_x),
                int(ymin * scale_y), 
                int(xmax * scale_x),
                int(ymax * scale_y)
            ]
            resized_bboxes.append(new_bbox)
        
        # 更新標註
        resized_annotations = annotations.copy()
        resized_annotations['bboxes'] = resized_bboxes
        resized_annotations['image_size'] = (target_w, target_h)
        
        return resized_image, resized_annotations
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        if idx >= len(self.data_list):
            raise IndexError(f"索引 {idx} 超出範圍 {len(self.data_list)}")
        
        try:
            data_info = self.data_list[idx]
            patient_id = data_info['patient_id']
            dicom_path = data_info['dicom_path']
            xml_path = data_info['xml_path']
            
            # 載入影像
            image = self._load_dicom_image(dicom_path)
            
            # 解析標註
            annotations = self.xml_parser.parse_xml(xml_path)
            
            # 確保影像為三維 (H, W, C)
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=2)  # 添加通道維度
            
            # 創建分割遮罩
            seg_mask = self.xml_parser.create_segmentation_mask(
                annotations['annotations'], 
                annotations['image_size'],
                multi_class=self.multi_class_segmentation
            )
            
            # 數據增強
            if self.transform and annotations['bboxes']:
                try:
                    # Albumentations 需要歸一化的邊界框座標
                    height, width = image.shape[:2]
                    normalized_bboxes = []
                    for bbox in annotations['bboxes']:
                        xmin, ymin, xmax, ymax = bbox
                        normalized_bboxes.append([xmin, ymin, xmax, ymax])  # Pascal VOC 格式
                    
                    transformed = self.transform(
                        image=image,
                        mask=seg_mask,
                        bboxes=normalized_bboxes,
                        class_labels=annotations['labels']
                    )
                    
                    image = transformed['image']
                    seg_mask = transformed['mask']
                    bboxes = transformed['bboxes']
                    labels = transformed['class_labels']
                    
                except Exception as e:
                    self.logger.warning(f"數據增強失敗 {patient_id}: {e}")
                    # 使用原始數據
                    image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
                    seg_mask = torch.from_numpy(seg_mask).long()
                    bboxes = annotations['bboxes']
                    labels = annotations['labels']
            else:
                # 沒有邊界框或不使用變換
                image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
                seg_mask = torch.from_numpy(seg_mask).long()
                bboxes = annotations['bboxes']
                labels = annotations['labels']
            
            # 構建目標字典
            target = {
                'boxes': torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long),
                'segmentation': seg_mask,
                'image_id': torch.tensor([idx]),
                'area': torch.tensor([(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in bboxes], dtype=torch.float32) if bboxes else torch.zeros((0,), dtype=torch.float32),
                'iscrowd': torch.zeros(len(bboxes), dtype=torch.int64) if bboxes else torch.zeros((0,), dtype=torch.int64),
                'patient_id': patient_id
            }
            
            return {
                'image': image,
                'target': target
            }
            
        except Exception as e:
            self.logger.error(f"載入數據失敗 {idx}: {e}")
            # 返回空白數據
            empty_image = torch.zeros((1, 512, 512), dtype=torch.float32)
            empty_target = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.long),
                'segmentation': torch.zeros((512, 512), dtype=torch.long),
                'image_id': torch.tensor([idx]),
                'area': torch.zeros((0,), dtype=torch.float32),
                'iscrowd': torch.zeros((0,), dtype=torch.int64),
                'patient_id': f'error_{idx}'
            }
            return {
                'image': empty_image,
                'target': empty_target
            }


def collate_fn(batch):
    """
    自定義批次整理函數
    處理 UNet++ 檢測數據的批次合併
    """
    import torch
    import torch.nn.functional as F
    
    images = []
    targets = []
    
    # 目標尺寸和通道數
    target_size = (512, 512)
    target_channels = 1  # 統一使用單通道（灰階）
    
    for item in batch:
        if isinstance(item, dict) and 'image' in item and 'target' in item:
            image = item['image']
            target = item['target']
            
            # 統一通道數 - 轉換為單通道
            if image.shape[0] == 3:  # RGB 轉灰階
                # 使用加權平均轉換為灰階
                image = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
                image = image.unsqueeze(0)  # 添加通道維度
            elif image.shape[0] != target_channels:
                # 如果不是目標通道數，取第一個通道
                image = image[0:target_channels]
            
            # 調整圖像尺寸
            if image.shape[-2:] != target_size:
                image = F.interpolate(
                    image.unsqueeze(0), 
                    size=target_size, 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
                
                # 調整分割遮罩尺寸
                if 'segmentation' in target:
                    seg_mask = target['segmentation']
                    if seg_mask.shape[-2:] != target_size:
                        seg_mask = F.interpolate(
                            seg_mask.unsqueeze(0).float(), 
                            size=target_size, 
                            mode='nearest'
                        ).squeeze(0).long()
                        target['segmentation'] = seg_mask
                
                # 調整檢測框座標
                if 'boxes' in target and len(target['boxes']) > 0:
                    boxes = target['boxes'].clone()
                    original_h, original_w = image.shape[-2:]
                    h_scale = target_size[0] / original_h
                    w_scale = target_size[1] / original_w
                    
                    # 調整邊界框座標
                    boxes[:, [0, 2]] *= w_scale  # x座標
                    boxes[:, [1, 3]] *= h_scale  # y座標
                    target['boxes'] = boxes
            
            images.append(image)
            targets.append(target)
        else:
            print(f"Unexpected batch item format: {type(item)}")
    
    # 將圖像堆疊成張量
    try:
        stacked_images = torch.stack(images)
    except RuntimeError as e:
        # 如果仍然有尺寸問題，打印詳細信息並嘗試修復
        print(f"圖像堆疊失敗: {e}")
        for i, img in enumerate(images):
            print(f"圖像 {i} 尺寸: {img.shape}")
        
        # 嘗試強制統一所有圖像的形狀
        unified_images = []
        for img in images:
            # 確保是單通道
            if img.dim() == 2:
                img = img.unsqueeze(0)
            elif img.shape[0] != target_channels:
                img = img[0:target_channels]
            
            # 確保尺寸正確
            if img.shape[-2:] != target_size:
                img = F.interpolate(
                    img.unsqueeze(0), 
                    size=target_size, 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
            
            unified_images.append(img)
        
        stacked_images = torch.stack(unified_images)
    
    return {
        'images': stacked_images,
        'targets': targets
    }


def create_data_loaders(data_dir: str, xml_dir: str, batch_size: int = 4, 
                       num_workers: int = 4, train_split: float = 0.8,
                       multi_class_segmentation: bool = False) -> Tuple[DataLoader, DataLoader]:
    """
    創建訓練和驗證數據載入器
    
    Args:
        data_dir: 數據目錄
        xml_dir: XML標註目錄  
        batch_size: 批次大小
        num_workers: 工作線程數
        train_split: 訓練集比例
        multi_class_segmentation: 是否使用多類別分割
    
    Returns:
        train_loader, val_loader
    """
    
    # 訓練數據變換
    train_transform = A.Compose([
        A.Resize(height=512, width=512),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(0.0, 50.0)),
            A.GaussianBlur(blur_limit=(1, 3)),
            A.MotionBlur(blur_limit=(3, 7)),
        ], p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.Normalize(mean=[0.485], std=[0.229]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    
    # 驗證數據變換
    val_transform = A.Compose([
        A.Resize(height=512, width=512),
        A.Normalize(mean=[0.485], std=[0.229]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    
    # 創建完整數據集
    full_dataset = UNetPPDetectionDataset(
        data_dir=data_dir,
        xml_dir=xml_dir,
        transform=None,  # 稍後分別設置
        multi_class_segmentation=multi_class_segmentation
    )
    
    # 分割數據集
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split_idx = int(np.floor(train_split * dataset_size))
    
    np.random.shuffle(indices)
    train_indices, val_indices = indices[:split_idx], indices[split_idx:]
    
    # 創建子集
    train_dataset = UNetPPDetectionDataset(
        data_dir=data_dir,
        xml_dir=xml_dir,
        transform=train_transform,
        multi_class_segmentation=multi_class_segmentation
    )
    
    val_dataset = UNetPPDetectionDataset(
        data_dir=data_dir,
        xml_dir=xml_dir,
        transform=val_transform,
        multi_class_segmentation=multi_class_segmentation
    )
    
    # 設置數據集索引
    train_dataset.data_list = [train_dataset.data_list[i] for i in train_indices]
    val_dataset.data_list = [val_dataset.data_list[i] for i in val_indices]
    
    # 創建數據載入器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader


def test_dataset():
    """測試數據集"""
    # 測試參數
    data_dir = r"E:\path\to\dicom\files"
    xml_dir = r"E:\path\to\xml\annotations"
    
    try:
        # 創建數據集
        dataset = UNetPPDetectionDataset(
            data_dir=data_dir,
            xml_dir=xml_dir,
            multi_class_segmentation=True
        )
        
        print(f"數據集大小: {len(dataset)}")
        
        if len(dataset) > 0:
            # 測試單個樣本
            sample = dataset[0]
            image = sample['image']
            target = sample['target']
            
            print(f"影像形狀: {image.shape}")
            print(f"分割遮罩形狀: {target['segmentation'].shape}")
            print(f"邊界框數量: {len(target['boxes'])}")
            print(f"標籤: {target['labels']}")
            
            # 創建數據載入器
            train_loader, val_loader = create_data_loaders(
                data_dir=data_dir,
                xml_dir=xml_dir,
                batch_size=2,
                num_workers=0  # 測試時使用0避免多進程問題
            )
            
            print(f"訓練批次數: {len(train_loader)}")
            print(f"驗證批次數: {len(val_loader)}")
            
            # 測試一個批次
            for batch in train_loader:
                images = batch['images']
                targets = batch['targets']
                print(f"批次影像數: {len(images)}")
                print(f"批次目標數: {len(targets)}")
                break
        
        print("數據集測試完成！")
        
    except Exception as e:
        print(f"測試失敗: {e}")


def collate_fn(batch):
    """
    自定義 collate 函數，處理不同尺寸的圖像
    將所有圖像調整到統一尺寸和通道數
    """
    # 目標尺寸和通道數
    target_size = (512, 512)
    target_channels = 1  # 統一使用單通道（灰階）
    
    images = []
    targets = []
    
    for sample in batch:
        image = sample['image']
        target = sample['target']
        
        # 統一通道數 - 轉換為單通道
        if image.shape[0] == 3:  # RGB 轉灰階
            # 使用加權平均轉換為灰階
            image = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
            image = image.unsqueeze(0)  # 添加通道維度
        elif image.shape[0] != target_channels:
            # 如果不是目標通道數，取第一個通道
            image = image[0:target_channels]
        
        # 調整圖像尺寸
        if image.shape[-2:] != target_size:
            image = F.interpolate(
                image.unsqueeze(0), 
                size=target_size, 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            
            # 調整分割遮罩尺寸
            if 'segmentation' in target:
                seg_mask = target['segmentation']
                if seg_mask.shape[-2:] != target_size:
                    seg_mask = F.interpolate(
                        seg_mask.unsqueeze(0).float(), 
                        size=target_size, 
                        mode='nearest'
                    ).squeeze(0).long()
                    target['segmentation'] = seg_mask
            
            # 調整檢測框座標
            if 'boxes' in target and len(target['boxes']) > 0:
                boxes = target['boxes'].clone()
                original_h, original_w = image.shape[-2:]
                h_scale = target_size[0] / original_h
                w_scale = target_size[1] / original_w
                
                # 調整邊界框座標
                boxes[:, [0, 2]] *= w_scale  # x座標
                boxes[:, [1, 3]] *= h_scale  # y座標
                target['boxes'] = boxes
        
        images.append(image)
        targets.append(target)
    
    # 將圖像堆疊成張量
    try:
        stacked_images = torch.stack(images)
    except RuntimeError as e:
        # 如果仍然有尺寸問題，打印詳細信息並嘗試修復
        print(f"圖像堆疊失敗: {e}")
        for i, img in enumerate(images):
            print(f"圖像 {i} 尺寸: {img.shape}")
        
        # 嘗試強制統一所有圖像的形狀
        unified_images = []
        for img in images:
            # 確保是單通道
            if img.dim() == 2:
                img = img.unsqueeze(0)
            elif img.shape[0] != target_channels:
                img = img[0:target_channels]
            
            # 確保尺寸正確
            if img.shape[-2:] != target_size:
                img = F.interpolate(
                    img.unsqueeze(0), 
                    size=target_size, 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
            
            unified_images.append(img)
        
        stacked_images = torch.stack(unified_images)
    
    return {
        'images': stacked_images,
        'targets': targets
    }


if __name__ == "__main__":
    test_dataset()