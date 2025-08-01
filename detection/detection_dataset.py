#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CT-ViT 目標檢測資料處理模組
處理XML標註檔案，生成目標檢測訓練資料

主要功能：
1. 解析XML標註檔案（Pascal VOC格式）
2. 提取邊界框和類別資訊
3. 生成目標檢測資料集
4. 資料擴增和預處理

作者: GitHub Copilot
日期: 2025-07-25
"""

import os
import json
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.data import Dataset
import pydicom
from PIL import Image
import cv2
from typing import Dict, List, Tuple, Optional, Any
import logging

class XMLAnnotationParser:
    """XML標註檔案解析器"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        
        # 類別映射 (根據你的資料集調整)
        self.class_mapping = {
            'Adenocarcinoma': 1,
            'A': 1,  # 惡性
            'B': 2,  # 良性
            'E': 3,  # E類
            'G': 4,  # G類
            'background': 0
        }
        
    def parse_xml(self, xml_path: str) -> Dict[str, Any]:
        """解析單個XML標註檔案"""
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
            
            # 解析所有物件
            for obj in root.findall('object'):
                name_elem = obj.find('name')
                bbox_elem = obj.find('bndbox')
                
                if name_elem is not None and bbox_elem is not None:
                    class_name = name_elem.text
                    class_id = self.class_mapping.get(class_name, 0)
                    
                    # 提取邊界框座標
                    xmin = int(bbox_elem.find('xmin').text)
                    ymin = int(bbox_elem.find('ymin').text)
                    xmax = int(bbox_elem.find('xmax').text)
                    ymax = int(bbox_elem.find('ymax').text)
                    
                    # 正規化座標到[0, 1]
                    xmin_norm = xmin / width
                    ymin_norm = ymin / height
                    xmax_norm = xmax / width
                    ymax_norm = ymax / height
                    
                    # 轉換為中心點+寬高格式
                    x_center = (xmin_norm + xmax_norm) / 2
                    y_center = (ymin_norm + ymax_norm) / 2
                    bbox_width = xmax_norm - xmin_norm
                    bbox_height = ymax_norm - ymin_norm
                    
                    annotations.append({
                        'class_name': class_name,
                        'class_id': class_id,
                        'bbox': [x_center, y_center, bbox_width, bbox_height],
                        'bbox_original': [xmin, ymin, xmax, ymax]
                    })
            
            return {
                'image_size': (width, height),
                'annotations': annotations,
                'num_objects': len(annotations)
            }
            
        except Exception as e:
            self.logger.error(f"解析XML檔案失敗 {xml_path}: {e}")
            return {
                'image_size': (512, 512),
                'annotations': [],
                'num_objects': 0
            }

class CTDetectionDataset(Dataset):
    """CT目標檢測資料集"""
    
    def __init__(self, data_root, split='train', transform=None, target_size=224, specific_patients=None):
        """
        Args:
            data_root: 資料根目錄 (matched_data_by_patient)
            split: 資料分割 ('train', 'val', 'test')
            transform: 影像變換
            target_size: 目標影像尺寸
            specific_patients: 指定的患者列表 (用於K-Fold交叉驗證)
        """
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.target_size = target_size
        self.specific_patients = specific_patients
        
        # 初始化解析器
        self.xml_parser = XMLAnnotationParser()
        
        # 載入資料路徑
        self.data_pairs = self._load_data_pairs()
        
        print(f"載入 {split} 資料集: {len(self.data_pairs)} 個樣本")
    
    def _load_data_pairs(self) -> List[Dict]:
        """載入DICOM-XML配對資料"""
        data_pairs = []
        
        # 如果指定了特定患者列表（K-Fold模式），使用該列表
        if self.specific_patients is not None:
            target_patients = self.specific_patients
            print(f"使用指定的患者列表: {len(target_patients)} 個患者")
        else:
            # 使用預分割的資料集 - 從splited_dataset載入
            if self.split == 'train':
                patient_list_file = os.path.join(self.data_root, 'train_patients.txt')
            elif self.split == 'test':
                patient_list_file = os.path.join(self.data_root, 'test_patients.txt')
            else:
                # 如果是val，但沒有指定specific_patients，則使用train的一部分
                patient_list_file = os.path.join(self.data_root, 'train_patients.txt')
            
            if os.path.exists(patient_list_file):
                # 從患者列表文件載入
                with open(patient_list_file, 'r') as f:
                    target_patients = [line.strip() for line in f if line.strip()]
                print(f"從 {patient_list_file} 載入了 {len(target_patients)} 個患者")
            else:
                # 如果沒有分割文件，使用所有患者
                print(f"找不到分割文件 {patient_list_file}，使用所有患者資料")
                # 查找實際的患者資料目錄
                data_dir = os.path.join(self.data_root, self.split)
                if os.path.exists(data_dir):
                    target_patients = [f for f in os.listdir(data_dir) 
                                      if os.path.isdir(os.path.join(data_dir, f))]
                else:
                    print(f"找不到資料目錄: {data_dir}")
                    return []
        
        # 根據split確定實際的資料路徑
        actual_data_dir = os.path.join(self.data_root, 'train')  # 所有患者資料都在train目錄下
        
        # 遍歷目標患者資料夾
        for patient_folder in target_patients:
            patient_path = os.path.join(actual_data_dir, patient_folder)
            
            if not os.path.exists(patient_path):
                print(f"警告: 找不到患者資料夾 {patient_path}")
                continue
            
            # 檢查資料夾結構
            dicom_path = os.path.join(patient_path, 'dicom_files')
            xml_path = os.path.join(patient_path, 'xml_annotations')
            json_file = os.path.join(patient_path, f'{patient_folder}_file_list.json')
            
            if not (os.path.exists(dicom_path) and os.path.exists(xml_path) and os.path.exists(json_file)):
                print(f"警告: 患者 {patient_folder} 缺少必要檔案")
                continue
            
            # 從JSON文件讀取配對關係
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                # 遍歷所有成功複製的文件對
                for file_pair in json_data.get('copied_files', []):
                    # 提取檔案名稱
                    dicom_filename = os.path.basename(file_pair['copied_dcm'])
                    xml_filename = os.path.basename(file_pair['copied_xml'])
                    
                    # 檢查檔案是否存在
                    dicom_full_path = os.path.join(dicom_path, dicom_filename)
                    xml_full_path = os.path.join(xml_path, xml_filename)
                    
                    if os.path.exists(dicom_full_path) and os.path.exists(xml_full_path):
                        data_pairs.append({
                            'patient_id': patient_folder,
                            'dicom_path': dicom_full_path,
                            'xml_path': xml_full_path,
                            'base_name': os.path.splitext(dicom_filename)[0],
                            'uid': file_pair['uid']
                        })
                    else:
                        print(f"警告: 檔案不存在 - DICOM: {dicom_full_path}, XML: {xml_full_path}")
                        
            except Exception as e:
                print(f"讀取JSON文件失敗 {json_file}: {e}")
                continue
        
        return data_pairs
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        data_pair = self.data_pairs[idx]
        
        # 載入DICOM影像
        image = self._load_dicom_image(data_pair['dicom_path'])
        
        # 解析XML標註
        annotation_data = self.xml_parser.parse_xml(data_pair['xml_path'])
        
        # 處理標註資料
        if annotation_data['num_objects'] > 0:
            # 有目標的情況
            first_annotation = annotation_data['annotations'][0]  # 取第一個目標
            label = first_annotation['class_id']
            bbox = torch.tensor(first_annotation['bbox'], dtype=torch.float32)
        else:
            # 沒有目標（背景類別）
            label = 0
            bbox = torch.tensor([0.5, 0.5, 0.0, 0.0], dtype=torch.float32)  # 中心點，無大小
        
        # 影像預處理
        if self.transform:
            image = self.transform(image)
        else:
            image = self._default_transform(image)
        
        return {
            'pixel_values': image,
            'labels': torch.tensor(label, dtype=torch.long),
            'bbox_targets': bbox,
            'patient_id': data_pair['patient_id'],
            'file_name': data_pair['base_name']
        }
    
    def _load_dicom_image(self, dicom_path: str) -> np.ndarray:
        """載入DICOM影像"""
        try:
            dicom_data = pydicom.dcmread(dicom_path)
            image = dicom_data.pixel_array.astype(np.float32)
            
            # 正規化到[0, 255]
            image = np.clip(image, np.percentile(image, 1), np.percentile(image, 99))
            image = (image - image.min()) / (image.max() - image.min()) * 255.0
            
            return image.astype(np.uint8)
            
        except Exception as e:
            print(f"載入DICOM檔案失敗 {dicom_path}: {e}")
            return np.zeros((512, 512), dtype=np.uint8)
    
    def _default_transform(self, image: np.ndarray) -> torch.Tensor:
        """預設影像變換"""
        # 調整大小
        image = cv2.resize(image, (self.target_size, self.target_size))
        
        # 轉換為3通道（為了相容ViT）
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=2)
        
        # 正規化到[0, 1]
        image = image.astype(np.float32) / 255.0
        
        # 轉換為張量 [C, H, W]
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        return image

def create_detection_dataloaders(data_root, batch_size=8, num_workers=4, target_size=224):
    """創建目標檢測資料載入器"""
    
    # 創建資料集
    train_dataset = CTDetectionDataset(
        data_root=data_root,
        split='train',
        target_size=target_size
    )
    
    val_dataset = CTDetectionDataset(
        data_root=data_root,
        split='val',
        target_size=target_size
    )
    
    # 創建資料載入器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=detection_collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=detection_collate_fn
    )
    
    return train_loader, val_loader

def detection_collate_fn(batch):
    """目標檢測批次整理函數"""
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    bbox_targets = torch.stack([item['bbox_targets'] for item in batch])
    
    return {
        'pixel_values': pixel_values,
        'labels': labels,
        'bbox_targets': bbox_targets,
        'patient_ids': [item['patient_id'] for item in batch],
        'file_names': [item['file_name'] for item in batch]
    }

# === 測試和驗證功能 ===
def visualize_detection_sample(data_root, sample_idx=0):
    """視覺化檢測樣本"""
    import matplotlib.pyplot as plt
    
    dataset = CTDetectionDataset(data_root)
    sample = dataset[sample_idx]
    
    # 取得影像和標註
    image = sample['pixel_values'].permute(1, 2, 0).numpy()
    label = sample['labels'].item()
    bbox = sample['bbox_targets'].numpy()
    
    # 顯示影像
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image[:, :, 0], cmap='gray')
    plt.title(f'原始影像 - 患者: {sample["patient_id"]}')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(image[:, :, 0], cmap='gray')
    
    # 繪製邊界框
    if bbox[2] > 0 and bbox[3] > 0:  # 如果有有效的邊界框
        x_center, y_center, width, height = bbox
        x1 = (x_center - width/2) * image.shape[1]
        y1 = (y_center - height/2) * image.shape[0]
        x2 = (x_center + width/2) * image.shape[1]
        y2 = (y_center + height/2) * image.shape[0]
        
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                           fill=False, color='red', linewidth=2)
        plt.gca().add_patch(rect)
    
    class_names = {0: 'Background', 1: 'A/Adenocarcinoma', 2: 'B', 3: 'E', 4: 'G'}
    plt.title(f'標註結果 - 類別: {class_names.get(label, "Unknown")}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 測試資料載入
    data_root = "matched_data_by_patient"
    
    if os.path.exists(data_root):
        # 創建資料集
        dataset = CTDetectionDataset(data_root)
        print(f"資料集大小: {len(dataset)}")
        
        # 測試單個樣本
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"樣本形狀:")
            print(f"  影像: {sample['pixel_values'].shape}")
            print(f"  標籤: {sample['labels']}")
            print(f"  邊界框: {sample['bbox_targets']}")
            print(f"  患者ID: {sample['patient_id']}")
            
            # 視覺化
            # visualize_detection_sample(data_root, 0)
    else:
        print(f"資料目錄不存在: {data_root}")
