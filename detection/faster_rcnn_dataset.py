#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Faster R-CNN 目標檢測資料處理模組
處理XML標註檔案，生成Faster R-CNN訓練資料

主要功能：
1. 解析XML標註檔案（Pascal VOC格式）
2. 二分類：背景(0) vs 病灶(1)
3. 生成Faster R-CNN格式的資料集
4. 資料擴增和預處理

作者: GitHub Copilot
日期: 2025-08-06
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
try:
    import pydicom
except ImportError:
    pydicom = None
    print("警告: pydicom 未安裝，DICOM 載入功能將不可用")

from PIL import Image
from typing import Dict, List, Optional, Any
import logging

class XMLAnnotationParser:
    """XML標註檔案解析器 - 二分類版本"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        
        # 二分類映射：背景(0) vs 病灶(1)
        self.class_mapping = {
            'A': 1,  # 所有病灶類型都歸類為1
            'B': 1, 
            'E': 1,
            'G': 1,
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
                    class_id = self.class_mapping.get(class_name, 1)  # 默認為病灶
                    
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
                            'bbox': [xmin, ymin, xmax, ymax],  # Faster R-CNN使用[x1,y1,x2,y2]格式
                        })
            
            return {
                'image_size': (width, height),
                'annotations': annotations,
                'num_objects': len(annotations)
            }
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"解析XML檔案失敗: {xml_path}, 錯誤: {e}")
            return {
                'image_size': (512, 512),
                'annotations': [],
                'num_objects': 0
            }

class CTDetectionDataset(Dataset):
    """
    CT目標檢測資料集 - Faster R-CNN版本
    二分類：背景 vs 病灶
    """
    
    def __init__(self, 
                 data_root: str,
                 split: str = 'train',
                 target_size: int = 512,
                 specific_patients: Optional[List[str]] = None,
                 transforms=None):
        """
        初始化資料集
        
        Args:
            data_root: 資料根目錄
            split: 資料分割 ('train' 或 'test')
            target_size: 目標影像尺寸
            specific_patients: 指定的患者列表（用於K-Fold）
            transforms: 資料擴增轉換
        """
        self.data_root = data_root
        self.split = split
        self.target_size = target_size
        self.transforms = transforms
        
        # 設置日誌
        self.logger = logging.getLogger(__name__)
        
        # 初始化XML解析器
        self.xml_parser = XMLAnnotationParser(self.logger)
        
        # 載入資料
        self.samples = self._load_samples(specific_patients)
        
        # 定義基本的影像轉換
        if transforms is not None:
            self.image_transform = transforms
        else:
            import torchvision.transforms as T
            self.image_transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485], std=[0.229])  # 灰階影像正規化
            ])
        
        self.logger.info(f"載入 {split} 資料集: {len(self.samples)} 個樣本")
    
    def _load_samples(self, specific_patients: Optional[List[str]] = None) -> List[Dict]:
        """載入樣本資料"""
        samples = []
        
        # 確定要載入的患者
        if specific_patients:
            patients_to_load = specific_patients
        else:
            # 讀取患者列表檔案
            patients_file = os.path.join(self.data_root, f'{self.split}_patients.txt')
            if os.path.exists(patients_file):
                with open(patients_file, 'r') as f:
                    patients_to_load = [line.strip() for line in f.readlines() if line.strip()]
            else:
                # 如果沒有患者列表檔案，掃描目錄
                split_dir = os.path.join(self.data_root, self.split)
                if os.path.exists(split_dir):
                    patients_to_load = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
                else:
                    patients_to_load = []
        
        # 為每個患者載入樣本
        total_patients = len(patients_to_load)
        print(f"開始載入 {total_patients} 位患者的資料...")
        
        for i, patient_id in enumerate(patients_to_load):
            # 更頻繁的進度顯示
            if i % 10 == 0 or i == total_patients - 1:
                print(f"載入進度: {i+1}/{total_patients} 患者 ({(i+1)/total_patients*100:.1f}%)")
            
            try:
                patient_dir = os.path.join(self.data_root, self.split, patient_id)
                if not os.path.exists(patient_dir):
                    continue
                
                # 檢查DICOM檔案和XML標註的目錄結構
                dicom_dir = os.path.join(patient_dir, 'dicom_files')
                xml_dir = os.path.join(patient_dir, 'xml_annotations')
                
                if os.path.exists(dicom_dir):
                    # 新的目錄結構：dicom_files/ 和 xml_annotations/
                    dicom_files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
                    
                    # 載入XML標註檔案，創建UID到XML路徑的映射
                    xml_uid_to_path = {}
                    if os.path.exists(xml_dir):
                        xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
                        for xml_file in xml_files:
                            # XML檔名就是SOPInstanceUID
                            xml_uid = os.path.splitext(xml_file)[0]
                            xml_uid_to_path[xml_uid] = os.path.join(xml_dir, xml_file)
                    
                    # 處理所有 DICOM 檔案（完全移除抽樣限制）
                    file_count = 0
                    for dcm_file in dicom_files:
                        dcm_path = os.path.join(dicom_dir, dcm_file)
                        
                        try:
                            # 讀取DICOM檔案的SOPInstanceUID（快速模式）
                            if pydicom is None:
                                sop_instance_uid = None
                            else:
                                # 只讀取必要的標籤，提高速度
                                ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
                                sop_instance_uid = str(ds.SOPInstanceUID)
                            
                            # 根據UID查找對應的XML標註
                            xml_path = xml_uid_to_path.get(sop_instance_uid) if sop_instance_uid else None
                            
                            samples.append({
                                'patient_id': patient_id,
                                'dcm_path': dcm_path,
                                'xml_path': xml_path,
                                'file_name': dcm_file,
                                'sop_instance_uid': sop_instance_uid,
                                'has_annotation': xml_path is not None
                            })
                            file_count += 1
                            
                        except Exception as e:
                            # 靜默處理個別檔案錯誤，避免中斷整個載入過程
                            samples.append({
                                'patient_id': patient_id,
                                'dcm_path': dcm_path,
                                'xml_path': None,
                                'file_name': dcm_file,
                                'sop_instance_uid': None,
                                'has_annotation': False
                            })
                            file_count += 1
                    
                    # 顯示患者檔案統計
                    if i < 10:  # 只顯示前10位患者的詳細信息
                        print(f"  患者 {patient_id}: {file_count} 個檔案")
                            
                else:
                    # 舊的目錄結構：直接在患者目錄下
                    dcm_files = [f for f in os.listdir(patient_dir) if f.endswith('.dcm')]
                    
                    for file_name in dcm_files:
                        dcm_path = os.path.join(patient_dir, file_name)
                        xml_path = os.path.join(patient_dir, file_name.replace('.dcm', '.xml'))
                        
                        samples.append({
                            'patient_id': patient_id,
                            'dcm_path': dcm_path,
                            'xml_path': xml_path if os.path.exists(xml_path) else None,
                            'file_name': file_name,
                            'has_annotation': os.path.exists(xml_path)
                        })
                        
            except Exception as e:
                print(f"  ⚠️  患者 {patient_id} 載入失敗: {e}")
                continue
        
        return samples
    
    def _load_dicom_image(self, dcm_path: str) -> np.ndarray:
        """載入DICOM影像"""
        try:
            if pydicom is None:
                # 如果沒有 pydicom，創建假的影像
                self.logger.warning(f"pydicom 未安裝，無法載入 DICOM 檔案: {dcm_path}")
                return np.zeros((512, 512), dtype=np.uint8)
                
            ds = pydicom.dcmread(dcm_path)
            image = ds.pixel_array.astype(np.float32)
            
            # 正規化到[0, 255]
            image_min = image.min()
            image_max = image.max()
            if image_max > image_min:
                image = ((image - image_min) / (image_max - image_min) * 255).astype(np.uint8)
            else:
                image = np.zeros_like(image, dtype=np.uint8)
            
            return image
            
        except Exception as e:
            self.logger.error(f"載入DICOM失敗: {dcm_path}, 錯誤: {e}")
            return np.zeros((512, 512), dtype=np.uint8)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 載入影像
        image = self._load_dicom_image(sample['dcm_path'])
        
        # 載入標註（如果有的話）
        if sample['xml_path'] is not None:
            annotations = self.xml_parser.parse_xml(sample['xml_path'])
        else:
            # 沒有標註的情況，創建空標註
            annotations = {
                'image_size': (image.shape[1], image.shape[0]),
                'annotations': [],
                'num_objects': 0
            }
        
        # 準備Faster R-CNN格式的資料
        boxes = []
        labels = []
        
        # 處理不同維度的圖像
        if len(image.shape) == 2:
            original_height, original_width = image.shape
        else:
            original_height, original_width = image.shape[:2]
        
        for ann in annotations['annotations']:
            # 獲取邊界框座標
            x1, y1, x2, y2 = ann['bbox']
            
            # 調整座標到目標尺寸
            x1 = (x1 / original_width) * self.target_size
            y1 = (y1 / original_height) * self.target_size
            x2 = (x2 / original_width) * self.target_size
            y2 = (y2 / original_height) * self.target_size
            
            # 確保邊界框有效
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(self.target_size, x2), min(self.target_size, y2)
            
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(ann['class_id'])
        
        # 如果沒有標註，創建一個假的背景邊界框
        if len(boxes) == 0:
            boxes = [[0, 0, 10, 10]]  # 小的假邊界框
            labels = [0]  # 背景類別
        
        # 轉換為tensor
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        # 影像預處理：調整尺寸到目標大小
        from PIL import Image as PILImage
        
        # 調整影像尺寸
        pil_image = PILImage.fromarray(image)
        pil_image_resized = pil_image.resize((self.target_size, self.target_size), PILImage.Resampling.LANCZOS)
        
        # 應用轉換
        image_tensor = self.image_transform(pil_image_resized)
        
        # 對於Faster R-CNN，我們需要返回字典格式
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }
        
        return {
            'image': image_tensor,
            'target': target,
            'patient_id': sample['patient_id'],
            'file_name': sample['file_name']
        }

def collate_fn(batch):
    """
    自定義批次整理函數
    """
    images = []
    targets = []
    
    for item in batch:
        images.append(item['image'])
        targets.append(item['target'])
    
    return images, targets

def create_detection_dataloaders(data_root: str,
                               train_patients: Optional[List[str]] = None,
                               val_patients: Optional[List[str]] = None,
                               batch_size: int = 8,
                               target_size: int = 512,
                               num_workers: int = 4):
    """
    創建檢測資料載入器
    
    Args:
        data_root: 資料根目錄
        train_patients: 訓練患者列表
        val_patients: 驗證患者列表
        batch_size: 批次大小
        target_size: 目標影像尺寸
        num_workers: 工作程序數
        
    Returns:
        train_loader, val_loader
    """
    
    # 創建訓練資料集
    train_dataset = CTDetectionDataset(
        data_root=data_root,
        split='train',
        target_size=target_size,
        specific_patients=train_patients
    )
    
    # 創建驗證資料集
    val_dataset = CTDetectionDataset(
        data_root=data_root,
        split='train',  # 從train中選取驗證資料
        target_size=target_size,
        specific_patients=val_patients
    )
    
    # 創建資料載入器
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

# 測試程式碼
if __name__ == "__main__":
    print("測試 Faster R-CNN 資料集...")
    data_root = "../../datasets/splited_dataset"
    
    try:
        dataset = CTDetectionDataset(data_root=data_root, split='train', target_size=512)
        print(f"✅ 資料集大小: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"✅ 樣本載入成功 - 患者: {sample['patient_id']}")
            print(f"   影像: {sample['image'].shape}")
            print(f"   標籤: {len(sample['target']['labels'])} 個")
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
