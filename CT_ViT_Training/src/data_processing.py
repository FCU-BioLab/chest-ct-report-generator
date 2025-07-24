#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CT-ViT 資料處理模組
包含DICOM影像處理和資料集類別

作者: GitHub Copilot
日期: 2025-07-22
"""

import os
from typing import Dict, List, Tuple, Any

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import pydicom
from transformers import ViTImageProcessor

from config import CTViTConfig

# === DICOM影像處理工具 ===
class DICOMProcessor:
    """DICOM影像處理器"""
    
    def __init__(self, config: CTViTConfig):
        self.config = config
        
    def load_dicom_file(self, dicom_path: str) -> np.ndarray:
        """載入DICOM文件並轉換為像素陣列"""
        try:
            ds = pydicom.dcmread(dicom_path)
            pixel_array = ds.pixel_array.astype(np.float32)
            
            # 應用救援斜率和截距
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
            
            return pixel_array
        except Exception as e:
            raise RuntimeError(f"無法載入DICOM文件 {dicom_path}: {str(e)}")
    
    def apply_window_level(self, pixel_array: np.ndarray) -> np.ndarray:
        """應用窗口調整（肺窗）"""
        window_min = self.config.window_center - self.config.window_width // 2
        window_max = self.config.window_center + self.config.window_width // 2
        
        # 限制像素值範圍
        windowed = np.clip(pixel_array, window_min, window_max)
        
        # 歸一化到 0-255
        windowed = ((windowed - window_min) / (window_max - window_min) * 255).astype(np.uint8)
        
        return windowed
    
    def normalize_hounsfield(self, pixel_array: np.ndarray) -> np.ndarray:
        """Hounsfield單位歸一化"""
        # 限制HU值範圍 (-1024 到 3071)
        pixel_array = np.clip(pixel_array, -1024, 3071)
        
        # 歸一化到 0-1
        normalized = (pixel_array + 1024) / (3071 + 1024)
        
        return normalized
    
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """調整影像大小"""
        if len(image.shape) == 2:
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
        else:
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
        
        return image
    
    def preprocess_dicom(self, dicom_path: str) -> np.ndarray:
        """完整的DICOM預處理流程"""
        # 載入DICOM
        pixel_array = self.load_dicom_file(dicom_path)
        
        # Hounsfield歸一化或窗口調整
        if self.config.normalize_hounsfield:
            processed = self.normalize_hounsfield(pixel_array)
            # 轉換為0-255範圍用於顯示
            processed = (processed * 255).astype(np.uint8)
        else:
            processed = self.apply_window_level(pixel_array)
        
        # 調整大小
        processed = self.resize_image(processed, (self.config.image_size, self.config.image_size))
        
        # 轉換為3通道（RGB）
        if len(processed.shape) == 2:
            processed = np.stack([processed, processed, processed], axis=-1)
        
        return processed

# === 資料集類 ===
class CTDataset(Dataset):
    """胸部CT資料集"""
    
    def __init__(self, data_dir: str, config: CTViTConfig, split: str = "train"):
        self.data_dir = data_dir
        self.config = config
        self.split = split
        self.dicom_processor = DICOMProcessor(config)
        self.image_processor = ViTImageProcessor.from_pretrained(config.model_name)
        
        # 載入資料列表
        self.data_list = self._load_data_list()
        
        # 類別映射
        self.label_map = {'A': 0, 'B': 1, 'E': 2, 'G': 3}
        self.label_names = ['A_series', 'B_series', 'E_series', 'G_series']
        
        print(f"載入 {split} 資料集: {len(self.data_list)} 個樣本")
    
    def _load_data_list(self) -> List[Dict]:
        """載入資料列表"""
        data_list = []
        
        if not os.path.exists(self.data_dir):
            raise ValueError(f"資料目錄不存在: {self.data_dir}")
        
        for patient_id in os.listdir(self.data_dir):
            patient_dir = os.path.join(self.data_dir, patient_id)
            
            if not os.path.isdir(patient_dir):
                continue
            
            # 檢查是否為有效的病例ID
            if not patient_id.startswith(('A', 'B', 'E', 'G')):
                continue
            
            dicom_dir = os.path.join(patient_dir, 'dicom_files')
            xml_dir = os.path.join(patient_dir, 'xml_annotations')
            
            if not (os.path.exists(dicom_dir) and os.path.exists(xml_dir)):
                continue
            
            # 獲取DICOM文件列表
            dicom_files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
            
            if len(dicom_files) == 0:
                continue
            
            # 根據切片選擇方法處理
            if self.config.slice_selection_method == "middle":
                # 選擇中間切片
                selected_files = [dicom_files[len(dicom_files) // 2]]
            elif self.config.slice_selection_method == "random":
                # 隨機選擇一個切片
                selected_files = [np.random.choice(dicom_files)]
            else:  # "all"
                # 使用所有切片
                selected_files = dicom_files
            
            for dicom_file in selected_files:
                data_list.append({
                    'patient_id': patient_id,
                    'dicom_path': os.path.join(dicom_dir, dicom_file),
                    'xml_dir': xml_dir,
                    'label': patient_id[0]  # A, B, E, G
                })
        
        return data_list
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """獲取單個樣本"""
        item = self.data_list[idx]
        
        try:
            # 載入和預處理DICOM影像
            image = self.dicom_processor.preprocess_dicom(item['dicom_path'])
            
            # 資料增強 (僅訓練時)
            if self.split == "train" and self.config.use_augmentation:
                image = self._apply_augmentation(image)
            
            # 使用ViT圖像處理器
            inputs = self.image_processor(image, return_tensors="pt", do_resize=False)
            pixel_values = inputs['pixel_values'].squeeze(0)  # 移除batch維度
            
            # 標籤
            label = self.label_map[item['label']]
            
            return {
                'pixel_values': pixel_values,
                'labels': torch.tensor(label, dtype=torch.long),
                'patient_id': item['patient_id'],
                'dicom_path': item['dicom_path']
            }
            
        except Exception as e:
            print(f"處理樣本時出錯 {item['dicom_path']}: {str(e)}")
            # 返回第一個樣本作為備選
            return self.__getitem__(0)
    
    def _apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        """應用資料增強"""
        h, w = image.shape[:2]
        
        # 旋轉
        if np.random.random() < 0.5:
            angle = np.random.uniform(-self.config.rotation_range, self.config.rotation_range)
            matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            image = cv2.warpAffine(image, matrix, (w, h))
        
        # 縮放
        if np.random.random() < 0.5:
            scale = np.random.uniform(1.0 - self.config.zoom_range, 1.0 + self.config.zoom_range)
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h))
            
            # 裁剪或填充回原始大小
            if scale > 1.0:  # 裁剪
                start_y = (new_h - h) // 2
                start_x = (new_w - w) // 2
                image = image[start_y:start_y+h, start_x:start_x+w]
            else:  # 填充
                pad_y = (h - new_h) // 2
                pad_x = (w - new_w) // 2
                image = cv2.copyMakeBorder(image, pad_y, h-new_h-pad_y, pad_x, w-new_w-pad_x, cv2.BORDER_CONSTANT, value=0)
        
        # 亮度調整
        if np.random.random() < 0.5:
            brightness = np.random.uniform(1.0 - self.config.brightness_range, 1.0 + self.config.brightness_range)
            image = np.clip(image * brightness, 0, 255).astype(np.uint8)
        
        # 對比度調整
        if np.random.random() < 0.5:
            contrast = np.random.uniform(1.0 - self.config.contrast_range, 1.0 + self.config.contrast_range)
            image = np.clip((image - 127.5) * contrast + 127.5, 0, 255).astype(np.uint8)
        
        return image
