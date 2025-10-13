#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
預處理資料集加載器 - 直接讀取 PNG + YOLO 標註

這個 Dataset 類用於載入已經預處理完成的資料（PNG 圖像 + YOLO 標註）
跳過即時的 DICOM 讀取和預處理，大幅提升訓練速度

使用場景：
- 已經執行 preprocess_for_yolo.py 預處理完成
- 資料結構支援兩種：
  1. 扁平結構：split/images/*.png + split/labels/*.txt
  2. 患者分組：split/patient_id/images/*.png + split/patient_id/labels/*.txt
- 標註格式：YOLO (class x_center y_center width height)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging
import cv2

LOGGER = logging.getLogger(__name__)


class PreprocessedYOLODataset(Dataset):
    """
    預處理 YOLO 資料集加載器
    
    直接讀取預處理後的 PNG 圖像和 YOLO 格式標註，
    避免重複的 DICOM 讀取和預處理操作
    
    資料結構：
        data_root/
        ├── images/
        │   ├── 000000.png
        │   ├── 000001.png
        │   └── ...
        └── labels/
            ├── 000000.txt
            ├── 000001.txt
            └── ...
    
    標註格式（YOLO）：
        class_id x_center y_center width height
        (所有值歸一化到 [0, 1])
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        img_size: int = 640,
        augment: bool = False,
    ):
        """
        Args:
            data_root: 預處理資料根目錄（包含 train/val/test 子目錄）
            split: 資料分割 ('train', 'val', 'test')
            img_size: 目標圖像尺寸
            augment: 是否啟用資料增強（將傳遞給後續的 DataLoader）
        """
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.augment = augment
        
        # 確定分割目錄
        self.split_dir = self.data_root / split
        
        # 驗證目錄存在
        if not self.split_dir.exists():
            raise ValueError(f"分割目錄不存在: {self.split_dir}")
        
        # 檢測資料結構類型
        self._detect_data_structure()
        
        # 載入所有圖像檔案路徑
        self._load_image_files()
        
        if len(self.image_files) == 0:
            raise ValueError(f"在 {self.split_dir} 中找不到圖像檔案")
        
        # 載入元數據（如果存在）
        self.metadata = None
        metadata_file = self.split_dir / "metadata.json"
        if metadata_file.exists():
            import json
            with open(metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        
        # 建立正負樣本索引（用於 oversampling）
        self._build_sample_indices()
        
        LOGGER.info(f"PreprocessedYOLODataset 初始化完成:")
        LOGGER.info(f"  分割: {split}")
        LOGGER.info(f"  資料結構: {self.data_structure}")
        LOGGER.info(f"  樣本數: {len(self.image_files)}")
        LOGGER.info(f"  正樣本: {len(self.positive_indices)}")
        LOGGER.info(f"  負樣本: {len(self.negative_indices)}")
        LOGGER.info(f"  圖像尺寸: {img_size}x{img_size}")
    
    def _detect_data_structure(self):
        """檢測資料結構類型：扁平結構 vs 患者分組"""
        # 檢查是否有 images/ 和 labels/ 目錄（扁平結構）
        images_dir = self.split_dir / "images"
        labels_dir = self.split_dir / "labels"
        
        if images_dir.exists() and labels_dir.exists():
            self.data_structure = "flat"
            self.images_dir = images_dir
            self.labels_dir = labels_dir
            LOGGER.info("檢測到扁平資料結構")
        else:
            # 檢查是否有患者目錄（患者分組結構）
            patient_dirs = [d for d in self.split_dir.iterdir() 
                          if d.is_dir() and (d / "images").exists() and (d / "labels").exists()]
            
            if patient_dirs:
                self.data_structure = "grouped"
                self.patient_dirs = sorted(patient_dirs)
                LOGGER.info(f"檢測到患者分組結構，共 {len(self.patient_dirs)} 位患者")
            else:
                raise ValueError(
                    f"無法識別資料結構。請確保資料目錄符合以下格式之一：\n"
                    f"1. 扁平結構: {self.split_dir}/images/ 和 {self.split_dir}/labels/\n"
                    f"2. 患者分組: {self.split_dir}/patient_id/images/ 和 {self.split_dir}/patient_id/labels/"
                )
    
    def _load_image_files(self):
        """根據資料結構載入所有圖像檔案"""
        self.image_files = []
        self.label_files = []
        
        if self.data_structure == "flat":
            # 扁平結構：直接從 images/ 目錄載入
            self.image_files = sorted(list(self.images_dir.glob("*.png")))
            self.label_files = [self.labels_dir / (img.stem + ".txt") for img in self.image_files]
        
        elif self.data_structure == "grouped":
            # 患者分組結構：遍歷所有患者目錄
            for patient_dir in self.patient_dirs:
                patient_images_dir = patient_dir / "images"
                patient_labels_dir = patient_dir / "labels"
                
                patient_images = sorted(list(patient_images_dir.glob("*.png")))
                for img_path in patient_images:
                    self.image_files.append(img_path)
                    self.label_files.append(patient_labels_dir / (img_path.stem + ".txt"))
    
    def _build_sample_indices(self):
        """建立正負樣本索引"""
        self.positive_indices = []
        self.negative_indices = []
        self._samples_cache = []  # 使用私有變量緩存 samples
        
        for idx in range(len(self.image_files)):
            # 對應的標註檔案
            label_file = self.label_files[idx]
            
            has_annotation = label_file.exists() and label_file.stat().st_size > 0
            
            if has_annotation:
                # 有標註 = 正樣本
                self.positive_indices.append(idx)
            else:
                # 無標註或空檔案 = 負樣本
                self.negative_indices.append(idx)
            
            # 添加到 samples 緩存（格式與 YOLOv7MedicalDataset 期望的一致）
            self._samples_cache.append({
                'has_annotation': has_annotation,
                'image_path': str(self.image_files[idx]),
                'label_path': str(label_file) if label_file.exists() else None,
                'index': idx,
            })
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        獲取單個樣本
        
        Returns:
            image: (3, H, W) 圖像張量，歸一化到 [0, 1]
            labels: (N, 5) 標註張量 [class_id, x_center, y_center, width, height]
            metadata: 元資料字典
        """
        try:
            # 讀取圖像
            img_path = self.image_files[idx]
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                LOGGER.error(f"無法讀取圖像: {img_path}")
                # Return a black image as fallback
                img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            
            # 調整尺寸（如果需要）
            if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
                img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            
            # 轉換為 RGB（3 通道）
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            # 轉換為張量 (C, H, W) 並歸一化到 [0, 1]
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            
            # 讀取標註
            label_path = self.label_files[idx]
            labels = []
            
            if label_path.exists() and label_path.stat().st_size > 0:
                try:
                    with open(label_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                # YOLO 格式: class_id x_center y_center width height
                                parts = line.split()
                                if len(parts) == 5:
                                    class_id = int(parts[0])
                                    x_center = float(parts[1])
                                    y_center = float(parts[2])
                                    width = float(parts[3])
                                    height = float(parts[4])
                                    labels.append([class_id, x_center, y_center, width, height])
                except Exception as e:
                    LOGGER.warning(f"讀取標註失敗 {label_path}: {e}")
            
            # 轉換為張量
            if len(labels) > 0:
                labels_tensor = torch.tensor(labels, dtype=torch.float32)
            else:
                labels_tensor = torch.zeros((0, 5), dtype=torch.float32)
            
            # 元資料
            metadata = {
                "index": idx,
                "image_path": str(img_path),
                "label_path": str(label_path) if label_path.exists() else "",
                "has_annotation": len(labels) > 0,
                "num_boxes": len(labels),
            }
            
            return img_tensor, labels_tensor, metadata
            
        except Exception as e:
            # Catch-all error handling to prevent worker crashes
            LOGGER.error(f"Error loading sample {idx}: {e}")
            # Return empty/black sample
            img_tensor = torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32)
            labels_tensor = torch.zeros((0, 5), dtype=torch.float32)
            metadata = {
                "index": idx,
                "image_path": str(self.image_files[idx]) if idx < len(self.image_files) else "",
                "label_path": "",
                "has_annotation": False,
                "num_boxes": 0,
                "error": str(e),
            }
            return img_tensor, labels_tensor, metadata
    
    @property
    def samples(self):
        """
        提供與 CTDetectionDataset 兼容的 samples 屬性
        用於 yolov7_dataset.py 中的快速索引建立
        
        使用緩存版本以提高性能
        """
        # 如果已經有緩存，直接返回
        if hasattr(self, '_samples_cache'):
            return self._samples_cache
        
        # 否則動態生成（向後兼容）
        samples = []
        for idx in range(len(self)):
            label_file = self.label_files[idx]
            has_annotation = label_file.exists() and label_file.stat().st_size > 0
            samples.append({
                'has_annotation': has_annotation,
                'index': idx,
            })
        return samples


def collate_fn(batch):
    """
    自訂 collate 函數，處理可變長度的標註
    
    Args:
        batch: List of (image, labels, metadata) tuples
    
    Returns:
        images: (B, 3, H, W)
        labels: (N, 6) - 添加 batch_idx
        metadata: List of dicts
    """
    images, labels, metadata = zip(*batch)
    
    # 堆疊圖像
    images = torch.stack(images, 0)
    
    # 為標註添加 batch index
    labels_with_batch = []
    for i, label in enumerate(labels):
        if label.shape[0] > 0:
            batch_idx = torch.full((label.shape[0], 1), i, dtype=label.dtype, device=label.device)
            label_with_idx = torch.cat([batch_idx, label], dim=1)
            labels_with_batch.append(label_with_idx)
    
    # 合併所有標註
    if labels_with_batch:
        labels = torch.cat(labels_with_batch, 0)
    else:
        labels = torch.zeros((0, 6), dtype=torch.float32)
    
    return images, labels, metadata


if __name__ == "__main__":
    # 測試程式碼
    print("PreprocessedYOLODataset - 預處理資料集加載器")
    print("用於載入已預處理的 PNG 圖像 + YOLO 標註")
    
    # 示例用法
    example_code = """
    # 使用範例
    from preprocessed_dataset import PreprocessedYOLODataset, collate_fn
    from torch.utils.data import DataLoader
    
    # 創建資料集
    train_dataset = PreprocessedYOLODataset(
        data_root="../../datasets/preprocessed_yolo",
        split="train",
        img_size=640,
        augment=True
    )
    
    # 創建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # 訓練迴圈
    for images, labels, metadata in train_loader:
        # images: (B, 3, 640, 640)
        # labels: (N, 6) - [batch_idx, class_id, x, y, w, h]
        pass
    """
    print(example_code)
