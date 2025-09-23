#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用目標檢測資料處理模組
處理XML標註檔案，支援多種檢測模型（YOLO、Faster R-CNN等）

主要功能：
1. 解析XML標註檔案（Pascal VOC格式）
2. 只載入有標註的DICOM檔案（病灶檢測）
3. 支援多種檢測模型格式的資料集
4. 資料擴增和預處理

作者: GitHub Copilot
日期: 2025-09-23
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
        
        # 單一病灶類別映射（只有病灶，無背景）
        self.class_mapping = {
            'A': 1,  # 所有病灶類型都歸類為1
            'B': 1, 
            'E': 1,
            'G': 1,
            'background': 0  # 保留，但實際不會使用
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
                            'bbox': [xmin, ymin, xmax, ymax],  # 使用[x1,y1,x2,y2]格式
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
    CT目標檢測資料集 - 通用版本
    支援載入有標註和無標註的資料（正負樣本）
    支援多種檢測模型格式（YOLO、Faster R-CNN等）
    """
    
    def __init__(self, 
                 data_root: str,
                 split: str = 'train',
                 target_size: int = 512,
                 specific_patients: Optional[List[str]] = None,
                 transforms=None,
                 include_negative_samples: bool = True,
                 max_negative_per_patient: int = 0,
                 format_type: str = 'fasterrcnn'):
        """
        初始化資料集
        
        Args:
            data_root: 資料根目錄
            split: 資料分割 ('train' 或 'test')
            target_size: 目標影像尺寸
            specific_patients: 指定的患者列表（用於K-Fold）
            transforms: 資料擴增轉換
            include_negative_samples: 是否包含負樣本（無標註的影像）
            max_negative_per_patient: 每位患者最大負樣本數量，0表示無限制（載入所有負樣本）
            format_type: 輸出格式類型 ('fasterrcnn', 'yolo', 'general')
        """
        self.data_root = data_root
        self.split = split
        self.target_size = target_size
        self.transforms = transforms
        self.include_negative_samples = include_negative_samples
        self.max_negative_per_patient = max_negative_per_patient
        self.format_type = format_type.lower()
        
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
        
        positive_count = len([s for s in self.samples if s['has_annotation']])
        negative_count = len([s for s in self.samples if not s['has_annotation']])
        
        if self.include_negative_samples:
            if self.max_negative_per_patient == 0:
                self.logger.info(f"載入 {split} 資料集: {len(self.samples)} 個樣本 (正樣本:{positive_count}, 負樣本:{negative_count}, 無限制)")
            else:
                self.logger.info(f"載入 {split} 資料集: {len(self.samples)} 個樣本 (正樣本:{positive_count}, 負樣本:{negative_count}, 每患者最多{self.max_negative_per_patient}個)")
        else:
            self.logger.info(f"載入 {split} 資料集: {len(self.samples)} 個有標註的樣本")
    
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
                if self.split == 'all':
                    # For 'all' split, scan directly in data_root for patient directories
                    if os.path.exists(self.data_root):
                        patients_to_load = [d for d in os.listdir(self.data_root) 
                                          if os.path.isdir(os.path.join(self.data_root, d)) 
                                          and d.startswith(('A', 'B', 'E', 'G'))]  # Patient ID patterns
                    else:
                        patients_to_load = []
                else:
                    # For train/val splits, scan in split subdirectory
                    split_dir = os.path.join(self.data_root, self.split)
                    if os.path.exists(split_dir):
                        patients_to_load = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
                    else:
                        patients_to_load = []
        
        # 檢查是否找到患者
        if len(patients_to_load) == 0:
            print(f"❌ 錯誤：在 {self.data_root} 中找不到任何患者資料")
            print(f"   請檢查路徑是否正確或患者列表檔案是否存在")
            
            # 詳細的路徑診斷
            abs_data_root = os.path.abspath(self.data_root)
            print(f"\n🔍 路徑診斷:")
            print(f"   相對路徑: {self.data_root}")
            print(f"   絕對路徑: {abs_data_root}")
            print(f"   路徑存在: {os.path.exists(abs_data_root)}")
            
            if os.path.exists(abs_data_root):
                print(f"   目錄內容:")
                try:
                    contents = os.listdir(abs_data_root)
                    for item in contents:
                        item_path = os.path.join(abs_data_root, item)
                        item_type = "📁" if os.path.isdir(item_path) else "📄"
                        print(f"     {item_type} {item}")
                except Exception as e:
                    print(f"     無法讀取目錄內容: {e}")
            
            patients_file = os.path.join(self.data_root, f'{self.split}_patients.txt')
            print(f"   患者列表檔案: {patients_file}")
            print(f"   檔案存在: {os.path.exists(patients_file)}")
            
            if self.split == 'all':
                print(f"   使用分割: all (平面目錄結構)")
                print(f"   患者目錄位於: {self.data_root}")
            else:
                split_dir = os.path.join(self.data_root, self.split)
                print(f"   分割目錄: {split_dir}")
                print(f"   目錄存在: {os.path.exists(split_dir)}")
            
            return samples
        
        # 為每個患者載入樣本
        total_patients = len(patients_to_load)
        print(f"開始載入 {total_patients} 位患者的有標註資料...")
        
        total_annotated_files = 0
        failed_patients = []
        patients_without_annotations = []
        pydicom_errors = []
        
        for i, patient_id in enumerate(patients_to_load):
            # 更頻繁的進度顯示
            if i % 10 == 0 or i == total_patients - 1:
                print(f"載入進度: {i+1}/{total_patients} 患者 ({(i+1)/total_patients*100:.1f}%)")
            
            try:
                # Handle different directory structures
                if self.split == 'all':
                    # For 'all' split, patients are directly in data_root (flat structure)
                    patient_dir = os.path.join(self.data_root, patient_id)
                else:
                    # For train/val splits, patients are in subdirectories
                    patient_dir = os.path.join(self.data_root, self.split, patient_id)
                    
                if not os.path.exists(patient_dir):
                    print(f"⚠️  患者目錄不存在: {patient_dir}")
                    failed_patients.append(patient_id)
                    continue
                
                # 檢查DICOM檔案和XML標註的目錄結構
                dicom_dir = os.path.join(patient_dir, 'dicom_files')
                xml_dir = os.path.join(patient_dir, 'xml_annotations')
                
                if os.path.exists(dicom_dir):
                    # 新的目錄結構：dicom_files/ 和 xml_annotations/
                    dicom_files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
                    
                    if len(dicom_files) == 0:
                        print(f"⚠️  患者 {patient_id}: 找不到DICOM檔案")
                        failed_patients.append(patient_id)
                        continue
                    
                    # 載入XML標註檔案，創建UID到XML路徑的映射
                    xml_uid_to_path = {}
                    if os.path.exists(xml_dir):
                        xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
                        if len(xml_files) == 0:
                            print(f"⚠️  患者 {patient_id}: 找不到XML標註檔案")
                            patients_without_annotations.append(patient_id)
                            continue
                        
                        for xml_file in xml_files:
                            # XML檔名就是SOPInstanceUID
                            xml_uid = os.path.splitext(xml_file)[0]
                            xml_uid_to_path[xml_uid] = os.path.join(xml_dir, xml_file)
                    else:
                        print(f"⚠️  患者 {patient_id}: XML標註目錄不存在")
                        patients_without_annotations.append(patient_id)
                        continue
                    
                    # 檢查pydicom可用性
                    if pydicom is None:
                        if patient_id not in pydicom_errors:
                            print(f"❌ 錯誤：pydicom未安裝，無法匹配DICOM檔案和XML標註")
                            print(f"   請執行: pip install pydicom")
                            pydicom_errors.append(patient_id)
                        continue
                    
                    # 處理所有 DICOM 檔案，載入有標註的正樣本和無標註的負樣本
                    file_count_positive = 0
                    file_count_negative = 0
                    negative_candidates = []  # 候選負樣本列表
                    
                    for dcm_file in dicom_files:
                        dcm_path = os.path.join(dicom_dir, dcm_file)
                        
                        try:
                            # 讀取DICOM檔案的SOPInstanceUID（快速模式）
                            ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
                            sop_instance_uid = str(ds.SOPInstanceUID)
                            
                            # 根據UID查找對應的XML標註
                            xml_path = xml_uid_to_path.get(sop_instance_uid)
                            
                            if xml_path is not None:
                                # 有標註的正樣本
                                samples.append({
                                    'patient_id': patient_id,
                                    'dcm_path': dcm_path,
                                    'xml_path': xml_path,
                                    'file_name': dcm_file,
                                    'sop_instance_uid': sop_instance_uid,
                                    'has_annotation': True
                                })
                                file_count_positive += 1
                                total_annotated_files += 1
                            elif self.include_negative_samples:
                                # 無標註的負樣本候選
                                negative_candidates.append({
                                    'patient_id': patient_id,
                                    'dcm_path': dcm_path,
                                    'xml_path': None,
                                    'file_name': dcm_file,
                                    'sop_instance_uid': sop_instance_uid,
                                    'has_annotation': False
                                })
                            
                        except Exception as e:
                            # 靜默處理個別檔案錯誤，避免中斷整個載入過程
                            if i < 5:  # 只顯示前5位患者的錯誤詳情
                                print(f"   ⚠️ 檔案 {dcm_file} 讀取失敗: {str(e)[:50]}...")
                            continue
                    
                    # 隨機選取負樣本（根據限制數量）
                    if self.include_negative_samples and negative_candidates:
                        if self.max_negative_per_patient == 0:
                            # 無限制：載入所有負樣本
                            samples.extend(negative_candidates)
                            file_count_negative = len(negative_candidates)
                        else:
                            # 有限制：隨機選取指定數量的負樣本
                            import random
                            selected_negative = random.sample(
                                negative_candidates, 
                                min(self.max_negative_per_patient, len(negative_candidates))
                            )
                            samples.extend(selected_negative)
                            file_count_negative = len(selected_negative)
                    
                    # 顯示患者檔案統計
                    if i < 10:  # 只顯示前10位患者的詳細信息
                        if file_count_positive > 0 or file_count_negative > 0:
                            status_msg = f"患者 {patient_id}: "
                            if file_count_positive > 0:
                                status_msg += f"{file_count_positive} 個正樣本"
                            if file_count_negative > 0:
                                status_msg += f", {file_count_negative} 個負樣本" if file_count_positive > 0 else f"{file_count_negative} 個負樣本"
                            print(f"  ✅ {status_msg}")
                        else:
                            print(f"  ⚠️  患者 {patient_id}: 沒有找到可用的檔案")
                            patients_without_annotations.append(patient_id)
                            
                else:
                    # 舊的目錄結構：直接在患者目錄下，只保留有XML標註的
                    dcm_files = [f for f in os.listdir(patient_dir) if f.endswith('.dcm')]
                    
                    if len(dcm_files) == 0:
                        print(f"⚠️  患者 {patient_id}: 舊格式目錄中找不到DICOM檔案")
                        failed_patients.append(patient_id)
                        continue
                    
                    file_count = 0
                    for file_name in dcm_files:
                        dcm_path = os.path.join(patient_dir, file_name)
                        xml_path = os.path.join(patient_dir, file_name.replace('.dcm', '.xml'))
                        
                        # 只保留有XML標註的DICOM檔案
                        if os.path.exists(xml_path):
                            samples.append({
                                'patient_id': patient_id,
                                'dcm_path': dcm_path,
                                'xml_path': xml_path,
                                'file_name': file_name,
                                'has_annotation': True
                            })
                            total_annotated_files += 1
                            file_count += 1
                    
                    if i < 10 and file_count > 0:
                        print(f"  ✅ 患者 {patient_id}: {file_count} 個有標註的檔案（舊格式）")
                        
            except Exception as e:
                print(f"  ⚠️  患者 {patient_id} 載入失敗: {e}")
                failed_patients.append(patient_id)
                continue
        
        # 詳細的載入結果報告
        positive_samples = [s for s in samples if s['has_annotation']]
        negative_samples = [s for s in samples if not s['has_annotation']]
        
        print(f"\n{'='*60}")
        print(f"📊 資料載入結果報告")
        print(f"{'='*60}")
        print(f"✅ 總樣本數: {len(samples)}")
        print(f"   - 正樣本（有標註）: {len(positive_samples)}")
        if self.include_negative_samples:
            print(f"   - 負樣本（無標註）: {len(negative_samples)}")
            if len(samples) > 0:
                print(f"   - 負樣本比例: {len(negative_samples)/len(samples)*100:.1f}%")
        print(f"👥 有效患者: {len([p for p in patients_to_load if any(s['patient_id'] == p for s in samples)])} 位")
        print(f"📁 總患者數: {total_patients} 位")
        
        if len(failed_patients) > 0:
            print(f"\n❌ 載入失敗的患者 ({len(failed_patients)} 位):")
            for patient in failed_patients[:10]:  # 只顯示前10個
                print(f"   - {patient}")
            if len(failed_patients) > 10:
                print(f"   ... 還有 {len(failed_patients) - 10} 位")
        
        if len(patients_without_annotations) > 0:
            print(f"\n⚠️ 沒有標註的患者 ({len(patients_without_annotations)} 位):")
            for patient in patients_without_annotations[:10]:
                print(f"   - {patient}")
            if len(patients_without_annotations) > 10:
                print(f"   ... 還有 {len(patients_without_annotations) - 10} 位")
        
        if len(pydicom_errors) > 0:
            print(f"\n🔧 需要安裝pydicom來讀取DICOM檔案:")
            print(f"   pip install pydicom")
        
        if len(samples) == 0:
            print(f"\n❌ 嚴重錯誤：沒有載入任何資料！")
            print(f"可能的原因：")
            print(f"1. pydicom未安裝 - 執行: pip install pydicom")
            print(f"2. 資料路徑錯誤 - 檢查: {self.data_root}")
            print(f"3. XML標註檔案和DICOM檔案不匹配")
            print(f"4. 資料集結構問題")
        
        print(f"{'='*60}")
        
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
    
    def _format_for_yolo(self, boxes, labels, image_width, image_height):
        """轉換為YOLO格式 (center_x, center_y, width, height) 正規化座標"""
        yolo_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2.0 / image_width
            center_y = (y1 + y2) / 2.0 / image_height
            width = (x2 - x1) / image_width
            height = (y2 - y1) / image_height
            yolo_boxes.append([center_x, center_y, width, height])
        
        return {
            'boxes': torch.tensor(yolo_boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64)
        }
    
    def _format_for_fasterrcnn(self, boxes, labels, idx):
        """轉換為Faster R-CNN格式"""
        if len(boxes) > 0:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            # 負樣本：空的tensor
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        
        return {
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'image_id': torch.tensor([idx])
        }
    
    def _format_general(self, boxes, labels):
        """通用格式：簡單的字典格式"""
        return {
            'boxes': boxes,  # 列表格式
            'labels': labels,  # 列表格式
            'num_objects': len(boxes)
        }
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 載入影像
        image = self._load_dicom_image(sample['dcm_path'])
        
        # 準備檢測格式的資料
        boxes = []
        labels = []
        
        # 處理不同維度的圖像
        if len(image.shape) == 2:
            original_height, original_width = image.shape
        else:
            original_height, original_width = image.shape[:2]
        
        # 檢查是否有標註
        if sample['has_annotation'] and sample['xml_path'] is not None:
            # 載入標註（正樣本）
            annotations = self.xml_parser.parse_xml(sample['xml_path'])
            
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
            
            # 如果解析後沒有有效的邊界框，跳過這個樣本（理論上不應該發生）
            if len(boxes) == 0:
                print(f"警告：患者 {sample['patient_id']} 的 {sample['file_name']} 沒有有效的標註")
                boxes = [[0, 0, 10, 10]]  # 建立最小的假邊界框
                labels = [1]  # 病灶類別
        else:
            # 負樣本：沒有標註
            pass  # boxes 和 labels 保持為空列表
        
        # 影像預處理：調整尺寸到目標大小
        from PIL import Image as PILImage
        
        # 調整影像尺寸
        pil_image = PILImage.fromarray(image)
        pil_image_resized = pil_image.resize((self.target_size, self.target_size), PILImage.Resampling.LANCZOS)
        
        # 應用轉換
        image_tensor = self.image_transform(pil_image_resized)
        
        # 根據格式類型格式化標註
        if self.format_type == 'yolo':
            target = self._format_for_yolo(boxes, labels, self.target_size, self.target_size)
        elif self.format_type == 'fasterrcnn':
            target = self._format_for_fasterrcnn(boxes, labels, idx)
        else:  # general
            target = self._format_general(boxes, labels)
        
        return {
            'image': image_tensor,
            'target': target,
            'patient_id': sample['patient_id'],
            'file_name': sample['file_name']
        }

def collate_fn_fasterrcnn(batch):
    """
    Faster R-CNN 自定義批次整理函數
    """
    images = []
    targets = []
    
    for item in batch:
        images.append(item['image'])
        targets.append(item['target'])
    
    return images, targets

def collate_fn_yolo(batch):
    """
    YOLO 自定義批次整理函數
    """
    images = torch.stack([item['image'] for item in batch])
    targets = [item['target'] for item in batch]
    
    return images, targets

def collate_fn_general(batch):
    """
    通用自定義批次整理函數
    """
    images = torch.stack([item['image'] for item in batch])
    targets = [item['target'] for item in batch]
    patient_ids = [item['patient_id'] for item in batch]
    file_names = [item['file_name'] for item in batch]
    
    return {
        'images': images,
        'targets': targets,
        'patient_ids': patient_ids,
        'file_names': file_names
    }

def create_detection_dataloaders(data_root: str,
                               train_patients: Optional[List[str]] = None,
                               val_patients: Optional[List[str]] = None,
                               batch_size: int = 8,
                               target_size: int = 512,
                               num_workers: int = 4,
                               format_type: str = 'fasterrcnn'):
    """
    創建檢測資料載入器
    
    Args:
        data_root: 資料根目錄
        train_patients: 訓練患者列表
        val_patients: 驗證患者列表
        batch_size: 批次大小
        target_size: 目標影像尺寸
        num_workers: 工作程序數
        format_type: 格式類型 ('fasterrcnn', 'yolo', 'general')
        
    Returns:
        train_loader, val_loader
    """
    
    # 選擇對應的collate_fn
    if format_type == 'yolo':
        collate_fn = collate_fn_yolo
    elif format_type == 'fasterrcnn':
        collate_fn = collate_fn_fasterrcnn
    else:
        collate_fn = collate_fn_general
    
    # 創建訓練資料集
    train_dataset = CTDetectionDataset(
        data_root=data_root,
        split='train',
        target_size=target_size,
        specific_patients=train_patients,
        format_type=format_type
    )
    
    # 創建驗證資料集
    val_dataset = CTDetectionDataset(
        data_root=data_root,
        split='train',  # 從train中選取驗證資料
        target_size=target_size,
        specific_patients=val_patients,
        format_type=format_type
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
    print("測試通用檢測資料集...")
    # 從detection目錄執行時，需要回到上一層目錄然後到datasets
    data_root = "../datasets/splited_dataset"
    
    print(f"🔍 當前工作目錄: {os.getcwd()}")
    print(f"📁 嘗試資料路徑: {data_root}")
    print(f"📍 絕對路徑: {os.path.abspath(data_root)}")
    print(f"✅ 路徑存在: {os.path.exists(data_root)}")
    print()
    
    # 測試不同格式
    for format_type in ['fasterrcnn', 'yolo', 'general']:
        print(f"\n測試 {format_type.upper()} 格式:")
        try:
            dataset = CTDetectionDataset(
                data_root=data_root, 
                split='train', 
                target_size=512,
                format_type=format_type
            )
            print(f"✅ {format_type} 資料集大小: {len(dataset)}")
            
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"✅ 樣本載入成功 - 患者: {sample['patient_id']}")
                print(f"   影像: {sample['image'].shape}")
                print(f"   目標格式: {type(sample['target'])}")
                if format_type == 'fasterrcnn':
                    print(f"   病灶標籤: {len(sample['target']['labels'])} 個")
                elif format_type == 'yolo':
                    print(f"   YOLO格式: {sample['target']['boxes'].shape if len(sample['target']['boxes']) > 0 else 'empty'}")
                else:
                    print(f"   通用格式: {sample['target']['num_objects']} 個物件")
        except Exception as e:
            print(f"❌ {format_type} 測試失敗: {e}")