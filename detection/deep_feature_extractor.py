#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Features 提取器 - 用於提取病例的深層特徵供LLM生成報告使用
基於已訓練的Faster R-CNN模型，提取多層次的視覺特徵

主要功能：
1. 提取backbone特徵（ResNet50）
2. 提取RPN特徵
3. 提取ROI特徵
4. 病例級別特徵聚合
5. 儲存特徵向量供LLM使用

作者: GitHub Copilot
日期: 2025-09-03
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import OrderedDict
import logging
from tqdm import tqdm
from datetime import datetime
import pickle

# 添加上級目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faster_rcnn_detection.faster_rcnn_model import FasterRCNN
from faster_rcnn_detection.faster_rcnn_dataset import CTDetectionDataset


class DeepFeatureExtractor(nn.Module):
    """
    深層特徵提取器，基於Faster R-CNN模型
    """
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        super(DeepFeatureExtractor, self).__init__()
        
        self.device = device
        self.model_path = model_path
        
        # 載入已訓練的模型
        self.model = self._load_model()
        self.model.eval()
        
        # 註冊hook來提取中間特徵
        self.features = {}
        self._register_hooks()
        
    def _load_model(self) -> nn.Module:
        """載入已訓練的Faster R-CNN模型"""
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        
        # 創建模型架構
        model = fasterrcnn_resnet50_fpn(weights=None)
        num_classes = 2  # 背景 + 病灶
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # 載入權重
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            logging.info(f"模型載入成功: {self.model_path}")
        else:
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        model.to(self.device)
        return model
    
    def _register_hooks(self):
        """註冊hook來提取中間層特徵"""
        
        def get_activation(name):
            def hook(model, input, output):
                # 處理不同類型的輸出
                if isinstance(output, torch.Tensor):
                    self.features[name] = output.detach()
                elif isinstance(output, (dict, OrderedDict)):
                    # 如果是字典類型（如FPN輸出），保存整個字典
                    detached_dict = {}
                    for k, v in output.items():
                        if isinstance(v, torch.Tensor):
                            detached_dict[k] = v.detach()
                        else:
                            detached_dict[k] = v
                    self.features[name] = detached_dict
                elif isinstance(output, (list, tuple)):
                    # 如果是列表或元組，處理每個元素
                    detached_list = []
                    for item in output:
                        if isinstance(item, torch.Tensor):
                            detached_list.append(item.detach())
                        else:
                            detached_list.append(item)
                    self.features[name] = detached_list
                else:
                    # 其他類型直接保存
                    self.features[name] = output
            return hook
        
        # 註冊backbone特徵提取hook
        # ResNet50 backbone的關鍵層
        self.model.backbone.body.layer1.register_forward_hook(get_activation('backbone_layer1'))
        self.model.backbone.body.layer2.register_forward_hook(get_activation('backbone_layer2'))
        self.model.backbone.body.layer3.register_forward_hook(get_activation('backbone_layer3'))
        self.model.backbone.body.layer4.register_forward_hook(get_activation('backbone_layer4'))
        
        # FPN層特徵
        self.model.backbone.fpn.register_forward_hook(get_activation('fpn_features'))
        
        # RPN特徵
        self.model.rpn.head.register_forward_hook(get_activation('rpn_features'))
        
        # ROI Head特徵 (在box_head之前)
        if hasattr(self.model.roi_heads, 'box_head'):
            self.model.roi_heads.box_head.register_forward_hook(get_activation('roi_features'))
    
    def extract_global_features(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        提取圖像的全局特徵
        
        Args:
            image: 輸入圖像 [C, H, W]
            
        Returns:
            包含各層特徵的字典
        """
        self.features.clear()
        
        # 確保圖像格式正確
        if image.dim() == 2:
            image = image.unsqueeze(0).repeat(3, 1, 1)  # [3, H, W]
        elif image.dim() == 3 and image.size(0) == 1:
            image = image.repeat(3, 1, 1)  # [3, H, W]
        
        image = image.to(self.device)
        
        with torch.no_grad():
            # 前向傳播觸發hook
            _ = self.model([image])
            
            # 提取並處理特徵
            global_features = {}
            
            # Backbone特徵 - 使用全局平均池化
            for layer_name in ['backbone_layer1', 'backbone_layer2', 'backbone_layer3', 'backbone_layer4']:
                if layer_name in self.features:
                    feat = self.features[layer_name]
                    # 全局平均池化 [B, C, H, W] -> [B, C]
                    global_feat = torch.mean(feat, dim=[2, 3])
                    global_features[layer_name] = global_feat.cpu()
            
            # FPN特徵
            if 'fpn_features' in self.features:
                fpn_feat = self.features['fpn_features']
                # FPN返回字典，提取每個尺度的特徵
                if isinstance(fpn_feat, (dict, OrderedDict)):
                    fpn_global = {}
                    for key, value in fpn_feat.items():
                        if isinstance(value, torch.Tensor) and value.dim() == 4:  # [B, C, H, W]
                            fpn_global[f'fpn_{key}'] = torch.mean(value, dim=[2, 3]).cpu()
                    if fpn_global:
                        global_features['fpn'] = fpn_global
                elif isinstance(fpn_feat, torch.Tensor) and fpn_feat.dim() == 4:
                    global_features['fpn'] = torch.mean(fpn_feat, dim=[2, 3]).cpu()
            
            # RPN特徵
            if 'rpn_features' in self.features:
                rpn_feat = self.features['rpn_features']
                if isinstance(rpn_feat, (list, tuple)):
                    # 如果是多個尺度的特徵
                    rpn_global = []
                    for i, feat in enumerate(rpn_feat):
                        if isinstance(feat, torch.Tensor) and feat.dim() == 4:  # [B, C, H, W]
                            rpn_global.append(torch.mean(feat, dim=[2, 3]).cpu())
                    if rpn_global:
                        global_features['rpn'] = torch.cat(rpn_global, dim=1)
                elif isinstance(rpn_feat, torch.Tensor):
                    if rpn_feat.dim() == 4:
                        global_features['rpn'] = torch.mean(rpn_feat, dim=[2, 3]).cpu()
                elif isinstance(rpn_feat, (dict, OrderedDict)):
                    # 如果RPN返回字典
                    rpn_global = []
                    for key, value in rpn_feat.items():
                        if isinstance(value, torch.Tensor) and value.dim() == 4:
                            rpn_global.append(torch.mean(value, dim=[2, 3]).cpu())
                    if rpn_global:
                        global_features['rpn'] = torch.cat(rpn_global, dim=1)
            
            # ROI特徵
            if 'roi_features' in self.features:
                roi_feat = self.features['roi_features']
                if isinstance(roi_feat, torch.Tensor):
                    if roi_feat.dim() == 2:  # [N, C] 已經是池化後的特徵
                        # 對所有ROI特徵取平均
                        global_features['roi'] = torch.mean(roi_feat, dim=0, keepdim=True).cpu()
                    elif roi_feat.dim() == 4:  # [N, C, H, W]
                        roi_pooled = torch.mean(roi_feat, dim=[2, 3])  # [N, C]
                        global_features['roi'] = torch.mean(roi_pooled, dim=0, keepdim=True).cpu()
        
        return global_features
    
    def extract_detection_features(self, image: torch.Tensor, confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """
        提取檢測結果和對應的特徵
        
        Args:
            image: 輸入圖像 [C, H, W]
            confidence_threshold: 檢測置信度閾值
            
        Returns:
            包含檢測結果和特徵的字典
        """
        # 確保圖像格式正確
        if image.dim() == 2:
            image = image.unsqueeze(0).repeat(3, 1, 1)
        elif image.dim() == 3 and image.size(0) == 1:
            image = image.repeat(3, 1, 1)
        
        image = image.to(self.device)
        
        with torch.no_grad():
            # 獲取檢測結果
            predictions = self.model([image])
            pred = predictions[0]
            
            # 過濾低置信度檢測
            high_conf_mask = pred['scores'] > confidence_threshold
            
            detection_features = {
                'num_detections': high_conf_mask.sum().item(),
                'boxes': pred['boxes'][high_conf_mask].cpu().numpy(),
                'scores': pred['scores'][high_conf_mask].cpu().numpy(),
                'labels': pred['labels'][high_conf_mask].cpu().numpy(),
            }
            
            # 計算檢測統計特徵
            if detection_features['num_detections'] > 0:
                boxes = detection_features['boxes']
                scores = detection_features['scores']
                
                # 邊界框統計
                box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                detection_features.update({
                    'avg_box_area': np.mean(box_areas),
                    'max_box_area': np.max(box_areas),
                    'min_box_area': np.min(box_areas),
                    'total_lesion_area': np.sum(box_areas),
                    'avg_confidence': np.mean(scores),
                    'max_confidence': np.max(scores),
                    'min_confidence': np.min(scores),
                })
            else:
                detection_features.update({
                    'avg_box_area': 0.0,
                    'max_box_area': 0.0,
                    'min_box_area': 0.0,
                    'total_lesion_area': 0.0,
                    'avg_confidence': 0.0,
                    'max_confidence': 0.0,
                    'min_confidence': 0.0,
                })
        
        return detection_features
    
    def extract_patient_features(self, images: List[torch.Tensor], patient_id: str, 
                               confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """
        提取單個病例的綜合特徵
        
        Args:
            images: 病例的所有切片圖像列表
            patient_id: 病例ID
            confidence_threshold: 檢測置信度閾值
            
        Returns:
            病例級別的特徵字典
        """
        logging.info(f"提取病例 {patient_id} 的特徵，共 {len(images)} 張切片")
        
        patient_features = {
            'patient_id': patient_id,
            'num_slices': len(images),
            'global_features': {},
            'detection_features': {},
            'slice_features': []
        }
        
        # 收集所有切片的特徵
        all_global_features = {}
        all_detection_stats = []
        
        for i, image in enumerate(tqdm(images, desc=f"處理切片", leave=False)):
            try:
                # 提取全局特徵
                global_feat = self.extract_global_features(image)
                
                # 提取檢測特徵
                detection_feat = self.extract_detection_features(image, confidence_threshold)
                
                # 保存切片級別特徵
                slice_info = {
                    'slice_idx': i,
                    'num_detections': detection_feat['num_detections'],
                    'total_lesion_area': detection_feat['total_lesion_area'],
                    'avg_confidence': detection_feat['avg_confidence'],
                    'max_confidence': detection_feat['max_confidence']
                }
                patient_features['slice_features'].append(slice_info)
                
                # 累積全局特徵
                for feat_name, feat_value in global_feat.items():
                    if feat_name == 'fpn':
                        # FPN特徵的特殊處理
                        if 'fpn' not in all_global_features:
                            all_global_features['fpn'] = {}
                        
                        # feat_value應該是一個字典
                        if isinstance(feat_value, dict):
                            for fpn_key, fpn_value in feat_value.items():
                                if fpn_key not in all_global_features['fpn']:
                                    all_global_features['fpn'][fpn_key] = []
                                all_global_features['fpn'][fpn_key].append(fpn_value)
                        else:
                            # 如果不是字典，作為普通特徵處理
                            if feat_name not in all_global_features:
                                all_global_features[feat_name] = []
                            all_global_features[feat_name].append(feat_value)
                    else:
                        # 普通特徵處理
                        if feat_name not in all_global_features:
                            all_global_features[feat_name] = []
                        all_global_features[feat_name].append(feat_value)
                
                # 累積檢測統計
                all_detection_stats.append(detection_feat)
                
            except Exception as e:
                logging.warning(f"處理切片 {i} 時發生錯誤: {str(e)}")
                continue
        
        # 聚合病例級別的全局特徵
        for feat_name, feat_data in all_global_features.items():
            if feat_name == 'fpn':
                # FPN特徵的特殊處理
                patient_features['global_features']['fpn'] = {}
                
                # feat_data應該是一個字典，每個key對應不同的FPN尺度
                if isinstance(feat_data, dict):
                    for fpn_key, fpn_list in feat_data.items():
                        if fpn_list:
                            try:
                                stacked = torch.stack(fpn_list, dim=0)  # [num_slices, feat_dim]
                                patient_features['global_features']['fpn'][fpn_key] = {
                                    'mean': torch.mean(stacked, dim=0).numpy(),
                                    'std': torch.std(stacked, dim=0).numpy(),
                                    'max': torch.max(stacked, dim=0)[0].numpy(),
                                    'min': torch.min(stacked, dim=0)[0].numpy()
                                }
                            except Exception as e:
                                logging.warning(f"處理FPN特徵 {fpn_key} 時發生錯誤: {str(e)}")
                else:
                    # 如果不是字典，作為普通特徵處理
                    if feat_data:
                        try:
                            stacked = torch.stack(feat_data, dim=0)  # [num_slices, feat_dim]
                            patient_features['global_features'][feat_name] = {
                                'mean': torch.mean(stacked, dim=0).numpy(),
                                'std': torch.std(stacked, dim=0).numpy(),
                                'max': torch.max(stacked, dim=0)[0].numpy(),
                                'min': torch.min(stacked, dim=0)[0].numpy()
                            }
                        except Exception as e:
                            logging.warning(f"處理特徵 {feat_name} 時發生錯誤: {str(e)}")
            else:
                # 普通特徵處理
                if feat_data:
                    try:
                        stacked = torch.stack(feat_data, dim=0)  # [num_slices, feat_dim]
                        patient_features['global_features'][feat_name] = {
                            'mean': torch.mean(stacked, dim=0).numpy(),
                            'std': torch.std(stacked, dim=0).numpy(),
                            'max': torch.max(stacked, dim=0)[0].numpy(),
                            'min': torch.min(stacked, dim=0)[0].numpy()
                        }
                    except Exception as e:
                        logging.warning(f"處理特徵 {feat_name} 時發生錯誤: {str(e)}")
        
        # 聚合病例級別的檢測特徵
        if all_detection_stats:
            total_detections = sum(stat['num_detections'] for stat in all_detection_stats)
            slices_with_lesions = sum(1 for stat in all_detection_stats if stat['num_detections'] > 0)
            
            # 收集所有非零值用於統計
            all_areas = [stat['total_lesion_area'] for stat in all_detection_stats if stat['total_lesion_area'] > 0]
            all_confidences = [stat['avg_confidence'] for stat in all_detection_stats if stat['avg_confidence'] > 0]
            
            patient_features['detection_features'] = {
                'total_detections': total_detections,
                'slices_with_lesions': slices_with_lesions,
                'lesion_ratio': slices_with_lesions / len(images) if len(images) > 0 else 0,
                'avg_detections_per_slice': total_detections / len(images) if len(images) > 0 else 0,
                'total_lesion_volume': sum(all_areas),  # 使用面積和近似體積
                'avg_lesion_area': np.mean(all_areas) if all_areas else 0,
                'max_lesion_area': np.max(all_areas) if all_areas else 0,
                'avg_confidence': np.mean(all_confidences) if all_confidences else 0,
                'max_confidence': np.max(all_confidences) if all_confidences else 0,
            }
        
        return patient_features


def extract_features_from_dataset(model_path: str, data_dir: str, save_dir: str, 
                                split: str = 'val', confidence_threshold: float = 0.5,
                                device: str = None):
    """
    從數據集中提取所有病例的深層特徵
    
    Args:
        model_path: 模型路徑
        data_dir: 數據集目錄
        save_dir: 特徵保存目錄
        split: 數據集分割 ('train', 'val', 'test')
        confidence_threshold: 檢測置信度閾值
        device: 計算設備
    """
    
    # 設置設備
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logging.info(f"開始提取深層特徵，使用設備: {device}")
    logging.info(f"模型路徑: {model_path}")
    logging.info(f"數據目錄: {data_dir}")
    logging.info(f"數據集分割: {split}")
    
    # 創建保存目錄
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化特徵提取器
    feature_extractor = DeepFeatureExtractor(model_path, device)
    
    # 載入數據集
    try:
        dataset = CTDetectionDataset(
            data_root=data_dir,
            split=split,
            transforms=None,  # 使用原始圖像
            include_negative_samples=True,
            max_negative_per_patient=0  # 包含所有切片
        )
        logging.info(f"載入數據集成功，共 {len(dataset)} 個樣本")
    except Exception as e:
        logging.error(f"載入數據集失敗: {str(e)}")
        return
    
    # 按病例組織數據
    patient_data = {}
    for i in range(len(dataset)):
        try:
            sample = dataset[i]
            patient_id = sample['patient_id']
            
            if patient_id not in patient_data:
                patient_data[patient_id] = []
            
            # 獲取原始圖像（未經預處理）
            image = sample['image']
            if isinstance(image, torch.Tensor):
                # 如果已經是tensor，確保在CPU上
                image = image.cpu()
            
            patient_data[patient_id].append(image)
            
        except Exception as e:
            logging.warning(f"處理樣本 {i} 時發生錯誤: {str(e)}")
            continue
    
    logging.info(f"共找到 {len(patient_data)} 個病例")
    
    # 提取每個病例的特徵
    all_patient_features = {}
    
    for patient_id, images in tqdm(patient_data.items(), desc="提取病例特徵"):
        try:
            # 提取病例特徵
            patient_features = feature_extractor.extract_patient_features(
                images, patient_id, confidence_threshold
            )
            
            all_patient_features[patient_id] = patient_features
            
            # 保存單個病例的特徵
            patient_dir = os.path.join(save_dir, patient_id)
            os.makedirs(patient_dir, exist_ok=True)
            
            patient_save_path = os.path.join(patient_dir, f"{patient_id}_features.pkl")
            with open(patient_save_path, 'wb') as f:
                pickle.dump(patient_features, f)
            
            # 同時保存JSON格式（去除numpy數組）
            json_features = convert_features_to_json(patient_features)
            json_save_path = os.path.join(patient_dir, f"{patient_id}_features.json")
            with open(json_save_path, 'w', encoding='utf-8') as f:
                json.dump(json_features, f, indent=2, ensure_ascii=False)
            
            logging.info(f"病例 {patient_id} 特徵提取完成，保存到 {patient_dir}")
            
            # 立即為這個病例生成特徵報告（可選）
            try:
                from feature_loader import FeatureLoader, FeatureVisualizer
                
                # 創建臨時的特徵加載器來生成報告
                temp_loader = FeatureLoader(save_dir)
                if patient_id in temp_loader.get_all_patient_ids():
                    visualizer = FeatureVisualizer(temp_loader)
                    report_path = os.path.join(patient_dir, f"{patient_id}_feature_report.md")
                    visualizer.create_patient_report(patient_id, report_path)
                    logging.info(f"病例 {patient_id} 特徵報告已生成: {report_path}")
            except Exception as e:
                logging.warning(f"生成病例 {patient_id} 即時報告失敗: {str(e)}")
            
        except Exception as e:
            logging.error(f"提取病例 {patient_id} 特徵時發生錯誤: {str(e)}")
            continue
    
    # 保存所有特徵的匯總
    summary_path = os.path.join(save_dir, "all_features_summary.pkl")
    with open(summary_path, 'wb') as f:
        pickle.dump(all_patient_features, f)
    
    # 創建特徵統計報告
    create_feature_report(all_patient_features, save_dir)
    
    logging.info(f"特徵提取完成，共處理 {len(all_patient_features)} 個病例")
    logging.info(f"特徵保存在: {save_dir}")


def convert_features_to_json(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    將包含numpy數組的特徵字典轉換為JSON可序列化的格式
    """
    def convert_value(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, torch.Tensor):
            return value.numpy().tolist()
        elif isinstance(value, dict):
            return {k: convert_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [convert_value(v) for v in value]
        else:
            return value
    
    return convert_value(features)


def create_feature_report(all_features: Dict[str, Dict], save_dir: str):
    """
    創建特徵提取報告
    """
    report = {
        'extraction_time': datetime.now().isoformat(),
        'total_patients': len(all_features),
        'feature_summary': {
            'patients_with_lesions': 0,
            'total_detections': 0,
            'avg_detections_per_patient': 0,
            'avg_slices_per_patient': 0,
        }
    }
    
    # 統計特徵
    patients_with_lesions = 0
    total_detections = 0
    total_slices = 0
    
    for patient_id, features in all_features.items():
        detection_features = features.get('detection_features', {})
        
        if detection_features.get('total_detections', 0) > 0:
            patients_with_lesions += 1
        
        total_detections += detection_features.get('total_detections', 0)
        total_slices += features.get('num_slices', 0)
    
    if len(all_features) > 0:
        report['feature_summary']['patients_with_lesions'] = patients_with_lesions
        report['feature_summary']['total_detections'] = total_detections
        report['feature_summary']['avg_detections_per_patient'] = total_detections / len(all_features)
        report['feature_summary']['avg_slices_per_patient'] = total_slices / len(all_features)
        report['feature_summary']['lesion_patient_ratio'] = patients_with_lesions / len(all_features)
    
    # 保存報告
    report_path = os.path.join(save_dir, 'feature_extraction_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logging.info(f"特徵提取報告保存到: {report_path}")


def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description="提取病例深層特徵供LLM使用")
    parser.add_argument('--model_path', type=str, required=True,
                       help="訓練好的模型路徑")
    parser.add_argument('--data_dir', type=str, required=True,
                       help="數據集目錄")
    parser.add_argument('--save_dir', type=str, default='./deep_features',
                       help="特徵保存目錄")
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val', 'test'],
                       help="數據集分割")
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                       help="檢測置信度閾值")
    parser.add_argument('--device', type=str, default=None,
                       help="計算設備 (cuda/cpu)")
    
    args = parser.parse_args()
    
    # 設置日誌
    log_dir = os.path.join(args.save_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'feature_extraction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # 開始特徵提取
    extract_features_from_dataset(
        model_path=args.model_path,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        split=args.split,
        confidence_threshold=args.confidence_threshold,
        device=args.device
    )


if __name__ == "__main__":
    main()
