#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CT-ViT 目標檢測推理腳本
使用訓練好的檢測模型進行腫瘤檢測和報告生成

使用方式:
python inference_detection.py --model_path CT_ViT_Detection/best_detection_model.pth --input_dicom path/to/dicom/file.dcm

主要功能：
1. 載入訓練好的檢測模型
2. 處理DICOM檔案
3. 執行腫瘤檢測
4. 生成結構化醫療報告
5. 視覺化檢測結果

作者: GitHub Copilot
日期: 2025-07-25
"""

import os
import sys
import json
import argparse
from datetime import datetime
import logging

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import cv2

# 添加src路徑
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from detection_model import CTViTForDetection
from transformers import ViTConfig

class CTDetectionInference:
    """CT目標檢測推理器"""
    
    def __init__(self, model_path, device='cuda'):
        """
        Args:
            model_path: 訓練好的模型路徑
            device: 計算設備
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        
        # 類別映射
        self.class_names = {
            0: 'Background',
            1: 'Adenocarcinoma (惡性腺癌)',
            2: 'Benign nodule (良性結節)',
            3: 'E-type lesion',
            4: 'G-type lesion'
        }
        
        # 風險等級映射
        self.risk_levels = {
            0: 'Normal',
            1: 'High Risk (惡性)',
            2: 'Low Risk (良性)',
            3: 'Medium Risk',
            4: 'Medium Risk'
        }
        
    def _load_model(self, model_path):
        """載入模型"""
        print(f"載入模型: {model_path}")
        
        # 載入檢查點
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 創建模型配置
        config = ViTConfig(
            image_size=224,
            patch_size=16,
            num_channels=3,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            num_labels=5,  # 包含背景
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        
        # 創建模型
        model = CTViTForDetection(config)
        
        # 載入權重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        print(f"模型載入成功，設備: {self.device}")
        return model
    
    def preprocess_dicom(self, dicom_path, target_size=224):
        """預處理DICOM檔案"""
        try:
            # 讀取DICOM
            dicom_data = pydicom.dcmread(dicom_path)
            image = dicom_data.pixel_array.astype(np.float32)
            
            # 正規化
            image = np.clip(image, np.percentile(image, 1), np.percentile(image, 99))
            image = (image - image.min()) / (image.max() - image.min()) * 255.0
            image = image.astype(np.uint8)
            
            # 調整大小
            image = cv2.resize(image, (target_size, target_size))
            
            # 轉換為3通道
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=2)
            
            # 正規化到[0, 1]並轉換為張量
            image = image.astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
            
            return image_tensor.to(self.device), image
            
        except Exception as e:
            print(f"DICOM預處理失敗: {e}")
            return None, None
    
    def detect(self, image_tensor, confidence_threshold=0.5):
        """執行檢測"""
        with torch.no_grad():
            outputs = self.model(pixel_values=image_tensor)
            
            # 處理輸出
            class_logits = outputs['class_logits']
            bbox_pred = outputs['bbox_pred']
            objectness_logits = outputs['objectness_logits']
            
            # 計算概率
            class_probs = F.softmax(class_logits, dim=-1)[0]
            objectness_prob = torch.sigmoid(objectness_logits)[0].item()
            
            # 獲取預測類別
            predicted_class = torch.argmax(class_probs).item()
            class_confidence = class_probs[predicted_class].item()
            
            # 獲取邊界框
            bbox = bbox_pred[0].cpu().numpy()
            
            # 檢測結果
            detection_result = {
                'has_lesion': objectness_prob > confidence_threshold,
                'objectness_score': objectness_prob,
                'predicted_class': predicted_class,
                'class_name': self.class_names[predicted_class],
                'class_confidence': class_confidence,
                'bbox': bbox,  # [x_center, y_center, width, height] 正規化座標
                'risk_level': self.risk_levels[predicted_class]
            }
            
            return detection_result
    
    def generate_report(self, detection_result, patient_id="Unknown", study_date=None):
        """生成結構化醫療報告"""
        if study_date is None:
            study_date = datetime.now().strftime("%Y-%m-%d")
        
        report = {
            "patient_id": patient_id,
            "study_date": study_date,
            "modality": "CT",
            "examination": "Chest CT",
            "ai_analysis": {
                "detection_confidence": detection_result['objectness_score'],
                "findings": [],
                "impression": "",
                "recommendations": []
            }
        }
        
        if detection_result['has_lesion']:
            # 計算病灶大小（假設影像尺寸為512x512mm）
            bbox = detection_result['bbox']
            lesion_width_mm = bbox[2] * 512  # 寬度(mm)
            lesion_height_mm = bbox[3] * 512  # 高度(mm)
            lesion_area_mm2 = lesion_width_mm * lesion_height_mm
            
            # 添加發現
            finding = {
                "description": f"檢測到疑似病灶",
                "location": {
                    "coordinates": {
                        "x_center": f"{bbox[0]:.3f}",
                        "y_center": f"{bbox[1]:.3f}",
                        "width": f"{bbox[2]:.3f}",
                        "height": f"{bbox[3]:.3f}"
                    },
                    "anatomical_region": self._determine_lung_region(bbox[0], bbox[1])
                },
                "characteristics": {
                    "type": detection_result['class_name'],
                    "size": {
                        "width_mm": f"{lesion_width_mm:.1f}",
                        "height_mm": f"{lesion_height_mm:.1f}",
                        "area_mm2": f"{lesion_area_mm2:.1f}"
                    },
                    "confidence": f"{detection_result['class_confidence']:.3f}"
                },
                "risk_assessment": detection_result['risk_level']
            }
            
            report["ai_analysis"]["findings"].append(finding)
            
            # 生成印象
            if detection_result['predicted_class'] == 1:  # 惡性
                report["ai_analysis"]["impression"] = f"檢測到疑似惡性病灶（腺癌），位於{finding['location']['anatomical_region']}，建議進一步檢查。"
                report["ai_analysis"]["recommendations"] = [
                    "建議PET-CT檢查確認病灶性質",
                    "建議胸腔外科會診評估手術可行性",
                    "建議腫瘤標記檢測",
                    "建議3個月內追蹤檢查"
                ]
            elif detection_result['predicted_class'] == 2:  # 良性
                report["ai_analysis"]["impression"] = f"檢測到疑似良性結節，位於{finding['location']['anatomical_region']}，建議定期追蹤。"
                report["ai_analysis"]["recommendations"] = [
                    "建議6-12個月追蹤CT檢查",
                    "如有症狀變化請及時就醫",
                    "保持健康生活方式"
                ]
            else:
                report["ai_analysis"]["impression"] = f"檢測到肺部病灶，位於{finding['location']['anatomical_region']}，性質待確定。"
                report["ai_analysis"]["recommendations"] = [
                    "建議專科醫師進一步評估",
                    "建議必要時進行組織學檢查",
                    "建議定期追蹤"
                ]
        else:
            report["ai_analysis"]["impression"] = "未檢測到明顯肺部病灶。"
            report["ai_analysis"]["recommendations"] = [
                "建議按照常規篩檢計劃追蹤",
                "如有呼吸道症狀請及時就醫"
            ]
        
        return report
    
    def _determine_lung_region(self, x_center, y_center):
        """根據座標判斷肺葉區域"""
        if x_center < 0.5:
            lung_side = "左肺"
        else:
            lung_side = "右肺"
        
        if y_center < 0.4:
            lung_lobe = "上葉"
        elif y_center < 0.7:
            lung_lobe = "中葉" if lung_side == "右肺" else "舌葉"
        else:
            lung_lobe = "下葉"
        
        return f"{lung_side}{lung_lobe}"
    
    def visualize_detection(self, image, detection_result, save_path=None):
        """視覺化檢測結果"""
        plt.figure(figsize=(12, 6))
        
        # 原始影像
        plt.subplot(1, 2, 1)
        plt.imshow(image[:, :, 0], cmap='gray')
        plt.title('原始CT影像')
        plt.axis('off')
        
        # 檢測結果
        plt.subplot(1, 2, 2)
        plt.imshow(image[:, :, 0], cmap='gray')
        
        if detection_result['has_lesion']:
            # 繪製邊界框
            bbox = detection_result['bbox']
            x_center, y_center, width, height = bbox
            
            # 轉換為像素座標
            img_h, img_w = image.shape[:2]
            x1 = (x_center - width/2) * img_w
            y1 = (y_center - height/2) * img_h
            x2 = (x_center + width/2) * img_w
            y2 = (y_center + height/2) * img_h
            
            # 繪製矩形
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, color='red', linewidth=2)
            plt.gca().add_patch(rect)
            
            # 添加標籤
            label = f"{detection_result['class_name']}\n信心度: {detection_result['class_confidence']:.3f}"
            plt.text(x1, y1-10, label, color='red', fontsize=8, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.title(f'檢測結果 - {detection_result["risk_level"]}')
        plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"檢測結果已保存到: {save_path}")
        
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='CT-ViT目標檢測推理')
    
    parser.add_argument('--model_path', type=str, required=True,
                      help='訓練好的模型路徑')
    parser.add_argument('--input_dicom', type=str, required=True,
                      help='輸入DICOM檔案路徑')
    parser.add_argument('--patient_id', type=str, default='Unknown',
                      help='患者ID')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                      help='檢測置信度閾值')
    parser.add_argument('--output_dir', type=str, default='detection_results',
                      help='結果輸出目錄')
    
    args = parser.parse_args()
    
    # 創建輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化推理器
    print("初始化CT檢測推理器...")
    inference = CTDetectionInference(args.model_path)
    
    # 預處理影像
    print(f"處理DICOM檔案: {args.input_dicom}")
    image_tensor, image_np = inference.preprocess_dicom(args.input_dicom)
    
    if image_tensor is None:
        print("DICOM檔案處理失敗！")
        return
    
    # 執行檢測
    print("執行病灶檢測...")
    detection_result = inference.detect(image_tensor, args.confidence_threshold)
    
    # 生成報告
    print("生成醫療報告...")
    report = inference.generate_report(detection_result, args.patient_id)
    
    # 保存結果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存檢測結果JSON
    result_path = os.path.join(args.output_dir, f'detection_result_{timestamp}.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump({
            'detection_result': detection_result,
            'medical_report': report
        }, f, indent=2, ensure_ascii=False)
    
    # 保存視覺化結果
    vis_path = os.path.join(args.output_dir, f'detection_visualization_{timestamp}.png')
    inference.visualize_detection(image_np, detection_result, vis_path)
    
    # 打印結果摘要
    print("\n" + "="*50)
    print("檢測結果摘要")
    print("="*50)
    print(f"患者ID: {args.patient_id}")
    print(f"檢測到病灶: {'是' if detection_result['has_lesion'] else '否'}")
    
    if detection_result['has_lesion']:
        print(f"病灶類型: {detection_result['class_name']}")
        print(f"分類信心度: {detection_result['class_confidence']:.3f}")
        print(f"檢測信心度: {detection_result['objectness_score']:.3f}")
        print(f"風險等級: {detection_result['risk_level']}")
        
        bbox = detection_result['bbox']
        print(f"病灶位置: 中心({bbox[0]:.3f}, {bbox[1]:.3f}), 大小({bbox[2]:.3f}, {bbox[3]:.3f})")
    
    print(f"\n結果已保存到:")
    print(f"  檢測結果: {result_path}")
    print(f"  視覺化圖片: {vis_path}")
    print("="*50)

if __name__ == "__main__":
    main()
