#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CT-ViT 推理和評估腳本
用於載入訓練好的CT-ViT模型進行推理和詳細評估

功能包括：
1. 單張影像推理
2. 批次推理
3. 模型性能評估
4. 注意力可視化
5. 特徵提取

作者: GitHub Copilot
日期: 2025-07-22
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# 添加src路徑到Python路徑
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import ViTImageProcessor, ViTForImageClassification
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# 導入自定義模組
from config import CTViTConfig
from data_processing import DICOMProcessor, CTDataset

class CTViTInference:
    """CT-ViT推理器"""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 載入模型和處理器
        self.model = ViTForImageClassification.from_pretrained(model_path)
        self.image_processor = ViTImageProcessor.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # 類別標籤
        self.label_names = ['A_series', 'B_series', 'E_series', 'G_series']
        self.label_map = {'A': 0, 'B': 1, 'E': 2, 'G': 3}
        
        # 載入配置
        if config_path and os.path.exists(config_path):
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config_dict = yaml.safe_load(f)
        else:
            self.config_dict = {}
        
        # DICOM處理器
        dummy_config = CTViTConfig()
        self.dicom_processor = DICOMProcessor(dummy_config)
        
        print(f"模型載入成功: {model_path}")
        print(f"使用設備: {self.device}")
    
    def predict_single_image(self, image_path: str, return_attention: bool = False) -> Dict[str, Any]:
        """單張影像推理"""
        
        # 載入和預處理影像
        if image_path.endswith('.dcm'):
            image = self.dicom_processor.preprocess_dicom(image_path)
        else:
            image = cv2.imread(image_path)
            image = cv2.resize(image, (224, 224))
        
        # 轉換為模型輸入格式
        inputs = self.image_processor(image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=return_attention)
            
            # 預測結果
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)
            predicted_class = torch.argmax(logits, dim=-1).item()
            confidence = probabilities[0, predicted_class].item()
            
            results = {
                'predicted_class': predicted_class,
                'predicted_label': self.label_names[predicted_class],
                'confidence': confidence,
                'probabilities': probabilities[0].cpu().numpy().tolist(),
                'class_probabilities': {
                    name: prob for name, prob in zip(self.label_names, probabilities[0].cpu().numpy())
                }
            }
            
            if return_attention:
                results['attention_weights'] = outputs.attentions
            
        return results
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """批次推理"""
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict_single_image(image_path)
                result['image_path'] = image_path
                results.append(result)
                print(f"✓ 處理完成: {os.path.basename(image_path)}")
            except Exception as e:
                print(f"✗ 處理失敗: {image_path} - {str(e)}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results
    
    def evaluate_dataset(self, dataset_path: str, output_dir: str) -> Dict[str, Any]:
        """評估整個資料集"""
        
        # 創建輸出目錄
        os.makedirs(output_dir, exist_ok=True)
        
        # 載入資料集
        dummy_config = CTViTConfig()
        dataset = CTDataset(dataset_path, dummy_config, "test")
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        print(f"評估資料集: {len(dataset)} 個樣本")
        
        # 批次處理
        for i, item in enumerate(dataset):
            if i % 50 == 0:
                print(f"處理進度: {i}/{len(dataset)}")
            
            try:
                # 準備輸入
                pixel_values = item['pixel_values'].unsqueeze(0).to(self.device)
                inputs = {'pixel_values': pixel_values}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probabilities = F.softmax(outputs.logits, dim=-1)
                    predicted_class = torch.argmax(outputs.logits, dim=-1).item()
                
                all_predictions.append(predicted_class)
                all_labels.append(item['labels'].item())
                all_probabilities.append(probabilities[0].cpu().numpy())
                
            except Exception as e:
                print(f"處理樣本 {i} 時出錯: {str(e)}")
                continue
        
        # 計算評估指標
        results = self._compute_detailed_metrics(
            all_labels, all_predictions, all_probabilities, output_dir
        )
        
        # 保存結果
        results_file = os.path.join(output_dir, "evaluation_results.json")
        
        def convert_numpy_types(obj):
            """遞迴轉換numpy類型為Python原生類型"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json_results = convert_numpy_types(results)
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        return results
    
    def _compute_detailed_metrics(self, y_true: List[int], y_pred: List[int], 
                                 y_prob: List[np.ndarray], output_dir: str) -> Dict[str, Any]:
        """計算詳細的評估指標"""
        
        # 基本指標
        accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
        
        # 分類報告
        class_report = classification_report(y_true, y_pred, target_names=self.label_names, output_dict=True)
        
        # 混淆矩陣
        cm = confusion_matrix(y_true, y_pred)
        
        # ROC曲線和AUC
        y_true_bin = label_binarize(y_true, classes=range(len(self.label_names)))
        y_prob_array = np.array(y_prob)
        
        roc_data = {}
        for i, class_name in enumerate(self.label_names):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob_array[:, i])
            roc_auc = auc(fpr, tpr)
            roc_data[class_name] = {
                'fpr': fpr.tolist(),  # 轉換為列表
                'tpr': tpr.tolist(),  # 轉換為列表
                'auc': float(roc_auc)  # 確保是Python float
            }
        
        # 繪製可視化圖表
        self._plot_confusion_matrix(cm, output_dir)
        self._plot_roc_curves(roc_data, output_dir)
        self._plot_class_distribution(y_true, y_pred, output_dir)
        
        return {
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'roc_data': roc_data,
            'predictions': y_pred,
            'true_labels': y_true,
            'probabilities': y_prob
        }
    
    def _plot_confusion_matrix(self, cm: np.ndarray, output_dir: str):
        """繪製混淆矩陣"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_names, yticklabels=self.label_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # 添加準確率信息
        for i in range(len(self.label_names)):
            accuracy_i = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
            plt.text(i + 0.5, i - 0.3, f'{accuracy_i:.2%}', 
                    ha='center', va='center', fontweight='bold', color='red')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curves(self, roc_data: Dict, output_dir: str):
        """繪製ROC曲線"""
        plt.figure(figsize=(12, 8))
        
        colors = ['blue', 'red', 'green', 'orange']
        for i, (class_name, data) in enumerate(roc_data.items()):
            plt.plot(data['fpr'], data['tpr'], color=colors[i],
                    label=f'{class_name} (AUC = {data["auc"]:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_class_distribution(self, y_true: List[int], y_pred: List[int], output_dir: str):
        """繪製類別分佈圖"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 真實標籤分佈
        true_counts = [y_true.count(i) for i in range(len(self.label_names))]
        ax1.bar(self.label_names, true_counts, color='skyblue', alpha=0.7)
        ax1.set_title('True Label Distribution')
        ax1.set_ylabel('Count')
        for i, count in enumerate(true_counts):
            ax1.text(i, count + 0.1, str(count), ha='center', va='bottom')
        
        # 預測標籤分佈
        pred_counts = [y_pred.count(i) for i in range(len(self.label_names))]
        ax2.bar(self.label_names, pred_counts, color='lightcoral', alpha=0.7)
        ax2.set_title('Predicted Label Distribution')
        ax2.set_ylabel('Count')
        for i, count in enumerate(pred_counts):
            ax2.text(i, count + 0.1, str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="CT-ViT 推理和評估工具")
    parser.add_argument("--model_path", required=True, help="訓練好的模型路徑")
    parser.add_argument("--mode", choices=["single", "batch", "evaluate"], 
                       default="single", help="運行模式")
    parser.add_argument("--input", help="輸入影像路徑或資料集路徑")
    parser.add_argument("--output", default="./inference_results", help="輸出目錄")
    parser.add_argument("--config", help="配置文件路徑")
    
    args = parser.parse_args()
    
    # 創建推理器
    inferencer = CTViTInference(args.model_path, args.config)
    
    if args.mode == "single":
        # 單張影像推理
        if not args.input:
            print("錯誤: 請提供輸入影像路徑 (--input)")
            return
        
        results = inferencer.predict_single_image(args.input)
        print(f"\n推理結果:")
        print(f"預測類別: {results['predicted_label']}")
        print(f"信心度: {results['confidence']:.3f}")
        print(f"各類別機率:")
        for name, prob in results['class_probabilities'].items():
            print(f"  {name}: {prob:.3f}")
    
    elif args.mode == "batch":
        # 批次推理
        if not args.input:
            print("錯誤: 請提供影像列表文件或目錄路徑")
            return
        
        # 獲取影像路徑列表
        if os.path.isfile(args.input):
            with open(args.input, 'r', encoding='utf-8') as f:
                image_paths = [line.strip() for line in f if line.strip()]
        elif os.path.isdir(args.input):
            image_paths = []
            for ext in ['*.dcm', '*.jpg', '*.png', '*.jpeg']:
                image_paths.extend(Path(args.input).glob(f"**/{ext}"))
            image_paths = [str(p) for p in image_paths]
        else:
            print("錯誤: 輸入路徑無效")
            return
        
        results = inferencer.predict_batch(image_paths)
        
        # 保存結果
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, "batch_results.json"), 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"批次推理完成，結果已保存到: {args.output}/batch_results.json")
    
    elif args.mode == "evaluate":
        # 資料集評估
        if not args.input:
            print("錯誤: 請提供資料集路徑")
            return
        
        results = inferencer.evaluate_dataset(args.input, args.output)
        print(f"\n評估完成!")
        print(f"準確率: {results['accuracy']:.4f}")
        print(f"詳細結果已保存到: {args.output}")
    
    else:
        print("錯誤: 未知的運行模式")

if __name__ == "__main__":
    main()
