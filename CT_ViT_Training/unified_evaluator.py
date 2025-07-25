#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CT-ViT 統一評估系統
支援分類模型和檢測模型的評估

功能:
- 分類模型評估 (原始CT-ViT)
- 檢測模型評估 (升級版CT-ViT)
- 性能指標計算和視覺化
- 詳細報告生成

作者: GitHub Copilot
日期: 2025-07-25
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, 
    precision_recall_fscore_support, average_precision_score
)
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

# 添加src路徑
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from detection_model import CTViTForDetection
    from detection_dataset import CTDetectionDataset
    from utils import setup_logging
    HAS_DETECTION_SUPPORT = True
except ImportError:
    HAS_DETECTION_SUPPORT = False
    print("⚠️  檢測模型支援未找到，僅支援分類評估")

class ModelEvaluator:
    """統一模型評估器"""
    
    def __init__(self, config_path: str = None):
        """初始化評估器"""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 類別映射
        self.class_names = {
            0: 'Background',
            1: 'Adenocarcinoma (A)',
            2: 'Benign (B)', 
            3: 'Emphysema (E)',
            4: 'Ground Glass (G)'
        }
        
        self.logger.info(f"評估器初始化完成，使用設備: {self.device}")
    
    def _load_config(self, config_path: str) -> Dict:
        """載入配置"""
        default_config = {
            "model": {
                "classification_model_path": "../CT_ViT/models/best_model.pth",
                "detection_model_path": "models/best_detection_model.pth",
                "model_type": "classification"  # classification or detection
            },
            "data": {
                "test_data_dir": "../../dataset_splits/test",
                "validation_data_dir": "../../dataset_splits/validation",
                "image_size": 224,
                "batch_size": 8
            },
            "evaluation": {
                "confidence_threshold": 0.5,
                "iou_threshold": 0.5,
                "save_visualizations": True,
                "save_detailed_results": True
            },
            "output": {
                "results_dir": "evaluation_results",
                "plots_dir": "evaluation_plots"
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            default_config.update(user_config)
        
        return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """設置日誌"""
        log_dir = Path(self.config["output"]["results_dir"]) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        
        return logging.getLogger(__name__)
    
    def load_classification_model(self, model_path: str):
        """載入分類模型"""
        try:
            from transformers import ViTForImageClassification
            
            if os.path.exists(model_path):
                self.logger.info(f"載入分類模型: {model_path}")
                model = ViTForImageClassification.from_pretrained(model_path)
            else:
                self.logger.warning(f"模型路徑不存在: {model_path}")
                return None
            
            model.to(self.device)
            model.eval()
            return model
            
        except Exception as e:
            self.logger.error(f"載入分類模型失敗: {e}")
            return None
    
    def load_detection_model(self, model_path: str):
        """載入檢測模型"""
        if not HAS_DETECTION_SUPPORT:
            self.logger.error("檢測模型支援未啟用")
            return None
        
        try:
            if not os.path.exists(model_path):
                self.logger.warning(f"檢測模型不存在: {model_path}")
                return None
            
            self.logger.info(f"載入檢測模型: {model_path}")
            
            # 載入檢測模型
            from transformers import ViTConfig
            
            config = ViTConfig(
                image_size=self.config["data"]["image_size"],
                patch_size=16,
                num_channels=3,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                num_labels=5
            )
            
            model = CTViTForDetection(config)
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            self.logger.error(f"載入檢測模型失敗: {e}")
            return None
    
    def evaluate_classification_model(self, model, test_dataset) -> Dict:
        """評估分類模型"""
        self.logger.info("開始分類模型評估...")
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config["data"]["batch_size"], 
            shuffle=False
        )
        
        all_predictions = []
        all_labels = []
        all_confidences = []
        
        with torch.no_grad():
            for batch in test_loader:
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = model(pixel_values)
                probabilities = F.softmax(outputs.logits, dim=-1)
                predictions = torch.argmax(probabilities, dim=-1)
                confidences = torch.max(probabilities, dim=-1)[0]
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
        
        # 計算評估指標
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )
        
        # 每個類別的指標
        per_class_metrics = precision_recall_fscore_support(
            all_labels, all_predictions, average=None, zero_division=0
        )
        
        results = {
            "model_type": "classification",
            "overall_metrics": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1)
            },
            "per_class_metrics": {},
            "predictions": all_predictions,
            "true_labels": all_labels,
            "confidences": all_confidences
        }
        
        # 每個類別的詳細指標
        unique_labels = sorted(set(all_labels))
        for i, label in enumerate(unique_labels):
            if i < len(per_class_metrics[0]):
                results["per_class_metrics"][self.class_names.get(label, f"Class_{label}")] = {
                    "precision": float(per_class_metrics[0][i]),
                    "recall": float(per_class_metrics[1][i]),
                    "f1_score": float(per_class_metrics[2][i]),
                    "support": int(per_class_metrics[3][i])
                }
        
        return results
    
    def evaluate_detection_model(self, model, test_dataset) -> Dict:
        """評估檢測模型"""
        if not HAS_DETECTION_SUPPORT:
            return {"error": "檢測模型支援未啟用"}
        
        self.logger.info("開始檢測模型評估...")
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config["data"]["batch_size"],
            shuffle=False
        )
        
        all_predictions = []
        all_bbox_predictions = []
        all_objectness_scores = []
        all_labels = []
        all_bbox_targets = []
        all_objectness_targets = []
        
        confidence_threshold = self.config["evaluation"]["confidence_threshold"]
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(self.device)
                labels = batch['labels'].to(self.device)
                bbox_targets = batch['boxes'].to(self.device)
                objectness_targets = batch['has_object'].to(self.device)
                
                outputs = model(pixel_values=images)
                
                # 分類預測
                class_probs = F.softmax(outputs['logits'], dim=-1)
                class_predictions = torch.argmax(class_probs, dim=-1)
                
                # 目標性預測
                objectness_probs = torch.sigmoid(outputs['objectness_logits'])
                
                # 收集結果
                all_predictions.extend(class_predictions.cpu().numpy())
                all_bbox_predictions.extend(outputs['bbox_pred'].cpu().numpy())
                all_objectness_scores.extend(objectness_probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_bbox_targets.extend(bbox_targets.cpu().numpy())
                all_objectness_targets.extend(objectness_targets.cpu().numpy())
        
        # 計算檢測指標
        detection_results = self._calculate_detection_metrics(
            all_predictions, all_bbox_predictions, all_objectness_scores,
            all_labels, all_bbox_targets, all_objectness_targets,
            confidence_threshold
        )
        
        return detection_results
    
    def _calculate_detection_metrics(self, pred_classes, pred_boxes, pred_objectness,
                                   true_classes, true_boxes, true_objectness, threshold) -> Dict:
        """計算檢測指標"""
        
        # 分類指標
        class_accuracy = accuracy_score(true_classes, pred_classes)
        class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
            true_classes, pred_classes, average='weighted', zero_division=0
        )
        
        # 目標檢測指標
        pred_has_object = (np.array(pred_objectness) > threshold).astype(int)
        true_has_object = np.array(true_objectness)
        
        objectness_accuracy = accuracy_score(true_has_object, pred_has_object)
        
        # 邊界框IoU計算
        ious = []
        for pred_box, true_box in zip(pred_boxes, true_boxes):
            if np.any(true_box > 0):  # 只有真實邊界框存在時才計算IoU
                iou = self._calculate_iou(pred_box, true_box)
                ious.append(iou)
        
        mean_iou = np.mean(ious) if ious else 0.0
        
        return {
            "model_type": "detection", 
            "classification_metrics": {
                "accuracy": float(class_accuracy),
                "precision": float(class_precision),
                "recall": float(class_recall),
                "f1_score": float(class_f1)
            },
            "detection_metrics": {
                "objectness_accuracy": float(objectness_accuracy),
                "mean_iou": float(mean_iou),
                "num_valid_boxes": len(ious)
            },
            "predictions": {
                "classes": pred_classes,
                "boxes": pred_boxes,
                "objectness": pred_objectness
            },
            "targets": {
                "classes": true_classes,
                "boxes": true_boxes,
                "objectness": true_objectness
            }
        }
    
    def _calculate_iou(self, box1, box2):
        """計算兩個邊界框的IoU"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # 轉換為座標格式
        x1_min, y1_min, x1_max, y1_max = x1 - w1/2, y1 - h1/2, x1 + w1/2, y1 + h1/2
        x2_min, y2_min, x2_max, y2_max = x2 - w2/2, y2 - h2/2, x2 + w2/2, y2 + h2/2
        
        # 計算交集
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def create_visualizations(self, results: Dict):
        """創建評估視覺化"""
        if not self.config["evaluation"]["save_visualizations"]:
            return
        
        plots_dir = Path(self.config["output"]["plots_dir"])
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("生成評估視覺化...")
        
        if results["model_type"] == "classification":
            self._create_classification_plots(results, plots_dir)
        elif results["model_type"] == "detection":
            self._create_detection_plots(results, plots_dir)
    
    def _create_classification_plots(self, results: Dict, plots_dir: Path):
        """創建分類模型視覺化"""
        
        # 1. 混淆矩陣
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(results["true_labels"], results["predictions"])
        
        # 獲取實際使用的類別
        unique_labels = sorted(set(results["true_labels"]))
        class_labels = [self.class_names.get(label, f"Class_{label}") for label in unique_labels]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_labels, yticklabels=class_labels)
        plt.title('分類模型混淆矩陣')
        plt.ylabel('真實標籤')
        plt.xlabel('預測標籤')
        plt.tight_layout()
        plt.savefig(plots_dir / 'confusion_matrix.png', dpi=300)
        plt.close()
        
        # 2. 信心度分布
        plt.figure(figsize=(10, 6))
        plt.hist(results["confidences"], bins=50, alpha=0.7, edgecolor='black')
        plt.title('預測信心度分布')
        plt.xlabel('信心度')
        plt.ylabel('頻率')
        plt.axvline(np.mean(results["confidences"]), color='red', linestyle='--', 
                   label=f'平均信心度: {np.mean(results["confidences"]):.3f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / 'confidence_distribution.png', dpi=300)
        plt.close()
        
        # 3. 每個類別的指標比較
        if results["per_class_metrics"]:
            classes = list(results["per_class_metrics"].keys())
            metrics = ['precision', 'recall', 'f1_score']
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            for i, metric in enumerate(metrics):
                values = [results["per_class_metrics"][cls][metric] for cls in classes]
                axes[i].bar(classes, values, alpha=0.7)
                axes[i].set_title(f'每類別{metric.title()}')
                axes[i].set_ylabel(metric.title())
                axes[i].tick_params(axis='x', rotation=45)
                
                # 添加數值標籤
                for j, v in enumerate(values):
                    axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center')
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'per_class_metrics.png', dpi=300)
            plt.close()
    
    def _create_detection_plots(self, results: Dict, plots_dir: Path):
        """創建檢測模型視覺化"""
        
        # 1. 分類準確度 vs 檢測準確度
        class_metrics = results["classification_metrics"]
        detection_metrics = results["detection_metrics"]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 分類指標
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        values = [class_metrics[m] for m in metrics]
        ax1.bar(metrics, values, alpha=0.7, color='skyblue')
        ax1.set_title('分類性能指標')
        ax1.set_ylabel('分數')
        ax1.set_ylim(0, 1)
        
        for i, v in enumerate(values):
            ax1.text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        # 檢測指標
        det_metrics = ['objectness_accuracy', 'mean_iou']
        det_values = [detection_metrics[m] for m in det_metrics]
        ax2.bar(det_metrics, det_values, alpha=0.7, color='lightcoral')
        ax2.set_title('檢測性能指標')
        ax2.set_ylabel('分數')
        ax2.set_ylim(0, 1)
        
        for i, v in enumerate(det_values):
            ax2.text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'detection_metrics.png', dpi=300)
        plt.close()
        
        # 2. IoU分布
        if 'predictions' in results and 'targets' in results:
            pred_boxes = results['predictions']['boxes']
            true_boxes = results['targets']['boxes']
            
            ious = []
            for pred_box, true_box in zip(pred_boxes, true_boxes):
                if np.any(true_box > 0):
                    iou = self._calculate_iou(pred_box, true_box)
                    ious.append(iou)
            
            if ious:
                plt.figure(figsize=(10, 6))
                plt.hist(ious, bins=30, alpha=0.7, edgecolor='black')
                plt.title('IoU分布')
                plt.xlabel('IoU')
                plt.ylabel('頻率')
                plt.axvline(np.mean(ious), color='red', linestyle='--', 
                           label=f'平均IoU: {np.mean(ious):.3f}')
                plt.legend()
                plt.tight_layout()
                plt.savefig(plots_dir / 'iou_distribution.png', dpi=300)
                plt.close()
    
    def save_results(self, results: Dict):
        """保存評估結果"""
        if not self.config["evaluation"]["save_detailed_results"]:
            return
        
        results_dir = Path(self.config["output"]["results_dir"])
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存詳細結果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"evaluation_results_{timestamp}.json"
        
        # 轉換numpy arrays為列表以便JSON序列化
        serializable_results = self._make_json_serializable(results)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"評估結果已保存: {results_file}")
        
        # 生成簡要報告
        self._generate_summary_report(results, results_dir / f"evaluation_summary_{timestamp}.txt")
    
    def _make_json_serializable(self, obj):
        """轉換對象為JSON可序列化格式"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
    
    def _generate_summary_report(self, results: Dict, report_path: Path):
        """生成摘要報告"""
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== CT-ViT 模型評估報告 ===\n\n")
            f.write(f"評估時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"模型類型: {results['model_type']}\n\n")
            
            if results["model_type"] == "classification":
                metrics = results["overall_metrics"]
                f.write("=== 分類性能 ===\n")
                f.write(f"準確度: {metrics['accuracy']:.4f}\n")
                f.write(f"精確度: {metrics['precision']:.4f}\n")
                f.write(f"召回率: {metrics['recall']:.4f}\n")
                f.write(f"F1分數: {metrics['f1_score']:.4f}\n\n")
                
                if "per_class_metrics" in results:
                    f.write("=== 各類別性能 ===\n")
                    for class_name, class_metrics in results["per_class_metrics"].items():
                        f.write(f"\n{class_name}:\n")
                        f.write(f"  精確度: {class_metrics['precision']:.4f}\n")
                        f.write(f"  召回率: {class_metrics['recall']:.4f}\n")
                        f.write(f"  F1分數: {class_metrics['f1_score']:.4f}\n")
                        f.write(f"  樣本數: {class_metrics['support']}\n")
            
            elif results["model_type"] == "detection":
                class_metrics = results["classification_metrics"]
                detection_metrics = results["detection_metrics"]
                
                f.write("=== 分類性能 ===\n")
                f.write(f"準確度: {class_metrics['accuracy']:.4f}\n")
                f.write(f"精確度: {class_metrics['precision']:.4f}\n")
                f.write(f"召回率: {class_metrics['recall']:.4f}\n")
                f.write(f"F1分數: {class_metrics['f1_score']:.4f}\n\n")
                
                f.write("=== 檢測性能 ===\n")
                f.write(f"目標檢測準確度: {detection_metrics['objectness_accuracy']:.4f}\n")
                f.write(f"平均IoU: {detection_metrics['mean_iou']:.4f}\n")
                f.write(f"有效邊界框數量: {detection_metrics['num_valid_boxes']}\n")
        
        self.logger.info(f"摘要報告已生成: {report_path}")
    
    def run_evaluation(self, model_path: str, data_path: str, model_type: str = "classification"):
        """運行完整評估"""
        self.logger.info("開始模型評估...")
        self.logger.info(f"模型類型: {model_type}")
        self.logger.info(f"模型路徑: {model_path}")
        self.logger.info(f"測試資料: {data_path}")
        
        # 載入模型
        if model_type == "classification":
            model = self.load_classification_model(model_path)
            if model is None:
                self.logger.error("無法載入分類模型")
                return None
            
            # 載入測試資料集 (這裡需要實現具體的資料載入邏輯)
            test_dataset = self._load_classification_dataset(data_path)
            if test_dataset is None:
                self.logger.error("無法載入測試資料集")
                return None
            
            # 評估
            results = self.evaluate_classification_model(model, test_dataset)
            
        elif model_type == "detection":
            model = self.load_detection_model(model_path)
            if model is None:
                self.logger.error("無法載入檢測模型")
                return None
            
            # 載入檢測測試資料集
            test_dataset = self._load_detection_dataset(data_path)
            if test_dataset is None:
                self.logger.error("無法載入檢測測試資料集")
                return None
            
            # 評估
            results = self.evaluate_detection_model(model, test_dataset)
        
        else:
            self.logger.error(f"不支援的模型類型: {model_type}")
            return None
        
        # 創建視覺化
        self.create_visualizations(results)
        
        # 保存結果
        self.save_results(results)
        
        # 顯示摘要
        self._print_summary(results)
        
        return results
    
    def _load_classification_dataset(self, data_path: str):
        """載入分類資料集"""
        # 這裡需要實現具體的分類資料集載入邏輯
        # 暫時返回None，需要根據實際資料格式實現
        self.logger.warning("分類資料集載入功能需要實現")
        return None
    
    def _load_detection_dataset(self, data_path: str):
        """載入檢測資料集"""
        if not HAS_DETECTION_SUPPORT:
            return None
        
        try:
            # 使用檢測資料集類
            dataset = CTDetectionDataset(
                data_path,
                image_size=self.config["data"]["image_size"],
                is_training=False
            )
            return dataset
        except Exception as e:
            self.logger.error(f"載入檢測資料集失敗: {e}")
            return None
    
    def _print_summary(self, results: Dict):
        """打印評估摘要"""
        print("\n" + "="*60)
        print("📊 評估結果摘要")
        print("="*60)
        
        if results["model_type"] == "classification":
            metrics = results["overall_metrics"]
            print(f"模型類型: 分類模型")
            print(f"準確度: {metrics['accuracy']:.4f}")
            print(f"精確度: {metrics['precision']:.4f}")
            print(f"召回率: {metrics['recall']:.4f}")
            print(f"F1分數: {metrics['f1_score']:.4f}")
            
        elif results["model_type"] == "detection":
            class_metrics = results["classification_metrics"]
            detection_metrics = results["detection_metrics"]
            
            print(f"模型類型: 檢測模型")
            print(f"分類準確度: {class_metrics['accuracy']:.4f}")
            print(f"檢測準確度: {detection_metrics['objectness_accuracy']:.4f}")
            print(f"平均IoU: {detection_metrics['mean_iou']:.4f}")
        
        print("="*60)


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='CT-ViT 統一評估系統')
    parser.add_argument('--model_path', type=str, required=True, help='模型檔案路徑')
    parser.add_argument('--data_path', type=str, required=True, help='測試資料路徑')
    parser.add_argument('--model_type', choices=['classification', 'detection'], 
                       default='classification', help='模型類型')
    parser.add_argument('--config', type=str, help='配置檔案路徑')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', 
                       help='輸出目錄')
    
    args = parser.parse_args()
    
    try:
        # 創建評估器
        evaluator = ModelEvaluator(args.config)
        
        # 更新輸出目錄
        if args.output_dir:
            evaluator.config["output"]["results_dir"] = args.output_dir
            evaluator.config["output"]["plots_dir"] = os.path.join(args.output_dir, "plots")
        
        # 運行評估
        results = evaluator.run_evaluation(
            model_path=args.model_path,
            data_path=args.data_path,
            model_type=args.model_type
        )
        
        if results:
            print("✅ 評估完成!")
        else:
            print("❌ 評估失敗!")
            
    except Exception as e:
        print(f"❌ 評估過程出現錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
