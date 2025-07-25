#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CT-ViT 模型評估腳本
對已訓練的模型進行驗證和測試評估

作者: GitHub Copilot
日期: 2025-07-24
"""

import os
import sys
import json
import logging
from pathlib import Path

# 禁用wandb
os.environ["WANDB_DISABLED"] = "true"

# 添加src路徑到Python路徑
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from transformers import ViTImageProcessor, ViTForImageClassification
from torch.utils.data import DataLoader

from config import CTViTConfig
from data_processing import CTDataset
from utils import setup_logging

def main():
    """主評估函數"""
    print("=== CT-ViT 模型評估 ===\n")
    
    # 創建配置
    config = CTViTConfig()
    
    # 設置日誌
    logger = setup_logging(config.log_dir)
    
    # 載入資料集
    logger.info("載入資料集...")
    val_dataset = CTDataset(config.val_dir, config, "validation")
    test_dataset = CTDataset(config.test_dir, config, "test")
    
    logger.info(f"驗證集: {len(val_dataset)} 樣本")
    logger.info(f"測試集: {len(test_dataset)} 樣本")
    
    # 檢查是否有已訓練的模型
    final_model_path = os.path.join(config.model_save_dir, "final_model")
    if not os.path.exists(final_model_path):
        print(f"找不到已訓練的模型: {final_model_path}")
        print("請先完成訓練")
        return
    
    # 載入已訓練的模型
    logger.info("載入已訓練的模型...")
    model = ViTForImageClassification.from_pretrained(final_model_path)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    def evaluate_dataset_simple(dataset, split_name):
        logger.info(f"評估模型 ({split_name})...")
        
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(pixel_values)
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)
        
        # 計算指標
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        
        # 動態獲取實際存在的類別
        unique_labels = sorted(list(set(y_true.tolist() + y_pred.tolist())))
        class_names = [['A_series', 'B_series', 'E_series', 'G_series'][i] for i in unique_labels]
        
        # 分類報告
        classification_rep = classification_report(
            y_true, y_pred, 
            labels=unique_labels,
            target_names=class_names, 
            output_dict=True,
            zero_division=0
        )
        
        # 混淆矩陣
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
        
        # 保存結果
        results = {
            'metrics': {
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall
            },
            'classification_report': classification_rep,
            'confusion_matrix': cm.tolist(),
            'predictions': {
                'y_true': y_true.tolist(),
                'y_pred': y_pred.tolist()
            }
        }
        
        # 保存到文件
        result_file = os.path.join(config.output_dir, f"evaluation_{split_name}.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 繪製混淆矩陣
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {split_name.title()}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # 保存圖片
        plt.savefig(os.path.join(config.output_dir, f'confusion_matrix_{split_name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"{split_name} 評估結果:")
        logger.info(f"  準確率: {accuracy:.4f}")
        logger.info(f"  F1分數: {f1:.4f}")
        logger.info(f"  精確度: {precision:.4f}")
        logger.info(f"  召回率: {recall:.4f}")
        
        return results
    
    try:
        # 執行驗證
        val_results = evaluate_dataset_simple(val_dataset, "validation")
        
        # 執行測試
        test_results = evaluate_dataset_simple(test_dataset, "test")
        
        print("\n🎉 評估完成！")
        print(f"結果保存位置: {config.output_dir}")
        print(f"驗證集準確率: {val_results['metrics']['accuracy']:.4f}")
        print(f"測試集準確率: {test_results['metrics']['accuracy']:.4f}")
        
    except Exception as e:
        logger.error(f"評估失敗: {str(e)}")
        raise

if __name__ == "__main__":
    main()
