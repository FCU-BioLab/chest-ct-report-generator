#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CT-ViT 訓練腳本
胸部CT影像Vision Transformer模型訓練、驗證和測試

此腳本實作了完整的CT-ViT訓練流程，包含：
1. DICOM影像載入和預處理
2. Vision Transformer模型架構
3. 訓練、驗證、測試循環
4. 模型保存和評估

作者: GitHub Copilot
日期: 2025-07-22
"""

import os
import sys
import json
import time
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# 添加src路徑到Python路徑
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import TrainingArguments

# 導入自定義模組
from config import CTViTConfig
from data_processing import DICOMProcessor, CTDataset
from model import CTViTTrainer
from utils import setup_logging, compute_metrics

# 忽略警告
warnings.filterwarnings('ignore')

# === 主訓練類 ===
class CTViTTrainingPipeline:
    """CT-ViT訓練流水線"""
    
    def __init__(self, config: CTViTConfig):
        self.config = config
        config.create_directories()
        
        # 設置日誌
        self.logger = setup_logging(config.log_dir)
        
        # 設置隨機種子
        self._set_random_seeds()
        
        # 載入資料集
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # 模型相關
        self.model = None
        self.image_processor = None
        self.trainer = None
        
        # TensorBoard
        self.writer = SummaryWriter(config.tensorboard_dir)
        
        self.logger.info("CT-ViT訓練流水線初始化完成")
    
    def _set_random_seeds(self):
        """設置隨機種子"""
        torch.manual_seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.random_seed)
            torch.cuda.manual_seed_all(self.config.random_seed)
    
    def load_datasets(self):
        """載入資料集"""
        self.logger.info("載入資料集...")
        
        try:
            self.train_dataset = CTDataset(self.config.train_dir, self.config, "train")
            self.val_dataset = CTDataset(self.config.val_dir, self.config, "validation")
            self.test_dataset = CTDataset(self.config.test_dir, self.config, "test")
            
            self.logger.info(f"訓練集: {len(self.train_dataset)} 樣本")
            self.logger.info(f"驗證集: {len(self.val_dataset)} 樣本")
            self.logger.info(f"測試集: {len(self.test_dataset)} 樣本")
            
            # 統計類別分佈
            self._log_class_distribution()
            
        except Exception as e:
            self.logger.error(f"載入資料集失敗: {str(e)}")
            raise
    
    def _log_class_distribution(self):
        """記錄類別分佈"""
        for dataset_name, dataset in [("訓練", self.train_dataset), ("驗證", self.val_dataset), ("測試", self.test_dataset)]:
            class_counts = {}
            for item in dataset.data_list:
                label = item['label']
                class_counts[label] = class_counts.get(label, 0) + 1
            
            self.logger.info(f"{dataset_name}集類別分佈: {class_counts}")
    
    def setup_model(self):
        """設置模型"""
        self.logger.info("設置模型...")
        
        try:
            # 載入預訓練的ViT模型
            self.model = ViTForImageClassification.from_pretrained(
                self.config.model_name,
                num_labels=self.config.num_labels,
                ignore_mismatched_sizes=True
            )
            
            # 載入圖像處理器
            self.image_processor = ViTImageProcessor.from_pretrained(self.config.model_name)
            
            self.logger.info(f"模型載入成功: {self.config.model_name}")
            self.logger.info(f"模型參數數量: {sum(p.numel() for p in self.model.parameters()):,}")
            
        except Exception as e:
            self.logger.error(f"模型設置失敗: {str(e)}")
            raise
    
    def setup_trainer(self):
        """設置訓練器"""
        self.logger.info("設置訓練器...")
        
        training_args = TrainingArguments(
            output_dir=self.config.model_save_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            fp16=self.config.fp16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            max_grad_norm=self.config.max_grad_norm,
            dataloader_num_workers=self.config.num_workers,
            remove_unused_columns=False,
            report_to="tensorboard",  # 只使用tensorboard，不使用wandb
            logging_dir=self.config.tensorboard_dir,
        )
        
        self.trainer = CTViTTrainer(
            config=self.config,
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=compute_metrics,
            logger=self.logger,
        )
        
        self.logger.info("訓練器設置完成")
    
    def train(self):
        """執行訓練"""
        self.logger.info("開始訓練...")
        
        try:
            # 開始訓練
            train_result = self.trainer.train()
            
            # 保存最終模型
            final_model_path = os.path.join(self.config.model_save_dir, "final_model")
            self.trainer.save_model(final_model_path)
            self.image_processor.save_pretrained(final_model_path)
            
            # 記錄訓練結果
            self.logger.info(f"訓練完成！最終損失: {train_result.training_loss:.4f}")
            
            # 保存訓練日誌
            self._save_training_log(train_result)
            
        except Exception as e:
            self.logger.error(f"訓練失敗: {str(e)}")
            raise
    
    def evaluate(self, dataset: Dataset = None, split_name: str = "validation") -> Dict:
        """評估模型"""
        if dataset is None:
            dataset = self.val_dataset
        
        self.logger.info(f"評估模型 ({split_name})...")
        
        # 執行評估
        eval_result = self.trainer.evaluate(eval_dataset=dataset)
        
        # 詳細分析
        predictions = self.trainer.predict(dataset)
        y_true = predictions.label_ids
        y_pred = np.argmax(predictions.predictions, axis=1)
        
        # 動態獲取實際存在的類別
        unique_labels = sorted(list(set(y_true.tolist() + y_pred.tolist())))
        class_names = [['A_series', 'B_series', 'E_series', 'G_series'][i] for i in unique_labels]
        
        # 分類報告 - 只針對實際存在的類別
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
            'metrics': eval_result,
            'classification_report': classification_rep,
            'confusion_matrix': cm.tolist(),
            'predictions': {
                'y_true': y_true.tolist(),
                'y_pred': y_pred.tolist()
            }
        }
        
        # 保存到文件
        result_file = os.path.join(self.config.output_dir, f"evaluation_{split_name}.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 繪製混淆矩陣
        self._plot_confusion_matrix(cm, class_names, split_name)
        
        self.logger.info(f"{split_name} 評估結果:")
        self.logger.info(f"  準確率: {eval_result['eval_accuracy']:.4f}")
        self.logger.info(f"  F1分數: {eval_result['eval_f1']:.4f}")
        
        return results
    
    def test(self) -> Dict:
        """測試模型"""
        return self.evaluate(self.test_dataset, "test")
    
    def _plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str], split_name: str):
        """繪製混淆矩陣"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {split_name.title()}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # 保存圖片
        plt.savefig(os.path.join(self.config.output_dir, f'confusion_matrix_{split_name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_training_log(self, train_result):
        """保存訓練日誌"""
        log_data = {
            'config': self.config.__dict__,
            'training_loss': train_result.training_loss,
            'training_time': train_result.metrics.get('train_runtime', 0),
            'train_samples_per_second': train_result.metrics.get('train_samples_per_second', 0),
            'model_info': {
                'model_name': self.config.model_name,
                'num_parameters': sum(p.numel() for p in self.model.parameters()),
                'num_trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            }
        }
        
        log_file = os.path.join(self.config.output_dir, "training_log.json")
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    def run_full_pipeline(self):
        """執行完整的訓練流水線"""
        self.logger.info("=== 開始CT-ViT完整訓練流水線 ===")
        
        start_time = time.time()
        
        try:
            # 1. 載入資料集
            self.load_datasets()
            
            # 2. 設置模型
            self.setup_model()
            
            # 3. 設置訓練器
            self.setup_trainer()
            
            # 4. 訓練
            self.train()
            
            # 5. 驗證
            self.evaluate(self.val_dataset, "validation")
            
            # 6. 測試
            self.test()
            
            end_time = time.time()
            total_time = end_time - start_time
            
            self.logger.info(f"=== 訓練流水線完成! 總耗時: {total_time/3600:.2f} 小時 ===")
            
        except Exception as e:
            self.logger.error(f"訓練流水線失敗: {str(e)}")
            raise
        finally:
            self.writer.close()

# === 主函數 ===
def main():
    """主函數"""
    print("=== CT-ViT 胸部CT影像分類模型訓練 ===\n")
    
    # 創建配置
    config = CTViTConfig()
    
    # 顯示配置
    print("訓練配置:")
    print(f"  資料集路徑: {config.dataset_root}")
    print(f"  輸出路徑: {config.output_dir}")
    print(f"  模型: {config.model_name}")
    print(f"  影像大小: {config.image_size}x{config.image_size}")
    print(f"  批次大小: {config.batch_size}")
    print(f"  學習率: {config.learning_rate}")
    print(f"  訓練輪數: {config.num_epochs}")
    print(f"  類別數量: {config.num_labels}")
    
    # 確認執行
    user_input = input("\n是否開始訓練? (y/N): ")
    if user_input.lower() != 'y':
        print("訓練已取消")
        return
    
    # 檢查GPU
    if torch.cuda.is_available():
        print(f"\n使用GPU: {torch.cuda.get_device_name()}")
        print(f"GPU記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("\n使用CPU (警告: 訓練速度會很慢)")
    
    try:
        # 創建並執行訓練流水線
        pipeline = CTViTTrainingPipeline(config)
        pipeline.run_full_pipeline()
        
        print("\n🎉 CT-ViT訓練完成！")
        print(f"模型保存位置: {config.model_save_dir}")
        print(f"日誌位置: {config.log_dir}")
        print(f"TensorBoard: tensorboard --logdir {config.tensorboard_dir}")
        
    except KeyboardInterrupt:
        print("\n訓練被使用者中斷")
    except Exception as e:
        print(f"\n訓練失敗: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
