#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CT-ViT 配置模組
包含所有訓練相關的配置設定

作者: GitHub Copilot
日期: 2025-07-22
"""

import os
from pathlib import Path

class CTViTConfig:
    """CT-ViT訓練配置"""
    
    def __init__(self):
        # 資料路徑
        self.dataset_root = "D:/GitHub/chest-ct-report-generator/dataset_splits"
        self.train_dir = os.path.join(self.dataset_root, "train")
        self.val_dir = os.path.join(self.dataset_root, "validation") 
        self.test_dir = os.path.join(self.dataset_root, "test")
        
        # 輸出路徑
        self.output_dir = "D:/GitHub/chest-ct-report-generator/CT_ViT"
        self.model_save_dir = os.path.join(self.output_dir, "models")
        self.log_dir = os.path.join(self.output_dir, "logs")
        self.tensorboard_dir = os.path.join(self.output_dir, "tensorboard")
        
        # 模型參數
        self.model_name = "google/vit-base-patch16-224"
        self.image_size = 224
        self.patch_size = 16
        self.num_labels = 4  # A, B, E, G 四種類型
        self.hidden_size = 768
        self.num_hidden_layers = 12
        self.num_attention_heads = 12
        
        # 訓練參數
        self.batch_size = 8
        self.learning_rate = 2e-5
        self.num_epochs = 50
        self.weight_decay = 0.01
        self.warmup_steps = 500
        self.save_steps = 1000
        self.eval_steps = 500
        self.logging_steps = 100
        
        # 資料增強參數
        self.use_augmentation = True
        self.rotation_range = 15
        self.zoom_range = 0.1
        self.brightness_range = 0.2
        self.contrast_range = 0.2
        
        # DICOM處理參數
        self.window_center = 40  # 肺窗
        self.window_width = 400
        self.normalize_hounsfield = True
        self.slice_selection_method = "middle"  # "middle", "random", "all"
        
        # 其他參數
        self.random_seed = 42
        self.num_workers = 4
        self.fp16 = True
        self.gradient_checkpointing = True
        self.max_grad_norm = 1.0

    def create_directories(self):
        """創建必要的目錄"""
        for dir_path in [self.output_dir, self.model_save_dir, self.log_dir, self.tensorboard_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def to_dict(self):
        """轉換為字典格式"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def from_dict(self, config_dict):
        """從字典載入配置"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def save_config(self, file_path: str):
        """保存配置到文件"""
        import json
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def load_config(self, file_path: str):
        """從文件載入配置"""
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
            self.from_dict(config_dict)
