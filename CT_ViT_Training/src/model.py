#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CT-ViT 模型模組
包含自定義的訓練器和模型相關功能

作者: GitHub Copilot
日期: 2025-07-22
"""

import logging
import torch
import torch.nn as nn
from transformers import Trainer

from config import CTViTConfig

# === 自定義Trainer ===
class CTViTTrainer(Trainer):
    """自定義的CT-ViT訓練器"""
    
    def __init__(self, config: CTViTConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.logger = kwargs.get('logger', logging.getLogger(__name__))
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """計算損失"""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(outputs.logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss
