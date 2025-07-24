#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CT-ViT 工具模組
包含日誌設置、評估指標等工具函數

作者: GitHub Copilot
日期: 2025-07-22
"""

import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers.trainer_utils import EvalPrediction

# === 設置日誌 ===
def setup_logging(log_dir: str) -> logging.Logger:
    """設置日誌記錄"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"ct_vit_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

# === 評估指標計算 ===
def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """計算評估指標"""
    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = np.argmax(predictions, axis=1)
    
    # 計算指標
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
