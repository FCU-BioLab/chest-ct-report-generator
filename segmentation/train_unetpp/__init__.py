#!/usr/bin/env python3
"""
UNet++ 肺病灶分割訓練模組
=========================

支援資料集:
    - LNDb: 肺結節分割 (236 病人)
    - MSD: 肺腫瘤分割 (Task06, 64+32 案例)

使用方式:
    # LNDb 訓練
    python -m segmentation.train_unetpp.main --dataset lndb --epochs 100
    
    # MSD 訓練
    python -m segmentation.train_unetpp.main --dataset msd --epochs 100
    
    # 查看所有選項
    python -m segmentation.train_unetpp.main --help
"""

from .config import Config, get_default_config
from .model import get_model, create_unetpp_model, count_parameters
from .losses import get_loss_function, BCEDiceLoss, AdaptiveLoss
from .trainer import UNetPPTrainer, MetricsCalculator
from .inference import Inferencer, load_model_for_inference, NoduleExtractor
from .dataset import LNDbSliceDataset, get_patient_split, get_fold_split, val_collate_fn
from .msd_dataset import MSDLungSliceDataset, get_msd_lung_cases, get_msd_train_val_split
from .preprocess import CTPreprocessor, preprocess_lndb_slices
from .utils import setup_logging, set_seed, get_device, plot_training_history, custom_collate_fn
from .patch_utils import compute_4patch_positions, extract_patch_with_lung_mask, stitch_4patches

__all__ = [
    # Config
    'Config',
    'get_default_config',
    # Model
    'get_model',
    'create_unetpp_model',
    'count_parameters',
    # Loss
    'get_loss_function',
    'BCEDiceLoss',
    'AdaptiveLoss',
    # Trainer
    'UNetPPTrainer',
    'MetricsCalculator',
    # Inference
    'Inferencer',
    'load_model_for_inference',
    'NoduleExtractor',
    # LNDb Dataset
    'LNDbSliceDataset',
    'get_patient_split',
    'get_fold_split',
    'val_collate_fn',
    # MSD Dataset
    'MSDLungSliceDataset',
    'get_msd_lung_cases',
    'get_msd_train_val_split',

    # Patch Utils
    'compute_4patch_positions',
    'extract_patch_with_lung_mask',
    'stitch_4patches',
    # Preprocess
    'CTPreprocessor',
    'preprocess_lndb_slices',
    # Utils
    'setup_logging',
    'set_seed',
    'get_device',
    'plot_training_history',
    'custom_collate_fn',
]
