#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Medical CT Tumor Detection - Configuration Presets

This file provides pre-configured settings for different scenarios in medical CT tumor detection.
"""

from typing import Dict, Any


class MedicalCTConfigs:
    """Pre-configured training settings for medical CT tumor detection."""
    
    @staticmethod
    def get_baseline_config() -> Dict[str, Any]:
        """
        Baseline configuration for initial training.
        Good starting point for most medical CT datasets.
        """
        return {
            "num_epochs": 100,
            "batch_size": 16,
            "learning_rate": 0.01,
            "imgsz": 640,
            "model_size": "m",
            "optimizer": "AdamW",
            "cos_lr": True,
            "warmup_epochs": 3,
            "use_medical_preprocessing": True,
            "hu_window": (-1000, 400),
            "pseudo_3d_slices": 3,
            "use_augmentation": True,
            "use_focal_loss": True,
            "mosaic": 1.0,
            "mixup": 0.1,
            "copy_paste": 0.1,
            "multi_scale": True,
            "conf_threshold": 0.25,
            "iou_threshold": 0.45,
        }
    
    @staticmethod
    def get_small_tumor_config() -> Dict[str, Any]:
        """
        Optimized configuration for detecting small tumors.
        Increases resolution and uses larger model for better feature extraction.
        """
        return {
            "num_epochs": 150,
            "batch_size": 8,  # Reduced due to larger image size
            "learning_rate": 0.008,
            "imgsz": 800,  # Higher resolution for small objects
            "model_size": "l",  # Larger model
            "optimizer": "AdamW",
            "cos_lr": True,
            "warmup_epochs": 5,
            "use_medical_preprocessing": True,
            "hu_window": (-1000, 400),
            "pseudo_3d_slices": 3,
            "use_augmentation": True,
            "use_focal_loss": True,
            "mosaic": 1.0,  # Strong mosaic for small objects
            "mixup": 0.2,   # Increased mixup
            "copy_paste": 0.3,  # More copy-paste for small objects
            "multi_scale": True,
            "scale_range": (0.5, 1.5),  # Wider scale range
            "conf_threshold": 0.20,  # Lower threshold for small objects
            "iou_threshold": 0.40,
            "weight_decay": 0.0005,
        }
    
    @staticmethod
    def get_high_precision_config() -> Dict[str, Any]:
        """
        Configuration optimized for high precision (low false positives).
        Suitable for clinical scenarios where false positives are costly.
        """
        return {
            "num_epochs": 120,
            "batch_size": 16,
            "learning_rate": 0.01,
            "imgsz": 640,
            "model_size": "l",
            "optimizer": "AdamW",
            "cos_lr": True,
            "warmup_epochs": 3,
            "use_medical_preprocessing": True,
            "hu_window": (-1000, 400),
            "pseudo_3d_slices": 3,
            "use_augmentation": True,
            "use_focal_loss": True,
            "focal_loss_gamma": 2.5,  # Higher gamma for focal loss
            "mosaic": 0.8,  # Reduced to maintain image quality
            "mixup": 0.05,
            "copy_paste": 0.05,
            "multi_scale": False,
            "conf_threshold": 0.35,  # Higher threshold
            "iou_threshold": 0.50,   # More aggressive NMS
            "include_negative_samples": True,
            "max_negative_per_patient": 30,  # More negative samples
        }
    
    @staticmethod
    def get_high_recall_config() -> Dict[str, Any]:
        """
        Configuration optimized for high recall (detect all tumors).
        Suitable for screening scenarios where missing a tumor is critical.
        """
        return {
            "num_epochs": 150,
            "batch_size": 16,
            "learning_rate": 0.01,
            "imgsz": 640,
            "model_size": "x",  # Largest model
            "optimizer": "AdamW",
            "cos_lr": True,
            "warmup_epochs": 5,
            "use_medical_preprocessing": True,
            "hu_window": (-1000, 400),
            "pseudo_3d_slices": 3,
            "use_augmentation": True,
            "use_focal_loss": True,
            "mosaic": 1.0,
            "mixup": 0.15,
            "copy_paste": 0.15,
            "multi_scale": True,
            "scale_range": (0.5, 1.5),
            "conf_threshold": 0.15,  # Lower threshold
            "iou_threshold": 0.40,   # Less aggressive NMS
            "max_det": 500,  # Allow more detections
        }
    
    @staticmethod
    def get_fast_training_config() -> Dict[str, Any]:
        """
        Configuration for fast prototyping and experimentation.
        Uses smaller model and fewer epochs.
        """
        return {
            "num_epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.01,
            "imgsz": 512,
            "model_size": "s",
            "optimizer": "AdamW",
            "cos_lr": True,
            "warmup_epochs": 2,
            "use_medical_preprocessing": True,
            "hu_window": (-1000, 400),
            "pseudo_3d_slices": 0,  # Disable pseudo-3D for speed
            "use_augmentation": True,
            "use_focal_loss": False,
            "mosaic": 1.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "multi_scale": False,
            "conf_threshold": 0.25,
            "iou_threshold": 0.45,
        }
    
    @staticmethod
    def get_production_config() -> Dict[str, Any]:
        """
        Production-ready configuration with balanced performance.
        Suitable for deployment after validation.
        """
        return {
            "num_epochs": 200,
            "batch_size": 16,
            "learning_rate": 0.01,
            "imgsz": 640,
            "model_size": "l",
            "optimizer": "AdamW",
            "cos_lr": True,
            "warmup_epochs": 5,
            "weight_decay": 0.0005,
            "momentum": 0.937,
            "use_medical_preprocessing": True,
            "hu_window": (-1000, 400),
            "pseudo_3d_slices": 3,
            "use_augmentation": True,
            "use_focal_loss": True,
            "focal_loss_gamma": 2.0,
            "mosaic": 1.0,
            "mixup": 0.1,
            "copy_paste": 0.1,
            "multi_scale": True,
            "scale_range": (0.5, 1.5),
            "conf_threshold": 0.25,
            "iou_threshold": 0.45,
            "max_det": 300,
            "include_negative_samples": True,
            "max_negative_per_patient": 20,
        }


def print_config_comparison():
    """Print comparison of all available configurations."""
    configs = MedicalCTConfigs()
    config_names = [
        "baseline",
        "small_tumor",
        "high_precision",
        "high_recall",
        "fast_training",
        "production",
    ]
    
    print("=" * 100)
    print("Medical CT Configuration Presets Comparison")
    print("=" * 100)
    print()
    
    for name in config_names:
        method_name = f"get_{name}_config"
        if hasattr(configs, method_name):
            config = getattr(configs, method_name)()
            print(f"\n{name.upper().replace('_', ' ')} Configuration:")
            print("-" * 100)
            
            # Key parameters
            key_params = [
                "model_size", "imgsz", "num_epochs", "batch_size",
                "conf_threshold", "optimizer", "use_focal_loss"
            ]
            
            for key in key_params:
                if key in config:
                    print(f"  {key:25s}: {config[key]}")
            
            print()


if __name__ == "__main__":
    print_config_comparison()
    
    print("\n" + "=" * 100)
    print("Usage Example:")
    print("=" * 100)
    print("""
from train_yolo_optimize import train_yolov11
from medical_ct_config import MedicalCTConfigs

# Get configuration
config = MedicalCTConfigs.get_small_tumor_config()

# Add dataset path
config['data_dir'] = './datasets/ct_tumor_data'

# Start training
summary = train_yolov11(**config)

# Print results
print(f"Training completed: {summary['metrics']}")
    """)
