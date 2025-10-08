#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example Usage: YOLOv8 Medical CT Tumor Detection

This script demonstrates various training scenarios for medical CT tumor detection.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from train_yolo_optimize import train_yolov11, DatasetCache, clear_dataset_cache
from medical_ct_config import MedicalCTConfigs


def example_1_basic_training():
    """Example 1: Basic training with default settings."""
    print("\n" + "=" * 80)
    print("Example 1: Basic Training")
    print("=" * 80)
    
    config = {
        "data_dir": "./datasets/ct_tumor_data",
        "num_epochs": 50,  # Reduced for demo
        "batch_size": 16,
        "model_size": "s",  # Small model for fast training
        "imgsz": 512,
        "save_dir": "./experiments/example_1_basic",
    }
    
    summary = train_yolov11(**config)
    print(f"\nTraining completed!")
    print(f"mAP@0.5: {summary['metrics'].get('mAP@0.5', 0):.3f}")
    print(f"Model saved to: {summary['model_path']}")
    
    return summary


def example_2_small_tumor_detection():
    """Example 2: Optimized for small tumor detection."""
    print("\n" + "=" * 80)
    print("Example 2: Small Tumor Detection")
    print("=" * 80)
    
    # Use pre-configured settings
    config = MedicalCTConfigs.get_small_tumor_config()
    config.update({
        "data_dir": "./datasets/ct_tumor_data",
        "num_epochs": 100,  # Reduced from 150 for demo
        "save_dir": "./experiments/example_2_small_tumor",
    })
    
    summary = train_yolov11(**config)
    print(f"\nTraining completed!")
    print(f"mAP@0.5: {summary['metrics'].get('mAP@0.5', 0):.3f}")
    print(f"Recall: {summary['metrics'].get('recall', 0):.3f}")
    
    return summary


def example_3_high_precision():
    """Example 3: High precision configuration (reduce false positives)."""
    print("\n" + "=" * 80)
    print("Example 3: High Precision Training")
    print("=" * 80)
    
    config = MedicalCTConfigs.get_high_precision_config()
    config.update({
        "data_dir": "./datasets/ct_tumor_data",
        "num_epochs": 80,  # Reduced for demo
        "save_dir": "./experiments/example_3_high_precision",
    })
    
    summary = train_yolov11(**config)
    print(f"\nTraining completed!")
    print(f"Precision: {summary['metrics'].get('precision', 0):.3f}")
    print(f"mAP@0.5: {summary['metrics'].get('mAP@0.5', 0):.3f}")
    
    return summary


def example_4_multiple_experiments_with_cache():
    """Example 4: Multiple experiments with shared dataset cache."""
    print("\n" + "=" * 80)
    print("Example 4: Multiple Experiments with Shared Cache")
    print("=" * 80)
    
    # Create shared cache
    cache = DatasetCache(cache_dir="./experiments/shared_cache")
    
    # Define different configurations to test
    experiments = [
        {
            "name": "yolov8s_512",
            "config": {
                "model_size": "s",
                "imgsz": 512,
                "num_epochs": 30,
            }
        },
        {
            "name": "yolov8m_640",
            "config": {
                "model_size": "m",
                "imgsz": 640,
                "num_epochs": 30,
            }
        },
        {
            "name": "yolov8m_with_focal",
            "config": {
                "model_size": "m",
                "imgsz": 640,
                "num_epochs": 30,
                "use_focal_loss": True,
            }
        },
    ]
    
    results = []
    
    for exp in experiments:
        print(f"\n{'=' * 80}")
        print(f"Running Experiment: {exp['name']}")
        print(f"{'=' * 80}")
        
        config = {
            "data_dir": "./datasets/ct_tumor_data",
            "batch_size": 16,
            "save_dir": f"./experiments/example_4_{exp['name']}",
            "dataset_cache": cache,  # Share cache across experiments
            **exp['config']
        }
        
        summary = train_yolov11(**config)
        
        results.append({
            "name": exp['name'],
            "metrics": summary['metrics'],
            "model_path": summary['model_path'],
        })
        
        print(f"\n{exp['name']} completed:")
        print(f"  mAP@0.5: {summary['metrics'].get('mAP@0.5', 0):.3f}")
        print(f"  F1 Score: {summary['metrics'].get('f1_score', 0):.3f}")
        print(f"  Cache datasets: {cache.cache_size()}")
    
    # Compare results
    print(f"\n{'=' * 80}")
    print("Experiment Comparison")
    print(f"{'=' * 80}")
    print(f"{'Experiment':<25} {'mAP@0.5':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"{'-' * 80}")
    
    for result in results:
        metrics = result['metrics']
        print(f"{result['name']:<25} "
              f"{metrics.get('mAP@0.5', 0):>10.3f} "
              f"{metrics.get('precision', 0):>10.3f} "
              f"{metrics.get('recall', 0):>10.3f} "
              f"{metrics.get('f1_score', 0):>10.3f}")
    
    return results


def example_5_custom_configuration():
    """Example 5: Custom configuration for specific requirements."""
    print("\n" + "=" * 80)
    print("Example 5: Custom Configuration")
    print("=" * 80)
    
    # Start from baseline and customize
    config = MedicalCTConfigs.get_baseline_config()
    
    # Customize for your specific needs
    config.update({
        "data_dir": "./datasets/ct_tumor_data",
        "num_epochs": 80,
        "batch_size": 16,
        "model_size": "m",
        "imgsz": 640,
        
        # Customize preprocessing
        "use_medical_preprocessing": True,
        "hu_window": (-1000, 400),  # Lung window
        "pseudo_3d_slices": 3,
        
        # Customize augmentation
        "mosaic": 0.9,
        "mixup": 0.15,
        "copy_paste": 0.15,
        
        # Customize detection
        "conf_threshold": 0.25,
        "iou_threshold": 0.45,
        
        # Optimizer settings
        "optimizer": "AdamW",
        "learning_rate": 0.01,
        "cos_lr": True,
        "warmup_epochs": 5,
        
        "save_dir": "./experiments/example_5_custom",
    })
    
    summary = train_yolov11(**config)
    print(f"\nTraining completed!")
    print(f"Final metrics:")
    for key, value in summary['metrics'].items():
        print(f"  {key}: {value:.3f}")
    
    return summary


def example_6_production_ready():
    """Example 6: Production-ready training with full optimization."""
    print("\n" + "=" * 80)
    print("Example 6: Production-Ready Training")
    print("=" * 80)
    
    config = MedicalCTConfigs.get_production_config()
    config.update({
        "data_dir": "./datasets/ct_tumor_data",
        "num_epochs": 150,  # Reduced from 200 for demo
        "save_dir": "./experiments/example_6_production",
        
        # Enable all optimizations
        "use_medical_preprocessing": True,
        "use_augmentation": True,
        "use_focal_loss": True,
        "multi_scale": True,
    })
    
    print("\nStarting production training with:")
    print(f"  Model: YOLOv8{config['model_size']}")
    print(f"  Image size: {config['imgsz']}x{config['imgsz']}")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"  Optimizer: {config['optimizer']}")
    
    summary = train_yolov11(**config)
    
    print(f"\n{'=' * 80}")
    print("Production Training Results")
    print(f"{'=' * 80}")
    print(f"Model path: {summary['model_path']}")
    print(f"\nMetrics:")
    for key, value in sorted(summary['metrics'].items()):
        print(f"  {key:<20}: {value:.4f}")
    
    return summary


def main():
    """Run selected examples."""
    print("=" * 80)
    print("YOLOv8 Medical CT Tumor Detection - Example Usage")
    print("=" * 80)
    print("\nAvailable examples:")
    print("  1. Basic training")
    print("  2. Small tumor detection")
    print("  3. High precision training")
    print("  4. Multiple experiments with cache")
    print("  5. Custom configuration")
    print("  6. Production-ready training")
    print("  0. Run all examples")
    
    choice = input("\nSelect example to run (0-6): ").strip()
    
    # Note: Update data_dir in each example before running
    print("\n⚠️  Remember to update 'data_dir' in the examples before running!")
    print("    Current default: ./datasets/ct_tumor_data")
    
    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return
    
    if choice == "1":
        example_1_basic_training()
    elif choice == "2":
        example_2_small_tumor_detection()
    elif choice == "3":
        example_3_high_precision()
    elif choice == "4":
        example_4_multiple_experiments_with_cache()
    elif choice == "5":
        example_5_custom_configuration()
    elif choice == "6":
        example_6_production_ready()
    elif choice == "0":
        print("\nRunning all examples sequentially...")
        example_1_basic_training()
        example_2_small_tumor_detection()
        example_3_high_precision()
        example_4_multiple_experiments_with_cache()
        example_5_custom_configuration()
        example_6_production_ready()
    else:
        print("Invalid choice!")
        return
    
    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
