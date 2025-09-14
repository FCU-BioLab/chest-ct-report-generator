#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv11 Simple Training Script - Simplified YOLOv11 training script
Based on the original train_detection_simple.py logic, using YOLOv11 for simple training

Main Features:
1. Simple single training (no cross-validation)
2. Automatic dataset preparation and format conversion
3. Training process monitoring and result saving
4. Basic performance evaluation and visualization

Author: Based on Faster R-CNN simple training logic
Date: 2025-09-06
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import numpy as np
from tqdm import tqdm

# Optional matplotlib import
# Optional dependencies
MATPLOTLIB_AVAILABLE = False
ULTRALYTICS_AVAILABLE = False
YOLO_MODULES_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    pass

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    pass

# Import YOLOv11 dataset module
try:
    # Add parent directory to Python path for importing detection modules
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from train_yolov11 import YOLOv11CTDataset
    YOLO_MODULES_AVAILABLE = True
except ImportError:
    pass

# Optional detection modules (for advanced features)
MODULES_IMPORTED = False
try:
    from metrics.detection_metrics import calculate_comprehensive_metrics, calculate_detection_metrics
    from metrics.roc_froc import calculate_roc_froc_curves
    from metrics.dataset_statistics import save_patient_lists
    from metrics.iou_calculations import calculate_iou_variants, calculate_iou_matrix, calculate_bbox_error
    from visualization import visualize_predictions, create_prediction_summary, create_comprehensive_summary
    from data_processing import create_kfold_datasets
    from evaluation import evaluate_model
    from utils import collate_fn
    MODULES_IMPORTED = True
except ImportError:
    pass

def check_dependencies() -> Dict[str, bool]:
    """Check if required dependencies are available"""
    dependencies = {
        'ultralytics': ULTRALYTICS_AVAILABLE,
        'matplotlib': MATPLOTLIB_AVAILABLE,
        'yolo_modules': YOLO_MODULES_AVAILABLE,
        'optional_modules': MODULES_IMPORTED
    }
    
    # Log dependency status
    logging.info("Dependency Status:")
    for dep, available in dependencies.items():
        status = "✓" if available else "✗"
        logging.info(f"  {dep}: {status}")
    
    return dependencies


def validate_requirements() -> bool:
    """Validate that critical requirements are met"""
    if not ULTRALYTICS_AVAILABLE:
        logging.error("ultralytics library is required but not installed")
        logging.error("Please install with: pip install ultralytics")
        return False
    
    if not YOLO_MODULES_AVAILABLE:
        logging.error("YOLOv11 modules are required but not available")
        logging.error("Please ensure train_yolov11.py exists and is importable")
        return False
    
    return True


def calculate_dataset_statistics(dataset, dataset_name: str = "Dataset") -> Dict[str, Any]:
    """Calculate detailed statistics of the dataset"""
    total_images = len(dataset)
    total_annotations = 0
    images_with_annotations = 0
    images_without_annotations = 0
    
    logging.info(f"Calculating {dataset_name} statistics...")
    
    for i in range(total_images):
        try:
            item = dataset[i]
            target = item['target']
            
            if 'boxes' in target and len(target['boxes']) > 0:
                num_boxes = len(target['boxes'])
                total_annotations += num_boxes
                images_with_annotations += 1
            else:
                images_without_annotations += 1
        except Exception as e:
            logging.warning(f"Error processing sample {i}: {e}")
            images_without_annotations += 1
    
    stats = {
        'dataset_name': dataset_name,
        'total_images': total_images,
        'total_annotations': total_annotations,
        'images_with_annotations': images_with_annotations,
        'images_without_annotations': images_without_annotations,
        'avg_annotations_per_image': total_annotations / total_images if total_images > 0 else 0,
        'avg_annotations_per_annotated_image': total_annotations / images_with_annotations if images_with_annotations > 0 else 0
    }
    
    # Log statistics
    logging.info(f"=== {dataset_name} Statistics ===")
    logging.info(f"Total images: {total_images}")
    logging.info(f"Total annotations: {total_annotations}")
    logging.info(f"Images with annotations: {images_with_annotations}")
    logging.info(f"Images without annotations: {images_without_annotations}")
    logging.info(f"Average annotations per image: {stats['avg_annotations_per_image']:.2f}")
    if images_with_annotations > 0:
        logging.info(f"Average annotations per annotated image: {stats['avg_annotations_per_annotated_image']:.2f}")
    
    return stats


def download_and_save_model(model_name: str, model_dir: str) -> str:
    """Download YOLO model and save to specified directory"""
    model_path = os.path.join(model_dir, model_name)
    
    if os.path.exists(model_path):
        logging.info(f"Model already exists at: {model_path}")
        return model_path
    
    if not ULTRALYTICS_AVAILABLE:
        raise ImportError("ultralytics library is required to download models")
    
    try:
        logging.info(f"Downloading {model_name}...")
        model = YOLO(model_name)
        
        # Copy the model to our directory by loading and saving
        model.save(model_path)
        logging.info(f"Model saved to: {model_path}")
        return model_path
        
    except Exception as e:
        logging.error(f"Failed to download {model_name}: {e}")
        raise


def setup_yolo_directories() -> str:
    """Setup YOLO detection directories"""
    base_dir = Path('detection') / 'yolo_detection'
    directories = [
        base_dir / 'models',
        base_dir / 'results',
        base_dir / 'checkpoints'
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created directory: {directory}")
    
    return str(base_dir)


def setup_logging(log_dir: str) -> str:
    """Setup logging configuration"""
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir_path / f'yolov11_simple_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                 datefmt='%Y-%m-%d %H:%M:%S')
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return str(log_file)


def create_train_val_split(data_dir: str, train_ratio: float = 0.8, 
                          random_seed: int = 42, include_negative_samples: bool = True,
                          max_negative_per_patient: int = 0) -> Tuple[YOLOv11CTDataset, YOLOv11CTDataset]:
    """
    Create training and validation dataset split
    
    Args:
        data_dir: Data directory
        train_ratio: Training set ratio
        random_seed: Random seed
        include_negative_samples: Whether to include negative samples
        max_negative_per_patient: Maximum negative samples per patient
        
    Returns:
        Tuple[YOLOv11CTDataset, YOLOv11CTDataset]: (training set, validation set)
    """
    from faster_rcnn_dataset import CTDetectionDataset
    import torchvision.transforms as transforms
    
    # First, scan the directory to get all patient IDs
    all_patient_ids = []
    if os.path.exists(data_dir):
        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path) and item.startswith(('A', 'B', 'E', 'G')):  # Patient ID patterns
                all_patient_ids.append(item)
    
    if not all_patient_ids:
        logging.error(f"No patient directories found in {data_dir}")
        logging.error("Expected directories like A0001, B0001, etc.")
        raise ValueError(f"No patient data found in {data_dir}")
    
    all_patient_ids = sorted(all_patient_ids)
    total_patients = len(all_patient_ids)
    
    logging.info(f"Found {total_patients} patient directories")
    logging.info(f"Sample patients: {all_patient_ids[:5]}...")
    
    # Set random seed and split patients
    np.random.seed(random_seed)
    np.random.shuffle(all_patient_ids)
    
    train_size = int(total_patients * train_ratio)
    train_patient_ids = all_patient_ids[:train_size]
    val_patient_ids = all_patient_ids[train_size:]
    
    logging.info(f"Training patients: {len(train_patient_ids)}")
    logging.info(f"Validation patients: {len(val_patient_ids)}")
    
    # Create datasets with the specific patient lists
    # Use 'all' as split and specify patient IDs to bypass the train/test directory structure
    train_dataset = YOLOv11CTDataset(
        data_dir=data_dir,
        split='train',  # This will be overridden by patient_ids
        include_negative_samples=include_negative_samples,
        max_negative_per_patient=max_negative_per_patient,
        patient_ids=train_patient_ids
    )
    
    val_dataset = YOLOv11CTDataset(
        data_dir=data_dir,
        split='val',  # This will be overridden by patient_ids
        include_negative_samples=include_negative_samples,
        max_negative_per_patient=max_negative_per_patient,
        patient_ids=val_patient_ids
    )
    
    # Calculate dataset statistics
    train_stats = calculate_dataset_statistics(train_dataset.rcnn_dataset, "Training Set")
    val_stats = calculate_dataset_statistics(val_dataset.rcnn_dataset, "Validation Set")
    
    return train_dataset, val_dataset


def train_yolov11_simple(data_dir: str, num_epochs: int = 100, batch_size: int = 16,
                        learning_rate: float = 0.01, save_dir: str = None,
                        log_dir: str = None, random_seed: int = 42,
                        train_ratio: float = 0.8, include_negative_samples: bool = True,
                        max_negative_per_patient: int = 0, imgsz: int = 640,
                        model_size: str = 'n', device: str = 'auto') -> Dict[str, Any]:
    """
    YOLOv11 Simple Training
    
    Args:
        data_dir: Data directory
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        save_dir: Save directory
        log_dir: Log directory
        random_seed: Random seed
        train_ratio: Training set ratio
        include_negative_samples: Whether to include negative samples
        max_negative_per_patient: Maximum negative samples per patient
        imgsz: Image size
        model_size: Model size
        device: Device type
        
    Returns:
        Dict[str, Any]: Training results
    """
    # Validate dependencies
    if not validate_requirements():
        raise RuntimeError("Critical dependencies not available")
    
    # Setup YOLO directories
    setup_yolo_directories()
    
    # Record training start time and create timestamped directories
    training_start_time = time.time()
    start_datetime = datetime.now()
    timestamp = start_datetime.strftime("%Y%m%d_%H%M%S")
    
    # Create timestamped save and log directories if not provided
    base_yolo_dir = Path('detection') / 'yolo_detection'
    if save_dir is None:
        save_dir = base_yolo_dir / 'results' / f'yolov11_training_{timestamp}'
    else:
        save_dir = Path(save_dir)
    
    if log_dir is None:
        log_dir = save_dir / 'logs'
    else:
        log_dir = Path(log_dir)
    
    # Setup logging
    log_file = setup_logging(str(log_dir))
    logging.info("Starting YOLOv11 simple training")
    logging.info(f"Training started at: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Results will be saved to: {save_dir}")
    
    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create training and validation datasets
        logging.info("Preparing dataset...")
        train_dataset, val_dataset = create_train_val_split(
            data_dir=data_dir,
            train_ratio=train_ratio,
            random_seed=random_seed,
            include_negative_samples=include_negative_samples,
            max_negative_per_patient=max_negative_per_patient
        )
        
        # Prepare YOLOv11 format data
        logging.info("Converting data format to YOLOv11 format...")
        train_config_path = train_dataset.prepare_yolo_format(str(save_dir))
        val_config_path = val_dataset.prepare_yolo_format(str(save_dir))
        
        # Merge training and validation dataset configurations
        combined_config_path = save_dir / 'dataset.yaml'
        yaml_content = f"""path: {save_dir}
train: yolo_dataset_train/images/train
val: yolo_dataset_val/images/val

nc: 1
names: ['lesion']
"""
        with open(combined_config_path, 'w') as f:
            f.write(yaml_content)
        
        logging.info(f"Dataset configuration file created: {combined_config_path}")
        
        # Initialize YOLOv11 model
        model_dir = base_yolo_dir / 'models'
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_name = f'yolo11{model_size}.pt'  # Use correct YOLO11 naming
        
        logging.info(f"Loading pretrained model: {model_name}")
        
        # Try to download and save YOLO11 model
        try:
            model_path = download_and_save_model(model_name, str(model_dir))
            model = YOLO(model_path)
        except Exception as e:
            # If YOLO11 is not available, try using YOLOv8 as backup
            backup_model_name = f'yolov8{model_size}.pt'
            logging.warning(f"YOLO11 model loading failed: {e}")
            logging.info(f"Trying backup model: {backup_model_name}")
            
            backup_model_path = download_and_save_model(backup_model_name, str(model_dir))
            model = YOLO(backup_model_path)
        
        # Training parameters
        # Auto-detect device: use CUDA if available, otherwise CPU
        if device == 'auto':
            try:
                if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                    device = 0  # Use integer for YOLO (first CUDA device)
                    logging.info(f"CUDA available: Using GPU device {device}")
                    logging.info(f"CUDA devices detected: {torch.cuda.device_count()}")
                    logging.info(f"Current CUDA device: {torch.cuda.get_device_name(0)}")
                else:
                    device = 'cpu'
                    logging.info("CUDA not detected: Using CPU")
                    logging.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
                    logging.info(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
            except Exception as e:
                logging.warning(f"Error detecting CUDA device: {e}")
                device = 'cpu'
                logging.info("Fallback to CPU due to CUDA detection error")
        elif device == 'cuda' or device == 'gpu':
            device = 0  # Convert 'cuda' to device index for YOLO
        elif device.isdigit():
            device = int(device)  # Convert string device index to integer
        
        logging.info(f"Selected device: {device}")
        
        train_args = {
            'data': str(combined_config_path),
            'epochs': num_epochs,
            'batch': batch_size,
            'lr0': learning_rate,
            'imgsz': imgsz,
            'device': device,
            'project': str(save_dir),
            'name': 'training',
            'save': True,
            'save_period': 10,  # Save every 10 epochs
            'val': True,
            'plots': True,
            'verbose': True,
            'patience': 50,  # Early stopping patience
            'workers': 4,  # Number of data loading processes
            'seed': random_seed,
            'exist_ok': True,  # Allow overwriting existing project
        }
        
        logging.info("Starting training...")
        logging.info(f"Training parameters: {train_args}")
        
        # Start training
        results = model.train(**train_args)
        
        # Calculate training time
        training_time = time.time() - training_start_time
        
        logging.info(f"Training completed! Time taken: {training_time/3600:.2f} hours")
        
        # Get best model path
        best_model_path = save_dir / 'training' / 'weights' / 'best.pt'
        last_model_path = save_dir / 'training' / 'weights' / 'last.pt'
        
        # YOLO training already includes built-in validation
        # The results object contains all the metrics we need
        logging.info("Using YOLO built-in evaluation metrics...")
        
        # Extract metrics from YOLO training results
        metrics = {}
        if hasattr(results, 'results_dict'):
            # Get metrics from the training results
            results_dict = results.results_dict
            metrics = {
                'final_precision': results_dict.get('metrics/precision(B)', 0.0),
                'final_recall': results_dict.get('metrics/recall(B)', 0.0),
                'final_map50': results_dict.get('metrics/mAP50(B)', 0.0),
                'final_map50_95': results_dict.get('metrics/mAP50-95(B)', 0.0),
                'final_fitness': results_dict.get('fitness', 0.0)
            }
        
        # Run validation on best model to get final metrics
        logging.info("Running final validation with best model...")
        best_model = YOLO(str(best_model_path))
        val_results = best_model.val(data=str(combined_config_path), split='val')
        
        if val_results:
            # Extract precision and recall first
            best_precision = float(val_results.box.p[0]) if hasattr(val_results.box, 'p') and len(val_results.box.p) > 0 else 0.0
            best_recall = float(val_results.box.r[0]) if hasattr(val_results.box, 'r') and len(val_results.box.r) > 0 else 0.0
            
            # Calculate F1 score
            best_f1_score = 2 * (best_precision * best_recall) / (best_precision + best_recall) if (best_precision + best_recall) > 0 else 0.0
            
            metrics.update({
                'best_precision': best_precision,
                'best_recall': best_recall,
                'best_map50': float(val_results.box.map50) if hasattr(val_results.box, 'map50') else 0.0,
                'best_map50_95': float(val_results.box.map) if hasattr(val_results.box, 'map') else 0.0,
                'best_f1_score': best_f1_score
            })
        
        # Create training summary
        training_summary = {
            'training_start_time': start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            'training_timestamp': timestamp,
            'training_time_seconds': training_time,
            'training_time_hours': training_time/3600,
            'training_args': train_args,
            'best_model_path': str(best_model_path),
            'last_model_path': str(last_model_path),
            'dataset_config_path': str(combined_config_path),
            'evaluation_metrics': metrics,
            'yolo_training_results': str(results) if results else 'No results available',
            'model_dir': str(model_dir),
            'yolo_detection_base': str(base_yolo_dir),
            'save_dir': str(save_dir),
            'log_dir': str(log_dir),
            'dataset_info': {
                'train_size': len(train_dataset.rcnn_dataset),
                'val_size': len(val_dataset.rcnn_dataset),
                'train_ratio': train_ratio,
                'include_negative_samples': include_negative_samples,
                'max_negative_per_patient': max_negative_per_patient
            },
            'config': {
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'imgsz': imgsz,
                'model_size': model_size,
                'random_seed': random_seed,
                'device': device
            }
        }
        
        # Save training summary
        summary_file = save_dir / 'training_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(training_summary, f, indent=2, ensure_ascii=False, default=str)
        
        # Create simple performance visualization
        create_training_visualization(training_summary, str(save_dir))
        
        # Output final results
        logging.info(f"\n{'='*60}")
        logging.info("Training Completion Summary")
        logging.info(f"{'='*60}")
        logging.info(f"Training started: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Training time: {training_time/3600:.2f} hours")
        logging.info(f"Best model: {best_model_path}")
        if metrics:
            logging.info(f"Best mAP50: {metrics.get('best_map50', 0.0):.3f}")
            logging.info(f"Best mAP50-95: {metrics.get('best_map50_95', 0.0):.3f}")
            logging.info(f"Best precision: {metrics.get('best_precision', 0.0):.3f}")
            logging.info(f"Best recall: {metrics.get('best_recall', 0.0):.3f}")
            logging.info(f"Best F1 score: {metrics.get('best_f1_score', 0.0):.3f}")
        logging.info(f"Results saved in: {save_dir}")
        logging.info(f"Training summary: {summary_file}")
        
        return training_summary
        
    except Exception as e:
        logging.error(f"Error during training: {e}")
        return {
            'error': str(e),
            'training_start_time': start_datetime.strftime('%Y-%m-%d %H:%M:%S') if 'start_datetime' in locals() else 'Unknown',
            'training_timestamp': timestamp if 'timestamp' in locals() else 'Unknown',
            'training_time': time.time() - training_start_time,
            'save_dir': str(save_dir)
        }


def create_training_visualization(training_summary: Dict[str, Any], save_dir: str):
    """Create training visualization charts"""
    if not MATPLOTLIB_AVAILABLE:
        logging.warning("matplotlib not available - skipping visualization")
        return
    
    # Import matplotlib here to avoid issues with early imports
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    
    vis_dir = Path(save_dir) / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Create training summary chart
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.axis('off')
    
    # Prepare text information
    config = training_summary.get('config', {})
    dataset_info = training_summary.get('dataset_info', {})
    metrics = training_summary.get('evaluation_metrics', {})
    
    # Format metrics safely
    def format_metric(value):
        if isinstance(value, (int, float)):
            return f"{value:.3f}"
        return "N/A"
    
    summary_text = f"""YOLOv11 Training Summary
    
Training Configuration:
• Model Size: {config.get('model_size', 'unknown')}
• Training Epochs: {config.get('num_epochs', 'unknown')}
• Batch Size: {config.get('batch_size', 'unknown')}
• Learning Rate: {config.get('learning_rate', 'unknown')}
• Image Size: {config.get('imgsz', 'unknown')}

Dataset Information:
• Training Set Size: {dataset_info.get('train_size', 'unknown')}
• Validation Set Size: {dataset_info.get('val_size', 'unknown')}
• Training Ratio: {dataset_info.get('train_ratio', 'unknown')}
• Include Negative Samples: {dataset_info.get('include_negative_samples', 'unknown')}

YOLO Performance Metrics:
• Best mAP50: {format_metric(metrics.get('best_map50'))}
• Best mAP50-95: {format_metric(metrics.get('best_map50_95'))}
• Best Precision: {format_metric(metrics.get('best_precision'))}
• Best Recall: {format_metric(metrics.get('best_recall'))}
• Best F1 Score: {format_metric(metrics.get('best_f1_score'))}

Training Information:
• Training Started: {training_summary.get('training_start_time', 'Unknown')}
• Training Time: {training_summary.get('training_time_hours', 0):.2f} hours
• Timestamp: {training_summary.get('training_timestamp', 'Unknown')}
"""
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    chart_path = vis_dir / 'training_summary.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Training visualization charts saved to: {vis_dir}")


def main():
    """Main function"""
    # Set console encoding (Windows)
    if sys.platform.startswith('win'):
        try:
            os.system('chcp 65001 >nul')
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except:
            pass
    
    parser = argparse.ArgumentParser(description='YOLOv11 Simple Training for CT Detection')
    parser.add_argument('--data_dir', type=str, 
                       default='datasets/all_patient_data',
                       help='Data directory path (default: datasets/all_patient_data)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default=None, 
                       help='Save directory (default: auto-generated with timestamp in detection/yolo_detection/results/)')
    parser.add_argument('--log_dir', type=str, default=None, 
                       help='Log directory (default: auto-generated with timestamp in save_dir/logs/)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training set ratio')
    parser.add_argument('--include_negative', action='store_true', default=True, help='Whether to include negative samples')
    parser.add_argument('--max_negative', type=int, default=20, help='Maximum negative samples per patient')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--model_size', type=str, default='s', choices=['n', 's', 'm', 'l', 'x'], help='Model size')
    parser.add_argument('--device', type=str, default='auto', help='Device type (auto, cpu, cuda, 0, 1, etc.)')
    parser.add_argument('--check_deps', action='store_true', help='Only check dependencies and exit')
    
    args = parser.parse_args()
    
    # Setup basic logging for dependency checking
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Check dependencies
    deps = check_dependencies()
    
    if args.check_deps:
        print("\nDependency Check Results:")
        print("-" * 30)
        for dep, available in deps.items():
            status = "✓ Available" if available else "✗ Not Available"
            print(f"{dep}: {status}")
        return
    
    # Validate requirements
    if not validate_requirements():
        print("\nCritical requirements not met. Please install missing dependencies.")
        print("Run with --check_deps to see detailed dependency status.")
        return
    
    # Display system information
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"Requested device: {args.device}")
    print()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory does not exist: {args.data_dir}")
        print("Please specify the correct data directory path, for example:")
        print("python detection/yolo_detection/train_yolov11_simple.py --data_dir datasets/all_patient_data")
        return
    
    # Start training
    try:
        results = train_yolov11_simple(
            data_dir=args.data_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            save_dir=args.save_dir,
            log_dir=args.log_dir,
            random_seed=args.seed,
            train_ratio=args.train_ratio,
            include_negative_samples=args.include_negative,
            max_negative_per_patient=args.max_negative,
            imgsz=args.imgsz,
            model_size=args.model_size,
            device=args.device
        )
        
        if 'error' in results:
            print(f"\nTraining failed: {results['error']}")
        else:
            print(f"\nTraining completed! Results saved in: {results.get('save_dir', args.save_dir)}")
            if 'evaluation_metrics' in results:
                metrics = results['evaluation_metrics']
                best_map50 = metrics.get('best_map50', 'N/A')
                best_map50_95 = metrics.get('best_map50_95', 'N/A')
                if isinstance(best_map50, (int, float)):
                    print(f"Best mAP50: {best_map50:.3f}")
                else:
                    print(f"Best mAP50: {best_map50}")
                if isinstance(best_map50_95, (int, float)):
                    print(f"Best mAP50-95: {best_map50_95:.3f}")
                else:
                    print(f"Best mAP50-95: {best_map50_95}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error during training: {e}")
        logging.exception("Full error details:")


if __name__ == "__main__":
    main()
