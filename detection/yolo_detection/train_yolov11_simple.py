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
import shutil  # Add shutil import at the top
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import torch
import numpy as np
from tqdm import tqdm

# Optional matplotlib import
MATPLOTLIB_AVAILABLE = False
ULTRALYTICS_AVAILABLE = False
YOLO_MODULES_AVAILABLE = False

try:
    import matplotlib
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    pass

try:
    if 'matplotlib' not in sys.modules and 'matplotlib.pyplot' not in sys.modules:
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
    # Import from the same directory
    from .train_yolov11 import YOLOv11CTDataset
    YOLO_MODULES_AVAILABLE = True
except ImportError:
    try:
        # Fallback: try direct import
        from train_yolov11 import YOLOv11CTDataset
        YOLO_MODULES_AVAILABLE = True
    except ImportError:
        YOLO_MODULES_AVAILABLE = False
        print("Warning: Could not import YOLOv11CTDataset. Some functionality may be limited.")

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


def organize_training_outputs(save_dir: Path, training_name: str = 'training') -> Dict[str, str]:
    """
    Organize training outputs by moving runs and creating checkpoints structure
    
    Args:
        save_dir: Main save directory
        training_name: Name of the training run
        
    Returns:
        Dict[str, str]: Dictionary with organized paths
    """
    import shutil  # Import shutil at the beginning of the function
    
    try:
        # Define directories
        checkpoints_dir = save_dir / 'checkpoints'
        runs_dir = save_dir / 'runs'
        training_dir = save_dir / training_name
        
        # Create directories if they don't exist
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        runs_dir.mkdir(parents=True, exist_ok=True)
        
        organized_paths = {
            'checkpoints_dir': str(checkpoints_dir),
            'runs_dir': str(runs_dir),
            'training_dir': str(training_dir)
        }
        
        # Move weights to checkpoints directory if training completed
        weights_dir = training_dir / 'weights'
        if weights_dir.exists():
            target_weights_dir = checkpoints_dir / 'weights'
            if target_weights_dir.exists():
                shutil.rmtree(target_weights_dir)
            shutil.move(str(weights_dir), str(target_weights_dir))
            logging.info(f"Moved model weights to: {target_weights_dir}")
            organized_paths['weights_dir'] = str(target_weights_dir)
            
            # Update paths in organized_paths
            organized_paths['best_model_path'] = str(target_weights_dir / 'best.pt')
            organized_paths['last_model_path'] = str(target_weights_dir / 'last.pt')
        
        # Move training plots and logs to runs directory
        plots_to_move = [
            'results.png', 'results.csv', 'confusion_matrix.png', 
            'confusion_matrix_normalized.png', 'BoxF1_curve.png',
            'BoxP_curve.png', 'BoxPR_curve.png', 'BoxR_curve.png'
        ]
        
        training_runs_dir = runs_dir / 'training'
        training_runs_dir.mkdir(parents=True, exist_ok=True)
        
        for plot_file in plots_to_move:
            source_path = training_dir / plot_file
            if source_path.exists():
                target_path = training_runs_dir / plot_file
                shutil.move(str(source_path), str(target_path))
                logging.info(f"Moved {plot_file} to runs directory")
        
        # Move validation images and other outputs
        val_files_pattern = ['val_batch*.jpg', 'train_batch*.jpg', 'labels.jpg']
        import glob
        for pattern in val_files_pattern:
            for file_path in glob.glob(str(training_dir / pattern)):
                filename = os.path.basename(file_path)
                target_path = training_runs_dir / filename
                shutil.move(file_path, str(target_path))
                logging.info(f"Moved {filename} to runs directory")
        
        # Check if any global runs directory exists and move it
        global_runs_dir = Path('runs')
        if global_runs_dir.exists():
            try:
                # Move contents of global runs to our organized runs directory
                target_global_runs = runs_dir / 'global_runs'
                
                # Remove existing target if it exists
                if target_global_runs.exists():
                    shutil.rmtree(target_global_runs)
                
                # Move the entire global runs directory
                shutil.move(str(global_runs_dir), str(target_global_runs))
                logging.info(f"Moved global runs directory to: {target_global_runs}")
                organized_paths['global_runs_dir'] = str(target_global_runs)
                
                # Also check for any detect directories in the workspace root and move them
                detect_dirs = list(Path('.').glob('runs/detect*'))
                if detect_dirs:
                    for detect_dir in detect_dirs:
                        detect_target = runs_dir / 'validation' / detect_dir.name
                        detect_target.parent.mkdir(parents=True, exist_ok=True)
                        if detect_target.exists():
                            shutil.rmtree(detect_target)
                        shutil.move(str(detect_dir), str(detect_target))
                        logging.info(f"Moved validation results: {detect_target}")
                        
            except Exception as move_error:
                logging.warning(f"Failed to move global runs directory: {move_error}")
        
        logging.info("Training outputs organized successfully")
        return organized_paths
        
    except Exception as e:
        logging.error(f"Error organizing training outputs: {e}")
        # Return basic structure even if organization fails
        return {
            'checkpoints_dir': str(save_dir / 'checkpoints'),
            'runs_dir': str(save_dir / 'runs'),
            'training_dir': str(save_dir / 'training'),
            'error': str(e)
        }


def validate_dataset_split(train_dataset, val_dataset, split_info: Dict[str, Any]) -> bool:
    """
    Validate that train and validation datasets have no patient overlap
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset  
        split_info: Split information dictionary
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    try:
        # Extract patient IDs from actual samples
        train_patients_from_data = set()
        val_patients_from_data = set()
        
        # Get patient IDs from training dataset samples
        for sample in train_dataset.samples:
            patient_id = sample.get('patient_id', 'unknown')
            if patient_id != 'unknown':
                train_patients_from_data.add(patient_id)
        
        # Get patient IDs from validation dataset samples  
        for sample in val_dataset.samples:
            patient_id = sample.get('patient_id', 'unknown')
            if patient_id != 'unknown':
                val_patients_from_data.add(patient_id)
        
        # Check for overlap in actual data
        data_overlap = train_patients_from_data & val_patients_from_data
        if data_overlap:
            logging.error(f"✗ Found patient overlap in actual dataset samples: {data_overlap}")
            return False
        
        # Verify consistency with split_info
        expected_train = set(split_info['train_patient_ids'])
        expected_val = set(split_info['val_patient_ids'])
        
        if train_patients_from_data != expected_train:
            logging.warning(f"Training dataset patients differ from expected split")
            logging.warning(f"Expected: {expected_train}")
            logging.warning(f"Actual: {train_patients_from_data}")
        
        if val_patients_from_data != expected_val:
            logging.warning(f"Validation dataset patients differ from expected split")
            logging.warning(f"Expected: {expected_val}")
            logging.warning(f"Actual: {val_patients_from_data}")
        
        logging.info(f"✓ Dataset split validation passed")
        logging.info(f"  Training patients in data: {len(train_patients_from_data)}")
        logging.info(f"  Validation patients in data: {len(val_patients_from_data)}")
        logging.info(f"  No patient overlap confirmed")
        
        return True
        
    except Exception as e:
        logging.error(f"Error during dataset split validation: {e}")
        return False


def load_config() -> Dict[str, Any]:
    """Load configuration from config.json"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config.json')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logging.info(f"Configuration loaded from: {config_path}")
        return config
    except Exception as e:
        logging.error(f"Failed to load config from {config_path}: {e}")
        # Return default config
        return {
            "data": {
                "dataset_splits_dir": "datasets/splited_dataset"
            }
        }


# Type hint fallbacks
if not YOLO_MODULES_AVAILABLE:
    DatasetType = Any
else:
    DatasetType = 'YOLOv11CTDataset'


def create_train_val_split(config: Dict[str, Any], train_ratio: float = 0.8, 
                          random_seed: int = 42, include_negative_samples: bool = True,
                          max_negative_per_patient: int = 0) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Create training and validation dataset split from config-defined dataset_splits_dir
    
    Args:
        config: Configuration dictionary from config.json
        train_ratio: Training set ratio (for further splitting the train set)
        random_seed: Random seed
        include_negative_samples: Whether to include negative samples
        max_negative_per_patient: Maximum negative samples per patient
        
    Returns:
        Tuple[Any, Any, Dict[str, Any]]: (training set, validation set, split info)
    """
    # Import CTDetectionDataset with proper error handling to avoid repeated warnings
    CTDetectionDataset = None
    try:
        from faster_rcnn_detection.faster_rcnn_dataset import CTDetectionDataset
    except ImportError:
        try:
            # Add parent directory to path and try again
            sys.path.append(os.path.dirname(os.path.dirname(__file__)))
            from faster_rcnn_detection.faster_rcnn_dataset import CTDetectionDataset
        except ImportError:
            logging.warning("CTDetectionDataset not available - using fallback dataset handling")
            # Continue without CTDetectionDataset - the YOLOv11CTDataset will handle this
    
    import torchvision.transforms as transforms
    
    # Get dataset splits directory from config
    dataset_splits_dir = config.get('data', {}).get('dataset_splits_dir', 'datasets/splited_dataset')
    train_data_dir = os.path.join(dataset_splits_dir, 'train')
    
    logging.info(f"Using dataset splits directory: {dataset_splits_dir}")
    logging.info(f"Training data directory: {train_data_dir}")
    
    # First, scan the train directory to get all patient IDs
    all_patient_ids = []
    if os.path.exists(train_data_dir):
        for item in os.listdir(train_data_dir):
            item_path = os.path.join(train_data_dir, item)
            if os.path.isdir(item_path) and item.startswith(('A', 'B', 'E', 'G')):  # Patient ID patterns
                all_patient_ids.append(item)
    
    if not all_patient_ids:
        logging.error(f"No patient directories found in {train_data_dir}")
        logging.error("Expected directories like A0001, B0001, etc.")
        raise ValueError(f"No patient data found in {train_data_dir}")
    
    all_patient_ids = sorted(all_patient_ids)
    total_patients = len(all_patient_ids)
    
    logging.info(f"Found {total_patients} patient directories in train split")
    logging.info(f"Sample patients: {all_patient_ids[:5]}...")
    
    # Set random seed and split patients into train/validation
    np.random.seed(random_seed)
    np.random.shuffle(all_patient_ids)
    
    train_size = int(total_patients * train_ratio)
    train_patient_ids = all_patient_ids[:train_size]
    val_patient_ids = all_patient_ids[train_size:]
    
    # Ensure no overlap between train and validation sets
    train_set = set(train_patient_ids)
    val_set = set(val_patient_ids)
    overlap = train_set & val_set
    
    if overlap:
        logging.error(f"Found overlapping patients between train and validation: {overlap}")
        raise ValueError("Train and validation sets must not have overlapping patients!")
    
    logging.info(f"Training patients: {len(train_patient_ids)}")
    logging.info(f"Validation patients: {len(val_patient_ids)}")
    logging.info(f"✓ Verified no patient overlap between train and validation sets")
    
    # Log sample patient IDs for verification
    logging.info(f"Sample training patients: {sorted(train_patient_ids)[:5]}")
    logging.info(f"Sample validation patients: {sorted(val_patient_ids)[:5]}")
    
    # Save patient split information for reproducibility
    split_info = {
        'random_seed': random_seed,
        'train_ratio': train_ratio,
        'total_patients': total_patients,
        'train_patients': len(train_patient_ids),
        'val_patients': len(val_patient_ids),
        'train_patient_ids': sorted(train_patient_ids),
        'val_patient_ids': sorted(val_patient_ids),
        'split_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Create datasets with the specific patient lists
    # Use train_data_dir as the base directory
    train_dataset = YOLOv11CTDataset(
        data_dir=train_data_dir,
        split='train',  # This will be overridden by patient_ids
        include_negative_samples=include_negative_samples,
        max_negative_per_patient=max_negative_per_patient,
        patient_ids=train_patient_ids
    )
    
    val_dataset = YOLOv11CTDataset(
        data_dir=train_data_dir,
        split='val',  # This will be overridden by patient_ids
        include_negative_samples=include_negative_samples,
        max_negative_per_patient=max_negative_per_patient,
        patient_ids=val_patient_ids
    )
    
    # Calculate dataset statistics
    train_stats = calculate_dataset_statistics(train_dataset.rcnn_dataset, "Training Set")
    val_stats = calculate_dataset_statistics(val_dataset.rcnn_dataset, "Validation Set")
    
    # Add split information to the datasets for later reference
    train_dataset.split_info = split_info
    val_dataset.split_info = split_info
    
    return train_dataset, val_dataset, split_info


def optimize_training_hyperparameters(config: Dict[str, Any], dataset_size: int) -> Dict[str, Any]:
    """
    Optimize training hyperparameters based on dataset size and medical imaging characteristics
    
    Args:
        config: Base configuration
        dataset_size: Size of training dataset
        
    Returns:
        Dict[str, Any]: Optimized hyperparameters
    """
    # Base hyperparameters for medical imaging
    optimized_params = {
        # Optimizer settings - AdamW works better for medical imaging
        'optimizer': 'AdamW',
        'lr0': 0.001,  # Lower learning rate for medical imaging
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        
        # Learning rate scheduling
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # Loss function weights - tuned for medical detection
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        
        # Data augmentation - DISABLED for medical imaging
        'hsv_h': 0.0,   # No hue changes for medical images
        'hsv_s': 0.0,   # No saturation changes
        'hsv_v': 0.0,   # No value changes
        'degrees': 0.0,  # No rotation
        'translate': 0.0,  # No translation
        'scale': 0.0,   # No scaling
        'shear': 0.0,   # No shearing
        'perspective': 0.0,  # No perspective changes
        'fliplr': 0.0,  # No horizontal flip
        'flipud': 0.0,  # No vertical flip
        'mosaic': 0.0,  # No mosaic augmentation
        'mixup': 0.0,   # No mixup
        'copy_paste': 0.0,  # No copy-paste
        'erasing': 0.0,  # No random erasing
        'auto_augment': None,  # Disable auto augmentation
        
        # Training settings
        'patience': 25,  # Early stopping
        'amp': True,    # Mixed precision training
        'save_period': 5,  # Save every 5 epochs
    }
    
    # Adjust based on dataset size
    if dataset_size < 500:
        # Small dataset adjustments - NO augmentation
        optimized_params.update({
            'lr0': 0.0005,  # Lower learning rate
            'warmup_epochs': 5.0,  # Longer warmup
            'patience': 15,  # Earlier stopping
            # Keep all augmentation disabled
        })
    elif dataset_size > 2000:
        # Large dataset adjustments - NO augmentation
        optimized_params.update({
            'lr0': 0.002,   # Higher learning rate
            'warmup_epochs': 2.0,  # Shorter warmup
            'patience': 35,  # More patience
            # Keep all augmentation disabled
        })
    
    logging.info(f"Optimized hyperparameters for dataset size {dataset_size} - DATA AUGMENTATION DISABLED")
    logging.info("All data augmentation techniques have been disabled for medical imaging accuracy")
    return optimized_params


def train_yolov11_simple(config: Dict[str, Any] = None, num_epochs: int = 100, batch_size: int = 16,
                        learning_rate: float = 0.01, save_dir: str = None,
                        log_dir: str = None, random_seed: int = 42,
                        train_ratio: float = 0.8, include_negative_samples: bool = True,
                        max_negative_per_patient: int = 0, imgsz: int = 640,
                        model_size: str = 'n', device: str = 'auto') -> Dict[str, Any]:
    """
    YOLOv11 Simple Training
    
    Args:
        config: Configuration dictionary from config.json
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        save_dir: Save directory
        log_dir: Log directory
        random_seed: Random seed
        train_ratio: Training set ratio (for splitting the train data into train/val)
        include_negative_samples: Whether to include negative samples
        max_negative_per_patient: Maximum negative samples per patient
        imgsz: Image size
        model_size: Model size
        device: Device type
        
    Returns:
        Dict[str, Any]: Training results
    """
    # Load config if not provided
    if config is None:
        config = load_config()
    
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
    
    # Log configuration source
    dataset_splits_dir = config.get('data', {}).get('dataset_splits_dir', 'datasets/splited_dataset')
    logging.info(f"Using dataset from config: {dataset_splits_dir}/train")
    
    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create training and validation datasets from config-defined train split
        logging.info("Preparing dataset from config-defined train split...")
        train_dataset, val_dataset, split_info = create_train_val_split(
            config=config,
            train_ratio=train_ratio,
            random_seed=random_seed,
            include_negative_samples=include_negative_samples,
            max_negative_per_patient=max_negative_per_patient
        )
        
        # Validate dataset split to ensure no patient overlap
        logging.info("Validating dataset split integrity...")
        if not validate_dataset_split(train_dataset, val_dataset, split_info):
            raise ValueError("Dataset split validation failed - found patient overlap!")
        
        # Save patient split information for future reference
        split_file = save_dir / 'patient_split_info.json'
        with open(split_file, 'w', encoding='utf-8') as f:
            json.dump(split_info, f, indent=2, ensure_ascii=False)
        logging.info(f"Patient split information saved to: {split_file}")
        
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
        
        # Create directories for organized outputs
        checkpoints_dir = save_dir / 'checkpoints'
        runs_dir = save_dir / 'runs'
        training_runs_dir = runs_dir / 'training'
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        runs_dir.mkdir(parents=True, exist_ok=True)
        training_runs_dir.mkdir(parents=True, exist_ok=True)
        
        # Get optimized hyperparameters based on dataset size
        train_dataset_size = len(train_dataset.rcnn_dataset)
        optimized_params = optimize_training_hyperparameters(config, train_dataset_size)
        
        # Override learning rate if explicitly provided
        if learning_rate != 0.01:  # Default value check
            optimized_params['lr0'] = learning_rate
        
        train_args = {
            'data': str(combined_config_path),
            'epochs': num_epochs,
            'batch': batch_size,
            'imgsz': imgsz,
            'device': device,
            'project': str(save_dir),  # YOLO will create subdirectories here
            'name': 'training',  # This will be the training subdirectory name
            'save': True,
            'val': True,
            'plots': True,
            'verbose': True,
            'workers': 4,  # Number of data loading processes
            'seed': random_seed,
            'exist_ok': True,  # Allow overwriting existing project
        }
        
        # Merge optimized parameters
        train_args.update(optimized_params)
        
        # Log the optimized training configuration
        logging.info("=== Optimized Training Configuration (NO AUGMENTATION) ===")
        logging.info(f"Dataset size: {train_dataset_size}")
        logging.info(f"Optimizer: {train_args['optimizer']}")
        logging.info(f"Learning rate: {train_args['lr0']}")
        logging.info(f"Weight decay: {train_args['weight_decay']}")
        logging.info(f"Warmup epochs: {train_args['warmup_epochs']}")
        logging.info(f"Patience: {train_args['patience']}")
        logging.info(f"Mixed precision: {train_args['amp']}")
        logging.info("Data Augmentation: DISABLED (All augmentation parameters set to 0)")
        logging.info("=" * 55)
        
        
        logging.info("Starting training...")
        logging.info(f"Training parameters: {train_args}")
        logging.info(f"Results will be organized in: {save_dir}")
        logging.info(f"Checkpoints directory: {checkpoints_dir}")
        logging.info(f"Runs directory: {runs_dir}")
        
        # Start training
        results = model.train(**train_args)
        
        # Calculate training time
        training_time = time.time() - training_start_time
        
        logging.info(f"Training completed! Time taken: {training_time/3600:.2f} hours")
        
        # Organize training outputs (move runs and checkpoints to result directory)
        logging.info("Organizing training outputs...")
        
        # Ensure the organize function gets called after training
        try:
            organized_paths = organize_training_outputs(save_dir, 'training')
            logging.info("Training outputs organized successfully")
        except Exception as organize_error:
            logging.warning(f"Failed to organize training outputs: {organize_error}")
            # Continue with fallback paths
            organized_paths = {
                'checkpoints_dir': str(save_dir / 'checkpoints'),
                'runs_dir': str(save_dir / 'runs'),
                'training_dir': str(save_dir / 'training')
            }
        
        # Get best model paths (updated after organization)
        if 'best_model_path' in organized_paths and organized_paths['best_model_path']:
            best_model_path = Path(organized_paths['best_model_path'])
            last_model_path = Path(organized_paths['last_model_path'])
        else:
            # Fallback to original paths if organization failed
            best_model_path = save_dir / 'training' / 'weights' / 'best.pt'
            last_model_path = save_dir / 'training' / 'weights' / 'last.pt'
            
            # If those don't exist, check in checkpoints
            if not best_model_path.exists():
                best_model_path = save_dir / 'checkpoints' / 'weights' / 'best.pt'
                last_model_path = save_dir / 'checkpoints' / 'weights' / 'last.pt'
        
        logging.info(f"Best model located at: {best_model_path}")
        logging.info(f"Last model located at: {last_model_path}")
        
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
        dataset_splits_dir = config.get('data', {}).get('dataset_splits_dir', 'datasets/splited_dataset')
        train_data_dir = os.path.join(dataset_splits_dir, 'train')
        
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
            'organized_outputs': organized_paths,  # Include organized paths information
            'data_source': {
                'dataset_splits_dir': dataset_splits_dir,
                'train_data_dir': train_data_dir,
                'config_based': True,
                'source_description': f'Using pre-split train data from {dataset_splits_dir}/train'
            },
            'dataset_info': {
                'train_size': len(train_dataset.rcnn_dataset),
                'val_size': len(val_dataset.rcnn_dataset),
                'train_ratio': train_ratio,
                'include_negative_samples': include_negative_samples,
                'max_negative_per_patient': max_negative_per_patient,
                'patient_split_info': split_info  # Include detailed split information
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
        
        # Log organized output structure
        if organized_paths:
            logging.info(f"\n{'='*40}")
            logging.info("Organized Output Structure:")
            logging.info(f"{'='*40}")
            logging.info(f"Main results directory: {save_dir}")
            if 'checkpoints_dir' in organized_paths:
                logging.info(f"Model checkpoints: {organized_paths['checkpoints_dir']}")
            if 'runs_dir' in organized_paths:
                logging.info(f"Training runs/plots: {organized_paths['runs_dir']}")
            if 'weights_dir' in organized_paths:
                logging.info(f"Model weights: {organized_paths['weights_dir']}")
            if 'global_runs_dir' in organized_paths:
                logging.info(f"Global runs moved to: {organized_paths['global_runs_dir']}")
        
        if metrics:
            logging.info(f"\n{'='*40}")
            logging.info("Performance Metrics:")
            logging.info(f"{'='*40}")
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
    parser.add_argument('--config', type=str, 
                       default='config.json',
                       help='Configuration file path (default: config.json)')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs (increased from 300)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (optimized for medical imaging)')
    parser.add_argument('--save_dir', type=str, default=None, 
                       help='Save directory (default: auto-generated with timestamp in detection/yolo_detection/results/)')
    parser.add_argument('--log_dir', type=str, default=None, 
                       help='Log directory (default: auto-generated with timestamp in save_dir/logs/)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training set ratio (for splitting train data into train/val)')
    parser.add_argument('--include_negative', action='store_true', default=True, help='Whether to include negative samples')
    parser.add_argument('--max_negative', type=int, default=15, help='Maximum negative samples per patient (reduced for better balance)')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--model_size', type=str, default='x', choices=['n', 's', 'm', 'l', 'x'], help='Model size')
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
    
    # Load configuration
    config = load_config()
    dataset_splits_dir = config.get('data', {}).get('dataset_splits_dir', 'datasets/splited_dataset')
    train_data_dir = os.path.join(dataset_splits_dir, 'train')
    
    # Display system information
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"Requested device: {args.device}")
    print(f"Dataset source: {train_data_dir}")
    print()
    
    # Check if train data directory exists
    if not os.path.exists(train_data_dir):
        print(f"Error: Train data directory does not exist: {train_data_dir}")
        print("Please check your config.json file and ensure the dataset_splits_dir is correctly set.")
        return
    
    # Start training
    try:
        results = train_yolov11_simple(
            config=config,
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
