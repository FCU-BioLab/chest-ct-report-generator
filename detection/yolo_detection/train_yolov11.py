#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv11 Training for CT Detection.

This script prepares CT detection datasets for YOLOv11, launches a single
Ultralytics training run, evaluates the trained model, and records metrics.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import textwrap
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

# Optional cv2 import (required for dataset export)
try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    CV2_AVAILABLE = False

# Optional ultralytics import (required for training and evaluation)
try:
    from ultralytics import YOLO

    ULTRALYTICS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    ULTRALYTICS_AVAILABLE = False

# Locate shared modules (dataset definitions, utilities)
try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
    sys.path.append(str(REPO_ROOT))
except Exception:  # pragma: no cover - best effort path adjustment
    REPO_ROOT = Path.cwd()

try:
    from detection_dataset import CTDetectionDataset

    DATASET_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback for alternate layout
    try:
        from faster_rcnn_detection.faster_rcnn_dataset import CTDetectionDataset

        DATASET_AVAILABLE = True
    except ImportError:  # pragma: no cover - cannot proceed without dataset
        DATASET_AVAILABLE = False

# Optional utilities from metrics/data_processing packages
try:
    from metrics.dataset_statistics import calculate_dataset_statistics as external_calculate_dataset_statistics
    from metrics.dataset_statistics import save_patient_lists as external_save_patient_lists
except ImportError:  # pragma: no cover - fallback implementations
    external_calculate_dataset_statistics = None
    external_save_patient_lists = None

try:
    from utils import setup_logging as external_setup_logging
except ImportError:  # pragma: no cover - use local logging setup
    external_setup_logging = None

LOGGER = logging.getLogger("yolov11_training")


def _ensure_utf8_console() -> None:
    """Best-effort UTF-8 console configuration for Windows."""
    if not sys.platform.startswith("win"):
        return
    try:
        os.system("chcp 65001 >nul")
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # pragma: no cover - console setup is best effort
        pass


_ensure_utf8_console()


@dataclass
class TrainingConfig:
    """Container for YOLOv11 training hyperparameters and paths."""

    data_dir: str
    num_epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.01
    save_dir: str = "./yolov11_models"
    log_dir: str = "./yolov11_logs"
    random_seed: int = 42
    include_negative_samples: bool = True
    max_negative_per_patient: int = 0
    imgsz: int = 640
    model_size: str = "n"
    val_ratio: float = 0.2
    train_split: str = "train"
    val_split: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def build_train_args(self, data_config: str, project_dir: Path, device: str) -> Dict[str, Any]:
        """Compose Ultralytics training arguments."""
        run_name = "training_run"
        return {
            "data": data_config,
            "epochs": self.num_epochs,
            "batch": self.batch_size,
            "lr0": self.learning_rate,
            "imgsz": self.imgsz,
            "device": device,
            "project": str(project_dir),
            "name": run_name,
            "save": True,
            "save_period": 10,
            "val": True,
            "plots": True,
            "verbose": True,
            "patience": 50,
            "workers": 4,
            "seed": self.random_seed,
        }


@dataclass
class DatasetArtifacts:
    """Filesystem artifacts created during dataset preparation."""

    export_root: Path
    combined_config: Path
    train_dataset_config: Path
    val_dataset_config: Optional[Path]


def _fallback_calculate_dataset_statistics(dataset: Any, dataset_name: str) -> Dict[str, Any]:
    """Minimal dataset statistics computation used when metrics module is absent."""
    total_images = len(dataset)
    total_annotations = 0
    images_with_annotations = 0
    images_without_annotations = 0

    LOGGER.info("Collecting statistics for %s (size=%s)", dataset_name, total_images)

    for idx in range(total_images):
        try:
            item = dataset[idx]
            if isinstance(item, dict) and "target" in item:
                target = item["target"]
            else:
                _, target = item

            boxes = target.get("boxes", []) if target else []
            if boxes and len(boxes) > 0:
                total_annotations += len(boxes)
                images_with_annotations += 1
            else:
                images_without_annotations += 1
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning("Failed to analyse sample %s in %s: %s", idx, dataset_name, exc)
            images_without_annotations += 1

    stats = {
        "dataset_name": dataset_name,
        "total_images": total_images,
        "total_annotations": total_annotations,
        "images_with_annotations": images_with_annotations,
        "images_without_annotations": images_without_annotations,
        "avg_annotations_per_image": total_annotations / total_images if total_images else 0.0,
        "avg_annotations_per_annotated_image": (
            total_annotations / images_with_annotations if images_with_annotations else 0.0
        ),
    }

    LOGGER.info(
        "[%s] images=%s, annotated=%s, empty=%s, avg_annotations=%.2f",
        dataset_name,
        total_images,
        images_with_annotations,
        images_without_annotations,
        stats["avg_annotations_per_image"],
    )
    return stats


DATASET_STATS_FN = (
    external_calculate_dataset_statistics if external_calculate_dataset_statistics else _fallback_calculate_dataset_statistics
)


def setup_logging(log_dir: str) -> str:
    """Create a timestamped log file and configure root logger."""
    if external_setup_logging:
        return external_setup_logging(log_dir)

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / f"yolov11_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Reset existing handlers to avoid duplicate logs between runs
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8", mode="w")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return str(log_file)


def select_device() -> str:
    """Return the preferred torch device available on the current machine."""
    if torch.cuda.is_available():
        return "cuda"
    mps_available = getattr(torch.backends, "mps", None)
    if mps_available and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_global_seed(seed: int) -> None:
    """Apply the same seed across numpy and torch for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_dependencies() -> None:
    """Ensure required dependencies are present before training starts."""
    if not DATASET_AVAILABLE:
        raise RuntimeError("CTDetectionDataset is not available. Please ensure the dataset module is importable.")
    if not ULTRALYTICS_AVAILABLE:
        raise RuntimeError("ultralytics package is required. Install with: pip install ultralytics")
    if not CV2_AVAILABLE:
        raise RuntimeError("OpenCV (cv2) is required to export images. Install with: pip install opencv-python")


class YOLOv11CTDataset:
    """Wrapper that adapts CTDetectionDataset samples to the YOLOv11 format."""

    def __init__(
        self,
        data_dir: str,
        split: str,
        include_negative_samples: bool,
        max_negative_per_patient: int,
        patient_ids: Optional[List[str]],
        image_size: int,
    ) -> None:
        # Keep the original split for directory structure, even when using specific patients
        actual_split = split  # Don't change to "all" when using patient_ids
        self.image_size = image_size
        self.split = split

        self.rcnn_dataset = CTDetectionDataset(
            data_root=data_dir,
            split=actual_split,
            target_size=image_size,
            specific_patients=patient_ids,
            transforms=transforms.Compose([transforms.ToTensor()]),
            include_negative_samples=include_negative_samples,
            max_negative_per_patient=max_negative_per_patient,
            format_type='yolo'  # 指定使用YOLO格式
        )

        self.samples = self.rcnn_dataset.samples

    def prepare_yolo_format(self, output_dir: Path) -> Path:
        """Convert samples to YOLO format and persist to disk."""
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV (cv2) is required to write YOLO format images.")

        base_dir = Path(output_dir) / f"yolo_dataset_{self.split}"
        images_dir = base_dir / "images" / self.split
        labels_dir = base_dir / "labels" / self.split

        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info("Preparing YOLOv11 dataset for %s split at %s", self.split, base_dir)

        for idx, sample in enumerate(
            tqdm(self.samples, desc=f"Converting {self.split} data", leave=False)
        ):
            item = self.rcnn_dataset[idx]
            image_tensor = item["image"]
            target = item.get("target", {})

            image_array = _tensor_to_image_array(image_tensor)
            patient_id = (sample.get("patient_id") or "unknown").replace("/", "_")
            sop_instance_uid = sample.get("sop_instance_uid") or f"uid_{idx:06d}"
            image_filename = f"{patient_id}_{sop_instance_uid}_{idx:06d}.png"
            image_path = images_dir / image_filename

            if image_array.ndim == 2:
                cv2.imwrite(str(image_path), image_array)
            else:
                cv2.imwrite(str(image_path), cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))

            label_filename = image_filename.replace(".png", ".txt")
            label_path = labels_dir / label_filename
            img_height, img_width = image_array.shape[:2]

            with open(label_path, "w", encoding="utf-8") as handle:
                boxes = target.get("boxes", [])
                labels = target.get("labels", [])
                if boxes is None or len(boxes) == 0:
                    continue
                
                # 新的數據集已經提供YOLO格式，直接使用
                if hasattr(boxes, 'shape') and len(boxes.shape) == 2 and boxes.shape[1] == 4:
                    # YOLO格式：(center_x, center_y, width, height) 已正規化
                    for i, (box, label) in enumerate(zip(boxes, labels)):
                        center_x, center_y, width, height = box.tolist() if hasattr(box, 'tolist') else box
                        class_id = max(_extract_label_id(label) - 1, 0)
                        handle.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
                else:
                    # 舊格式轉換（向後兼容）
                    for box, label in zip(boxes, labels):
                        x1, y1, x2, y2 = _extract_box_coordinates(box)
                        center_x = ((x1 + x2) / 2.0) / img_width
                        center_y = ((y1 + y2) / 2.0) / img_height
                        width = (x2 - x1) / img_width
                        height = (y2 - y1) / img_height

                        class_id = max(_extract_label_id(label) - 1, 0)
                        handle.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")

        dataset_config = base_dir / "dataset.yaml"
        dataset_config.write_text(
            textwrap.dedent(
                f"""
                path: {base_dir}
                train: images/{self.split if self.split == 'train' else 'train'}
                val: images/{self.split if self.split == 'val' else 'val'}

                nc: 1
                names: ['lesion']
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )

        LOGGER.info("YOLO dataset configuration saved to %s", dataset_config)
        return dataset_config


def _tensor_to_image_array(tensor: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor (C,H,W) or (H,W) into a uint8 numpy array."""
    array = tensor.detach().cpu()
    if array.dim() == 3 and array.shape[0] in (1, 3):
        if array.shape[0] == 1:
            image = array.squeeze(0).numpy()
        else:
            image = array.permute(1, 2, 0).numpy()
    elif array.dim() == 2:
        image = array.numpy()
    else:
        raise ValueError(f"Unexpected tensor shape for image conversion: {tuple(array.shape)}")

    image = np.clip(image, 0.0, 1.0) if image.max() <= 1.0 else np.clip(image, 0.0, 255.0)
    image_uint8 = (image * 255.0 if image.max() <= 1.0 else image).astype(np.uint8)
    return image_uint8


def _extract_box_coordinates(box: Any) -> Tuple[float, float, float, float]:
    """Return bounding box coordinates from tensor/list inputs."""
    if hasattr(box, "tolist"):
        x1, y1, x2, y2 = box.tolist()
    else:
        x1, y1, x2, y2 = map(float, box)
    return float(x1), float(y1), float(x2), float(y2)


def _extract_label_id(label: Any) -> int:
    """Extract integer class label from tensor-like or numeric inputs."""
    if hasattr(label, "item"):
        return int(label.item())
    return int(label)


def _resolve_model_name(model_size: str) -> str:
    valid_sizes = {"n", "s", "m", "l", "x"}
    if model_size not in valid_sizes:
        raise ValueError(f"Unsupported model_size '{model_size}'. Expected one of {sorted(valid_sizes)}")
    
    # Try to find model files in yolo_detection/models directory first
    current_dir = Path(__file__).resolve().parent
    models_dir = current_dir / "models"
    model_filename = f"yolo11{model_size}.pt"  # Note: no 'v' in filename
    local_model_path = models_dir / model_filename
    
    if local_model_path.exists():
        return str(local_model_path)
    
    # Try repository root as fallback
    repo_model_path = REPO_ROOT / model_filename
    if repo_model_path.exists():
        return str(repo_model_path)
    
    # Fallback to standard YOLOv11 naming (with 'v') - will download if needed
    standard_name = f"yolov11{model_size}.pt"
    return standard_name


def _write_combined_dataset_yaml(export_root: Path, train_config: Path, val_config: Optional[Path]) -> Path:
    """Create a dataset.yaml that references train and validation splits."""
    combined_config = export_root / "combined_dataset.yaml"

    train_images = (train_config.parent / "images" / "train").resolve()
    if val_config:
        val_images = (val_config.parent / "images" / "val").resolve()
    else:
        val_images = train_images

    train_rel = Path(os.path.relpath(train_images, export_root)).as_posix()
    val_rel = Path(os.path.relpath(val_images, export_root)).as_posix()

    combined_config.write_text(
        textwrap.dedent(
            f"""
            path: {export_root}
            train: {train_rel}
            val: {val_rel}

            nc: 1
            names: ['lesion']
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return combined_config


def _collect_patient_ids(dataset: YOLOv11CTDataset) -> List[str]:
    patient_ids: List[str] = []
    for idx, sample in enumerate(dataset.samples):
        patient_id = sample.get("patient_id")
        if not patient_id:
            patient_id = f"unknown_{idx:06d}"
        patient_ids.append(str(patient_id))
    return sorted(set(patient_ids))



def _split_patient_ids(patient_ids: List[str], val_ratio: float) -> Tuple[List[str], List[str]]:
    """Split patient identifiers into train/validation subsets by ratio."""
    if not patient_ids:
        raise ValueError("No patient IDs available for splitting.")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1 when deriving validation data from the train split.")
    if len(patient_ids) < 2:
        raise ValueError("At least two unique patient IDs are required to create a validation subset.")

    shuffled = np.random.permutation(patient_ids)
    val_count = max(1, int(round(len(patient_ids) * val_ratio)))
    val_count = min(val_count, len(patient_ids) - 1)

    val_ids = [str(pid) for pid in shuffled[:val_count]]
    train_ids = [str(pid) for pid in shuffled[val_count:]]
    return train_ids, val_ids


def prepare_single_run_datasets(config: TrainingConfig, export_root: Path) -> Tuple[DatasetArtifacts, Dict[str, Any]]:
    """Create YOLO-ready train/validation datasets for a single training run."""
    base_train_dataset = YOLOv11CTDataset(
        data_dir=config.data_dir,
        split=config.train_split,
        include_negative_samples=config.include_negative_samples,
        max_negative_per_patient=config.max_negative_per_patient,
        patient_ids=None,
        image_size=config.imgsz,
    )

    train_dataset: YOLOv11CTDataset = base_train_dataset
    val_dataset: Optional[YOLOv11CTDataset] = None

    train_patient_ids = _collect_patient_ids(base_train_dataset)
    val_patient_ids: List[str] = []

    if config.val_split:
        val_dataset = YOLOv11CTDataset(
            data_dir=config.data_dir,
            split=config.val_split,
            include_negative_samples=config.include_negative_samples,
            max_negative_per_patient=config.max_negative_per_patient,
            patient_ids=None,
            image_size=config.imgsz,
        )
        val_patient_ids = _collect_patient_ids(val_dataset)
    elif 0.0 < config.val_ratio < 1.0:
        train_ids, val_ids = _split_patient_ids(train_patient_ids, config.val_ratio)
        LOGGER.info("Using %.0f%% of training patients for validation (%d train / %d val)", config.val_ratio * 100, len(train_ids), len(val_ids))
        train_dataset = YOLOv11CTDataset(
            data_dir=config.data_dir,
            split=config.train_split,
            include_negative_samples=config.include_negative_samples,
            max_negative_per_patient=config.max_negative_per_patient,
            patient_ids=train_ids,
            image_size=config.imgsz,
        )
        val_dataset = YOLOv11CTDataset(
            data_dir=config.data_dir,
            split=config.train_split,  # validation數據也來自train split
            include_negative_samples=config.include_negative_samples,
            max_negative_per_patient=config.max_negative_per_patient,
            patient_ids=val_ids,
            image_size=config.imgsz,
        )
        train_patient_ids = _collect_patient_ids(train_dataset)
        val_patient_ids = _collect_patient_ids(val_dataset)
    else:
        val_dataset = None
        val_patient_ids = []

    train_stats_label = f"{config.train_split.title()} Split"
    train_stats = DATASET_STATS_FN(train_dataset.rcnn_dataset, train_stats_label)

    val_stats = None
    if val_dataset:
        if config.val_split:
            val_stats_label = f"{config.val_split.title()} Split"
        else:
            val_stats_label = 'Validation Split (from train)'
        val_stats = DATASET_STATS_FN(val_dataset.rcnn_dataset, val_stats_label)

    export_root.mkdir(parents=True, exist_ok=True)
    train_config_path = train_dataset.prepare_yolo_format(export_root)
    val_config_path = val_dataset.prepare_yolo_format(export_root) if val_dataset else None
    combined_config = _write_combined_dataset_yaml(
        export_root,
        Path(train_config_path),
        Path(val_config_path) if val_config_path else None,
    )

    train_dataset_size = len(train_dataset.rcnn_dataset)
    val_dataset_size = len(val_dataset.rcnn_dataset) if val_dataset else 0
    total_annotations = train_stats.get('total_annotations', 0) + (val_stats.get('total_annotations', 0) if val_stats else 0)

    dataset_statistics = {
        'train_stats': train_stats,
        'val_stats': val_stats,
        'train_patient_ids': train_patient_ids,
        'val_patient_ids': val_patient_ids,
        'total_dataset_size': train_dataset_size + val_dataset_size,
        'total_annotations': total_annotations,
        'validation_strategy': 'existing_split' if config.val_split else ('ratio_split' if val_dataset else 'none'),
        'val_ratio': config.val_ratio if not config.val_split else None,
    }

    if external_save_patient_lists:
        try:
            lists_root = export_root / 'patient_lists'
            lists_root.mkdir(parents=True, exist_ok=True)
            payload = {
                'train_patient_ids': train_patient_ids,
                'val_patient_ids': val_patient_ids,
                'train_stats': train_stats,
                'val_stats': val_stats or {},
                'total_patient_count': len(train_patient_ids) + len(val_patient_ids),
            }
            external_save_patient_lists(lists_root, payload)
        except Exception as exc:  # pragma: no cover - best effort export
            LOGGER.warning('Failed to save patient lists: %s', exc)

    artifacts = DatasetArtifacts(
        export_root=export_root,
        combined_config=combined_config,
        train_dataset_config=Path(train_config_path),
        val_dataset_config=Path(val_config_path) if val_config_path else None,
    )

    return artifacts, dataset_statistics


def calculate_yolo_metrics(results: Any) -> Dict[str, float]:
    """Extract common metrics from Ultralytics validation output."""
    metrics: Dict[str, float] = {}

    try:
        if hasattr(results, "results") and results.results:
            last_result = results.results[-1]
            box_metrics = getattr(last_result, "box", None)
            if box_metrics is not None:
                metrics["precision"] = float(getattr(box_metrics, "p", 0.0))
                metrics["recall"] = float(getattr(box_metrics, "r", 0.0))
                metrics["mAP@0.5"] = float(getattr(box_metrics, "map50", 0.0))
                metrics["mAP@[0.5:0.95]"] = float(getattr(box_metrics, "map", 0.0))

        val_metrics = getattr(results, "val", None)
        if val_metrics is not None and hasattr(val_metrics, "box"):
            box_metrics = val_metrics.box
            metrics.setdefault("precision", float(getattr(box_metrics, "p", 0.0)))
            metrics.setdefault("recall", float(getattr(box_metrics, "r", 0.0)))
            metrics.setdefault("mAP@0.5", float(getattr(box_metrics, "map50", 0.0)))
            metrics.setdefault("mAP@[0.5:0.95]", float(getattr(box_metrics, "map", 0.0)))

        precision = metrics.get("precision", 0.0)
        recall = metrics.get("recall", 0.0)
        metrics["f1_score"] = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.warning("Failed to extract YOLO metrics: %s", exc)

    return metrics


def evaluate_yolo_model(model_path: str, dataset_config: str, device: str = "auto") -> Dict[str, Any]:
    """Evaluate a trained YOLO model on the validation split."""
    try:
        model = YOLO(model_path)
        val_results = model.val(data=dataset_config, device=device)
        metrics = calculate_yolo_metrics(val_results)
        return {"metrics": metrics, "val_results": val_results, "model_path": model_path}
    except Exception as exc:  # pragma: no cover - evaluation should not crash run
        LOGGER.error("Validation failed for model %s: %s", model_path, exc)
        return {"metrics": {}, "error": str(exc)}


def run_training(config: TrainingConfig, artifacts: DatasetArtifacts, device: str) -> Tuple[Dict[str, Any], float]:
    """Train and evaluate a YOLO model once."""
    start_time = time.time()

    model_name = _resolve_model_name(config.model_size)
    LOGGER.info("Loading YOLO base model: %s", model_name)
    model = YOLO(model_name)

    train_args = config.build_train_args(
        data_config=str(artifacts.combined_config),
        project_dir=Path(config.save_dir),
        device=device,
    )

    LOGGER.info("Starting training with data config %s", artifacts.combined_config)
    results = model.train(**train_args)

    run_dir = Path(config.save_dir) / train_args["name"]
    best_model_path = run_dir / "weights" / "best.pt"
    if not best_model_path.exists():
        LOGGER.warning("best.pt not found, falling back to last.pt")
        alternative = run_dir / "weights" / "last.pt"
        if alternative.exists():
            best_model_path = alternative

    eval_results = evaluate_yolo_model(str(best_model_path), str(artifacts.combined_config), device=device)
    elapsed = time.time() - start_time

    run_summary = {
        "training_time": elapsed,
        "model_path": str(best_model_path),
        "config_path": str(artifacts.combined_config),
        "train_dataset_config": str(artifacts.train_dataset_config),
        "val_dataset_config": str(artifacts.val_dataset_config) if artifacts.val_dataset_config else None,
        "metrics": eval_results.get("metrics", {}),
        "train_args": train_args,
        "results": results,
    }

    return run_summary, elapsed


def save_results_summary(save_dir: Path, summary: Dict[str, Any]) -> Path:
    """Persist the aggregated results to JSON."""
    save_dir.mkdir(parents=True, exist_ok=True)
    results_file = save_dir / "yolov11_training_results.json"
    results_file.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    LOGGER.info("Results saved to %s", results_file)
    return results_file


def log_training_summary(summary: Dict[str, Any]) -> None:
    """Emit training summary to the logger."""
    LOGGER.info("\n%s", "=" * 60)
    LOGGER.info("YOLOv11 training complete")
    LOGGER.info("%s\n", "=" * 60)

    total_time = summary.get("training_time", 0.0)
    LOGGER.info("Total training time: %.2f hours", total_time / 3600 if total_time else 0.0)

    metrics = summary.get("metrics", {})
    if metrics:
        LOGGER.info("Evaluation metrics:")
        for key, value in metrics.items():
            LOGGER.info("  %s: %.3f", key, value)


def _train_yolov11_with_config(config: TrainingConfig) -> Dict[str, Any]:
    """Internal implementation that consumes a TrainingConfig instance."""
    _validate_dependencies()

    log_file = setup_logging(config.log_dir)
    LOGGER.info("Log file created at %s", log_file)

    device = select_device()
    LOGGER.info("Using device: %s", device)

    set_global_seed(config.random_seed)

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    export_root = save_dir / "dataset_exports"
    artifacts, dataset_statistics = prepare_single_run_datasets(config, export_root)

    run_summary, elapsed = run_training(config, artifacts, device)

    summary = {
        "metrics": run_summary.get("metrics", {}),
        "training_time": elapsed,
        "model_path": run_summary.get("model_path"),
        "config": config.as_dict(),
        "dataset_statistics": dataset_statistics,
        "log_file": log_file,
        "train_args": run_summary.get("train_args", {}),
        "train_dataset_config": run_summary.get("train_dataset_config"),
        "val_dataset_config": run_summary.get("val_dataset_config"),
    }

    results_file = save_results_summary(save_dir, summary)
    summary["results_file"] = str(results_file)

    log_training_summary(summary)

    return summary


def train_yolov11(
    data_dir: str,
    num_epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 0.01,
    save_dir: str = "./yolov11_models",
    log_dir: str = "./yolov11_logs",
    random_seed: int = 42,
    include_negative_samples: bool = True,
    max_negative_per_patient: int = 0,
    imgsz: int = 640,
    model_size: str = "n",
    val_ratio: float = 0.2,
    train_split: str = "train",
    val_split: Optional[str] = None,
) -> Dict[str, Any]:
    """Public API for running a single YOLOv11 training session."""
    config = TrainingConfig(
        data_dir=data_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        save_dir=save_dir,
        log_dir=log_dir,
        random_seed=random_seed,
        include_negative_samples=include_negative_samples,
        max_negative_per_patient=max_negative_per_patient,
        imgsz=imgsz,
        model_size=model_size,
        val_ratio=val_ratio,
        train_split=train_split,
        val_split=val_split,
    )
    return _train_yolov11_with_config(config)


def main() -> None:
    """Entry point for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="YOLOv11 Training for CT Detection"
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset root directory")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--save_dir", type=str, default="./yolov11_models", help="Directory to store trained models")
    parser.add_argument("--log_dir", type=str, default="./yolov11_logs", help="Directory to store logs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--include_negative",
        default=True,
        action="store_true",
        help="Include negative samples during training",
    )
    parser.add_argument(
        "--max_negative",
        type=int,
        default=40,
        help="Maximum negative samples per patient (0 for no limit)",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size for YOLO")
    parser.add_argument(
        "--model_size",
        type=str,
        default="n",
        choices=["n", "s", "m", "l", "x"],
        help="YOLOv11 model variant",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        help="Dataset split name used for training",
    )
    parser.add_argument(
        "--val_split",
        type=str,
        default="",
        help="Dataset split name used for validation (set empty string to use val_ratio from train split)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Ratio of training patients allocated to validation when --val_split is disabled",
    )

    args = parser.parse_args()

    if not ULTRALYTICS_AVAILABLE:
        print("Error: ultralytics package not installed. Install with: pip install ultralytics")
        sys.exit(1)

    val_split = args.val_split or None

    summary = train_yolov11(
        data_dir=args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        random_seed=args.seed,
        include_negative_samples=args.include_negative,
        max_negative_per_patient=args.max_negative,
        imgsz=args.imgsz,
        model_size=args.model_size,
        val_ratio=args.val_ratio,
        train_split=args.train_split,
        val_split=val_split,
    )

    print("\nTraining complete! Results saved to:", summary.get("results_file", args.save_dir))


if __name__ == "__main__":
    main()
