#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run YOLOv11 predictions on a folder of DICOM files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import numpy as np

try:
    import pydicom  # type: ignore
    from pydicom.errors import InvalidDicomError  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime
    pydicom = None  # type: ignore
    InvalidDicomError = Exception  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLOv11 predictions on DICOM files."
    )
    parser.add_argument(
        "--dicom_dir",
        required=True,
        type=Path,
        help="Directory containing DICOM files."
    )
    parser.add_argument(
        "--weights",
        required=True,
        type=Path,
        help="Path to YOLOv11 weights file (best.pt)."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./yolov11_predictions"),
        help="Directory to store prediction outputs."
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run inference on (auto, cpu, cuda, cuda:0, etc.)."
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="yolov11_inference",
        help="Name of the prediction run directory."
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for DICOM files."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on the number of DICOM files to process."
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Disable saving annotated prediction images."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose ultralytics output."
    )
    return parser.parse_args()


def collect_dicom_files(dicom_dir: Path, recursive: bool) -> List[Path]:
    if not dicom_dir.exists():
        raise FileNotFoundError(f"DICOM directory not found: {dicom_dir}")
    if not dicom_dir.is_dir():
        raise NotADirectoryError(f"Provided DICOM path is not a directory: {dicom_dir}")

    pattern = "**/*.dcm" if recursive else "*.dcm"
    files = sorted(dicom_dir.rglob("*.dcm") if recursive else dicom_dir.glob("*.dcm"))

    if not files:
        raise FileNotFoundError(f"No .dcm files found using pattern '{pattern}' in {dicom_dir}")

    return files


def _first_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0]
    if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
        try:
            return value[0]
        except Exception:  # pragma: no cover - defensive
            return value
    return value


def apply_window(array: np.ndarray, center: Any, width: Any) -> np.ndarray:
    if center is None or width is None:
        return array

    center = _first_value(center)
    width = _first_value(width)

    try:
        center = float(center)
        width = float(width)
    except (TypeError, ValueError):
        return array

    if width <= 0:
        return array

    lower = center - width / 2.0
    upper = center + width / 2.0
    return np.clip(array, lower, upper)


def normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32)
    min_val = float(np.min(array))
    array = array - min_val
    max_val = float(np.max(array))
    if max_val > 0:
        array = array / max_val
    return np.clip(array * 255.0, 0, 255).astype(np.uint8)


def load_dicom_image(path: Path) -> np.ndarray:
    if pydicom is None:  # pragma: no cover - dependency guard
        raise RuntimeError("pydicom is not installed. Install with 'pip install pydicom'.")

    try:
        dataset = pydicom.dcmread(str(path))
    except InvalidDicomError as exc:
        raise ValueError(f"Invalid DICOM file: {path}") from exc

    if not hasattr(dataset, "pixel_array"):
        raise ValueError(f"DICOM file has no pixel data: {path}")

    pixel_array = dataset.pixel_array.astype(np.float32)
    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    pixel_array = pixel_array * slope + intercept

    if pixel_array.ndim == 3 and pixel_array.shape[-1] == 3:
        pass
    elif pixel_array.ndim == 3 and pixel_array.shape[0] == 3:
        pixel_array = np.transpose(pixel_array, (1, 2, 0))
    elif pixel_array.ndim == 3:
        index = pixel_array.shape[0] // 2
        pixel_array = pixel_array[index, :, :]

    photometric = str(getattr(dataset, "PhotometricInterpretation", "")).upper()
    if photometric == "MONOCHROME1":
        pixel_array = pixel_array.max() - pixel_array

    pixel_array = apply_window(pixel_array, getattr(dataset, "WindowCenter", None), getattr(dataset, "WindowWidth", None))

    if pixel_array.ndim == 2:
        image = normalize_to_uint8(pixel_array)
        return np.stack([image, image, image], axis=-1)

    if pixel_array.ndim == 3 and pixel_array.shape[-1] == 3:
        return normalize_to_uint8(pixel_array)

    raise ValueError(f"Unsupported pixel array shape {pixel_array.shape} in {path}")


def extract_predictions(result: Any) -> List[Tuple[str, float, Tuple[float, float, float, float]]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    names = getattr(result, "names", {}) or {}

    xyxy_values = boxes.xyxy
    cls_values = boxes.cls if hasattr(boxes, "cls") else None
    conf_values = boxes.conf if hasattr(boxes, "conf") else None

    xyxy_list = xyxy_values.tolist() if hasattr(xyxy_values, "tolist") else list(xyxy_values)
    cls_list = cls_values.tolist() if cls_values is not None and hasattr(cls_values, "tolist") else []
    conf_list = conf_values.tolist() if conf_values is not None and hasattr(conf_values, "tolist") else []

    predictions: List[Tuple[str, float, Tuple[float, float, float, float]]] = []
    for idx, bbox in enumerate(xyxy_list):
        cls_id = int(cls_list[idx]) if idx < len(cls_list) else -1
        label = names.get(cls_id, str(cls_id))
        conf = float(conf_list[idx]) if idx < len(conf_list) else 0.0
        x1, y1, x2, y2 = (float(v) for v in bbox)
        predictions.append((label, conf, (x1, y1, x2, y2)))

    return predictions


def print_prediction_summary(index: int, total: int, path: Path, predictions: Sequence[Tuple[str, float, Tuple[float, float, float, float]]]) -> None:
    prefix = f"[{index}/{total}] {path.name}"
    if not predictions:
        print(f"{prefix}: no detections")
        return

    print(f"{prefix}: {len(predictions)} detections")
    for label, conf, bbox in predictions:
        x1, y1, x2, y2 = (round(value, 2) for value in bbox)
        print(f"  - {label} @ {conf:.3f} [{x1}, {y1}, {x2}, {y2}]")


def run_inference(
    model: Any,
    dicom_paths: Sequence[Path],
    output_dir: Path,
    imgsz: int,
    conf: float,
    device: str,
    run_name: str,
    save_images: bool,
    verbose: bool,
) -> None:
    save_root = output_dir / run_name
    save_root.mkdir(parents=True, exist_ok=True)

    total = len(dicom_paths)
    for index, dicom_path in enumerate(dicom_paths, start=1):
        try:
            image = load_dicom_image(dicom_path)
        except Exception as exc:
            print(f"[{index}/{total}] Skipping {dicom_path.name}: {exc}")
            continue

        results = model.predict(
            source=image,
            imgsz=imgsz,
            conf=conf,
            device=device,
            project=str(output_dir),
            name=run_name,
            exist_ok=True,
            save=save_images,
            verbose=verbose,
        )

        if not results:
            print(f"[{index}/{total}] {dicom_path.name}: no results returned")
            continue

        predictions = extract_predictions(results[0])
        print_prediction_summary(index, total, dicom_path, predictions)


def main() -> None:
    args = parse_args()

    dicom_dir = args.dicom_dir.expanduser().resolve()
    weights_path = args.weights.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if pydicom is None:
        print("Error: pydicom is not installed. Install it with 'pip install pydicom'.", file=sys.stderr)
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - dependency guard
        print(f"Error: ultralytics package is not installed: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        import torch
        torch_available = True
    except ImportError:  # pragma: no cover - torch required by ultralytics but guard for clarity
        torch_available = False

    if not weights_path.exists():
        print(f"Error: weights file not found: {weights_path}", file=sys.stderr)
        sys.exit(1)

    dicom_paths = collect_dicom_files(dicom_dir, args.recursive)
    if args.limit > 0:
        dicom_paths = dicom_paths[: args.limit]

    device = args.device
    if device == "auto":
        if torch_available:
            import torch  # safe: already imported above when available
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = "cpu"

    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or weights_path.stem
    save_images = not args.no_save

    print("Loading YOLO model...")
    model = YOLO(str(weights_path))
    print(f"Model loaded from {weights_path}")
    print(f"Processing {len(dicom_paths)} DICOM files from {dicom_dir}")
    print(f"Saving outputs to {output_dir / run_name}\n")

    try:
        run_inference(
            model=model,
            dicom_paths=dicom_paths,
            output_dir=output_dir,
            imgsz=args.imgsz,
            conf=args.conf,
            device=device,
            run_name=run_name,
            save_images=save_images,
            verbose=args.verbose,
        )
    except KeyboardInterrupt:  # pragma: no cover - user interruption
        print("\nInference interrupted by user.")


if __name__ == "__main__":
    main()
