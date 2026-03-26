#!/usr/bin/env python3
"""
Convert LUNA16-New (TCIA/NBIA downloaded LIDC-IDRI DICOM + XML) into
RetinaNet-ready dataset JSON.

Output JSON schema matches detection.retinanet.prepare_data:
{
  "training": [ {"image": "...mhd", "box": [[x1,y1,z1,x2,y2,z2], ...], "label":[1,...], ...}, ... ],
  "validation": [...],
  "testing": [...]
}

Notes:
- Boxes are generated from LIDC XML ROI edge points.
- Physical coordinates are converted from LPS to RAS.
- One class only: nodule (label=1).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


XML_NS = {"ns": "http://www.nih.gov"}


def _default_output_json() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "detection" / "manifests" / "dataset_luna16_new.json"


@dataclass
class SeriesEntry:
    series_uid: str
    series_dir: Path


def _normalize_file_location(manifest_dir: Path, location: str) -> Path:
    cleaned = str(location).strip().replace("\\", "/")
    if cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return (manifest_dir / cleaned).resolve()


def discover_series(base_dir: Path) -> Dict[str, SeriesEntry]:
    series_map: Dict[str, SeriesEntry] = {}
    metadata_files = sorted(base_dir.rglob("metadata.csv"))
    if not metadata_files:
        raise FileNotFoundError(f"No metadata.csv found under: {base_dir}")

    for meta in metadata_files:
        manifest_dir = meta.parent
        try:
            df = pd.read_csv(meta)
        except Exception as exc:
            logger.warning("Skip unreadable metadata: %s (%s)", meta, exc)
            continue

        if "Series UID" not in df.columns or "File Location" not in df.columns:
            logger.warning("Skip metadata without required columns: %s", meta)
            continue

        for _, row in df.iterrows():
            uid = str(row["Series UID"]).strip()
            loc = str(row["File Location"]).strip()
            if not uid or uid == "nan" or not loc or loc == "nan":
                continue
            series_dir = _normalize_file_location(manifest_dir, loc)
            if not series_dir.exists():
                continue
            if uid not in series_map:
                series_map[uid] = SeriesEntry(series_uid=uid, series_dir=series_dir)

    return series_map


def _get_series_filenames(series_dir: Path) -> List[str]:
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(series_dir))
    if not series_ids:
        return []
    # LIDC series directory should contain one series id.
    series_id = series_ids[0]
    return list(reader.GetGDCMSeriesFileNames(str(series_dir), series_id))


def _read_slice_meta(file_path: str) -> Tuple[Optional[str], Optional[float]]:
    r = sitk.ImageFileReader()
    r.SetFileName(file_path)
    r.LoadPrivateTagsOn()
    r.ReadImageInformation()

    sop_uid: Optional[str] = None
    z_pos: Optional[float] = None

    if r.HasMetaDataKey("0008|0018"):
        sop_uid = r.GetMetaData("0008|0018").strip()
    if r.HasMetaDataKey("0020|0032"):
        ipp = r.GetMetaData("0020|0032")
        parts = [p.strip() for p in ipp.split("\\")]
        if len(parts) >= 3:
            try:
                z_pos = float(parts[2])
            except ValueError:
                z_pos = None

    return sop_uid, z_pos


def _load_volume_and_maps(
    series_dir: Path,
    output_image_path: Path,
    write_image: bool,
) -> Tuple[sitk.Image, Dict[str, int], List[Optional[float]]]:
    file_names = _get_series_filenames(series_dir)
    if not file_names:
        raise RuntimeError(f"No readable DICOM series under: {series_dir}")

    # Build SOP UID -> slice index map, plus z-position list for fallback.
    sop_to_k: Dict[str, int] = {}
    z_positions: List[Optional[float]] = []
    for k, fn in enumerate(file_names):
        sop_uid, z_pos = _read_slice_meta(fn)
        if sop_uid:
            sop_to_k[sop_uid] = k
        z_positions.append(z_pos)

    if output_image_path.exists():
        image = sitk.ReadImage(str(output_image_path))
        return image, sop_to_k, z_positions

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(file_names)
    image = reader.Execute()

    if write_image:
        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(image, str(output_image_path))

    return image, sop_to_k, z_positions


def _nearest_k_by_z(target_z: float, z_positions: List[Optional[float]]) -> Optional[int]:
    pairs = [(idx, z) for idx, z in enumerate(z_positions) if z is not None]
    if not pairs:
        return None
    arr = np.array([p[1] for p in pairs], dtype=np.float64)
    nearest_idx = int(np.argmin(np.abs(arr - target_z)))
    return int(pairs[nearest_idx][0])


def _parse_nodule_boxes_from_xml(
    xml_path: Path,
    image: sitk.Image,
    sop_to_k: Dict[str, int],
    z_positions: List[Optional[float]],
    min_diameter_mm: float,
) -> List[List[float]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_x, size_y, size_z = image.GetSize()
    boxes: List[List[float]] = []

    for nodule in root.findall(".//ns:unblindedReadNodule", XML_NS):
        points_ras: List[Tuple[float, float, float]] = []

        for roi in nodule.findall("ns:roi", XML_NS):
            inclusion = (roi.findtext("ns:inclusion", default="", namespaces=XML_NS) or "").strip().upper()
            if inclusion and inclusion != "TRUE":
                continue

            sop_uid = (roi.findtext("ns:imageSOP_UID", default="", namespaces=XML_NS) or "").strip()
            k = sop_to_k.get(sop_uid)

            if k is None:
                z_text = (roi.findtext("ns:imageZposition", default="", namespaces=XML_NS) or "").strip()
                if z_text:
                    try:
                        z_target = float(z_text)
                        k = _nearest_k_by_z(z_target, z_positions)
                    except ValueError:
                        k = None

            if k is None:
                continue

            for edge in roi.findall("ns:edgeMap", XML_NS):
                x_text = (edge.findtext("ns:xCoord", default="", namespaces=XML_NS) or "").strip()
                y_text = (edge.findtext("ns:yCoord", default="", namespaces=XML_NS) or "").strip()
                if not x_text or not y_text:
                    continue

                try:
                    x = int(float(x_text))
                    y = int(float(y_text))
                except ValueError:
                    continue

                x = min(max(x, 0), size_x - 1)
                y = min(max(y, 0), size_y - 1)
                k_clamped = min(max(int(k), 0), size_z - 1)

                lps = image.TransformIndexToPhysicalPoint((x, y, k_clamped))
                ras = (-float(lps[0]), -float(lps[1]), float(lps[2]))
                points_ras.append(ras)

        if not points_ras:
            continue

        arr = np.asarray(points_ras, dtype=np.float64)
        pmin = arr.min(axis=0)
        pmax = arr.max(axis=0)
        est_diameter = float(np.max(pmax - pmin))
        if est_diameter < min_diameter_mm:
            continue

        box = [float(pmin[0]), float(pmin[1]), float(pmin[2]), float(pmax[0]), float(pmax[1]), float(pmax[2])]
        boxes.append(box)

    return boxes


def split_dataset(
    samples: List[dict],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict:
    random.seed(seed)
    random.shuffle(samples)

    n_total = len(samples)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    return {
        "training": samples[:n_train],
        "validation": samples[n_train:n_train + n_val],
        "testing": samples[n_train + n_val:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare RetinaNet dataset JSON from LUNA16-New DICOM/XML.")
    parser.add_argument("--base_dir", type=str, required=True, help="LUNA16-New root directory.")
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="Output dataset json path. Default: detection/manifests/dataset_luna16_new.json",
    )
    parser.add_argument(
        "--output_image_dir",
        type=str,
        default="",
        help="Directory to store converted .mhd files. Default: <base_dir>/retina_mhd",
    )
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_diameter_mm", type=float, default=3.0)
    parser.add_argument("--max_series", type=int, default=0, help="Debug only: limit number of series (0=no limit).")
    parser.add_argument("--no_image_write", action="store_true", help="Do not write .mhd (requires existing files).")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"Base dir not found: {base_dir}")

    if args.output_json:
        output_json = Path(args.output_json).resolve()
    else:
        output_json = _default_output_json().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    if args.output_image_dir:
        image_dir = Path(args.output_image_dir).resolve()
    else:
        image_dir = (base_dir / "retina_mhd").resolve()
    image_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Discovering series from metadata.csv ...")
    series_map = discover_series(base_dir)
    if not series_map:
        raise RuntimeError("No series discovered from metadata.csv")

    series_items = sorted(series_map.items(), key=lambda kv: kv[0])
    if args.max_series > 0:
        series_items = series_items[: args.max_series]
    logger.info("Series discovered: %d", len(series_items))

    samples: List[dict] = []
    stats = {
        "series_total": len(series_items),
        "series_converted": 0,
        "series_error": 0,
        "series_no_xml": 0,
        "series_no_xml_included": 0,
        "samples_missing_image_skipped": 0,
        "boxes_total": 0,
    }

    for idx, (uid, entry) in enumerate(series_items, start=1):
        if idx % 50 == 0:
            logger.info("Progress: %d / %d", idx, len(series_items))

        image_path = image_dir / f"{uid}.mhd"
        xml_files = sorted(entry.series_dir.glob("*.xml"))
        if not xml_files:
            stats["series_no_xml"] += 1
            try:
                # Ensure negative samples also have a valid image file.
                _load_volume_and_maps(
                    series_dir=entry.series_dir,
                    output_image_path=image_path,
                    write_image=(not args.no_image_write),
                )
                if args.no_image_write and (not image_path.exists()):
                    raise FileNotFoundError(
                        f"--no_image_write was set but image file does not exist: {image_path}"
                    )

                samples.append(
                    {
                        "image": str(image_path),
                        "box": [],
                        "label": [],
                        "seriesuid": uid,
                        "dataset_type": "luna16_new",
                    }
                )
                stats["series_no_xml_included"] += 1
            except Exception as exc:
                logger.warning("Series failed (no xml, %s): %s", uid, exc)
                stats["series_error"] += 1
            continue

        # Typical LIDC series has one xml. If multiple, combine all boxes.
        try:
            image, sop_to_k, z_positions = _load_volume_and_maps(
                series_dir=entry.series_dir,
                output_image_path=image_path,
                write_image=(not args.no_image_write),
            )
            if args.no_image_write and (not image_path.exists()):
                raise FileNotFoundError(
                    f"--no_image_write was set but image file does not exist: {image_path}"
                )

            all_boxes: List[List[float]] = []
            for xml_path in xml_files:
                boxes = _parse_nodule_boxes_from_xml(
                    xml_path=xml_path,
                    image=image,
                    sop_to_k=sop_to_k,
                    z_positions=z_positions,
                    min_diameter_mm=args.min_diameter_mm,
                )
                all_boxes.extend(boxes)

            labels = [1] * len(all_boxes)
            stats["boxes_total"] += len(all_boxes)
            stats["series_converted"] += 1

            samples.append(
                {
                    "image": str(image_path),
                    "box": all_boxes,
                    "label": labels,
                    "seriesuid": uid,
                    "dataset_type": "luna16_new",
                }
            )
        except Exception as exc:
            logger.warning("Series failed (%s): %s", uid, exc)
            stats["series_error"] += 1

    # Final safety check: do not write samples with missing image paths.
    filtered_samples: List[dict] = []
    for s in samples:
        image_path = Path(str(s.get("image", "")))
        if image_path.exists():
            filtered_samples.append(s)
        else:
            stats["samples_missing_image_skipped"] += 1
            logger.warning("Skip sample with missing image: %s", image_path)
    samples = filtered_samples

    logger.info("Building split json ...")
    dataset = split_dataset(
        samples=samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    summary = {
        **stats,
        "output_json": str(output_json),
        "output_image_dir": str(image_dir),
        "samples_total": len(samples),
        "samples_training": len(dataset["training"]),
        "samples_validation": len(dataset["validation"]),
        "samples_testing": len(dataset["testing"]),
        "min_diameter_mm": args.min_diameter_mm,
    }
    summary_path = output_json.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("Done. dataset json: %s", output_json)
    logger.info("Summary: %s", summary_path)
    logger.info(
        "Samples=%d, boxes=%d, errors=%d, no_xml=%d",
        summary["samples_total"],
        summary["boxes_total"],
        summary["series_error"],
        summary["series_no_xml"],
    )


if __name__ == "__main__":
    main()
