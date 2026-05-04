#!/usr/bin/env python3
"""
Analyze false-positive box size distribution from exported case analysis.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


def _load_summaries(case_analysis_dir: Path) -> Iterable[Dict]:
    for path in sorted(case_analysis_dir.rglob("summary.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["_summary_path"] = str(path)
        yield data


def _max_diameter_mm(box: Sequence[float], spacing: Sequence[float]) -> float:
    dy = max(float(box[3]) - float(box[0]), 0.0) * float(spacing[0])
    dx = max(float(box[4]) - float(box[1]), 0.0) * float(spacing[1])
    dz = max(float(box[5]) - float(box[2]), 0.0) * float(spacing[2])
    return float(max(dy, dx, dz))


def _volume_mm3(box: Sequence[float], spacing: Sequence[float]) -> float:
    dy = max(float(box[3]) - float(box[0]), 0.0) * float(spacing[0])
    dx = max(float(box[4]) - float(box[1]), 0.0) * float(spacing[1])
    dz = max(float(box[5]) - float(box[2]), 0.0) * float(spacing[2])
    return float(dy * dx * dz)


def _size_bucket(diameter_mm: float, small_mm: float, medium_mm: float) -> str:
    if diameter_mm <= small_mm:
        return "small"
    if diameter_mm <= medium_mm:
        return "medium"
    return "large"


def _summarize(rows: List[Dict], prefix: str) -> Dict:
    values = np.asarray([float(row["diameter_mm"]) for row in rows], dtype=np.float32)
    return {
        f"{prefix}_count": int(len(rows)),
        f"{prefix}_diameter_mean_mm": float(np.mean(values)) if len(values) else 0.0,
        f"{prefix}_diameter_median_mm": float(np.median(values)) if len(values) else 0.0,
        f"{prefix}_diameter_p90_mm": float(np.percentile(values, 90)) if len(values) else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze FP size distribution from case_analysis summary.json files.")
    parser.add_argument("--case_analysis_dir", required=True, help="Directory produced by --export_case_analysis")
    parser.add_argument("--output_json", default=None, help="Summary JSON path")
    parser.add_argument("--output_csv", default=None, help="Per-prediction CSV path")
    parser.add_argument("--spacing", type=float, nargs=3, default=[0.703125, 0.703125, 1.25], metavar=("SY", "SX", "SZ"))
    parser.add_argument("--small_mm", type=float, default=8.0, help="small bucket upper diameter")
    parser.add_argument("--medium_mm", type=float, default=20.0, help="medium bucket upper diameter")
    args = parser.parse_args()

    case_analysis_dir = Path(args.case_analysis_dir)
    output_json = Path(args.output_json) if args.output_json else case_analysis_dir / "fp_size_report.json"
    output_csv = Path(args.output_csv) if args.output_csv else case_analysis_dir / "fp_size_rows.csv"

    rows: List[Dict] = []
    missing_box_summaries = 0
    for summary in _load_summaries(case_analysis_dir):
        boxes = summary.get("pred_boxes_yxz")
        scores = summary.get("pred_scores", [])
        is_tp = summary.get("pred_is_tp")
        best_iou = summary.get("pred_best_iou", [])
        if boxes is None or is_tp is None:
            missing_box_summaries += 1
            continue
        case_name = str(summary.get("case_name", Path(summary["_summary_path"]).parent.name))
        for idx, box in enumerate(boxes):
            diameter = _max_diameter_mm(box, args.spacing)
            row = {
                "case_name": case_name,
                "pred_index": idx,
                "label": "TP" if bool(is_tp[idx]) else "FP",
                "score": float(scores[idx]) if idx < len(scores) else None,
                "best_iou": float(best_iou[idx]) if idx < len(best_iou) else None,
                "diameter_mm": diameter,
                "volume_mm3": _volume_mm3(box, args.spacing),
                "size_bucket": _size_bucket(diameter, args.small_mm, args.medium_mm),
                "box_y1": float(box[0]),
                "box_x1": float(box[1]),
                "box_z1": float(box[2]),
                "box_y2": float(box[3]),
                "box_x2": float(box[4]),
                "box_z2": float(box[5]),
            }
            rows.append(row)

    fp_rows = [row for row in rows if row["label"] == "FP"]
    tp_rows = [row for row in rows if row["label"] == "TP"]

    def bucket_counts(subset: List[Dict]) -> Dict[str, int]:
        return {
            "small": sum(1 for row in subset if row["size_bucket"] == "small"),
            "medium": sum(1 for row in subset if row["size_bucket"] == "medium"),
            "large": sum(1 for row in subset if row["size_bucket"] == "large"),
        }

    report = {
        "case_analysis_dir": str(case_analysis_dir),
        "spacing_yxz": [float(v) for v in args.spacing],
        "small_mm": float(args.small_mm),
        "medium_mm": float(args.medium_mm),
        "n_predictions": int(len(rows)),
        "n_missing_box_summaries": int(missing_box_summaries),
        "fp": {
            **_summarize(fp_rows, "fp"),
            "bucket_counts": bucket_counts(fp_rows),
        },
        "tp": {
            **_summarize(tp_rows, "tp"),
            "bucket_counts": bucket_counts(tp_rows),
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_name",
        "pred_index",
        "label",
        "score",
        "best_iou",
        "diameter_mm",
        "volume_mm3",
        "size_bucket",
        "box_y1",
        "box_x1",
        "box_z1",
        "box_y2",
        "box_x2",
        "box_z2",
    ]
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved JSON: {output_json}")
    print(f"Saved CSV: {output_csv}")


if __name__ == "__main__":
    main()
