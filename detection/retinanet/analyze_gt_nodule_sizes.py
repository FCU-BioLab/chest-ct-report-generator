#!/usr/bin/env python3
"""
Analyze GT nodule size distribution and which nodules are missed in evaluation.
"""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Optional, Sequence

import numpy as np

from .dataset import prepare_datalist


SIZE_BINS_MM = [
    ("<=4", None, 4.0),
    ("4-6", 4.0, 6.0),
    ("6-8", 6.0, 8.0),
    ("8-10", 8.0, 10.0),
    ("10-15", 10.0, 15.0),
    ("15-20", 15.0, 20.0),
    (">20", 20.0, None),
]


def _safe_float(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _size_bin_label(max_diameter_mm: float) -> str:
    for label, lo, hi in SIZE_BINS_MM:
        if lo is None and max_diameter_mm <= hi:
            return label
        if hi is None and max_diameter_mm > lo:
            return label
        if lo is not None and hi is not None and lo < max_diameter_mm <= hi:
            return label
    return "unknown"


def _summarize(values: Sequence[float]) -> Dict[str, Optional[float]]:
    vals = [float(v) for v in values]
    if not vals:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "p25": None,
            "p75": None,
        }
    arr = np.asarray(vals, dtype=np.float32)
    return {
        "count": int(arr.size),
        "mean": float(mean(vals)),
        "median": float(median(vals)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }


def _load_case_summary(case_analysis_dir: Path, case_name: str) -> Optional[Dict]:
    summary_path = case_analysis_dir / case_name / "summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_trace_rows(case_analysis_dir: Path, case_name: str, summary: Optional[Dict]) -> List[Dict]:
    if summary and isinstance(summary.get("fpr_trace"), list):
        return summary["fpr_trace"]
    trace_path = case_analysis_dir / case_name / "fpr_trace.json"
    if not trace_path.exists():
        return []
    with open(trace_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def _classify_miss_stage(gt_index: int, found_final: bool, trace_rows: Sequence[Dict]) -> str:
    if found_final:
        return "found_final"

    gt_rows = [
        row for row in trace_rows
        if bool(row.get("candidate_is_tp", False)) and int(row.get("matched_gt_index", -1)) == gt_index
    ]
    if not gt_rows:
        return "no_tp_candidate"
    if any(bool(row.get("keep_after_final_score", False)) for row in gt_rows):
        return "removed_after_final_postprocess"
    if any(bool(row.get("keep_after_fpr", False)) for row in gt_rows):
        return "removed_by_final_threshold_or_nms"
    return "removed_by_fpr"


def _make_row(case_name: str, record: Dict, gt_index: int, box: Sequence[float], found_final: bool, summary: Optional[Dict], trace_rows: Sequence[Dict]) -> Dict:
    x1, y1, z1, x2, y2, z2 = [float(v) for v in box]
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)
    max_d = max(dx, dy, dz)
    mean_d = (dx + dy + dz) / 3.0
    vol = dx * dy * dz
    miss_stage = _classify_miss_stage(gt_index, found_final, trace_rows)
    return {
        "case_name": case_name,
        "seriesuid": str(record.get("seriesuid", "")),
        "image": str(record.get("image", "")),
        "gt_index": int(gt_index),
        "found_final": bool(found_final),
        "miss_stage": miss_stage,
        "dx_mm": dx,
        "dy_mm": dy,
        "dz_mm": dz,
        "max_diameter_mm": max_d,
        "mean_diameter_mm": mean_d,
        "box_volume_mm3": vol,
        "size_bin": _size_bin_label(max_d),
        "case_n_gt": int(summary.get("n_gt", 0)) if summary else None,
        "case_n_pred": int(summary.get("n_pred", 0)) if summary else None,
        "case_n_fn": int(summary.get("n_fn", 0)) if summary else None,
        "case_f1": float(summary.get("case_f1", 0.0)) if summary else None,
        "case_precision": float(summary.get("case_precision", 0.0)) if summary else None,
        "case_recall": float(summary.get("case_recall", 0.0)) if summary else None,
    }


def _aggregate_bins(rows: Sequence[Dict]) -> List[Dict]:
    out = []
    for label, _, _ in SIZE_BINS_MM:
        items = [row for row in rows if row["size_bin"] == label]
        missed = [row for row in items if not row["found_final"]]
        out.append({
            "size_bin": label,
            "n_gt": int(len(items)),
            "n_missed": int(len(missed)),
            "miss_rate": float(len(missed) / len(items)) if items else None,
            "mean_max_diameter_mm": float(mean([row["max_diameter_mm"] for row in items])) if items else None,
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze GT nodule sizes and which nodules are missed.")
    parser.add_argument("--data_path", required=True, help="dataset JSON path")
    parser.add_argument("--case_analysis_dir", required=True, help="case_analysis output directory")
    parser.add_argument("--eval_split", default="val", choices=["train", "val", "test", "training", "validation", "testing"], help="dataset split corresponding to case names")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_prefix", default=None, help="output prefix without extension; defaults under case_analysis_dir")
    args = parser.parse_args()

    case_analysis_dir = Path(args.case_analysis_dir)
    if not case_analysis_dir.exists():
        raise FileNotFoundError(f"case_analysis_dir not found: {case_analysis_dir}")

    split_items = prepare_datalist(
        data_path=args.data_path,
        section=args.eval_split,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    prefix = Path(args.output_prefix) if args.output_prefix else case_analysis_dir / f"gt_nodule_size_analysis_{args.eval_split}"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    json_path = prefix.with_suffix(".json")

    rows: List[Dict] = []
    missing_case_summaries: List[str] = []

    split_prefix = "val" if args.eval_split in {"val", "validation"} else ("test" if args.eval_split in {"test", "testing"} else "train")

    for idx, record in enumerate(split_items):
        case_name = f"{split_prefix}_{idx:04d}"
        summary = _load_case_summary(case_analysis_dir, case_name)
        if summary is None:
            missing_case_summaries.append(case_name)
            continue

        trace_rows = _load_trace_rows(case_analysis_dir, case_name, summary)
        matched_final = {int(v) for v in summary.get("pred_match_gt", []) if int(v) >= 0}
        boxes = record.get("box", []) or []

        for gt_index, box in enumerate(boxes):
            rows.append(
                _make_row(
                    case_name=case_name,
                    record=record,
                    gt_index=gt_index,
                    box=box,
                    found_final=(gt_index in matched_final),
                    summary=summary,
                    trace_rows=trace_rows,
                )
            )

    fieldnames = [
        "case_name",
        "seriesuid",
        "image",
        "gt_index",
        "found_final",
        "miss_stage",
        "dx_mm",
        "dy_mm",
        "dz_mm",
        "max_diameter_mm",
        "mean_diameter_mm",
        "box_volume_mm3",
        "size_bin",
        "case_n_gt",
        "case_n_pred",
        "case_n_fn",
        "case_f1",
        "case_precision",
        "case_recall",
    ]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    missed_rows = [row for row in rows if not row["found_final"]]
    found_rows = [row for row in rows if row["found_final"]]

    miss_stage_counts: Dict[str, int] = {}
    for row in missed_rows:
        miss_stage_counts[row["miss_stage"]] = miss_stage_counts.get(row["miss_stage"], 0) + 1

    report = {
        "data_path": args.data_path,
        "case_analysis_dir": str(case_analysis_dir),
        "eval_split": args.eval_split,
        "n_cases_in_split": int(len(split_items)),
        "n_cases_with_summary": int(len(split_items) - len(missing_case_summaries)),
        "n_missing_case_summaries": int(len(missing_case_summaries)),
        "missing_case_summaries": missing_case_summaries,
        "n_gt_total": int(len(rows)),
        "n_gt_found_final": int(len(found_rows)),
        "n_gt_missed_final": int(len(missed_rows)),
        "miss_rate": float(len(missed_rows) / len(rows)) if rows else None,
        "all_gt_max_diameter_mm": _summarize([row["max_diameter_mm"] for row in rows]),
        "found_gt_max_diameter_mm": _summarize([row["max_diameter_mm"] for row in found_rows]),
        "missed_gt_max_diameter_mm": _summarize([row["max_diameter_mm"] for row in missed_rows]),
        "all_gt_box_volume_mm3": _summarize([row["box_volume_mm3"] for row in rows]),
        "missed_gt_box_volume_mm3": _summarize([row["box_volume_mm3"] for row in missed_rows]),
        "miss_stage_counts": miss_stage_counts,
        "size_bin_stats": _aggregate_bins(rows),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("GT nodule size analysis complete.")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"GT total: {report['n_gt_total']} | missed: {report['n_gt_missed_final']} | miss_rate: {report['miss_rate']:.4f}" if rows else "No GT rows found.")
    if missed_rows:
        print(
            "Missed GT max diameter (mm): "
            f"median={report['missed_gt_max_diameter_mm']['median']:.3f} "
            f"p25={report['missed_gt_max_diameter_mm']['p25']:.3f} "
            f"p75={report['missed_gt_max_diameter_mm']['p75']:.3f}"
        )
    if miss_stage_counts:
        print(f"Miss stages: {miss_stage_counts}")


if __name__ == "__main__":
    main()
