#!/usr/bin/env python3
"""
Apply MedSAM2 mask-morphology rules to an existing candidate feature CSV.

Use this after analyze_medsam2_candidate_features.py so rule tuning does not
require rerunning MedSAM2 segmentation.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


def _read_rows(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _safe_float(row: Dict, key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        if value is None or value == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _apply_round_rule(row: Dict, args: argparse.Namespace) -> Tuple[bool, str]:
    score = _safe_float(row, "score")
    diameter = _safe_float(row, "bbox_max_diameter_mm")
    mask_valid = _safe_float(row, "mask_valid")
    round_planes = _safe_float(row, "round_planes_elong_le2")
    pca_elongation = _safe_float(row, "pca_elongation", 999.0)
    mask_bbox_fill = _safe_float(row, "mask_bbox_fill")
    side_area_ratio = _safe_float(row, "best_side_area_ratio")

    if score >= float(args.protect_score):
        return True, "protect_score"
    if float(args.protect_small_mm) > 0 and diameter <= float(args.protect_small_mm):
        return True, "protect_small"
    if mask_valid <= 0:
        return True, "segmentation_failed_keep"

    bad = []
    if round_planes < float(args.min_round_planes):
        bad.append("not_enough_round_planes")
    if pca_elongation > float(args.max_pca_elongation):
        bad.append("elongated_3d")
    if mask_bbox_fill < float(args.min_mask_bbox_fill):
        bad.append("low_bbox_fill")
    if side_area_ratio < float(args.min_side_area_ratio):
        bad.append("poor_axial_side_consistency")

    if "not_enough_round_planes" not in bad:
        return True, "round_planes_ok"
    if len(bad) >= int(args.min_bad_conditions):
        return False, ";".join(bad)
    return True, "round_planes_bad_but_not_enough_evidence"


def _summarize(rows: List[Dict]) -> Dict:
    summary = {}
    for label in ("TP", "FP"):
        subset = [r for r in rows if str(r.get("label", "")).upper() == label]
        removed = [r for r in subset if str(r.get("medsam2_morph_keep", "")).lower() == "false"]
        summary[label] = {
            "count": len(subset),
            "removed": len(removed),
            "kept": len(subset) - len(removed),
            "removed_rate": float(len(removed) / len(subset)) if subset else 0.0,
        }
    return summary


def _write_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply round-plane MedSAM2 morphology rule to candidate CSV.")
    parser.add_argument("--features_csv", required=True)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--protect_score", type=float, default=0.95)
    parser.add_argument("--protect_small_mm", type=float, default=6.0)
    parser.add_argument("--min_round_planes", type=float, default=2.0)
    parser.add_argument("--max_pca_elongation", type=float, default=8.0)
    parser.add_argument("--min_mask_bbox_fill", type=float, default=0.01)
    parser.add_argument("--min_side_area_ratio", type=float, default=0.10)
    parser.add_argument("--min_bad_conditions", type=int, default=2)
    args = parser.parse_args()

    features_csv = Path(args.features_csv)
    output_csv = Path(args.output_csv) if args.output_csv else features_csv.with_name(features_csv.stem + "_roundrule.csv")
    output_json = Path(args.output_json) if args.output_json else features_csv.with_name(features_csv.stem + "_roundrule.json")

    rows = _read_rows(features_csv)
    for row in rows:
        keep, reason = _apply_round_rule(row, args)
        row["medsam2_morph_keep"] = bool(keep)
        row["medsam2_morph_remove_reason"] = reason

    report = {
        "features_csv": str(features_csv),
        "output_csv": str(output_csv),
        "rule": {
            "mode": "round_planes_plus_bad_condition",
            "protect_score": float(args.protect_score),
            "protect_small_mm": float(args.protect_small_mm),
            "min_round_planes": float(args.min_round_planes),
            "max_pca_elongation": float(args.max_pca_elongation),
            "min_mask_bbox_fill": float(args.min_mask_bbox_fill),
            "min_side_area_ratio": float(args.min_side_area_ratio),
            "min_bad_conditions": int(args.min_bad_conditions),
            "description": "Remove only when candidate has fewer than min_round_planes and enough additional bad morphology evidence.",
        },
        "rule_summary": _summarize(rows),
    }

    _write_csv(rows, output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved CSV: {output_csv}")
    print(f"Saved JSON: {output_json}")


if __name__ == "__main__":
    main()
