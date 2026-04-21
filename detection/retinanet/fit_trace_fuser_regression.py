#!/usr/bin/env python3
"""
Fit a final Stage-2 regression model from exported FPR traces.

This learns a logistic regression on proposal-level labels using only inference-available
features:
  - det_score
  - fpr_prob
  - interaction = det_score * fpr_prob
  - optional morphology features exported by case_analysis traces

It reports:
  - best-F1 operating threshold
  - coverage-safe threshold: keep at least one matched proposal per covered GT
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class TraceRow:
    case_name: str
    proposal_index: int
    det_score: float
    fpr_prob: float
    interaction: float
    features: Dict[str, float]
    candidate_is_tp: bool
    candidate_is_fp: bool
    matched_gt_index: int
    best_iou: float


def _load_rows(case_analysis_dir: Path) -> Tuple[List[TraceRow], Dict[str, int]]:
    rows: List[TraceRow] = []
    gt_count_by_case: Dict[str, int] = {}

    cases_json = case_analysis_dir / "cases.json"
    if cases_json.exists():
        summaries = json.loads(cases_json.read_text(encoding="utf-8"))
        for item in summaries:
            gt_count_by_case[str(item.get("case_name"))] = int(item.get("n_gt", 0))

    for case_dir in sorted(p for p in case_analysis_dir.iterdir() if p.is_dir()):
        trace_path = case_dir / "fpr_trace.json"
        if not trace_path.exists():
            continue
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        for item in trace:
            det_score = float(item.get("det_score", 0.0))
            fpr_prob = float(item.get("fpr_prob", 0.0))
            box = item.get("box", [0, 0, 0, 0, 0, 0])
            dy = max(0.0, float(box[3]) - float(box[0])) if len(box) >= 6 else 0.0
            dx = max(0.0, float(box[4]) - float(box[1])) if len(box) >= 6 else 0.0
            dz = max(0.0, float(box[5]) - float(box[2])) if len(box) >= 6 else 0.0
            min_axis = max(min(dy, dx, dz), 1e-3)
            max_axis = max(dy, dx, dz)
            volume = dy * dx * dz
            features = {
                "det_score": det_score,
                "fpr_prob": fpr_prob,
                "interaction": det_score * fpr_prob,
                "log_volume": float(np.log1p(volume)),
                "elongation": float(max_axis / min_axis),
                "max_diameter_mm": float(item.get("max_diameter_mm", max_axis)),
                "size_aware_small": 1.0 if bool(item.get("size_aware_small", False)) else 0.0,
            }
            for key in (
                "morph_axial_elongation",
                "morph_coronal_elongation",
                "morph_sagittal_elongation",
                "morph_axial_fill",
                "morph_coronal_fill",
                "morph_sagittal_fill",
                "morph_max_elongation",
                "morph_mean_elongation",
                "morph_min_fill",
                "morph_mean_fill",
                "morph_bad_plane_count",
                "morph_good_plane_count",
                "morph_valid_plane_count",
            ):
                if key in item:
                    features[key] = float(item.get(key, 0.0))
            rows.append(
                TraceRow(
                    case_name=case_dir.name,
                    proposal_index=int(item.get("proposal_index", -1)),
                    det_score=det_score,
                    fpr_prob=fpr_prob,
                    interaction=det_score * fpr_prob,
                    features=features,
                    candidate_is_tp=bool(item.get("candidate_is_tp", False)),
                    candidate_is_fp=bool(item.get("candidate_is_fp", False)),
                    matched_gt_index=int(item.get("matched_gt_index", -1)),
                    best_iou=float(item.get("best_iou", 0.0)),
                )
            )
    return rows, gt_count_by_case


def _select_must_keep(rows: Sequence[TraceRow], gt_count_by_case: Dict[str, int]) -> Tuple[List[int], Dict[str, List[int]]]:
    best_by_gt: Dict[Tuple[str, int], Tuple[int, float, float]] = {}
    for idx, row in enumerate(rows):
        if row.matched_gt_index < 0 or not row.candidate_is_tp:
            continue
        key = (row.case_name, row.matched_gt_index)
        value = (idx, row.best_iou, row.det_score)
        current = best_by_gt.get(key)
        if current is None or (value[1] > current[1]) or (value[1] == current[1] and value[2] > current[2]):
            best_by_gt[key] = value

    must_keep = sorted(v[0] for v in best_by_gt.values())
    missing_gt_cases: Dict[str, List[int]] = {}
    for case_name, n_gt in gt_count_by_case.items():
        matched = {gt_idx for (cn, gt_idx), _ in best_by_gt.items() if cn == case_name}
        missing = [gt_idx for gt_idx in range(n_gt) if gt_idx not in matched]
        if missing:
            missing_gt_cases[case_name] = missing
    return must_keep, missing_gt_cases


def _evaluate_threshold(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = probs >= threshold
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _search_best_f1(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    best = None
    for threshold in np.arange(0.05, 0.951, 0.01):
        row = _evaluate_threshold(y_true, probs, float(threshold))
        if best is None or row["f1"] > best["f1"]:
            best = row
    return best or _evaluate_threshold(y_true, probs, 0.5)


def _feature_names(feature_set: str) -> List[str]:
    base = ["det_score", "fpr_prob", "interaction"]
    extended = base + ["log_volume", "elongation", "max_diameter_mm", "size_aware_small"]
    morph = extended + [
        "morph_max_elongation",
        "morph_mean_elongation",
        "morph_min_fill",
        "morph_mean_fill",
        "morph_bad_plane_count",
        "morph_good_plane_count",
    ]
    morph_planes = morph + [
        "morph_axial_elongation",
        "morph_coronal_elongation",
        "morph_sagittal_elongation",
        "morph_axial_fill",
        "morph_coronal_fill",
        "morph_sagittal_fill",
    ]
    choices = {
        "base": base,
        "extended": extended,
        "morph": morph,
        "morph_planes": morph_planes,
    }
    if feature_set not in choices:
        raise ValueError(f"Unsupported feature_set: {feature_set}")
    return choices[feature_set]


def _build_feature_matrix(rows: Sequence[TraceRow], feature_names: Sequence[str]) -> np.ndarray:
    missing = sorted({name for name in feature_names for row in rows if name not in row.features})
    if missing:
        raise RuntimeError(
            "Requested features are missing from fpr_trace.json: "
            + ", ".join(missing)
            + ". Re-run test with the updated code and --export_case_analysis."
        )
    return np.asarray([[row.features[name] for name in feature_names] for row in rows], dtype=np.float32)


def _coverage_safe_threshold(probs: np.ndarray, must_keep_indices: Sequence[int]) -> float:
    if not must_keep_indices:
        return 1.0
    return float(np.min(probs[np.asarray(must_keep_indices, dtype=np.int64)]))


def _covered_gt_count(rows: Sequence[TraceRow], probs: np.ndarray, threshold: float) -> int:
    covered = set()
    for i, row in enumerate(rows):
        if row.matched_gt_index < 0:
            continue
        if probs[i] >= threshold:
            covered.add((row.case_name, row.matched_gt_index))
    return len(covered)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit logistic regression on detector+FPR traces.")
    parser.add_argument("--case_analysis_dir", required=True, help="Path to case_analysis directory.")
    parser.add_argument("--output_json", default=None, help="Output report path. Default: <case_analysis_dir>/trace_fuser_regression.json")
    parser.add_argument("--class_weight", default="balanced", help="scikit-learn class_weight. Default: balanced")
    parser.add_argument("--c_value", type=float, default=1.0, help="Inverse regularization strength for logistic regression.")
    parser.add_argument(
        "--feature_set",
        choices=["base", "extended", "morph", "morph_planes"],
        default="morph",
        help="Feature set for the logistic regression. Default: morph",
    )
    args = parser.parse_args()

    case_analysis_dir = Path(args.case_analysis_dir)
    rows, gt_count_by_case = _load_rows(case_analysis_dir)
    if not rows:
        raise RuntimeError(f"No trace rows found under: {case_analysis_dir}")

    must_keep_indices, missing_gt_cases = _select_must_keep(rows, gt_count_by_case)

    feature_names = _feature_names(args.feature_set)
    X = _build_feature_matrix(rows, feature_names)
    y = np.asarray([1 if r.candidate_is_tp else 0 for r in rows], dtype=np.int32)

    model = LogisticRegression(
        penalty="l2",
        C=float(args.c_value),
        solver="lbfgs",
        max_iter=1000,
        class_weight=args.class_weight if args.class_weight != "none" else None,
    )
    model.fit(X, y)
    probs = model.predict_proba(X)[:, 1].astype(np.float32, copy=False)

    best_f1 = _search_best_f1(y, probs)
    safe_tau = _coverage_safe_threshold(probs, must_keep_indices)
    coverage_safe = _evaluate_threshold(y, probs, safe_tau)
    coverage_safe["covered_gt"] = int(_covered_gt_count(rows, probs, safe_tau))
    coverage_safe["n_gt_total"] = int(sum(gt_count_by_case.values()))

    coef = model.coef_[0].astype(np.float32, copy=False)
    intercept = float(model.intercept_[0])
    equation = "logit(p) = " + f"{intercept:.6f}" + "".join(
        f" + {float(c):.6f} * {name}" for name, c in zip(feature_names, coef)
    )

    report = {
        "case_analysis_dir": str(case_analysis_dir),
        "n_cases": int(len({r.case_name for r in rows})),
        "n_proposals": int(len(rows)),
        "n_gt_total": int(sum(gt_count_by_case.values())),
        "n_positive_proposals": int(np.sum(y == 1)),
        "n_negative_proposals": int(np.sum(y == 0)),
        "n_must_keep": int(len(must_keep_indices)),
        "missing_gt_cases": missing_gt_cases,
        "model": {
            "type": "logistic_regression",
            "class_weight": args.class_weight,
            "c_value": float(args.c_value),
            "feature_set": args.feature_set,
            "feature_names": list(feature_names),
            "coefficients": [float(v) for v in coef.tolist()],
            "intercept": intercept,
            "coef_det_score": float(coef[feature_names.index("det_score")]) if "det_score" in feature_names else 0.0,
            "coef_fpr_prob": float(coef[feature_names.index("fpr_prob")]) if "fpr_prob" in feature_names else 0.0,
            "coef_interaction": float(coef[feature_names.index("interaction")]) if "interaction" in feature_names else 0.0,
            "equation": equation,
        },
        "best_f1_threshold": best_f1,
        "coverage_safe_threshold": coverage_safe,
        "notes": [
            f"Features are inference-available only. feature_set={args.feature_set}.",
            "coverage_safe_threshold is set so all must-keep proposals remain above threshold.",
            "If missing_gt_cases is non-empty, detector proposals already miss some GTs before Stage-2.",
        ],
    }

    output_json = Path(args.output_json) if args.output_json else (case_analysis_dir / "trace_fuser_regression.json")
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Trace fuser regression fitted.")
    print(f"Equation: {equation}")
    print(
        "Best F1 threshold: "
        f"{best_f1['threshold']:.4f} | F1={best_f1['f1']:.4f} "
        f"(P={best_f1['precision']:.4f}, R={best_f1['recall']:.4f})"
    )
    print(
        "Coverage-safe threshold: "
        f"{coverage_safe['threshold']:.6f} | covered_gt={coverage_safe['covered_gt']}/{coverage_safe['n_gt_total']} "
        f"| FP={coverage_safe['fp']}"
    )
    if missing_gt_cases:
        print(f"Cases with missing detector GT coverage: {len(missing_gt_cases)}")
    print(f"Saved report: {output_json}")


if __name__ == "__main__":
    main()
