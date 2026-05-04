#!/usr/bin/env python3
"""
Train a lightweight candidate-level fuser from mask-feature CSV rows.

The input is produced by analyze_candidate_mask_features.py. This script is
intended for analysis before integrating any rule/model into inference.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURES = [
    "score",
    "bbox_max_diameter_mm",
    "bbox_volume_mm3",
    "mask_volume_mm3",
    "mask_bbox_fill",
    "elongation_3d",
    "bbox_fill_3d",
    "axial_elongation",
    "axial_solidity",
    "axial_fill",
    "sagittal_elongation",
    "sagittal_solidity",
    "sagittal_fill",
    "coronal_elongation",
    "coronal_solidity",
    "coronal_fill",
    "best_side_area_ratio",
    "side_elongation_delta",
    "side_solidity_delta",
    "side_fill_delta",
]


def _read_rows(csv_path: Path) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _safe_float(value: str) -> float:
    try:
        if value is None or value == "":
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _build_arrays(rows: Sequence[Dict[str, str]], feature_names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray([[ _safe_float(row.get(name)) for name in feature_names ] for row in rows], dtype=np.float32)
    y = np.asarray([1 if str(row.get("label", "")).upper() == "TP" else 0 for row in rows], dtype=np.int64)
    groups = np.asarray([str(row.get("case_name", "")) for row in rows], dtype=object)
    return x, y, groups


def _metrics_at_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    protected_mask: np.ndarray = None,
) -> Dict:
    pred = probs >= float(threshold)
    if protected_mask is not None:
        pred = pred | protected_mask.astype(bool, copy=False)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        pred.astype(np.int64),
        average="binary",
        zero_division=0,
    )
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _threshold_search(
    y_true: np.ndarray,
    probs: np.ndarray,
    target_recall: float,
    protected_mask: np.ndarray = None,
) -> Dict:
    thresholds = np.arange(0.01, 0.991, 0.01, dtype=np.float32)
    rows = [_metrics_at_threshold(y_true, probs, float(t), protected_mask=protected_mask) for t in thresholds]
    best_f1 = max(rows, key=lambda r: (r["f1"], r["recall"], r["precision"]))
    recall_safe = [r for r in rows if r["recall"] >= float(target_recall)]
    if recall_safe:
        best_recall_safe = max(recall_safe, key=lambda r: (r["precision"], r["f1"], r["threshold"]))
    else:
        best_recall_safe = max(rows, key=lambda r: (r["recall"], r["precision"], r["f1"]))
    return {
        "best_f1": best_f1,
        "recall_safe": best_recall_safe,
        "thresholds": rows,
    }


def _build_protected_mask(rows: Sequence[Dict[str, str]], protect_score: float, protect_small_mm: float) -> np.ndarray:
    protected = []
    for row in rows:
        score = _safe_float(row.get("score"))
        diameter = _safe_float(row.get("bbox_max_diameter_mm"))
        keep = False
        if np.isfinite(score) and score >= float(protect_score):
            keep = True
        if protect_small_mm > 0 and np.isfinite(diameter) and diameter <= float(protect_small_mm):
            keep = True
        protected.append(keep)
    return np.asarray(protected, dtype=bool)


def _coef_report(model: Pipeline, feature_names: Sequence[str]) -> Dict[str, float]:
    clf = model.named_steps["clf"]
    coefs = clf.coef_[0]
    return {name: float(coef) for name, coef in zip(feature_names, coefs)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight fuser from candidate mask features.")
    parser.add_argument("--features_csv", required=True, help="candidate_mask_features.csv")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--val_ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_recall", type=float, default=0.95)
    parser.add_argument("--protect_score", type=float, default=0.95, help="always keep candidates with detector score >= this value")
    parser.add_argument("--protect_small_mm", type=float, default=8.0, help="always keep candidates with max bbox diameter <= this value; 0 disables")
    parser.add_argument("--features", nargs="*", default=DEFAULT_FEATURES)
    args = parser.parse_args()

    features_csv = Path(args.features_csv)
    output_json = Path(args.output_json) if args.output_json else features_csv.with_name("candidate_mask_fuser_report.json")
    rows = _read_rows(features_csv)
    if not rows:
        raise RuntimeError(f"No rows found: {features_csv}")

    x, y, groups = _build_arrays(rows, args.features)
    if len(np.unique(y)) < 2:
        raise RuntimeError("Need both TP and FP rows to train a fuser.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=float(args.val_ratio), random_state=int(args.seed))
    train_idx, val_idx = next(splitter.split(x, y, groups=groups))

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, solver="lbfgs")),
        ]
    )
    model.fit(x[train_idx], y[train_idx])

    train_probs = model.predict_proba(x[train_idx])[:, 1]
    val_probs = model.predict_proba(x[val_idx])[:, 1]
    protected_all = _build_protected_mask(rows, args.protect_score, args.protect_small_mm)
    train_search = _threshold_search(y[train_idx], train_probs, target_recall=args.target_recall)
    val_search = _threshold_search(y[val_idx], val_probs, target_recall=args.target_recall)
    train_protected_search = _threshold_search(
        y[train_idx],
        train_probs,
        target_recall=args.target_recall,
        protected_mask=protected_all[train_idx],
    )
    val_protected_search = _threshold_search(
        y[val_idx],
        val_probs,
        target_recall=args.target_recall,
        protected_mask=protected_all[val_idx],
    )

    report = {
        "features_csv": str(features_csv),
        "feature_names": list(args.features),
        "seed": int(args.seed),
        "val_ratio": float(args.val_ratio),
        "target_recall": float(args.target_recall),
        "protected_policy": {
            "enabled": True,
            "protect_score": float(args.protect_score),
            "protect_small_mm": float(args.protect_small_mm),
            "description": "keep candidate if detector score is high or candidate is small; otherwise require fuser probability above threshold",
        },
        "n_rows": int(len(rows)),
        "n_cases": int(len(set(groups.tolist()))),
        "train": {
            "n_rows": int(len(train_idx)),
            "n_tp": int(np.sum(y[train_idx] == 1)),
            "n_fp": int(np.sum(y[train_idx] == 0)),
            "roc_auc": float(roc_auc_score(y[train_idx], train_probs)) if len(np.unique(y[train_idx])) > 1 else None,
            "pr_ap": float(average_precision_score(y[train_idx], train_probs)) if len(np.unique(y[train_idx])) > 1 else None,
            **train_search,
            "protected": train_protected_search,
        },
        "val": {
            "n_rows": int(len(val_idx)),
            "n_tp": int(np.sum(y[val_idx] == 1)),
            "n_fp": int(np.sum(y[val_idx] == 0)),
            "roc_auc": float(roc_auc_score(y[val_idx], val_probs)) if len(np.unique(y[val_idx])) > 1 else None,
            "pr_ap": float(average_precision_score(y[val_idx], val_probs)) if len(np.unique(y[val_idx])) > 1 else None,
            **val_search,
            "protected": val_protected_search,
        },
        "model": {
            "type": "logistic_regression",
            "intercept": float(model.named_steps["clf"].intercept_[0]),
            "coefficients_standardized": _coef_report(model, args.features),
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps({
        "output_json": str(output_json),
        "train_best_f1": report["train"]["best_f1"],
        "val_best_f1": report["val"]["best_f1"],
        "val_recall_safe": report["val"]["recall_safe"],
        "val_protected_best_f1": report["val"]["protected"]["best_f1"],
        "val_protected_recall_safe": report["val"]["protected"]["recall_safe"],
        "val_roc_auc": report["val"]["roc_auc"],
        "val_pr_ap": report["val"]["pr_ap"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
