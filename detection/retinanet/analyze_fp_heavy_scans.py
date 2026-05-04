#!/usr/bin/env python3
"""
Analyze how metrics change after excluding FP-heavy scans.

This is a diagnostic tool only. Excluding cases from evaluation should not be
reported as model performance; it helps identify scans that dominate FP burden.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


FROC_FP_RATES = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def _load_summaries(case_analysis_dir: Path) -> List[Dict]:
    rows = []
    for path in sorted(case_analysis_dir.rglob("summary.json")):
        with open(path, "r", encoding="utf-8") as f:
            item = json.load(f)
        item["_summary_path"] = str(path)
        rows.append(item)
    return rows


def _case_name(summary: Dict) -> str:
    return str(summary.get("case_name") or Path(summary.get("_summary_path", "")).parent.name)


def _collect_predictions(summaries: Sequence[Dict]) -> tuple[np.ndarray, np.ndarray]:
    scores = []
    labels = []
    for summary in summaries:
        pred_scores = summary.get("pred_scores") or []
        pred_is_tp = summary.get("pred_is_tp")
        if pred_is_tp is None:
            match = summary.get("pred_match_gt") or []
            pred_is_tp = [int(v) >= 0 for v in match]
        for i, score in enumerate(pred_scores):
            scores.append(float(score))
            labels.append(1 if i < len(pred_is_tp) and bool(pred_is_tp[i]) else 0)
    return np.asarray(scores, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def _froc(summaries: Sequence[Dict]) -> Dict:
    n_scans = max(len(summaries), 1)
    n_gt = int(sum(int(s.get("n_gt", 0)) for s in summaries))
    scores, labels = _collect_predictions(summaries)
    if n_gt <= 0 or len(scores) == 0:
        return {
            "froc_score": 0.0,
            **{f"sensitivity_at_{rate:g}_fp_per_scan": 0.0 for rate in FROC_FP_RATES},
        }

    order = np.argsort(-scores)
    labels = labels[order]
    tp_cum = np.cumsum(labels == 1)
    fp_cum = np.cumsum(labels == 0)
    fp_per_scan = fp_cum.astype(np.float32) / float(n_scans)
    sens = tp_cum.astype(np.float32) / float(n_gt)

    result = {}
    for rate in FROC_FP_RATES:
        valid = np.where(fp_per_scan <= float(rate))[0]
        value = float(np.max(sens[valid])) if len(valid) else 0.0
        result[f"sensitivity_at_{rate:g}_fp_per_scan"] = value
    result["froc_score"] = float(np.mean([result[f"sensitivity_at_{rate:g}_fp_per_scan"] for rate in FROC_FP_RATES]))
    return result


def _aggregate(summaries: Sequence[Dict]) -> Dict:
    n_scans = int(len(summaries))
    n_gt = int(sum(int(s.get("n_gt", 0)) for s in summaries))
    n_pred = int(sum(int(s.get("n_pred", 0)) for s in summaries))
    n_tp = int(sum(int(s.get("n_tp", 0)) for s in summaries))
    n_fp = int(sum(int(s.get("n_fp", 0)) for s in summaries))
    n_fn = int(sum(int(s.get("n_fn", 0)) for s in summaries))
    precision = n_tp / max(n_tp + n_fp, 1)
    recall = n_tp / max(n_tp + n_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    scan_tp = int(sum(1 for s in summaries if int(s.get("n_gt", 0)) > 0 and int(s.get("n_pred", 0)) > 0))
    scan_fp = int(sum(1 for s in summaries if int(s.get("n_gt", 0)) == 0 and int(s.get("n_pred", 0)) > 0))
    scan_tn = int(sum(1 for s in summaries if int(s.get("n_gt", 0)) == 0 and int(s.get("n_pred", 0)) == 0))
    scan_fn = int(sum(1 for s in summaries if int(s.get("n_gt", 0)) > 0 and int(s.get("n_pred", 0)) == 0))
    return {
        "n_scans": n_scans,
        "n_gt": n_gt,
        "n_pred": n_pred,
        "n_tp": n_tp,
        "n_fp": n_fp,
        "n_fn": n_fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fp_per_scan": float(n_fp / max(n_scans, 1)),
        "scan_tp": scan_tp,
        "scan_fp": scan_fp,
        "scan_tn": scan_tn,
        "scan_fn": scan_fn,
        **_froc(summaries),
    }


def _case_rows(summaries: Sequence[Dict]) -> List[Dict]:
    rows = []
    for summary in summaries:
        n_gt = int(summary.get("n_gt", 0))
        n_pred = int(summary.get("n_pred", 0))
        n_tp = int(summary.get("n_tp", 0))
        n_fp = int(summary.get("n_fp", 0))
        n_fn = int(summary.get("n_fn", 0))
        rows.append(
            {
                "case_name": _case_name(summary),
                "n_gt": n_gt,
                "n_pred": n_pred,
                "n_tp": n_tp,
                "n_fp": n_fp,
                "n_fn": n_fn,
                "case_precision": float(summary.get("case_precision", n_tp / max(n_pred, 1))),
                "case_recall": float(summary.get("case_recall", n_tp / max(n_gt, 1))),
                "case_f1": float(summary.get("case_f1", 0.0)),
                "summary_path": str(summary.get("_summary_path", "")),
            }
        )
    return sorted(rows, key=lambda r: (-r["n_fp"], r["n_fn"], -r["n_tp"], r["case_name"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze metric changes after excluding FP-heavy scans.")
    parser.add_argument("--case_analysis_dir", required=True)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--exclude_top_fp", type=int, nargs="*", default=[1, 2, 3, 5])
    args = parser.parse_args()

    case_analysis_dir = Path(args.case_analysis_dir)
    output_json = Path(args.output_json) if args.output_json else case_analysis_dir / "fp_heavy_scan_report.json"
    output_csv = Path(args.output_csv) if args.output_csv else case_analysis_dir / "fp_heavy_scan_rows.csv"

    summaries = _load_summaries(case_analysis_dir)
    if not summaries:
        raise RuntimeError(f"No summary.json files found under {case_analysis_dir}")

    rows = _case_rows(summaries)
    name_to_summary = {_case_name(s): s for s in summaries}
    scenarios = {"baseline": _aggregate(summaries)}
    excluded = {}
    for n in args.exclude_top_fp:
        n = int(n)
        excluded_names = [row["case_name"] for row in rows[:n]]
        keep = [s for s in summaries if _case_name(s) not in set(excluded_names)]
        scenarios[f"exclude_top_{n}_fp_scans"] = _aggregate(keep)
        excluded[f"exclude_top_{n}_fp_scans"] = [row for row in rows[:n]]

    report = {
        "case_analysis_dir": str(case_analysis_dir),
        "note": "Diagnostic only: excluding scans from evaluation is not valid model performance.",
        "top_fp_scans": rows,
        "scenarios": scenarios,
        "excluded_scans": excluded,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({"output_json": str(output_json), "output_csv": str(output_csv), "scenarios": scenarios, "excluded_scans": excluded}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
