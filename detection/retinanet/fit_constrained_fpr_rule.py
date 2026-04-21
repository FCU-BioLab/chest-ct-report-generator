#!/usr/bin/env python3
"""
Fit a constrained Stage-2 scoring rule from exported case-analysis traces.

Goal:
- keep at least one matched proposal for every GT
- suppress as many FP proposals as possible

The fitted rule is a linear scoring equation:

    score = w_det * det_score + w_fpr * fpr_prob + w_inter * (det_score * fpr_prob)

with non-negative weights constrained to sum to 1.

Threshold tau is chosen as the minimum score among the selected must-keep proposals
(one representative proposal per GT, selected by highest IoU then detector score).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class ProposalRow:
    case_name: str
    proposal_index: int
    det_score: float
    fpr_prob: float
    interaction: float
    matched_gt_index: int
    best_iou: float
    candidate_is_tp: bool
    candidate_is_fp: bool


def _load_case_trace_rows(case_analysis_dir: Path) -> Tuple[List[ProposalRow], Dict[str, int]]:
    rows: List[ProposalRow] = []
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
        case_name = case_dir.name
        for item in trace:
            det_score = float(item.get("det_score", 0.0))
            fpr_prob = float(item.get("fpr_prob", 0.0))
            rows.append(
                ProposalRow(
                    case_name=case_name,
                    proposal_index=int(item.get("proposal_index", -1)),
                    det_score=det_score,
                    fpr_prob=fpr_prob,
                    interaction=det_score * fpr_prob,
                    matched_gt_index=int(item.get("matched_gt_index", -1)),
                    best_iou=float(item.get("best_iou", 0.0)),
                    candidate_is_tp=bool(item.get("candidate_is_tp", False)),
                    candidate_is_fp=bool(item.get("candidate_is_fp", False)),
                )
            )
    return rows, gt_count_by_case


def _select_must_keep_indices(rows: Sequence[ProposalRow], gt_count_by_case: Dict[str, int]) -> Tuple[List[int], Dict[str, List[int]]]:
    case_gt_to_best: Dict[Tuple[str, int], Tuple[int, float, float]] = {}
    for idx, row in enumerate(rows):
        if row.matched_gt_index < 0 or not row.candidate_is_tp:
            continue
        key = (row.case_name, row.matched_gt_index)
        current = case_gt_to_best.get(key)
        candidate_tuple = (idx, row.best_iou, row.det_score)
        if current is None or (candidate_tuple[1] > current[1]) or (
            candidate_tuple[1] == current[1] and candidate_tuple[2] > current[2]
        ):
            case_gt_to_best[key] = candidate_tuple

    must_keep = sorted(v[0] for v in case_gt_to_best.values())
    missing_gt_by_case: Dict[str, List[int]] = {}
    for case_name, n_gt in gt_count_by_case.items():
        matched = {gt_idx for (cn, gt_idx), _ in case_gt_to_best.items() if cn == case_name}
        missing = [gt_idx for gt_idx in range(n_gt) if gt_idx not in matched]
        if missing:
            missing_gt_by_case[case_name] = missing
    return must_keep, missing_gt_by_case


def _evaluate_rule(rows: Sequence[ProposalRow], must_keep_indices: Sequence[int], weights: Tuple[float, float, float]) -> Dict[str, float]:
    w_det, w_fpr, w_inter = weights
    score_vec = np.asarray(
        [w_det * row.det_score + w_fpr * row.fpr_prob + w_inter * row.interaction for row in rows],
        dtype=np.float32,
    )
    tau = float(np.min(score_vec[np.asarray(must_keep_indices, dtype=np.int64)]))
    keep = score_vec >= tau

    total_fp = int(sum(row.candidate_is_fp for row in rows))
    total_tp_candidates = int(sum(row.candidate_is_tp for row in rows))
    kept_fp = int(sum(bool(keep[i]) and rows[i].candidate_is_fp for i in range(len(rows))))
    kept_tp_candidates = int(sum(bool(keep[i]) and rows[i].candidate_is_tp for i in range(len(rows))))

    gt_covered = len(must_keep_indices)
    fp_suppression = 1.0 - (kept_fp / total_fp) if total_fp > 0 else 0.0
    tp_candidate_keep_rate = kept_tp_candidates / total_tp_candidates if total_tp_candidates > 0 else 0.0
    mean_margin = float(np.mean(score_vec[np.asarray(must_keep_indices, dtype=np.int64)] - tau)) if must_keep_indices else 0.0

    return {
        "w_det": float(w_det),
        "w_fpr": float(w_fpr),
        "w_inter": float(w_inter),
        "tau": tau,
        "fp_kept": kept_fp,
        "fp_total": total_fp,
        "fp_suppression": float(fp_suppression),
        "tp_candidates_kept": kept_tp_candidates,
        "tp_candidates_total": total_tp_candidates,
        "tp_candidate_keep_rate": float(tp_candidate_keep_rate),
        "gt_covered": int(gt_covered),
        "mean_positive_margin": float(mean_margin),
    }


def _search_weights(rows: Sequence[ProposalRow], must_keep_indices: Sequence[int], grid_step: float) -> List[Dict[str, float]]:
    candidates: List[Dict[str, float]] = []
    n_steps = int(round(1.0 / grid_step))
    for a in range(n_steps + 1):
        for b in range(n_steps + 1 - a):
            c = n_steps - a - b
            weights = (a * grid_step, b * grid_step, c * grid_step)
            result = _evaluate_rule(rows, must_keep_indices, weights)
            candidates.append(result)
    candidates.sort(
        key=lambda r: (
            -r["fp_suppression"],
            -r["mean_positive_margin"],
            -r["tp_candidate_keep_rate"],
            -r["w_inter"],
            -r["w_fpr"],
        )
    )
    return candidates


def _format_equation(best: Dict[str, float]) -> str:
    return (
        f"score = {best['w_det']:.3f} * det_score + "
        f"{best['w_fpr']:.3f} * fpr_prob + "
        f"{best['w_inter']:.3f} * (det_score * fpr_prob)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a constrained linear FPR scoring rule from case-analysis traces.")
    parser.add_argument(
        "--case_analysis_dir",
        required=True,
        help="Path to case_analysis directory containing case subfolders and fpr_trace.json files.",
    )
    parser.add_argument(
        "--grid_step",
        type=float,
        default=0.05,
        help="Simplex search step size for weight search. Smaller is slower but finer. Default: 0.05",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Number of top candidate rules to save. Default: 20",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        help="Optional path to save the search report JSON. Default: <case_analysis_dir>/constrained_fpr_rule.json",
    )
    args = parser.parse_args()

    case_analysis_dir = Path(args.case_analysis_dir)
    rows, gt_count_by_case = _load_case_trace_rows(case_analysis_dir)
    if not rows:
        raise RuntimeError(f"No fpr_trace.json rows found under: {case_analysis_dir}")

    must_keep_indices, missing_gt_by_case = _select_must_keep_indices(rows, gt_count_by_case)
    total_gt = int(sum(gt_count_by_case.values()))
    if not must_keep_indices:
        raise RuntimeError("No matched GT proposals found in traces. Detector candidates do not cover GTs well enough for constrained fitting.")

    top_rules = _search_weights(rows, must_keep_indices, grid_step=float(args.grid_step))
    best = top_rules[0]

    report = {
        "case_analysis_dir": str(case_analysis_dir),
        "n_cases": int(len({row.case_name for row in rows})),
        "n_proposals": int(len(rows)),
        "n_gt_total": total_gt,
        "n_must_keep": int(len(must_keep_indices)),
        "missing_gt_cases": missing_gt_by_case,
        "best_rule": {
            **best,
            "equation": _format_equation(best),
        },
        "top_rules": [
            {
                **row,
                "equation": _format_equation(row),
            }
            for row in top_rules[: max(1, int(args.top_k))]
        ],
        "notes": [
            "This constrained rule guarantees coverage only for GTs that already have at least one matched candidate in the exported traces.",
            "Threshold tau is set to the minimum score among the selected must-keep proposals, so every represented GT remains covered.",
            "Objective is proposal-level FP suppression, not end-to-end detection F1.",
        ],
    }

    output_json = Path(args.output_json) if args.output_json else (case_analysis_dir / "constrained_fpr_rule.json")
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Constrained rule fitted.")
    print(f"Cases: {report['n_cases']}")
    print(f"Proposals: {report['n_proposals']}")
    print(f"GT total: {report['n_gt_total']}")
    print(f"Must-keep proposals: {report['n_must_keep']}")
    if missing_gt_by_case:
        print(f"Cases with missing GT coverage in detector proposals: {len(missing_gt_by_case)}")
    print("Best equation:")
    print(f"  {_format_equation(best)}")
    print(f"  tau = {best['tau']:.6f}")
    print(f"  FP kept = {best['fp_kept']} / {best['fp_total']}")
    print(f"  FP suppression = {best['fp_suppression']:.4f}")
    print(f"  TP candidate keep rate = {best['tp_candidate_keep_rate']:.4f}")
    print(f"Saved report: {output_json}")


if __name__ == "__main__":
    main()
