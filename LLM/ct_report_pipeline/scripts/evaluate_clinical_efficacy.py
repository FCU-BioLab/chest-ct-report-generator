"""Clinical efficacy evaluation for generated CT chest reports.

This script evaluates whether generated reports preserve clinically important
labels derived from structured nodule features. It is intentionally lightweight:
no model inference is performed, and no external packages are required.

Supported inputs:
- End-to-end case directory, e.g. F:\\chest-ct-report-output\\case-006
- Generated report JSON files from Llama 3.2 report-generation output directories

The evaluation is label-based rather than text-overlap-based. It checks labels
such as nodule presence, count, largest-size bucket, attenuation type,
Lung-RADS category, and recommendation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from lung_rads import CATEGORY_INFO, assess_exam  # noqa: E402


SIZE_LABELS = ("size_lt_6mm", "size_6_8mm", "size_8_15mm", "size_ge_15mm")
TYPE_LABELS = ("solid", "part_solid", "ground_glass")
LUNG_RADS_LABELS = ("lung_rads_2", "lung_rads_3", "lung_rads_4a", "lung_rads_4b")
RECOMMENDATION_LABELS = (
    "annual_screening",
    "six_month_followup",
    "three_month_followup",
    "pet_ct_or_biopsy",
)


@dataclass
class EvalSample:
    sample_id: str
    expected: Dict[str, bool]
    predicted: Dict[str, bool]
    expected_summary: Dict[str, Any]
    predicted_summary: Dict[str, Any]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def size_bucket(size_mm: Optional[float]) -> Optional[str]:
    if size_mm is None:
        return None
    if size_mm < 6:
        return "size_lt_6mm"
    if size_mm < 8:
        return "size_6_8mm"
    if size_mm < 15:
        return "size_8_15mm"
    return "size_ge_15mm"


def category_from_size(size_mm: Optional[float]) -> Optional[str]:
    """Fallback category for legacy generated JSON without per-nodule features."""

    if size_mm is None:
        return None
    if size_mm < 6:
        return "lung_rads_2"
    if size_mm < 8:
        return "lung_rads_3"
    if size_mm < 15:
        return "lung_rads_4a"
    return "lung_rads_4b"


def category_to_label(category: Optional[str]) -> Optional[str]:
    if not category:
        return None
    normalized = str(category).strip().lower()
    if normalized in {"2", "3", "4a", "4b"}:
        return f"lung_rads_{normalized}"
    return None


def recommendation_from_category(category: Optional[str]) -> Optional[str]:
    return {
        "lung_rads_2": "annual_screening",
        "lung_rads_3": "six_month_followup",
        "lung_rads_4a": "three_month_followup",
        "lung_rads_4b": "pet_ct_or_biopsy",
    }.get(category or "")


def label_dict() -> Dict[str, bool]:
    labels = {
        "nodule_present": False,
        "count_correct": False,
    }
    labels.update({label: False for label in SIZE_LABELS})
    labels.update({label: False for label in TYPE_LABELS})
    labels.update({label: False for label in LUNG_RADS_LABELS})
    labels.update({label: False for label in RECOMMENDATION_LABELS})
    return labels


def labels_from_structured_features(sample_id: str, features: Any) -> Tuple[Dict[str, bool], Dict[str, Any]]:
    labels = label_dict()

    nodules: List[Dict[str, Any]]
    if isinstance(features, list):
        nodules = [x for x in features if isinstance(x, dict)]
    elif isinstance(features, dict) and "lesions" in features:
        nodules = [x for x in features.get("lesions", []) if isinstance(x, dict)]
    elif isinstance(features, dict) and "total_lesions" in features:
        count = int(features.get("total_lesions") or 0)
        max_size = as_float(features.get("max_diameter_mm"))
        summary = {
            "sample_id": sample_id,
            "nodule_count": count,
            "max_size_mm": max_size,
            "attenuation_types": [],
            "lung_rads_category": category_from_size(max_size),
            "lung_rads_reason": "Legacy generated JSON contains only total_lesions/max_diameter_mm; per-nodule attenuation and solid-component features are unavailable.",
            "lung_rads_source": "fallback_size_only_for_legacy_input_features",
            "lung_rads_limitations": ["missing_per_nodule_features"],
        }
        labels["nodule_present"] = count > 0
        bucket = size_bucket(max_size)
        if bucket:
            labels[bucket] = True
        category = summary["lung_rads_category"]
        if category:
            labels[category] = True
            rec = recommendation_from_category(category)
            if rec:
                labels[rec] = True
        return labels, summary
    else:
        nodules = []

    count = len(nodules)
    sizes = [
        as_float(n.get("longest_axis_mm"))
        or as_float(n.get("equivalent_diameter_mm"))
        or as_float(n.get("diameter_mm"))
        for n in nodules
    ]
    sizes = [x for x in sizes if x is not None]
    max_size = max(sizes) if sizes else None

    raw_types = [
        str(n.get("attenuation_type") or n.get("type") or "").lower()
        for n in nodules
    ]
    type_labels = {attenuation_to_label(t) for t in raw_types}
    type_labels.discard(None)

    labels["nodule_present"] = count > 0
    bucket = size_bucket(max_size)
    if bucket:
        labels[bucket] = True
    for t in type_labels:
        labels[t] = True

    assessment = assess_exam(nodules)
    exam = assessment.get("exam", {})
    category = category_to_label(exam.get("category"))
    if category:
        labels[category] = True
        rec = recommendation_from_category(category)
        if rec:
            labels[rec] = True

    return labels, {
        "sample_id": sample_id,
        "nodule_count": count,
        "max_size_mm": max_size,
        "attenuation_types": sorted(type_labels),
        "lung_rads_category": category,
        "lung_rads_reason": exam.get("reason"),
        "lung_rads_management": exam.get("management"),
        "lung_rads_limitations": exam.get("limitations", []) + assessment.get("limitations", []),
        "lung_rads_source": "llm/ct_report_pipeline/lung_rads.py::assess_exam; deterministic Lung-RADS v2022 helper",
    }


def labels_from_report_text(text: str) -> Tuple[Dict[str, bool], Dict[str, Any]]:
    labels = label_dict()
    norm = normalize_text(text)

    nodule_mentions = len(re.findall(r"\bnodule(?:s)?\b", norm))
    labels["nodule_present"] = bool(re.search(r"\b(nodule|nodules|lesion|mass)\b", norm))

    # Match linear measurements in mm, but exclude volume units such as mm3/mm³.
    sizes = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*mm(?!\s*(?:3|³|\^3))", norm)]
    max_size = max(sizes) if sizes else None
    bucket = size_bucket(max_size)
    if bucket:
        labels[bucket] = True

    if re.search(r"\bpart[-\s]?solid\b|\bsubsolid\b", norm):
        labels["part_solid"] = True
    if re.search(r"\bground[-\s]?glass\b|\bggo\b", norm):
        labels["ground_glass"] = True
    # Count "solid" only when not part-solid/subsolid wording.
    if re.search(r"\bsolid\b", norm) and not labels["part_solid"]:
        labels["solid"] = True

    category = extract_lung_rads_category(norm)
    if category:
        labels[category] = True

    rec = extract_recommendation(norm)
    if rec:
        labels[rec] = True

    count = extract_reported_count(norm)

    return labels, {
        "nodule_mentions": nodule_mentions,
        "reported_nodule_count": count,
        "max_size_mm": max_size,
        "lung_rads_category": category,
        "recommendation": rec,
    }


def compare_count(expected_summary: Dict[str, Any], predicted_summary: Dict[str, Any], predicted: Dict[str, bool]) -> None:
    expected_count = expected_summary.get("nodule_count")
    predicted_count = predicted_summary.get("reported_nodule_count")
    if expected_count is None or predicted_count is None:
        predicted["count_correct"] = False
    else:
        predicted["count_correct"] = int(expected_count) == int(predicted_count)


def extract_reported_count(norm: str) -> Optional[int]:
    # Prefer impression phrases like "3 pulmonary nodule(s)".
    patterns = [
        r"\b(\d+)\s+pulmonary\s+nodule",
        r"\b(\d+)\s+nodule",
        r"\b(\d+)\s+lung\s+nodule",
    ]
    for pattern in patterns:
        match = re.search(pattern, norm)
        if match:
            return int(match.group(1))

    enumerated = re.findall(r"(?:^|\s)(\d+)\.\s+(?:a\s+)?(?:\d+(?:\.\d+)?\s*mm\s+)?(?:solid|part[-\s]?solid|ground[-\s]?glass)?\s*pulmonary\s+nodule", norm)
    if enumerated:
        return len(set(enumerated))
    if "nodule" in norm:
        return 1
    return 0


def extract_lung_rads_category(norm: str) -> Optional[str]:
    match = re.search(r"(?:lung[-\s]?rads.*?category|category)\s*[:\-]?\s*(4a|4b|2|3)\b", norm)
    if not match:
        match = re.search(r"lung[-\s]?rads\s*(4a|4b|2|3)\b", norm)
    if not match:
        return None
    return f"lung_rads_{match.group(1).lower()}"


def extract_recommendation(norm: str) -> Optional[str]:
    if re.search(r"annual|12[-\s]?month|yearly", norm):
        return "annual_screening"
    if re.search(r"6[-\s]?month|six[-\s]?month", norm):
        return "six_month_followup"
    if re.search(r"3[-\s]?month|three[-\s]?month", norm):
        return "three_month_followup"
    if re.search(r"pet[/\-\s]?ct|biopsy|tissue sampling", norm):
        return "pet_ct_or_biopsy"
    return None


def attenuation_to_label(value: str) -> Optional[str]:
    value = normalize_text(value)
    if "part" in value and "solid" in value:
        return "part_solid"
    if "ground" in value or "glass" in value or value == "ggo":
        return "ground_glass"
    if "solid" in value:
        return "solid"
    return None


def as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_metrics(samples: Iterable[EvalSample]) -> Dict[str, Any]:
    sample_list = list(samples)
    all_labels = list(label_dict().keys())
    per_label: Dict[str, Dict[str, Any]] = {}

    for label in all_labels:
        tp = fp = fn = tn = 0
        for sample in sample_list:
            ref = bool(sample.expected.get(label, False))
            hyp = bool(sample.predicted.get(label, False))
            if ref and hyp:
                tp += 1
            elif not ref and hyp:
                fp += 1
            elif ref and not hyp:
                fn += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        accuracy = (tp + tn) / len(sample_list) if sample_list else 0.0
        support = tp + fn
        per_label[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        }

    supported = [v for v in per_label.values() if v["support"] > 0]
    macro = {
        "precision": mean([v["precision"] for v in supported]),
        "recall": mean([v["recall"] for v in supported]),
        "f1": mean([v["f1"] for v in supported]),
        "accuracy": mean([v["accuracy"] for v in supported]),
    }

    exact_match = 0
    for sample in sample_list:
        labels_with_support = [k for k in all_labels if sample.expected.get(k, False)]
        if labels_with_support and all(sample.predicted.get(k, False) for k in labels_with_support):
            exact_match += 1

    return {
        "num_samples": len(sample_list),
        "macro_supported": macro,
        "exact_supported_label_match_rate": exact_match / len(sample_list) if sample_list else 0.0,
        "per_label": per_label,
        "samples": [
            {
                "sample_id": sample.sample_id,
                "expected_summary": sample.expected_summary,
                "predicted_summary": sample.predicted_summary,
                "expected_labels": sample.expected,
                "predicted_labels": sample.predicted,
            }
            for sample in sample_list
        ],
    }


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_case_dir(case_dir: Path) -> EvalSample:
    features_path = case_dir / "04_feature" / "lesion_features.json"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing feature file: {features_path}")

    report_files = sorted((case_dir / "05_report").glob("AUTO_*.json"))
    if not report_files:
        raise FileNotFoundError(f"Missing AUTO report JSON under: {case_dir / '05_report'}")

    features = json.loads(features_path.read_text(encoding="utf-8"))
    report_json = json.loads(report_files[0].read_text(encoding="utf-8"))
    report_text = str(report_json.get("text") or report_json.get("generated_report") or "")

    expected, expected_summary = labels_from_structured_features(case_dir.name, features)
    predicted, predicted_summary = labels_from_report_text(report_text)
    compare_count(expected_summary, predicted_summary, predicted)
    expected["count_correct"] = True

    return EvalSample(case_dir.name, expected, predicted, expected_summary, predicted_summary)


def load_generated_report(path: Path) -> EvalSample:
    data = json.loads(path.read_text(encoding="utf-8"))
    sample_id = str(data.get("patient_id") or path.stem)
    structured_input = data.get("structured_input") if isinstance(data.get("structured_input"), dict) else {}
    structured_features = structured_input.get("image_features") if isinstance(structured_input.get("image_features"), dict) else {}
    features = structured_features.get("nodules") or data.get("input_features") or {}
    report_text = str(data.get("generated_report") or data.get("text") or "")

    reference_report = str(data.get("reference_report") or "")
    if reference_report:
        expected, reference_summary = labels_from_report_text(reference_report)
        expected_summary = {
            "sample_id": sample_id,
            "nodule_count": reference_summary.get("reported_nodule_count"),
            "max_size_mm": reference_summary.get("max_size_mm"),
            "lung_rads_category": reference_summary.get("lung_rads_category"),
            "recommendation": reference_summary.get("recommendation"),
            "reference_source": "reference_report",
        }
    else:
        expected, expected_summary = labels_from_structured_features(sample_id, features)
    predicted, predicted_summary = labels_from_report_text(report_text)
    compare_count(expected_summary, predicted_summary, predicted)
    expected["count_correct"] = True

    return EvalSample(sample_id, expected, predicted, expected_summary, predicted_summary)


def write_markdown(results: Dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Clinical Efficacy Evaluation",
        "",
        f"- Samples: {results['num_samples']}",
        f"- Macro precision: {results['macro_supported']['precision']:.4f}",
        f"- Macro recall: {results['macro_supported']['recall']:.4f}",
        f"- Macro F1: {results['macro_supported']['f1']:.4f}",
        f"- Supported-label exact match rate: {results['exact_supported_label_match_rate']:.4f}",
        "",
        "## Per-label Metrics",
        "",
        "| Label | Support | TP | FP | FN | Precision | Recall | F1 | Accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, metric in results["per_label"].items():
        lines.append(
            f"| {label} | {metric['support']} | {metric['tp']} | {metric['fp']} | "
            f"{metric['fn']} | {metric['precision']:.4f} | {metric['recall']:.4f} | "
            f"{metric['f1']:.4f} | {metric['accuracy']:.4f} |"
        )
    lines.extend(["", "## Samples", ""])
    for sample in results["samples"]:
        lines.append(f"### {sample['sample_id']}")
        lines.append("")
        lines.append(f"- Expected: `{json.dumps(sample['expected_summary'], ensure_ascii=False)}`")
        lines.append(f"- Predicted: `{json.dumps(sample['predicted_summary'], ensure_ascii=False)}`")
        missed = [
            label
            for label, value in sample["expected_labels"].items()
            if value and not sample["predicted_labels"].get(label, False)
        ]
        false_pos = [
            label
            for label, value in sample["predicted_labels"].items()
            if value and not sample["expected_labels"].get(label, False)
        ]
        lines.append(f"- Missed labels: {', '.join(missed) if missed else 'none'}")
        lines.append(f"- Extra labels: {', '.join(false_pos) if false_pos else 'none'}")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", action="append", type=Path, default=[], help="End-to-end case directory")
    parser.add_argument("--generated-json", action="append", type=Path, default=[], help="Generated report JSON file")
    parser.add_argument("--generated-json-dir", type=Path, help="Directory containing generated report JSON files")
    parser.add_argument("--output-json", type=Path, required=True, help="Path for JSON metrics")
    parser.add_argument("--output-md", type=Path, help="Optional Markdown summary path")
    args = parser.parse_args()

    samples: List[EvalSample] = []
    for case_dir in args.case_dir:
        samples.append(load_case_dir(case_dir))
    for path in args.generated_json:
        samples.append(load_generated_report(path))
    if args.generated_json_dir:
        for path in sorted(args.generated_json_dir.glob("*.json")):
            samples.append(load_generated_report(path))

    if not samples:
        raise SystemExit("No samples were provided. Use --case-dir, --generated-json, or --generated-json-dir.")

    results = compute_metrics(samples)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(results, args.output_md)

    print(json.dumps({
        "num_samples": results["num_samples"],
        "macro_supported": results["macro_supported"],
        "exact_supported_label_match_rate": results["exact_supported_label_match_rate"],
        "output_json": str(args.output_json),
        "output_md": str(args.output_md) if args.output_md else "",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
