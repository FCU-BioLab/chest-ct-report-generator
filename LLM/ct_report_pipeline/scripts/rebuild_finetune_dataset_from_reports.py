"""
Rebuild CT report fine-tuning datasets from radiology reports.

This script derives the prompt side from the source report text, then formats it
with the same structured JSON prompt used by the runtime LLM pipeline.

Default source:
E:/radiology_reports_dataset/report_data/splited_reports/*.txt
"""

import argparse
import json
import math
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from lung_rads import build_structured_report_input
from prompt_templates import build_report_prompt


LOCATION_PATTERN = re.compile(
    r"\b(RUL|RML|RLL|LUL|LLL|right upper lobe|right middle lobe|right lower lobe|"
    r"left upper lobe|left lower lobe|lingula)\b",
    re.IGNORECASE,
)


def read_text(path: Path) -> str:
    """Read text with tolerant encodings used by the source report files."""
    for encoding in ("utf-8-sig", "utf-8", "cp950", "big5", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def normalize_text(text: str) -> str:
    """Normalize whitespace and common superscript-3 mojibake."""
    text = text.replace("\ufeff", "")
    text = text.replace("mm糧", "mm^3")
    text = text.replace("mm³", "mm^3")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_report(text: str, report_id: str) -> str:
    """Remove noisy headers/footers while preserving source text for traceability."""
    text = normalize_text(text)
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        if stripped.startswith("==="):
            break
        if report_id in stripped and len(stripped) <= len(report_id) + 20:
            continue
        if "嚙" in stripped:
            continue
        cleaned_lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned_lines)).strip()


def estimate_volume_mm3(size_mm: float) -> float:
    radius = size_mm / 2.0
    return round((4.0 / 3.0) * math.pi * (radius ** 3), 1)


def normalize_location(location: Optional[str]) -> Optional[str]:
    if not location:
        return None
    value = location.strip().lower()
    mapping = {
        "right upper lobe": "RUL",
        "right middle lobe": "RML",
        "right lower lobe": "RLL",
        "left upper lobe": "LUL",
        "left lower lobe": "LLL",
        "lingula": "lingula",
    }
    return mapping.get(value, location.upper())


def infer_attenuation_type(context: str) -> str:
    text = context.lower()
    if re.search(r"ground[- ]glass|\bggo\b|\bggn\b|non[- ]solid", text):
        return "ground-glass"
    if re.search(r"part[- ]solid|subsolid", text):
        return "part-solid"
    if "calcif" in text:
        return "calcified"
    if re.search(r"nodule|nodular|opacity|lesion|mass|tumou?r", text):
        return "solid"
    return "indeterminate"


def extract_lesion_features(report_text: str) -> List[Dict]:
    """
    Extract nodule-like measurements from the report.

    The source reports are semi-structured free text, so this intentionally keeps
    derived fields transparent and marks uncertain attenuation as indeterminate.
    """
    text = normalize_text(report_text)
    features = []
    seen = set()

    size_pattern = re.compile(r"(?:size\s*)?(\d+(?:\.\d+)?)\s*mm\b", re.IGNORECASE)
    for match in size_pattern.finditer(text):
        size_mm = float(match.group(1))
        if size_mm < 2 or size_mm > 100:
            continue

        start = max(0, match.start() - 180)
        end = min(len(text), match.end() + 180)
        context = text[start:end]
        context_lower = context.lower()

        nodule_terms = [
            "nodule",
            "nodular",
            "opacity",
            "lesion",
            "mass",
            "tumor",
            "tumour",
            "ggo",
            "ground glass",
            "lung",
            "pulmonary",
            "lobe",
            "rul",
            "rml",
            "rll",
            "lul",
            "lll",
            "lingula",
        ]
        skip_terms = ["aorta", "heart", "vertebr", "spine", "lymph node", "lymph nodes"]
        if not any(term in context_lower for term in nodule_terms):
            continue
        if any(term in context_lower for term in skip_terms):
            continue

        location_match = LOCATION_PATTERN.search(context)
        location = normalize_location(location_match.group(0)) if location_match else None
        attenuation_type = infer_attenuation_type(context)
        key = (round(size_mm, 1), location, attenuation_type)
        if key in seen:
            continue
        seen.add(key)

        features.append(
            {
                "nodule_id": len(features) + 1,
                "longest_axis_mm": round(size_mm, 1),
                "equivalent_diameter_mm": round(size_mm, 1),
                "mean_diameter_mm": round(size_mm, 1),
                "volume_mm3": estimate_volume_mm3(size_mm),
                "attenuation_type": attenuation_type,
                "attenuation_confidence": 0.75 if attenuation_type != "indeterminate" else 0.0,
                "location": location or "Not evaluated",
                "mean_hu": None,
                "source": "derived_from_report_text",
            }
        )

    return features


def build_pipeline_response(report_id: str, scan_date: str, structured_input: Dict) -> str:
    """Build a normalized assistant target matching the pipeline report format."""
    nodules = structured_input["nodules"]
    lung_rads = structured_input["lung_rads"]
    exam = lung_rads["exam"]

    if nodules:
        lung_lines = []
        for nodule in nodules:
            location = nodule.get("location") or "Not evaluated"
            attenuation = nodule.get("attenuation_type") or "indeterminate"
            size = nodule.get("longest_axis_mm")
            volume = nodule.get("volume_mm3")
            lung_lines.append(
                f"{nodule.get('nodule_id')}. {location} {attenuation} pulmonary nodule, "
                f"size {size:.1f} mm, estimated volume {volume:.1f} mm^3."
            )
        largest = max(nodules, key=lambda item: item.get("longest_axis_mm") or 0)
        impression = (
            f"{len(nodules)} pulmonary nodule(s), largest {largest['longest_axis_mm']:.1f} mm "
            f"in {largest.get('location') or 'Not evaluated'}; Lung-RADS Category {exam['category']}."
        )
    else:
        lung_lines = ["No measurable pulmonary nodules were derived from the source report."]
        impression = f"No measurable pulmonary nodule; Lung-RADS Category {exam['category']}."

    return f"""Report ID: {report_id}
Date: {scan_date}

Technique:
Non-contrast CT chest.

Findings:

Lungs:
{chr(10).join(lung_lines)}

Mediastinum: Not evaluated.
Pleura: Not evaluated.

Lung-RADS Assessment:
Category: {exam['category']}
Descriptor: {exam['descriptor']}
Management: {exam['management']}

Impression:
1. {impression}

Recommendation:
{exam['management']}"""


def build_examples(root: Path, report_dir_name: str, scan_date: str, response_mode: str) -> List[Dict]:
    report_dir = root / report_dir_name
    if not report_dir.exists():
        raise FileNotFoundError(f"Report directory not found: {report_dir}")

    examples = []
    skipped = []
    for report_file in sorted(report_dir.glob("*.txt")):
        report_id = report_file.stem
        source_report = read_text(report_file)
        cleaned_report = clean_report(source_report, report_id)
        if not cleaned_report:
            skipped.append({"report_id": report_id, "reason": "empty_report"})
            continue

        lesion_features = extract_lesion_features(cleaned_report)
        structured_input = build_structured_report_input(
            lesion_features,
            report_id=report_id,
            scan_date=scan_date,
        )
        prompt = build_report_prompt(
            lesion_features,
            report_id=report_id,
            scan_date=scan_date,
            structured_input=structured_input,
        )
        if response_mode == "original_report":
            response = cleaned_report
        else:
            response = build_pipeline_response(report_id, scan_date, structured_input)

        examples.append(
            {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
                "report_id": report_id,
                "source_report": str(report_file),
                "nodule_count": len(lesion_features),
                "has_nodule": bool(lesion_features),
                "derived_lesion_features": lesion_features,
                "structured_input": structured_input,
                "source_clean_report": cleaned_report,
            }
        )

    return examples, skipped


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_examples(examples: List[Dict], train_ratio: float, val_ratio: float, seed: int):
    rows = examples[:]
    random.Random(seed).shuffle(rows)
    train_end = int(len(rows) * train_ratio)
    val_end = train_end + int(len(rows) * val_ratio)
    return rows[:train_end], rows[train_end:val_end], rows[val_end:]


def summarize(rows: List[Dict]) -> Dict:
    nodule_counts = [row["nodule_count"] for row in rows]
    return {
        "count": len(rows),
        "with_nodule": sum(1 for row in rows if row["has_nodule"]),
        "without_nodule": sum(1 for row in rows if not row["has_nodule"]),
        "min_nodule_count": min(nodule_counts) if nodule_counts else 0,
        "max_nodule_count": max(nodule_counts) if nodule_counts else 0,
        "avg_nodule_count": sum(nodule_counts) / len(nodule_counts) if nodule_counts else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild fine-tune JSONL from source report text.")
    parser.add_argument("--source_root", default=r"E:\radiology_reports_dataset\report_data")
    parser.add_argument("--report_dir", default="splited_reports")
    parser.add_argument("--output_dir", default="assets/data")
    parser.add_argument("--prefix", default="finetune_reportdata_pipeline")
    parser.add_argument("--scan_date", default="Not evaluated")
    parser.add_argument(
        "--response_mode",
        choices=["pipeline_template", "original_report"],
        default="pipeline_template",
        help="Use normalized pipeline output or cleaned original report as assistant target.",
    )
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_root = Path(args.source_root)
    output_dir = Path(args.output_dir)

    examples, skipped = build_examples(source_root, args.report_dir, args.scan_date, args.response_mode)
    train, val, test = split_examples(examples, args.train_ratio, args.val_ratio, args.seed)

    full_path = output_dir / f"{args.prefix}_full.jsonl"
    train_path = output_dir / f"{args.prefix}_train.jsonl"
    val_path = output_dir / f"{args.prefix}_val.jsonl"
    test_path = output_dir / f"{args.prefix}_test.jsonl"
    manifest_path = output_dir / f"{args.prefix}_manifest.json"

    write_jsonl(full_path, examples)
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)
    write_jsonl(test_path, test)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_root": str(source_root),
        "report_dir": args.report_dir,
        "prompt_source": "derived_from_report_text",
        "prompt_format": "pipeline_structured_json_prompt",
        "response_mode": args.response_mode,
        "output_files": {
            "full": str(full_path),
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
        },
        "split": {
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": 1.0 - args.train_ratio - args.val_ratio,
        },
        "summary": {
            "full": summarize(examples),
            "train": summarize(train),
            "val": summarize(val),
            "test": summarize(test),
            "skipped": skipped,
            "skipped_count": len(skipped),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
