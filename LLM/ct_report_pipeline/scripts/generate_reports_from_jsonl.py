"""Generate CT report JSON files from chat-format JSONL validation data.

The output format is compatible with evaluate_clinical_efficacy.py.
Each output JSON contains the original prompt, optional reference report,
parsed nodule features, and the generated report.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if line:
                yield line_number, json.loads(line)


def extract_messages(record: Dict[str, Any]) -> Tuple[str, str]:
    messages = record.get("messages") or []
    user = ""
    assistant = ""
    for message in messages:
        if message.get("role") == "user" and not user:
            user = str(message.get("content") or "")
        elif message.get("role") == "assistant" and not assistant:
            assistant = str(message.get("content") or "")
    return user, assistant


def extract_report_id(prompt: str, fallback: str) -> str:
    match = re.search(r"Report ID:\s*([^\n\r]+)", prompt, flags=re.IGNORECASE)
    if match:
        return sanitize_filename(match.group(1).strip())
    return fallback


def extract_scan_date(prompt: str) -> str:
    match = re.search(r"Date:\s*([^\n\r]+)", prompt, flags=re.IGNORECASE)
    return match.group(1).strip() if match else ""


def parse_nodule_features(prompt: str) -> Dict[str, Any]:
    nodules: List[Dict[str, Any]] = []
    pattern = re.compile(
        r"Nodule\s+(?P<id>\d+):(?P<body>.*?)(?=\n\s*Nodule\s+\d+:|\n\s*LUNG-RADS GUIDE:|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(prompt):
        body = match.group("body")
        size = extract_float(body, r"Size:\s*([0-9]+(?:\.[0-9]+)?)\s*mm")
        volume = extract_float(body, r"Volume:\s*([0-9]+(?:\.[0-9]+)?)")
        nodule_type = extract_text(body, r"Type:\s*([^\n\r]+)")
        nodules.append(
            {
                "nodule_id": int(match.group("id")),
                "longest_axis_mm": size,
                "volume_mm3": volume,
                "attenuation_type": normalize_attenuation(nodule_type),
            }
        )
    max_size = max([n["longest_axis_mm"] for n in nodules if n["longest_axis_mm"] is not None], default=None)
    return {
        "lesions": nodules,
        "total_lesions": len(nodules),
        "max_diameter_mm": max_size,
    }


def build_structured_input(prompt: str, sample_id: str) -> Dict[str, Any]:
    features = parse_nodule_features(prompt)
    lung_rads = determine_lung_rads(features["lesions"])
    return {
        "schema_version": "ct-report-structured-input-v1-rule-lungrads-llm-report",
        "report_id": sample_id,
        "scan_date": extract_scan_date(prompt),
        "task": "Generate a CT chest radiology report from structured nodule data.",
        "decision_policy": {
            "lung_rads_category": "Already determined from image-derived structured features. LLM must use it exactly.",
            "malignancy_risk": "LLM must determine from the provided Lung-RADS category using the few-shot mapping.",
            "recommendation": "LLM must determine from the provided Lung-RADS category using the few-shot mapping.",
            "do_not_fabricate": [
                "extra nodules",
                "location",
                "patient history",
                "unprovided mediastinal or pleural findings",
            ],
        },
        "image_features": {
            "nodules": features["lesions"],
            "total_lesions": features["total_lesions"],
            "max_diameter_mm": features["max_diameter_mm"],
        },
        "lung_rads": lung_rads,
        "output_requirements": {
            "must_include": [
                "all listed nodules",
                "provided Lung-RADS category",
                "malignancy risk corresponding to Lung-RADS",
                "recommendation corresponding to Lung-RADS",
            ]
        },
    }


def build_structured_prompt(structured_input: Dict[str, Any]) -> str:
    structured_json = json.dumps(structured_input, ensure_ascii=False, indent=2)
    return f"""You are a radiologist. Write a CT chest report from the structured JSON payload below.

RULES:
- Use ONLY the structured JSON values.
- Do NOT add nodules, locations, clinical history, mediastinal findings, or pleural findings that are not provided.
- Use the provided lung_rads.exam.category exactly. Do NOT recalculate or change the category.
- Determine malignancy risk and recommendation from the provided Lung-RADS category using the examples below.
- Output in English only.

FEW-SHOT RECOMMENDATION EXAMPLES:
Example 1:
Provided Lung-RADS category: 2
Provided malignancy risk: <1%
Correct recommendation: Continue annual LDCT screening.
Incorrect recommendation: 6-month LDCT follow-up.

Example 2:
Provided Lung-RADS category: 3
Provided malignancy risk: 1-2%
Correct recommendation: 6-month LDCT follow-up.

Example 3:
Provided Lung-RADS category: 4A
Provided malignancy risk: 5-15%
Correct recommendation: 3-month LDCT or PET/CT.

Example 4:
Provided Lung-RADS category: 4B
Provided malignancy risk: >15%
Correct recommendation: PET/CT or tissue sampling.
Incorrect recommendation: 6-month LDCT follow-up.

STRUCTURED_JSON:
{structured_json}

Write the report:

Report ID: {structured_input.get("report_id", "")}
Date: {structured_input.get("scan_date", "")}

Technique:
Non-contrast CT chest.

Findings:

Lungs:
[Describe each nodule: size, type, volume]

Mediastinum: Not evaluated.
Pleura: Not evaluated.

Lung-RADS Assessment:
Category: [Use structured_input.lung_rads.exam.category exactly]
Malignancy Risk: [Determine from the provided Lung-RADS category]

Impression:
[Summarize nodule count, largest size, Lung-RADS category]

Recommendation:
[Determine from the provided Lung-RADS category]"""


def determine_lung_rads(nodules: List[Dict[str, Any]]) -> Dict[str, Any]:
    nodule_assessments = [determine_nodule_lung_rads(nodule) for nodule in nodules]
    if not nodule_assessments:
        category = "1"
        reason = "No measurable pulmonary nodules were provided."
        most_suspicious = None
    else:
        most_suspicious = max(nodule_assessments, key=lambda item: category_rank(item["category"]))
        category = most_suspicious["category"]
        reason = f"Exam category is determined by the most suspicious nodule: {most_suspicious['reason']}"
    return {
        "source": "deterministic_from_structured_image_features",
        "version": "project_size_and_attenuation_rule",
        "exam": {
            "category": category,
            "reason": reason,
            "most_suspicious_nodule_id": most_suspicious.get("nodule_id") if most_suspicious else None,
        },
        "nodules": nodule_assessments,
    }


def determine_nodule_lung_rads(nodule: Dict[str, Any]) -> Dict[str, Any]:
    size = nodule.get("longest_axis_mm")
    attenuation = str(nodule.get("attenuation_type") or "").lower()
    if size is None:
        category = "0"
        reason = "Nodule size is unavailable."
    elif attenuation == "ground-glass":
        if size < 30:
            category = "2"
            reason = "Ground-glass nodule <30 mm."
        else:
            category = "3"
            reason = "Ground-glass nodule >=30 mm."
    elif size < 6:
        category = "2"
        reason = "Largest nodule diameter <6 mm."
    elif size < 8:
        category = "3"
        reason = "Largest nodule diameter 6 to <8 mm."
    elif size < 15:
        category = "4A"
        reason = "Largest nodule diameter 8 to <15 mm."
    else:
        category = "4B"
        reason = "Largest nodule diameter >=15 mm."
    return {
        "nodule_id": nodule.get("nodule_id"),
        "category": category,
        "reason": reason,
        "size_mm": size,
        "attenuation_type": nodule.get("attenuation_type"),
    }


def category_rank(category: str) -> int:
    return {"0": 0, "1": 1, "2": 2, "3": 3, "4A": 4, "4B": 5, "4X": 6}.get(category, 0)


def category_output_mapping(category: str) -> Dict[str, str]:
    return {
        "2": {
            "malignancy_risk": "<1%",
            "recommendation": "Continue annual LDCT screening.",
        },
        "3": {
            "malignancy_risk": "1-2%",
            "recommendation": "6-month LDCT follow-up.",
        },
        "4A": {
            "malignancy_risk": "5-15%",
            "recommendation": "3-month LDCT or PET/CT.",
        },
        "4B": {
            "malignancy_risk": ">15%",
            "recommendation": "PET/CT or tissue sampling.",
        },
    }.get(category, {"malignancy_risk": "Not encoded", "recommendation": "Additional evaluation is needed."})


def validate_and_fix_report(report_text: str, structured_input: Dict[str, Any]) -> Tuple[str, List[Dict[str, str]]]:
    """Enforce final report consistency with the structured Lung-RADS category."""
    category = str(structured_input.get("lung_rads", {}).get("exam", {}).get("category") or "").strip()
    mapping = category_output_mapping(category)
    fixed = report_text
    fixes: List[Dict[str, str]] = []

    fixed, changed = replace_or_insert_line(
        fixed,
        field_name="Category",
        value=category,
        insert_after="Lung-RADS Assessment:",
    )
    if changed:
        fixes.append({"field": "category", "value": category})

    fixed, changed = replace_or_insert_line(
        fixed,
        field_name="Malignancy Risk",
        value=mapping["malignancy_risk"],
        insert_after="Category:",
    )
    if changed:
        fixes.append({"field": "malignancy_risk", "value": mapping["malignancy_risk"]})

    fixed, changed = replace_section(
        fixed,
        heading="Recommendation",
        body=mapping["recommendation"],
    )
    if changed:
        fixes.append({"field": "recommendation", "value": mapping["recommendation"]})

    return fixed, fixes


def replace_or_insert_line(text: str, field_name: str, value: str, insert_after: str) -> Tuple[str, bool]:
    replacement = f"{field_name}: {value}"
    pattern = rf"(?im)^{re.escape(field_name)}:\s*.*$"
    match = re.search(pattern, text)
    if match:
        if match.group(0).strip() == replacement:
            return text, False
        return re.sub(pattern, replacement, text, count=1), True

    anchor_pattern = rf"(?im)^({re.escape(insert_after)}\s*)$"
    anchor = re.search(anchor_pattern, text)
    if anchor:
        insert_at = anchor.end()
        return text[:insert_at] + "\n" + replacement + text[insert_at:], True
    return text.rstrip() + "\n\n" + replacement, True


def replace_section(text: str, heading: str, body: str) -> Tuple[str, bool]:
    section = f"{heading}:\n{body}"
    pattern = rf"(?ims)^{re.escape(heading)}:\s*\n?.*?(?=\n\n[A-Z][A-Za-z -]*:|\Z)"
    match = re.search(pattern, text)
    if match:
        if match.group(0).strip() == section:
            return text, False
        return re.sub(pattern, section, text, count=1), True
    return text.rstrip() + "\n\n" + section, True


def extract_float(text: str, pattern: str) -> Optional[float]:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def extract_text(text: str, pattern: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else ""


def normalize_attenuation(value: str) -> str:
    value = value.strip().lower().replace("_", "-")
    if value in {"ground glass", "ground-glass", "ggo"}:
        return "ground-glass"
    if value in {"part solid", "part-solid", "subsolid"}:
        return "part-solid"
    if value == "solid":
        return "solid"
    return value


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "sample"


def format_prompt(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"


def load_model(base_model: str, adapter: Optional[Path], device: str) -> Tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "auto":
        device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device_map = device
    dtype = torch.float16 if str(device_map).startswith("cuda") and torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
    return tokenizer, model.eval()


def generate_one(tokenizer: Any, model: Any, prompt: str, max_new_tokens: int) -> Tuple[str, Dict[str, Any]]:
    import torch

    formatted = format_prompt(tokenizer, prompt)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    start = time.time()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - start
    generated_ids = output[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return text, {
        "inference_time_sec": elapsed,
        "input_tokens": int(inputs["input_ids"].shape[1]),
        "generated_tokens": int(generated_ids.shape[0]),
        "tokens_per_sec": float(generated_ids.shape[0] / elapsed) if elapsed > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda:0"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prompt-mode", choices=["original", "structured-json"], default="structured-json")
    parser.add_argument("--validate-output", action="store_true", help="Apply rule-based consistency fixes after LLM generation")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.adapter and not args.adapter.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {args.adapter}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, model = load_model(args.base_model, args.adapter, args.device)

    count = 0
    for line_number, record in iter_jsonl(args.input_jsonl):
        if args.limit and count >= args.limit:
            break
        prompt, reference = extract_messages(record)
        if not prompt:
            continue
        sample_id = extract_report_id(prompt, f"sample_{line_number:04d}")
        output_path = args.output_dir / f"{line_number:04d}_{sample_id}.json"
        if output_path.exists() and not args.overwrite:
            count += 1
            continue

        structured_input = build_structured_input(prompt, sample_id)
        generation_prompt = build_structured_prompt(structured_input) if args.prompt_mode == "structured-json" else prompt
        generated, stats = generate_one(tokenizer, model, generation_prompt, args.max_new_tokens)
        final_report = generated
        validation_fixes: List[Dict[str, str]] = []
        if args.validate_output:
            final_report, validation_fixes = validate_and_fix_report(generated, structured_input)
        output = {
            "patient_id": sample_id,
            "source_jsonl": str(args.input_jsonl),
            "line_number": line_number,
            "prompt": generation_prompt,
            "source_prompt": prompt,
            "prompt_mode": args.prompt_mode,
            "reference_report": reference,
            "structured_input": structured_input,
            "input_features": {
                **structured_input["image_features"],
                "lung_rads_category": structured_input["lung_rads"]["exam"]["category"],
            },
            "generated_report_raw": generated,
            "generated_report": final_report,
            "validation_fixes": validation_fixes,
            "stats": stats,
        }
        output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        count += 1
        print(f"[{count}] wrote {output_path}")

    print(json.dumps({"generated_or_existing": count, "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
