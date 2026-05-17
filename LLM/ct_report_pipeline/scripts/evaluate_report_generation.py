"""
Evaluate CT report generation models.

The script can either:
1. Generate reports from a base model plus optional LoRA adapter, then evaluate them.
2. Evaluate an existing JSONL prediction file with reference/generated fields.

Outputs:
- generated_reports.jsonl
- evaluation_metrics.json
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from extras.evaluation.metrics import ClinicalEfficacyMetrics, NLGMetrics


SYSTEM_PROMPT = """You are an experienced radiologist assistant. Generate professional CT chest reports based on provided nodule measurements.

Rules:
1. Use ONLY the provided measurements - do not fabricate data
2. Follow the standard radiology report structure
3. Output in English only
4. Include Lung-RADS 2022 category assessment
5. Leave uncertain fields empty or state "Not evaluated"
6. Be concise and clinically relevant"""


def load_report_pairs(jsonl_path: Path) -> List[Dict[str, str]]:
    """Load prompt/reference pairs from OpenAI-style or prompt/response JSONL."""
    pairs = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "messages" in item:
                messages = item["messages"]
                prompt = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
                reference = next((m.get("content", "") for m in messages if m.get("role") == "assistant"), "")
            else:
                prompt = item.get("prompt", "")
                reference = item.get("response", item.get("reference", ""))

            if not prompt or not reference:
                raise ValueError(f"Missing prompt/reference at {jsonl_path}:{line_no}")

            pairs.append({"prompt": prompt, "reference": reference})
    return pairs


def load_predictions(jsonl_path: Path) -> List[Dict[str, str]]:
    """Load existing predictions for metric-only evaluation."""
    rows = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            reference = item.get("reference", "")
            generated = item.get("generated", item.get("prediction", item.get("generated_report", "")))
            prompt = item.get("prompt", "")
            if not reference or not generated:
                raise ValueError(f"Missing reference/generated at {jsonl_path}:{line_no}")
            rows.append({"prompt": prompt, "reference": reference, "generated": generated})
    return rows


def format_prompt(tokenizer, prompt: str) -> str:
    """Format prompt with the same system instruction used during fine-tuning."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"### System:\n{SYSTEM_PROMPT}\n\n### User:\n{prompt}\n\n### Assistant:\n"


def generate_reports(
    rows: List[Dict[str, str]],
    model_name: str,
    lora_path: Optional[str],
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
) -> List[Dict[str, str]]:
    """Generate reports using a Hugging Face causal LM and optional PEFT adapter."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()

    outputs = []
    for idx, row in enumerate(rows, start=1):
        formatted = format_prompt(tokenizer, row["prompt"])
        inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = tokenizer.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()
        outputs.append(
            {
                "index": idx,
                "prompt": row["prompt"],
                "reference": row["reference"],
                "generated": generated,
            }
        )
    return outputs


def compute_format_compliance(reports: List[str]) -> Dict[str, float]:
    """Measure whether generated reports contain required report sections."""
    checks = {
        "report_id": re.compile(r"\bReport ID\s*:", re.IGNORECASE),
        "technique": re.compile(r"\bTechnique\s*:", re.IGNORECASE),
        "findings": re.compile(r"\b(Findings|Lungs)\s*:", re.IGNORECASE),
        "lung_rads": re.compile(r"\b(Lung-RADS|Category)\b", re.IGNORECASE),
        "recommendation": re.compile(r"\bRecommendation\s*:", re.IGNORECASE),
    }
    if not reports:
        return {key: 0.0 for key in [*checks.keys(), "macro_avg"]}

    scores = {}
    for name, pattern in checks.items():
        scores[name] = sum(bool(pattern.search(report)) for report in reports) / len(reports)
    scores["macro_avg"] = sum(scores.values()) / len(checks)
    return scores


def compute_metrics(rows: List[Dict[str, str]]) -> Dict[str, object]:
    """Compute NLG, clinical efficacy, and format metrics."""
    references = [row["reference"] for row in rows]
    generated = [row["generated"] for row in rows]

    metrics = {
        "sample_count": len(rows),
        "nlg": NLGMetrics().compute_all(references, generated),
        "clinical_efficacy": ClinicalEfficacyMetrics().compute_metrics(references, generated),
        "format_compliance": compute_format_compliance(generated),
    }
    return metrics


def write_jsonl(path: Path, rows: List[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CT report generation quality.")
    parser.add_argument("--model_name", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--val_data_path", default="assets/data/finetune_val.jsonl")
    parser.add_argument("--predictions_path", default=None)
    parser.add_argument("--output_dir", default="assets/evaluation")
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all samples")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--do_sample", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / datetime.now().strftime("eval_%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.predictions_path:
        rows = load_predictions(Path(args.predictions_path))
    else:
        rows = load_report_pairs(Path(args.val_data_path))
        if args.max_samples and args.max_samples > 0:
            rows = rows[: args.max_samples]
        rows = generate_reports(
            rows,
            model_name=args.model_name,
            lora_path=args.lora_path,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample,
        )

    predictions_file = output_dir / "generated_reports.jsonl"
    metrics_file = output_dir / "evaluation_metrics.json"

    write_jsonl(predictions_file, rows)
    metrics = compute_metrics(rows)
    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Predictions saved to: {predictions_file}")
    print(f"Metrics saved to: {metrics_file}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
