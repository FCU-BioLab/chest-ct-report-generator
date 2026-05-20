"""
Report Generator Module

Generates structured CT radiology reports using Llama LLM.
Supports both pre-trained and fine-tuned models with Lung-RADS 2022 classification.
"""

import os
import sys
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
from datetime import datetime

try:
    import torch
except ImportError:
    torch = None

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

# Set HuggingFace token from environment
hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from prompt_templates import (
    SYSTEM_PROMPT_BILINGUAL,
    build_report_prompt,
    format_nodule_descriptions,
    classify_nodule_type_from_features,
)
from lung_rads import assess_exam, build_structured_report_input


def category_output_mapping(category: str) -> Dict[str, str]:
    """Map deterministic Lung-RADS category to report-facing risk and recommendation."""
    return {
        "1": {
            "malignancy_risk": "<1%",
            "recommendation": "Continue annual LDCT screening.",
        },
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
        "4X": {
            "malignancy_risk": ">15%",
            "recommendation": "Diagnostic evaluation as appropriate for a very suspicious Lung-RADS finding.",
        },
    }.get(category, {"malignancy_risk": "Not encoded", "recommendation": "Additional evaluation is needed."})


def build_llm_structured_input(structured_input: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the LLM-facing JSON payload without prefilled risk/recommendation fields."""
    source_nodules = [dict(item) for item in structured_input.get("nodules", [])]
    lung_rads = deepcopy(dict(structured_input.get("lung_rads", {})))
    exam = dict(lung_rads.get("exam", {}))
    assessed_by_id = {
        item.get("nodule_id"): item
        for item in lung_rads.get("nodules", [])
        if isinstance(item, Mapping)
    }

    compact_nodules = []
    for index, nodule in enumerate(source_nodules, 1):
        nodule_id = nodule.get("nodule_id", index)
        assessed = assessed_by_id.get(nodule_id, {})
        compact = {
            "nodule_id": nodule_id,
            "longest_axis_mm": _round_or_none(nodule.get("longest_axis_mm")),
            "equivalent_diameter_mm": _round_or_none(nodule.get("equivalent_diameter_mm")),
            "volume_mm3": _round_or_none(nodule.get("volume_mm3"), digits=1),
            "attenuation_type": nodule.get("attenuation_type"),
            "attenuation_confidence": _round_or_none(nodule.get("attenuation_confidence"), digits=2),
            "mean_hu": _round_or_none(nodule.get("mean_hu"), digits=1),
            "lung_rads_category": assessed.get("category"),
            "lung_rads_reason": assessed.get("reason"),
        }
        if "solid_component_mm" in assessed:
            compact["solid_component_mm"] = _round_or_none(assessed.get("solid_component_mm"))
        if isinstance(nodule.get("anatomical_location"), Mapping):
            compact["anatomical_location"] = {
                "lobe": nodule["anatomical_location"].get("lobe_full") or nodule["anatomical_location"].get("lobe"),
                "side": nodule["anatomical_location"].get("side"),
            }
        compact_nodules.append({k: v for k, v in compact.items() if v not in (None, "", [])})

    payload = {
        "schema_version": "ct-report-structured-input-v1-rule-lungrads-llm-report",
        "report_id": structured_input.get("report_id"),
        "scan_date": structured_input.get("scan_date"),
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
                "training IDs",
            ],
        },
        "image_features": {
            "nodules": compact_nodules,
            "total_lesions": len(compact_nodules),
            "max_diameter_mm": max([_to_float(item.get("longest_axis_mm")) or 0.0 for item in compact_nodules] or [0.0]),
        },
        "lung_rads": {
            "source": "deterministic_from_structured_image_features",
            "exam": {
                "category": exam.get("category"),
                "descriptor": exam.get("descriptor"),
                "reason": exam.get("reason"),
                "limitations": exam.get("limitations", []),
                "most_suspicious_nodule_id": exam.get("most_suspicious_nodule_id"),
            },
            "nodules": [
                {
                    "nodule_id": item.get("nodule_id"),
                    "category": item.get("category"),
                    "reason": item.get("reason"),
                    "limitations": item.get("limitations", []),
                }
                for item in lung_rads.get("nodules", [])
            ],
            "limitations": lung_rads.get("limitations", []),
        },
        "output_requirements": {
            "must_include": [
                "all listed nodules",
                "provided Lung-RADS category",
                "malignancy risk corresponding to Lung-RADS",
                "recommendation corresponding to Lung-RADS",
            ]
        },
    }
    _remove_report_answer_fields(payload)
    return payload


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_or_none(value: Any, digits: int = 2) -> Optional[float]:
    number = _to_float(value)
    return round(number, digits) if number is not None else None


def _remove_report_answer_fields(value: Any) -> None:
    """Remove fields that would give the LLM the final risk/recommendation answer."""
    if isinstance(value, dict):
        for key in list(value.keys()):
            normalized = str(key).lower()
            if normalized in {"management", "recommendation", "malignancy_risk", "risk", "probability"}:
                value.pop(key, None)
            else:
                _remove_report_answer_fields(value[key])
    elif isinstance(value, list):
        for item in value:
            _remove_report_answer_fields(item)


def validate_and_fix_report(report_text: str, structured_input: Mapping[str, Any]) -> Tuple[str, List[Dict[str, str]]]:
    """Enforce final report consistency with the structured Lung-RADS category."""
    category = str(structured_input.get("lung_rads", {}).get("exam", {}).get("category") or "").strip()
    mapping = category_output_mapping(category)
    fixed = report_text
    fixes: List[Dict[str, str]] = []

    fixed, changed = _replace_or_insert_line(
        fixed,
        field_name="Category",
        value=category,
        insert_after="Lung-RADS Assessment:",
    )
    if changed:
        fixes.append({"field": "category", "value": category})

    fixed, changed = _replace_or_insert_line(
        fixed,
        field_name="Malignancy Risk",
        value=mapping["malignancy_risk"],
        insert_after="Category:",
    )
    if changed:
        fixes.append({"field": "malignancy_risk", "value": mapping["malignancy_risk"]})

    fixed, changed = _replace_section(
        fixed,
        heading="Recommendation",
        body=mapping["recommendation"],
    )
    if changed:
        fixes.append({"field": "recommendation", "value": mapping["recommendation"]})

    return fixed, fixes


def _replace_or_insert_line(text: str, field_name: str, value: str, insert_after: str) -> Tuple[str, bool]:
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


def _replace_section(text: str, heading: str, body: str) -> Tuple[str, bool]:
    section = f"{heading}:\n{body}"
    pattern = rf"(?ims)^{re.escape(heading)}:\s*\n?.*?(?=\n\n[A-Z][A-Za-z -]*:|\Z)"
    match = re.search(pattern, text)
    if match:
        if match.group(0).strip() == section:
            return text, False
        return re.sub(pattern, section, text, count=1), True
    return text.rstrip() + "\n\n" + section, True


def _extract_report_like_text(text: str) -> str:
    for marker in ("Report ID:", "Technique:", "Findings:", "Lung-RADS Assessment:"):
        index = text.find(marker)
        if index >= 0:
            return text[index:].strip()
    match = re.search(r"(?im)^Report:\s*$", text)
    if match:
        return text[match.end() :].strip()
    return text


def _remove_contaminated_tail(text: str) -> str:
    stop_patterns = (
        r"(?im)^Lungs-RADS Training Data:",
        r"(?im)^Lung-RADS Training Data:",
        r"(?im)^Write Report:",
        r"(?im)^Training Data IDs:",
        r"<END REPORT>",
        r"<end_of_train_set>",
    )
    end = len(text)
    for pattern in stop_patterns:
        match = re.search(pattern, text)
        if match:
            end = min(end, match.start())
    return text[:end].strip()


def _is_contaminated_report(text: str) -> bool:
    lowered = text.lower()
    forbidden = (
        "train_",
        "test_",
        "<end_of_train_set",
        "<end report",
        "select <option",
        "<select",
        '"bbox_',
        '"source_feature_keys"',
        '"lungs_data"',
        "training data",
        "write report:",
    )
    if any(token in lowered for token in forbidden):
        return True
    required = ("Technique:", "Findings:", "Lung-RADS Assessment:", "Impression:", "Recommendation:")
    return sum(1 for marker in required if marker.lower() in lowered) < 4


def _render_guardrailed_report(structured_input: Mapping[str, Any], report_id: str, scan_date: str) -> str:
    """Render a clean final report when LLM text is contaminated by prompt/training artifacts."""
    nodules = list(structured_input.get("nodules", []))
    lung_rads = structured_input.get("lung_rads", {})
    exam = lung_rads.get("exam", {}) if isinstance(lung_rads, Mapping) else {}
    exam_category = str(exam.get("category") or "0")
    mapping = category_output_mapping(exam_category)
    assessed_by_id = {
        item.get("nodule_id"): item
        for item in lung_rads.get("nodules", [])
        if isinstance(item, Mapping)
    }

    lines = [
        f"Report ID: {report_id}",
        f"Date: {scan_date}",
        "",
        "Technique:",
        "Non-contrast CT chest.",
        "",
        "Findings:",
        "",
        "Lungs:",
    ]
    for index, nodule in enumerate(nodules, 1):
        nodule_id = nodule.get("nodule_id", index)
        assessed = assessed_by_id.get(nodule_id, {})
        attenuation = nodule.get("attenuation_type") or "indeterminate"
        size = _to_float(nodule.get("longest_axis_mm") or nodule.get("equivalent_diameter_mm"))
        volume = _to_float(nodule.get("volume_mm3"))
        location = ""
        if isinstance(nodule.get("anatomical_location"), Mapping):
            location = nodule["anatomical_location"].get("lobe_full") or nodule["anatomical_location"].get("lobe") or ""
        location_text = f" in the {location}" if location else ""
        size_text = f"{size:.1f} mm" if size is not None else "size not available"
        volume_text = f", volume {volume:.1f} mm3" if volume is not None else ""
        category_text = assessed.get("category") or "not assessed"
        lines.append(
            f"{index}. {attenuation} pulmonary nodule{location_text} measuring {size_text}{volume_text} "
            f"(nodule Lung-RADS {category_text})."
        )

    if not nodules:
        lines.append("No measurable pulmonary nodules were provided.")

    limitations = list(dict.fromkeys((lung_rads.get("limitations", []) if isinstance(lung_rads, Mapping) else []) + exam.get("limitations", [])))
    largest = max([_to_float(item.get("longest_axis_mm") or item.get("equivalent_diameter_mm")) or 0.0 for item in nodules] or [0.0])
    lines.extend(
        [
            "",
            "Mediastinum: Not evaluated.",
            "Pleura: Not evaluated.",
            "",
            "Lung-RADS Assessment:",
            f"Category: {exam_category}",
            f"Malignancy Risk: {mapping['malignancy_risk']}",
            "",
            "Impression:",
            f"{len(nodules)} pulmonary nodule(s), largest {largest:.1f} mm; exam-level Lung-RADS Category {exam_category}.",
            "",
            "Recommendation:",
            mapping["recommendation"],
        ]
    )
    if limitations:
        lines.extend(["", "Limitations:"])
        lines.extend(f"- {item}" for item in limitations)
    return "\n".join(lines)


class ReportGenerator:
    """
    Generates structured CT reports using Llama LLM.
    
    Features:
    - Local Llama model support with LoRA fine-tuning
    - Professional English radiology report format
    - Lung-RADS 2022 classification
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        device: str = None,
        use_lora: bool = False,
        lora_path: str = None,
        load_in_8bit: bool = False,
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
        **kwargs,  # Accept extra kwargs for compatibility
    ):
        """Initialize the report generator."""
        self.model_name = model_name
        self.device = device or ("cuda" if torch is not None and torch.cuda.is_available() else "cpu")
        self.use_lora = use_lora
        self.lora_path = lora_path
        self.load_in_8bit = load_in_8bit
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
    
    def load_model(self):
        """Load the Llama model and tokenizer."""
        if self.is_loaded:
            return
        if torch is None:
            raise ImportError("PyTorch is required for LLM report generation but is not installed.")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            
            print(f"Loading model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            if self.load_in_8bit and self.device == "cuda":
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True,
                )
                if self.device == "cpu":
                    self.model = self.model.to(self.device)
            
            # Load LoRA weights if specified
            if self.use_lora and self.lora_path:
                self._load_lora_weights()
            
            self.model.eval()
            self.is_loaded = True
            print("??Model loaded successfully")
            
        except Exception as e:
            print(f"??Failed to load model: {e}")
            raise
    
    def _load_lora_weights(self):
        """Load LoRA adapter weights."""
        try:
            from peft import PeftModel
            print(f"  Loading LoRA weights from: {self.lora_path}")
            self.model = PeftModel.from_pretrained(self.model, self.lora_path)
            print("  ??LoRA weights loaded")
        except Exception as e:
            print(f"  ??Failed to load LoRA weights: {e}")
    
    def generate_report(
        self,
        lesion_features: Union[Dict, List[Dict]],
        report_id: str = None,
        scan_date: str = None,
        return_xml: bool = False,
        structured_input: Optional[Dict] = None,
        lung_rads_assessment: Optional[Dict] = None,
        validate_output: bool = True,
    ) -> Dict[str, str]:
        """Generate a structured CT report from lesion features."""
        if not self.is_loaded:
            self.load_model()
        
        # Ensure lesion_features is a list
        if isinstance(lesion_features, dict):
            lesion_features = [lesion_features]
        
        # Generate report ID and date if not provided
        if not report_id:
            report_id = f"AUTO_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        if not scan_date:
            scan_date = datetime.now().strftime("%Y/%m/%d")

        if structured_input is None:
            structured_input = build_structured_report_input(
                lesion_features,
                report_id=report_id,
                scan_date=scan_date,
            )
        if lung_rads_assessment is None:
            lung_rads_assessment = structured_input.get("lung_rads")

        llm_structured_input = build_llm_structured_input(structured_input)
        
        # Build prompt
        prompt = build_report_prompt(
            lesion_features,
            report_id=report_id,
            scan_date=scan_date,
            structured_input=llm_structured_input,
        )
        
        if os.environ.get("CHEST_CT_LOG_LLM_PROMPT", "").lower() in {"1", "true", "yes"}:
            print("\n" + "=" * 60)
            print("[LLM PROMPT]")
            print("=" * 60)
            print(prompt)
            print("=" * 60 + "\n")
        
        # Generate response
        generated_text = self._generate(prompt)
        
        # Post-process
        raw_report_text, postprocess_fixes = self._postprocess(
            generated_text,
            report_id,
            scan_date,
            structured_input=structured_input,
        )
        report_text = raw_report_text
        validation_fixes: List[Dict[str, str]] = []
        if validate_output:
            report_text, validation_fixes = validate_and_fix_report(raw_report_text, structured_input)
        
        return {
            "text": report_text,
            "raw_text": raw_report_text,
            "xml": None,
            "parsed": {
                "structured_input": structured_input,
                "llm_structured_input": llm_structured_input,
                "lung_rads": lung_rads_assessment,
                "postprocess_fixes": postprocess_fixes,
                "validation_fixes": validation_fixes,
            },
            "report_id": report_id,
            "scan_date": scan_date,
        }
    
    def _generate(self, prompt: str) -> str:
        """Generate text using the loaded model."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_BILINGUAL},
            {"role": "user", "content": prompt},
        ]
        
        # Format prompt
        try:
            if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                formatted_prompt = f"### System:\n{SYSTEM_PROMPT_BILINGUAL}\n\n### User:\n{prompt}\n\n### Assistant:\n"
        except Exception:
            formatted_prompt = f"### System:\n{SYSTEM_PROMPT_BILINGUAL}\n\n### User:\n{prompt}\n\n### Assistant:\n"
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            generate_kwargs = {
                **inputs,
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.temperature > 0,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "repetition_penalty": 1.2,
            }
            if self.temperature > 0:
                generate_kwargs["temperature"] = self.temperature
            outputs = self.model.generate(
                **generate_kwargs,
            )
        
        # Decode
        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        
        return generated.strip()
    
    def _postprocess(
        self,
        text: str,
        report_id: str,
        scan_date: str,
        structured_input: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[str, List[Dict[str, str]]]:
        """Clean up the generated report."""
        text = text.strip()
        fixes: List[Dict[str, str]] = []
        
        # Remove markers
        if text.startswith("---"):
            text = text[3:].strip()
        if text.endswith("---"):
            text = text[:-3].strip()

        text = _extract_report_like_text(text)
        text = _remove_contaminated_tail(text)

        if structured_input is not None and _is_contaminated_report(text):
            text = _render_guardrailed_report(structured_input, report_id, scan_date)
            fixes.append({"field": "report_body", "value": "guardrailed_from_structured_input"})
        
        # Ensure header exists
        if not text.startswith("Report ID:"):
            text = f"Report ID: {report_id}\nDate: {scan_date}\n\n{text}"
        
        return text, fixes
    
    def save_report(self, report: Dict, output_dir: str, formats: List[str] = ["txt", "json"]) -> Dict[str, str]:
        """Save report in multiple formats."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_id = report.get("report_id", "report")
        saved_files = {}
        
        if "txt" in formats:
            txt_path = output_dir / f"{report_id}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(report.get("text", ""))
            saved_files["txt"] = str(txt_path)
        
        if "json" in formats:
            json_path = output_dir / f"{report_id}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "report_id": report_id,
                    "scan_date": report.get("scan_date", ""),
                    "text": report.get("text", ""),
                    "raw_text": report.get("raw_text", ""),
                    "parsed": report.get("parsed"),
                }, f, ensure_ascii=False, indent=2)
            saved_files["json"] = str(json_path)
        
        return saved_files


class SimpleReportGenerator:
    """
    Template-based report generator (no LLM required).
    Uses Lung-RADS 2022 classification.
    """
    
    @staticmethod
    def get_nodule_type(features_or_mean_hu: Union[Dict, float]) -> str:
        """Determine nodule type from HU statistics."""
        if isinstance(features_or_mean_hu, dict):
            return classify_nodule_type_from_features(features_or_mean_hu)
        return classify_nodule_type_from_features({"mean_hu": float(features_or_mean_hu)})
    
    @staticmethod
    def get_lung_rads_category(size_mm: float, nodule_type: str) -> Dict:
        """Calculate Lung-RADS 2022 category."""
        assessment = assess_exam(
            [{"nodule_id": 1, "attenuation_type": nodule_type, "longest_axis_mm": size_mm}]
        )["exam"]
        return {
            "category": assessment["category"],
            "description": assessment["descriptor"],
            "probability": "Not encoded by Lung-RADS v2022 rule table",
            "recommendation": assessment["management"],
        }
    
    def generate_report(
        self,
        lesion_features: Union[Dict, List[Dict]],
        report_id: str = None,
        scan_date: str = None,
        return_xml: bool = False,
        structured_input: Optional[Dict] = None,
        lung_rads_assessment: Optional[Dict] = None,
        validate_output: bool = True,
    ) -> Dict[str, str]:
        """Generate a template-based report."""
        if isinstance(lesion_features, dict):
            lesion_features = [lesion_features]
        
        if not report_id:
            report_id = f"AUTO_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        if not scan_date:
            scan_date = datetime.now().strftime("%Y/%m/%d")

        if structured_input is None:
            structured_input = build_structured_report_input(
                lesion_features,
                report_id=report_id,
                scan_date=scan_date,
            )
        if lung_rads_assessment is None:
            lung_rads_assessment = structured_input.get("lung_rads") or assess_exam(lesion_features)
        exam_assessment = lung_rads_assessment.get("exam", {})
        nodule_assessments = {
            item.get("nodule_id"): item for item in lung_rads_assessment.get("nodules", [])
        }
        
        # Build report sections
        lines = [
            f"Report ID: {report_id}",
            f"Date: {scan_date}",
            "",
            "Technique:",
            "Non-contrast computed tomography of the chest was performed using standard protocols.",
            "",
            "Findings:",
            "",
            "Lungs:",
        ]
        
        # Process each nodule
        for i, features in enumerate(lesion_features, 1):
            size_mm = features.get("equivalent_diameter_mm", 0)
            volume_mm3 = features.get("volume_mm3", 0)
            nodule_type = self.get_nodule_type(features)
            nodule_id = features.get("nodule_id", i)
            nodule_lung_rads = nodule_assessments.get(nodule_id, {})
            location = ""
            if isinstance(features.get("anatomical_location"), dict):
                location = features["anatomical_location"].get("lobe_full") or features["anatomical_location"].get("lobe") or ""
            
            location_text = f" in the {location}" if location else ""
            category_text = nodule_lung_rads.get("category", "not assessed")
            lines.append(
                f"{i}. A {nodule_type} pulmonary nodule{location_text} measuring {size_mm:.1f} mm (ESD) "
                f"with volume of {volume_mm3:.1f} mm3 is identified "
                f"(nodule Lung-RADS {category_text})."
            )
        
        lines.extend([
            "",
            "Mediastinum:",
            "The mediastinal structures appear intact. No masses or lymphadenopathy noted.",
            "",
            "Pleura:",
            "No pleural effusion or pneumothorax.",
            "",
            "Bony Structures:",
            "The visualized osseous structures are unremarkable.",
            "",
            "Lung-RADS 2022 Assessment:",
            f"Category: {exam_assessment.get('category', '0')}",
            f"Description: {exam_assessment.get('descriptor', '')}",
            "",
            "Impression:",
            f"1. {len(lesion_features)} pulmonary nodule(s) identified - Lung-RADS Category {exam_assessment.get('category', '0')}.",
            "",
            "Recommendation:",
            exam_assessment.get("management", ""),
        ])

        limitations = lung_rads_assessment.get("limitations", []) + exam_assessment.get("limitations", [])
        if limitations:
            lines.extend(["", "Limitations:"])
            for item in dict.fromkeys(limitations):
                lines.append(f"- {item}")
        
        report_text = "\n".join(lines)
        
        return {
            "text": report_text,
            "xml": None,
            "parsed": {
                "structured_input": structured_input,
                "lung_rads": lung_rads_assessment,
            },
            "report_id": report_id,
            "scan_date": scan_date,
        }
    
    def save_report(self, report: Dict, output_dir: str, formats: List[str] = ["txt", "json"]) -> Dict[str, str]:
        """Save report in multiple formats."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_id = report.get("report_id", "report")
        saved_files = {}
        
        if "txt" in formats:
            txt_path = output_dir / f"{report_id}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(report.get("text", ""))
            saved_files["txt"] = str(txt_path)
        
        if "json" in formats:
            json_path = output_dir / f"{report_id}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "report_id": report_id,
                    "scan_date": report.get("scan_date", ""),
                    "text": report.get("text", ""),
                    "parsed": report.get("parsed"),
                }, f, ensure_ascii=False, indent=2)
            saved_files["json"] = str(json_path)
        
        return saved_files


def get_report_generator(use_llm: bool = True, **kwargs) -> Union[ReportGenerator, SimpleReportGenerator]:
    """Factory function to get appropriate report generator."""
    if use_llm:
        try:
            from config import load_config, get_llm_config, get_device
            
            config = load_config()
            llm_config = get_llm_config(config)
            device = get_device(config)
            
            # Get LoRA path from config
            lora_weights = llm_config.get('lora_weights', {})
            lora_path = lora_weights.get('latest', '')
            
            generator_kwargs = {
                'model_name': llm_config.get('model_name', 'meta-llama/Llama-3.2-1B-Instruct'),
                'device': device,
                'load_in_8bit': llm_config.get('load_in_8bit', False),
                'max_new_tokens': llm_config.get('max_length', 1024),
                'temperature': llm_config.get('temperature', 0.3),
                'use_lora': bool(lora_path),
                'lora_path': lora_path,
            }
            generator_kwargs.update(kwargs)
            
            return ReportGenerator(**generator_kwargs)
        except Exception as e:
            print(f"??Failed to initialize LLM generator: {e}")
            return SimpleReportGenerator()
    else:
        return SimpleReportGenerator()



