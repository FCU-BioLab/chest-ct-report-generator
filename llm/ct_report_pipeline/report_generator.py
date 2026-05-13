"""
Report Generator Module

Generates structured CT radiology reports using Llama LLM.
Supports both pre-trained and fine-tuned models with Lung-RADS 2022 classification.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
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
        
        # Build prompt
        prompt = build_report_prompt(
            lesion_features,
            report_id=report_id,
            scan_date=scan_date,
            structured_input=structured_input,
        )
        
        # Log prompt for debugging
        print("\n" + "="*60)
        print("[LLM PROMPT]")
        print("="*60)
        print(prompt)
        print("="*60 + "\n")
        
        # Generate response
        generated_text = self._generate(prompt)
        
        # Post-process
        report_text = self._postprocess(generated_text, report_id, scan_date)
        
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
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,
            )
        
        # Decode
        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        
        return generated.strip()
    
    def _postprocess(self, text: str, report_id: str, scan_date: str) -> str:
        """Clean up the generated report."""
        text = text.strip()
        
        # Remove markers
        if text.startswith("---"):
            text = text[3:].strip()
        if text.endswith("---"):
            text = text[:-3].strip()
        
        # Ensure header exists
        if not text.startswith("Report ID:"):
            text = f"Report ID: {report_id}\nDate: {scan_date}\n\n{text}"
        
        return text
    
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



