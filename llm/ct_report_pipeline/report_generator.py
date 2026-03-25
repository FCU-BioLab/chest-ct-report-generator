"""
Report Generator Module

Generates structured CT radiology reports using Llama LLM.
Supports both pre-trained and fine-tuned models with Lung-RADS 2022 classification.
"""

import os
import sys
import torch
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

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
)


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
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
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
        
        # Build prompt
        prompt = build_report_prompt(
            lesion_features,
            report_id=report_id,
            scan_date=scan_date,
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
            "parsed": None,
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
                }, f, ensure_ascii=False, indent=2)
            saved_files["json"] = str(json_path)
        
        return saved_files


class SimpleReportGenerator:
    """
    Template-based report generator (no LLM required).
    Uses Lung-RADS 2022 classification.
    """
    
    @staticmethod
    def get_nodule_type(mean_hu: float) -> str:
        """Determine nodule type from HU value."""
        if mean_hu < -600:
            return "ground-glass"
        elif mean_hu < -300:
            return "part-solid"
        elif mean_hu > 200:
            return "calcified"
        else:
            return "solid"
    
    @staticmethod
    def get_lung_rads_category(size_mm: float, nodule_type: str) -> Dict:
        """Calculate Lung-RADS 2022 category."""
        if nodule_type == "calcified":
            return {
                "category": "1",
                "description": "Negative - Calcified nodule (benign)",
                "probability": "<1%",
                "recommendation": "Continue annual LDCT screening."
            }
        
        if nodule_type == "ground-glass":
            if size_mm < 30:
                return {
                    "category": "2",
                    "description": "Benign appearance - GGN <30mm",
                    "probability": "<1%",
                    "recommendation": "Continue annual LDCT screening."
                }
            else:
                return {
                    "category": "3",
                    "description": "Probably benign - GGN ??0mm",
                    "probability": "1-2%",
                    "recommendation": "6-month LDCT follow-up."
                }
        
        if nodule_type == "part-solid":
            if size_mm < 6:
                return {
                    "category": "2",
                    "description": "Benign appearance - Part-solid <6mm",
                    "probability": "<1%",
                    "recommendation": "Continue annual LDCT screening."
                }
            else:
                return {
                    "category": "4A",
                    "description": "Suspicious - Part-solid ??mm",
                    "probability": "5-15%",
                    "recommendation": "3-month LDCT or PET/CT; tissue sampling if PET positive."
                }
        
        # Solid nodule
        if size_mm < 6:
            return {
                "category": "2",
                "description": "Benign appearance - Solid <6mm",
                "probability": "<1%",
                "recommendation": "Continue annual LDCT screening."
            }
        elif size_mm < 8:
            return {
                "category": "3",
                "description": "Probably benign - Solid 6-<8mm",
                "probability": "1-2%",
                "recommendation": "6-month LDCT follow-up."
            }
        elif size_mm < 15:
            return {
                "category": "4A",
                "description": "Suspicious - Solid 8-<15mm",
                "probability": "5-15%",
                "recommendation": "3-month LDCT or PET/CT."
            }
        else:
            return {
                "category": "4B",
                "description": "Very suspicious - Solid ??5mm",
                "probability": ">15%",
                "recommendation": "PET/CT and/or tissue sampling."
            }
    
    def generate_report(
        self,
        lesion_features: Union[Dict, List[Dict]],
        report_id: str = None,
        scan_date: str = None,
        return_xml: bool = False,
    ) -> Dict[str, str]:
        """Generate a template-based report."""
        if isinstance(lesion_features, dict):
            lesion_features = [lesion_features]
        
        if not report_id:
            report_id = f"AUTO_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        if not scan_date:
            scan_date = datetime.now().strftime("%Y/%m/%d")
        
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
        max_category = "1"
        max_lung_rads = None
        
        for i, features in enumerate(lesion_features, 1):
            size_mm = features.get("equivalent_diameter_mm", 0)
            volume_mm3 = features.get("volume_mm3", 0)
            mean_hu = features.get("mean_hu", 0)
            
            nodule_type = self.get_nodule_type(mean_hu)
            lung_rads = self.get_lung_rads_category(size_mm, nodule_type)
            
            # Track highest category
            if lung_rads["category"] > max_category:
                max_category = lung_rads["category"]
                max_lung_rads = lung_rads
            
            lines.append(f"{i}. A {nodule_type} pulmonary nodule measuring {size_mm:.1f} mm (ESD) "
                        f"with volume of {volume_mm3:.1f} mm糧 is identified.")
        
        if max_lung_rads is None:
            max_lung_rads = self.get_lung_rads_category(
                lesion_features[0].get("equivalent_diameter_mm", 0),
                self.get_nodule_type(lesion_features[0].get("mean_hu", 0))
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
            f"Category: {max_lung_rads['category']}",
            f"Description: {max_lung_rads['description']}",
            f"Malignancy Probability: {max_lung_rads['probability']}",
            "",
            "Impression:",
            f"1. {len(lesion_features)} pulmonary nodule(s) identified - Lung-RADS Category {max_lung_rads['category']}.",
            "",
            "Recommendation:",
            max_lung_rads['recommendation'],
        ])
        
        report_text = "\n".join(lines)
        
        return {
            "text": report_text,
            "xml": None,
            "parsed": None,
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


