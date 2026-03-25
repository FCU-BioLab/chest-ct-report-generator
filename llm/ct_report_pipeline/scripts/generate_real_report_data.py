"""
Generate Fine-Tuning Data from Real CT Reports

This script:
1. Parses reports from splited_reports/
2. Extracts nodule information (size, type, location)
3. Generates prompts matching segmentation feature format
4. Adds Lung-RADS 2022 assessment to responses
5. Creates JSONL training data

Output format matches interactive_segmentation.py output.
"""

import os
import re
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import random

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config, get_reports_config


def estimate_volume_from_diameter(diameter_mm: float) -> float:
    """Estimate volume assuming spherical nodule."""
    radius = diameter_mm / 2
    return (4/3) * math.pi * (radius ** 3)


def get_nodule_type(text_context: str) -> str:
    """Infer nodule type from surrounding text."""
    text_lower = text_context.lower()
    if "ground glass" in text_lower or "ggo" in text_lower or "ggn" in text_lower:
        return "ground-glass"
    elif "part-solid" in text_lower or "part solid" in text_lower:
        return "part-solid"
    elif "calcif" in text_lower:
        return "calcified"
    else:
        return "solid"


def get_lung_rads(size_mm: float, nodule_type: str) -> Dict:
    """Calculate Lung-RADS 2022 classification."""
    if nodule_type == "calcified":
        return {
            "category": "1",
            "risk": "<1%",
            "recommendation": "Continue annual LDCT screening."
        }
    
    if nodule_type == "ground-glass":
        if size_mm < 30:
            return {"category": "2", "risk": "<1%", "recommendation": "Continue annual LDCT screening."}
        else:
            return {"category": "3", "risk": "1-2%", "recommendation": "6-month LDCT follow-up."}
    
    if nodule_type == "part-solid":
        if size_mm < 6:
            return {"category": "2", "risk": "<1%", "recommendation": "Continue annual LDCT screening."}
        else:
            return {"category": "4A", "risk": "5-15%", "recommendation": "3-month LDCT or PET/CT."}
    
    # Solid nodule
    if size_mm < 6:
        return {"category": "2", "risk": "<1%", "recommendation": "Continue annual LDCT screening."}
    elif size_mm < 8:
        return {"category": "3", "risk": "1-2%", "recommendation": "6-month LDCT follow-up."}
    elif size_mm < 15:
        return {"category": "4A", "risk": "5-15%", "recommendation": "3-month LDCT or PET/CT."}
    else:
        return {"category": "4B", "risk": ">15%", "recommendation": "PET/CT or tissue sampling."}


def extract_nodules_from_report(report_text: str) -> List[Dict]:
    """Extract nodule information from report text."""
    nodules = []
    
    # Pattern to match size mentions (e.g., "size 18.6mm", "16.3mm", "35mm nodule")
    size_pattern = r'(?:size\s+)?(\d+(?:\.\d+)?)\s*mm'
    
    # Find all size mentions
    for match in re.finditer(size_pattern, report_text, re.IGNORECASE):
        size_mm = float(match.group(1))
        
        # Skip very small or very large (likely not nodules)
        if size_mm < 2 or size_mm > 100:
            continue
        
        # Get context around the match (150 chars before and after)
        start = max(0, match.start() - 150)
        end = min(len(report_text), match.end() + 150)
        context = report_text[start:end].lower()
        
        # Check if this is a nodule/opacity mention (not bone, heart, etc.)
        nodule_keywords = ['nodule', 'nodular', 'opacity', 'lesion', 'mass', 'tumor', 'ggo', 
                           'ground glass', 'lung', 'pulmonary', 'lobe', 'rul', 'rml', 'rll', 
                           'lul', 'lll', 'lingula']
        
        is_nodule = any(kw in context for kw in nodule_keywords)
        
        # Skip if in bone/heart context
        skip_keywords = ['spine', 'vertebr', 'heart', 'aort', 'lymph node']
        is_skip = any(kw in context for kw in skip_keywords)
        
        if is_nodule and not is_skip:
            nodule_type = get_nodule_type(context)
            volume_mm3 = estimate_volume_from_diameter(size_mm)
            
            nodules.append({
                "size_mm": size_mm,
                "volume_mm3": round(volume_mm3, 1),
                "type": nodule_type,
            })
    
    # Remove duplicates (same size)
    seen_sizes = set()
    unique_nodules = []
    for n in nodules:
        if n["size_mm"] not in seen_sizes:
            seen_sizes.add(n["size_mm"])
            unique_nodules.append(n)
    
    return unique_nodules


def generate_prompt(nodules: List[Dict], report_id: str) -> str:
    """Generate a prompt matching the segmentation feature format."""
    nodule_data_lines = []
    for i, nod in enumerate(nodules, 1):
        nodule_data_lines.append(f"""Nodule {i}:
- Size: {nod['size_mm']:.1f} mm
- Volume: {nod['volume_mm3']:.1f} mm³
- Type: {nod['type']}""")
    
    nodule_data = "\n\n".join(nodule_data_lines)
    scan_date = datetime.now().strftime("%Y/%m/%d")
    
    prompt = f"""You are a radiologist. Write a CT chest report for the nodule(s) below.

RULES:
- Report ONLY the nodules listed - do NOT add extra nodules
- Do NOT fabricate location, patient ID, or clinical history
- SELECT the correct Lung-RADS category based on nodule size

NODULE DATA:
{nodule_data}

LUNG-RADS GUIDE:
- Solid <6mm → Category 2, <1% malignancy, annual screening
- Solid 6-8mm → Category 3, 1-2% malignancy, 6-month follow-up
- Solid 8-15mm → Category 4A, 5-15% malignancy, 3-month follow-up or PET/CT
- Solid ≥15mm → Category 4B, >15% malignancy, PET/CT or biopsy

Write the report:

Report ID: {report_id}
Date: {scan_date}

Technique:
Non-contrast CT chest.

Findings:

Lungs:
[Describe each nodule: size, type, volume]

Mediastinum: No masses or lymphadenopathy.
Pleura: No effusion.

Lung-RADS Assessment:
Category: [SELECT ONE: 2 or 3 or 4A or 4B]
Malignancy Risk: [SELECT ONE: <1% or 1-2% or 5-15% or >15%]

Impression:
[Summarize: count, largest size, Lung-RADS category]

Recommendation:
[SELECT based on category: annual screening / 6-month CT / 3-month CT / PET-CT]"""
    
    return prompt


def enhance_report_with_lung_rads(report_text: str, nodules: List[Dict], report_id: str) -> str:
    """Enhance the original report with Lung-RADS 2022 assessment."""
    
    # Get highest Lung-RADS category
    max_category = "1"
    max_lung_rads = None
    largest_size = 0
    
    for nod in nodules:
        lr = get_lung_rads(nod['size_mm'], nod['type'])
        if nod['size_mm'] > largest_size:
            largest_size = nod['size_mm']
        if lr['category'] > max_category:
            max_category = lr['category']
            max_lung_rads = lr
    
    if max_lung_rads is None and nodules:
        max_lung_rads = get_lung_rads(nodules[0]['size_mm'], nodules[0]['type'])
    elif max_lung_rads is None:
        max_lung_rads = {"category": "1", "risk": "<1%", "recommendation": "Continue annual LDCT screening."}
    
    # Clean the original report (remove garbled footer)
    clean_report = re.sub(r'===.*===', '', report_text).strip()
    clean_report = re.sub(r'查單號:.*\n', '', clean_report).strip()
    
    # Generate nodule descriptions
    nodule_lines = []
    for i, nod in enumerate(nodules, 1):
        nodule_lines.append(f"{i}. A {nod['size_mm']:.1f} mm {nod['type']} pulmonary nodule with volume of {nod['volume_mm3']:.1f} mm³.")
    
    scan_date = datetime.now().strftime("%Y/%m/%d")
    
    # Build enhanced report
    enhanced_report = f"""Report ID: {report_id}
Date: {scan_date}

Technique:
Non-contrast CT chest.

Findings:

Lungs:
{chr(10).join(nodule_lines) if nodule_lines else "No significant lung nodules identified."}

Mediastinum: No masses or lymphadenopathy.
Pleura: No effusion.

Lung-RADS Assessment:
Category: {max_lung_rads['category']}
Malignancy Risk: {max_lung_rads['risk']}

Impression:
1. {len(nodules)} pulmonary nodule(s), largest {largest_size:.1f} mm - Lung-RADS Category {max_lung_rads['category']}

Recommendation:
{max_lung_rads['recommendation']}"""
    
    return enhanced_report


def process_reports(
    reports_dir: str,
    output_jsonl: str,
    max_reports: int = None,
) -> None:
    """Process all reports and generate JSONL training file."""
    reports_dir = Path(reports_dir)
    
    training_data = []
    report_files = list(reports_dir.glob("*.txt"))
    
    if max_reports:
        report_files = report_files[:max_reports]
    
    for i, report_file in enumerate(report_files):
        try:
            with open(report_file, 'r', encoding='utf-8', errors='ignore') as f:
                report_text = f.read()
            
            # Extract report ID
            report_id = report_file.stem
            
            # Extract nodules
            nodules = extract_nodules_from_report(report_text)
            
            # Skip reports without nodules
            if not nodules:
                continue
            
            # Generate prompt
            prompt = generate_prompt(nodules, report_id)
            
            # Generate enhanced response with Lung-RADS
            response = enhance_report_with_lung_rads(report_text, nodules, report_id)
            
            # Add to training data
            training_data.append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ],
                "report_id": report_id,
                "nodule_count": len(nodules),
            })
            
            if (i + 1) % 20 == 0:
                print(f"Processed {i + 1}/{len(report_files)} reports ({len(training_data)} with nodules)")
            
        except Exception as e:
            print(f"Error processing {report_file.name}: {e}")
    
    # Save JSONL training file
    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"{'='*60}")
    print(f"Total reports processed: {len(report_files)}")
    print(f"Reports with nodules: {len(training_data)}")
    print(f"Training data saved to: {output_jsonl}")
    
    # Print example
    if training_data:
        print(f"\n{'='*60}")
        print("EXAMPLE TRAINING PAIR:")
        print(f"{'='*60}")
        print("\n[USER PROMPT]")
        print(training_data[0]["messages"][0]["content"][:600] + "...")
        print("\n[ASSISTANT RESPONSE]")
        print(training_data[0]["messages"][1]["content"])


def main():
    import argparse
    
    # Load config for default paths
    config = load_config()
    reports_config = get_reports_config(config)
    
    reports_dir_default = str(reports_config['raw_reports'])
    output_default = str(reports_config['processed_reports'])
    
    parser = argparse.ArgumentParser(description="Generate fine-tuning data from real CT reports")
    parser.add_argument(
        "--reports_dir",
        type=str,
        default=reports_dir_default,
        help=f"Directory containing original reports (default: {reports_dir_default})"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=output_default,
        help=f"Path for JSONL training file (default: {output_default})"
    )
    parser.add_argument(
        "--max_reports",
        type=int,
        default=None,
        help="Maximum number of reports to process"
    )
    
    args = parser.parse_args()
    
    process_reports(
        args.reports_dir,
        args.output,
        args.max_reports,
    )


if __name__ == "__main__":
    main()
