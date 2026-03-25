"""
Generate Fine-Tuning Data for CT Report LLM

Creates training pairs with:
- Input: Nodule data (size, volume, type)
- Output: Correctly formatted CT report

This teaches the model to:
1. Follow the exact output format
2. NOT fabricate information
3. Correctly apply Lung-RADS classification
"""

import json
import random
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config, get_output_paths

# Generate synthetic training examples with correct format
def generate_training_examples() -> List[Dict]:
    """Generate synthetic training examples with proper input/output pairs."""
    
    examples = []
    
    # Various nodule configurations
    nodule_configs = [
        # Single small nodule
        [{"size": 3.5, "volume": 22.4, "type": "solid"}],
        [{"size": 4.2, "volume": 38.8, "type": "solid"}],
        [{"size": 5.5, "volume": 87.1, "type": "solid"}],
        
        # Single medium nodule
        [{"size": 6.8, "volume": 164.6, "type": "solid"}],
        [{"size": 7.2, "volume": 195.4, "type": "solid"}],
        
        # Single larger nodule
        [{"size": 9.5, "volume": 449.0, "type": "solid"}],
        [{"size": 12.3, "volume": 974.0, "type": "solid"}],
        [{"size": 16.0, "volume": 2144.7, "type": "solid"}],
        
        # Part-solid nodules
        [{"size": 8.0, "volume": 268.1, "type": "part-solid"}],
        [{"size": 10.5, "volume": 606.1, "type": "part-solid"}],
        
        # Ground-glass nodules
        [{"size": 15.0, "volume": 1767.1, "type": "ground-glass"}],
        [{"size": 25.0, "volume": 8181.2, "type": "ground-glass"}],
        [{"size": 35.0, "volume": 22449.3, "type": "ground-glass"}],
        
        # Multiple nodules
        [{"size": 4.0, "volume": 33.5, "type": "solid"}, 
         {"size": 5.2, "volume": 73.6, "type": "solid"}],
        [{"size": 3.8, "volume": 28.7, "type": "solid"}, 
         {"size": 7.5, "volume": 220.9, "type": "solid"},
         {"size": 4.1, "volume": 36.1, "type": "solid"}],
        [{"size": 6.2, "volume": 124.8, "type": "solid"}, 
         {"size": 8.8, "volume": 356.8, "type": "part-solid"}],
    ]
    
    for config in nodule_configs:
        # Generate input prompt
        nodule_data_lines = []
        for i, nod in enumerate(config, 1):
            nodule_data_lines.append(f"""Nodule {i}:
- Size: {nod['size']:.1f} mm
- Volume: {nod['volume']:.1f} mm³
- Type: {nod['type']}""")
        
        nodule_data = "\n\n".join(nodule_data_lines)
        
        # Create the input prompt (same as REPORT_GENERATION_PROMPT)
        report_id = f"TRAIN_{random.randint(100000, 999999)}"
        scan_date = f"2024/{random.randint(1,12):02d}/{random.randint(1,28):02d}"
        
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
        
        # Generate the expected response
        response = generate_correct_response(config, report_id, scan_date)
        
        examples.append({
            "prompt": prompt,
            "response": response,
            "nodules": config,
        })
    
    return examples


def get_lung_rads(size_mm: float, nodule_type: str) -> Dict:
    """Get Lung-RADS classification."""
    if nodule_type == "ground-glass":
        if size_mm < 30:
            return {"category": "2", "risk": "<1%", "rec": "Continue annual LDCT screening."}
        else:
            return {"category": "3", "risk": "1-2%", "rec": "6-month LDCT follow-up."}
    elif nodule_type == "part-solid":
        if size_mm < 6:
            return {"category": "2", "risk": "<1%", "rec": "Continue annual LDCT screening."}
        else:
            return {"category": "4A", "risk": "5-15%", "rec": "3-month LDCT or PET/CT."}
    else:  # solid
        if size_mm < 6:
            return {"category": "2", "risk": "<1%", "rec": "Continue annual LDCT screening."}
        elif size_mm < 8:
            return {"category": "3", "risk": "1-2%", "rec": "6-month LDCT follow-up."}
        elif size_mm < 15:
            return {"category": "4A", "risk": "5-15%", "rec": "3-month LDCT or PET/CT."}
        else:
            return {"category": "4B", "risk": ">15%", "rec": "PET/CT or tissue sampling."}


def generate_correct_response(nodules: List[Dict], report_id: str, scan_date: str) -> str:
    """Generate a correctly formatted response."""
    
    # Describe nodules
    lung_findings = []
    for i, nod in enumerate(nodules, 1):
        lung_findings.append(
            f"{i}. A {nod['size']:.1f} mm {nod['type']} pulmonary nodule with volume of {nod['volume']:.1f} mm³."
        )
    
    # Get highest Lung-RADS category
    max_category = "2"
    max_lung_rads = None
    largest_size = 0
    
    for nod in nodules:
        lr = get_lung_rads(nod['size'], nod['type'])
        if nod['size'] > largest_size:
            largest_size = nod['size']
        if lr['category'] > max_category:
            max_category = lr['category']
            max_lung_rads = lr
    
    if max_lung_rads is None:
        max_lung_rads = get_lung_rads(nodules[0]['size'], nodules[0]['type'])
    
    response = f"""Report ID: {report_id}
Date: {scan_date}

Technique:
Non-contrast CT chest.

Findings:

Lungs:
{chr(10).join(lung_findings)}

Mediastinum: No masses or lymphadenopathy.
Pleura: No effusion.

Lung-RADS Assessment:
Category: {max_lung_rads['category']}
Malignancy Risk: {max_lung_rads['risk']}

Impression:
1. {len(nodules)} pulmonary nodule(s), largest {largest_size:.1f} mm - Lung-RADS Category {max_lung_rads['category']}

Recommendation:
{max_lung_rads['rec']}"""
    
    return response


def main():
    # Load config and get output paths
    config = load_config()
    output_dir = get_output_paths(config)['training_data']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating fine-tuning training data...")
    
    # Generate examples
    examples = generate_training_examples()
    
    # Duplicate and shuffle for more training data
    all_examples = examples * 10  # 10x for more training
    random.shuffle(all_examples)
    
    # Split into train/val
    split_idx = int(len(all_examples) * 0.9)
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]
    
    # Save training data
    train_file = output_dir / "finetune_train.jsonl"
    with open(train_file, 'w', encoding='utf-8') as f:
        for ex in train_examples:
            # Format for Llama fine-tuning
            data = {
                "messages": [
                    {"role": "user", "content": ex["prompt"]},
                    {"role": "assistant", "content": ex["response"]}
                ]
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print(f"Training data saved: {train_file} ({len(train_examples)} examples)")
    
    # Save validation data
    val_file = output_dir / "finetune_val.jsonl"
    with open(val_file, 'w', encoding='utf-8') as f:
        for ex in val_examples:
            data = {
                "messages": [
                    {"role": "user", "content": ex["prompt"]},
                    {"role": "assistant", "content": ex["response"]}
                ]
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print(f"Validation data saved: {val_file} ({len(val_examples)} examples)")
    
    # Print example
    print("\n" + "="*60)
    print("EXAMPLE TRAINING PAIR:")
    print("="*60)
    print("\n[INPUT PROMPT]")
    print(examples[0]["prompt"][:500] + "...")
    print("\n[EXPECTED OUTPUT]")
    print(examples[0]["response"])


if __name__ == "__main__":
    main()
