#!/usr/bin/env python3
"""
LLM Feature Usage Example
=========================

This script demonstrates how to use the extracted features for LLM-based analysis.

Usage:
    python llm_feature_usage_example.py --patient_id A0001
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List


def find_latest_result_dir(base_dir: str = "segmentation_result") -> Path:
    """Find the latest timestamped result directory"""
    base_path = Path(base_dir)
    if not base_path.exists():
        return None
    
    # Check for timestamped directories (format: YYYYMMDD_HHMMSS)
    timestamped_dirs = [d for d in base_path.iterdir() 
                       if d.is_dir() and len(d.name) == 15 and d.name[8] == '_']
    
    if timestamped_dirs:
        # Return the latest timestamped directory
        return sorted(timestamped_dirs)[-1]
    else:
        # If no timestamped directories, use base directory directly
        return base_path


def load_patient_features(patient_id: str, features_dir: str = "segmentation_result") -> Dict:
    """Load extracted features for a patient"""
    base_path = Path(features_dir)
    
    # If features_dir looks like a timestamped directory, use it directly
    if len(base_path.name) == 15 and base_path.name[8] == '_':
        result_dir = base_path
    else:
        # Try to find the latest timestamped directory
        result_dir = find_latest_result_dir(features_dir)
        if result_dir is None:
            print(f"No result directory found in {features_dir}")
            return {}
    
    feature_path = result_dir / patient_id / "llm_features"
    
    if not feature_path.exists():
        print(f"No features found for patient {patient_id} in {result_dir}")
        return {}
    
    # Find the latest features file
    json_files = list(feature_path.glob("features_*.json"))
    if not json_files:
        print(f"No feature JSON files found for patient {patient_id}")
        return {}
    
    latest_file = sorted(json_files)[-1]
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        features = json.load(f)
    
    print(f"Loaded features from: {result_dir.name}/{patient_id}/llm_features/{latest_file.name}")
    return features


def generate_llm_prompt(features: Dict) -> str:
    """Generate a comprehensive prompt for LLM analysis"""
    
    if not features:
        return "No features available for analysis."
    
    patient_id = features.get('patient_id', 'Unknown')
    total_slices = features.get('total_slices', 0)
    
    prompt = f"""# Medical Image Analysis Report Request

## Patient Information
- Patient ID: {patient_id}
- Total CT Slices Analyzed: {total_slices}

## Lesion Analysis Data

"""
    
    # Add detailed lesion information
    for i, slice_data in enumerate(features.get('slices', []), 1):
        prompt += f"### Slice {i}: {slice_data['dicom_file']}\n\n"
        
        for j, lesion in enumerate(slice_data.get('lesions', []), 1):
            prompt += f"**Lesion {j}**\n\n"
            prompt += f"{lesion.get('description', 'No description available')}\n\n"
            
            # Add quantitative details
            prompt += "**Quantitative Measurements:**\n"
            prompt += f"- Area: {lesion.get('area_mm2', 0):.2f} mm²\n"
            prompt += f"- Equivalent Diameter: {lesion.get('equivalent_diameter_mm', 0):.2f} mm\n"
            prompt += f"- Major Axis: {lesion.get('major_axis_length_mm', 0):.2f} mm\n"
            prompt += f"- Minor Axis: {lesion.get('minor_axis_length_mm', 0):.2f} mm\n"
            prompt += f"- Circularity: {lesion.get('circularity', 0):.3f}\n"
            prompt += f"- Solidity: {lesion.get('solidity', 0):.3f}\n"
            prompt += f"- Mean Intensity: {lesion.get('mean_intensity', 0):.1f}\n"
            prompt += f"- Contrast to Background: {lesion.get('contrast_to_background', 0):.1f}\n"
            prompt += f"- Texture Entropy: {lesion.get('entropy', 0):.2f}\n"
            prompt += f"- Edge Strength: {lesion.get('edge_strength', 0):.2f}\n"
            prompt += f"- Slice Location: {lesion.get('slice_location', 0):.1f} mm\n\n"
        
        prompt += "---\n\n"
    
    # Add analysis request
    prompt += """## Analysis Request

Based on the above lesion characteristics, please provide:

1. **Lesion Classification**: Analyze the morphological and intensity features to suggest possible lesion types (e.g., nodule, mass, consolidation, ground-glass opacity)

2. **Clinical Significance**: Assess the clinical relevance based on size, shape, and texture characteristics

3. **Comparison Analysis**: If multiple lesions are present, compare their characteristics and discuss any patterns

4. **Recommendations**: Suggest appropriate follow-up actions or additional imaging if needed

5. **Summary**: Provide a concise summary suitable for a radiology report

Please structure your response in a clear, clinical format appropriate for medical documentation.
"""
    
    return prompt


def generate_structured_summary(features: Dict) -> Dict:
    """Generate a structured summary of all lesions for programmatic use"""
    
    summary = {
        'patient_id': features.get('patient_id', 'Unknown'),
        'total_lesions': 0,
        'lesion_statistics': {
            'total_area_mm2': 0,
            'mean_area_mm2': 0,
            'mean_diameter_mm': 0,
            'mean_circularity': 0,
            'mean_intensity': 0,
        },
        'lesion_locations': [],
        'lesion_details': []
    }
    
    all_lesions = []
    
    for slice_data in features.get('slices', []):
        for lesion in slice_data.get('lesions', []):
            all_lesions.append(lesion)
            summary['lesion_details'].append({
                'name': lesion.get('lesion_name', 'Unknown'),
                'slice': slice_data['dicom_file'],
                'area_mm2': lesion.get('area_mm2', 0),
                'diameter_mm': lesion.get('equivalent_diameter_mm', 0),
                'position': f"({lesion.get('relative_position_x', 0):.2f}, {lesion.get('relative_position_y', 0):.2f})",
                'slice_location': lesion.get('slice_location', 0)
            })
    
    summary['total_lesions'] = len(all_lesions)
    
    if all_lesions:
        summary['lesion_statistics']['total_area_mm2'] = sum(l.get('area_mm2', 0) for l in all_lesions)
        summary['lesion_statistics']['mean_area_mm2'] = summary['lesion_statistics']['total_area_mm2'] / len(all_lesions)
        summary['lesion_statistics']['mean_diameter_mm'] = sum(l.get('equivalent_diameter_mm', 0) for l in all_lesions) / len(all_lesions)
        summary['lesion_statistics']['mean_circularity'] = sum(l.get('circularity', 0) for l in all_lesions) / len(all_lesions)
        summary['lesion_statistics']['mean_intensity'] = sum(l.get('mean_intensity', 0) for l in all_lesions) / len(all_lesions)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="LLM Feature Usage Example")
    parser.add_argument("--patient_id", type=str, required=True, help="Patient ID to analyze")
    parser.add_argument("--features_dir", type=str, default="segmentation_result", 
                       help="Features directory (can be base dir or specific timestamped dir)")
    parser.add_argument("--output_prompt", type=str, help="Output file for LLM prompt")
    
    args = parser.parse_args()
    
    # Load features
    features = load_patient_features(args.patient_id, args.features_dir)
    
    if not features:
        print("\nAvailable timestamped directories:")
        base_path = Path(args.features_dir)
        if base_path.exists():
            timestamped_dirs = sorted([d.name for d in base_path.iterdir() 
                                     if d.is_dir() and len(d.name) == 15 and d.name[8] == '_'])
            if timestamped_dirs:
                for dir_name in timestamped_dirs:
                    print(f"  - {dir_name}")
                print(f"\nTry: python llm_feature_usage_example.py --patient_id {args.patient_id} --features_dir {args.features_dir}/{timestamped_dirs[-1]}")
            else:
                print(f"  No timestamped directories found in {args.features_dir}")
        return
    
    # Generate LLM prompt
    prompt = generate_llm_prompt(features)
    print("\n" + "="*80)
    print("GENERATED LLM PROMPT")
    print("="*80 + "\n")
    print(prompt)
    
    # Save prompt if output file specified
    if args.output_prompt:
        with open(args.output_prompt, 'w', encoding='utf-8') as f:
            f.write(prompt)
        print(f"\nPrompt saved to: {args.output_prompt}")
    
    # Generate structured summary
    summary = generate_structured_summary(features)
    print("\n" + "="*80)
    print("STRUCTURED SUMMARY")
    print("="*80 + "\n")
    print(json.dumps(summary, indent=2))
    
    # Save summary
    # Determine the correct path for saving
    base_path = Path(args.features_dir)
    if len(base_path.name) == 15 and base_path.name[8] == '_':
        result_dir = base_path
    else:
        result_dir = find_latest_result_dir(args.features_dir)
    
    summary_path = result_dir / args.patient_id / "llm_features" / "structured_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"\nStructured summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
