#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate YOLO format dataset structure before training.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def validate_yolo_label(label_path: Path) -> Tuple[bool, str]:
    """
    Validate YOLO format label file.
    
    Returns:
        (is_valid, error_message)
    """
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            # Empty file is valid (no objects)
            return True, ""
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) != 5:
                return False, f"Line {i+1}: Expected 5 values, got {len(parts)}"
            
            try:
                cls, x, y, w, h = map(float, parts)
            except ValueError:
                return False, f"Line {i+1}: Values must be numeric"
            
            # Check class ID (should be 0 for single class)
            if cls != 0:
                return False, f"Line {i+1}: Class ID should be 0, got {cls}"
            
            # Check normalized coordinates (should be 0-1)
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                return False, f"Line {i+1}: Coordinates must be normalized (0-1)"
        
        return True, ""
        
    except Exception as e:
        return False, f"Failed to read file: {e}"


def validate_patient_directory(patient_dir: Path) -> Dict:
    """Validate a single patient directory."""
    result = {
        "patient_id": patient_dir.name,
        "valid": True,
        "images_count": 0,
        "labels_count": 0,
        "matched_pairs": 0,
        "errors": [],
        "warnings": [],
    }
    
    images_dir = patient_dir / "images"
    labels_dir = patient_dir / "labels"
    
    # Check directory existence
    if not images_dir.exists():
        result["valid"] = False
        result["errors"].append("Missing images/ directory")
        return result
    
    if not labels_dir.exists():
        result["valid"] = False
        result["errors"].append("Missing labels/ directory")
        return result
    
    # Collect image files
    image_files = list(images_dir.glob("*.png"))
    result["images_count"] = len(image_files)
    
    if not image_files:
        result["warnings"].append("No PNG images found")
    
    # Check labels
    matched = 0
    for img_file in image_files:
        label_file = labels_dir / f"{img_file.stem}.txt"
        
        if not label_file.exists():
            result["warnings"].append(f"Missing label: {img_file.name}")
            continue
        
        # Validate label format
        is_valid, error_msg = validate_yolo_label(label_file)
        if not is_valid:
            result["errors"].append(f"{label_file.name}: {error_msg}")
            result["valid"] = False
        else:
            matched += 1
    
    result["matched_pairs"] = matched
    result["labels_count"] = len(list(labels_dir.glob("*.txt")))
    
    # Check for extra labels
    if result["labels_count"] > result["images_count"]:
        result["warnings"].append(
            f"More labels ({result['labels_count']}) than images ({result['images_count']})"
        )
    
    return result


def validate_dataset(data_dir: Path, verbose: bool = False) -> Dict:
    """Validate entire dataset."""
    print(f"🔍 Validating dataset: {data_dir}")
    print("=" * 80)
    
    if not data_dir.exists():
        print(f"❌ Error: Directory not found: {data_dir}")
        return None
    
    # Collect patient directories
    patient_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    if not patient_dirs:
        print(f"❌ Error: No patient directories found in {data_dir}")
        return None
    
    print(f"Found {len(patient_dirs)} patient directories")
    print()
    
    # Validate each patient
    results = []
    total_images = 0
    total_matched = 0
    total_errors = 0
    total_warnings = 0
    
    for patient_dir in sorted(patient_dirs):
        result = validate_patient_directory(patient_dir)
        results.append(result)
        
        total_images += result["images_count"]
        total_matched += result["matched_pairs"]
        total_errors += len(result["errors"])
        total_warnings += len(result["warnings"])
        
        # Print patient result
        if verbose or not result["valid"] or result["warnings"]:
            status = "✅" if result["valid"] else "❌"
            print(f"{status} {result['patient_id']}: {result['matched_pairs']}/{result['images_count']} images")
            
            if result["errors"]:
                for error in result["errors"]:
                    print(f"   ❌ {error}")
            
            if result["warnings"] and verbose:
                for warning in result["warnings"]:
                    print(f"   ⚠️  {warning}")
    
    # Summary
    print()
    print("=" * 80)
    print("📊 Validation Summary")
    print("=" * 80)
    print(f"Total Patients: {len(patient_dirs)}")
    print(f"Total Images: {total_images}")
    print(f"Matched Pairs: {total_matched}")
    print(f"Errors: {total_errors}")
    print(f"Warnings: {total_warnings}")
    print()
    
    valid_patients = sum(1 for r in results if r["valid"])
    invalid_patients = len(patient_dirs) - valid_patients
    
    if invalid_patients == 0:
        print("✅ Dataset validation PASSED!")
        print(f"   All {len(patient_dirs)} patients are valid")
    else:
        print(f"⚠️  Dataset validation FAILED!")
        print(f"   Valid patients: {valid_patients}/{len(patient_dirs)}")
        print(f"   Invalid patients: {invalid_patients}")
    
    print("=" * 80)
    
    # Return summary
    summary = {
        "total_patients": len(patient_dirs),
        "valid_patients": valid_patients,
        "invalid_patients": invalid_patients,
        "total_images": total_images,
        "matched_pairs": total_matched,
        "total_errors": total_errors,
        "total_warnings": total_warnings,
        "patient_results": results,
    }
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Validate YOLO format dataset structure"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../../datasets/splited_dataset/train",
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed information for each patient"
    )
    parser.add_argument(
        "--save_report",
        type=str,
        default="",
        help="Save validation report to JSON file"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    summary = validate_dataset(data_dir, verbose=args.verbose)
    
    if summary and args.save_report:
        report_path = Path(args.save_report)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Report saved to: {report_path}")
    
    # Exit code based on validation result
    if summary and summary["invalid_patients"] == 0:
        exit(0)
    else:
        exit(1)


if __name__ == "__main__":
    main()
