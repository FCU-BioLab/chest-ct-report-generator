"""
Dataset Preparation Script

Converts LNDb dataset to the intermediate JSON format required by the pipeline.
Parses LNDb annotations and uses pre-annotated masks from radiologists.
"""

import json
import argparse
import csv
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from typing import List, Dict, Tuple
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config, get_lndb_root, get_output_paths




def load_lndb_annotations(lndb_root: Path) -> Tuple[Dict, Dict]:
    """
    Load LNDb annotations from trainset_csv directory.
    
    Args:
        lndb_root: Root directory of LNDb dataset
    
    Returns:
        Tuple of (nodule_annotations, ct_info)
    """
    annotations = {}
    ct_info = {}
    
    # LNDb has trainset_csv directory with annotation files
    trainset_dir = lndb_root / "trainset_csv"
    
    if not trainset_dir.exists():
        print(f"Warning: trainset_csv directory not found: {trainset_dir}")
        return annotations, ct_info
    
    print(f"Found trainset_csv directory: {trainset_dir}")
    
    # Load trainNodules.csv - contains nodule locations and info
    nodules_csv = trainset_dir / "trainNodules.csv"
    if nodules_csv.exists():
        with open(nodules_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                lndb_id = int(row['LNDbID'])
                rad_id = int(row['RadID'])
                finding_id = int(row['FindingID'])
                
                # Create scan key
                scan_key = f"LNDb-{lndb_id:04d}"
                
                if scan_key not in annotations:
                    annotations[scan_key] = []
                
                # Store nodule info
                nodule_info = {
                    'radiologist': rad_id,
                    'finding_id': finding_id,
                    'x': float(row['x']),
                    'y': float(row['y']),
                    'z': float(row['z']),
                    'is_nodule': int(row['Nodule']) == 1,
                    'volume': float(row['Volume']) if row['Volume'] else 0.0,
                    'text': row.get('Text', '')
                }
                
                annotations[scan_key].append(nodule_info)
    
    # Load trainCTs.csv - contains CT scan info (just RadN - number of radiologists)
    cts_csv = trainset_dir / "trainCTs.csv"
    if cts_csv.exists():
        with open(cts_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                lndb_id = int(row['LNDbID'])
                scan_key = f"LNDb-{lndb_id:04d}"
                ct_info[scan_key] = {
                    'num_radiologists': int(row.get('RadN', 1))
                }
    
    print(f"Loaded annotations for {len(annotations)} scans")
    print(f"Loaded CT info for {len(ct_info)} scans")
    
    return annotations, ct_info


def find_mask_files(lndb_root: Path, scan_id: str) -> List[Path]:
    """
    Find all mask files for a given scan.
    
    Args:
        lndb_root: Root directory of LNDb dataset
        scan_id: Scan identifier (e.g., LNDb-0001)
    
    Returns:
        List of mask file paths
    """
    masks_dir = lndb_root / "masks" / "masks"
    
    if not masks_dir.exists():
        return []
    
    # LNDb masks are named like: LNDb-0001_rad1.mhd, LNDb-0001_rad2.mhd, etc.
    mask_files = list(masks_dir.glob(f"{scan_id}_rad*.mhd"))
    
    return sorted(mask_files)


def create_region_entry(
    region_id: int,
    mask_path: str,
    region_type: str = "nodule",
    text_gt: str = "",
    nodule_info: Dict = None
) -> Dict:
    """
    Create a region entry in the intermediate format.
    
    Args:
        region_id: Region identifier
        mask_path: Path to mask file
        region_type: Type of region (default: nodule)
        text_gt: Ground truth text description
        nodule_info: Additional nodule information
    
    Returns:
        Region dictionary
    """
    region = {
        "region_id": region_id,
        "mask": mask_path,
        "region_type": region_type,
        "text_gt": text_gt
    }
    
    # Add additional nodule info if available
    if nodule_info:
        region["metadata"] = {
            "volume_mm3": nodule_info.get('volume', 0.0),
            "center_x": nodule_info.get('x', 0.0),
            "center_y": nodule_info.get('y', 0.0),
            "center_z": nodule_info.get('z', 0.0)
        }
    
    return region


def process_single_scan(
    scan_id: str,
    ct_path: Path,
    annotations: List[Dict],
    lndb_root: Path,
    output_dir: Path
) -> Dict:
    """
    Process a single CT scan and create intermediate format.
    
    Args:
        scan_id: Scan identifier (e.g., LNDb-0001)
        ct_path: Path to CT volume
        annotations: List of nodule annotations for this scan
        lndb_root: LNDb root directory
        output_dir: Output directory for processed data
    
    Returns:
        Scan dictionary in intermediate format
    """
    # Create output directory for this scan
    scan_output_dir = output_dir / scan_id
    scan_output_dir.mkdir(parents=True, exist_ok=True)
    
    regions = []
    
    # Find mask files for this scan
    mask_files = find_mask_files(lndb_root, scan_id)
    
    if mask_files:
        print(f"    Found {len(mask_files)} mask files (radiologist annotations)")
        
        # Use masks from radiologists
        # Typically LNDb has 3 radiologists (rad1, rad2, rad3)
        # We'll use rad1 as the primary annotation
        primary_mask = None
        for mask_file in mask_files:
            if "_rad1" in mask_file.stem:
                primary_mask = mask_file
                break
        
        if primary_mask is None and mask_files:
            primary_mask = mask_files[0]
        
        if primary_mask:
            # Load mask to count regions
            try:
                sitk_mask = sitk.ReadImage(str(primary_mask))
                mask_array = sitk.GetArrayFromImage(sitk_mask)
                
                # Find unique labels (excluding background 0)
                unique_labels = np.unique(mask_array)
                unique_labels = unique_labels[unique_labels > 0]
                
                print(f"    Mask contains {len(unique_labels)} labeled regions")
                
                # Create region entries for each labeled region
                for idx, label in enumerate(unique_labels):
                    # Find corresponding annotation if available
                    nodule_info = None
                    text_description = ""
                    
                    if annotations and idx < len(annotations):
                        nodule_info = annotations[idx]
                        text_description = nodule_info.get('text', '')
                        if not text_description:
                            # Generate basic description
                            volume = nodule_info.get('volume', 0.0)
                            text_description = f"Nodule with volume {volume:.2f} mm³"
                    
                    region = create_region_entry(
                        region_id=idx,
                        mask_path=str(primary_mask),
                        region_type="nodule",
                        text_gt=text_description,
                        nodule_info=nodule_info
                    )
                    
                    # Add label information
                    region["mask_label"] = int(label)
                    
                    regions.append(region)
                    
            except Exception as e:
                print(f"    Warning: Could not load mask {primary_mask}: {e}")
    
    else:
        print(f"    No mask files found")
    
    # Generate global report from individual nodule descriptions
    global_report = ""
    if regions:
        nodule_descriptions = [r.get('text_gt', '') for r in regions if r.get('text_gt')]
        if nodule_descriptions:
            global_report = " ".join(nodule_descriptions)
    
    scan_data = {
        "scan_id": scan_id,
        "ct_volume": str(ct_path),
        "regions": regions,
        "global_report_gt": global_report
    }
    
    return scan_data


def main():
    # Load configuration
    config = load_config()
    
    # Get default paths from config
    lndb_default = str(get_lndb_root(config))
    output_default = str(get_output_paths(config)['processed_data'])
    
    parser = argparse.ArgumentParser(
        description="Convert LNDb dataset to intermediate JSON format"
    )
    parser.add_argument(
        "--lndb_root",
        type=str,
        default=lndb_default,
        help=f"Root directory of LNDb dataset (default: {lndb_default})"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=output_default,
        help=f"Output directory for processed data (default: {output_default})"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="dataset.json",
        help="Output JSON file name"
    )
    
    args = parser.parse_args()
    
    lndb_root = Path(args.lndb_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing LNDb dataset from {lndb_root}")
    print(f"Output directory: {output_dir}")
    print()
    
    if not lndb_root.exists():
        print(f"Error: LNDb root directory not found: {lndb_root}")
        print("Please check the path in config/config.yaml")
        return
    
    # Load annotations
    print("Loading LNDb annotations...")
    annotations_dict, ct_info_dict = load_lndb_annotations(lndb_root)

    print()
    
    # Process all scans
    dataset = []
    
    # LNDb structure: data0, data1, data2, data3, data4, data5 directories
    for data_dir in sorted(lndb_root.glob("data*")):
        if not data_dir.is_dir():
            continue
        
        print(f"Processing {data_dir.name}...")
        
        # Find all .mhd files (LNDb uses MHD format)
        mhd_files = sorted(data_dir.glob("**/*.mhd"))
        
        for mhd_path in mhd_files:
            scan_id = mhd_path.stem
            print(f"  Processing {scan_id}...")
            
            # Get annotations for this scan
            scan_annotations = annotations_dict.get(scan_id, [])
            
            scan_data = process_single_scan(
                scan_id,
                mhd_path,
                scan_annotations,
                lndb_root,
                output_dir
            )
            
            dataset.append(scan_data)
    
    # Save dataset JSON
    output_json_path = output_dir / args.output_json
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    total_regions = sum(len(scan['regions']) for scan in dataset)
    scans_with_regions = sum(1 for scan in dataset if len(scan['regions']) > 0)
    
    print(f"\n{'='*60}")
    print(f"Dataset preparation complete!")
    print(f"{'='*60}")
    print(f"Total scans processed: {len(dataset)}")
    print(f"Scans with regions: {scans_with_regions}")
    print(f"Total regions: {total_regions}")
    print(f"Average regions per scan: {total_regions/len(dataset):.2f}")
    print(f"Output JSON: {output_json_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
