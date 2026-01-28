
import logging
from pathlib import Path
from typing import List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

def get_split_files(
    npz_dir: Path, 
    split: str, 
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15), 
    seed: int = 42
) -> List[Path]:
    """
    Get file list for a specific split using dynamic splitting logic.
    
    Args:
        npz_dir: Directory containing NPZ files (or subdirs)
        split: 'train', 'val', or 'test'
        ratios: (train_ratio, val_ratio, test_ratio)
        seed: Random seed for deterministic splitting
        
    Returns:
        List of file paths for the requested split
    """
    if not npz_dir.exists():
        logger.warning(f"⚠️ Directory not found: {npz_dir}")
        return []

    # 1. Try to load from "flat" structure (all files in npz_dir)
    #    or "nested" structure (files in npz_dir/split)
    
    # Priority 1: Check if specific split folder exists (legacy support or manual split)
    split_dir = npz_dir / split
    if split_dir.exists() and any(split_dir.glob("*.npz")):
        logger.info(f"📂 Found legacy split folder: {split_dir}")
        return sorted(list(split_dir.glob("*.npz")))
        
    # Priority 2: Load all files from root and split dynamically
    files = sorted(list(npz_dir.glob("*.npz")))
    if not files:
        # Check one level deeper just in case user pointed to parent
        files = sorted(list(npz_dir.glob("*/*.npz")))
        
    if not files:
        logger.warning(f"⚠️ No NPZ files found in {npz_dir}")
        return []
        
    # Group by patient ID to ensure no leakage
    # Assumption: filename format starts with "LNDb-0001" or "lung_001"
    # We group by the first part of the filename
    patient_files = {}
    for f in files:
        # Heuristic: Patient ID is usually the part before the first underscore or second hyphen
        # But LNDb is LNDb-0001_lesion01.npz -> LNDb-0001
        # MSD is lung_001_lesion01.npz -> lung_001
        
        stem = f.stem
        if "LNDb" in stem:
            # LNDb-0001_lesion...
            pid = stem.split("_")[0] 
        elif "lung_" in stem:
             # lung_001_lesion...
             parts = stem.split("_")
             if len(parts) >= 2:
                 pid = f"{parts[0]}_{parts[1]}"
             else:
                 pid = stem
        else:
            # Fallback: assume whole filename or part before last underscore
            pid = stem.rsplit("_", 1)[0]
            
        if pid not in patient_files:
            patient_files[pid] = []
        patient_files[pid].append(f)
        
    patient_ids = sorted(list(patient_files.keys()))
    
    # Shuffle patients
    rng = np.random.RandomState(seed)
    rng.shuffle(patient_ids)
    
    # Calculate split indices
    n = len(patient_ids)
    train_end = int(n * ratios[0])
    val_end = train_end + int(n * ratios[1])
    
    if split == 'train':
        selected_pids = patient_ids[:train_end]
    elif split == 'val':
        selected_pids = patient_ids[train_end:val_end]
    elif split == 'test':
        selected_pids = patient_ids[val_end:]
    else:
        # 'all' or unknown
        return files
        
    # Collect all files for selected patients
    selected_files = []
    for pid in selected_pids:
        selected_files.extend(patient_files[pid])
        
    logger.info(f"🔄 Dynamic Split ({split}): {len(selected_pids)} patients, {len(selected_files)} samples")
    return sorted(selected_files)
