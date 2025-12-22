import numpy as np
import os
from pathlib import Path

cache_dir = Path(r'c:\GitHub\chest-ct-report-generator\segmentation\cache\lndb_slices')
patients = sorted(os.listdir(cache_dir))[:5]

print("=" * 80)
print("檢查 slice npz：mask (binary)、lung_mask (binary)、對齊情況")
print("=" * 80)

for p in patients:
    patient_dir = cache_dir / p
    if not patient_dir.is_dir():
        continue
    
    slices = [f for f in os.listdir(patient_dir) if f.startswith('slice_') and f.endswith('.npz')]
    
    # 找有結節的切片
    positive_slices = []
    for s in slices[:100]:
        data = np.load(patient_dir / s, allow_pickle=True)
        mask = data['mask']
        if mask.max() > 0:
            positive_slices.append((s, data))
        if len(positive_slices) >= 3:
            break
    
    print(f"\n{'='*40}")
    print(f"Patient: {p}")
    print(f"{'='*40}")
    
    for s, data in positive_slices:
        image = data['image']
        mask = data['mask']
        lung_mask = data['lung_mask']
        
        # Shape 檢查
        print(f"\n  {s}:")
        print(f"    Shapes: image={image.shape}, mask={mask.shape}, lung_mask={lung_mask.shape}")
        shapes_match = image.shape == mask.shape and mask.shape == lung_mask.shape[:2] if lung_mask.ndim > 2 else mask.shape == lung_mask.shape
        print(f"    Shapes match: {shapes_match}")
        
        # Mask 檢查 (binary)
        mask_unique = np.unique(mask)
        print(f"    mask: min={mask.min():.4f}, max={mask.max():.4f}, unique={mask_unique[:10]}")
        is_binary_mask = len(mask_unique) <= 2 and mask.max() <= 1
        print(f"    mask is binary: {is_binary_mask}")
        
        # Lung mask 檢查 (binary)
        lung_unique = np.unique(lung_mask)
        print(f"    lung_mask: unique={lung_unique}")
        is_binary_lung = set(lung_unique).issubset({0, 1, False, True})
        print(f"    lung_mask is binary: {is_binary_lung}")
        
        # 面積統計
        lung_area = int((lung_mask > 0).sum())
        mask_area = int((mask > 0).sum())
        mask_in_lung = int(((mask > 0) & (lung_mask > 0)).sum())
        
        print(f"    lung_area: {lung_area} px")
        print(f"    mask_area (nodule): {mask_area} px")
        print(f"    mask_in_lung: {mask_in_lung} px")
        
        if mask_area > 0:
            ratio = mask_in_lung / mask_area * 100
            print(f"    nodule in lung ratio: {ratio:.1f}%")
        else:
            print(f"    nodule in lung ratio: N/A (no nodule)")

print("\n" + "=" * 80)
print("檢查完成")
