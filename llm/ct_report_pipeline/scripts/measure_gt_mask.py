"""
Measure ground truth mask size from LNDb dataset.
This script calculates the lesion size from the ground truth mask
to verify measurement accuracy.
"""

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from scipy.spatial.distance import cdist
from pathlib import Path
import argparse


def measure_mask(mask_path: str, ct_path: str = None):
    """
    Measure the size of a lesion from a mask file.
    
    Args:
        mask_path: Path to the mask MHD file
        ct_path: Optional path to CT for spacing (if mask doesn't have it)
    """
    print(f"\n{'='*60}")
    print(f"Measuring: {Path(mask_path).name}")
    print(f"{'='*60}")
    
    # Load mask
    sitk_mask = sitk.ReadImage(str(mask_path))
    mask = sitk.GetArrayFromImage(sitk_mask)
    
    # Get spacing (x, y, z) from SimpleITK
    spacing_xyz = np.array(sitk_mask.GetSpacing())
    spacing_x, spacing_y, spacing_z = spacing_xyz
    
    print(f"\nMask shape (D, H, W): {mask.shape}")
    print(f"Spacing (x, y, z): {spacing_xyz} mm")
    print(f"Unique values in mask: {np.unique(mask)}")
    
    # Find all unique lesion labels (exclude 0 which is background)
    unique_labels = np.unique(mask)
    lesion_labels = unique_labels[unique_labels > 0]
    
    print(f"\nNumber of lesions: {len(lesion_labels)}")
    
    for label in lesion_labels:
        print(f"\n--- Lesion {label} ---")
        lesion_mask = mask == label
        
        voxel_count = np.sum(lesion_mask)
        if voxel_count == 0:
            print("  Empty mask, skipping")
            continue
            
        # Volume
        voxel_volume = spacing_x * spacing_y * spacing_z
        volume_mm3 = voxel_count * voxel_volume
        volume_cm3 = volume_mm3 / 1000
        
        # Equivalent Sphere Diameter
        esd = 2 * (3 * volume_mm3 / (4 * np.pi)) ** (1/3)
        
        # Bounding box
        coords = np.where(lesion_mask)
        z_min, z_max = coords[0].min(), coords[0].max()
        y_min, y_max = coords[1].min(), coords[1].max()
        x_min, x_max = coords[2].min(), coords[2].max()
        
        n_z = z_max - z_min + 1
        n_y = y_max - y_min + 1
        n_x = x_max - x_min + 1
        
        bbox_x = n_x * spacing_x
        bbox_y = n_y * spacing_y
        bbox_z = n_z * spacing_z
        
        print(f"  Voxel count: {voxel_count}")
        print(f"  Volume: {volume_mm3:.2f} mm³ ({volume_cm3:.4f} cm³)")
        print(f"  ESD: {esd:.2f} mm")
        print(f"  Bounding box voxels: x={n_x}, y={n_y}, z={n_z}")
        print(f"  Bounding box size: x={bbox_x:.2f}mm, y={bbox_y:.2f}mm, z={bbox_z:.2f}mm")
        
        # Boundary-based measurement
        from scipy.ndimage import binary_erosion
        eroded = binary_erosion(lesion_mask)
        boundary = lesion_mask & ~eroded
        
        boundary_coords = np.column_stack(np.where(boundary))
        print(f"  Boundary voxels: {len(boundary_coords)}")
        
        if len(boundary_coords) > 0:
            # Convert to mm
            boundary_mm = np.zeros_like(boundary_coords, dtype=np.float64)
            boundary_mm[:, 0] = boundary_coords[:, 0] * spacing_z
            boundary_mm[:, 1] = boundary_coords[:, 1] * spacing_y
            boundary_mm[:, 2] = boundary_coords[:, 2] * spacing_x
            
            # Sample if too many points
            if len(boundary_mm) > 5000:
                indices = np.random.choice(len(boundary_mm), 5000, replace=False)
                sample_mm = boundary_mm[indices]
            else:
                sample_mm = boundary_mm
            
            # Longest axis
            distances = cdist(sample_mm, sample_mm)
            longest_axis = np.max(distances)
            
            print(f"  Longest axis (Feret): {longest_axis:.2f} mm")
            
            # PCA for principal axes
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=min(3, len(boundary_mm)))
                pca.fit(boundary_mm)
                projected = pca.transform(boundary_mm)
                
                axis_extents = []
                for i in range(projected.shape[1]):
                    extent = projected[:, i].max() - projected[:, i].min()
                    axis_extents.append(extent)
                
                print(f"  PCA axes: {', '.join([f'{e:.2f}mm' for e in sorted(axis_extents, reverse=True)])}")
                mean_diameter = np.mean(axis_extents)
                print(f"  Mean diameter: {mean_diameter:.2f} mm")
            except Exception as e:
                print(f"  PCA failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Measure ground truth mask size")
    parser.add_argument("--mask", type=str, required=True, help="Path to mask MHD file")
    parser.add_argument("--ct", type=str, help="Optional path to CT for spacing")
    args = parser.parse_args()
    
    measure_mask(args.mask, args.ct)


if __name__ == "__main__":
    main()
