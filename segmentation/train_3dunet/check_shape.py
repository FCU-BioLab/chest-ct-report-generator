import numpy as np
import sys

path = r"cache\msd_volume_npz\train\lung_001_lesion01.npz"
try:
    data = np.load(path, allow_pickle=True)
    print(f"Frames shape: {data['frames'].shape}")
    print(f"Masks shape: {data['masks'].shape}")
except Exception as e:
    print(f"Error: {e}")
