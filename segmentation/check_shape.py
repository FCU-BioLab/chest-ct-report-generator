
import numpy as np
import sys

try:
    data = np.load(r"c:\GitHub\chest-ct-report-generator\segmentation\cache\lndb_slices\LNDb-0001\slice_0225.npz")
    print(f"Shape: {data['image'].shape}")
    print(f"Mask Shape: {data['mask'].shape}")
except Exception as e:
    print(f"Error: {e}")
