
import unittest
import numpy as np
from detection.train_3dunet.segmentation import generate_lung_mask, compute_lung_bbox

class TestLungCropping(unittest.TestCase):
    def setUp(self):
        # Create a synthetic volume (D, H, W)
        self.D, self.H, self.W = 10, 100, 100
        self.volume = np.ones((self.D, self.H, self.W), dtype=np.float32) # Background = 1 (Bone/Tissue)
        
        # Create "Lungs" (Low intensity = 0.2)
        # Left Lung: (20:80, 20:45)
        self.volume[:, 20:80, 20:45] = 0.2
        # Right Lung: (20:80, 55:80) 
        self.volume[:, 20:80, 55:80] = 0.2
        
        # Create "Air" background (Low intensity = 0.0) -> Should be ignored if we handle it right, 
        # but generate_lung_mask simple version might pick it up. 
        # In our logic, we filter by 'largest 2 regions' excluding corners or just largest 2.
        # Let's clean the corners to 1 (Tissue) to simulate a cropped body scan first, 
        # or set to 0 to test robustness.
        # Let's set corners to 1 to simulate body crop (common in LNDb).
        
    def test_segmentation_mask(self):
        mask = generate_lung_mask(self.volume, threshold=0.45)
        
        # Check specific points
        # Lung center should be 1
        self.assertEqual(mask[5, 50, 30], 1, "Left lung center should be 1") 
        self.assertEqual(mask[5, 50, 65], 1, "Right lung center should be 1")
        
        # Background should be 0
        self.assertEqual(mask[5, 10, 10], 0, "Top-left background should be 0")
        self.assertEqual(mask[5, 90, 50], 0, "Bottom area should be 0")
        
        # Mediastinum (50, 50) might be 1 due to dilation (closing gap) so we skip strict check there
        # but 10, 10 is definitely out.
        
    def test_bbox_computation(self):
        mask = generate_lung_mask(self.volume, threshold=0.45)
        min_x, min_y, max_x, max_y = compute_lung_bbox(mask, margin=0)
        
        # Expected:
        # y: 20 to 80
        # x: 20 to 80 (covering both lungs 20-45 and 55-80)
        
        print(f"Computed BBox: x=[{min_x}, {max_x}], y=[{min_y}, {max_y}]")
        
        # Dilation of 5mm (approx 5 pixels) expands the mask by ~5 pixels on each side.
        # So 20 -> ~15, 80 -> ~85.
        
        # Allow wider tolerance due to morphology operations
        self.assertTrue(10 <= min_y <= 25)
        self.assertTrue(75 <= max_y <= 90)
        self.assertTrue(10 <= min_x <= 25)
        self.assertTrue(75 <= max_x <= 90)

    def test_cropping(self):
        mask = generate_lung_mask(self.volume, threshold=0.45) # dilate=5
        min_x, min_y, max_x, max_y = compute_lung_bbox(mask, margin=5) # margin=5
        
        cropped_vol = self.volume[:, min_y:max_y, min_x:max_x]
        
        print(f"Original shape: {self.volume.shape}")
        print(f"Cropped shape: {cropped_vol.shape}")
        
        # Original width 100. Lung span 20-80 = 60. 
        # Dilate 5 -> 15-85 (span 70)
        # BBox Margin 5 -> 10-90 (span 80)
        # So we expect around 80.
        
        self.assertTrue(60 <= cropped_vol.shape[1] <= 90)
        self.assertTrue(60 <= cropped_vol.shape[2] <= 90)

if __name__ == '__main__':
    unittest.main()
