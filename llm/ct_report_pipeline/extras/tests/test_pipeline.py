"""
Unit tests for the CT report generation pipeline.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFeatureExtractor(unittest.TestCase):
    """Test feature extraction module."""
    
    def setUp(self):
        from features import FeatureExtractor
        self.extractor = FeatureExtractor(device="cpu")
    
    def test_texture_feature_shape(self):
        """Test that texture features have correct shape."""
        ct_volume = np.random.randn(64, 128, 128)
        mask = np.zeros_like(ct_volume)
        mask[20:40, 50:70, 50:70] = 1
        
        texture_feat = self.extractor.extract_texture_feature(ct_volume, mask)
        
        self.assertEqual(texture_feat.shape[0], self.extractor.llm_dim)
    
    def test_global_feature_shape(self):
        """Test that global features have correct shape."""
        ct_volume = np.random.randn(64, 128, 128)
        
        global_feat = self.extractor.extract_global_feature(ct_volume)
        
        self.assertEqual(global_feat.shape[0], self.extractor.llm_dim)
    
    def test_multiple_regions(self):
        """Test extraction for multiple regions."""
        ct_volume = np.random.randn(64, 128, 128)
        masks = [
            np.zeros_like(ct_volume),
            np.zeros_like(ct_volume)
        ]
        masks[0][10:20, 30:40, 30:40] = 1
        masks[1][40:50, 60:70, 60:70] = 1
        
        features = self.extractor.extract_all_features(ct_volume, masks)
        
        self.assertIn('global', features)
        self.assertIn('local', features)
        self.assertEqual(len(features['local']), 2)


class TestNLGMetrics(unittest.TestCase):
    """Test NLG metrics."""
    
    def setUp(self):
        from evaluation import NLGMetrics
        self.metrics = NLGMetrics()
    
    def test_bleu_identical(self):
        """Test BLEU score for identical texts."""
        refs = ["This is a test sentence"]
        hyps = ["This is a test sentence"]
        
        scores = self.metrics.compute_bleu(refs, hyps)
        
        # Identical texts should have high BLEU scores
        self.assertGreater(scores['BLEU-1'], 0.9)
    
    def test_rouge_l(self):
        """Test ROUGE-L computation."""
        refs = ["The patient has a small nodule in the right lung"]
        hyps = ["The patient has a nodule in the right lung"]
        
        score = self.metrics.compute_rouge_l(refs, hyps)
        
        # Should have reasonable overlap
        self.assertGreater(score, 0.5)


class TestClinicalEfficacyMetrics(unittest.TestCase):
    """Test clinical efficacy metrics."""
    
    def setUp(self):
        from evaluation import ClinicalEfficacyMetrics
        self.metrics = ClinicalEfficacyMetrics()
    
    def test_label_extraction(self):
        """Test label extraction from text."""
        text = "Small solid nodule with calcification in the right upper lobe"
        
        labels = self.metrics.extract_labels(text)
        
        self.assertTrue(labels['nodule_present'])
        self.assertTrue(labels['size_small'])
        self.assertTrue(labels['solid'])
        self.assertTrue(labels['calcification'])
    
    def test_metrics_computation(self):
        """Test precision/recall/F1 computation."""
        refs = [
            "Solid nodule with calcification",
            "Subsolid ground-glass opacity"
        ]
        hyps = [
            "Solid nodule present",
            "Ground-glass opacity noted"
        ]
        
        results = self.metrics.compute_metrics(refs, hyps)
        
        self.assertIn('nodule_present', results)
        self.assertIn('macro_avg', results)
        self.assertIn('precision', results['macro_avg'])


if __name__ == "__main__":
    unittest.main()
