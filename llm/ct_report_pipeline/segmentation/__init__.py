"""
Segmentation module for CT report generation pipeline.

Provides wrappers for:
- MedSAM2: Prompt-based segmentation (requires bounding box or points)
"""

from .medsam2_infer import MedSAM2Segmenter

__all__ = ['MedSAM2Segmenter']
