#!/usr/bin/env python3
"""
Coordinate-based lung lobe estimation shared across detection pipelines.
"""

from typing import Dict, Optional, Tuple

import numpy as np


class LungLocationEstimator:
    """Estimate approximate anatomical lung lobe from normalized coordinates."""

    LOBES = {
        "RUL": "Right Upper Lobe",
        "RML": "Right Middle Lobe",
        "RLL": "Right Lower Lobe",
        "LUL": "Left Upper Lobe",
        "Lingula": "Left Lingula",
        "LLL": "Left Lower Lobe",
    }

    def __init__(self, total_slices: int = 100):
        self.total_slices = total_slices

    def estimate_location(
        self,
        relative_x: float,
        relative_y: float,
        slice_ratio: float,
        slice_index: Optional[int] = None,
        slice_location_mm: Optional[float] = None,
    ) -> Dict[str, object]:
        result = {
            "lobe": "unknown",
            "lobe_full": "Unknown",
            "side": "unknown",
            "vertical_zone": "unknown",
            "confidence": 0.0,
            "description": "",
            "coordinates": {
                "relative_x": relative_x,
                "relative_y": relative_y,
                "slice_ratio": slice_ratio,
            },
        }

        if relative_x < 0.42:
            side = "left"
            result["side"] = "left"
        elif relative_x > 0.58:
            side = "right"
            result["side"] = "right"
        else:
            side = "left" if relative_x < 0.5 else "right"
            result["side"] = side
            result["confidence"] = max(0.3, result["confidence"])

        if slice_ratio < 0.35:
            vertical_zone = "upper"
        elif slice_ratio < 0.65:
            vertical_zone = "middle"
        else:
            vertical_zone = "lower"
        result["vertical_zone"] = vertical_zone

        lobe, confidence = self._determine_lobe(side, vertical_zone, relative_y, slice_ratio)
        result["lobe"] = lobe
        result["lobe_full"] = self.LOBES.get(lobe, lobe)
        result["confidence"] = confidence
        result["description"] = self._generate_description(result)
        return result

    def _determine_lobe(
        self,
        side: str,
        vertical_zone: str,
        relative_y: float,
        slice_ratio: float,
    ) -> Tuple[str, float]:
        if side == "right":
            if vertical_zone == "upper":
                return "RUL", 0.85
            if vertical_zone == "lower":
                return "RLL", 0.85
            if relative_y < 0.5:
                return "RML", 0.75
            return ("RUL", 0.6) if slice_ratio < 0.5 else ("RLL", 0.6)

        if vertical_zone == "upper":
            if slice_ratio > 0.25 and relative_y < 0.45:
                return "Lingula", 0.7
            return "LUL", 0.85
        if vertical_zone == "lower":
            return "LLL", 0.85
        if relative_y < 0.45:
            return "Lingula", 0.7
        return ("LUL", 0.6) if slice_ratio < 0.5 else ("LLL", 0.6)

    def _generate_description(self, result: Dict[str, object]) -> str:
        side_text = "right" if result["side"] == "right" else "left"
        confidence = float(result["confidence"])
        confidence_suffix = ""
        if confidence < 0.6:
            confidence_suffix = " (low confidence)"
        elif confidence < 0.75:
            confidence_suffix = " (moderate confidence)"
        return f"{side_text} {result['lobe']}{confidence_suffix}"

    def estimate_from_features(self, lesion_features: Dict[str, float], total_slices: int = None) -> Dict[str, object]:
        if total_slices:
            self.total_slices = total_slices

        relative_x = lesion_features.get("relative_position_x", 0.5)
        relative_y = lesion_features.get("relative_position_y", 0.5)
        slice_location = lesion_features.get("slice_location", 0)
        slice_index = lesion_features.get("slice_index", 0)

        if slice_index and self.total_slices:
            slice_ratio = slice_index / self.total_slices
        else:
            slice_ratio = (slice_location + 400) / 500 if slice_location else 0.5
            slice_ratio = np.clip(slice_ratio, 0, 1)

        return self.estimate_location(
            relative_x=relative_x,
            relative_y=relative_y,
            slice_ratio=slice_ratio,
            slice_index=slice_index,
            slice_location_mm=slice_location,
        )


def add_location_to_features(features_dict: Dict[str, object], total_slices: int = None) -> Dict[str, object]:
    estimator = LungLocationEstimator(total_slices or features_dict.get("metadata", {}).get("total_slices", 100))
    features_dict["anatomical_location"] = {
        "lesion_locations": [],
        "estimation_method": "coordinate_based",
        "note": "Approximate lobe estimation without explicit lobe segmentation.",
    }
    if "relative_position_x" in features_dict:
        features_dict["anatomical_location"]["lesion_locations"].append(
            estimator.estimate_from_features(features_dict, total_slices)
        )
    return features_dict


def get_location_for_report(relative_x: float, relative_y: float, slice_ratio: float, total_slices: int = 100) -> str:
    estimator = LungLocationEstimator(total_slices)
    return estimator.estimate_location(relative_x, relative_y, slice_ratio)["lobe"]
