"""
Unit tests for deterministic Lung-RADS v2022 classification.
"""

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lung_rads import assess_exam, assess_nodule, build_structured_report_input


class TestLungRadsAssessment(unittest.TestCase):
    def test_no_nodules_is_category_1(self):
        result = assess_exam([])

        self.assertEqual(result["exam"]["category"], "1")
        self.assertEqual(result["nodules"], [])

    def test_solid_baseline_thresholds(self):
        self.assertEqual(
            assess_nodule({"nodule_id": 1, "attenuation_type": "solid", "longest_axis_mm": 5.9})["category"],
            "2",
        )
        self.assertEqual(
            assess_nodule({"nodule_id": 1, "attenuation_type": "solid", "longest_axis_mm": 6.0})["category"],
            "3",
        )
        self.assertEqual(
            assess_nodule({"nodule_id": 1, "attenuation_type": "solid", "longest_axis_mm": 8.0})["category"],
            "4A",
        )
        self.assertEqual(
            assess_nodule({"nodule_id": 1, "attenuation_type": "solid", "longest_axis_mm": 15.0})["category"],
            "4B",
        )

    def test_ground_glass_threshold(self):
        self.assertEqual(
            assess_nodule({"nodule_id": 1, "attenuation_type": "ground-glass", "longest_axis_mm": 29.9})["category"],
            "2",
        )
        self.assertEqual(
            assess_nodule({"nodule_id": 1, "attenuation_type": "ground-glass", "longest_axis_mm": 30.0})["category"],
            "3",
        )

    def test_part_solid_uses_solid_component(self):
        result = assess_nodule(
            {
                "nodule_id": 2,
                "attenuation_type": "part-solid",
                "longest_axis_mm": 12.0,
                "solid_component_mm": 7.0,
            }
        )

        self.assertEqual(result["category"], "4A")
        self.assertEqual(result["solid_component_source"], "solid_component_mm")

    def test_exam_uses_most_suspicious_nodule(self):
        result = assess_exam(
            [
                {"nodule_id": 1, "attenuation_type": "solid", "longest_axis_mm": 6.5},
                {"nodule_id": 2, "attenuation_type": "solid", "longest_axis_mm": 16.0},
            ]
        )

        self.assertEqual(result["exam"]["category"], "4B")
        self.assertEqual(result["exam"]["most_suspicious_nodule_id"], 2)

    def test_structured_input_schema(self):
        payload = build_structured_report_input(
            [{"nodule_id": 1, "attenuation_type": "solid", "longest_axis_mm": 7.0}],
            report_id="AUTO_TEST",
            scan_date="2026-05-13",
        )

        self.assertEqual(payload["schema_version"], "ct-report-structured-input-v1")
        self.assertEqual(payload["lung_rads"]["exam"]["category"], "3")
        self.assertEqual(payload["report_id"], "AUTO_TEST")


if __name__ == "__main__":
    unittest.main()
