"""
Deterministic Lung-RADS v2022 assessment helpers.

This module classifies only the findings represented by the current pipeline
feature schema. Findings that need unavailable upstream signals (for example
airway nodules, atypical cysts, infection, or longitudinal comparison) are
reported in the limitations instead of being inferred silently.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Mapping, Optional


LUNG_RADS_VERSION = "v2022"
STRUCTURED_INPUT_SCHEMA_VERSION = "ct-report-structured-input-v1"

CATEGORY_RANK = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4A": 4,
    "4B": 5,
    "4X": 6,
}

CATEGORY_INFO: Dict[str, Dict[str, str]] = {
    "0": {
        "descriptor": "Incomplete",
        "management": "Additional lung cancer screening CT imaging or comparison is needed.",
    },
    "1": {
        "descriptor": "Negative",
        "management": "Continue annual screening LDCT in 12 months.",
    },
    "2": {
        "descriptor": "Benign appearance or behavior",
        "management": "Continue annual screening LDCT in 12 months.",
    },
    "3": {
        "descriptor": "Probably benign",
        "management": "6-month LDCT follow-up.",
    },
    "4A": {
        "descriptor": "Suspicious",
        "management": "3-month LDCT; PET/CT may be considered when there is a solid nodule or solid component >=8 mm.",
    },
    "4B": {
        "descriptor": "Very suspicious",
        "management": "Diagnostic chest CT with or without contrast, PET/CT when appropriate, tissue sampling, and/or referral for clinical evaluation.",
    },
    "4X": {
        "descriptor": "Very suspicious with additional suspicious features",
        "management": "Diagnostic evaluation as appropriate for a very suspicious Lung-RADS finding.",
    },
}


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _category(category: str, reason: str, limitations: Optional[List[str]] = None, **extra: Any) -> Dict[str, Any]:
    info = CATEGORY_INFO[category]
    result: Dict[str, Any] = {
        "category": category,
        "rank": CATEGORY_RANK[category],
        "descriptor": info["descriptor"],
        "management": info["management"],
        "reason": reason,
        "limitations": limitations or [],
    }
    result.update(extra)
    return result


def _normalize_attenuation_type(features: Mapping[str, Any]) -> str:
    raw = str(features.get("attenuation_type") or features.get("nodule_type") or "").strip().lower()
    aliases = {
        "ggo": "ground-glass",
        "ground glass": "ground-glass",
        "ground_glass": "ground-glass",
        "non-solid": "ground-glass",
        "nonsolid": "ground-glass",
        "subsolid": "part-solid",
        "part solid": "part-solid",
        "part_solid": "part-solid",
        "solid": "solid",
        "calcified": "calcified",
        "fat": "fat-containing",
        "fat-containing": "fat-containing",
    }
    return aliases.get(raw, raw or "indeterminate")


def _nodule_size_mm(features: Mapping[str, Any]) -> Optional[float]:
    for key in (
        "mean_diameter_mm",
        "average_diameter_mm",
        "longest_axis_mm",
        "equivalent_diameter_mm",
        "approx_diameter_mm",
    ):
        value = _to_float(features.get(key))
        if value is not None:
            return value
    return None


def _solid_component_mm(features: Mapping[str, Any]) -> Dict[str, Any]:
    for key in (
        "solid_component_mm",
        "solid_component_diameter_mm",
        "solid_component_mean_diameter_mm",
    ):
        value = _to_float(features.get(key))
        if value is not None:
            return {"value": value, "source": key}

    volume_mm3 = _to_float(features.get("volume_mm3"))
    soft_fraction = _to_float(features.get("soft_tissue_fraction"))
    if volume_mm3 is None or soft_fraction is None:
        return {"value": None, "source": "unavailable"}

    soft_fraction = min(1.0, max(0.0, soft_fraction))
    if soft_fraction <= 0:
        return {"value": 0.0, "source": "estimated_from_soft_tissue_fraction"}

    solid_volume = volume_mm3 * soft_fraction
    diameter = 2.0 * ((3.0 * solid_volume) / (4.0 * 3.141592653589793)) ** (1.0 / 3.0)
    return {"value": float(diameter), "source": "estimated_from_soft_tissue_fraction"}


def _is_new(features: Mapping[str, Any]) -> bool:
    return _to_bool(features.get("is_new")) or str(features.get("growth_status", "")).lower() == "new"


def _is_growing(features: Mapping[str, Any]) -> bool:
    growth_mm = _to_float(features.get("growth_mm_12mo"))
    if growth_mm is not None and growth_mm > 1.5:
        return True
    return str(features.get("growth_status", "")).strip().lower() in {"growing", "growth"}


def _is_stable_or_decreased(features: Mapping[str, Any]) -> bool:
    return str(features.get("growth_status", "")).strip().lower() in {
        "stable",
        "decreased",
        "decreasing",
        "slow_growing",
        "slow-growing",
    }


def _has_suspicious_features(features: Mapping[str, Any]) -> bool:
    if _to_bool(features.get("suspicious_features")):
        return True
    values = features.get("suspicious_feature_list")
    return isinstance(values, list) and bool(values)


def assess_nodule(features: Mapping[str, Any]) -> Dict[str, Any]:
    """Assess one nodule using Lung-RADS v2022 size/composition criteria."""
    nodule = deepcopy(dict(features))
    nodule_id = nodule.get("nodule_id", nodule.get("id"))
    size_mm = _nodule_size_mm(nodule)
    attenuation_type = _normalize_attenuation_type(nodule)
    limitations: List[str] = []

    if size_mm is None:
        return _category(
            "0",
            "Nodule size is unavailable; Lung-RADS size criteria cannot be applied.",
            ["missing_nodule_size"],
            nodule_id=nodule_id,
            attenuation_type=attenuation_type,
            size_mm=None,
        )

    category: Dict[str, Any]

    if attenuation_type in {"calcified", "fat-containing"} or _to_bool(nodule.get("benign_features")):
        category = _category(
            "1",
            "Nodule has benign features (calcification or fat-containing feature).",
            nodule_id=nodule_id,
            attenuation_type=attenuation_type,
            size_mm=size_mm,
        )
    elif attenuation_type == "solid":
        category = _assess_solid_nodule(nodule, size_mm)
    elif attenuation_type == "ground-glass":
        category = _assess_ground_glass_nodule(nodule, size_mm)
    elif attenuation_type == "part-solid":
        category = _assess_part_solid_nodule(nodule, size_mm)
    else:
        category = _category(
            "0",
            "Nodule attenuation type is unavailable or low confidence; composition-specific Lung-RADS criteria cannot be applied.",
            ["missing_or_indeterminate_attenuation_type"],
            nodule_id=nodule_id,
            attenuation_type=attenuation_type,
            size_mm=size_mm,
        )

    category.setdefault("nodule_id", nodule_id)
    category.setdefault("attenuation_type", attenuation_type)
    category.setdefault("size_mm", size_mm)

    if _has_suspicious_features(nodule) and category["rank"] >= CATEGORY_RANK["3"]:
        prior = dict(category)
        category = _category(
            "4X",
            "Category 3 or 4 nodule has additional suspicious features.",
            prior.get("limitations", []),
            nodule_id=nodule_id,
            attenuation_type=attenuation_type,
            size_mm=size_mm,
            base_category=prior["category"],
            base_reason=prior["reason"],
        )

    category["source_feature_keys"] = sorted(str(k) for k in nodule.keys())
    category["limitations"] = [*category.get("limitations", []), *limitations]
    return category


def _assess_solid_nodule(features: Mapping[str, Any], size_mm: float) -> Dict[str, Any]:
    is_new = _is_new(features)
    is_growing = _is_growing(features)
    context = "new" if is_new else "growing" if is_growing else "baseline"

    if is_new:
        if size_mm < 4:
            category = "2"
            reason = "New solid nodule <4 mm."
        elif size_mm < 6:
            category = "3"
            reason = "New solid nodule 4 to <6 mm."
        elif size_mm < 8:
            category = "4A"
            reason = "New solid nodule 6 to <8 mm."
        else:
            category = "4B"
            reason = "New solid nodule >=8 mm."
    elif is_growing:
        if size_mm < 8:
            category = "4A"
            reason = "Growing solid nodule <8 mm."
        else:
            category = "4B"
            reason = "Growing solid nodule >=8 mm."
    elif size_mm < 6:
        category = "2"
        reason = "Baseline solid nodule <6 mm."
    elif size_mm < 8:
        category = "3"
        reason = "Baseline solid nodule >=6 to <8 mm."
    elif size_mm < 15:
        category = "4A"
        reason = "Baseline solid nodule >=8 to <15 mm."
    else:
        category = "4B"
        reason = "Baseline solid nodule >=15 mm."

    return _category(category, reason, nodule_context=context)


def _assess_ground_glass_nodule(features: Mapping[str, Any], size_mm: float) -> Dict[str, Any]:
    if size_mm < 30:
        return _category("2", "Non-solid ground-glass nodule <30 mm.")
    if _is_stable_or_decreased(features):
        return _category("2", "Non-solid ground-glass nodule >=30 mm that is stable or slowly growing.")
    return _category("3", "Non-solid ground-glass nodule >=30 mm at baseline or new.")


def _assess_part_solid_nodule(features: Mapping[str, Any], size_mm: float) -> Dict[str, Any]:
    solid_component = _solid_component_mm(features)
    solid_mm = solid_component["value"]
    limitations: List[str] = []
    if solid_component["source"] == "estimated_from_soft_tissue_fraction":
        limitations.append("solid_component_estimated_from_hu_fraction")

    if size_mm < 6:
        return _category(
            "2",
            "Part-solid nodule <6 mm total mean diameter.",
            limitations,
            solid_component_mm=solid_mm,
            solid_component_source=solid_component["source"],
        )

    if solid_mm is None:
        return _category(
            "0",
            "Part-solid nodule >=6 mm requires solid-component diameter, but it is unavailable.",
            ["missing_solid_component_diameter"],
            solid_component_mm=None,
            solid_component_source=solid_component["source"],
        )

    if _is_new(features) or _is_growing(features):
        if solid_mm < 4:
            return _category(
                "4A",
                "New or growing part-solid nodule with solid component <4 mm.",
                limitations,
                solid_component_mm=solid_mm,
                solid_component_source=solid_component["source"],
            )
        return _category(
            "4B",
            "New or growing part-solid nodule with solid component >=4 mm.",
            limitations,
            solid_component_mm=solid_mm,
            solid_component_source=solid_component["source"],
        )

    if solid_mm < 6:
        category = "3"
        reason = "Baseline part-solid nodule >=6 mm with solid component <6 mm."
    elif solid_mm < 8:
        category = "4A"
        reason = "Baseline part-solid nodule >=6 mm with solid component >=6 to <8 mm."
    else:
        category = "4B"
        reason = "Baseline part-solid nodule with solid component >=8 mm."

    return _category(
        category,
        reason,
        limitations,
        solid_component_mm=solid_mm,
        solid_component_source=solid_component["source"],
    )


def assess_exam(
    lesion_features: Iterable[Mapping[str, Any]],
    exam_context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Assess all nodules and choose the most suspicious exam-level category."""
    features = [dict(item) for item in lesion_features]
    context = dict(exam_context or {})
    limitations = [
        "single_exam_only_no_prior_comparison",
        "airway_nodule_cyst_infection_and_s_modifier_not_evaluated",
    ]

    if _to_bool(context.get("incomplete_exam")):
        return {
            "lung_rads_version": LUNG_RADS_VERSION,
            "exam": _category("0", "Exam is marked incomplete.", ["incomplete_exam"]),
            "nodules": [],
            "limitations": limitations,
        }

    if not features:
        return {
            "lung_rads_version": LUNG_RADS_VERSION,
            "exam": _category("1", "No measurable pulmonary nodules were provided."),
            "nodules": [],
            "limitations": limitations,
        }

    nodules = [assess_nodule(item) for item in features]
    most_suspicious = max(nodules, key=lambda item: item["rank"])
    exam = _category(
        most_suspicious["category"],
        f"Exam category is determined by the most suspicious nodule: {most_suspicious.get('reason', '')}",
        list(most_suspicious.get("limitations", [])),
        most_suspicious_nodule_id=most_suspicious.get("nodule_id"),
    )

    return {
        "lung_rads_version": LUNG_RADS_VERSION,
        "exam": exam,
        "nodules": nodules,
        "limitations": limitations,
    }


def build_structured_report_input(
    lesion_features: Iterable[Mapping[str, Any]],
    report_id: str,
    scan_date: str,
    exam_context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the fixed JSON payload used by template and LLM report generation."""
    features = [deepcopy(dict(item)) for item in lesion_features]
    assessment = assess_exam(features, exam_context=exam_context)
    return {
        "schema_version": STRUCTURED_INPUT_SCHEMA_VERSION,
        "report_id": report_id,
        "scan_date": scan_date,
        "lung_rads": assessment,
        "nodules": features,
    }
