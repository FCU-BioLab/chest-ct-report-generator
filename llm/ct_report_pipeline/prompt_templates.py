"""
Prompt Templates for CT Report Generation

Professional radiology report templates (English only) for generating
structured CT reports from segmentation features with Lung-RADS 2022 scoring.
"""

# System prompt for the LLM
SYSTEM_PROMPT_BILINGUAL = """You are an experienced radiologist assistant. Generate professional CT chest reports based on provided nodule measurements.

Rules:
1. Use ONLY the provided measurements - do not fabricate data
2. Follow the standard radiology report structure
3. Output in English only
4. Include Lung-RADS 2022 category assessment
5. Leave uncertain fields empty or state "Not evaluated"
6. Be concise and clinically relevant"""

# Professional radiology report prompt (English only, with Lung-RADS 2022)
REPORT_GENERATION_PROMPT = """You are a radiologist. Write a CT chest report for the nodule(s) below.

RULES:
- Report ONLY the nodules listed - do NOT add extra nodules
- Do NOT fabricate location, patient ID, or clinical history
- SELECT the correct Lung-RADS category based on nodule size

NODULE DATA:
{nodule_descriptions}

LUNG-RADS GUIDE:
- Solid <6mm → Category 2, <1% malignancy, annual screening
- Solid 6-8mm → Category 3, 1-2% malignancy, 6-month follow-up
- Solid 8-15mm → Category 4A, 5-15% malignancy, 3-month follow-up or PET/CT
- Solid ≥15mm → Category 4B, >15% malignancy, PET/CT or biopsy

Write the report:

Report ID: {report_id}
Date: {scan_date}

Technique:
Non-contrast CT chest.

Findings:

Lungs:
[Describe each nodule: size, type, volume]

Mediastinum: No masses or lymphadenopathy.
Pleura: No effusion.

Lung-RADS Assessment:
Category: [SELECT ONE: 2 or 3 or 4A or 4B]
Malignancy Risk: [SELECT ONE: <1% or 1-2% or 5-15% or >15%]

Impression:
[Summarize: count, largest size, Lung-RADS category]

Recommendation:
[SELECT based on category: annual screening / 6-month CT / 3-month CT / PET-CT]"""

# Nodule description template (HU used internally for classification, not shown in report)
NODULE_DESCRIPTION_TEMPLATE = """Nodule {nodule_id}:
- Size: {size_mm:.1f} mm
- Volume: {volume_mm3:.1f} mm³
- Type: {nodule_type}
"""

# Fleischner criteria for nodule follow-up
FLEISCHNER_CRITERIA = {
    "solid": {
        "<6mm": "No routine follow-up required",
        "6-8mm": "CT at 6-12 months",
        ">8mm": "Consider CT at 3 months or PET/CT",
    },
    "subsolid": {
        "<6mm": "No routine follow-up",
        ">=6mm": "CT at 3-6 months",
    },
}


def get_fleischner_recommendation(size_mm: float, nodule_type: str = "solid") -> str:
    """Get Fleischner guideline recommendation based on size (English only)."""
    if nodule_type == "solid":
        if size_mm < 6:
            return "No routine follow-up required"
        elif size_mm <= 8:
            return "CT at 6-12 months"
        else:
            return "Consider CT at 3 months or PET/CT"
    else:
        if size_mm < 6:
            return "No routine follow-up"
        else:
            return "CT at 3-6 months"


def format_nodule_descriptions(lesion_features_list: list) -> str:
    """Format lesion features into text descriptions for the prompt."""
    descriptions = []
    
    for i, features in enumerate(lesion_features_list, 1):
        # Determine nodule type from HU value
        mean_hu = features.get("mean_hu", 0)
        if mean_hu < -600:
            nodule_type = "ground-glass"
        elif mean_hu < -300:
            nodule_type = "part-solid"
        elif mean_hu > 200:
            nodule_type = "calcified"
        else:
            nodule_type = "solid"
        
        desc = NODULE_DESCRIPTION_TEMPLATE.format(
            nodule_id=i,
            size_mm=features.get("equivalent_diameter_mm", 0),
            volume_mm3=features.get("volume_mm3", 0),
            nodule_type=nodule_type,
        )
        descriptions.append(desc)
    
    return "\n".join(descriptions)


def build_report_prompt(
    lesion_features_list: list,
    report_id: str = "",
    scan_date: str = "",
) -> str:
    """Build the complete prompt for report generation."""
    from datetime import datetime
    
    if not scan_date:
        scan_date = datetime.now().strftime("%Y/%m/%d")
    
    if not report_id:
        report_id = f"AUTO_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    nodule_descriptions = format_nodule_descriptions(lesion_features_list)
    
    prompt = REPORT_GENERATION_PROMPT.format(
        scan_date=scan_date,
        report_id=report_id,
        nodule_descriptions=nodule_descriptions,
    )
    
    return prompt

