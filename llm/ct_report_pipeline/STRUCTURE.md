# Pipeline Structure (Organized)

## Core (for pipeline runtime)
- config/
- features/
- segmentation/
- scripts/
- report_generator.py
- prompt_templates.py
- quick_start.py

## Assets (large/training artifacts)
- assets/data/
- assets/models/

## Extras (optional modules)
- extras/dataset_process/
- extras/evaluation/
- extras/tests/
- extras/PROJECT_DOCUMENTATION.md

## Notes
- Config paths already updated to the new project location.
- Default LoRA/data paths now point to `assets/`.
