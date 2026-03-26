# dataset_process

This directory was cleaned to remove legacy, unreferenced YOLO/viewer utilities.

## Kept

- `create_lidc_minimal_manifests.ps1`
  - Purpose: generate minimal NBIA/LIDC manifest batches.
  - Typical use: dataset acquisition support for LIDC/LUNA16-New workflows.
- `normalize_retinanet_jsons.py`
  - Purpose: normalize RetinaNet dataset JSON schema (`training`/`validation`/`testing`) and key aliases.
  - Typical use: one-time cleanup when migrating old dataset JSON files.

## Notes

- Current active dataset preparation for RetinaNet is under:
  - `detection/retinanet/prepare_data.py`
  - `detection/retinanet/prepare_luna16_new.py`
