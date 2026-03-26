# dataset_process

此目錄保留資料前處理與 manifest 相關工具，已移除舊版且未被使用的 YOLO / viewer 程式。

## 保留工具

- `create_lidc_minimal_manifests.ps1`
  - 用途：產生最小化 NBIA/LIDC manifest 批次
  - 場景：資料下載與整理（LIDC / LUNA16-New）

- `normalize_retinanet_jsons.py`
  - 用途：統一 RetinaNet dataset JSON 結構（`training` / `validation` / `testing`）與欄位別名
  - 場景：遷移舊版 dataset JSON 時的一次性整理

## 相關主流程

目前 RetinaNet 使用的資料生成腳本位於：

- `detection/retinanet/prepare_data.py`
- `detection/retinanet/prepare_luna16_new.py`
- `detection/retinanet/make_kfold_json.py`
