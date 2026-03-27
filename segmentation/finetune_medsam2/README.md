# MedSAM2 微調（胸部 CT 分割）

此模組現在支援兩種資料來源：
- `cache`：沿用既有 `.npz` 快取流程（相容舊訓練）
- `manifest`：直接使用 `NIfTI/MHD + JSON manifest`（推薦，便於串接 Detection）

## 1. 主要更新

- 新增 `--data_mode manifest`
- 新增 `--manifest` 輸入
- 新增 `--prompt_mode {gt,det,hybrid}`
- 新增 `--det_prompt_json`（可選）
- 新增 `--prompt_jitter_px`（訓練時增加 prompt 擾動）
- 新增 `build_manifest.py` 產生分割 manifest

## 2. Manifest 格式

```json
{
  "training": [
    {
      "patient_id": "case001",
      "image": "path/to/case001.nii.gz",
      "mask": "path/to/case001.nii.gz",
      "boxes": [[x1, y1, z1, x2, y2, z2]]
    }
  ],
  "validation": [],
  "testing": []
}
```

欄位說明：
- `image`：必要，CT 影像
- `mask`：建議提供（訓練/驗證需要 GT mask）
- `boxes`：可選，Detection 3D 框（可轉成 segmentation prompt）

## 3. 建立 Manifest

```bash
python segmentation/finetune_medsam2/build_manifest.py \
  --image_dir E:/dataset/images \
  --mask_dir E:/dataset/masks \
  --output_json segmentation/manifests/dataset_segmentation.json \
  --relative_paths
```

若要把 detection 推論框一起帶入：

```bash
python segmentation/finetune_medsam2/build_manifest.py \
  --image_dir E:/dataset/images \
  --mask_dir E:/dataset/masks \
  --det_report_dir E:/dataset/detection_reports \
  --output_json segmentation/manifests/dataset_segmentation.json
```

## 4. 訓練（Manifest 模式）

使用 GT mask 產生 prompt：

```bash
python segmentation/finetune_medsam2/main.py \
  --data_mode manifest \
  --manifest segmentation/manifests/dataset_segmentation.json \
  --prompt_mode gt \
  --epochs 100
```

使用 Detection prompt（無 Detection 框時會回退 GT）：

```bash
python segmentation/finetune_medsam2/main.py \
  --data_mode manifest \
  --manifest segmentation/manifests/dataset_segmentation.json \
  --prompt_mode det \
  --det_prompt_json segmentation/manifests/det_prompts.json \
  --prompt_jitter_px 4 \
  --epochs 100
```

混合模式（優先 Detection 框）：

```bash
python segmentation/finetune_medsam2/main.py \
  --data_mode manifest \
  --manifest segmentation/manifests/dataset_segmentation.json \
  --prompt_mode hybrid \
  --epochs 100
```

## 5. 訓練（舊 Cache 模式，保留相容）

```bash
python segmentation/finetune_medsam2/main.py \
  --data_mode cache \
  --cache_dir cache \
  --cache_dataset_type lndb \
  --epochs 100
```

## 6. 測試與特徵輸出

```bash
python segmentation/finetune_medsam2/main.py \
  --test \
  --resume segmentation/result/segmentation_xxx/best_model.pth
```

## 7. Prompt 模式建議

- `gt`：最穩定，適合初次訓練 baseline
- `det`：模擬真實串接 Detection 後的誤差
- `hybrid`：實務推薦；有 detection 框就用，否則回退 GT
