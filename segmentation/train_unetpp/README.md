# UNet++ 肺結節分割訓練

使用 LNDb 資料集和 PyTorch 訓練 UNet++ 模型進行肺結節自動分割。

## 特點

- **Patch-based 訓練**：解決極端類別不平衡
- **2.5D 固定 spacing**：統一到 1.0mm 各向同性
- **軟共識標註**：整合多位放射科醫師的標註
- **多指標評估**：ROI Dice、Lesion-wise F1、FP/scan
- **推論後處理**：連通區域過濾、肺野遮罩
- **結節屬性輸出**：JSON 格式供 LLM 使用

## 安裝

```bash
pip install -r requirements.txt
```

## 使用方式

### 1. 預處理資料集

```bash
python -m segmentation.train_unetpp.main --preprocess
```

### 2. 訓練模型

```bash
# 單次訓練（預設）
python train_unetpp\main.py --epochs 50

# 5-fold CV 訓練
python train_unetpp\main.py --cv --epochs 100

# 只訓練特定 fold
python train_unetpp\main.py --cv --fold 0 --epochs 100

# 快速測試
python train_unetpp\main.py --data_fraction 0.1 --epochs 5
```

### 3. 推論

```bash
python train_unetpp\main.py --inference --model_path path/to/model.pth
```

## 參數說明

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--epochs` | 訓練 epochs | 100 |
| `--batch_size` | Batch size | 16 |
| `--lr` | 學習率 | 1e-4 |
| `--patch_size` | Patch 大小 | 160 |
| `--encoder` | 編碼器名稱 | efficientnet-b4 |
| `--cv` | 啟用 5-fold CV | False |
| `--fold` | 指定 fold | None |
| `--data_fraction` | 資料比例 | 1.0 |

## 目標指標

| 指標 | 目標值 |
|------|--------|
| ROI Dice | ≥ 0.85 |
| Lesion Sensitivity | ≥ 0.90 |
| FP per Scan | ≤ 2.0 |

## 輸出結構

```
segmentation/result/train_unetpp/
├── unetpp_lndb_YYYYMMDD_HHMMSS/
│   ├── config.json
│   ├── best_model.pth
│   ├── final_model.pth
│   ├── history.json
│   └── training_curves.png
└── inference_results/
    ├── LNDb-0001_nodules.json
    ├── LNDb-0001_pred.nii.gz
    └── inference_summary.json
```

## JSON 輸出範例

結節屬性 JSON（供 LLM 使用）：

```json
{
  "patient_id": "LNDb-0001",
  "nodules": [
    {
      "id": 1,
      "centroid_mm": [120.5, 85.2, 45.0],
      "volume_mm3": 523.6,
      "max_diameter_mm": 10.2,
      "mean_hu": -450
    }
  ]
}
```

## 資料載入邏輯

### 整體流程

```
1. 預處理 (preprocess.py)
   LNDb .mhd/.raw → CTPreprocessor → cache/*.npz
   • Resample 到 1.0mm³ 各向同性
   • HU Windowing [-1000, 200]
   • 肺野分割與 ROI 裁切

2. 資料分割 (main.py)
   236 病人按 7:2:1 分割 → train(165) / val(47) / test(24)

3. 建立樣本索引 (dataset.py)
   每個病人的每個切片 = 一個樣本
   165 病人 → 約 52,000 個原始樣本

4. 正樣本 Oversampling
   正樣本（含結節切片）只佔 ~4%
   max_samples_per_epoch = 20,000
   正負比例調整為 7:3

5. 預載入到記憶體 (preload=True)
   首次啟動讀取所有 .npz 到記憶體
   避免訓練時重複讀取磁碟

6. __getitem__ 取樣
   ① 取出病人資料
   ② 創建軟共識遮罩 (多醫師平均)
   ③ 提取 2.5D 切片 (center ± 2mm)
   ④ 隨機採樣 160×160 Patch
   ⑤ 資料增強
   ⑥ 返回 image(3,160,160) + mask(1,160,160)

7. DataLoader
   batch_size=32 → 625 批次/epoch
```

### 設計決策

| 設計 | 原因 |
|------|------|
| Preload | 避免每次讀取磁碟，加速 100 倍 |
| 2.5D 切片 | 提供 Z 軸上下文，比 3D 省記憶體 |
| Patch-based | 處理類別不平衡，聚焦結節區域 |
| 軟共識遮罩 | 整合多醫師標註，處理標註不確定性 |
| max_samples_per_epoch | 限制樣本數，避免過長的 epoch |
