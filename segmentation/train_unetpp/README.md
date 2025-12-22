# UNet++ 肺結節/肺腫瘤分割訓練

使用 LNDb 資料集和 MSD Lung Tumours 資料集，以 PyTorch 訓練 UNet++ 模型進行肺部病灶自動分割。

## 特點

- **2.5D 輸入**：使用 z-1, z, z+1 三個連續切片作為 RGB 輸入
- **Slice-Level 預處理**：切片式快取，加速訓練載入
- **Lungmask 分割**：自動生成肺部遮罩用於 ROI 裁切
- **4-Patch 驗證**：Val/Test 使用 4-patch stitch → full-slice 評估
- **多指標評估**：Global Dice、Boundary IoU、Lesion F1

## 支援資料集

| 資料集 | 格式 | 病人數 | 任務 |
|--------|------|--------|------|
| **LNDb** | MHD/RAW | 236 | 肺結節分割 |
| **MSD Task06** | NIfTI | 64+32 | 肺腫瘤分割 |

## 安裝

```bash
pip install -r requirements.txt
```

## 使用方式

### LNDb 資料集

```bash
cd segmentation

# 1. 切片式預處理（推薦）
python train_unetpp\main.py --preprocess-slices

# 2. 訓練
python train_unetpp\main.py --epochs 100

# 3. 5-fold CV
python train_unetpp\main.py --cv --epochs 100
```

### MSD Lung Tumours 資料集

```bash
cd segmentation

# 1. 預處理
python train_unetpp\train_msd.py --preprocess

# 2. 訓練
python train_unetpp\train_msd.py --epochs 100
```

## 參數說明

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--epochs` | 訓練 epochs | 100 |
| `--batch_size` | Batch size | 16 |
| `--lr` | 學習率 | 1e-4 |
| `--patch_size` | Patch 大小 | 224 |
| `--encoder` | 編碼器名稱 | efficientnet-b4 |
| `--cv` | 啟用 5-fold CV | False |

## 評估指標

| 指標 | 說明 | 計算方式 |
|------|------|----------|
| **Global Dice** | 全局像素級 Dice | Stitch 後 full-slice 計算 |
| **Boundary IoU** | 邊界 IoU (d=2) | 只算有 GT 的 slice |
| **Lesion F1** | Slice-level 偵測 F1 | TP/FP/FN 按 slice 計算 |
| **Val Loss** | Patch-level loss | 4 個 patch 平均 |

## 目錄結構

```
train_unetpp/
├── main.py           # LNDb 訓練入口
├── train_msd.py      # MSD Lung 訓練入口
├── config.py         # 配置管理
├── dataset.py        # LNDb 資料集 (含 val_collate_fn)
├── msd_dataset.py    # MSD 資料集
├── trainer.py        # 訓練器 (4-patch stitch 驗證)
├── model.py          # UNet++ 模型
├── losses.py         # 損失函數
├── preprocess.py     # 預處理器
├── sampler.py        # Patch 採樣器
├── inference.py      # 推論模組
└── utils.py          # 工具函數
```

## 輸出結構

```
segmentation/result/unetpp_lndb_YYYYMMDD_HHMMSS/
├── config.json              # 訓練配置
├── data_split.json          # 資料分割
├── best_model.pth           # 最佳模型
├── history.json             # 訓練歷史
├── training_curves.png      # 訓練曲線
├── train.log                # 訓練日誌
└── validation_samples/      # 驗證視覺化
    └── epoch_XXX.png        # 每 5 epoch 儲存
```

## Val/Test 4-Patch Stitch 流程

```
1. Dataset 返回:
   - images_4patch: (4, 3, 224, 224)
   - positions: [(y1,x1), ...]
   - full_mask: (1, H, W)
   - full_image_mid: (H, W)

2. Trainer 處理:
   - Forward 4 patches → outputs (4,1,224,224)
   - 裁切 GT patches → 計算 patch-level loss
   - Max stitch → full_pred (1, H, W)
   - 計算 full-slice metrics

3. 視覺化:
   - Full Image | GT | Pred | Overlay
```

## 設計決策

| 設計 | 原因 |
|------|------|
| 2.5D 輸入 | 提供 Z 軸上下文，比 3D 省記憶體 |
| Slice-Level Cache | 避免每次讀取整個 volume，加速 10x |
| 4-Patch Stitch | 驗證時覆蓋整個 slice，避免 patch 邊界問題 |
| Lungmask ROI | 使用 lungmask 套件生成肺部遮罩 |
| Oversampling | 正樣本 oversample 到 70%，處理不平衡 |
