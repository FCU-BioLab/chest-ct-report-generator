# MedSAM2 Fine-tuning for Chest CT Tumor Segmentation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

專門用於微調 MedSAM2 模型進行胸部 CT 腫瘤分割的訓練套件。

## 🎯 功能特點

- **Prompt-based 分割**: 使用 BBox prompts 進行精準分割
- **凍結 Image Encoder**: 只訓練 Prompt Encoder + Mask Decoder (~11.7M params)
- **快取資料集模式**: 預處理的 .npz 切片，大幅加速訓練
- **LNDb + MSD 資料集**: 支援兩種肺部腫瘤資料集
- **自動訓練曲線**: 每 epoch 自動更新 training_curves.png
- **LLM 特徵提取**: 測試階段可提取病灶特徵供下游任務

## 🏗️ 模型架構

```
┌──────────────────────────────────────────────────────┐
│                    MedSAM2                           │
├──────────────────────────────────────────────────────┤
│  🔒 Image Encoder (凍結)              ~38M params    │
│     └─ Hiera Transformer backbone                   │
├──────────────────────────────────────────────────────┤
│  🔓 Prompt Encoder (可訓練)                          │
│     └─ BBox → sparse/dense embeddings               │
├──────────────────────────────────────────────────────┤
│  🔓 Mask Decoder (可訓練)            ~11.7M params   │
│     └─ embeddings → segmentation mask               │
└──────────────────────────────────────────────────────┘
```

## � 完整訓練流程 (Pipeline)

### Stage 1: 資料預處理

```
原始 CT (MHD/NIfTI)
        │
        ▼
┌─────────────────────────────────────┐
│ 1. 載入 CT 影像                     │
│    └─ load_mhd() / SimpleITK        │
├─────────────────────────────────────┤
│ 2. Spacing Resample                 │
│    └─ 統一到 (1.0, 1.0, 1.0) mm     │
├─────────────────────────────────────┤
│ 3. HU Windowing (Lung Window)       │
│    └─ Center=-400, Width=1200       │
│    └─ 輸出範圍: [-1000, 200] HU     │
│    └─ 歸一化到 [0, 1]               │
├─────────────────────────────────────┤
│ 4. 肺部遮罩 (Lungmask)              │
│    └─ 3D U-Net 肺野分割             │
├─────────────────────────────────────┤
│ 5. 軟共識遮罩 (Soft Consensus)      │
│    └─ 多位醫師標註平均              │
├─────────────────────────────────────┤
│ 6. 2D 切片輸出                      │
│    └─ slice_{z}.npz                 │
│    └─ 包含: image, mask, lung_mask  │
└─────────────────────────────────────┘
        │
        ▼
cache/{dataset}_slices/{patient}/slice_XXXX.npz
```

**預處理指令：**
```bash
# 預處理 LNDb 資料集
python train_unetpp/preprocess.py --data_dir path/to/LNDb --output_dir cache/lndb_slices
```

### Stage 2: 資料載入與增強

```
slice_XXXX.npz
        │
        ▼
┌─────────────────────────────────────┐
│ CachedSliceDataset                  │
├─────────────────────────────────────┤
│ 1. 載入 NPZ (image, mask, lung)     │
│ 2. 2.5D 格式 (Z-1, Z, Z+1)          │
│ 3. Resize to 512x512                │
│ 4. 轉為 RGB 格式 (3 channel)        │
│ 5. 從 Mask 產生 BBox prompts        │
├─────────────────────────────────────┤
│ 資料增強 (可選):                    │
│    └─ RandomFlip                    │
│    └─ RandomRotation                │
│    └─ RandomBrightness              │
│    └─ ElasticDeformation (強增強)   │
└─────────────────────────────────────┘
        │
        ▼
   DataLoader (batch_size=16)
```

### Stage 3: 模型訓練

```
每個 Batch:
┌─────────────────────────────────────┐
│ Input: image [B, 3, 512, 512]       │
│        bbox  [B, N, 4]              │
│        mask  [B, 1, 512, 512]       │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│ Image Encoder (凍結)                │
│    └─ image → image_embedding       │
├─────────────────────────────────────┤
│ Prompt Encoder                      │
│    └─ bbox → sparse/dense embed     │
├─────────────────────────────────────┤
│ Mask Decoder                        │
│    └─ embeddings → pred_mask        │
├─────────────────────────────────────┤
│ Loss: DiceLoss + BCEWithLogitsLoss  │
├─────────────────────────────────────┤
│ Optimizer: AdamW (lr=5e-6)          │
│ Scheduler: Warmup + CosineAnnealing │
└─────────────────────────────────────┘
        │
        ▼
    Backpropagation (只更新 Decoder)
```

### Stage 4: 驗證與評估

```
每個 Epoch 結束:
┌─────────────────────────────────────┐
│ 評估指標:                           │
│    ├─ Dice Score (3D Volume Dice)   │
│    ├─ IoU / Jaccard                 │
│    ├─ Precision / Recall            │
│    ├─ Specificity / Accuracy        │
│    └─ Hausdorff Distance 95         │
├─────────────────────────────────────┤
│ 輸出:                               │
│    ├─ best_model.pth (Dice 最佳)    │
│    ├─ training_curves.png           │
│    ├─ history.json                  │
│    └─ best_metrics.json             │
└─────────────────────────────────────┘
```

### Stage 5: 測試與特徵提取

```
python main.py --test --resume best_model.pth
        │
        ▼
┌─────────────────────────────────────┐
│ 測試集評估                          │
│    └─ 計算所有指標                  │
├─────────────────────────────────────┤
│ 特徵提取 (可選)                     │
│    ├─ 形態學特徵                    │
│    ├─ 強度統計特徵                  │
│    ├─ 深層特徵向量                  │
│    └─ 病灶分類 (惡性/良性)          │
│    └─ 模擬點擊 (Click Prompt)       │
├─────────────────────────────────────┤
│ 後處理 (Post-processing)            │
│    └─ NPY → NIfTI (3D Reconstruct)  │
└─────────────────────────────────────┘
        │
        ▼
   features/*.json (供 LLM 使用)
```


## �🚀 快速開始

### 基本訓練（使用快取資料）

```bash
cd segmentation

# LNDb 資料集訓練 (212 患者, ~60k 切片)
python finetune_medsam2/main.py --cache_dataset_type lndb --epochs 100

# MSD Lung 資料集訓練
python finetune_medsam2/main.py --cache_dataset_type msd --epochs 100

# 完整資料集 (LNDb + MSD, 275 患者)
python finetune_medsam2/main.py --cache_dataset_type both --augmentation --epochs 100

# 使用增強損失函數（推薦用於高 DSC）
python finetune_medsam2/main.py --cache_dataset_type lndb --loss_type enhanced --epochs 100

# 使用 MedSAM2 原生損失函數（與 MedSAM2 官方訓練一致）
python finetune_medsam2/main.py --cache_dataset_type lndb --loss_type native --epochs 100

# 快速測試
python finetune_medsam2/main.py --data_fraction 0.1 --epochs 5
```

## ⚙️ 配置系統

所有預設值定義於 `config.py`：

```python
from finetune_medsam2.config import get_default_config

config = get_default_config()
print(config.training.epochs)      # 100
print(config.training.batch_size)  # 16
print(config.training.learning_rate)  # 5e-6
```

## 📋 參數說明

### 資料參數
| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--cache_dir` | `cache` | 快取目錄 |
| `--cache_dataset_type` | `lndb` | `lndb` / `msd` / `both` |
| `--data_fraction` | `1.0` | 資料使用比例 |
| `--use_2_5d` | `True` | 使用 2.5D 輸入 (Z-1, Z, Z+1) |
| `--no_2_5d` | - | 禁用 2.5D，使用傳統 2D |

### 訓練參數
| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--epochs` | `100` | 訓練輪數 |
| `--batch_size` | `32` | 批次大小 |
| `--lr` | `5e-6` | 學習率 (SAM 需低 LR) |
| `--weight_decay` | `1e-6` | 權重衰減 |
| `--early_stopping_patience` | `10` | 早停容忍 epochs |
| `--loss_type` | `combined` | 損失函數: `combined`/`enhanced`/`native`/`tversky`/`focal` |
| `--warmup_epochs` | `5` | Warmup epochs |
| `--accumulation_steps` | `1` | 梯度累積步數 |
| `--augmentation` | - | 啟用基本資料增強 |
| `--strong_augmentation` | - | 啟用強資料增強 |

### 模型參數
| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--config` | `sam2.1_hiera_t512.yaml` | SAM2 配置 |
| `--checkpoint` | `MedSAM2_CTLesion.pt` | 預訓練權重 |
| `--seed` | `42` | 隨機種子 |

## 📁 輸出結構

```
result/segmentation_{TIMESTAMP}/
├── training_*.log           # 完整訓練日誌
├── best_model.pth           # 最佳模型權重
├── final_model.pth          # 最終模型權重
├── best_metrics.json        # 最佳驗證指標
├── history.json             # 完整訓練歷史
├── training_curves.png      # 訓練曲線 (每 epoch 更新)
├── training_config.json     # 訓練配置
├── dataset_split.json       # 資料集分割
├── first_epoch_samples/     # 第一 epoch 樣本視覺化
│   ├── train_sample_*.png
│   └── val_sample_*.png
└── features/                # 測試特徵輸出
```

## 📊 預期效能

在 LNDb 資料集上的典型結果：

| 指標 | 數值 |
|------|------|
| **Dice** | 0.985+ |
| **IoU** | 0.978+ |
| **Precision** | 0.985+ |
| **Recall** | 0.990+ |
| **訓練時間** | ~33 min/epoch (RTX 3060 Ti) |

## 🧪 測試與特徵提取

```bash
# 在測試集上評估
python finetune_medsam2/main.py --test --resume result/.../best_model.pth

# 提取深層特徵向量 (供 LLM 使用)
python finetune_medsam2/main.py --test --extract_features --resume result/.../best_model.pth
```

## 📦 模組說明

| 模組 | 說明 |
|------|------|
| `main.py` | 程式入口與 CLI |
| `config.py` | 配置 dataclass 定義 |
| `trainer.py` | 訓練邏輯、驗證、特徵提取 |
| `dataset.py` | LNDbDataset + CachedSliceDataset |
| `losses.py` | 損失函數 (使用 MedSAM2 原生 Dice/Focal + 自訂擴充) |
| `utils.py` | 工具函數、logging |

## 📊 損失函數選項

| 類型 | 說明 | 推薦場景 |
|------|------|----------|
| `combined` | Dice + BCE（預設） | 通用訓練 |
| `enhanced` | Dice + Focal + Tversky + Boundary | 高 DSC 目標 |
| `native` | MedSAM2 原生 dice_loss + sigmoid_focal_loss | 與官方訓練一致 |
| `tversky` | Tversky Loss (alpha=0.7, beta=0.3) | 減少漏檢 |
| `focal` | Focal Loss | 類別不平衡 |

> **注意**: `native`、`enhanced`、`combined` 中的 Dice 和 Focal 損失已使用 MedSAM2 原生實作（來自 `training/loss_fns.py`）

## 📚 快取資料結構

```
cache/
├── lndb_slices/              # LNDb 快取 (212 患者)
│   └── LNDb-XXXX/
│       ├── meta.json         # 患者 metadata
│       └── slice_XXXX.npz    # image + mask + lung_mask
└── msd_lung_slices/          # MSD 快取 (63 患者)
    └── lung_XXX/
        ├── meta.json
        └── slice_XXXX.npz
```

## 🔧 進階用法

```bash
# 從 checkpoint 繼續訓練
python finetune_medsam2/main.py --resume result/.../best_model.pth

# 使用強資料增強
python finetune_medsam2/main.py --strong_augmentation --epochs 100

# 使用增強損失函數（推薦用於小型結節）
python finetune_medsam2/main.py --loss_type enhanced --epochs 100

# 梯度累積 (模擬更大 batch)
python finetune_medsam2/main.py --accumulation_steps 4 --batch_size 8

# 過濾小結節 (提高 Dice)
python finetune_medsam2/main.py --min_nodule_diameter 4 --epochs 100

# 禁用 2.5D 模式（使用傳統 2D）
python finetune_medsam2/main.py --no_2_5d --epochs 100

# 使用固定資料分割 (復現實驗)
python finetune_medsam2/main.py --split_file result/.../dataset_split.json

# 只評估模型

# 模擬醫生點擊測試 (Click Simulation)
python finetune_medsam2/main.py --test --test_prompt_type point --resume result/.../best_model.pth

# 後處理：將預測結果轉換為 NIfTI 格式
python finetune_medsam2/postprocess.py --result_dir result/segmentation_...
```

## 🧠 評估指標說明 (Volume Dice vs Slice Dice)

本專案採用 **Volume Dice** 作為主要評估指標，與傳統 Slice Dice 相比：
- **Volume Dice** (`2*Intersection / (Vol_Pred + Vol_GT)`): 把整個 3D 結節視為一個整體計算。更符合視覺感受，不會被邊緣小切片的低分影響。
- **Slice Dice**: 計算每張切片的 Dice 後取平均。容易低估 3D 分割品質。

> ⚠️ 注意: 訓練過程中看到的 Dice 已經修正為 Volume Dice。

## 🛠️ 後處理工具 (Post-processing)

`postprocess.py` 用於將模型預測的 2D 切片 (`.npy`) 重組回原始 3D CT 空間 (`.nii.gz`)。

```bash
python finetune_medsam2/postprocess.py --result_dir <RESULT_DIR>
```

**功能:**
- 自動讀取 `training_config.json` 找到原始 CT 資訊
- 重組 3D Mask 並還原幾何資訊 (Spacing, Origin, Direction)
- **LCC (Largest Connected Component)**: 只保留最大連通區 (去除雜訊)
- **Smoothing**: 形態學平滑處理 (Closing)
- **Small Object Removal**: 去除小於指定體積的雜訊

