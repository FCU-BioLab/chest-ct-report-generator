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

## 🚀 快速開始

### 使用快取資料訓練（推薦）

```bash
cd segmentation

# LNDb 資料集訓練 (212 患者, ~60k 切片)
python finetune_medsam2/main.py --use_cache --cache_dataset_type lndb

# 完整資料集 (LNDb + MSD, 275 患者)
python finetune_medsam2/main.py --use_cache --cache_dataset_type both --augmentation

# 快速測試
python finetune_medsam2/main.py --use_cache --data_fraction 0.1 --epochs 5
```

### 使用原始資料訓練

```bash
python finetune_medsam2/main.py --data_dir path/to/LNDb
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
| `--use_cache` | False | 啟用快取模式 |
| `--cache_dir` | `cache` | 快取目錄 |
| `--cache_dataset_type` | `both` | `lndb` / `msd` / `both` |
| `--data_fraction` | `1.0` | 資料使用比例 |

### 訓練參數
| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--epochs` | `100` | 訓練輪數 |
| `--batch_size` | `16` | 批次大小 |
| `--lr` | `5e-6` | 學習率 (SAM 需低 LR) |
| `--weight_decay` | `1e-6` | 權重衰減 |
| `--early_stopping_patience` | `20` | 早停容忍 epochs |
| `--loss_type` | `combined` | 損失函數 (Dice+BCE) |

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
| `losses.py` | 損失函數 (Dice, BCE, Combined) |
| `utils.py` | 工具函數、logging |

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
python finetune_medsam2/main.py --use_cache --strong_augmentation

# 梯度累積 (模擬更大 batch)
python finetune_medsam2/main.py --use_cache --accumulation_steps 4

# 使用固定資料分割 (復現實驗)
python finetune_medsam2/main.py --split_file result/.../dataset_split.json
```
