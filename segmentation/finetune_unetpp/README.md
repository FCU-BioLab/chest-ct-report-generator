# UNet++ Fine-tuning for Chest Tumor Segmentation

使用 UNet++ (Nested U-Net) 架構進行胸部 CT 腫瘤分割微調。

## 📋 功能特點

- **SMP UNet++**: 使用 [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) 的 UNet++
- **預訓練權重**: 支援 ImageNet 預訓練的多種 Encoder (ResNet, EfficientNet, MobileNet 等)
- **多種損失函數**: Dice, Combined, Enhanced, Tversky, Focal
- **資料增強**: 翻轉、旋轉、亮度調整、噪聲添加
- **混合精度訓練**: 加速訓練，減少記憶體使用
- **完整評估**: Dice, IoU, Precision, Recall, HD95

## 🏗️ 模型架構

```
UNet++ 架構圖:

X_0,0 ----→ X_0,1 ----→ X_0,2 ----→ X_0,3 ----→ X_0,4 → Output
  ↓           ↑           ↑           ↑           ↑
X_1,0 ----→ X_1,1 ----→ X_1,2 ----→ X_1,3
  ↓           ↑           ↑           ↑
X_2,0 ----→ X_2,1 ----→ X_2,2
  ↓           ↑           ↑
X_3,0 ----→ X_3,1
  ↓           ↑
X_4,0 (Bottleneck)
```

## 📦 安裝

```bash
pip install -r requirements.txt
```

## 🚀 使用方法

### 基本訓練 (使用 SMP)

```bash
# 預設使用 SMP UNet++ (ResNet34 + ImageNet 預訓練)
python main.py --epochs 100

# 使用不同的 Encoder
python main.py --encoder_name efficientnet-b0 --epochs 100

# 使用輕量 MobileNet (適合快速推論)
python main.py --encoder_name mobilenet_v2 --epochs 100

# 不使用預訓練權重
python main.py --encoder_weights None --epochs 100
```

### 使用自定義實現

```bash
# 使用自定義 UNet++ (帶深度監督)
python main.py --no_smp --model_type standard --epochs 100

# 使用輕量版自定義模型
python main.py --no_smp --model_type lite --epochs 100
```

### 資料增強與損失函數

```bash
# 啟用資料增強
python main.py --augmentation --epochs 100

# 使用增強版損失函數
python main.py --loss_type enhanced --epochs 100

# 快速測試（10% 資料）
python main.py --data_fraction 0.1 --epochs 5
```

### 🔧 CT 前處理選項

```bash
# 使用 2x2 網格 patch 模式 (224x224)
python main.py --use_patches --patch_size 224 --epochs 100

# 啟用 HSV 色彩空間增強
python main.py --augmentation --hsv_augmentation --epochs 100

# 完整前處理流程
python main.py --use_patches --augmentation --hsv_augmentation --epochs 100
```

**前處理步驟：**
1. CT 值裁剪: `[-1000, 800] HU`
2. 正規化到 `[0, 1]`
3. 2x2 網格裁剪 (可選): 4 個 224×224 patches
4. HSV 增強 (可選): 色相/飽和度/亮度隨機調整

### 繼續訓練 / 評估

```bash
# 從 checkpoint 繼續訓練
python main.py --resume result/unetpp_XXXXXXXX/best_model.pth --epochs 150

# 只評估模型
python main.py --eval_only --resume result/unetpp_XXXXXXXX/best_model.pth
```

### 🧠 LLM 特徵提取

提取深層特徵供 LLM 生成報告使用：

```bash
# 訓練後自動提取 LLM 特徵
python main.py --epochs 100 --extract_llm_features

# 從已訓練模型提取特徵
python main.py --eval_only --resume best_model.pth --extract_llm_features
```

**輸出檔案：**
```
result/unetpp_XXXXXXXX/llm_features/
├── llm_features.json      # 完整特徵 (512-dim deep features + morphological)
└── features_summary.txt   # 可讀摘要
```

**JSON 格式：**
```json
{
  "extraction_info": {
    "total_samples": 31,
    "deep_feature_dim": 512
  },
  "samples": [
    {
      "patient_id": "70",
      "slice_idx": 155,
      "deep_features": [0.123, -0.456, ...],
      "predicted_morphological": {
        "has_lesion": true,
        "area_pixels": 2840,
        "circularity": 0.478,
        "centroid_x": 0.37,
        "centroid_y": 0.58
      },
      "dice_score": 0.0014
    }
  ]
}
```

## 📊 模型選項

### SMP UNet++ (推薦)

| Encoder | 參數量 | 特點 |
|---------|--------|------|
| `resnet34` | ~26M | 推薦，平衡性能與速度 |
| `resnet50` | ~44M | 更大容量 |
| `efficientnet-b0` | ~8M | 高效輕量 |
| `mobilenet_v2` | ~6M | 最輕量，適合推論 |

### 自定義 UNet++ (--no_smp)

| 類型 | 參數量 | 特點 |
|------|--------|------|
| `standard` | ~36M | 完整功能，帶深度監督 |
| `lite` | ~3M | 輕量版 |

## 🎯 損失函數

### 抗崩潰損失函數 (Anti-Collapse) ⭐
| 名稱 | 描述 | 預設參數 | 適用場景 |
|------|------|----------|----------|
| `stable` | **[預設]** Weighted BCE + Dice | `pos_weight=10, bce=0.3, dice=0.7` | 一般訓練，抗模型崩潰 |
| `focal_dice` | **[推薦]** Focal + Dice | `alpha=0.75, gamma=2.0` | 極度不平衡，小病灶分割 |

### 其他損失函數
| 名稱 | 描述 | 適用場景 |
|------|------|----------|
| `dice` | Dice Loss | 基礎分割 |
| `combined` | Dice + BCE (舊版) | 傳統訓練 |
| `enhanced` | Dice + Focal + Tversky + Boundary | 追求高 DSC |
| `tversky` | Tversky Loss | 調控 FP/FN 權重 |
| `focal` | Focal Loss | 純 Focal (不含 Dice) |

### ⚠️ 模型崩潰問題
如果訓練時出現 `pred_positive_ratio` 快速下降到 0%，或 Dice 持續為 0，這是**模型崩潰**的徵兆。

**解決方案：**
```bash
# 使用 focal_dice 損失 (最強抗崩潰)
python main.py --loss_type focal_dice --use_patches --augmentation --epochs 50

# 或調整 stable 損失的 pos_weight
python main.py --loss_type stable --epochs 50
```

## 📁 輸出結構

```
result/unetpp_YYYYMMDD_HHMMSS/
├── config.json              # 訓練配置
├── dataset_split.json       # 資料集分割資訊
├── best_model.pth           # 最佳模型
├── checkpoint_epoch_X.pth   # 定期 checkpoint
├── training_history.json    # 訓練歷史
├── training_curves.png      # 訓練曲線圖
├── test_results.json        # 測試結果
├── training_*.log           # 訓練日誌
└── visualizations/          # 分割結果可視化
```

## 🔧 命令列參數

### SMP 模型參數 (預設)
| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--use_smp` | True | 使用 SMP UNet++ |
| `--encoder_name` | resnet34 | Encoder 名稱 |
| `--encoder_weights` | imagenet | 預訓練權重 |
| `--encoder_depth` | 5 | Encoder 深度 (3-5) |

### 自定義模型參數 (--no_smp)
| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--model_type` | standard | 模型類型 |
| `--features` | [64,128,256,512,1024] | 各層特徵通道 |
| `--deep_supervision` | True | 使用深度監督 |

### 資料參數
| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--dataset_type` | lndb | 資料集類型 |
| `--data_dir` | ../datasets/... | 資料集目錄 |
| `--rad_id` | consensus | 放射科醫師 ID |
| `--axis` | 2 | 切片軸向 |
| `--data_fraction` | 1.0 | 使用資料比例 |
| `--target_size` | 256 256 | 目標影像大小 |

### 訓練參數
| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--epochs` | 100 | 訓練輪數 |
| `--batch_size` | 8 | 批次大小 |
| `--lr` | 1e-4 | 學習率 |
| `--weight_decay` | 1e-4 | 權重衰減 |
| `--early_stopping_patience` | 30 | 早停耐心值 |
| `--loss_type` | stable | 損失函數類型 (stable/focal_dice/combined) |
| `--augmentation` | False | 啟用資料增強 |
| `--use_amp` | True | 使用混合精度 |
| `--threshold` | 0.5 | 預測二值化閾值 |

## 📈 預期效能

| 資料集 | 模型 | Dice | IoU |
|--------|------|------|-----|
| LNDb | SMP (resnet34) | 0.85-0.90 | 0.75-0.82 |
| LNDb | Custom Standard | 0.85-0.90 | 0.75-0.82 |
| LNDb | Custom Lite | 0.80-0.85 | 0.70-0.75 |

## 🔗 參考文獻

- Zhou et al., "UNet++: A Nested U-Net Architecture for Medical Image Segmentation" ([arXiv](https://arxiv.org/abs/1807.10165))
- [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)

## 📝 License

MIT License
