# YOLOv7 Medical Image Detection Training

完整的 YOLOv7 醫學影像偵測訓練框架，專為胸部 CT 腫瘤偵測設計。

## 📋 概述

本專案將原始的 YOLOv11 Ultralytics 訓練腳本改寫為 YOLOv7 原生訓練流程，並整合了專為醫學影像設計的注意力模組：

### 核心特性

1. **醫學影像預處理管線**
   - HU 視窗化 (Hounsfield Unit Windowing)
   - CLAHE 對比度增強
   - 魯棒性百分位拉伸

2. **醫學專用模組** (可選啟用/關閉)
   - **CBAM** (Convolutional Block Attention Module) - 在每個 ELAN stage 後插入
   - **Swin Transformer Block** - 在 backbone 倒數第二個 stage 插入
   - **BiFPN** (Bidirectional Feature Pyramid Network) - 取代原 PAN neck
   - **SimAM** (Simple, Parameter-Free Attention) - 在偵測頭前插入

3. **進階訓練技術**
   - Exponential Moving Average (EMA)
   - 混合精度訓練 (AMP)
   - 餘弦退火學習率排程
   - 梯度裁剪
   - 多 GPU 支援

4. **資料集快取機制**
   - 保留原有的 dataset caching
   - 持久化索引儲存

## 📁 專案結構

```
detection/yolo_detection/
├── models/
│   ├── custom_layers.py         # 醫學模組實作 (CBAM, SimAM, Swin, BiFPN)
│   ├── yolov7_medical.yaml      # YOLOv7 + 醫學模組配置
│   └── yolov7_baseline.yaml     # 純 YOLOv7 基線配置
├── train_yolov7_medical.py      # 主訓練腳本
├── yolov7_model.py              # 模型載入器與建構器
├── yolov7_dataset.py            # 資料集適配器 (含醫學預處理)
├── yolov7_utils.py              # 訓練工具 (loss, EMA, scheduler, metrics)
└── README_YOLOV7.md             # 本文件
```

## 🚀 快速開始

### 安裝依賴

```bash
pip install torch torchvision
pip install numpy opencv-python pyyaml tqdm
pip install pydicom  # 如需處理 DICOM 檔案
```

### 基本訓練

```bash
# 使用醫學模組訓練 (預設)
python train_yolov7_medical.py \
    --data_dir ./datasets/ct_data \
    --epochs 120 \
    --batch_size 16 \
    --imgsz 640

# 純 YOLOv7 基線訓練 (不使用醫學模組)
python train_yolov7_medical.py \
    --data_dir ./datasets/ct_data \
    --epochs 120 \
    --use_medical_modules 0 \
    --model_config models/yolov7_baseline.yaml
```

### 進階訓練範例

```bash
# 完整訓練配置 (大圖 + 強增強 + 醫學預處理)
python train_yolov7_medical.py \
    --data_dir ./datasets/ct_data \
    --epochs 150 \
    --batch_size 32 \
    --imgsz 800 \
    --lr 0.001 \
    --optimizer AdamW \
    --warmup_epochs 10 \
    --window_center -600 \
    --window_width 1500 \
    --enable_clahe 1 \
    --use_ema \
    --mixed_precision \
    --multi_scale \
    --device 0,1,2,3  # 多 GPU
```

### 關閉 HU 視窗化 (如資料已預處理)

```bash
python train_yolov7_medical.py \
    --data_dir ./datasets/ct_data \
    --enable_hu_windowing 0 \
    --enable_clahe 1
```

## 📊 命令列參數

### 資料參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--data_dir` | (必填) | 資料集根目錄 |
| `--train_split` | `train` | 訓練集名稱 |
| `--val_split` | `` | 驗證集名稱 (空則使用 ratio 切分) |
| `--val_ratio` | `0.2` | 驗證集比例 |
| `--include_negative` | `True` | 包含負樣本 |
| `--max_negative` | `20` | 每位病人最大負樣本數 |

### 訓練參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--epochs` | `120` | 訓練輪數 |
| `--batch_size` | `16` | 批次大小 |
| `--lr` | `0.001` | 學習率 |
| `--imgsz` | `640` | 影像大小 |

### 模型參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--model_config` | `models/yolov7_medical.yaml` | 模型配置檔 |
| `--use_medical_modules` | `1` | 啟用醫學模組 (1/0) |
| `--pretrained` | `` | 預訓練權重路徑 |

### 優化器參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--optimizer` | `Adam` | 優化器 (SGD/Adam/AdamW) |
| `--weight_decay` | `5e-4` | 權重衰減 |
| `--momentum` | `0.937` | 動量 (SGD) |
| `--warmup_epochs` | `5` | Warmup 輪數 |
| `--cos_lr` | `1` | 使用餘弦學習率 |

### 醫學預處理參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--enable_hu_windowing` | `1` | 啟用 HU 視窗化 |
| `--window_center` | `-600.0` | 視窗中心 (肺窗) |
| `--window_width` | `1500.0` | 視窗寬度 (肺窗) |
| `--enable_clahe` | `1` | 啟用 CLAHE |
| `--clahe_clip_limit` | `2.0` | CLAHE 限制 |
| `--clahe_tile_grid` | `8` | CLAHE 網格大小 |

### 進階參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--use_ema` | `True` | 使用 EMA |
| `--ema_decay` | `0.9999` | EMA 衰減率 |
| `--gradient_clip` | `10.0` | 梯度裁剪 |
| `--mixed_precision` | `True` | 混合精度訓練 |
| `--multi_scale` | `True` | 多尺度訓練 |

## 🏗️ 架構細節

### 醫學模組整合點

#### Backbone (CBAM + Swin Transformer)
```
Input → Conv → ELAN1 → [CBAM] → ELAN2 → [CBAM] → 
        ELAN3 → [CBAM] → [Swin Transformer] → ELAN4 → [CBAM]
```

#### Neck (BiFPN)
```
P3, P4, P5 (from backbone) → [BiFPN] → P3_out, P4_out, P5_out
```

#### Head (SimAM)
```
P3_out → [SimAM] → Detect_P3
P4_out → [SimAM] → Detect_P4
P5_out → [SimAM] → Detect_P5
```

### 模組說明

1. **CBAM (Convolutional Block Attention Module)**
   - 通道注意力 + 空間注意力
   - 參數量：~16x 縮減比
   - 位置：每個 ELAN stage 後

2. **Swin Transformer Block**
   - 視窗化多頭自注意力
   - 參數量：~4x MLP ratio
   - 位置：backbone 倒數第二個 stage

3. **BiFPN (Bidirectional Feature Pyramid Network)**
   - 雙向特徵融合
   - 可學習權重
   - 位置：取代 PAN neck

4. **SimAM (Simple, Parameter-Free Attention)**
   - 無參數注意力
   - 計算量極小
   - 位置：偵測頭前

## 📈 訓練輸出

### 目錄結構

```
yolov7_models/
└── run_YYYYMMDD_HHMMSS/
    ├── weights/
    │   ├── best.pt          # 最佳模型
    │   ├── last.pt          # 最後一輪模型
    │   └── epoch_*.pt       # 週期性檢查點
    ├── training_history.json  # 訓練歷史
    └── summary.json         # 訓練摘要
```

### 日誌檔案

```
yolov7_logs/
└── yolov7_training_YYYYMMDD_HHMMSS.log
```

## 🔧 自訂模組

### 新增自訂注意力模組

1. 在 `models/custom_layers.py` 中實作模組：

```python
class MyAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        # 實作你的注意力模組
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 前向傳播
        return x
```

2. 在 `models/yolov7_medical.yaml` 中使用：

```yaml
backbone:
  [
    # ... 其他層 ...
    [-1, 1, MyAttention, [256]],  # 插入自訂模組
    # ... 其他層 ...
  ]
```

3. 在 `yolov7_model.py` 的 `_parse_model` 中註冊：

```python
module_dict['MyAttention'] = MyAttention
```

## 📊 效能比較

### Baseline vs Medical Modules

| 配置 | 參數量 | 醫學模組參數 | 訓練時間 | mAP@0.5 |
|------|--------|--------------|----------|---------|
| YOLOv7 Baseline | ~37M | 0 | 1.0x | 待測試 |
| YOLOv7 + Medical | ~42M | ~5M (12%) | 1.2x | 待測試 |

### 醫學模組參數貢獻

- CBAM: ~1.5M (3%)
- Swin Transformer: ~2M (5%)
- BiFPN: ~1M (2%)
- SimAM: 0 (無參數)

## 🐛 故障排除

### 常見問題

1. **ImportError: No module named 'cv2'**
   ```bash
   pip install opencv-python
   ```

2. **CUDA out of memory**
   - 減少 batch size: `--batch_size 8`
   - 減少影像大小: `--imgsz 512`
   - 關閉混合精度: 移除 `--mixed_precision`

3. **模組找不到**
   - 確認執行目錄：應在 `detection/yolo_detection/` 目錄下執行
   - 檢查 Python path

4. **醫學模組載入失敗**
   - 檢查 `models/custom_layers.py` 是否存在
   - 使用基線配置: `--use_medical_modules 0`

## 📝 引用

如果本專案對您的研究有幫助，請考慮引用：

```bibtex
@software{yolov7_medical_detection,
  title={YOLOv7 Medical Image Detection Framework},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo/chest-ct-report-generator}
}
```

## 📄 授權

本專案遵循與主專案相同的授權條款。

## 🤝 貢獻

歡迎提交 Issue 與 Pull Request！

## 📧 聯絡

如有問題或建議，請透過 Issue 聯繫。

---

**注意**：本框架為研究與開發用途，請在臨床應用前進行充分驗證。
