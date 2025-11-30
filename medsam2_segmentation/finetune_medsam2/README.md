# MedSAM2 Fine-tuning for Chest Tumor Segmentation

這是一個專門用於微調 MedSAM2 模型以進行胸部 CT 腫瘤分割的套件。

## 功能特點

- **自動化資料處理**: 支援 NIfTI 格式，自動提取切片與 Bounding Box Prompts。
- **記憶體優化**: 針對有限 VRAM (8GB) 進行優化，包含梯度累積與 Embedding 緩存管理。
- **完整評估指標**: 包含 Dice, IoU, Precision, Recall, Specificity, Hausdorff Distance (95%)。
- **錯誤分析**: 自動生成低分病例報告，協助診斷模型弱點。
- **訓練穩定性**: 整合早停機制 (Early Stopping) 與學習率調度 (Cosine Annealing)。
- **LLM 特徵提取**: 支援測試階段提取病灶特徵用於 LLM Fine-Tuning。

## 安裝需求

請參考 `requirements.txt` 安裝必要套件：

```bash
pip install -r requirements.txt
```

此外，需要確保 `MedSAM2` 專案已正確安裝並位於上層目錄。

## 使用方法

### 1. 開始訓練

```bash
# 基本訓練 (預設參數)
python main.py

# 自定義參數訓練
python main.py --epochs 50 --batch_size 4 --lr 1e-5 --accumulation_steps 2
```

### 2. 從 Checkpoint 恢復訓練

```bash
python main.py --resume result/segmentation_TIMESTAMP/checkpoint_epoch_10.pth
```

### 3. 僅進行評估 (Evaluation Only)

```bash
python main.py --eval_only --checkpoint result/segmentation_TIMESTAMP/best_model.pth
```

### 4. 測試並提取特徵（用於 LLM Fine-Tuning）

這是用於後續 LLM Fine-Tuning 的關鍵功能，可在測試階段提取病灶的多層次特徵：

```bash
# 基本測試與特徵提取
python main.py --test --resume result/segmentation_TIMESTAMP/best_model.pth

# 提取深層特徵向量（需要更多記憶體）
python main.py --test --resume result/segmentation_TIMESTAMP/best_model.pth --extract_features

# 指定特徵輸出目錄
python main.py --test --resume result/segmentation_TIMESTAMP/best_model.pth --feature_output_dir ./llm_features
```

#### 提取的特徵類型

1. **形態學特徵 (Morphological Features)**
   - 面積 (mm²)、周長 (mm)、等效直徑 (mm)
   - 主軸/副軸長度、離心率
   - 圓形度、實心度、緊密度
   - 邊界框面積、填充率

2. **強度特徵 (Intensity Features)**
   - 平均/標準差/最大/最小 CT 值
   - 中位數、四分位數
   - 偏度、峰度、熵
   - 與背景對比度

3. **深層特徵 (Deep Features)** - 可選
   - Image Encoder 全局特徵
   - Prompt Encoder 稀疏/密集嵌入
   - 多尺度高解析度特徵

4. **文字描述 (Text Description)**
   - 自動生成的病灶描述文字
   - 可直接用於 LLM 輸入

#### 輸出檔案結構（每個患者獨立資料夾）

```
features/
├── full_features_{timestamp}.json       # 完整特徵摘要（不含向量）
├── test_summary_{timestamp}.json        # 測試摘要
├── llm_data/                            # LLM 訓練資料
│   ├── llm_training_data_all_{timestamp}.json  # 所有患者整合版
│   ├── {patient_id}_llm.json            # 每個患者獨立 LLM 資料
│   └── ...
├── patients/                            # 患者獨立特徵（每個患者一個資料夾）
│   ├── {patient_id}/
│   │   ├── metadata.json                # 患者基本資訊和摘要
│   │   ├── features.json                # 完整特徵（不含深層向量）
│   │   ├── deep_features.npz            # 深層特徵向量（NumPy 格式）
│   │   ├── llm_input.txt                # LLM 輸入文字
│   │   ├── llm_training_sample.json     # 完整 LLM 訓練樣本
│   │   └── slices/                      # 切片級別特徵
│   │       ├── slice_0001_deep.npz      # 切片深層特徵
│   │       └── ...
│   └── ...
└── predictions/                         # 預測遮罩
    ├── {patient_id}/
    │   ├── slice_0001_pred.npy
    │   └── ...
    └── ...
```

#### 患者獨立特徵檔案說明

每個患者資料夾包含：

| 檔案 | 說明 |
|------|------|
| `metadata.json` | 患者 ID、摘要統計、切片數量 |
| `features.json` | 完整形態學與強度特徵（JSON 格式，不含向量） |
| `deep_features.npz` | 深層特徵向量（NumPy 壓縮格式），包含聚合特徵 |
| `llm_input.txt` | 純文字版病灶描述，可直接用於 LLM 輸入 |
| `llm_training_sample.json` | 完整 LLM 訓練樣本（含數值特徵與深層向量） |
| `slices/` | 切片級別的深層特徵 |

#### 深層特徵向量格式 (deep_features.npz)

使用 NumPy 載入：

```python
import numpy as np

# 載入患者深層特徵
data = np.load("patients/{patient_id}/deep_features.npz")

# 可用的特徵
print(data.files)
# ['image_embeddings', 'sparse_embeddings', 'dense_embeddings', 
#  'slice_indices', 'lesion_indices', 
#  'aggregated_image_embedding', 'aggregated_sparse_embedding', 'aggregated_dense_embedding']

# 使用聚合特徵（推薦用於 LLM）
aggregated_embedding = data['aggregated_image_embedding']  # shape: (dim,)

# 或使用所有切片的特徵
all_embeddings = data['image_embeddings']  # shape: (num_lesions, dim)
```

#### LLM 訓練資料格式

`llm_training_sample.json` 包含以下結構：

```json
{
  "patient_id": "1.3.6.1.4.1.14519.5.2.1...",
  "safe_patient_id": "1_3_6_1_4_1_14519_5_2_1___",
  "input": "# 胸部 CT 病灶分析報告\n\n## 摘要\n...",
  "numerical_features": {
    "total_lesions": 3,
    "total_slices": 5,
    "avg_area_mm2": 45.2,
    "max_area_mm2": 120.5,
    "avg_diameter_mm": 7.5,
    "max_diameter_mm": 12.3,
    "avg_circularity": 0.85,
    "avg_solidity": 0.92,
    "avg_confidence": 0.78,
    "dice": 0.85,
    "iou": 0.74,
    "precision": 0.88,
    "recall": 0.82
  },
  "deep_features_dim": 256,
  "deep_features": [0.123, -0.456, ...],
  "output": "",
  "metadata": {
    "timestamp": "20251127_143052",
    "feature_version": "1.0"
  }
}
```

## 輸出結構

訓練結果將自動保存於 `result/segmentation_{TIMESTAMP}/` 目錄下：

- `training_*.log`: 完整訓練日誌
- `best_model.pth`: 驗證集上表現最佳的模型權重
- `training_curves.png`: 訓練過程的 Loss 與各項指標曲線圖
- `test_patient_metrics.json`: 測試集所有患者的詳細評估數據
- `test_error_cases.json`: 表現不佳 (Dice < 0.5) 的病例清單
- `features/`: 測試階段提取的病灶特徵（用於 LLM Fine-Tuning）

## 模組說明

- `main.py`: 程式入口，處理參數與流程控制。
- `trainer.py`: 訓練核心邏輯，包含模型載入、訓練迴圈、驗證與特徵提取。
  - `LesionFeatureExtractor`: 病灶特徵提取器
  - `test_and_extract_features()`: 測試並提取特徵的主要方法
- `dataset.py`: 資料集類別，處理 NIfTI 讀取與資料增強。
- `losses.py`: 定義損失函數 (Dice + BCE)。
- `utils.py`: 提供 Logging、指標計算與結果分析工具。
