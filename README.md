# 🏥 胸部CT報告生成系統

基於深度學習的胸部CT影像分割、結節偵測與醫療報告自動生成系統。

## 📋 功能概覽

| 模組 | 功能 | 技術 |
|------|------|------|
| **影像分割** | 肺結節/腫瘤分割 | UNet++ (2.5D), MedSAM2 |
| **結節偵測** | 肺結節 3D 偵測 | 3D U-Net (SE Block + Attention Gate) |
| **報告生成** | 智能醫療報告 | RAG + LLM (Gemma, LLaMA) |
| **資料處理** | DICOM 處理與視覺化 | 完整資料 pipeline (Splitter, Viewer, Preprocessor) |

## 📁 專案結構

```
chest-ct-report-generator/
├── segmentation/              # 🔬 分割系統
│   ├── train_unetpp/          # UNet++ 分割訓練 (2.5D)
│   ├── finetune_medsam2/      # MedSAM2 微調
│   └── MedSAM2/               # MedSAM2 模型核心
├── detection/                 # 🔍 偵測系統
│   └── train_3dunet/          # 3D U-Net 結節偵測
│       ├── model.py           # 3D U-Net 模型定義
│       ├── detector.py        # 結節偵測與後處理
│       └── README.md          # 偵測模組詳細說明
├── llm/                       # 🤖 LLM 報告生成
│   ├── RAG/                   # RAG 報告生成系統 (GUI)
│   └── Fine_Tune/             # LLM 微調腳本
├── dataset_process/           # 🛠️ 資料處理工具集
│   ├── preprocess_original_dataset.py  # DICOM 轉檔預處理
│   ├── dataset_splitter.py             # 資料集分割 (K-Fold)
│   ├── visualize_patient_yolo.py       # YOLO/標註視覺化
│   ├── dicom_viewer.py                 # DICOM 3D 檢視器
│   └── README.md                       # 工具集說明
└── datasets/                  # 資料集目錄 (需自行建立)
```

## 🚀 快速開始

### 環境設置

```bash
# 克隆專案
git clone https://github.com/FCU-BioLab/chest-ct-report-generator.git
cd chest-ct-report-generator

# 創建虛擬環境
python -m venv venv
venv\Scripts\activate  # Windows

# 安裝依賴
pip install -r requirements.txt
```

### 1. 資料處理 (Dataset Process)

使用 `dataset_process` 工具將原始 DICOM 轉換為訓練格式：

```bash
cd dataset_process

# 1. 整理與分析資料
python dataset_analysis.py

# 2. 預處理 DICOM -> PNG/YOLO 格式
python preprocess_original_dataset.py

# 3. 分割訓練/測試集
python dataset_splitter.py --source_dir ../datasets/preprocessed --output_dir ../datasets/splits
```

### 2. UNet++ 分割訓練 (Segmentation)

```bash
cd segmentation

# LNDb 資料集訓練
python train_unetpp\main.py --preprocess-slices  # 預處理
python train_unetpp\main.py --epochs 100         # 訓練

# MSD Lung Tumours 資料集訓練
python train_unetpp\main.py --dataset msd --preprocess
python train_unetpp\main.py --dataset msd --epochs 100
```

### 3. 3D U-Net 結節偵測 (Detection)

```bash
# 1. 數據轉換 (DICOM -> NPZ Volume)
python -m detection.retinanet.prepare_data --dataset lndb --base_dir "cache/LNDb" --output "dataset_lndb.json"

# 2. 訓練偵測模型
python -m detection.retinanet.main train --data_path "dataset_lndb.json" --epochs 300 --output_dir "results/experiment_1"

# 3. 推論與生成報告
python -m detection.retinanet.inference --input_path "data/patient_01" --model_path "results/experiment_1/model_best.pt" --output_dir "results/patient_01"
```

### 4. RAG 報告生成 (LLM)

```bash
cd llm/RAG
python GUI.py
```

## 🔬 核心模組詳情

### 1. 分割模組 (Segmentation)

- **架構**: UNet++ (EfficientNet-B4 Encoder) 採用 2.5D 輸入 (z-1, z, z+1)。
- **困難負樣本挖掘 (Hard Negative Mining)**:
    - 訓練採樣策略：70% 正樣本 (結節中心) + 20% 困難負樣本 (高假陽性區域) + 10% 隨機背景。
    - 有效降低假陽性率 (False Positives)。
- **MedSAM2**: 整合最新的 Medical SAM 2 進行互動式分割微調。

### 2. 偵測模組 (Detection)

- **架構**: 3D U-Net 結合 **SE Block** (Squeeze-and-Excitation) 與 **Attention Gate**。
- **功能**:
    - **全體積輸入 (Full Volume)**: 支援大尺寸 CT 體積輸入。
    - **結節分析**: 自動計算結節體積 (mm³)、直徑 (mm) 與肺葉位置。
    - **結構化報告**: 輸出 JSON 格式的偵測報告與 GIF 視覺化結果。

### 3. 資料處理工具 (Dataset Process)

提供完整的醫學影像處理工具鏈：
- **Viewer**: `dicom_viewer.py` (3D DICOM 瀏覽), `visualize_patient_yolo.py` (標註檢查)。
- **Preprocessor**: 自動窗位調整 (Windowing)、CLAHE 增強、轉檔 (NIfTI/PNG)。
- **Splitter**: 支援分層隨機抽樣 (Stratified Sampling) 的資料集分割工具。

## 📊 輸出範例

### 分割與偵測結果
- **3D 視覺化**: 結節 Overlay 的 GIF 動畫。
- **指標報告**: Dice, IoU, F1-Score, Recall, Precision。

### 醫療報告 (RAG)
利用 LLM 結合偵測到的結節特徵，生成符合 Lung-RADS 標準的醫療報告草稿。

## ⚙️ 系統需求

- **OS**: Windows / Linux
- **Python**: 3.8+
- **GPU**: NVIDIA RTX 3060+ (建議 12GB+ VRAM 以進行 3D 訓練)
- **RAM**: 32GB+ (建議)

## 📚 技術參考

- [UNet++](https://arxiv.org/abs/1807.10165)
- [MedSAM](https://github.com/bowang-lab/MedSAM)
- [Lung-RADS Assessment Categories](https://www.acr.org/Clinical-Resources/Reporting-and-Data-Systems/Lung-Rads)

## ⚠️ 免責聲明

本系統**僅供研究和教育用途**，不得用於實際臨床診斷。任何醫療決策應由合格醫師做出。

## 📝 授權

MIT License

---

**維護**: FCU-BioLab
