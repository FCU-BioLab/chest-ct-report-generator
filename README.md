# 🏥 胸部CT報告生成系統

基於深度學習的胸部CT影像分割與醫療報告自動生成系統。

## 📋 功能概覽

| 模組 | 功能 | 技術 |
|------|------|------|
| **影像分割** | 肺結節/腫瘤分割 | UNet++ (2.5D), MedSAM2 |
| **報告生成** | 智能醫療報告 | RAG + LLM (Gemma, LLaMA) |
| **特徵提取** | 深層特徵分析 | CNN Encoder Features |

## 📁 專案結構

```
chest-ct-report-generator/
├── segmentation/              # 🔬 分割系統
│   ├── train_unetpp/          # UNet++ 分割訓練
│   │   ├── main.py            # LNDb 訓練入口
│   │   ├── train_msd.py       # MSD Lung 訓練入口
│   │   ├── trainer.py         # 訓練器 (4-patch stitch)
│   │   ├── dataset.py         # LNDb 資料集
│   │   ├── msd_dataset.py     # MSD Lung 資料集
│   │   └── README.md          # 詳細說明
│   ├── finetune_medsam2/      # MedSAM2 微調
│   ├── MedSAM2/               # MedSAM2 模型
│   └── cache/                 # 預處理快取
├── llm/                       # 🤖 LLM 報告生成
│   ├── RAG/                   # RAG 報告系統
│   │   ├── GUI.py             # 圖形介面
│   │   └── lung_rads_criteria.txt
│   └── Fine_Tune/             # LLM 微調
├── detection/                 # 特徵提取
├── dataset_process/           # 資料處理工具
│   ├── lndb_viewer.py         # LNDb 資料檢視
│   ├── msd_lung_viewer.py     # MSD Lung 資料檢視
│   └── dicom_viewer.py        # DICOM 檢視器
└── datasets/                  # 資料集目錄
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
# source venv/bin/activate  # Linux/Mac

# 安裝依賴
pip install -r requirements.txt
```

### UNet++ 分割訓練

```bash
cd segmentation

# LNDb 資料集訓練
python train_unetpp\main.py --preprocess-slices  # 預處理
python train_unetpp\main.py --epochs 100         # 訓練

# MSD Lung Tumours 資料集訓練
python train_unetpp\train_msd.py --preprocess    # 預處理
python train_unetpp\train_msd.py --epochs 100    # 訓練
```

### RAG 報告生成

```bash
cd llm/RAG
python GUI.py
```

## 🔬 分割模組詳情

### 支援的資料集

| 資料集 | 格式 | 任務 | 預處理命令 |
|--------|------|------|-----------|
| LNDb | MHD/RAW | 肺結節 | `--preprocess-slices` |
| MSD Task06 | NIfTI | 肺腫瘤 | `train_msd.py --preprocess` |

### 訓練特性

- **2.5D 輸入**: z-1, z, z+1 三切片作為 RGB
- **4-Patch 驗證**: Stitch 回 full-slice 評估
- **Lungmask 整合**: 自動肺部分割
- **正樣本 Oversampling**: 處理類別不平衡

### 訓練參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--epochs` | 100 | 訓練輪數 |
| `--batch_size` | 16 | 批次大小 |
| `--lr` | 1e-4 | 學習率 |
| `--patch_size` | 224 | Patch 大小 |
| `--encoder` | efficientnet-b4 | 編碼器 |

### 評估指標

| 指標 | 說明 |
|------|------|
| Global Dice | Full-slice 像素級 Dice |
| Boundary IoU | 邊界 IoU (d=2) |
| Lesion F1 | Slice-level 偵測 F1 |

## 📊 輸出結構

```
segmentation/result/unetpp_lndb_YYYYMMDD_HHMMSS/
├── config.json              # 訓練配置
├── data_split.json          # 資料分割
├── best_model.pth           # 最佳模型
├── history.json             # 訓練歷史
├── training_curves.png      # 訓練曲線
├── train.log                # 訓練日誌
└── validation_samples/      # 視覺化
```

## ⚙️ 系統需求

- **Python**: 3.8+
- **GPU**: NVIDIA RTX 3060+ (推薦)
- **CUDA**: 11.0+
- **RAM**: 16GB+

## 📚 技術參考

- [UNet++](https://arxiv.org/abs/1807.10165) - Nested U-Net Architecture
- [CSEA-Net](https://www.sciencedirect.com/science/article/abs/pii/S1746809423007693) - 2.5D Approach Reference
- [MedSAM2](https://github.com/bowang-lab/MedSAM) - Medical SAM
- [Lung-RADS](https://www.acr.org/Clinical-Resources/Reporting-and-Data-Systems/Lung-Rads) - 分類標準

## ⚠️ 免責聲明

本系統**僅供研究和教育用途**，不得用於實際臨床診斷。任何醫療決策應由合格醫師做出。

## 📝 授權

MIT License

---

**版本**: v2.3.0 | **更新**: 2025-12-22 | **維護**: FCU-BioLab
