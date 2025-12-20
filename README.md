# 🏥 胸部CT報告生成系統

基於深度學習的胸部CT影像分割與醫療報告自動生成系統。

## 📋 功能概覽

| 模組 | 功能 | 技術 |
|------|------|------|
| **影像分割** | 肺結節/腫瘤分割 | UNet++, MedSAM2 |
| **報告生成** | 智能醫療報告 | RAG + LLM (Gemma, LLaMA) |
| **特徵提取** | 深層特徵分析 | CNN Encoder Features |

## 📁 專案結構

```
chest-ct-report-generator/
├── segmentation/              # 🔬 分割系統
│   ├── finetune_unetpp/       # UNet++ 分割訓練
│   │   ├── main.py            # 訓練入口
│   │   ├── trainer.py         # 訓練器 (pos-GT-only 指標)
│   │   ├── losses.py          # 抗崩潰損失函數
│   │   ├── dataset.py         # Patch 資料集
│   │   └── README.md          # 詳細說明
│   ├── finetune_medsam2/      # MedSAM2 微調
│   ├── MedSAM2/               # MedSAM2 模型
│   └── sam_seg.py             # 分割腳本
├── llm/                       # 🤖 LLM 報告生成
│   ├── RAG/                   # RAG 報告系統
│   │   ├── GUI.py             # 圖形介面
│   │   └── lung_rads_criteria.txt
│   └── Fine_Tune/             # LLM 微調
│       ├── llama3.2/
│       └── google_gemma_3/
├── detection/                 # 特徵提取
│   ├── deep_feature_extractor.py
│   └── feature_loader.py
├── dataset_process/           # 資料處理工具
│   ├── lndb_viewer.py         # LNDb 資料檢視
│   ├── dicom_viewer.py        # DICOM 檢視器
│   └── dataset_splitter.py    # 資料集分割
├── datasets/                  # 資料集目錄
│   └── aLL_patients_data/     # 患者資料
└── config.json                # 系統配置
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

# 基本訓練 (使用 patch mode + 資料增強)
python -m finetune_unetpp.main --use_patches --augmentation --epochs 50

# 使用抗崩潰損失 (推薦)
python -m finetune_unetpp.main --use_patches --augmentation --loss_type focal_dice --epochs 50

# 快速測試 (10% 資料)
python -m finetune_unetpp.main --data_fraction 0.1 --epochs 5 --use_patches
```

### RAG 報告生成

```bash
cd llm/RAG
python GUI.py
```

## 🔬 分割模組詳情

### 損失函數 (抗模型崩潰)

| 類型 | 描述 | 適用場景 |
|------|------|----------|
| `stable` | Weighted BCE + Dice | 預設，一般訓練 |
| `focal_dice` | Focal + Dice | 極度不平衡，小病灶 |

### 訓練參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--epochs` | 100 | 訓練輪數 |
| `--batch_size` | 8 | 批次大小 |
| `--lr` | 1e-4 | 學習率 |
| `--use_patches` | False | 使用 224x224 patch 模式 |
| `--augmentation` | False | 啟用資料增強 |
| `--loss_type` | stable | 損失函數類型 |
| `--threshold` | 0.5 | 預測二值化閾值 |

## 📊 輸出結構

```
segmentation/result/unetpp_YYYYMMDD_HHMMSS/
├── config.json              # 訓練配置
├── dataset_split.json       # 資料集分割
├── best_model.pth           # 最佳模型
├── training_history.json    # 訓練歷史
├── training_curves.png      # 訓練曲線
├── test_results.json        # 測試結果
└── visualizations/          # 分割可視化
```

## ⚙️ 系統需求

- **Python**: 3.8+
- **GPU**: NVIDIA RTX 3060+ (推薦)
- **CUDA**: 11.0+
- **RAM**: 16GB+

## 📚 技術參考

- [UNet++](https://arxiv.org/abs/1807.10165) - Nested U-Net Architecture
- [MedSAM2](https://github.com/bowang-lab/MedSAM) - Medical SAM
- [Lung-RADS](https://www.acr.org/Clinical-Resources/Reporting-and-Data-Systems/Lung-Rads) - 分類標準

## ⚠️ 免責聲明

本系統**僅供研究和教育用途**，不得用於實際臨床診斷。任何醫療決策應由合格醫師做出。

## 📝 授權

MIT License

---

**版本**: v2.2.0 | **更新**: 2025-12-20 | **維護**: FCU-BioLab
