# DeepLung: 3D Faster R-CNN for Lung Nodule Detection

本目錄包含針對肺結節檢測的 **3D Faster R-CNN (DeepLung)** 實作。此模型直接處理 3D CT 體積數據，而非 2D 切片，能更有效地捕捉結節的空間特徵。

## 📂 目錄結構

```
detection/deep_lung/
├── dataset.py       # 3D Dataset (讀取 .npz, 3D Augmentation)
├── model.py         # 模型定義 (ResNet3D Backbone + 3D RPN + Head)
├── preprocess.py    # 數據預處理 (DICOM/LNDb -> 3D NPZ)
├── train.py         # 訓練腳本
└── README.md        # 本文件
```

## 🛠️ 環境需求

請確保安裝了以下依賴 (專案根目錄的 `requirements.txt` 應已包含)：
- `torch`, `torchvision`
- `SimpleITK` (用於醫學影像讀取)
- `numpy`, `pandas`, `tqdm`

## 🚀 數據準備 (Data Preparation)

DeepLung 模型需要統一 Spacing (1mm x 1mm x 1mm) 的 3D `.npz` 格式數據。我們提供了一個強大的預處理腳本 `preprocess.py`。

### 1. 支援的數據來源

#### A. LNDb 資料集 (MHD + CSV)
如果您使用的是 LNDb 資料集 (如 `LNDb-0001.mhd` 和 `trainNodules_gt.csv`)：

```bash
python detection/deep_lung/preprocess.py \
    --data_root "E:\lung_ct_lesion_dataset\LNDb" \
    --dataset_type "lndb" \
    --output_dir "cache/deep_lung_cache"
```
*   **功能**: 自動讀取 CSV 中的結節座標 (世界座標 mm)，轉換為體積索引，並計算半徑。
*   **分割**: 自動將數據按 70/15/15 比例分割為 `train`, `val`, `test`。

#### B. 通用 DICOM 資料集 (Generic)
如果您有標準的 DICOM 文件夾結構 (每個病患一個資料夾，內含 XML 標註)：

```bash
python detection/deep_lung/preprocess.py \
    --data_root "datasets/my_dicom_data" \
    --dataset_type "generic" \
    --output_dir "cache/deep_lung_cache"
```

### 2. 輸出結構
預處理後，`output_dir` (預設 `cache/deep_lung_cache`) 將包含：
```
cache/deep_lung_cache/
├── train/
│   ├── LNDb-0001.npz
│   └── ...
├── val/
└── test/
```
每個 `.npz` 檔案包含：
- `image`: 歸一化後的 3D 體積 (0-1)。
- `boxes`: 3D Bounding Boxes `[z, y, x, d, h, w]`。
- `spacing`, `origin`: 原始影像屬性。

## 🏋️‍♂️ 訓練模型 (Training)

訓練腳本預設會讀取 `cache/deep_lung_cache/train` 目錄。

```bash
# 在專案根目錄運行
python -m detection.deep_lung.train
```

**參數設定**:
您可以直接修改 `train.py` 中的配置變數，或未來透過 argparse 擴充：
- `BATCH_SIZE`: 預設 2 (視 GPU 顯存調整)。
- `LR`: 學習率。
- `EPOCHS`: 訓練輪數。

## 🧠 模型架構 (Architecture)

我們實作了一個簡化版的 DeepLung 架構：
1.  **Backbone**: 3D ResNet (類似 ResNet-18 的 3D 版本/C3D)，用於提取體積特徵。
2.  **RPN (Region Proposal Network)**: 3D 卷積層，用於生成 3D 候選框 (Anchors)。
3.  **Detector Head**: 分類與回歸層，輸出最終的結節機率與邊框修正。

## 📝 引用
- [DeepLung: Deep 3D Dual Path Nets for Automated Pulmonary Nodule Detection and Classification](https://arxiv.org/abs/1801.09555)
