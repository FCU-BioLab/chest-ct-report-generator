# MedSAM Segmentation with Spatial Alignment

整合並最佳化的 MedSAM 胸腔腫瘤分割系統，具備完整的空間對齊驗證功能，確保分割結果在 3D Slicer 中完美對齊。

## ✨ 主要特色

- � **智能分割**: 基於 MedSAM 的高精度腫瘤分割
- � **空間對齊**: 完整的 DICOM 到 NIfTI 座標系統轉換
- ✅ **自動驗證**: 內建空間對齊驗證機制
- � **3D Slicer 相容**: 確保分割結果在 3D Slicer 中完美顯示
- � **一站式工具**: 整合分割、驗證、測試於單一程式
- � **模式切換**: PyTorch 不可用時自動切換到模擬模式

## 📁 檔案結構

```
CT_ViT_Training/
├── medsam_seg.py               # 🎯 整合的主程式 (新版本)
├── medsam_demo.py              # 📋 原始版本 (已最佳化)
├── test_spatial_alignment.py   # 🔧 空間對齊測試工具
├── verify_alignment.py         # ✅ NIfTI 對齊驗證工具
└── README_MedSAM.md           # 📖 本說明文件
```

## 🛠️ 環境安裝

### 1. 基本依賴
```bash
# 安裝核心依賴
pip install numpy nibabel pydicom opencv-python matplotlib

# 安裝 MedSAM 相關依賴 (可選，無則使用模擬模式)
pip install torch transformers
```

### 2. 數據目錄結構
```
all_patient_data/
├── A0001/
│   ├── dicom/                 # DICOM 檔案目錄
│   │   ├── slice001.dcm
│   │   └── slice002.dcm
│   └── xml/                   # XML 標註目錄
│       ├── annotation001.xml
│       └── annotation002.xml
├── A0002/
└── ...
```

## 🚀 使用方法

### 基本命令

```bash
# 列出所有可用病例
python medsam_seg.py --list_patients

# 處理單一病例 (預設行為)
python medsam_seg.py --patient_id A0001

# 完整的空間對齊測試
python medsam_seg.py --test_alignment --patient_id A0001

# 只創建 DICOM 參考 NIfTI
python medsam_seg.py --create_reference --patient_id A0001

# 驗證兩個 NIfTI 檔案的空間對齊
python medsam_seg.py --verify_alignment reference.nii.gz segmentation.nii.gz
```

### 進階選項

```bash
# 指定數據目錄
python medsam_seg.py --data_dir /path/to/patient/data --patient_id A0001

# 完整工作流程示例
python medsam_seg.py --test_alignment --patient_id A0001 --data_dir all_patient_data
```

## 📊 核心功能

### 1. MedSAMSegmentator 類別

主要的分割處理類別，包含：

- **`load_patient_data()`**: 載入 DICOM 和 XML 檔案
- **`segment_with_medsam()`**: 執行 MedSAM 分割 (支援模擬模式)
- **`create_3d_mask_volume()`**: 創建空間對齊的 3D 分割體積
- **`save_masks_as_nifti()`**: 儲存為 3D Slicer 相容的 NIfTI 格式
- **`create_reference_nifti()`**: 創建 DICOM 參考 NIfTI 檔案

### 2. AlignmentVerifier 類別

專門的空間對齊驗證工具：

- **`verify_alignment()`**: 比較兩個 NIfTI 檔案的空間屬性
- **`print_verification_results()`**: 輸出易讀的驗證報告

### 3. test_spatial_alignment() 函數

整合的測試工作流程：
1. 創建 DICOM 參考 NIfTI
2. 執行 MedSAM 分割
3. 自動驗證空間對齊
4. 提供 3D Slicer 使用指導

## 📈 輸出結果

### 1. 分割結果

```
segmentation_result/
├── A0001/
│   ├── segmentation_20250729_120000.nii.gz    # 分割遮罩
│   ├── reference_20250729_120000.nii.gz       # DICOM 參考
│   └── medsam_seg.log                          # 處理日誌
└── ...
```

### 2. 空間對齊驗證報告

```
Spatial Alignment Verification
==================================================
Overall aligned: ✓ YES
Shapes match: ✓
Affines match: ✓
Spacing match: ✓
Origin distance: 0.000 mm
Reference shape: (512, 512, 20)
Segmentation shape: (512, 512, 20)

🎉 Files should align perfectly in 3D Slicer!
```

## 🔧 技術特點

### 空間對齊修復

- **座標系統轉換**: 正確處理 DICOM RAS+ 到 NIfTI LPS+ 轉換
- **仿射矩陣計算**: 精確的空間變換矩陣
- **體素排序**: 按空間位置正確排序切片
- **體積轉置**: 正確的軸向轉換 (z,y,x) → (x,y,z)

### 智能模式切換

```python
# 自動檢測 PyTorch 可用性
if TORCH_AVAILABLE:
    # 使用真實的 MedSAM 模型
    masks = segment_with_medsam(image, bboxes)
else:
    # 自動切換到模擬模式
    masks = generate_mock_masks(image, bboxes)
```

### 強健的錯誤處理

- 完整的異常捕獲和日誌記錄
- 優雅的降級處理
- 詳細的錯誤信息和解決建議

## 📋 完整工作流程示例

```bash
# 1. 查看可用病例
python medsam_seg.py --list_patients

# 2. 執行完整的空間對齊測試
python medsam_seg.py --test_alignment --patient_id A0001

# 3. 在 3D Slicer 中驗證結果
# 載入 segmentation_result/A0001/ 中的兩個 .nii.gz 檔案
# 它們應該完美對齊
```

輸出範例：
```
Testing spatial alignment for patient: A0001
============================================================

1. Creating DICOM reference NIfTI...
   Reference created: reference_20250729_120000.nii.gz

2. Processing segmentation...
   Segmentation created: segmentation_20250729_120000.nii.gz
   Processed 15 slices

3. Verifying spatial alignment...
Spatial Alignment Verification
==================================================
Overall aligned: ✓ YES
Shapes match: ✓
Affines match: ✓
Spacing match: ✓
Origin distance: 0.000 mm

🎉 Files should align perfectly in 3D Slicer!

3D Slicer Instructions:
1. Load both NIfTI files
2. Use 'Volumes' module to overlay them
3. Adjust opacity to see alignment
```

## 🆚 版本對比

| 功能 | 舊版本 (medsam_demo.py) | 新版本 (medsam_seg.py) |
|------|----------------------|----------------------|
| 程式碼行數 | ~1400 行 | ~660 行 |
| 檔案數量 | 3 個分散檔案 | 1 個整合檔案 |
| 空間對齊 | ✅ 已修復 | ✅ 完整整合 |
| 對齊驗證 | 需要分別執行 | ✅ 內建自動驗證 |
| 測試工具 | 分散在多個檔案 | ✅ 統一介面 |
| 模擬模式 | 基本支援 | ✅ 完整的降級機制 |

## 🔍 故障排除

### 常見問題

1. **找不到病例數據**
   ```bash
   # 檢查數據目錄結構是否正確
   python medsam_seg.py --list_patients --data_dir your_data_path
   ```

2. **PyTorch 相關錯誤**
   - 程式會自動切換到模擬模式，不影響空間對齊測試

3. **空間對齊驗證失敗**
   ```bash
   # 使用驗證工具詳細檢查
   python medsam_seg.py --verify_alignment ref.nii.gz seg.nii.gz
   ```

### 調試選項

```python
# 在程式中啟用詳細日誌
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## 📄 授權

本專案遵循原專案授權條款。MedSAM 模型遵循 Apache 2.0 授權。

---

**💡 提示**: 新的 `medsam_seg.py` 是推薦使用的版本，它整合了所有功能並提供更好的使用體驗！

## 🔍 故障排除

### 常見問題

1. **找不到病例數據**
   ```bash
   # 檢查數據目錄結構是否正確
   python medsam_seg.py --list_patients --data_dir your_data_path
   ```

2. **PyTorch 相關錯誤**
   - 程式會自動切換到模擬模式，不影響空間對齊測試

3. **空間對齊驗證失敗**
   ```bash
   # 使用驗證工具詳細檢查
   python medsam_seg.py --verify_alignment ref.nii.gz seg.nii.gz
   ```

### 調試選項

```python
# 在程式中啟用詳細日誌
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## 📄 授權

本專案遵循原專案授權條款。MedSAM 模型遵循 Apache 2.0 授權。

---

**💡 提示**: 新的 `medsam_seg.py` 是推薦使用的版本，它整合了所有功能並提供更好的使用體驗！
