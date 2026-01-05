# MedSAM2 Fine-tuning - Cache-Only Mode (2.5D)

✅ **已完成**: 2.5D 模式實作 + 簡化為僅快取模式

## 🎯 重要變更

### 1. **僅支援快取模式**
- ❌ 移除原始 LNDb 資料集直接載入
- ✅ 只使用預處理的快取資料 (`segmentation/cache`)
- ✅ 更快的訓練速度
- ✅ 統一的資料格式

### 2. **預設啟用 2.5D**
- ✅ 三個通道分別是 Z-1, Z, Z+1 切片
- ✅ 提供上下文資訊
- ✅ 預期 Dice +0.5-1%

---

## 🚀 快速開始

### 基本訓練

```bash
cd segmentation

# LNDb 資料集（2.5D 自動啟用）
python finetune_medsam2/main.py --cache_dataset_type lndb

# 完整資料集（LNDb + MSD）
python finetune_medsam2/main.py --cache_dataset_type both

# 快速測試
python finetune_medsam2/main.py --data_fraction 0.1 --epochs 5
```

### 進階訓練

```bash
# 強資料增強 + enhanced loss
python finetune_medsam2/main.py \
    --cache_dataset_type both \
    --strong_augmentation \
    --loss_type enhanced \
    --epochs 150

# 禁用 2.5D（使用傳統 2D）
python finetune_medsam2/main.py --no_2_5d
```

---

## 📋 命令列參數

### 資料參數
| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--cache_dir` | `cache` | 快取資料目錄 |
| `--cache_dataset_type` | `both` | `lndb` / `msd` / `both` |
| `--data_fraction` | `1.0` | 使用資料比例（測試用） |
| `--use_2_5d` | `True` | 使用 2.5D 輸入 |
| `--no_2_5d` | - | 禁用 2.5D |

### 訓練參數
| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--epochs` | `100` | 訓練輪數 |
| `--batch_size` | `32` | 批次大小 |
| `--lr` | `5e-6` | 學習率 |
| `--loss_type` | `combined` | 損失函數類型 |
| `--augmentation` | `False` | 啟用基本增強 |
| `--strong_augmentation` | `False` | 啟用強增強 |

---

## 📁 快取資料結構

```
segmentation/cache/
├── lndb_slices/              # LNDb 快取 (212 患者)
│   └── LNDb-XXXX/
│       ├── meta.json         # 患者 metadata
│       └── slice_XXXX.npz    # image + mask + lung_mask
└── msd_lung_slices/          # MSD 快取 (63 患者)
    └── lung_XXX/
        ├── meta.json
        └── slice_XXXX.npz
```

---

## 🔧 預處理（如果快取不存在）

如果 `cache` 目錄不存在，需要先執行預處理：

```bash
# 預處理 LNDb 資料集
python train_unetpp/preprocess.py \
    --data_dir path/to/LNDb \
    --output_dir cache/lndb_slices

# 預處理 MSD 資料集
python train_unetpp/preprocess.py \
    --data_dir path/to/MSD \
    --output_dir cache/msd_lung_slices
```

---

## 📊 2.5D 模式說明

### 輸入格式

#### 2.5D（預設）
```python
# 三個通道是相鄰切片
ct_rgb = np.stack([slice_z_minus_1, slice_z, slice_z_plus_1], axis=-1)
# Shape: [512, 512, 3]
# 通道 0: Z-1 (前一張切片)
# 通道 1: Z   (當前切片)
# 通道 2: Z+1 (後一張切片)
```

#### 2D（使用 --no_2_5d）
```python
# 三個通道都是同一張切片
ct_rgb = np.stack([slice_z, slice_z, slice_z], axis=-1)
# Shape: [512, 512, 3]
# 通道 0, 1, 2: 都是 Z
```

### 預期效果

| 指標 | 2D | 2.5D | 提升 |
|------|----|----|------|
| Dice | 0.985 | 0.990+ | +0.5% |
| 小結節檢測 | 中等 | 好 | +5-10% |
| 訓練時間 | 33 min/epoch | 35 min/epoch | +5% |

---

## 🎓 最佳實踐

### 推薦配置

```bash
python finetune_medsam2/main.py \
    --cache_dataset_type both \
    --strong_augmentation \
    --loss_type enhanced \
    --epochs 150 \
    --batch_size 16
```

預期 Dice: **0.992+** 🎯

### 對比實驗

```bash
# 實驗 1: 2D 基準
python finetune_medsam2/main.py \
    --no_2_5d \
    --output_dir result/exp_2d

# 實驗 2: 2.5D
python finetune_medsam2/main.py \
    --use_2_5d \
    --output_dir result/exp_2_5d
```

---

## 📚 相關文件

- `docs/2.5D_MODE_GUIDE.md` - 2.5D 模式詳細指南
- `docs/2.5D_IMPLEMENTATION_SUMMARY.md` - 實作總結
- `test_2_5d.py` - 測試腳本

---

## ✅ 測試

```bash
# 執行測試
python finetune_medsam2/test_2_5d.py

# 預期輸出
🎉 所有測試通過！2.5D 實作正確！
```

---

## 🔍 故障排除

### Q: 快取目錄不存在？
**A**: 執行預處理腳本生成快取資料（見上方「預處理」章節）

### Q: 找不到患者資料？
**A**: 檢查 `cache/lndb_slices` 或 `cache/msd_lung_slices` 是否存在且包含資料

### Q: 記憶體不足？
**A**: 降低 `--batch_size`（2.5D 不會增加記憶體使用）

---

## 📝 更新日誌

- **2026-01-03**: 
  - ✅ 實作 2.5D 模式（預設啟用）
  - ✅ 簡化為僅快取模式
  - ✅ 移除原始資料集載入邏輯
  - ✅ 添加詳細錯誤訊息
