# 🚀 MedSAM2 快速參考

## 最簡單的使用方式

```bash
cd segmentation
python finetune_medsam2/main.py
```

就這樣！會自動使用 2.5D 模式訓練。

---

## 常用命令

```bash
# 基本訓練（LNDb + MSD, 2.5D）
python finetune_medsam2/main.py

# 只用 LNDb
python finetune_medsam2/main.py --cache_dataset_type lndb

# 強資料增強
python finetune_medsam2/main.py --strong_augmentation

# 最佳配置（推薦）
python finetune_medsam2/main.py \
    --strong_augmentation \
    --loss_type enhanced \
    --epochs 150

# 快速測試
python finetune_medsam2/main.py --data_fraction 0.1 --epochs 5

# 使用 2D（禁用 2.5D）
python finetune_medsam2/main.py --no_2_5d

# 從 checkpoint 繼續
python finetune_medsam2/main.py --resume result/.../best_model.pth

# 測試模式
python finetune_medsam2/main.py --test --resume result/.../best_model.pth
```

---

## 重要參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--cache_dataset_type` | `both` | `lndb`/`msd`/`both` |
| `--epochs` | `100` | 訓練輪數 |
| `--batch_size` | `32` | 批次大小 |
| `--lr` | `5e-6` | 學習率 |
| `--loss_type` | `combined` | `combined`/`enhanced`/`tversky`/`focal` |
| `--use_2_5d` | `True` | 2.5D 模式（自動啟用） |
| `--no_2_5d` | - | 禁用 2.5D |
| `--augmentation` | `False` | 基本資料增強 |
| `--strong_augmentation` | `False` | 強資料增強 |

---

## 預期結果

| 配置 | Dice | 訓練時間/epoch |
|------|------|---------------|
| 基本 (2.5D) | 0.990 | 35 min |
| + 強增強 | 0.992 | 37 min |
| + enhanced loss | 0.993+ | 38 min |

---

## 測試

```bash
# 執行測試
python finetune_medsam2/test_2_5d.py

# 應該看到
🎉 所有測試通過！2.5D 實作正確！
```

---

## 故障排除

### 快取目錄不存在？
```bash
# 執行預處理
python train_unetpp/preprocess.py \
    --data_dir path/to/LNDb \
    --output_dir cache/lndb_slices
```

### 記憶體不足？
```bash
# 降低 batch size
python finetune_medsam2/main.py --batch_size 8
```

---

## 文件

- `README_CACHE_ONLY.md` - 完整使用指南
- `docs/2.5D_MODE_GUIDE.md` - 2.5D 詳細說明
- `IMPROVEMENTS_SUMMARY.md` - 改進總結
- `docs/LESION_STATISTICS.md` - 有病灶切片統計功能

---

## 📊 有病灶切片統計

訓練時會自動統計並保存有病灶的切片資訊：

### 輸出位置
```
result/segmentation_*/first_epoch_samples/
├── train_positive_sample_0.png  # 訓練樣本（只有病灶的）
├── train_positive_sample_1.png
├── ...
├── val_positive_sample_0.png    # 驗證樣本（只有病灶的）
├── val_positive_sample_1.png
├── ...
└── lesion_statistics.txt        # 統計報告
```

### 統計報告範例
```
訓練集:
  - 總切片數: 42,000
  - 有病灶切片數: 8,400
  - 比例: 20.00%

驗證集:
  - 總切片數: 6,000
  - 有病灶切片數: 1,200
  - 比例: 20.00%
```

詳細說明請參考 `docs/LESION_STATISTICS.md`
