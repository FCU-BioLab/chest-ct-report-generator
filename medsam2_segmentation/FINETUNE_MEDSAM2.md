# MedSAM2 胸部腫瘤微調指南

本文件說明 `finetune_medsam2.py` 如何準備資料、訓練、評估並保存微調後的 MedSAM2 權重，並涵蓋設定方式、產出內容與常見問題排解。

## 1. 概述

`finetune_medsam2.py` 會以 NIfTI 格式的胸腔電腦斷層 (CT) 影像及其腫瘤遮罩為輸入，建立一個端到端的微調流程，步驟如下：

1. **資料索引**：掃描病患資料夾、挑出含腫瘤的切片並建立索引。
2. **資料載入**：視需要載入 CT/腫瘤體積，正規化後切成 2D 影像，轉成 RGB 並計算外接框。
3. **訓練迴圈**：凍結 MedSAM2 影像編碼器，只優化提示編碼器與遮罩解碼器，損失為 Dice + BCE。
4. **驗證**：透過 predictor 進行推論並計算多項指標（Dice、IoU、Precision、Recall、Specificity、Hausdorff95）。
5. **檢查點與紀錄**：追蹤最佳 Dice，保存模型、學習曲線與設定檔。

## 2. 目錄與資料需求

```
medsam2_segmentation/
├── MedSAM2/                # 上游 MedSAM2 專案（已包含）
├── finetune_medsam2.py     # 訓練入口腳本
├── FINETUNE_MEDSAM2.md     # 本指南
└── ...

datasets/
└── all_patient_data/
    └── {patient_id}/
        ├── {patient_id}_CT.nii.gz
        └── {patient_id}_tumor.nii.gz
```

每個病患資料夾需包含一組 CT 影像與其腫瘤遮罩。腳本以固定比例（預設 70/15/15）將病患 ID 分成 train/val/test。

## 3. 關鍵組件

### 3.1 `ChestTumorDataset`
- 建立於指定軸向（預設 axial，`axis=2`）中含腫瘤體素的切片清單。
- 需求時再載入對應 NIfTI 影像，抽取切片、正規化為 [0, 255]，複製為 RGB，並從連通元件取得外接框。
- 可透過 `--cache_data` 將所有樣本快取於記憶體以加速重複實驗。

### 3.2 `MedSAM2Trainer`
- 透過 Hydra 設定載入 MedSAM2，凍結影像編碼器，並使用 `_prepare_image_features` 只計算一次每個切片的影像嵌入。
- 使用 `CombinedLoss`（0.5 Dice + 0.5 BCEWithLogits）作為訓練目標。
- 訓練時遍歷切片與外接框，經提示編碼器與遮罩解碼器得到 logits，插值回原始解析度並累計可微分損失。
- 驗證流程重用快取的嵌入並呼叫 predictor API，於不需梯度情況下計算各項指標。
- 保留訓練歷史以繪製損失/指標曲線並保存最佳權重。

### 3.3 日誌與雜訊抑制
- `setup_logging` 會同時輸出到 `finetune_logs/finetune_<timestamp>.log` 與終端。
- `suppress_noisy_logs` 會過濾 `sam2_image_predictor` 的冗長訊息並抑制 PyTorch SDPA（FlashAttention/CuDNN）重複警告，讓輸出更乾淨。

## 4. 命令列參數

| 參數 | 預設值 | 說明 |
| --- | --- | --- |
| `--data_dir` | `../datasets/all_patient_data` | 病患資料夾的根目錄。 |
| `--axis` | `2` | 切片軸向（0：矢狀、1：冠狀、2：軸向）。 |
| `--epochs` | `50` | 訓練輪數。 |
| `--batch_size` | `4` | DataLoader 批次大小；實際損失仍逐切片累計。 |
| `--lr` | `1e-5` | AdamW 學習率。 |
| `--weight_decay` | `1e-4` | AdamW 權重衰減。 |
| `--config` | `sam2.1_hiera_t512.yaml` | MedSAM2 設定檔（相對於 `MedSAM2/sam2/configs`）。 |
| `--checkpoint` | `MedSAM2/checkpoints/MedSAM2_latest.pt` | 預訓練權重；若提供 `--resume` 則忽略。 |
| `--resume` | `None` | 先前儲存的微調檢查點路徑。 |
| `--output_dir` | `finetune_output` | 儲存檢查點、曲線與設定的目錄。 |
| `--num_workers` | `4` | DataLoader worker 數量。 |
| `--cache_data` | flag | 將所有樣本快取於記憶體。 |
| `--eval_only` | flag | 僅執行驗證而不訓練。 |
| `--seed` | `42` | 控制隨機數的種子，確保可重現切分與訓練。 |

## 5. 常見流程

### 5.1 全新微調
```powershell
cd E:\GitHub\chest-ct-report-generator\medsam2_segmentation
python finetune_medsam2.py --epochs 50 --batch_size 8 --lr 5e-6
```

### 5.2 繼續訓練
```powershell
python finetune_medsam2.py --resume finetune_output/checkpoint_epoch_20.pth --epochs 50
```

### 5.3 僅驗證
```powershell
python finetune_medsam2.py --eval_only --checkpoint finetune_output/best_model.pth
```

## 6. 產出內容

- `finetune_output/best_model.pth`：以驗證 Dice 為準的最佳模型。
- `finetune_output/checkpoint_epoch_*.pth`：每 10 個 epoch 保存一次的檢查點。
- `finetune_output/training_curves.png`：損失與各項指標的網格圖。
- `finetune_output/training_config.json`：命令列參數與最佳 Dice，方便重現。
- `finetune_logs/`：以時間戳命名的訓練日誌，包含資料切分、各輪摘要與警告。

## 7. 疑難排解

| 症狀 | 可能原因 | 解決方案 |
| --- | --- | --- |
| `MaskDecoder.forward() missing 1 required positional argument: 'repeat_image'` | 上游 MedSAM2 需要 `repeat_image` 旗標。 | 腳本已傳入 `repeat_image=False`，請確保使用最新版。 |
| SDPA/FlashAttention 警告過多 | PyTorch 組建缺乏 FlashAttention 核心。 | 已自動抑制；若想看完整輸出可移除 `suppress_noisy_logs()` 或升級 PyTorch。 |
| 使用 GPU 仍緩慢 | 影像嵌入以切片為單位計算，batch size 主要影響 DataLoader 併發。 | 若記憶體足夠，可啟用 `--cache_data`。 |
| 缺少套件（如 `hydra`、`skimage`） | 執行環境尚未完整安裝依賴。 | 透過 `pip install -r requirements.txt` 安裝。 |
| 資料集為空 | CT 或腫瘤檔案遺失或命名不符。 | 確認 `{patient_id}_CT.nii.gz`/`{patient_id}_tumor.nii.gz` 命名與目錄結構。 |

## 8. 擴充腳本

- 在 `ChestTumorDataset` 加入自訂 transform 以實作資料增強。
- 透過調整 `CombinedLoss` 或 `compute_all_metrics` 更改損失權重或新增指標。
- 若需要即時儀表板，可利用 `train_history` 串接外部實驗追蹤工具。

在新的輔助函式與雜訊抑制工具下，腳本更易讀、維護成本更低，並能保持訓練流程的清晰與穩定。
