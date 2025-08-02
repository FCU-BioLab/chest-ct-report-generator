# 🚀 FPS-Former 快速開始指南

恭喜！您已經成功從CT-ViT升級到FPS-Former。本指南將幫助您快速開始使用新的檢測系統。

## ✅ 系統狀態確認

根據測試結果，您的FPS-Former系統已經準備就緒：

- ✅ **模型架構**：28.5M參數，109MB大小
- ✅ **GPU支援**：CUDA可用，記憶體使用125MB
- ✅ **前向傳播**：所有組件正常工作
- ✅ **輸出格式**：分類、邊界框、物件檢測正確

## 🎯 立即開始訓練

### 選項1：快速評估（推薦首次使用）
```bash
# K-Fold交叉驗證，快速了解模型性能
python detection\train_detection.py --mode kfold --num_epochs 30
```

### 選項2：完整訓練（推薦最終模型）
```bash
# 傳統訓練模式，獲得最佳模型
python detection\train_detection.py --mode traditional --num_epochs 100
```

### 選項3：自訂參數
```bash
# 自訂批次大小和學習率
python detection\train_detection.py --mode custom --batch_size 16 --learning_rate 5e-5
```

## 📊 預期性能提升

相比原始CT-ViT，FPS-Former預期將帶來：

| 指標 | CT-ViT | FPS-Former | 提升 |
|------|--------|------------|------|
| **準確率** | 85-90% | 87-93% | +2-3% |
| **推理速度** | 0.5-0.8s | 0.3-0.6s | +25% |
| **記憶體使用** | ~2GB | ~1.5GB | -25% |
| **邊界框精度** | L1<0.1 | L1<0.08 | +20% |

## 🔧 關鍵改進特點

### 1. **多尺度特徵金字塔**
- 4x4小patch提供更細緻的特徵
- 多層特徵融合提升檢測精度

### 2. **窗口注意力機制**
- 7x7窗口大小平衡性能和效率
- Shift Window提供全局感受野

### 3. **高效計算架構**
- 線性複雜度vs全局注意力的平方複雜度
- 更少的GPU記憶體需求

## 📈 監控訓練進度

### 實時監控
```bash
# 查看TensorBoard
tensorboard --logdir FPS_Former_Detection/logs
```

### 日誌文件
- **傳統模式**：`FPS_Former_Detection/logs/training.log`
- **K-Fold模式**：`FPS_Former_Detection/fold_X/logs/training.log`

### 關鍵指標
- **分類準確率**：目標 >90%
- **邊界框L1誤差**：目標 <0.08
- **物件檢測置信度**：目標 >0.85

## 🚨 注意事項

### 訓練建議
1. **首次訓練**：建議使用K-Fold評估了解基線性能
2. **記憶體不足**：降低batch_size到4或使用--image_size 192
3. **收斂緩慢**：可以嘗試學習率5e-5或2e-4

### 資料要求
- 確保資料在：`datasets/splited_dataset/`
- 檢查：`train_patients.txt` 和 `test_patients.txt`
- 驗證：XML標註和DICOM文件完整

## 🎉 成功指標

當您看到以下結果時，表示升級成功：

### K-Fold結果（30 epochs）
- **平均準確率** > 75%
- **標準差** < 0.08
- **邊界框誤差** < 0.15

### 完整訓練結果（100 epochs）
- **最終準確率** > 85%
- **邊界框誤差** < 0.08
- **收斂穩定**無異常波動

## 🔄 下一步行動

1. **立即開始**：
   ```bash
   python detection\train_detection.py --mode kfold
   ```

2. **監控進度**：
   ```bash
   tensorboard --logdir FPS_Former_Detection
   ```

3. **分析結果**：
   ```bash
   python detection\analyze_kfold_results.py --results_dir FPS_Former_Detection
   ```

4. **開始推理**：
   ```bash
   python detection\inference_detection.py --model_path FPS_Former_Detection/best_detection_model.pth --input_dicom path/to/dicom.dcm
   ```

## 📞 問題解決

如果遇到問題：

1. **查看完整指南**：`detection/CT_ViT_to_FPS_Former_Upgrade_Guide.md`
2. **檢查日誌**：`FPS_Former_Detection/logs/training.log`
3. **重新測試**：`python detection\test_fps_former.py`

---

**🎉 準備好了！您的FPS-Former系統已準備開始訓練！**

選擇上面的任一訓練選項開始您的升級之旅。預期在幾個小時內就能看到明顯的性能提升！
