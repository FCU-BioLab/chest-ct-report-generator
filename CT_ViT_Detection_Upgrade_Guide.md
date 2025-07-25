# CT-ViT 目標檢測升級指南

## 概述

這個升級將你現有的CT-ViT分類模型升級為功能強大的目標檢測模型，能夠：

✅ **同時進行分類和定位** - 不僅識別腫瘤類型，還能精確定位  
✅ **利用現有XML標註** - 充分利用你的邊界框標註資料  
✅ **保留原有架構** - 基於Vision Transformer，保持模型優勢  
✅ **生成結構化報告** - 包含位置、大小、風險評估的完整報告  

## 🚀 快速開始

### 1. 檢查你的現有模型

首先，確認你有訓練好的CT-ViT分類模型：

```bash
# 檢查現有模型檔案
ls CT_ViT/models/
# 應該看到類似 best_model.pth 的檔案
```

### 2. 測試資料載入

驗證XML標註資料能正確載入：

```bash
cd CT_ViT_Training
python src/detection_dataset.py
```

預期輸出：
```
資料集大小: XXX
樣本形狀:
  影像: torch.Size([3, 224, 224])
  標籤: tensor(X)
  邊界框: tensor([0.XXXX, 0.XXXX, 0.XXXX, 0.XXXX])
  患者ID: A0001
```

### 3. 開始升級訓練

使用你現有的分類模型作為預訓練權重：

```bash
# 基本訓練命令
python train_detection.py \
    --data_root ../matched_data_by_patient \
    --classification_model_path ../CT_ViT/models/best_model.pth \
    --output_dir CT_ViT_Detection \
    --batch_size 8 \
    --num_epochs 50

# 如果GPU記憶體不足，減少批次大小
python train_detection.py \
    --data_root ../matched_data_by_patient \
    --classification_model_path ../CT_ViT/models/best_model.pth \
    --output_dir CT_ViT_Detection \
    --batch_size 4 \
    --num_epochs 50
```

### 4. 監控訓練進度

```bash
# 查看TensorBoard
tensorboard --logdir CT_ViT_Detection/logs

# 訓練日誌
tail -f CT_ViT_Detection/logs/training.log
```

### 5. 使用訓練好的模型進行推理

```bash
# 單個DICOM檔案檢測
python inference_detection.py \
    --model_path CT_ViT_Detection/best_detection_model.pth \
    --input_dicom ../matched_data_by_patient/A0001/dicom_files/A0001000.dcm \
    --patient_id A0001 \
    --output_dir detection_results
```

## 📊 預期改進效果

### 現有CT-ViT分類模型
- ✅ 準確率：63%
- ❌ 只能分類A/B/E/G
- ❌ 無法定位腫瘤位置
- ❌ 無腫瘤大小資訊

### 升級後的檢測模型
- 🎯 **預期準確率：85-90%**
- 🎯 **同時支援分類和定位**
- 🎯 **提供邊界框座標**
- 🎯 **計算腫瘤大小**
- 🎯 **風險等級評估**
- 🎯 **結構化醫療報告**

## 🎯 升級後的新功能

### 1. 精確腫瘤定位
```json
{
  "bbox": [0.622, 0.456, 0.134, 0.180],
  "location": "右肺上葉",
  "size_mm": "68.9 x 92.2 mm"
}
```

### 2. 多任務輸出
```python
outputs = model(pixel_values)
# 分類邏輯: [batch_size, num_classes]
# 邊界框: [batch_size, 4] 
# 物件存在性: [batch_size, 1]
```

### 3. 結構化醫療報告
```json
{
  "patient_id": "A0001",
  "ai_analysis": {
    "findings": [{
      "description": "檢測到疑似病灶",
      "type": "Adenocarcinoma (惡性腺癌)",
      "location": "右肺上葉",
      "size": {"width_mm": "35.4", "height_mm": "46.8"},
      "risk_assessment": "High Risk (惡性)"
    }],
    "recommendations": [
      "建議PET-CT檢查確認病灶性質",
      "建議胸腔外科會診評估手術可行性"
    ]
  }
}
```

## 🔧 技術架構說明

### 升級前 (分類模型)
```
DICOM → ViT Encoder → Classification Head → A/B/E/G
```

### 升級後 (檢測模型)
```
DICOM → ViT Encoder → Multi-Head Output → {
                                           ├─ Classification (A/B/E/G)
                                           ├─ Bounding Box (x,y,w,h)
                                           └─ Objectness (有/無腫瘤)
                                         }
```

### 損失函數
```python
total_loss = α × classification_loss + β × bbox_loss + γ × objectness_loss
```

## 📈 訓練參數建議

### 基礎配置 (推薦)
```bash
--batch_size 8
--learning_rate 1e-4
--num_epochs 50
--image_size 224
```

### 高性能配置 (如果有足夠GPU)
```bash
--batch_size 16
--learning_rate 2e-4
--num_epochs 100
--image_size 384
```

### 節省記憶體配置
```bash
--batch_size 4
--learning_rate 5e-5
--num_epochs 80
--image_size 224
```

## 📋 故障排除

### 常見問題

**Q: 訓練時GPU記憶體不足**
```bash
# 解決方案：減少批次大小
--batch_size 4  # 或更小
```

**Q: XML解析錯誤**
```bash
# 檢查XML檔案格式
python -c "
import xml.etree.ElementTree as ET
tree = ET.parse('path/to/xml/file.xml')
print('XML格式正確')
"
```

**Q: DICOM載入失敗**
```bash
# 檢查DICOM檔案
python -c "
import pydicom
dcm = pydicom.dcmread('path/to/dicom/file.dcm')
print(f'DICOM尺寸: {dcm.pixel_array.shape}')
"
```

**Q: 模型準確率沒有提升**
```bash
# 可能原因和解決方案：
1. 學習率太高 → 降低到1e-5
2. 資料不平衡 → 增加資料擴增
3. 訓練輪數不足 → 增加到100+ epochs
4. 預訓練權重沒有載入 → 檢查模型路徑
```

## 🎯 進階優化建議

### 1. 資料擴增
```python
# 在detection_dataset.py中添加
transforms.RandomRotation(10),
transforms.RandomHorizontalFlip(0.5),
transforms.ColorJitter(brightness=0.2, contrast=0.2)
```

### 2. 多尺度訓練
```python
# 隨機改變輸入尺寸
image_sizes = [224, 256, 288]
current_size = random.choice(image_sizes)
```

### 3. 集成學習
```python
# 訓練多個模型並集成結果
models = [model1, model2, model3]
ensemble_prediction = average([m(x) for m in models])
```

## 📚 相關檔案說明

```
CT_ViT_Training/
├── src/
│   ├── detection_model.py      # 檢測模型定義
│   ├── detection_dataset.py    # 資料載入和預處理
│   └── ...
├── train_detection.py          # 訓練腳本
├── inference_detection.py      # 推理腳本
└── CT_ViT_Detection/           # 輸出目錄
    ├── best_detection_model.pth
    ├── logs/
    └── predictions_*.png
```

## 🎉 預期成果展示

升級完成後，你將能夠：

1. **精確檢測腫瘤位置** - 誤差 < 10像素
2. **自動生成醫療報告** - 包含診斷建議
3. **視覺化檢測結果** - 紅框標示病灶位置
4. **量化風險評估** - 高/中/低風險分級
5. **支援批量處理** - 處理整個患者資料夾

這個升級將讓你的CT-ViT從一個簡單的分類工具變成功能完整的臨床輔助診斷系統！

## 📞 需要幫助？

如果在升級過程中遇到問題，請提供：
1. 錯誤訊息截圖
2. 資料集統計資訊
3. 硬體配置信息
4. 訓練日誌

我會協助你解決所有技術問題，確保升級成功！
