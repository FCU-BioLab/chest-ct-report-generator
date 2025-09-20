# UNet++ Detection Module

UNet++ 檢測模組是一個基於 UNet++ 架構的醫學影像病灶檢測系統，專門設計用於胸部 CT 影像的病灶分割和檢測。

## 特點

### 🏗️ 架構特色
- **嵌套跳躍連接**: 實現多尺度特徵融合
- **深度監督**: 提升訓練穩定性和性能
- **端到端訓練**: 同時進行分割和檢測任務
- **多任務學習**: 結合語義分割和目標檢測

### 🔧 技術實現
- **分割分支**: 生成精確的病灶分割遮罩
- **檢測分支**: 預測邊界框和類別標籤
- **組合損失函數**: 平衡分割和檢測任務
- **數據增強**: 提升模型泛化能力

## 安裝要求

### 系統要求
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+ (推薦)

### 依賴套件
```bash
pip install torch torchvision torchaudio
pip install numpy opencv-python matplotlib
pip install scikit-learn tqdm
pip install albumentations
pip install pydicom  # 用於 DICOM 文件處理
pip install tensorboard  # 用於訓練監控
```

## 快速開始

### 1. 數據準備

組織您的數據如下結構：
```
data/
├── images/
│   ├── patient001.dcm
│   ├── patient002.dcm
│   └── ...
└── annotations/
    ├── patient001.xml
    ├── patient002.xml
    └── ...
```

XML 標註格式（Pascal VOC 格式）：
```xml
<annotation>
    <size>
        <width>512</width>
        <height>512</height>
    </size>
    <object>
        <name>A</name>  <!-- 病灶類型: A, B, E, G -->
        <bndbox>
            <xmin>100</xmin>
            <ymin>100</ymin>
            <xmax>200</xmax>
            <ymax>200</ymax>
        </bndbox>
    </object>
</annotation>
```

### 2. 訓練模型

#### 基本訓練
```bash
python -m detection.unet++_detection.train_unetpp \
    --data_dir /path/to/images \
    --xml_dir /path/to/annotations \
    --batch_size 4 \
    --num_epochs 100 \
    --learning_rate 1e-4
```

#### 使用配置文件
```bash
# 創建配置文件 config.json
{
    "batch_size": 8,
    "num_epochs": 200,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "multi_class_segmentation": false,
    "save_dir": "./checkpoints",
    "log_dir": "./logs"
}

# 使用配置文件訓練
python -m detection.unet++_detection.train_unetpp \
    --data_dir /path/to/images \
    --xml_dir /path/to/annotations \
    --config config.json
```

#### 恢復訓練
```bash
python -m detection.unet++_detection.train_unetpp \
    --data_dir /path/to/images \
    --xml_dir /path/to/annotations \
    --resume_from ./checkpoints/best_model.pth
```

### 3. 模型評估

```bash
python -m detection.unet++_detection.test_unetpp \
    --model_path ./checkpoints/best_model.pth \
    --data_dir /path/to/test/images \
    --xml_dir /path/to/test/annotations \
    --output_dir ./evaluation_results
```

### 4. 使用 Python API

```python
from detection.unet_plus_plus_detection import UNetPPDetector, UNetPPDetectionDataset
import torch

# 載入預訓練模型
model = UNetPPDetector(in_channels=1, num_classes=2)
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 創建數據集
dataset = UNetPPDetectionDataset(
    data_dir='/path/to/images',
    xml_dir='/path/to/annotations'
)

# 推理
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

with torch.no_grad():
    sample = dataset[0]
    image = sample['image'].unsqueeze(0).to(device)
    predictions = model(image)
    
    # 獲取結果
    segmentation = predictions['segmentation']
    bbox_pred = predictions['bbox_pred']
    cls_pred = predictions['cls_pred']
```

## 模型架構

### UNet++ Backbone
```
編碼器 (Encoder):
- Conv Block 00 (64 channels)
- Conv Block 10 (128 channels)
- Conv Block 20 (256 channels)
- Conv Block 30 (512 channels)
- Conv Block 40 (1024 channels)

嵌套跳躍連接:
- Level 1: X01, X11, X21, X31
- Level 2: X02, X12, X22
- Level 3: X03, X13
- Level 4: X04 (最終輸出)
```

### 檢測頭
```
分割分支:
- 深度監督輸出: [X01, X02, X03, X04]
- 最終分割遮罩: X04

檢測分支:
- 邊界框回歸頭: 預測 (x1, y1, x2, y2)
- 分類頭: 預測類別概率
```

## 配置選項

### 訓練配置
| 參數 | 預設值 | 說明 |
|------|--------|------|
| `batch_size` | 4 | 批次大小 |
| `num_epochs` | 100 | 訓練輪數 |
| `learning_rate` | 1e-4 | 學習率 |
| `weight_decay` | 1e-5 | 權重衰減 |
| `num_workers` | 4 | 數據載入線程數 |
| `multi_class_segmentation` | false | 是否使用多類別分割 |

### 模型配置
| 參數 | 預設值 | 說明 |
|------|--------|------|
| `in_channels` | 1 | 輸入通道數 |
| `num_classes` | 2 | 檢測類別數 |
| `segmentation_classes` | 1 | 分割類別數 |
| `feature_scale` | 4 | 特徵縮放因子 |
| `num_anchors` | 9 | 錨點數量 |

### 損失函數配置
| 參數 | 預設值 | 說明 |
|------|--------|------|
| `seg_weight` | 1.0 | 分割損失權重 |
| `det_weight` | 1.0 | 檢測損失權重 |
| `classification_weight` | 1.0 | 分類損失權重 |
| `use_focal_loss` | true | 是否使用 Focal Loss |

## 評估指標

### 分割指標
- **Dice Coefficient**: 分割重疊度
- **IoU (Intersection over Union)**: 交集比聯集
- **Pixel Accuracy**: 像素準確率
- **Sensitivity (Recall)**: 敏感性
- **Specificity**: 特異性

### 檢測指標
- **mAP (mean Average Precision)**: 平均精度均值
- **AP@0.5**: IoU 閾值 0.5 時的平均精度
- **AP@0.75**: IoU 閾值 0.75 時的平均精度
- **Precision**: 精確率
- **Recall**: 召回率

## 輸出格式

### 訓練輸出
```
checkpoints/
├── best_model.pth          # 最佳模型
├── latest_checkpoint.pth   # 最新檢查點
└── checkpoint_epoch_X.pth  # 定期保存

logs/
└── tensorboard logs        # TensorBoard 日誌
```

### 評估輸出
```
evaluation_results/
├── evaluation_results.json # 評估指標
├── prediction_0.png        # 預測可視化
├── prediction_1.png
└── ...
```

## 進階功能

### 1. 多類別分割
啟用多類別分割以區分不同類型的病灶：
```python
model = UNetPPDetector(
    segmentation_classes=5,  # 背景 + 4種病灶類型
    multi_class_segmentation=True
)
```

### 2. 自定義數據增強
```python
import albumentations as A

custom_transform = A.Compose([
    A.Resize(512, 512),
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussNoise(p=0.3),
    A.Normalize(mean=[0.485], std=[0.229]),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
```

### 3. 學習率調度
```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=10,
    verbose=True
)
```

### 4. 模型集成
```python
# 載入多個模型進行集成
models = []
for model_path in model_paths:
    model = UNetPPDetector()
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    models.append(model)

# 集成預測
ensemble_predictions = ensemble_predict(models, input_image)
```

## 故障排除

### 常見問題

1. **CUDA 內存不足**
   ```bash
   # 解決方案：減小批次大小
   --batch_size 2
   ```

2. **數據載入錯誤**
   ```bash
   # 檢查數據路徑和格式
   --num_workers 0  # 禁用多進程
   ```

3. **訓練不收斂**
   ```bash
   # 調整學習率
   --learning_rate 1e-5
   ```

4. **分割結果不佳**
   - 增加分割損失權重：`seg_weight=2.0`
   - 使用更多數據增強
   - 增加訓練輪數

### 性能優化

1. **混合精度訓練**
   ```python
   from torch.cuda.amp import GradScaler, autocast
   
   scaler = GradScaler()
   
   with autocast():
       predictions = model(images)
       loss = criterion(predictions, targets)
   
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

2. **數據載入優化**
   ```python
   # 使用更多工作線程
   DataLoader(dataset, num_workers=8, pin_memory=True)
   ```

3. **模型剪枝**
   ```python
   import torch.nn.utils.prune as prune
   
   # 剪枝卷積層
   prune.l1_unstructured(model.backbone.conv00, name='weight', amount=0.2)
   ```

## 版本歷史

- **v1.0.0** (2025-09-18)
  - 初始版本發布
  - 實現基本的 UNet++ 檢測功能
  - 支援分割和檢測的端到端訓練
  - 包含完整的訓練和評估工具

## 貢獻指南

歡迎貢獻代碼！請遵循以下步驟：

1. Fork 本項目
2. 創建功能分支：`git checkout -b feature/new-feature`
3. 提交更改：`git commit -m 'Add new feature'`
4. 推送分支：`git push origin feature/new-feature`
5. 創建 Pull Request

## 授權協議

本項目採用 MIT 授權協議 - 詳見 [LICENSE](LICENSE) 文件。

## 聯繫方式

- 作者: GitHub Copilot
- 電子郵件: copilot@github.com
- 項目地址: https://github.com/FCU-BioLab/chest-ct-report-generator

## 致謝

- 感謝 UNet++ 原始論文作者的開創性工作
- 感謝 PyTorch 社區提供的優秀框架
- 感謝所有貢獻者的支持和反饋