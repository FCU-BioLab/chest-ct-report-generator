# YOLOv7 Medical Architecture - Visual Guide

## 🏗️ 整體架構流程圖

```
Input Image (640x640)
        ↓
  [Medical Preprocessing]
  - HU Windowing (optional)
  - CLAHE Enhancement
  - Percentile Stretch
        ↓
    Backbone
        ↓
   ┌────────────┐
   │  Conv Stem │
   └────────────┘
        ↓
   ┌────────────┐
   │   ELAN 1   │ → [CBAM] ──→ P2 features
   └────────────┘
        ↓
   ┌────────────┐
   │   ELAN 2   │ → [CBAM] ──→ P3 features
   └────────────┘
        ↓
   ┌────────────┐
   │   ELAN 3   │ → [CBAM] → [Swin Transformer] ──→ P4 features
   └────────────┘
        ↓
   ┌────────────┐
   │   ELAN 4   │ → [CBAM] ──→ P5 features
   └────────────┘
        ↓
      Neck
        ↓
   ┌────────────┐
   │   BiFPN    │ (replaces PAN)
   │  3 layers  │
   └────────────┘
     ↓  ↓  ↓
   P3  P4  P5  (fused features)
     ↓  ↓  ↓
   [SimAM] × 3
     ↓  ↓  ↓
  Detection Heads
     ↓  ↓  ↓
  Predictions
```

---

## 🔍 模組詳細插入位置

### Backbone with Medical Modules

```
Layer Index │ Module              │ Output Channels │ Medical Module
────────────┼────────────────────┼────────────────┼─────────────────
0-2         │ Stem (Conv)        │ 64             │ -
3-11        │ ELAN Stage 1       │ 256            │ -
12          │ CBAM               │ 256            │ ✓ CBAM
13-25       │ ELAN Stage 2       │ 512            │ -
26          │ CBAM               │ 512            │ ✓ CBAM
27-39       │ ELAN Stage 3       │ 1024           │ -
40          │ CBAM               │ 1024           │ ✓ CBAM
41          │ Swin Transformer   │ 1024           │ ✓ Swin
42-54       │ ELAN Stage 4       │ 1024           │ -
55          │ CBAM               │ 1024           │ ✓ CBAM
```

### Neck with BiFPN

```
Input:  P3 (512), P4 (1024), P5 (1024)
         ↓
    Feature Projection
         ↓
    P3 (256), P4 (256), P5 (256)
         ↓
    ┌─────────────────────────┐
    │  BiFPN Layer 1          │
    │  - Top-down pathway     │
    │  - Bottom-up pathway    │
    │  - Weighted fusion      │
    └─────────────────────────┘
         ↓
    ┌─────────────────────────┐
    │  BiFPN Layer 2          │
    └─────────────────────────┘
         ↓
    ┌─────────────────────────┐
    │  BiFPN Layer 3          │
    └─────────────────────────┘
         ↓
Output: P3' (256), P4' (256), P5' (256)
```

### Detection Heads with SimAM

```
P3' (256) ──→ [SimAM] ──→ Conv ──→ Detect (80×80 grid)
P4' (256) ──→ [SimAM] ──→ Conv ──→ Detect (40×40 grid)
P5' (256) ──→ [SimAM] ──→ Conv ──→ Detect (20×20 grid)
```

---

## 📊 模組參數分布

```
Total Parameters: ~42M
├─ YOLOv7 Baseline: ~37M (88%)
└─ Medical Modules: ~5M (12%)
    ├─ CBAM (4×): ~1.5M (3.6%)
    │   ├─ Channel Attention
    │   └─ Spatial Attention
    ├─ Swin Transformer: ~2M (4.8%)
    │   ├─ Window Attention
    │   └─ MLP
    ├─ BiFPN (3 layers): ~1M (2.4%)
    │   ├─ Depthwise Separable Conv
    │   └─ Learnable Weights
    └─ SimAM (3×): 0 (0%)
        └─ Parameter-free
```

---

## 🔄 資料流程

### Training Pipeline

```
1. Dataset Loading
   ├─ CTDetectionDataset
   ├─ Patient IDs filtering
   └─ Negative samples handling
        ↓
2. Medical Preprocessing
   ├─ HU Windowing (-600 ± 750)
   ├─ CLAHE (clip=2.0, tile=8)
   └─ Resize to 640×640
        ↓
3. Batch Formation
   ├─ Collate with batch indices
   └─ Normalize to [0, 1]
        ↓
4. Model Forward
   ├─ Backbone (with CBAM, Swin)
   ├─ Neck (BiFPN)
   └─ Head (with SimAM)
        ↓
5. Loss Computation
   ├─ Box Loss (CIoU)
   ├─ Object Loss (BCE)
   └─ Class Loss (BCE)
        ↓
6. Backward & Update
   ├─ Gradient clipping
   ├─ Optimizer step
   └─ EMA update
```

### Inference Pipeline

```
Input CT Image
     ↓
Medical Preprocessing
     ↓
Model Forward (best.pt)
     ↓
NMS (IoU threshold)
     ↓
Post-processing
     ↓
Detections [x, y, w, h, conf, cls]
```

---

## 🎯 醫學模組功能說明

### CBAM (Convolutional Block Attention Module)
```
Purpose: 強化重要特徵，抑制不重要特徵
Location: 每個 ELAN stage 後
Mechanism:
    Channel Attention (全域池化 + FC)
         ↓
    Spatial Attention (空間池化 + Conv)
         ↓
    Element-wise multiplication
```

### Swin Transformer Block
```
Purpose: 捕捉長距離依賴關係
Location: Backbone 倒數第二個 stage
Mechanism:
    Window Partition (7×7)
         ↓
    Multi-head Self-Attention
         ↓
    Window Merge
         ↓
    MLP (Feed-forward)
```

### BiFPN (Bidirectional Feature Pyramid Network)
```
Purpose: 多尺度特徵融合
Location: Neck (取代 PAN)
Mechanism:
    Top-down Pathway
         ↓
    Bottom-up Pathway
         ↓
    Weighted Feature Fusion
```

### SimAM (Simple, Parameter-Free Attention)
```
Purpose: 輕量化注意力機制
Location: Detection heads 前
Mechanism:
    Energy Function (無參數)
         ↓
    Sigmoid Activation
         ↓
    Element-wise multiplication
```

---

## 📈 效能預期

### 計算複雜度 (FLOPs)

```
Component           │ FLOPs    │ Percentage
────────────────────┼──────────┼───────────
Baseline Backbone   │ ~100G    │ 70%
Medical Modules     │ ~15G     │ 10%
Neck (BiFPN)        │ ~20G     │ 15%
Detection Heads     │ ~5G      │ 5%
────────────────────┼──────────┼───────────
Total               │ ~140G    │ 100%
```

### 記憶體使用 (估計)

```
Batch Size │ Image Size │ GPU Memory (with AMP)
───────────┼────────────┼──────────────────────
8          │ 640        │ ~6 GB
16         │ 640        │ ~10 GB
32         │ 640        │ ~18 GB
8          │ 1024       │ ~12 GB
```

---

## 🔧 模組開關對照表

```
Configuration       │ Baseline │ Medical │ Difference
────────────────────┼──────────┼─────────┼───────────
CBAM                │ ✗        │ ✓       │ +1.5M
Swin Transformer    │ ✗        │ ✓       │ +2.0M
BiFPN               │ ✗        │ ✓       │ +1.0M
SimAM               │ ✗        │ ✓       │ +0M
────────────────────┼──────────┼─────────┼───────────
Total Params        │ ~37M     │ ~42M    │ +5M (12%)
Training Time       │ 1.0x     │ 1.2x    │ +20%
Inference Time      │ 1.0x     │ 1.15x   │ +15%
mAP (expected)      │ baseline │ +2-5%   │ improvement
```

---

## 🎨 視覺化範例

### CBAM Attention Maps (示意)

```
Original Feature Map    Channel Attention      Spatial Attention
┌────────────────┐      ┌────────────────┐     ┌────────────────┐
│ ▓▓▓▓░░░░▓▓▓▓  │      │ ████░░░░████   │     │ ▓▓▓▓▓▓▓▓░░░░  │
│ ▓▓▓▓░░░░▓▓▓▓  │  →   │ ████░░░░████   │  →  │ ▓▓▓▓▓▓▓▓░░░░  │
│ ░░░░▓▓▓▓░░░░  │      │ ░░░░████░░░░   │     │ ░░░░▓▓▓▓▓▓▓▓  │
│ ░░░░▓▓▓▓░░░░  │      │ ░░░░████░░░░   │     │ ░░░░▓▓▓▓▓▓▓▓  │
└────────────────┘      └────────────────┘     └────────────────┘
  (Input)                 (Channel weights)      (Spatial weights)
                                ↓
                         Enhanced Feature Map
                        ┌────────────────┐
                        │ ████████░░░░   │
                        │ ████████░░░░   │
                        │ ░░░░████████   │
                        │ ░░░░████████   │
                        └────────────────┘
```

### BiFPN Fusion (示意)

```
      P5 (20×20)
        ↓  ↑
        ↓  ↑ (weighted)
      P4 (40×40)
        ↓  ↑
        ↓  ↑ (weighted)
      P3 (80×80)
        
Top-down: ↓ (upsample + add)
Bottom-up: ↑ (downsample + add)
```

---

## 📚 配置檔案對照

### yolov7_medical.yaml (with modules)
```yaml
backbone:
  - [layers...]
  - CBAM        # ← 插入點 1
  - [layers...]
  - CBAM        # ← 插入點 2
  - [layers...]
  - CBAM        # ← 插入點 3
  - SwinBlock   # ← 插入點 4
  - [layers...]
  - CBAM        # ← 插入點 5

head:
  - BiFPN       # ← 取代 PAN
  - SimAM       # ← 插入點 6
  - SimAM       # ← 插入點 7
  - SimAM       # ← 插入點 8
  - Detect
```

### yolov7_baseline.yaml (without modules)
```yaml
backbone:
  - [standard YOLOv7 layers]

head:
  - [standard PAN neck]
  - [standard detection heads]
```

---

## 🎓 最佳實踐建議

### 1. 訓練策略
```
Small Dataset (< 500):
  - epochs: 200
  - batch_size: 8
  - imgsz: 512
  - use_medical_modules: 1

Large Dataset (> 5000):
  - epochs: 100
  - batch_size: 32
  - imgsz: 640
  - use_medical_modules: 1

High Resolution:
  - epochs: 150
  - batch_size: 8
  - imgsz: 1024
  - use_medical_modules: 1
```

### 2. 醫學預處理
```
Lung Window:
  - window_center: -600
  - window_width: 1500

Mediastinum Window:
  - window_center: 40
  - window_width: 400

Bone Window:
  - window_center: 300
  - window_width: 1500
```

### 3. 實驗對照
```
Step 1: Baseline
  --use_medical_modules 0

Step 2: Individual Modules
  - Only CBAM
  - Only Swin
  - Only BiFPN
  - Only SimAM

Step 3: Full Medical
  --use_medical_modules 1

Step 4: Analysis
  - Compare mAP
  - Compare training time
  - Compare inference speed
```

---

**提示**: 這份視覺化指南可以幫助你理解整個架構的運作方式。建議搭配 `README_YOLOV7.md` 一起閱讀。
