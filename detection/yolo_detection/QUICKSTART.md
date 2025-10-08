# YOLOv7 Medical Training - Quick Reference Card

## 🚀 Quick Start (3 Steps)

### 1. Setup & Verify
```bash
cd detection/yolo_detection
python setup_yolov7.py
```

### 2. Train with Medical Modules (Recommended)
```bash
python train_yolov7_medical.py \
    --data_dir ./datasets/ct_data \
    --epochs 120 \
    --batch_size 16
```

### 3. Train Baseline (for Comparison)
```bash
python train_yolov7_medical.py \
    --data_dir ./datasets/ct_data \
    --epochs 120 \
    --use_medical_modules 0 \
    --model_config models/yolov7_baseline.yaml
```

---

## 📊 Common Configurations

### 小數據集 (< 500 樣本)
```bash
python train_yolov7_medical.py \
    --data_dir ./datasets/ct_data \
    --epochs 200 \
    --batch_size 8 \
    --imgsz 512 \
    --lr 0.0005
```

### 大數據集 (> 5000 樣本)
```bash
python train_yolov7_medical.py \
    --data_dir ./datasets/ct_data \
    --epochs 100 \
    --batch_size 32 \
    --imgsz 640 \
    --lr 0.001 \
    --device 0,1,2,3  # 多 GPU
```

### 高解析度訓練 (小病灶偵測)
```bash
python train_yolov7_medical.py \
    --data_dir ./datasets/ct_data \
    --epochs 150 \
    --batch_size 8 \
    --imgsz 1024 \
    --lr 0.0005
```

### 快速實驗 (Debug)
```bash
python train_yolov7_medical.py \
    --data_dir ./datasets/ct_data \
    --epochs 10 \
    --batch_size 4 \
    --imgsz 320
```

---

## 🔧 常用參數組合

### 最佳效能配置
```bash
--use_medical_modules 1
--enable_hu_windowing 1
--window_center -600
--window_width 1500
--enable_clahe 1
--use_ema
--mixed_precision
--multi_scale
```

### CPU 訓練 (測試用)
```bash
--device cpu
--batch_size 2
--imgsz 320
--workers 2
--use_ema 0
--mixed_precision 0
```

### 節省記憶體
```bash
--batch_size 4
--imgsz 512
--gradient_clip 5.0
--workers 2
```

---

## 📁 輸出結構

```
yolov7_models/run_20250108_123456/
├── weights/
│   ├── best.pt          ← 使用這個進行推論
│   ├── last.pt
│   └── epoch_*.pt
├── training_history.json
└── summary.json
```

---

## 🎯 醫學預處理參數

### 肺窗 (Lung Window)
```bash
--window_center -600
--window_width 1500
```

### 縱膈窗 (Mediastinum Window)
```bash
--window_center 40
--window_width 400
```

### 骨窗 (Bone Window)
```bash
--window_center 300
--window_width 1500
```

### 關閉 HU 視窗化 (資料已預處理)
```bash
--enable_hu_windowing 0
--enable_clahe 1
```

---

## 🐛 故障排除

### CUDA OOM (記憶體不足)
```bash
# 減少 batch size
--batch_size 4

# 減少影像大小
--imgsz 512

# 關閉混合精度
--mixed_precision 0
```

### 訓練太慢
```bash
# 增加 workers
--workers 8

# 啟用混合精度
--mixed_precision

# 使用多 GPU
--device 0,1,2,3
```

### 收斂不佳
```bash
# 增加 warmup
--warmup_epochs 10

# 調整學習率
--lr 0.0005

# 使用 EMA
--use_ema
```

---

## 📈 監控訓練

### 即時監控
```bash
# 在另一個終端執行
tail -f yolov7_logs/yolov7_training_*.log
```

### 查看訓練歷史
```bash
# 使用 Python
python -c "import json; print(json.dumps(json.load(open('yolov7_models/run_*/training_history.json')), indent=2))"
```

---

## 🎓 進階技巧

### 恢復訓練
```bash
python train_yolov7_medical.py \
    --pretrained yolov7_models/run_*/weights/last.pt \
    --data_dir ./datasets/ct_data \
    --epochs 200  # 總輪數
```

### 遷移學習
```bash
python train_yolov7_medical.py \
    --pretrained yolov7_pretrained.pt \
    --data_dir ./new_dataset \
    --epochs 50 \
    --lr 0.0001  # 較小的學習率
```

### 比較實驗
```bash
# Baseline
python train_yolov7_medical.py --use_medical_modules 0 --save_dir ./exp_baseline

# With Medical Modules
python train_yolov7_medical.py --use_medical_modules 1 --save_dir ./exp_medical
```

---

## 📞 需要幫助？

1. 檢查 `README_YOLOV7.md` 完整文件
2. 執行 `python setup_yolov7.py` 驗證設置
3. 執行 `python train_yolov7_medical.py --help` 查看所有參數

---

**注意**: 所有命令都應在 `detection/yolo_detection/` 目錄下執行
