# MedSAM2 視頻模式訓練 (Video Mode Training)

## 🎬 概述

這個模組將 CT 切片序列視為「影片」，利用 MedSAM2 的時序傳播 (temporal propagation) 能力來學習病灶分割。

### 核心概念

```
傳統 2D 方法:          視頻模式:
                       
  Slice N-2  [獨立]      Slice N-2  ←───┐
  Slice N-1  [獨立]      Slice N-1  ←───┤  時序傳播
  Slice N    [獨立]  →   Slice N    ←───┤  (Propagation)
  Slice N+1  [獨立]      Slice N+1  ←───┤
  Slice N+2  [獨立]      Slice N+2  ←───┘
```

**優勢**:
- ✅ 學習病灶的 3D 空間連續性
- ✅ 跨切片的形態變化
- ✅ 時序一致的分割結果
- ✅ 減少單切片噪音干擾

---

## 📁 模組結構

```
finetune_medsam2_video/
├── __init__.py           # 模組入口
├── config.py             # 配置類別
├── video_dataset.py      # 視頻資料集
├── npz_converter.py      # 資料轉換器
├── video_trainer.py      # 視頻訓練器
├── visualizer.py         # 視覺化工具
├── utils.py              # 工具函數
├── main.py               # 主程式入口
└── README.md             # 使用說明
```

---

## 🚀 快速開始

### 1. 資料轉換

首先將 LNDb 或 MSD 資料集轉換為 NPZ 視頻格式：

```cmd
cd segmentation

REM 轉換 LNDb 資料集
python finetune_medsam2_video\main.py convert --dataset lndb --input_dir E:\lung_ct_lesion_dataset\LNDb --output_dir cache\lndb_video_npz --context_slices 6 --min_diameter 4.0

REM 轉換 MSD Lung 資料集
python finetune_medsam2_video\main.py convert --dataset msd --input_dir D:\Data\Task06_Lung --output_dir cache\msd_video_npz
```

**轉換參數說明**:

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--context_slices` | 6 | 中心切片前後各取幾個切片（視頻長度 = 2×6+1 = 13 幀）|
| `--max_video_length` | 32 | 最大視頻長度（避免 OOM）|
| `--min_diameter` | 4.0 | 最小結節直徑過濾 (mm) |
| `--image_size` | 512 | 輸出影像大小 |

### 2. 訓練模型

```cmd
python finetune_medsam2_video\main.py train --npz_dir cache\lndb_video_npz --output_dir segmentation\video_result\lndb_video_output --epochs 50 --learning_rate 1e-5 --propagation_steps 3
```

**訓練參數說明**:

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--epochs` | 50 | 訓練 epochs |
| `--batch_size` | 1 | 視頻模式通常使用 batch_size=1 |
| `--learning_rate` | 1e-5 | 學習率 |
| `--propagation_steps` | 3 | 每次前向/後向傳播步數 |
| `--patience` | 10 | 早停 patience |
| `--no_amp` | False | 禁用混合精度訓練 |

### 3. 查看統計

```cmd
python finetune_medsam2_video\main.py stats --npz_dir cache\lndb_video_npz
```

---

## 📹 視覺化工具

轉換完成後，可以使用視覺化工具播放和預覽視頻：

### 命令列使用

```cmd
cd segmentation

REM 列出所有樣本
python finetune_medsam2_video\visualizer.py --npz_dir cache\lndb_video_npz list

REM 互動式播放（有滑桿和播放按鈕）
python finetune_medsam2_video\visualizer.py --npz_dir cache\lndb_video_npz play --index 0

REM 動畫自動播放
python finetune_medsam2_video\visualizer.py --npz_dir cache\lndb_video_npz animate --index 0 --interval 200

REM 保存為 GIF
python finetune_medsam2_video\visualizer.py --npz_dir cache\lndb_video_npz gif --index 0 --fps 5

REM 批量生成所有 GIF
python finetune_medsam2_video\visualizer.py --npz_dir cache\lndb_video_npz batch --output_dir video_gifs --fps 5

REM 網格預覽（多個樣本的中心幀）
python finetune_medsam2_video\visualizer.py --npz_dir cache\lndb_video_npz grid --num 9 --split train
```

### Python 程式碼使用

```python
from finetune_medsam2_video import VideoVisualizer

viz = VideoVisualizer(npz_dir="video_npz")

# 列出樣本
print(f"總樣本數: {len(viz.samples)}")

# 互動式播放（帶滑桿控制）
viz.play_interactive(index=0)

# 動畫自動播放
viz.show_sample_animation(index=0, interval=200)

# 保存 GIF
viz.save_gif(index=0, output_path="lesion_video.gif", fps=5)

# 保存 MP4（需要 OpenCV）
viz.save_mp4(index=0, output_path="lesion_video.mp4", fps=10)

# 網格預覽
viz.preview_grid(num_samples=9, split='train')
```

### 視覺化功能

| 功能 | 說明 |
|------|------|
| `play` | 互動式播放，有滑桿、播放/停止按鈕 |
| `animate` | 自動循環播放動畫 |
| `gif` | 保存為 GIF 動畫 |
| `batch` | 批量生成所有 GIF |
| `grid` | 多樣本網格預覽 |
| `list` | 列出所有樣本資訊 |

播放時會：
- ⭐ 標記中心幀（有完整標註的那一幀）
- 🔴 紅色疊加顯示病灶 mask
- 顯示病灶資訊（直徑、幀數等）

---

## 📊 NPZ 視頻格式

每個 NPZ 檔案包含一個病灶的視頻資料：

```python
{
    # 視頻資料
    'frames': np.ndarray,        # (D, H, W) uint8, CT 切片序列
    'masks': np.ndarray,         # (D, H, W) uint8, 分割遮罩序列
    
    # 索引資訊
    'center_idx': int,           # 中心幀索引（有完整標註）
    'slice_indices': list,       # 對應原始體積的切片索引
    
    # 病灶資訊
    'patient_id': str,           # 患者 ID
    'lesion_id': int,            # 病灶 ID
    'diameter_mm': float,        # 病灶直徑 (mm)
    'volume_mm3': float,         # 病灶體積 (mm³)
    
    # 空間資訊
    'spacing': np.ndarray,       # (3,) voxel spacing
    'origin': np.ndarray,        # (3,) 世界座標原點
    'original_shape': tuple,     # 原始體積大小
    
    # Prompt
    'bbox': np.ndarray,          # (4,) bounding box
}
```

---

## 🎯 訓練策略

### 核心流程

1. **載入視頻**: 以病灶為中心的連續 CT 切片
2. **中心幀 Prompt**: 在中心幀給定 bbox/point prompt
3. **雙向傳播**: 從中心幀向前後傳播分割
4. **損失計算**: 所有幀的 Dice + Focal + 一致性損失

### 損失函數

```python
Loss = λ₁ × Dice + λ₂ × Focal + λ₃ × Consistency

# 預設權重
λ₁ = 1.0   # Dice Loss
λ₂ = 0.5   # Focal Loss
λ₃ = 0.3   # Propagation Consistency Loss
```

**Propagation Consistency Loss**: 確保相鄰幀的預測變化與 GT 變化一致。

---

## 💡 進階用法

### 使用自訂配置

```python
from finetune_medsam2_video import VideoConfig, MedSAM2VideoTrainer

config = VideoConfig()
config.data.npz_dir = "my_video_npz"
config.training.epochs = 100
config.training.propagation_steps = 5
config.training.dice_weight = 1.5

trainer = MedSAM2VideoTrainer(config)
trainer.train()
```

### 載入已訓練模型

```python
config = VideoConfig.load("video_output/config.json")
trainer = MedSAM2VideoTrainer(config)
trainer.load_checkpoint("video_output/checkpoints/best_model.pt")
```

### 完整訓練流程（Windows CMD）

```cmd
cd C:\GitHub\chest-ct-report-generator\segmentation

REM Step 1: 轉換資料
python finetune_medsam2_video\main.py convert --dataset lndb --input_dir D:\Data\LNDb --output_dir cache\lndb_video_npz

REM Step 2: 預覽資料
python finetune_medsam2_video\visualizer.py --npz_dir cache\lndb_video_npz list
python finetune_medsam2_video\visualizer.py --npz_dir cache\lndb_video_npz play --index 0

REM Step 3: 訓練模型
python finetune_medsam2_video\main.py train --npz_dir cache\lndb_video_npz --output_dir result\lndb_video_output --epochs 50

REM Step 4: 查看訓練結果
python finetune_medsam2_video\main.py stats --npz_dir cache\lndb_video_npz
```

---

## 📈 預期效果

相比傳統 2D 模式，視頻模式預期：

| 指標 | 2D 模式 | 視頻模式 | 提升 |
|------|---------|----------|------|
| Dice Score | ~0.70 | ~0.78 | +8% |
| IoU | ~0.55 | ~0.65 | +10% |
| 3D 一致性 | 低 | 高 | ⭐ |
| 邊界平滑度 | 一般 | 良好 | ⭐ |

---

## ⚠️ 注意事項

1. **記憶體**: 視頻模式比 2D 需要更多 GPU 記憶體
   - 建議 GPU VRAM ≥ 16GB
   - 可透過 `--max_video_length` 控制

2. **訓練時間**: 因為需要傳播，每個 epoch 會比 2D 慢
   - 約慢 2-3 倍

3. **Batch Size**: 視頻模式通常使用 batch_size=1
   - 可透過 gradient accumulation 模擬較大 batch

---

## 🔗 相關資源

- [MedSAM2 原始專案](https://github.com/bowang-lab/MedSAM2)
- [SAM2 論文](https://arxiv.org/abs/2408.00714)
- [finetune_medsam2 (2D/2.5D 模式)](../finetune_medsam2/)
