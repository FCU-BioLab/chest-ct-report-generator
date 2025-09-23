# YOLOv11 CT病灶检测训练和测试

本目录包含基于原有Faster R-CNN训练逻辑改写的YOLOv11版本训练和测试脚本。

## 文件说明

### 训练脚本
- `train_yolov11.py` - YOLOv11 K-fold交叉验证训练脚本（完整版）
- `train_yolov11_simple.py` - YOLOv11简单训练脚本（单次训练）

### 测试脚本
- `test_yolov11.py` - YOLOv11模型测试和评估脚本

### 支持文件
- `README_YOLOv11.md` - 本说明文件

## 环境要求

### 安装依赖
```bash
# 安装所有依赖（包括YOLOv11支持）
pip install -r requirements.txt
```

### 核心依赖说明
- **ultralytics>=8.0.0** - YOLOv11核心包
- **torch>=2.1.0** - PyTorch深度学习框架
- **torchvision>=0.16.0** - 计算机视觉工具
- **opencv-python>=4.8.0** - 图像处理
- **albumentations>=1.3.0** - 数据增强
- **matplotlib>=3.7.0** - 可视化

### 验证安装
```bash
# 验证YOLOv11安装
python -c "from ultralytics import YOLO; print('YOLOv11 ready!')"
```

### 可选包（用于DICOM处理）
```bash
pip install pydicom
```

## 使用方法

### 1. K-fold交叉验证训练

```bash
python train_yolov11.py \
    --data_dir "path/to/your/data" \
    --k_folds 5 \
    --epochs 100 \
    --batch_size 16 \
    --lr 0.01 \
    --save_dir "./yolov11_models" \
    --log_dir "./yolov11_logs" \
    --model_size n \
    --imgsz 640 \
    --include_negative \
    --seed 42
```

参数说明：
- `--data_dir`: 数据目录路径
- `--k_folds`: K折交叉验证的fold数量（默认5）
- `--epochs`: 训练轮数（默认100）
- `--batch_size`: 批次大小（默认16）
- `--lr`: 学习率（默认0.01）
- `--save_dir`: 模型保存目录
- `--log_dir`: 日志保存目录
- `--model_size`: YOLOv11模型大小，可选 'n', 's', 'm', 'l', 'x'（默认'n'）
- `--imgsz`: 输入图像尺寸（默认640）
- `--include_negative`: 是否包含负样本
- `--max_negative`: 每个病例最大负样本数
- `--seed`: 随机种子

### 2. 简单训练

```bash
python train_yolov11_simple.py \
    --data_dir "path/to/your/data" \
    --epochs 100 \
    --batch_size 16 \
    --lr 0.01 \
    --save_dir "./yolov11_simple_training" \
    --train_ratio 0.8 \
    --model_size n \
    --device auto
```

参数说明：
- `--train_ratio`: 训练集比例（默认0.8）
- `--device`: 设备类型，可选 'auto', 'cpu', 'cuda', 'mps'（默认'auto'）
- 其他参数与K-fold训练类似

### 3. 模型测试

```bash
python test_yolov11.py \
    --model_path "path/to/your/model.pt" \
    --data_dir "path/to/your/data" \
    --save_dir "./test_results_yolov11" \
    --confidence_thresholds 0.3 0.5 0.7 \
    --iou_thresholds 0.3 0.5 0.7 \
    --visualize_samples 15 \
    --include_negative
```

参数说明：
- `--model_path`: 训练好的模型文件路径（.pt文件）
- `--confidence_thresholds`: 置信度阈值列表
- `--iou_thresholds`: IoU阈值列表
- `--visualize_samples`: 可视化样本数量
- `--specific_patients`: 指定测试的病例ID（可选）

## 特性对比

| 特性 | Faster R-CNN版本 | YOLOv11版本 |
|------|-----------------|-------------|
| 模型架构 | Two-stage检测器 | One-stage检测器 |
| 训练速度 | 较慢 | 较快 |
| 推理速度 | 较慢 | 较快 |
| 精度 | 通常较高 | 平衡精度和速度 |
| 内存占用 | 较高 | 较低 |
| 模型大小 | 较大 | 可选择多种大小 |
| 部署难度 | 较复杂 | 较简单 |

## 输出结果

### 训练输出
训练完成后会生成以下文件：

```
save_dir/
├── fold_1/                     # K-fold训练时的各fold结果
│   ├── yolo_dataset_train/     # 训练数据
│   ├── yolo_dataset_val/       # 验证数据
│   ├── combined_dataset.yaml   # 数据集配置
│   └── fold_1_training/        # 训练结果
│       ├── weights/
│       │   ├── best.pt         # 最佳模型
│       │   └── last.pt         # 最后epoch模型
│       ├── results.png         # 训练曲线
│       └── confusion_matrix.png
├── fold_2/
├── ...
├── yolov11_kfold_results.json  # K-fold总结果
└── kfold_summary.png           # K-fold摘要图
```

### 测试输出
测试完成后会生成以下文件：

```
test_results/
├── visualizations/             # 可视化结果
│   ├── conf_0.3_iou_0.5/      # 不同阈值组合的结果
│   │   ├── sample_1.png
│   │   └── ...
│   └── ...
├── comprehensive_report/       # 综合报告
│   ├── performance_comparison.png
│   ├── confidence_analysis.png
│   ├── evaluation_report.txt
│   └── evaluation_results.json
└── overall_test_results.json   # 总体测试结果
```

## 评估指标

脚本计算以下评估指标：
- **基础指标**: Precision, Recall, F1-Score
- **检测指标**: mAP@0.5, mAP@[0.5:0.95]
- **IoU变体**: Standard IoU, GIoU, DIoU, CIoU
- **位置精度**: 边界框中心误差、尺寸误差
- **敏感度指标**: 病例级敏感度、病灶级敏感度
- **假阳性分析**: 每张图像平均假阳性数

## 数据格式

脚本支持与原有Faster R-CNN相同的数据格式：
- DICOM图像文件
- Pascal VOC格式的XML标注文件
- 按病例组织的目录结构

数据会自动转换为YOLOv11所需的格式：
- 图像转换为PNG格式
- 标注转换为YOLO格式（中心点坐标+宽高，归一化）

## 注意事项

1. **内存使用**: YOLOv11相比Faster R-CNN内存占用较小，但仍建议使用GPU训练
2. **批次大小**: 根据GPU内存调整batch_size，RTX 3090建议使用16-32
3. **模型选择**: 
   - YOLOv11n: 最快，适合快速实验
   - YOLOv11s: 平衡速度和精度
   - YOLOv11m/l/x: 更高精度，需要更多计算资源
4. **数据增强**: YOLOv11内置数据增强，无需额外配置
5. **早停**: 默认启用早停机制，patience=50个epoch

## 故障排除

### 常见问题

1. **导入错误**: 确保安装了ultralytics包
   ```bash
   pip install ultralytics
   ```

2. **CUDA内存不足**: 减小batch_size或使用更小的模型
   ```bash
   --batch_size 8 --model_size n
   ```

3. **数据格式错误**: 确保数据目录结构正确，XML标注文件存在

4. **训练不收敛**: 调整学习率或增加训练轮数
   ```bash
   --lr 0.001 --epochs 200
   ```

### 性能优化建议

1. **使用混合精度训练**: 在训练参数中自动启用
2. **多GPU训练**: YOLOv11支持自动多GPU训练
3. **数据预处理**: 预先转换数据格式可加速训练
4. **模型蒸馏**: 使用大模型训练小模型

## 与原有系统集成

这些YOLOv11脚本设计为与现有的Faster R-CNN系统兼容：
- 使用相同的数据加载器和预处理流程
- 输出相似的评估指标和可视化结果
- 支持相同的数据集分割和病例管理
- 可与现有的分析和报告生成流程集成

## 许可证

基于原有Faster R-CNN训练脚本的逻辑开发，遵循相同的许可证条款。
