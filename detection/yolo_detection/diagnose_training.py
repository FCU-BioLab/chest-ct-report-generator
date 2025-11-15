#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练效果诊断工具

分析训练结果，识别问题并给出优化建议
"""

import pandas as pd
from pathlib import Path
import numpy as np

def analyze_training_results(results_csv_path):
    """分析训练结果CSV文件"""
    
    print("\n" + "="*80)
    print("🔍 训练效果诊断报告")
    print("="*80 + "\n")
    
    # 读取数据
    df = pd.read_csv(results_csv_path)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(df) == 0:
        print("❌ 无有效数据")
        return
    
    # 提取关键列（兼容不同 YOLO 版本的列名）
    epoch = df.iloc[:, 0].values  # 第一列是 epoch
    
    # 尝试不同的列名格式
    def get_column(df, possible_names):
        for name in possible_names:
            if name in df.columns:
                return df[name].values
        return None
    
    train_box_loss = get_column(df, ['train/box_loss', 'box_loss'])
    train_cls_loss = get_column(df, ['train/cls_loss', 'cls_loss'])
    val_box_loss = get_column(df, ['val/box_loss', 'val_box_loss'])
    val_cls_loss = get_column(df, ['val/cls_loss', 'val_cls_loss'])
    
    precision = get_column(df, ['metrics/precision(B)', 'precision'])
    recall = get_column(df, ['metrics/recall(B)', 'recall'])
    map50 = get_column(df, ['metrics/mAP50(B)', 'mAP50'])
    map5095 = get_column(df, ['metrics/mAP50-95(B)', 'mAP50-95'])
    
    # === 1. 基础指标分析 ===
    print("📊 最终性能指标（Last Epoch）:")
    print(f"  - mAP@0.5:      {map50[-1]:.4f}  {'✅ 优秀' if map50[-1] >= 0.85 else '⚠️ 需改善' if map50[-1] >= 0.70 else '❌ 较差'}")
    print(f"  - mAP@0.5:0.95: {map5095[-1]:.4f}  {'✅ 优秀' if map5095[-1] >= 0.60 else '⚠️ 需改善' if map5095[-1] >= 0.45 else '❌ 较差'}")
    print(f"  - Precision:    {precision[-1]:.4f}  {'✅ 优秀' if precision[-1] >= 0.80 else '⚠️ 需改善' if precision[-1] >= 0.65 else '❌ 较差'}")
    print(f"  - Recall:       {recall[-1]:.4f}  {'✅ 优秀' if recall[-1] >= 0.75 else '⚠️ 需改善' if recall[-1] >= 0.60 else '❌ 较差'}")
    print()
    
    # === 2. 损失分析 ===
    print("📉 损失函数分析:")
    print(f"  训练集分类损失: {train_cls_loss[-1]:.4f} (首轮: {train_cls_loss[0]:.4f})")
    print(f"  验证集分类损失: {val_cls_loss[-1]:.4f} (首轮: {val_cls_loss[0]:.4f})")
    
    # 关键诊断：验证集分类损失是否过高
    val_train_ratio = val_cls_loss[-1] / (train_cls_loss[-1] + 1e-6)
    print(f"  验证/训练比例: {val_train_ratio:.2f}x ", end="")
    
    if val_train_ratio > 3.0:
        print("🔴 严重过拟合！验证集损失是训练集的 {:.1f} 倍".format(val_train_ratio))
        print("     建议：降低模型复杂度、增加正则化、启用数据增强")
    elif val_train_ratio > 2.0:
        print("⚠️ 轻度过拟合")
        print("     建议：增加 dropout、label smoothing")
    else:
        print("✅ 损失平衡良好")
    print()
    
    # === 3. 收敛性分析 ===
    print("📈 收敛性分析:")
    
    # 检查最后 20% 的 epoch 是否还在改善
    last_20_percent = int(len(df) * 0.8)
    map50_improvement = map50[-1] - map50[last_20_percent]
    
    print(f"  最后 20% epochs 的 mAP@0.5 提升: {map50_improvement:+.4f}")
    if abs(map50_improvement) < 0.005:
        print("  ⚠️ 已收敛（建议提前停止或降低学习率）")
    elif map50_improvement > 0:
        print("  ✅ 仍在改善（可继续训练）")
    else:
        print("  🔴 性能下降（可能过拟合，应回退到早期模型）")
    print()
    
    # === 4. 最佳 epoch 识别 ===
    best_map_epoch = np.argmax(map50)
    print(f"🏆 最佳 mAP@0.5 出现在 Epoch {best_map_epoch + 1}:")
    print(f"  - mAP@0.5:      {map50[best_map_epoch]:.4f}")
    print(f"  - Precision:    {precision[best_map_epoch]:.4f}")
    print(f"  - Recall:       {recall[best_map_epoch]:.4f}")
    print(f"  - Val Cls Loss: {val_cls_loss[best_map_epoch]:.4f}")
    
    if best_map_epoch < len(df) * 0.7:
        print("  ⚠️ 最佳模型出现较早，建议使用 best.pt 而非 last.pt")
    print()
    
    # === 5. 诊断建议 ===
    print("💡 优化建议:")
    suggestions = []
    
    if val_train_ratio > 2.5:
        suggestions.append("1. 降低分类损失权重（cls: 2.0 → 1.0）")
        suggestions.append("2. 增加 dropout（0.2 → 0.3）")
        suggestions.append("3. 启用 label_smoothing=0.05")
    
    if map50[-1] < 0.75:
        suggestions.append("4. 启用数据增强（mosaic=0.5, mixup=0.1）")
        suggestions.append("5. 降低学习率（lr=0.0005 → 0.0003）")
    
    if recall[-1] < 0.65:
        suggestions.append("6. 增加正样本过采样（oversample_positive=3.0）")
        suggestions.append("7. 调整置信度阈值（降低检测门槛）")
    
    if precision[-1] < 0.70:
        suggestions.append("8. 增加 box loss 权重（box: 5.0 → 6.0）")
        suggestions.append("9. 减少负样本比例（max_negative_ratio=0.3 → 0.2）")
    
    if len(suggestions) == 0:
        print("  ✅ 训练效果良好，无明显问题")
    else:
        for s in suggestions:
            print(f"  {s}")
    
    print("\n" + "="*80 + "\n")
    
    print("✅ 诊断完成！请根据以上建议调整训练配置。")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # 默认分析最新的训练结果
        yolo_runs = Path("yolo_runs")
        if not yolo_runs.exists():
            print("❌ 未找到 yolo_runs 目录")
            sys.exit(1)
        
        # 找到最新的训练目录
        train_dirs = sorted([d for d in yolo_runs.glob("train_*") if d.is_dir()], 
                           key=lambda x: x.name, reverse=True)
        
        if len(train_dirs) == 0:
            print("❌ 未找到训练结果")
            sys.exit(1)
        
        latest_train = train_dirs[0]
        results_files = list(latest_train.rglob("results.csv"))
        
        if len(results_files) == 0:
            print(f"❌ 在 {latest_train} 中未找到 results.csv")
            sys.exit(1)
        
        csv_path = results_files[0]
        print(f"📁 分析文件: {csv_path}")
    
    analyze_training_results(csv_path)
