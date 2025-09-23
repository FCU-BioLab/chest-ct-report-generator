#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可視化模組
包含預測結果可視化和統計圖表生成功能
"""

import os
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visualize_predictions(images, predictions, targets, save_dir, num_samples=10, 
                         confidence_threshold=0.5, prefix="final_predictions"):
    """簡化的可視化預測結果"""
    os.makedirs(save_dir, exist_ok=True)
    num_samples = min(num_samples, len(images))
    
    for i in range(num_samples):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # 處理圖像格式
        image = images[i]
        if isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.shape[0] in [1, 3]:
                image_np = image.permute(1, 2, 0).cpu().numpy()
            else:
                image_np = image.cpu().numpy()
            
            if image_np.shape[-1] == 1:
                image_np = np.repeat(image_np, 3, axis=-1)
            elif len(image_np.shape) == 2:
                image_np = np.stack([image_np] * 3, axis=-1)
        else:
            image_np = image
            
        if image_np.max() > 1.0:
            image_np = image_np / 255.0
        
        # 顯示真實標註
        ax1.imshow(image_np, cmap='gray' if image_np.shape[-1] == 1 else None)
        ax1.set_title(f'Ground Truth (Sample {i+1})')
        ax1.axis('off')
        
        if 'boxes' in targets[i] and len(targets[i]['boxes']) > 0:
            gt_boxes = targets[i]['boxes'].cpu().numpy() if isinstance(targets[i]['boxes'], torch.Tensor) else targets[i]['boxes']
            for box in gt_boxes:
                x1, y1, x2, y2 = box
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       linewidth=2, edgecolor='green', facecolor='none')
                ax1.add_patch(rect)
        
        # 顯示預測結果
        ax2.imshow(image_np, cmap='gray' if image_np.shape[-1] == 1 else None)
        ax2.set_title(f'Predictions (Sample {i+1})')
        ax2.axis('off')
        
        if 'boxes' in predictions[i] and len(predictions[i]['boxes']) > 0:
            pred_boxes = predictions[i]['boxes'].cpu().numpy() if isinstance(predictions[i]['boxes'], torch.Tensor) else predictions[i]['boxes']
            pred_scores = predictions[i]['scores'].cpu().numpy() if isinstance(predictions[i]['scores'], torch.Tensor) else predictions[i]['scores']
            
            # 過濾低置信度預測
            valid_indices = pred_scores > confidence_threshold
            pred_boxes = pred_boxes[valid_indices]
            pred_scores = pred_scores[valid_indices]
            
            for box, score in zip(pred_boxes, pred_scores):
                x1, y1, x2, y2 = box
                color = 'red' if score > 0.7 else 'orange' if score > 0.5 else 'yellow'
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       linewidth=2, edgecolor=color, facecolor='none')
                ax2.add_patch(rect)
                ax2.text(x1, y1-5, f'{score:.2f}', color=color, fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'{prefix}_sample_{i+1}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    logging.info(f"可視化結果已保存到: {save_dir}")
    return save_dir


def create_prediction_summary(predictions, targets, save_dir, prefix="prediction_summary"):
    """創建簡化的預測結果統計圖"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 統計數據
    pred_counts = [len(pred['boxes']) for pred in predictions]
    target_counts = [len(target['boxes']) for target in targets]
    confidence_scores = []
    for pred in predictions:
        if len(pred['scores']) > 0:
            confidence_scores.extend(pred['scores'].cpu().numpy())
    
    # 創建2x2統計圖
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 預測框數量分佈
    axes[0, 0].hist(pred_counts, bins=20, alpha=0.7, color='blue', label='Predictions')
    axes[0, 0].hist(target_counts, bins=20, alpha=0.7, color='green', label='Ground Truth')
    axes[0, 0].set_xlabel('Number of Boxes per Image')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Box Count Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 置信度分佈
    if confidence_scores:
        axes[0, 1].hist(confidence_scores, bins=50, alpha=0.7, color='orange')
        axes[0, 1].axvline(x=0.5, color='red', linestyle='--', label='Threshold=0.5')
        axes[0, 1].set_xlabel('Confidence Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Confidence Score Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 預測 vs 真實框數量散點圖
    axes[1, 0].scatter(target_counts, pred_counts, alpha=0.6, color='purple')
    max_count = max(max(pred_counts), max(target_counts))
    axes[1, 0].plot([0, max_count], [0, max_count], 'r--', label='Perfect Prediction')
    axes[1, 0].set_xlabel('Ground Truth Box Count')
    axes[1, 0].set_ylabel('Predicted Box Count')
    axes[1, 0].set_title('Predicted vs Ground Truth')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 置信度區間統計
    if confidence_scores:
        thresholds = [0.3, 0.5, 0.7, 0.9]
        counts = [sum(1 for score in confidence_scores if score > thresh) for thresh in thresholds]
        
        axes[1, 1].bar([f'>{thresh}' for thresh in thresholds], counts, color='lightblue')
        axes[1, 1].set_xlabel('Confidence Threshold')
        axes[1, 1].set_ylabel('Number of Predictions')
        axes[1, 1].set_title('Predictions by Confidence')
        axes[1, 1].grid(True, alpha=0.3)
        
        for i, count in enumerate(counts):
            axes[1, 1].text(i, count + max(counts)*0.01, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{prefix}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f"預測統計摘要已保存到: {save_path}")
    return save_path


def create_comprehensive_summary(metrics, save_dir, prefix="comprehensive_metrics"):
    """創建簡化的指標摘要報告"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 創建2x3的圖表布局
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 核心檢測指標雷達圖
    ax1 = plt.subplot(2, 3, 1, projection='polar')
    core_metrics = ['precision', 'sensitivity_recall', 'f1_score', 'mAP@0.5', 'case_level_sensitivity']
    core_values = [metrics.get(m, 0) for m in core_metrics]
    core_labels = ['Precision', 'Sensitivity', 'F1-Score', 'mAP@0.5', 'Case Sensitivity']
    
    angles = np.linspace(0, 2 * np.pi, len(core_metrics), endpoint=False).tolist()
    core_values += core_values[:1]
    angles += angles[:1]
    
    ax1.plot(angles, core_values, 'o-', linewidth=2)
    ax1.fill(angles, core_values, alpha=0.25)
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(core_labels)
    ax1.set_ylim(0, 1)
    ax1.set_title('Core Detection Metrics', y=1.08)
    ax1.grid(True)
    
    # 2. IoU品質指標
    ax2 = axes[0, 1]
    iou_metrics = ['iou', 'mean_giou', 'mean_diou', 'mean_ciou']
    iou_values = [metrics.get(m, 0) for m in iou_metrics]
    iou_labels = ['IoU', 'GIoU', 'DIoU', 'CIoU']
    
    bars = ax2.bar(iou_labels, iou_values, color=['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon'])
    ax2.set_ylabel('Score')
    ax2.set_title('IoU Quality Metrics')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    for bar, value in zip(bars, iou_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    # 3. 錯誤分析餅圖
    ax3 = axes[0, 2]
    error_data = [metrics.get('tp', 0), metrics.get('fp', 0), metrics.get('fn', 0)]
    error_labels = ['True Positives', 'False Positives', 'False Negatives']
    colors = ['green', 'red', 'orange']
    
    ax3.pie(error_data, labels=error_labels, colors=colors, autopct='%1.0f', startangle=90)
    ax3.set_title('Detection Results Distribution')
    
    # 4. 敏感度比較
    ax4 = axes[1, 0]
    sensitivity_types = ['Lesion-level', 'Case-level', 'Overall']
    sensitivity_values = [
        metrics.get('lesion_level_sensitivity', 0),
        metrics.get('case_level_sensitivity', 0),
        metrics.get('sensitivity_recall', 0)
    ]
    
    ax4.bar(sensitivity_types, sensitivity_values, color=['lightcyan', 'lightsteelblue', 'lightslategray'])
    ax4.set_ylabel('Sensitivity')
    ax4.set_title('Sensitivity Comparison')
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)
    
    # 5. mAP和假陽性率
    ax5 = axes[1, 1]
    metrics_values = [metrics.get('mAP@0.5', 0), metrics.get('fp_per_image', 0)]
    metrics_labels = ['mAP@0.5', 'FP per Image']
    
    ax5.bar(metrics_labels, metrics_values, color=['gold', 'lightcoral'])
    ax5.set_ylabel('Score / Count')
    ax5.set_title('mAP and False Positives')
    ax5.grid(True, alpha=0.3)
    
    # 6. 統計摘要表格
    ax6 = axes[1, 2]
    ax6.axis('tight')
    ax6.axis('off')
    
    stats_data = [
        ['Total Images', metrics.get('total_images', 0)],
        ['True Positives', metrics.get('tp', 0)],
        ['False Positives', metrics.get('fp', 0)],
        ['False Negatives', metrics.get('fn', 0)],
        ['Precision', f"{metrics.get('precision', 0):.3f}"],
        ['Recall', f"{metrics.get('sensitivity_recall', 0):.3f}"],
        ['F1-Score', f"{metrics.get('f1_score', 0):.3f}"],
        ['mAP@0.5', f"{metrics.get('mAP@0.5', 0):.3f}"]
    ]
    
    table = ax6.table(cellText=stats_data, 
                     colLabels=['Metric', 'Value'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax6.set_title('Summary Statistics', y=0.9)
    
    plt.tight_layout()
    
    # 保存圖片
    save_path = os.path.join(save_dir, f'{prefix}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 保存簡化的文字報告
    report_path = os.path.join(save_dir, f'{prefix}_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== 檢測指標評估報告 ===\n\n")
        f.write("核心檢測指標:\n")
        f.write(f"  • IoU: {metrics.get('iou', 0):.4f}\n")
        f.write(f"  • mAP@0.5: {metrics.get('mAP@0.5', 0):.4f}\n")
        f.write(f"  • Precision: {metrics.get('precision', 0):.4f}\n")
        f.write(f"  • Recall: {metrics.get('sensitivity_recall', 0):.4f}\n")
        f.write(f"  • F1-score: {metrics.get('f1_score', 0):.4f}\n\n")
        
        f.write("敏感度分析:\n")
        f.write(f"  • Lesion-level: {metrics.get('lesion_level_sensitivity', 0):.4f}\n")
        f.write(f"  • Case-level: {metrics.get('case_level_sensitivity', 0):.4f}\n\n")
        
        f.write("IoU變體:\n")
        f.write(f"  • GIoU: {metrics.get('mean_giou', 0):.4f}\n")
        f.write(f"  • DIoU: {metrics.get('mean_diou', 0):.4f}\n")
        f.write(f"  • CIoU: {metrics.get('mean_ciou', 0):.4f}\n\n")
        
        f.write("統計摘要:\n")
        f.write(f"  • Total Images: {metrics.get('total_images', 0)}\n")
        f.write(f"  • True Positives: {metrics.get('tp', 0)}\n")
        f.write(f"  • False Positives: {metrics.get('fp', 0)}\n")
        f.write(f"  • False Negatives: {metrics.get('fn', 0)}\n")
        f.write(f"  • FP per Image: {metrics.get('fp_per_image', 0):.4f}\n")
    
    logging.info(f"指標摘要已保存到: {save_path}")
    return save_path, report_path


def create_kfold_summary_plots(fold_results, save_dir):
    """Create summary plots for K-fold cross-validation results
    
    Args:
        fold_results: List of fold results
        save_dir: Directory to save plots
    """
    try:
        import matplotlib.pyplot as plt
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Extract metrics from each fold
        fold_nums = []
        f1_scores = []
        precisions = []
        recalls = []
        
        for i, result in enumerate(fold_results):
            fold_nums.append(i + 1)
            f1_scores.append(result.get('f1_score', 0))
            precisions.append(result.get('precision', 0))
            recalls.append(result.get('recall', 0))
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # F1 Score plot
        axes[0, 0].plot(fold_nums, f1_scores, 'bo-')
        axes[0, 0].set_title('F1 Score by Fold')
        axes[0, 0].set_xlabel('Fold')
        axes[0, 0].set_ylabel('F1 Score')
        axes[0, 0].grid(True)
        
        # Precision plot
        axes[0, 1].plot(fold_nums, precisions, 'ro-')
        axes[0, 1].set_title('Precision by Fold')
        axes[0, 1].set_xlabel('Fold')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].grid(True)
        
        # Recall plot
        axes[1, 0].plot(fold_nums, recalls, 'go-')
        axes[1, 0].set_title('Recall by Fold')
        axes[1, 0].set_xlabel('Fold')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].grid(True)
        
        # Summary statistics
        axes[1, 1].axis('off')
        stats_text = f"""K-Fold Summary Statistics:
        
F1 Score:
  Mean: {np.mean(f1_scores):.3f}
  Std:  {np.std(f1_scores):.3f}
  
Precision:
  Mean: {np.mean(precisions):.3f}
  Std:  {np.std(precisions):.3f}
  
Recall:
  Mean: {np.mean(recalls):.3f}
  Std:  {np.std(recalls):.3f}
"""
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'kfold_summary.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f"K-fold summary plots saved to: {save_path}")
        return save_path
        
    except ImportError:
        logging.warning("matplotlib not available - K-fold summary plots disabled")
        return None
    except Exception as e:
        logging.error(f"Error creating K-fold summary plots: {e}")
        return None
