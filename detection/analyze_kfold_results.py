#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K-Fold 結果分析腳本

使用方式:
python analyze_kfold_results.py --results_dir CT_ViT_Detection

作者: GitHub Copilot
日期: 2025-08-01
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_kfold_results(results_dir):
    """載入K-Fold結果"""
    results_file = os.path.join(results_dir, 'kfold_final_results.json')
    
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"找不到結果檔案: {results_file}")
    
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    return results

def plot_fold_comparison(results, save_dir):
    """繪製各Fold準確率比較圖"""
    accuracies = results['individual_accuracies']
    folds = list(range(1, len(accuracies) + 1))
    
    plt.figure(figsize=(10, 6))
    
    # 繪製柱狀圖
    bars = plt.bar(folds, accuracies, alpha=0.7, color='skyblue', edgecolor='navy')
    
    # 添加平均線
    mean_acc = results['mean_accuracy']
    plt.axhline(y=mean_acc, color='red', linestyle='--', 
                label=f'平均準確率: {mean_acc:.4f}')
    
    # 添加標準差區間
    std_acc = results['std_accuracy']
    plt.fill_between(folds, mean_acc - std_acc, mean_acc + std_acc, 
                     alpha=0.2, color='red', label=f'±1標準差: {std_acc:.4f}')
    
    # 在柱狀圖上添加數值
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{acc:.4f}', ha='center', va='bottom')
    
    plt.xlabel('Fold')
    plt.ylabel('驗證準確率')
    plt.title(f'{results["k_folds"]}-Fold 交叉驗證結果比較')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存圖片
    save_path = os.path.join(save_dir, 'fold_accuracy_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"準確率比較圖已保存到: {save_path}")

def plot_training_curves(results, results_dir, save_dir):
    """繪製各Fold的訓練曲線"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('各Fold訓練過程', fontsize=16)
    
    k_folds = results['k_folds']
    
    for fold_idx in range(k_folds):
        fold_num = fold_idx + 1
        row = fold_idx // 3
        col = fold_idx % 3
        
        if fold_idx >= 6:  # 最多顯示6個fold
            break
            
        ax = axes[row, col]
        
        # 載入該fold的詳細結果
        fold_dir = os.path.join(results_dir, f'fold_{fold_num}')
        fold_result_file = os.path.join(fold_dir, 'fold_results.json')
        
        if os.path.exists(fold_result_file):
            with open(fold_result_file, 'r', encoding='utf-8') as f:
                fold_data = json.load(f)
            
            history = fold_data['history']
            epochs = [h['epoch'] for h in history]
            train_losses = [h['train_loss'] for h in history]
            val_losses = [h['val_loss'] for h in history]
            val_accuracies = [h['val_accuracy'] for h in history]
            
            # 繪製損失曲線
            ax2 = ax.twinx()
            line1 = ax.plot(epochs, train_losses, 'b-', label='訓練損失', alpha=0.7)
            line2 = ax.plot(epochs, val_losses, 'r-', label='驗證損失', alpha=0.7)
            line3 = ax2.plot(epochs, val_accuracies, 'g-', label='驗證準確率', alpha=0.7)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('損失', color='black')
            ax2.set_ylabel('準確率', color='g')
            ax.set_title(f'Fold {fold_num}')
            
            # 合併圖例
            lines = line1 + line2 + line3
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper right')
            
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'Fold {fold_num}\n資料未找到', 
                   ha='center', va='center', transform=ax.transAxes)
    
    # 隱藏未使用的子圖
    if k_folds < 6:
        for idx in range(k_folds, 6):
            row = idx // 3
            col = idx % 3
            axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # 保存圖片
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"訓練曲線圖已保存到: {save_path}")

def generate_report(results, save_dir):
    """生成詳細報告"""
    report_content = f"""
# K-Fold 交叉驗證結果報告

## 基本資訊
- K-Fold數量: {results['k_folds']}
- 平均準確率: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}

## 各Fold結果
"""
    
    for i, acc in enumerate(results['individual_accuracies']):
        fold_num = i + 1
        report_content += f"- Fold {fold_num}: {acc:.4f}\n"
    
    report_content += f"""
## 統計分析
- 最佳Fold: {np.argmax(results['individual_accuracies']) + 1} (準確率: {np.max(results['individual_accuracies']):.4f})
- 最差Fold: {np.argmin(results['individual_accuracies']) + 1} (準確率: {np.min(results['individual_accuracies']):.4f})
- 準確率範圍: {np.min(results['individual_accuracies']):.4f} - {np.max(results['individual_accuracies']):.4f}
- 變異係數: {(results['std_accuracy'] / results['mean_accuracy']) * 100:.2f}%

## 結論
K-Fold交叉驗證顯示模型的平均性能為 {results['mean_accuracy']:.4f}，
標準差為 {results['std_accuracy']:.4f}，表明模型性能{"相對穩定" if results['std_accuracy'] < 0.05 else "存在一定變異"}。
"""
    
    # 保存報告
    report_path = os.path.join(save_dir, 'kfold_analysis_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"分析報告已保存到: {report_path}")
    print("\n" + "="*50)
    print("K-FOLD 交叉驗證結果總結")
    print("="*50)
    print(f"平均準確率: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
    print(f"各Fold準確率: {[f'{acc:.4f}' for acc in results['individual_accuracies']]}")
    print(f"最佳Fold: {np.argmax(results['individual_accuracies']) + 1} ({np.max(results['individual_accuracies']):.4f})")
    print(f"最差Fold: {np.argmin(results['individual_accuracies']) + 1} ({np.min(results['individual_accuracies']):.4f})")
    
def main():
    parser = argparse.ArgumentParser(description='K-Fold結果分析')
    parser.add_argument('--results_dir', type=str, 
                       default=os.path.join(os.path.dirname(__file__), 'CT_ViT_Detection'),
                       help='結果目錄路徑')
    
    args = parser.parse_args()
    
    # 檢查結果目錄
    if not os.path.exists(args.results_dir):
        print(f"錯誤: 找不到結果目錄 {args.results_dir}")
        return
    
    try:
        # 載入結果
        print("載入K-Fold結果...")
        results = load_kfold_results(args.results_dir)
        
        # 創建分析輸出目錄
        analysis_dir = os.path.join(args.results_dir, 'analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        
        # 生成分析圖表和報告
        print("生成準確率比較圖...")
        plot_fold_comparison(results, analysis_dir)
        
        print("生成訓練曲線圖...")
        plot_training_curves(results, args.results_dir, analysis_dir)
        
        print("生成分析報告...")
        generate_report(results, analysis_dir)
        
        print(f"\n所有分析結果已保存到: {analysis_dir}")
        
    except Exception as e:
        print(f"分析過程中發生錯誤: {e}")

if __name__ == "__main__":
    main()
