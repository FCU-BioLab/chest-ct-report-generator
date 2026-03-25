#!/usr/bin/env python3
"""
診斷可視化問題的腳本
"""

import sys
from pathlib import Path
import numpy as np
import torch

# 添加路徑
medsam2_path = Path(__file__).parent.parent / "MedSAM2"
if str(medsam2_path) not in sys.path:
    sys.path.insert(0, str(medsam2_path))

package_parent = Path(__file__).parent.parent
if str(package_parent) not in sys.path:
    sys.path.insert(0, str(package_parent))

from finetune_medsam2.dataset import ChestTumorDataset
from finetune_medsam2.utils import custom_collate_fn
from torch.utils.data import DataLoader

def main():
    print("="*60)
    print("診斷可視化問題")
    print("="*60)
    
    # 載入一個樣本
    data_dir = Path("../datasets/aLL_patients_data")
    
    # 找幾個測試患者
    all_patients = []
    for i in range(10):
        subset_dir = data_dir / f"subset{i}"
        if subset_dir.exists():
            for f in subset_dir.glob("*.mhd"):
                all_patients.append(f.stem)
    
    all_patients = sorted(list(set(all_patients)))[:5]  # 只取 5 個
    
    print(f"測試患者數: {len(all_patients)}")
    
    # 建立資料集
    dataset = ChestTumorDataset(
        str(data_dir),
        all_patients,
        axis=2,
        cache_data=False
    )
    
    print(f"資料集樣本數: {len(dataset)}")
    
    if len(dataset) == 0:
        print("❌ 沒有樣本！")
        return
    
    # 找一個有腫瘤的樣本
    sample = None
    for i in range(min(len(dataset), 50)):
        s = dataset[i]
        if s['mask'].sum() > 0:
            sample = s
            print(f"找到有腫瘤的樣本: index={i}")
            break
    
    if sample is None:
        print("❌ 前 50 個樣本都沒有腫瘤！")
        sample = dataset[0]
    
    image = sample['image']  # [3, H, W]
    mask = sample['mask']    # [1, H, W]
    bboxes = sample['bboxes']
    patient_id = sample['patient_id']
    slice_idx = sample['slice_index']
    
    print(f"\n樣本資訊:")
    print(f"  Patient ID: {patient_id}")
    print(f"  Slice Index: {slice_idx}")
    print(f"  Image shape: {image.shape}")
    print(f"  Image dtype: {image.dtype}")
    print(f"  Image range: [{image.min():.2f}, {image.max():.2f}]")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Mask dtype: {mask.dtype}")
    print(f"  Mask unique: {torch.unique(mask).numpy()}")
    print(f"  Mask sum (腫瘤像素數): {mask.sum().item()}")
    print(f"  BBoxes shape: {bboxes.shape}")
    print(f"  BBoxes: {bboxes.numpy()}")
    
    # 檢查 bbox 是否合理
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox.numpy()
        print(f"  BBox {i}: x1={x1:.0f}, y1={y1:.0f}, x2={x2:.0f}, y2={y2:.0f}, w={x2-x1:.0f}, h={y2-y1:.0f}")
    
    # 視覺化檢查
    print("\n生成診斷圖片...")
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 1. 原始影像 (通道 1 - Lung Window)
    img_ch1 = image[0].numpy()
    axes[0].imshow(img_ch1, cmap='gray')
    axes[0].set_title(f'Channel 1 (Lung)\nRange: [{img_ch1.min():.1f}, {img_ch1.max():.1f}]')
    axes[0].axis('off')
    
    # 2. GT Mask
    mask_np = mask.squeeze().numpy()
    axes[1].imshow(mask_np, cmap='gray')
    axes[1].set_title(f'GT Mask\nSum: {mask_np.sum():.0f}')
    axes[1].axis('off')
    
    # 3. 影像 + GT Mask 疊加
    axes[2].imshow(img_ch1, cmap='gray')
    mask_overlay = np.zeros((*mask_np.shape, 4))
    mask_overlay[mask_np > 0.5] = [0, 1, 0, 0.5]
    axes[2].imshow(mask_overlay)
    # 繪製 bbox
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox.numpy()
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                             fill=False, edgecolor='red', linewidth=2)
        axes[2].add_patch(rect)
    axes[2].set_title('Image + GT + BBox')
    axes[2].axis('off')
    
    # 4. 檢查 BBox 區域是否包含 mask
    if len(bboxes) > 0:
        x1, y1, x2, y2 = bboxes[0].numpy().astype(int)
        # 確保座標在範圍內
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(mask_np.shape[1], x2), min(mask_np.shape[0], y2)
        
        bbox_region = mask_np[y1:y2, x1:x2]
        axes[3].imshow(bbox_region, cmap='gray')
        axes[3].set_title(f'BBox Region\nSum: {bbox_region.sum():.0f}')
        axes[3].axis('off')
    
    plt.suptitle(f'Patient: {patient_id[:40]}... | Slice: {slice_idx}', fontsize=12)
    plt.tight_layout()
    
    output_path = Path(__file__).parent / 'debug_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ 診斷圖片已保存: {output_path}")
    
    # 額外檢查：確認 bbox 座標順序
    print("\n📋 BBox 座標檢查:")
    print("  預期格式: [x1, y1, x2, y2] (左上角 x, 左上角 y, 右下角 x, 右下角 y)")
    print("  在 imshow 中: x 對應列 (水平), y 對應行 (垂直)")
    print("  Rectangle((x1, y1), width, height) 應該是正確的")
    
    # 檢查 mask 和 bbox 的對應
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox.numpy().astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(mask_np.shape[1], x2), min(mask_np.shape[0], y2)
        
        region_sum = mask_np[y1:y2, x1:x2].sum()
        total_sum = mask_np.sum()
        
        print(f"  BBox {i}: 區域內 mask 佔 {region_sum/total_sum*100:.1f}% ({region_sum:.0f}/{total_sum:.0f})")
        
        if region_sum < total_sum * 0.5:
            print(f"  ⚠️ 警告: BBox 沒有完全覆蓋 mask！可能座標順序有問題")

if __name__ == "__main__":
    main()
