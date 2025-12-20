#!/usr/bin/env python3
"""
診斷模型推論和可視化問題
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

# 添加路徑
medsam2_path = Path(__file__).parent.parent / "MedSAM2"
if str(medsam2_path) not in sys.path:
    sys.path.insert(0, str(medsam2_path))

package_parent = Path(__file__).parent.parent
if str(package_parent) not in sys.path:
    sys.path.insert(0, str(package_parent))

# 初始化 Hydra
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()

config_dir = str((medsam2_path / "sam2" / "configs").absolute())
if Path(config_dir).exists():
    initialize_config_dir(config_dir=config_dir, version_base="1.2")

from finetune_medsam2.dataset import ChestTumorDataset
from finetune_medsam2.trainer import MedSAM2Trainer
from finetune_medsam2.utils import compute_all_metrics

def main():
    print("="*60)
    print("診斷模型推論問題")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")
    
    # 載入資料
    data_dir = Path("../datasets/aLL_patients_data")
    
    all_patients = []
    for i in range(10):
        subset_dir = data_dir / f"subset{i}"
        if subset_dir.exists():
            for f in subset_dir.glob("*.mhd"):
                all_patients.append(f.stem)
    
    all_patients = sorted(list(set(all_patients)))[:10]
    
    dataset = ChestTumorDataset(str(data_dir), all_patients, axis=2)
    
    # 找一個有腫瘤的樣本
    sample = None
    sample_idx = 0
    for i in range(min(len(dataset), 100)):
        s = dataset[i]
        if s['mask'].sum() > 100:  # 至少有 100 個腫瘤像素
            sample = s
            sample_idx = i
            break
    
    if sample is None:
        print("❌ 找不到足夠大的腫瘤樣本")
        return
    
    print(f"\n找到樣本 index={sample_idx}")
    print(f"  Patient: {sample['patient_id']}")
    print(f"  Slice: {sample['slice_index']}")
    print(f"  Mask sum: {sample['mask'].sum().item()}")
    
    # 載入模型（使用預訓練權重，不是 fine-tuned）
    print("\n載入模型...")
    trainer = MedSAM2Trainer(
        model_config="sam2.1_hiera_t512.yaml",
        checkpoint_path="MedSAM2/checkpoints/MedSAM2_latest.pt",
        device=device,
        output_dir="./debug_output"
    )
    
    trainer.model.eval()
    
    # 準備資料
    image = sample['image'].to(device)  # [3, H, W]
    gt_mask = sample['mask'].to(device)  # [1, H, W]
    bboxes = sample['bboxes'].to(device)  # [N, 4]
    
    print(f"\n輸入資料:")
    print(f"  Image shape: {image.shape}")
    print(f"  Image range: [{image.min():.2f}, {image.max():.2f}]")
    print(f"  GT Mask shape: {gt_mask.shape}")
    print(f"  BBoxes: {bboxes}")
    
    # 進行推論
    print("\n進行推論...")
    
    with torch.no_grad():
        # 計算 image embedding
        image_embedding, high_res_feats = trainer._prepare_image_features(image)
        
        print(f"  Image embedding shape: {image_embedding.shape}")
        print(f"  High res feats: {[f.shape for f in high_res_feats] if high_res_feats else None}")
        
        all_pred_masks = []
        
        for bbox in bboxes:
            if bbox.sum() == 0:
                continue
            
            print(f"\n  處理 BBox: {bbox.cpu().numpy()}")
            
            box_torch = bbox.unsqueeze(0)
            
            # Prompt Encoder
            sparse_embeddings, dense_embeddings = trainer.model.sam_prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
            
            print(f"  Sparse embeddings shape: {sparse_embeddings.shape}")
            print(f"  Dense embeddings shape: {dense_embeddings.shape}")
            
            # Mask Decoder
            low_res_masks, iou_predictions, _, _ = trainer.model.sam_mask_decoder(
                image_embeddings=image_embedding,
                image_pe=trainer.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res_feats,
            )
            
            print(f"  Low res masks shape: {low_res_masks.shape}")
            print(f"  Low res masks range: [{low_res_masks.min():.4f}, {low_res_masks.max():.4f}]")
            print(f"  IoU predictions: {iou_predictions}")
            
            # 上採樣到原始大小
            pred_mask = F.interpolate(
                low_res_masks,
                size=(gt_mask.shape[-2], gt_mask.shape[-1]),
                mode='bilinear',
                align_corners=False
            )
            
            print(f"  Pred mask shape: {pred_mask.shape}")
            print(f"  Pred mask (before sigmoid) range: [{pred_mask.min():.4f}, {pred_mask.max():.4f}]")
            
            pred_sigmoid = torch.sigmoid(pred_mask)
            print(f"  Pred mask (after sigmoid) range: [{pred_sigmoid.min():.4f}, {pred_sigmoid.max():.4f}]")
            
            pred_binary = (pred_sigmoid > 0.5).float()
            print(f"  Pred binary sum: {pred_binary.sum().item()}")
            
            all_pred_masks.append(pred_binary.squeeze().cpu().numpy())
            
            # 計算指標
            metrics = compute_all_metrics(pred_mask.squeeze(), gt_mask.squeeze())
            print(f"  Dice: {metrics['dice']:.4f}")
            print(f"  IoU: {metrics['iou']:.4f}")
    
    # 可視化
    print("\n生成診斷可視化...")
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 原始影像
    img_np = image[0].cpu().numpy()
    axes[0, 0].imshow(img_np, cmap='gray')
    axes[0, 0].set_title(f'Original Image\nRange: [{img_np.min():.1f}, {img_np.max():.1f}]')
    axes[0, 0].axis('off')
    
    # GT Mask
    gt_np = gt_mask.squeeze().cpu().numpy()
    axes[0, 1].imshow(gt_np, cmap='gray')
    axes[0, 1].set_title(f'GT Mask\nSum: {gt_np.sum():.0f}')
    axes[0, 1].axis('off')
    
    # Pred Mask
    if all_pred_masks:
        pred_combined = np.zeros_like(all_pred_masks[0])
        for pm in all_pred_masks:
            pred_combined = np.maximum(pred_combined, pm)
        axes[0, 2].imshow(pred_combined, cmap='gray')
        axes[0, 2].set_title(f'Pred Mask\nSum: {pred_combined.sum():.0f}')
    axes[0, 2].axis('off')
    
    # Image + GT
    axes[1, 0].imshow(img_np, cmap='gray')
    gt_overlay = np.zeros((*gt_np.shape, 4))
    gt_overlay[gt_np > 0.5] = [0, 1, 0, 0.5]
    axes[1, 0].imshow(gt_overlay)
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox.cpu().numpy()
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='cyan', linewidth=2)
        axes[1, 0].add_patch(rect)
    axes[1, 0].set_title('Image + GT + BBox')
    axes[1, 0].axis('off')
    
    # Image + Pred
    axes[1, 1].imshow(img_np, cmap='gray')
    if all_pred_masks:
        pred_overlay = np.zeros((*pred_combined.shape, 4))
        pred_overlay[pred_combined > 0.5] = [1, 0, 0, 0.5]
        axes[1, 1].imshow(pred_overlay)
    axes[1, 1].set_title('Image + Pred')
    axes[1, 1].axis('off')
    
    # Comparison
    axes[1, 2].imshow(img_np, cmap='gray')
    if all_pred_masks:
        overlap = gt_np * pred_combined
        gt_only = gt_np * (1 - pred_combined)
        pred_only = pred_combined * (1 - gt_np)
        
        comp_overlay = np.zeros((*gt_np.shape, 4))
        comp_overlay[gt_only > 0] = [0, 1, 0, 0.5]
        comp_overlay[pred_only > 0] = [1, 0, 0, 0.5]
        comp_overlay[overlap > 0] = [1, 1, 0, 0.5]
        axes[1, 2].imshow(comp_overlay)
    axes[1, 2].set_title('Comparison\nGreen=GT only, Red=Pred only, Yellow=Overlap')
    axes[1, 2].axis('off')
    
    plt.suptitle(f'Patient: {sample["patient_id"][:40]}... | Slice: {sample["slice_index"]}', fontsize=14)
    plt.tight_layout()
    
    output_path = Path(__file__).parent / 'debug_inference.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ 診斷圖片已保存: {output_path}")

if __name__ == "__main__":
    main()
