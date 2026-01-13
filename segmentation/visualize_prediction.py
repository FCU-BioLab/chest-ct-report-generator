#!/usr/bin/env python3
"""
UNet++ 預測結果視覺化工具
========================

功能：
1. 載入訓練好的模型
2. 對 Test Set 進行推論
3. 生成並保存視覺化結果（Input, GT, Pred, Overlay）
4. 重點展示：False Negatives (漏檢) 和 False Positives (誤檢)
"""

import sys
from pathlib import Path
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# 確保可以導入本地模組
sys.path.insert(0, str(Path(__file__).parent.parent))

from train_unetpp.config import Config
from train_unetpp.model import get_model
from train_unetpp.dataset import CachedPatchDataset, get_cached_patch_split
from train_unetpp.utils import custom_collate_fn, get_device

# 設定 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def visualize_predictions(run_dir: Path, model_name: str = "best_model.pth", num_samples: int = 50):
    """視覺化預測結果"""
    run_path = Path(run_dir)
    config_path = run_path / "config.json"
    
    if not config_path.exists():
        # 嘗試在上一層找 (如果是 fold 目錄)
        config_path = run_path.parent / "config.json"
        
    if not config_path.exists():
        logger.error(f"找不到配置檔: {config_path}")
        return

    # 載入配置
    config = Config.load(str(config_path))
    device = get_device(config.device)
    
    # 載入模型
    model_path = run_path / model_name
    if not model_path.exists():
        logger.error(f"找不到模型: {model_path}")
        return
        
    logger.info(f"載入模型: {model_path}")
    model = get_model(config).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 準備資料
    cache_dir = config.data.cache_dir
    # 如果 config 中的路徑是相對路徑且找不到，嘗試修正
    if not Path(cache_dir).exists():
        project_root = Path(__file__).parent.parent
        cache_dir = str(project_root / "segmentation" / "cache" / "lndb_patches")
        logger.info(f"修正 cache_dir 為: {cache_dir}")
        
    _, _, test_ids = get_cached_patch_split(
        cache_dir,
        config.data.train_ratio,
        config.data.val_ratio,
        config.seed
    )
    
    if not test_ids:
        logger.warning("沒有找到測試集病人，嘗試使用驗證集...")
        _, test_ids, _ = get_cached_patch_split(...) # Retry with val if test empty
    
    logger.info(f"測試集包含 {len(test_ids)} 個病人")
    
    # 建立 Dataset
    # 注意：這裡我們想要看 *所有* 類型的 patch，而不僅僅是正樣本
    # 但為了效率，我們先看有病灶的 (positive) 和 易混淆的 (hard_negative)
    config.data.val_num_patches = 100 # 增加取樣數以確保覆蓋
    
    test_dataset = CachedPatchDataset(cache_dir, test_ids, config, mode="val")
    
    loader = DataLoader(
        test_dataset,
        batch_size=1, # 逐張處理方便視覺化
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn
    )
    
    # 輸出目錄
    vis_dir = run_path / "viz_predictions"
    vis_dir.mkdir(exist_ok=True)
    
    logger.info(f"開始推論並保存圖片至 {vis_dir}...")
    
    count = 0
    saved_count = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, total=min(len(loader), num_samples * 10)): # 掃描更多，但只保存有趣的
            if saved_count >= num_samples:
                break
                
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            
            # 轉換為 numpy
            # Image: (B, C, H, W) -> 取中間 slice -> (H, W)
            # 2.5D input channel 2 is the center slice
            img_np = images[0, 2, :, :].cpu().numpy()
            
            mask_np = masks[0, 0, :, :].cpu().numpy()
            pred_np = preds[0, 0, :, :].cpu().numpy()
            
            # 簡單分類：TP, FN, FP
            # 只有當有東西時才感興趣
            has_gt = mask_np.sum() > 0
            has_pred = (pred_np > 0.5).sum() > 0
            
            if not has_gt and not has_pred:
                # True Negative，只有 5% 機率保存，避免存太多背景
                if np.random.rand() > 0.05:
                    continue
            
            # 判斷類型
            case_type = "TN"
            if has_gt and has_pred:
                # 檢查 IoU 判斷好壞
                intersection = ((pred_np > 0.5) & (mask_np > 0.5)).sum()
                union = ((pred_np > 0.5) | (mask_np > 0.5)).sum()
                iou = intersection / union if union > 0 else 0
                case_type = f"TP_IoU{iou:.2f}"
            elif has_gt and not has_pred:
                case_type = "FN_Missed"
            elif not has_gt and has_pred:
                case_type = "FP_FalseAlarm"
            
            # 繪圖
            fig, ax = plt.subplots(1, 4, figsize=(16, 4))
            
            # 1. Original CT
            ax[0].imshow(img_np, cmap='gray')
            ax[0].set_title(f"Input CT\n{batch['patient_id'][0]} - slice {batch['slice_idx'][0]}")
            ax[0].axis('off')
            
            # 2. GT Mask
            ax[1].imshow(img_np, cmap='gray')
            ax[1].imshow(mask_np, cmap='Reds', alpha=0.5 if has_gt else 0, vmin=0, vmax=1)
            ax[1].set_title("Ground Truth")
            ax[1].axis('off')
            
            # 3. Prediction
            ax[2].imshow(img_np, cmap='gray')
            ax[2].imshow(pred_np, cmap='Blues', alpha=0.5, vmin=0, vmax=1)
            ax[2].set_title(f"Prediction (Prob)\nMax: {pred_np.max():.2f}")
            ax[2].axis('off')
            
            # 4. Overlay (Green=TP, Red=FN, Blue=FP)
            # 製作 RGB 遮罩
            overlay_rgb = np.zeros((*img_np.shape, 3), dtype=float)
            
            gt_bin = mask_np > 0.5
            pred_bin = pred_np > 0.5
            
            # TP = Green
            overlay_rgb[gt_bin & pred_bin] = [0, 1, 0]
            # FN = Red
            overlay_rgb[gt_bin & ~pred_bin] = [1, 0, 0]
            # FP = Blue
            overlay_rgb[~gt_bin & pred_bin] = [0, 0.5, 1]
            
            # Mask alpha
            mask_alpha = (gt_bin | pred_bin).astype(float) * 0.4
            
            ax[3].imshow(img_np, cmap='gray')
            ax[3].imshow(overlay_rgb, alpha=mask_alpha)
            
            # Legend hack
            ax[3].text(5, 10, "TP", color="green", fontweight="bold")
            ax[3].text(5, 25, "FN (Miss)", color="red", fontweight="bold")
            ax[3].text(5, 40, "FP (Noise)", color="blue", fontweight="bold")
            
            ax[3].set_title(f"Overlay: {case_type}")
            ax[3].axis('off')
            
            save_path = vis_dir / f"{case_type}_{batch['patient_id'][0]}_{batch['slice_idx'][0]}_{batch['patch_idx'][0]}.png"
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close(fig)
            
            saved_count += 1

    logger.info(f"完成！已保存 {saved_count} 張圖片至 {vis_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True, help='訓練輸出目錄 (e.g. result/unetpp_lndb_...)')
    parser.add_argument('--num_samples', type=int, default=50, help='保存圖片數量')
    args = parser.parse_args()
    
    visualize_predictions(args.run_dir, num_samples=args.num_samples)
