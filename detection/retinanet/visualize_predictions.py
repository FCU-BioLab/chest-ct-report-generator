#!/usr/bin/env python3
"""
生成 RetinaNet 預測結果的 GIF 動畫
====================================

對驗證集中有結節的 CT 掃描執行推論，
在每個 axial slice 上繪製預測的 bounding box，
然後輸出逐 slice 播放的 GIF。

用法:
    python -m detection.retinanet.visualize_predictions \
        --data_path dataset_luna16.json \
        --num_samples 5 \
        --score_thresh 0.3
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def draw_boxes_on_slice(
    slice_2d: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    z_idx: int,
    color_pred=(1.0, 0.3, 0.3),  # 紅色 - 預測
    color_gt=(0.3, 1.0, 0.3),    # 綠色 - GT
    gt_boxes: np.ndarray = None,
):
    """
    在 2D slice 上繪製 bounding box。
    boxes 格式: [x1, y1, z1, x2, y2, z2] (voxel 座標)
    只繪製 z1 <= z_idx <= z2 的框。
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=80)
    ax.imshow(slice_2d.T, cmap="gray", origin="lower", vmin=0, vmax=1)
    ax.set_axis_off()

    # 繪製 GT boxes
    if gt_boxes is not None and len(gt_boxes) > 0:
        for box in gt_boxes:
            x1, y1, z1, x2, y2, z2 = box
            if z1 <= z_idx <= z2:
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor=color_gt, facecolor="none",
                    linestyle="--",
                )
                ax.add_patch(rect)
                ax.text(
                    x1, y1 - 3, "GT",
                    color=color_gt, fontsize=8, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.5),
                )

    # 繪製預測 boxes
    if len(boxes) > 0:
        for box, score in zip(boxes, scores):
            x1, y1, z1, x2, y2, z2 = box
            if z1 <= z_idx <= z2:
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor=color_pred, facecolor="none",
                )
                ax.add_patch(rect)
                ax.text(
                    x1, y1 - 3, f"{score:.2f}",
                    color=color_pred, fontsize=8, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.5),
                )

    ax.set_title(f"Slice {z_idx}", fontsize=10, color="white",
                 bbox=dict(facecolor="black", alpha=0.7))

    fig.tight_layout(pad=0.3)
    fig.canvas.draw()

    # Convert to numpy array (compatible with newer matplotlib)
    buf = np.asarray(fig.canvas.buffer_rgba())
    # RGBA -> RGB
    buf = buf[:, :, :3].copy()
    plt.close(fig)
    return buf


def create_prediction_gif(
    image: np.ndarray,
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    output_path: str,
    score_thresh: float = 0.3,
    fps: int = 8,
    skip_empty: bool = False,
):
    """
    建立一個 GIF 動畫，逐 axial slice 顯示預測框。
    image: [C, H, W, D] 或 [H, W, D]
    boxes: [N, 6] — [x1, y1, z1, x2, y2, z2]
    """
    from PIL import Image as PILImage

    if image.ndim == 4:
        image = image[0]  # 取第一個 channel

    # 只保留高於閾值的預測
    if len(pred_scores) > 0:
        mask = pred_scores >= score_thresh
        pred_boxes = pred_boxes[mask]
        pred_scores = pred_scores[mask]

    H, W, D = image.shape
    frames = []

    # 找出有框的 slice 範圍（擴展前後各 5 slices）
    z_with_content = set()
    for box in pred_boxes:
        z1, z2 = int(box[2]), int(box[5])
        for z in range(max(0, z1 - 5), min(D, z2 + 6)):
            z_with_content.add(z)
    if gt_boxes is not None:
        for box in gt_boxes:
            z1, z2 = int(box[2]), int(box[5])
            for z in range(max(0, z1 - 5), min(D, z2 + 6)):
                z_with_content.add(z)

    # 如果沒有任何框，顯示中間 30 slices
    if not z_with_content:
        mid = D // 2
        z_with_content = set(range(max(0, mid - 15), min(D, mid + 15)))

    z_slices = sorted(z_with_content)

    logger.info(f"    生成 {len(z_slices)} 個 frames (D={D})...")

    for z in z_slices:
        frame = draw_boxes_on_slice(
            image[:, :, z],
            pred_boxes, pred_scores, z,
            gt_boxes=gt_boxes,
        )
        frames.append(PILImage.fromarray(frame))

    if frames:
        duration_ms = int(1000 / fps)
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
        )
        logger.info(f"    ✅ GIF 已儲存: {output_path} ({len(frames)} frames)")
    else:
        logger.warning(f"    ⚠️ 無 frames 可生成 GIF")


def main():
    parser = argparse.ArgumentParser(description="生成 RetinaNet 預測 GIF")
    parser.add_argument("--data_path", default="dataset_luna16.json", help="資料路徑")
    parser.add_argument("--pretrained_weights", default=None, help="預訓練權重路徑")
    parser.add_argument("--num_samples", type=int, default=5, help="生成幾個 GIF")
    parser.add_argument("--score_thresh", type=float, default=0.3, help="顯示的分數閾值")
    parser.add_argument("--output_dir", default=None, help="輸出目錄")
    parser.add_argument("--device", default="cuda", help="裝置")
    parser.add_argument("--no_amp", action="store_true", help="停用 AMP")
    parser.add_argument("--fps", type=int, default=8, help="GIF FPS")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)
    args = parser.parse_args()

    # 設定輸出目錄
    if args.output_dir is None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"detection/results/predict_gifs_{ts}"

    os.makedirs(args.output_dir, exist_ok=True)

    # 建立模型
    from .config import RetinaNetConfig
    from .trainer import RetinaNetTrainer
    from .dataset import prepare_datalist, build_val_transform

    config = RetinaNetConfig(
        data_path=args.data_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
        amp=not args.no_amp,
        device=args.device,
        num_workers=0,
        cache_dataset=False,  # 不需要快取，只跑幾個樣本
    )
    if args.pretrained_weights is not None:
        config.pretrained_weights = args.pretrained_weights

    config.output_dir = args.output_dir

    trainer = RetinaNetTrainer(config)

    # 取得驗證集中有結節的樣本
    val_data = prepare_datalist(
        config.data_path, "val",
        config.train_ratio, config.val_ratio, config.test_ratio, config.split_seed,
    )
    # 挑選有結節的樣本
    nodule_samples = [d for d in val_data if len(d.get("box", [])) > 0]
    logger.info(f"驗證集中有結節的樣本: {len(nodule_samples)}/{len(val_data)}")

    # 取前 N 個
    samples = nodule_samples[:args.num_samples]
    logger.info(f"將生成 {len(samples)} 個 GIF 動畫")

    # 建立 transform
    val_transform = build_val_transform(
        spacing=config.spacing,
        hu_min=config.hu_min,
        hu_max=config.hu_max,
    )

    trainer.detector.eval()

    for i, sample in enumerate(samples):
        sample_name = Path(sample["image"]).stem[:30]
        logger.info(f"\n🔍 [{i+1}/{len(samples)}] 處理: {sample_name}")

        # 套用 transform
        t0 = time.time()
        transformed = val_transform(sample.copy())

        image = transformed["image"]  # [C, H, W, D]
        gt_boxes = transformed["box"].numpy() if torch.is_tensor(transformed["box"]) else np.array(transformed["box"])

        logger.info(f"    影像 shape: {image.shape}, GT boxes: {len(gt_boxes)}")

        # 推論
        image_input = image.unsqueeze(0).to(config.device)  # [1, C, H, W, D]

        use_inferer = image_input[0, 0].numel() > np.prod(config.val_patch_size)

        with torch.no_grad():
            if config.amp:
                with torch.amp.autocast("cuda"):
                    outputs = trainer.detector(
                        [image_input[0]], use_inferer=use_inferer
                    )
            else:
                outputs = trainer.detector(
                    [image_input[0]], use_inferer=use_inferer
                )

        pred_out = outputs[0]
        pred_boxes = pred_out[trainer.detector.target_box_key].cpu().numpy()
        pred_scores = pred_out[trainer.detector.pred_score_key].cpu().numpy()

        elapsed = time.time() - t0
        n_above = (pred_scores >= args.score_thresh).sum() if len(pred_scores) > 0 else 0
        logger.info(f"    推論耗時: {elapsed:.1f}s")
        logger.info(f"    預測框: {len(pred_boxes)} (score≥{args.score_thresh}: {n_above})")

        # 生成 GIF
        image_np = image.cpu().numpy()
        gif_path = os.path.join(args.output_dir, f"pred_{i+1}_{sample_name}.gif")

        create_prediction_gif(
            image_np, pred_boxes, pred_scores, gt_boxes,
            output_path=gif_path,
            score_thresh=args.score_thresh,
            fps=args.fps,
        )

        # 釋放 GPU 記憶體
        del image_input, outputs
        torch.cuda.empty_cache()

    logger.info(f"\n🎉 完成！GIF 已儲存至: {args.output_dir}")


if __name__ == "__main__":
    main()
