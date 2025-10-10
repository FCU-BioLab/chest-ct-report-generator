#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv7 Evaluation Visualizer

功能：
- 每個 epoch 結束後的預測可視化
- TP/FP/FN 標註
- 低 confidence threshold 評估
- 統計預測錯誤類型

Usage:
    visualizer = YOLOv7EvalVisualizer(save_dir='./yolov7_logs')
    visualizer.visualize_epoch(model, val_loader, epoch=1, conf_threshold=0.001)
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import json
import logging

LOGGER = logging.getLogger(__name__)


@dataclass
class DetectionStats:
    """檢測統計"""
    epoch: int
    tp: int  # True Positives
    fp: int  # False Positives
    fn: int  # False Negatives
    total_predictions: int
    total_gt: int
    precision: float
    recall: float
    f1: float


def box_iou_numpy(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    計算兩個 box 的 IoU
    
    Args:
        box1, box2: [x_center, y_center, w, h] normalized
    
    Returns:
        IoU value
    """
    # Convert to x1, y1, x2, y2
    b1_x1 = box1[0] - box1[2] / 2
    b1_y1 = box1[1] - box1[3] / 2
    b1_x2 = box1[0] + box1[2] / 2
    b1_y2 = box1[1] + box1[3] / 2
    
    b2_x1 = box2[0] - box2[2] / 2
    b2_y1 = box2[1] - box2[3] / 2
    b2_x2 = box2[0] + box2[2] / 2
    b2_y2 = box2[1] + box2[3] / 2
    
    # Intersection area
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    # Union area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area
    
    iou = inter_area / union_area if union_area > 0 else 0
    return iou


def non_max_suppression_simple(
    predictions: np.ndarray,
    conf_threshold: float = 0.001,
    iou_threshold: float = 0.45
) -> np.ndarray:
    """
    簡單的 NMS 實作
    
    Args:
        predictions: (N, 6) [x, y, w, h, conf, class]
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold
    
    Returns:
        Filtered predictions
    """
    if len(predictions) == 0:
        return np.zeros((0, 6))
    
    # Filter by confidence
    keep = predictions[:, 4] >= conf_threshold
    predictions = predictions[keep]
    
    if len(predictions) == 0:
        return np.zeros((0, 6))
    
    # Sort by confidence
    order = predictions[:, 4].argsort()[::-1]
    predictions = predictions[order]
    
    keep_boxes = []
    while len(predictions) > 0:
        # Keep highest confidence box
        keep_boxes.append(predictions[0])
        
        if len(predictions) == 1:
            break
        
        # Calculate IoU with remaining boxes
        ious = np.array([box_iou_numpy(predictions[0][:4], box[:4]) for box in predictions[1:]])
        
        # Remove boxes with high IoU
        keep_mask = ious < iou_threshold
        predictions = predictions[1:][keep_mask]
    
    return np.array(keep_boxes) if keep_boxes else np.zeros((0, 6))


class YOLOv7EvalVisualizer:
    """
    YOLOv7 評估可視化器
    
    每個 epoch 後進行：
    1. 低 conf threshold 預測
    2. 計算 TP/FP/FN
    3. 繪製對比圖
    4. 統計並保存報告
    """
    
    def __init__(
        self,
        save_dir: str = "./yolov7_logs",
        num_vis_samples: int = 20,
        iou_threshold: float = 0.5
    ):
        """
        Args:
            save_dir: 保存目錄
            num_vis_samples: 可視化樣本數
            iou_threshold: IoU threshold for TP
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.num_vis_samples = num_vis_samples
        self.iou_threshold = iou_threshold
        
        # 統計歷史
        self.stats_history: List[DetectionStats] = []
    
    @torch.no_grad()
    def predict_batch(
        self,
        model: nn.Module,
        images: torch.Tensor,
        conf_threshold: float = 0.001,
        nms_iou: float = 0.45
    ) -> List[np.ndarray]:
        """
        對一批圖像進行預測
        
        Args:
            model: YOLOv7 model
            images: (B, C, H, W)
            conf_threshold: Confidence threshold
            nms_iou: NMS IoU threshold
        
        Returns:
            List of predictions per image [(N, 6) for x, y, w, h, conf, class]
        """
        model.eval()
        outputs = model(images)
        
        # Handle different output formats
        if isinstance(outputs, tuple):
            outputs = outputs[0]  # Take first output
        
        batch_predictions = []
        
        for i in range(len(images)):
            if isinstance(outputs, list):
                # Multiple detection layers
                pred = outputs[0][i]  # Use first layer
            else:
                pred = outputs[i]
            
            # Convert to numpy
            if isinstance(pred, torch.Tensor):
                pred = pred.cpu().numpy()
            
            # Apply NMS
            pred_nms = non_max_suppression_simple(pred, conf_threshold, nms_iou)
            batch_predictions.append(pred_nms)
        
        return batch_predictions
    
    def compute_tp_fp_fn(
        self,
        predictions: np.ndarray,
        ground_truths: np.ndarray,
        iou_threshold: float = 0.5
    ) -> Tuple[int, int, int, List[int], List[int]]:
        """
        計算 TP, FP, FN
        
        Args:
            predictions: (N, 6) [x, y, w, h, conf, class]
            ground_truths: (M, 5) [class, x, y, w, h]
            iou_threshold: IoU threshold for TP
        
        Returns:
            (tp, fp, fn, matched_pred_idx, matched_gt_idx)
        """
        if len(predictions) == 0 and len(ground_truths) == 0:
            return 0, 0, 0, [], []
        
        if len(predictions) == 0:
            return 0, 0, len(ground_truths), [], []
        
        if len(ground_truths) == 0:
            return 0, len(predictions), 0, [], []
        
        # Match predictions to ground truths
        matched_pred = set()
        matched_gt = set()
        
        # Sort predictions by confidence
        pred_order = predictions[:, 4].argsort()[::-1]
        
        for pred_idx in pred_order:
            pred = predictions[pred_idx]
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx in range(len(ground_truths)):
                if gt_idx in matched_gt:
                    continue
                
                gt = ground_truths[gt_idx]
                # GT format: [class, x, y, w, h]
                iou = box_iou_numpy(pred[:4], gt[1:5])
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold:
                matched_pred.add(pred_idx)
                matched_gt.add(best_gt_idx)
        
        tp = len(matched_pred)
        fp = len(predictions) - tp
        fn = len(ground_truths) - len(matched_gt)
        
        return tp, fp, fn, list(matched_pred), list(matched_gt)
    
    def draw_detection_comparison(
        self,
        image: np.ndarray,
        predictions: np.ndarray,
        ground_truths: np.ndarray,
        matched_pred_idx: List[int],
        matched_gt_idx: List[int],
        title: str = ""
    ) -> np.ndarray:
        """
        繪製檢測對比圖
        
        Args:
            image: (H, W, 3) RGB image
            predictions: (N, 6) [x, y, w, h, conf, class]
            ground_truths: (M, 5) [class, x, y, w, h]
            matched_pred_idx: Matched prediction indices (TP)
            matched_gt_idx: Matched GT indices
            title: Image title
        
        Returns:
            Annotated image
        """
        img_vis = image.copy()
        h, w = img_vis.shape[:2]
        
        # Draw ground truths (green for matched, blue for unmatched FN)
        for gt_idx, gt in enumerate(ground_truths):
            cls_id, x_c, y_c, box_w, box_h = gt
            
            x1 = int((x_c - box_w / 2) * w)
            y1 = int((y_c - box_h / 2) * h)
            x2 = int((x_c + box_w / 2) * w)
            y2 = int((y_c + box_h / 2) * h)
            
            if gt_idx in matched_gt_idx:
                color = (0, 255, 0)  # Green (TP)
                label = "GT(TP)"
            else:
                color = (255, 255, 0)  # Yellow (FN)
                label = "GT(FN)"
            
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw predictions (green for TP, red for FP)
        for pred_idx, pred in enumerate(predictions):
            x_c, y_c, box_w, box_h, conf, cls_id = pred
            
            x1 = int((x_c - box_w / 2) * w)
            y1 = int((y_c - box_h / 2) * h)
            x2 = int((x_c + box_w / 2) * w)
            y2 = int((y_c + box_h / 2) * h)
            
            if pred_idx in matched_pred_idx:
                color = (0, 255, 0)  # Green (TP)
                label = f"TP({conf:.2f})"
            else:
                color = (255, 0, 0)  # Red (FP)
                label = f"FP({conf:.2f})"
            
            # Draw dashed line for predictions
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 1, lineType=cv2.LINE_4)
            cv2.putText(img_vis, label, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add title
        if title:
            cv2.putText(img_vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return img_vis
    
    @torch.no_grad()
    def visualize_epoch(
        self,
        model: nn.Module,
        val_loader: torch.utils.data.DataLoader,
        epoch: int,
        conf_threshold: float = 0.001,
        nms_iou: float = 0.45,
        device: str = 'cuda'
    ) -> DetectionStats:
        """
        可視化一個 epoch 的評估結果
        
        Args:
            model: YOLOv7 model
            val_loader: Validation dataloader
            epoch: Current epoch
            conf_threshold: Confidence threshold
            nms_iou: NMS IoU threshold
            device: Device
        
        Returns:
            DetectionStats
        """
        model.eval()
        
        # Create epoch save directory
        epoch_dir = self.save_dir / f"vis_epoch_{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_predictions = 0
        total_gt = 0
        
        # Collect samples for visualization
        vis_samples = []
        
        LOGGER.info(f"\n{'='*80}")
        LOGGER.info(f"Epoch {epoch} Evaluation Visualization")
        LOGGER.info(f"  Conf threshold: {conf_threshold}, NMS IoU: {nms_iou}")
        LOGGER.info(f"{'='*80}\n")
        
        for batch_idx, (images, labels, metadata) in enumerate(tqdm(val_loader, desc=f"Evaluating Epoch {epoch}")):
            images = images.to(device)
            
            # Predict
            batch_predictions = self.predict_batch(model, images, conf_threshold, nms_iou)
            
            # Process each image in batch
            for i in range(len(images)):
                # Get predictions
                predictions = batch_predictions[i]
                
                # Get ground truths (labels format: [batch_idx, class, x, y, w, h])
                img_labels = labels[labels[:, 0] == i][:, 1:]  # Remove batch_idx
                ground_truths = img_labels.cpu().numpy()
                
                # Compute TP/FP/FN
                tp, fp, fn, matched_pred, matched_gt = self.compute_tp_fp_fn(
                    predictions, ground_truths, self.iou_threshold
                )
                
                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_predictions += len(predictions)
                total_gt += len(ground_truths)
                
                # Save for visualization
                if len(vis_samples) < self.num_vis_samples:
                    # Convert image to numpy
                    img_np = images[i].cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
                    img_np = (img_np * 255).astype(np.uint8)
                    
                    vis_samples.append({
                        'image': img_np,
                        'predictions': predictions,
                        'ground_truths': ground_truths,
                        'matched_pred': matched_pred,
                        'matched_gt': matched_gt,
                        'metadata': metadata[i],
                        'tp': tp,
                        'fp': fp,
                        'fn': fn
                    })
        
        # Calculate metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        stats = DetectionStats(
            epoch=epoch,
            tp=total_tp,
            fp=total_fp,
            fn=total_fn,
            total_predictions=total_predictions,
            total_gt=total_gt,
            precision=precision,
            recall=recall,
            f1=f1
        )
        
        # Save statistics
        self.stats_history.append(stats)
        
        # Generate visualizations
        LOGGER.info(f"\nGenerating visualizations for {len(vis_samples)} samples...")
        
        for idx, sample in enumerate(tqdm(vis_samples, desc="Rendering")):
            img_vis = self.draw_detection_comparison(
                image=sample['image'],
                predictions=sample['predictions'],
                ground_truths=sample['ground_truths'],
                matched_pred_idx=sample['matched_pred'],
                matched_gt_idx=sample['matched_gt'],
                title=f"Epoch {epoch} | TP:{sample['tp']} FP:{sample['fp']} FN:{sample['fn']}"
            )
            
            # Save image
            save_path = epoch_dir / f"sample_{idx:03d}_{sample['metadata']['patient_id']}.png"
            cv2.imwrite(str(save_path), cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))
        
        # Generate summary report
        self._generate_epoch_report(epoch_dir, stats, vis_samples)
        
        # Print summary
        LOGGER.info(f"\n{'='*80}")
        LOGGER.info(f"Epoch {epoch} Evaluation Summary")
        LOGGER.info(f"{'='*80}")
        LOGGER.info(f"Total Predictions: {total_predictions}")
        LOGGER.info(f"Total Ground Truths: {total_gt}")
        LOGGER.info(f"TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
        LOGGER.info(f"Precision: {precision:.4f}")
        LOGGER.info(f"Recall: {recall:.4f}")
        LOGGER.info(f"F1-Score: {f1:.4f}")
        LOGGER.info(f"Visualizations saved to: {epoch_dir}")
        LOGGER.info(f"{'='*80}\n")
        
        return stats
    
    def _generate_epoch_report(
        self,
        epoch_dir: Path,
        stats: DetectionStats,
        samples: List[Dict]
    ):
        """生成 epoch 報告"""
        report_path = epoch_dir / "evaluation_report.md"
        
        md = f"""# Epoch {stats.epoch} Evaluation Report

## 📊 Overall Statistics

| Metric | Value |
|--------|-------|
| Total Predictions | {stats.total_predictions} |
| Total Ground Truths | {stats.total_gt} |
| True Positives (TP) | {stats.tp} |
| False Positives (FP) | {stats.fp} |
| False Negatives (FN) | {stats.fn} |
| **Precision** | **{stats.precision:.4f}** |
| **Recall** | **{stats.recall:.4f}** |
| **F1-Score** | **{stats.f1:.4f}** |

## 🔍 Sample Analysis

"""
        
        for idx, sample in enumerate(samples[:20]):
            md += f"""
### Sample #{idx+1}: {sample['metadata']['patient_id']}

- **TP**: {sample['tp']}, **FP**: {sample['fp']}, **FN**: {sample['fn']}
- Predictions: {len(sample['predictions'])}, Ground Truths: {len(sample['ground_truths'])}

![Sample {idx+1}](sample_{idx:03d}_{sample['metadata']['patient_id']}.png)

---
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(md)
        
        # Save JSON stats
        json_path = epoch_dir / "stats.json"
        with open(json_path, 'w') as f:
            json.dump(asdict(stats), f, indent=2)
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """繪製訓練曲線"""
        if not self.stats_history:
            return
        
        epochs = [s.epoch for s in self.stats_history]
        precisions = [s.precision for s in self.stats_history]
        recalls = [s.recall for s in self.stats_history]
        f1s = [s.f1 for s in self.stats_history]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].plot(epochs, precisions, marker='o')
        axes[0].set_title('Precision')
        axes[0].set_xlabel('Epoch')
        axes[0].grid(True)
        
        axes[1].plot(epochs, recalls, marker='o', color='orange')
        axes[1].set_title('Recall')
        axes[1].set_xlabel('Epoch')
        axes[1].grid(True)
        
        axes[2].plot(epochs, f1s, marker='o', color='green')
        axes[2].set_title('F1-Score')
        axes[2].set_xlabel('Epoch')
        axes[2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(self.save_dir / "training_curves.png", dpi=150, bbox_inches='tight')
        
        plt.close()


if __name__ == "__main__":
    print("YOLOv7 Evaluation Visualizer")
    print("Usage: Import and use YOLOv7EvalVisualizer class in training script")

