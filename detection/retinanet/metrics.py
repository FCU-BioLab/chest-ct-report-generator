"""
RetinaNet 偵測評估指標
======================

計算所有偵測相關指標，包括：
- COCO mAP / AP / AR（多 IoU 閾值）
- FROC（LUNA16 標準指標）
- Detection F1 / Precision / Recall（最佳閾值）
- ROC-AUC / PR-AUC
- 每個 IoU 閾值的詳細 AP
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# LUNA16 FROC 標準 FP/scan 閾值
FROC_FP_RATES = [0.125, 0.25, 0.5, 1, 2, 4, 8]


def compute_froc(
    pred_boxes_list: List[np.ndarray],
    pred_scores_list: List[np.ndarray],
    gt_boxes_list: List[np.ndarray],
    iou_thresh: float = 0.1,
    fp_rates: List[float] = None,
    box_iou_fn=None,
) -> Dict[str, float]:
    """
    計算 FROC (Free-Response ROC) 指標 — LUNA16 challenge 標準。

    Returns:
        dict: 包含 froc_score 和各 FP/scan 閾值的 sensitivity
    """
    if fp_rates is None:
        fp_rates = FROC_FP_RATES

    n_scans = len(pred_boxes_list)
    if n_scans == 0:
        return {"froc_score": 0.0}

    # 收集所有預測，附帶 scan index 和 score
    all_preds = []
    n_gt_total = 0

    for scan_idx in range(n_scans):
        pred_boxes = pred_boxes_list[scan_idx]
        pred_scores = pred_scores_list[scan_idx]
        gt_boxes = gt_boxes_list[scan_idx]
        n_gt_total += len(gt_boxes)

        for j in range(len(pred_boxes)):
            all_preds.append({
                "scan_idx": scan_idx,
                "box": pred_boxes[j],
                "score": float(pred_scores[j]),
            })

    if n_gt_total == 0:
        return {"froc_score": 0.0, "n_gt": 0}

    # 按分數降序排列
    all_preds.sort(key=lambda x: -x["score"])

    # 追蹤每個 scan 的已匹配 GT
    gt_matched = [np.zeros(len(gt_boxes_list[i]), dtype=bool) for i in range(n_scans)]

    # 逐預測計算 TP/FP
    tp_list = []
    fp_per_scan = np.zeros(n_scans)

    for pred in all_preds:
        scan_idx = pred["scan_idx"]
        pred_box = pred["box"]
        gt_boxes = gt_boxes_list[scan_idx]

        is_tp = False
        if len(gt_boxes) > 0:
            # 計算 IoU
            if box_iou_fn is not None:
                ious = box_iou_fn(
                    pred_box.reshape(1, -1), gt_boxes
                ).flatten()
            else:
                ious = _box_iou_numpy(pred_box.reshape(1, -1), gt_boxes).flatten()

            best_gt = np.argmax(ious)
            if ious[best_gt] >= iou_thresh and not gt_matched[scan_idx][best_gt]:
                is_tp = True
                gt_matched[scan_idx][best_gt] = True

        if is_tp:
            tp_list.append(1)
        else:
            tp_list.append(0)
            fp_per_scan[scan_idx] += 1

    # 計算各 FP/scan rate 下的 sensitivity
    cum_tp = np.cumsum(tp_list)
    cum_fp = np.cumsum([1 - t for t in tp_list])

    results = {}
    sensitivities = []

    for fp_rate in fp_rates:
        max_fp_total = fp_rate * n_scans
        # 找到 FP 數量 <= max_fp_total 的最大索引
        valid = np.where(cum_fp <= max_fp_total)[0]
        if len(valid) > 0:
            sens = cum_tp[valid[-1]] / n_gt_total
        else:
            sens = 0.0
        results[f"sensitivity_at_{fp_rate}_fp_per_scan"] = float(sens)
        sensitivities.append(sens)

    results["froc_score"] = float(np.mean(sensitivities))
    results["cpm_score"] = float(np.mean(sensitivities))  # LUNA16 CPM is exactly the average of these 7 sensitivities
    results["n_gt"] = n_gt_total
    results["n_scans"] = n_scans

    return results


def compute_detection_f1(
    pred_boxes_list: List[np.ndarray],
    pred_scores_list: List[np.ndarray],
    gt_boxes_list: List[np.ndarray],
    iou_thresh: float = 0.1,
    score_thresholds: List[float] = None,
    box_iou_fn=None,
) -> Dict[str, float]:
    """
    計算 Detection F1, Precision, Recall（逐閾值 + 最佳 F1）。

    Returns:
        dict: 包含 best_f1, best_precision, best_recall, best_threshold 等
    """
    if score_thresholds is None:
        # 從 0.05 到 0.95 每 0.05 一個
        score_thresholds = [round(t * 0.05, 2) for t in range(1, 20)]

    n_gt_total = sum(len(gt) for gt in gt_boxes_list)
    if n_gt_total == 0:
        return {"best_f1": 0.0, "best_precision": 0.0, "best_recall": 0.0, "best_threshold": 0.0}

    best_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0
    best_threshold = 0.0
    per_threshold = {}

    for thresh in score_thresholds:
        tp = 0
        fp = 0
        fn = 0

        for scan_idx in range(len(pred_boxes_list)):
            pred_boxes = pred_boxes_list[scan_idx]
            pred_scores = pred_scores_list[scan_idx]
            gt_boxes = gt_boxes_list[scan_idx]

            # 過濾低分預測
            if len(pred_scores) > 0:
                mask = pred_scores >= thresh
                filtered_boxes = pred_boxes[mask]
            else:
                filtered_boxes = np.array([]).reshape(0, 6)

            gt_matched = np.zeros(len(gt_boxes), dtype=bool)

            for pred_box in filtered_boxes:
                if len(gt_boxes) > 0:
                    if box_iou_fn is not None:
                        ious = box_iou_fn(
                            pred_box.reshape(1, -1), gt_boxes
                        ).flatten()
                    else:
                        ious = _box_iou_numpy(pred_box.reshape(1, -1), gt_boxes).flatten()

                    best_gt = np.argmax(ious)
                    if ious[best_gt] >= iou_thresh and not gt_matched[best_gt]:
                        tp += 1
                        gt_matched[best_gt] = True
                    else:
                        fp += 1
                else:
                    fp += 1

            fn += np.sum(~gt_matched)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_threshold[f"thresh_{thresh:.2f}"] = {
            "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn,
        }

        if f1 > best_f1:
            best_f1 = f1
            best_precision = precision
            best_recall = recall
            best_threshold = thresh

    return {
        "best_f1": float(best_f1),
        "best_precision": float(best_precision),
        "best_recall": float(best_recall),
        "best_threshold": float(best_threshold),
        "per_threshold": per_threshold,
    }


def compute_all_metrics(
    pred_boxes_list: List[np.ndarray],
    pred_scores_list: List[np.ndarray],
    pred_labels_list: List[np.ndarray],
    gt_boxes_list: List[np.ndarray],
    gt_labels_list: List[np.ndarray],
    iou_thresholds_coco: List[float] = None,
    iou_thresh_match: float = 0.1,
    coco_metric=None,
    matching_batch_fn=None,
    box_iou_fn=None,
) -> Dict:
    """
    計算所有偵測指標，統一入口。

    Returns:
        dict: 包含所有指標
    """
    from sklearn.metrics import (
        roc_curve, auc, precision_recall_curve, average_precision_score,
    )

    results = {}

    # ─── 1. COCO mAP / AP / AR ───
    if coco_metric is not None and matching_batch_fn is not None:
        logger.info("  📐 計算 COCO 指標...")
        match_results = matching_batch_fn(
            iou_fn=box_iou_fn,
            iou_thresholds=coco_metric.iou_thresholds,
            pred_boxes=[b for b in pred_boxes_list],
            pred_classes=[l for l in pred_labels_list],
            pred_scores=[s for s in pred_scores_list],
            gt_boxes=[b for b in gt_boxes_list],
            gt_classes=[l for l in gt_labels_list],
        )
        coco_dict = coco_metric(match_results)[0]
        results["coco"] = {k: float(v) for k, v in coco_dict.items()}
        results["mAP"] = float(sum(coco_dict.values()) / len(coco_dict)) if coco_dict else 0.0

        for k, v in coco_dict.items():
            logger.info(f"    {k}: {v:.4f}")

    # ─── 2. FROC (LUNA16 標準) ───
    logger.info("  📐 計算 FROC / CPM (LUNA16) 指標...")
    froc = compute_froc(
        pred_boxes_list, pred_scores_list, gt_boxes_list,
        iou_thresh=iou_thresh_match,
        box_iou_fn=box_iou_fn,
    )
    results["froc"] = froc
    logger.info(f"    FROC score / CPM: {froc['cpm_score']:.4f}")
    for fp_rate in FROC_FP_RATES:
        key = f"sensitivity_at_{fp_rate}_fp_per_scan"
        if key in froc:
            logger.info(f"    {key}: {froc[key]:.4f}")

    # ─── 3. Detection F1 / Precision / Recall ───
    logger.info("  📐 計算 Detection F1...")
    f1_metrics = compute_detection_f1(
        pred_boxes_list, pred_scores_list, gt_boxes_list,
        iou_thresh=iou_thresh_match,
        box_iou_fn=box_iou_fn,
    )
    results["detection_f1"] = f1_metrics["best_f1"]
    results["detection_precision"] = f1_metrics["best_precision"]
    results["detection_recall"] = f1_metrics["best_recall"]
    results["detection_best_threshold"] = f1_metrics["best_threshold"]
    results["f1_per_threshold"] = f1_metrics["per_threshold"]

    logger.info(f"    Best F1: {f1_metrics['best_f1']:.4f} "
                f"(P={f1_metrics['best_precision']:.4f}, "
                f"R={f1_metrics['best_recall']:.4f}, "
                f"thresh={f1_metrics['best_threshold']:.2f})")

    # ─── 4. Scan-level Classification Accuracy ───
    logger.info("  📐 計算 Scan-level 分類準確率...")
    tp_scan, fp_scan, tn_scan, fn_scan = 0, 0, 0, 0
    thresh = f1_metrics["best_threshold"]
    
    for i in range(len(pred_boxes_list)):
        has_gt = len(gt_boxes_list[i]) > 0
        has_pred = len(pred_scores_list[i][pred_scores_list[i] >= thresh]) > 0
        
        if has_gt and has_pred:
            tp_scan += 1
        elif has_gt and not has_pred:
            fn_scan += 1
        elif not has_gt and not has_pred:
            tn_scan += 1
        elif not has_gt and has_pred:
            fp_scan += 1
            
    total_scans = tp_scan + tn_scan + fp_scan + fn_scan
    scan_accuracy = (tp_scan + tn_scan) / total_scans if total_scans > 0 else 0.0
    scan_specificity = tn_scan / (tn_scan + fp_scan) if (tn_scan + fp_scan) > 0 else 0.0
    
    results["scan_accuracy"] = float(scan_accuracy)
    results["scan_specificity"] = float(scan_specificity)
    results["scan_tp"] = int(tp_scan)
    results["scan_tn"] = int(tn_scan)
    results["scan_fp"] = int(fp_scan)
    results["scan_fn"] = int(fn_scan)

    logger.info(f"    Scan Accuracy: {scan_accuracy:.4f} (TP={tp_scan}, TN={tn_scan}, FP={fp_scan}, FN={fn_scan})")

    # ─── 5. ROC / PR curves ───
    logger.info("  📐 計算 ROC / PR 曲線...")
    y_true_all = []
    y_score_all = []

    for i in range(len(pred_boxes_list)):
        gt_boxes = gt_boxes_list[i]
        pred_boxes = pred_boxes_list[i]
        pred_scores = pred_scores_list[i]

        if len(pred_boxes) == 0:
            continue

        if len(gt_boxes) > 0:
            if box_iou_fn is not None:
                iou_matrix = box_iou_fn(pred_boxes, gt_boxes)
            else:
                iou_matrix = _box_iou_numpy(pred_boxes, gt_boxes)
            max_iou = np.max(iou_matrix, axis=1)
            matches = max_iou >= iou_thresh_match
            y_true_all.extend(matches.astype(int))
        else:
            y_true_all.extend(np.zeros(len(pred_boxes), dtype=int))
        y_score_all.extend(pred_scores)

    roc_auc_val, pr_ap_val = 0.0, 0.0
    roc_fpr, roc_tpr = [], []
    pr_precision, pr_recall = [], []

    if len(y_score_all) > 0:
        y_true_all = np.array(y_true_all)
        y_score_all = np.array(y_score_all)

        if len(np.unique(y_true_all)) > 1:
            roc_fpr, roc_tpr, _ = roc_curve(y_true_all, y_score_all)
            roc_auc_val = auc(roc_fpr, roc_tpr)
            pr_precision, pr_recall, _ = precision_recall_curve(y_true_all, y_score_all)
            pr_ap_val = average_precision_score(y_true_all, y_score_all)

    results["roc_auc"] = float(roc_auc_val)
    results["pr_ap"] = float(pr_ap_val)
    # 只在完整 metrics dict 中保留曲線數據（不印出）
    results["_curves"] = {
        "roc_fpr": [float(x) for x in roc_fpr],
        "roc_tpr": [float(x) for x in roc_tpr],
        "pr_precision": [float(x) for x in pr_precision],
        "pr_recall": [float(x) for x in pr_recall],
    }

    logger.info(f"    ROC-AUC: {roc_auc_val:.4f}")
    logger.info(f"    PR-AP: {pr_ap_val:.4f}")

    # ─── 5. 統計摘要 ───
    n_gt = sum(len(gt) for gt in gt_boxes_list)
    n_pred = sum(len(p) for p in pred_boxes_list)
    n_scans = len(pred_boxes_list)
    n_scans_with_gt = sum(1 for gt in gt_boxes_list if len(gt) > 0)
    avg_fp_per_scan = sum(
        len(pred_scores_list[i][pred_scores_list[i] >= f1_metrics["best_threshold"]])
        for i in range(n_scans)
    ) / n_scans if n_scans > 0 else 0

    results["summary"] = {
        "n_scans": n_scans,
        "n_scans_with_gt": n_scans_with_gt,
        "n_gt_total": n_gt,
        "n_pred_total": n_pred,
        "avg_fp_per_scan_at_best_thresh": float(avg_fp_per_scan),
    }

    logger.info(f"    Scans: {n_scans} ({n_scans_with_gt} with nodule)")
    logger.info(f"    GT boxes: {n_gt}, Pred boxes: {n_pred}")

    return results


def _box_iou_numpy(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    計算 3D bounding box IoU (numpy fallback)。
    boxes: [N, 6] 格式 [x1, y1, z1, x2, y2, z2]
    """
    n1 = len(boxes1)
    n2 = len(boxes2)
    iou_matrix = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            x1 = max(boxes1[i, 0], boxes2[j, 0])
            y1 = max(boxes1[i, 1], boxes2[j, 1])
            z1 = max(boxes1[i, 2], boxes2[j, 2])
            x2 = min(boxes1[i, 3], boxes2[j, 3])
            y2 = min(boxes1[i, 4], boxes2[j, 4])
            z2 = min(boxes1[i, 5], boxes2[j, 5])

            inter = max(0, x2 - x1) * max(0, y2 - y1) * max(0, z2 - z1)

            vol1 = ((boxes1[i, 3] - boxes1[i, 0]) *
                     (boxes1[i, 4] - boxes1[i, 1]) *
                     (boxes1[i, 5] - boxes1[i, 2]))
            vol2 = ((boxes2[j, 3] - boxes2[j, 0]) *
                     (boxes2[j, 4] - boxes2[j, 1]) *
                     (boxes2[j, 5] - boxes2[j, 2]))

            union = vol1 + vol2 - inter
            iou_matrix[i, j] = inter / union if union > 0 else 0.0

    return iou_matrix
