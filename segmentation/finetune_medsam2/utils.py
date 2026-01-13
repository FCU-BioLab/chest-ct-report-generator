#!/usr/bin/env python3
"""
工具函數模組
提供 logging、評估指標、資料分割等通用功能
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import warnings

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, binary_erosion


# =============================================================================
# 通用工具函數
# =============================================================================

def convert_to_serializable(obj):
    """
    將 NumPy 和 PyTorch 類型轉換為 JSON 可序列化的 Python 原生類型
    
    Args:
        obj: 要轉換的物件（支援嵌套的 dict/list）
        
    Returns:
        轉換後的 Python 原生物件
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    return obj


def setup_logging(log_dir: str = "finetune_logs") -> logging.Logger:
    """
    設定日誌系統（支援多進程安全）
    
    Args:
        log_dir: 日誌輸出目錄（將自動使用輸出目錄）
        
    Returns:
        配置好的 logger 物件
    """
    # ✅ 修正：只在主進程中創建 log 檔案
    if os.environ.get('PYTORCH_WORKER_ID') is not None:
        # DataLoader worker 進程：只輸出到終端
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            )
            logger.addHandler(console_handler)
            logger.propagate = False
        return logger
    
    # 主進程：創建完整的 logging 系統
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"training_{timestamp}.log"
    
    # ✅ 使用 root logger 確保所有模組的 log 都寫入同一個檔案
    root_logger = logging.getLogger()
    
    # 清空現有 handlers 避免重複
    root_logger.handlers.clear()
    
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # 檔案 handler
    file_handler = logging.FileHandler(
        str(log_path / log_filename), 
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # 終端 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 返回 root logger 供呼叫者使用
    return root_logger


def suppress_noisy_logs() -> None:
    """過濾 MedSAM2 和 Attention 機制的冗餘日誌"""
    
    class SAM2PredictorFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            pathname = (record.pathname or "").replace("\\", "/")
            return "sam2/sam2_image_predictor" not in pathname
    
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.addFilter(SAM2PredictorFilter())
    
    # 過濾 Attention 警告
    attention_messages = (
        "Memory efficient kernel not used because",
        "Memory Efficient attention has been runtime disabled",
        "Flash attention kernel not used because",
        "Torch was not compiled with flash attention",
        "CuDNN attention kernel not used because",
        "Expected query, key and value to all be of dtype",
    )
    
    for msg in attention_messages:
        warnings.filterwarnings("ignore", message=f".*{msg}.*", category=UserWarning)


def split_dataset(
    patient_ids: List[str], 
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: Optional[int] = None
) -> Tuple[List[str], List[str], List[str]]:
    """
    分割資料集為訓練集、驗證集、測試集
    
    Args:
        patient_ids: 患者ID列表
        train_ratio: 訓練集比例
        val_ratio: 驗證集比例
        test_ratio: 測試集比例
        seed: 隨機種子（None 表示隨機分割）
        
    Returns:
        (train_ids, val_ids, test_ids)
    """
    import random
    
    logger = logging.getLogger(__name__)
    
    if seed is not None:
        random.seed(seed)
        logger.info(f"🎲 使用固定隨機種子: {seed} (可重現切分)")
    else:
        logger.info(f"🎲 使用隨機切分 (每次執行都不同)")
    
    shuffled = patient_ids.copy()
    random.shuffle(shuffled)
    
    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_ids = shuffled[:train_end]
    val_ids = shuffled[train_end:val_end]
    test_ids = shuffled[val_end:]
    
    # 驗證無重複
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)
    
    assert len(train_set & val_set) == 0, "訓練集與驗證集有重複患者！"
    assert len(train_set & test_set) == 0, "訓練集與測試集有重複患者！"
    assert len(val_set & test_set) == 0, "驗證集與測試集有重複患者！"
    
    logger.info("✅ 資料集分割驗證通過：各集合間無重複患者")
    
    return train_ids, val_ids, test_ids


def compute_dice(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> float:
    """計算 Dice 係數 (F1 Score)"""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice.item()


def compute_iou(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> float:
    """計算 IoU (Intersection over Union / Jaccard Index)"""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def compute_precision(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """計算 Precision (陽性預測值)"""
    pred = (pred > 0.5).float()
    true_positive = (pred * target).sum()
    predicted_positive = pred.sum()
    precision = (true_positive + smooth) / (predicted_positive + smooth)
    return precision.item()


def compute_recall(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """計算 Recall (Sensitivity / True Positive Rate)"""
    pred = (pred > 0.5).float()
    true_positive = (pred * target).sum()
    actual_positive = target.sum()
    recall = (true_positive + smooth) / (actual_positive + smooth)
    return recall.item()


def compute_specificity(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """計算 Specificity (True Negative Rate)"""
    pred = (pred > 0.5).float()
    true_negative = ((1 - pred) * (1 - target)).sum()
    actual_negative = (1 - target).sum()
    specificity = (true_negative + smooth) / (actual_negative + smooth)
    return specificity.item()


def compute_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """計算 Accuracy (Pixel-wise Accuracy)"""
    pred = (pred > 0.5).float()
    correct = (pred == target).sum()
    total = target.numel()
    accuracy = correct / total
    return accuracy.item()


def compute_hausdorff_distance(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    計算 95th percentile Hausdorff Distance
    
    ✅ 修正：使用 binary_erosion 正確提取邊界
    ✅ 修正：空遮罩返回 NaN 以便排除統計
    """
    try:
        pred_np = (pred > 0.5).cpu().numpy().astype(bool)
        target_np = target.cpu().numpy().astype(bool)
        
        # 如果兩者都為空，返回 0（完美匹配空遮罩）
        if not pred_np.any() and not target_np.any():
            return 0.0
        
        # 如果只有一個是空的，返回 NaN（無法計算有意義的距離）
        if not pred_np.any() or not target_np.any():
            return float('nan')
        
        # ✅ 修正：使用 binary_erosion 提取邊界
        # 邊界 = 原始 mask - erosion(mask)
        struct = np.ones((3, 3), dtype=bool)
        pred_border = pred_np & ~binary_erosion(pred_np, structure=struct)
        target_border = target_np & ~binary_erosion(target_np, structure=struct)
        
        if not pred_border.any() or not target_border.any():
            return 0.0
        
        # 計算距離變換
        pred_dt = distance_transform_edt(~pred_np)
        target_dt = distance_transform_edt(~target_np)
        
        # 計算邊界距離
        pred_distances = pred_dt[target_border]
        target_distances = target_dt[pred_border]
        
        all_distances = np.concatenate([pred_distances, target_distances])
        
        if len(all_distances) == 0:
            return 0.0
        
        # 返回 95th percentile
        return float(np.percentile(all_distances, 95))
    
    except ImportError:
        logging.warning("scipy 未安裝，無法計算 Hausdorff Distance")
        return 0.0
    except Exception as e:
        logging.warning(f"Hausdorff Distance 計算失敗: {e}")
        return 0.0


def compute_all_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    計算所有評估指標
    
    Args:
        pred: 預測遮罩 (logits 或機率值)
        target: 目標遮罩
        
    Returns:
        包含所有指標的字典
        
    指標說明:
        - DSC (Dice Similarity Coefficient) = dice
        - IoU (Intersection over Union) = iou
        - SEN (Sensitivity) = recall
        - PPV (Positive Predictive Value) = precision
    """
    # 確保預測值為機率值
    pred_sigmoid = torch.sigmoid(pred) if pred.min() < 0 else pred
    
    # 計算基礎指標
    dice_score = compute_dice(pred_sigmoid, target)
    iou_score = compute_iou(pred_sigmoid, target)
    precision_score = compute_precision(pred_sigmoid, target)
    recall_score = compute_recall(pred_sigmoid, target)
    specificity_score = compute_specificity(pred_sigmoid, target)
    accuracy_score = compute_accuracy(pred_sigmoid, target)
    hd95_score = compute_hausdorff_distance(pred_sigmoid.squeeze(), target.squeeze())
    
    # ✅ 處理 HD95 的 NaN 值（空遮罩情況）
    import math
    if math.isnan(hd95_score):
        hd95_score = 0.0  # 或者可以排除此樣本
    
    metrics = {
        # 標準名稱
        'dice': dice_score,
        'iou': iou_score,
        'precision': precision_score,
        'recall': recall_score,
        'specificity': specificity_score,
        'accuracy': accuracy_score,
        'hausdorff_95': hd95_score,
    }
    
    return metrics


def compute_lightweight_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    ✅ 計算輕量級評估指標（跳過耗時的 Hausdorff Distance）
    
    適用於訓練過程中的驗證階段，加快驗證速度
    HD95 只在最終測試時計算
    
    Args:
        pred: 預測遮罩 (logits 或機率值)
        target: 目標遮罩
        
    Returns:
        包含輕量級指標的字典 (不含 hausdorff_95)
    """
    # 確保預測值為機率值
    pred_sigmoid = torch.sigmoid(pred) if pred.min() < 0 else pred
    
    # 計算基礎指標（跳過 HD95）
    metrics = {
        'dice': compute_dice(pred_sigmoid, target),
        'iou': compute_iou(pred_sigmoid, target),
        'precision': compute_precision(pred_sigmoid, target),
        'recall': compute_recall(pred_sigmoid, target),
        'specificity': compute_specificity(pred_sigmoid, target),
        'accuracy': compute_accuracy(pred_sigmoid, target),
        'hausdorff_95': 0.0,  # ✅ 佔位符，在測試時會重新計算
    }
    
    return metrics


class EarlyStopping:
    """
    早停機制：當驗證指標不再改善時停止訓練
    
    Args:
        patience: 容忍多少個 epoch 沒有改善
        min_delta: 最小改善幅度（對於 Dice 建議 0.005）
        mode: 'max' 表示指標越大越好，'min' 表示越小越好
    """
    
    def __init__(self, patience: int = 7, min_delta: float = 0.005, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, epoch: int, val_score: float) -> bool:
        """
        檢查是否應該早停
        
        Returns:
            True 表示應該停止訓練
        """
        if self.best_score is None:
            self.best_score = val_score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'max':
            improved = val_score > (self.best_score + self.min_delta)
        else:
            improved = val_score < (self.best_score - self.min_delta)
        
        if improved:
            self.best_score = val_score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                self.logger.info(f"⚠️ 早停觸發：{self.patience} 個 epoch 沒有改善")
                self.logger.info(f"   最佳分數: {self.best_score:.4f} (Epoch {self.best_epoch + 1})")
                return True
        
        return False


def custom_collate_fn(batch):
    """
    自定義 collate function 處理不同數量的 bounding boxes
    
    Args:
        batch: Dataset 輸出的 batch
        
    Returns:
        處理後的 batch 字典
    """
    images = torch.stack([item['image'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    bboxes = [item['bboxes'] for item in batch]  # 保持為 list
    patient_ids = [item['patient_id'] for item in batch]
    slice_indices = [item['slice_index'] for item in batch]
    
    return {
        'image': images,
        'mask': masks,
        'bboxes': bboxes,
        'patient_id': patient_ids,
        'slice_index': slice_indices
    }


def save_dataset_split_info(
    train_patients, 
    val_patients, 
    test_patients, 
    output_dir: str,
    original_split_file: str = None,
    filter_info: dict = None
):
    """
    保存資料集分割資訊到 JSON 檔案
    
    Args:
        train_patients: 訓練集患者列表
        val_patients: 驗證集患者列表
        test_patients: 測試集患者列表
        output_dir: 輸出目錄
        original_split_file: 原始分割檔案路徑（若有過濾）
        filter_info: 過濾資訊（若有過濾）
    """
    import json
    from pathlib import Path
    from datetime import datetime
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    split_info = {
        'split_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'train': {
            'count': len(train_patients),
            'patient_ids': sorted([str(p) for p in train_patients])
        },
        'val': {
            'count': len(val_patients),
            'patient_ids': sorted([str(p) for p in val_patients])
        },
        'test': {
            'count': len(test_patients),
            'patient_ids': sorted([str(p) for p in test_patients])
        },
        'total': len(train_patients) + len(val_patients) + len(test_patients)
    }
    
    # 加入過濾資訊（如果有）
    if original_split_file:
        split_info['original_split_file'] = original_split_file
    if filter_info:
        split_info['filter_info'] = filter_info
    
    save_path = output_path / 'dataset_split.json'
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)
    
    logger = logging.getLogger(__name__)
    logger.info(f"✅ 資料集分割資訊已保存: {save_path}")
    
    return split_info


def load_dataset_split_info(split_file_path: str):
    """
    從既有的 dataset_split.json 載入資料集分割資訊
    
    Args:
        split_file_path: dataset_split.json 檔案路徑
        
    Returns:
        tuple: (train_ids, val_ids, test_ids)
    """
    import json
    from pathlib import Path
    
    logger = logging.getLogger(__name__)
    
    split_path = Path(split_file_path)
    if not split_path.exists():
        raise FileNotFoundError(f"找不到分割檔案: {split_file_path}")
    
    with open(split_path, 'r', encoding='utf-8') as f:
        split_info = json.load(f)
    
    train_ids = split_info['train']['patient_ids']
    val_ids = split_info['val']['patient_ids']
    test_ids = split_info['test']['patient_ids']
    
    logger.info(f"✅ 從既有檔案載入資料集分割: {split_file_path}")
    logger.info(f"   原始分割日期: {split_info.get('split_date', 'N/A')}")
    logger.info(f"   Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    
    return train_ids, val_ids, test_ids


def compute_bbox_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    計算兩個 bounding box 的 IoU
    
    Args:
        bbox1, bbox2: [x1, y1, x2, y2] 格式
        
    Returns:
        IoU 值 (0-1)
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # 計算交集
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)
    
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    
    # 計算聯集
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def aggregate_3d_nodules(
    patient_features: Dict,
    iou_threshold: float = 0.3
) -> Tuple[int, Dict[int, List[int]]]:
    """
    將 2D 切片中的 bbox 聚合為 3D 結節
    
    使用連續切片中 bbox 的 IoU 來判斷是否為同一個 3D 結節
    
    Args:
        patient_features: 患者特徵字典，包含每個切片的 lesions
        iou_threshold: IoU 閾值，超過此值視為同一結節
        
    Returns:
        (unique_nodule_count, nodule_mapping: {nodule_id: [slice_indices]})
    """
    # 收集所有切片的 lesions
    slice_data = []
    for slice_idx, slice_info in patient_features.get('slices', {}).items():
        for lesion in slice_info.get('lesions', []):
            bbox = lesion.get('bbox', [])
            if len(bbox) >= 4:
                slice_data.append({
                    'slice_idx': int(slice_idx),
                    'bbox': bbox[:4],
                    'nodule_id': None
                })
    
    if not slice_data:
        return 0, {}
    
    # 按切片索引排序
    slice_data.sort(key=lambda x: x['slice_idx'])
    
    # 使用 Union-Find 來聚合結節
    current_nodule_id = 0
    nodule_mapping = {}  # nodule_id -> [slice_indices]
    
    for i, data in enumerate(slice_data):
        if data['nodule_id'] is not None:
            continue
            
        # 建立新結節
        data['nodule_id'] = current_nodule_id
        nodule_mapping[current_nodule_id] = [data['slice_idx']]
        
        # 檢查後續切片是否有相連的 bbox
        current_bbox = data['bbox']
        current_slice = data['slice_idx']
        
        for j in range(i + 1, len(slice_data)):
            other = slice_data[j]
            
            # 只檢查相鄰切片 (距離 <= 3)
            if other['slice_idx'] - current_slice > 3:
                break
            
            if other['nodule_id'] is not None:
                continue
            
            # 計算 IoU
            iou = compute_bbox_iou(current_bbox, other['bbox'])
            if iou >= iou_threshold:
                other['nodule_id'] = current_nodule_id
                nodule_mapping[current_nodule_id].append(other['slice_idx'])
                # 更新當前 bbox 為較新切片的 bbox (追蹤變化)
                current_bbox = other['bbox']
                current_slice = other['slice_idx']
        
        current_nodule_id += 1
    
    return len(nodule_mapping), nodule_mapping


class PatientMetricsTracker:
    """
    追蹤每個患者的評估指標，用於錯誤分析
    """

    
    def __init__(self):
        self.patient_metrics = {}  # {patient_id: {'slices': [], 'avg_metrics': {}, 'volume_metrics': {}}}
    
    def add_slice_metrics(self, patient_id: str, slice_idx: int, metrics: Dict[str, float], 
                         vol_stats: Optional[Dict[str, float]] = None):
        """
        記錄單個切片的指標
        
        Args:
            patient_id: 患者 ID
            slice_idx: 切片索引
            metrics: 評估指標字典
            vol_stats: (Optional) 體積統計 {'intersection': float, 'pred_area': float, 'gt_area': float}
        """
        if patient_id not in self.patient_metrics:
            self.patient_metrics[patient_id] = {'slices': [], 'vol_stats': {'intersection': 0.0, 'pred_area': 0.0, 'gt_area': 0.0, 'union': 0.0}}
        
        slice_data = {'slice_index': slice_idx, **metrics}
        self.patient_metrics[patient_id]['slices'].append(slice_data)
        
        # 累積體積統計
        if vol_stats:
            p_stats = self.patient_metrics[patient_id]['vol_stats']
            p_stats['intersection'] += vol_stats.get('intersection', 0.0)
            p_stats['pred_area'] += vol_stats.get('pred_area', 0.0)
            p_stats['gt_area'] += vol_stats.get('gt_area', 0.0)
            p_stats['union'] += vol_stats.get('union', 0.0)
    
    def compute_patient_averages(self):
        """
        計算每個患者的平均指標和體積指標
        """
        for patient_id, data in self.patient_metrics.items():
            if not data['slices']:
                continue
            
            # 1. 計算 slice-average metrics (原有機制)
            metric_names = [k for k in data['slices'][0].keys() if k != 'slice_index']
            avg_metrics = {}
            
            for metric in metric_names:
                values = [s[metric] for s in data['slices'] if metric in s]
                avg_metrics[metric] = sum(values) / len(values) if values else 0.0
            
            # 2. 計算 Volume Metrics (更準確的 3D 指標)
            vol_stats = data.get('vol_stats', {})
            intersection = vol_stats.get('intersection', 0.0)
            pred_area = vol_stats.get('pred_area', 0.0)
            gt_area = vol_stats.get('gt_area', 0.0)
            union = vol_stats.get('union', 0.0)
            
            smooth = 1e-6
            
            # Volume Dice
            vol_dice = (2.0 * intersection + smooth) / (pred_area + gt_area + smooth)
            
            # Volume IoU
            vol_iou = (intersection + smooth) / (union + smooth) if union > 0 else 0.0
            
            # Volume Recall / Precision
            vol_recall = (intersection + smooth) / (gt_area + smooth)
            vol_precision = (intersection + smooth) / (pred_area + smooth)
            
            # 將 Volume Metrics 加入 avg_metrics，並覆寫 Dice/IoU 為 Volume 版本 (因為這才是使用者關心的)
            # 或者使用前綴 'vol_' 來區分
            avg_metrics['vol_dice'] = vol_dice
            avg_metrics['vol_iou'] = vol_iou
            avg_metrics['vol_recall'] = vol_recall
            avg_metrics['vol_precision'] = vol_precision
            
            # ✅ 重要變更：將主 Dice/IoU 更新為 Volume Dice，這樣報告會顯示更合理的數值
            avg_metrics['slice_avg_dice'] = avg_metrics['dice'] # 保留舊的 slice avg 作為參考
            avg_metrics['dice'] = vol_dice
            avg_metrics['iou'] = vol_iou
            
            data['avg_metrics'] = avg_metrics
            data['num_slices'] = len(data['slices'])
    
    def get_poor_performers(self, metric_name: str = 'dice', threshold: float = 0.5):
        """
        找出表現不佳的患者
        
        Args:
            metric_name: 要檢查的指標名稱
            threshold: 閾值（低於此值視為表現不佳）
            
        Returns:
            [(patient_id, score), ...] 按分數排序
        """
        self.compute_patient_averages()
        
        poor_cases = []
        for patient_id, data in self.patient_metrics.items():
            if 'avg_metrics' in data and metric_name in data['avg_metrics']:
                score = data['avg_metrics'][metric_name]
                if score < threshold:
                    poor_cases.append((patient_id, score))
        
        # 按分數從低到高排序
        poor_cases.sort(key=lambda x: x[1])
        return poor_cases
    
    def save_report(self, output_dir: str, split_name: str = 'test', error_threshold: float = 0.75):
        """
        保存詳細的錯誤分析報告
        
        Args:
            output_dir: 輸出目錄
            split_name: 資料集名稱（train/val/test）
            error_threshold: 低分閾值（Dice 低於此值視為表現不佳）
        """
        import json
        from pathlib import Path
        from datetime import datetime
        
        self.compute_patient_averages()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 準備報告資料
        report = {
            'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'split': split_name,
            'total_patients': len(self.patient_metrics),
            'patients': {}
        }
        
        for patient_id, data in self.patient_metrics.items():
            report['patients'][patient_id] = {
                'num_slices': data.get('num_slices', 0),
                'avg_metrics': data.get('avg_metrics', {}),
                'slices': data.get('slices', [])
            }
        
        # 保存完整報告
        report_path = output_path / f'{split_name}_patient_metrics.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 保存低分案例列表（使用傳入的閾值）
        poor_performers = self.get_poor_performers(metric_name='dice', threshold=error_threshold)
        
        # 總是寫入 error_cases.json（即使沒有低分案例）
        error_report = {
            'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'split': split_name,
            'threshold': error_threshold,
            'metric': 'dice',
            'total_poor_cases': len(poor_performers),
            'cases': [
                {
                    'patient_id': pid,
                    'dice_score': float(score),
                    'num_slices': self.patient_metrics[pid].get('num_slices', 0),
                    'all_metrics': self.patient_metrics[pid].get('avg_metrics', {})
                }
                for pid, score in poor_performers
            ]
        }
        
        error_path = output_path / f'{split_name}_error_cases.json'
        with open(error_path, 'w', encoding='utf-8') as f:
            json.dump(error_report, f, indent=2, ensure_ascii=False)
        
        logger = logging.getLogger(__name__)
        if poor_performers:
            logger.info(f"⚠️ 發現 {len(poor_performers)} 個低分病例（Dice < {error_threshold}）")
        else:
            logger.info(f"✅ 沒有低分病例（Dice < {error_threshold}）")
        logger.info(f"   錯誤案例清單已保存: {error_path}")
        
        logger.info(f"✅ 患者評估報告已保存: {report_path}")
        
        return report_path, poor_performers
