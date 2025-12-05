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
    log_filename = f"training_{timestamp}.log"  # ✅ 改名為 training_*.log
    
    logger = logging.getLogger(__name__)
    
    # ✅ 修正：清空現有 handlers 避免重複
    if logger.handlers:
        logger.handlers.clear()
    
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # 檔案 handler
    file_handler = logging.FileHandler(
        str(log_path / log_filename), 
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 終端 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.propagate = False
    
    return logger


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
    """
    try:
        pred_np = (pred > 0.5).cpu().numpy().astype(bool)
        target_np = target.cpu().numpy().astype(bool)
        
        # 如果其中一個是空的
        if not pred_np.any() or not target_np.any():
            return 100.0
        
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
    
    metrics = {
        # 標準名稱
        'dice': dice_score,
        'iou': iou_score,
        'precision': precision_score,
        'recall': recall_score,
        'specificity': specificity_score,
        'accuracy': accuracy_score,
        'hausdorff_95': hd95_score,
        # 學術標準別名 (Academic Standard Aliases)
        'DSC': dice_score,           # Dice Similarity Coefficient
        'IoU': iou_score,            # Intersection over Union
        'SEN': recall_score,         # Sensitivity = Recall = TPR
        'PPV': precision_score,      # Positive Predictive Value = Precision
    }
    
    return metrics


class EarlyStopping:
    """
    早停機制：當驗證指標不再改善時停止訓練
    
    Args:
        patience: 容忍多少個 epoch 沒有改善
        min_delta: 最小改善幅度
        mode: 'max' 表示指標越大越好，'min' 表示越小越好
    """
    
    def __init__(self, patience: int = 7, min_delta: float = 0.001, mode: str = 'max'):
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


def save_dataset_split_info(train_patients, val_patients, test_patients, output_dir: str):
    """
    保存資料集分割資訊到 JSON 檔案
    
    Args:
        train_patients: 訓練集患者列表
        val_patients: 驗證集患者列表
        test_patients: 測試集患者列表
        output_dir: 輸出目錄
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
    
    save_path = output_path / 'dataset_split.json'
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)
    
    logger = logging.getLogger(__name__)
    logger.info(f"✅ 資料集分割資訊已保存: {save_path}")
    
    return split_info


class PatientMetricsTracker:
    """
    追蹤每個患者的評估指標，用於錯誤分析
    """
    
    def __init__(self):
        self.patient_metrics = {}  # {patient_id: {'slices': [], 'avg_metrics': {}}}
    
    def add_slice_metrics(self, patient_id: str, slice_idx: int, metrics: Dict[str, float]):
        """
        記錄單個切片的指標
        
        Args:
            patient_id: 患者 ID
            slice_idx: 切片索引
            metrics: 評估指標字典
        """
        if patient_id not in self.patient_metrics:
            self.patient_metrics[patient_id] = {'slices': []}
        
        slice_data = {'slice_index': slice_idx, **metrics}
        self.patient_metrics[patient_id]['slices'].append(slice_data)
    
    def compute_patient_averages(self):
        """
        計算每個患者的平均指標
        """
        for patient_id, data in self.patient_metrics.items():
            if not data['slices']:
                continue
            
            # 計算平均值
            metric_names = [k for k in data['slices'][0].keys() if k != 'slice_index']
            avg_metrics = {}
            
            for metric in metric_names:
                values = [s[metric] for s in data['slices'] if metric in s]
                avg_metrics[metric] = sum(values) / len(values) if values else 0.0
            
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
    
    def save_report(self, output_dir: str, split_name: str = 'test'):
        """
        保存詳細的錯誤分析報告
        
        Args:
            output_dir: 輸出目錄
            split_name: 資料集名稱（train/val/test）
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
        
        # 保存低分案例列表
        poor_performers = self.get_poor_performers(metric_name='dice', threshold=0.5)
        
        if poor_performers:
            error_report = {
                'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'split': split_name,
                'threshold': 0.5,
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
            logger.info(f"⚠️ 發現 {len(poor_performers)} 個低分病例（Dice < 0.5）")
            logger.info(f"   錯誤案例清單已保存: {error_path}")
        
        logger = logging.getLogger(__name__)
        logger.info(f"✅ 患者評估報告已保存: {report_path}")
        
        return report_path, poor_performers
