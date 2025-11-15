#!/usr/bin/env python3
"""
MedSAM2 Fine-tuning for Chest Tumor Segmentation
================================================

使用NIfTI格式的胸部CT腫瘤資料集微調MedSAM2模型

主要功能:
1. 載入NIfTI格式的CT和腫瘤遮罩
2. 自動生成bounding box prompts
3. 資料增強 (Data Augmentation)
4. 微調MedSAM2模型
5. 驗證和評估
6. 保存最佳模型

使用範例:
    # 開始訓練
    python finetune_medsam2.py --epochs 50 --batch_size 4 --lr 1e-5
    
    # 從checkpoint繼續訓練
    python finetune_medsam2.py --resume checkpoints/best_model.pth
    
    # 只評估模型
    python finetune_medsam2.py --eval_only --checkpoint checkpoints/best_model.pth
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import json
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import nibabel as nib
from skimage import measure
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import MedSAM2
try:
    medsam2_path = Path(__file__).parent / "MedSAM2"
    if str(medsam2_path) not in sys.path:
        sys.path.insert(0, str(medsam2_path))
    
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    MEDSAM2_AVAILABLE = True
except ImportError as e:
    print(f"❌ MedSAM2 import error: {e}")
    MEDSAM2_AVAILABLE = False
    sys.exit(1)


def setup_logging(log_dir: str = "finetune_logs") -> logging.Logger:
    """設定日誌"""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"finetune_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(str(log_path / log_filename), encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()


def suppress_noisy_logs() -> None:
    """Filter verbose MedSAM2 predictor logs and recurring attention warnings."""

    class _SAM2PredictorFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            pathname = (record.pathname or "").replace("\\", "/")
            return "sam2/sam2_image_predictor" not in pathname

    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.addFilter(_SAM2PredictorFilter())

    attention_messages = (
        "Memory efficient kernel not used because",
        "Memory Efficient attention has been runtime disabled",
        "Flash attention kernel not used because",
        "Torch was not compiled with flash attention",
        "CuDNN attention kernel not used because",
        "Experimental cuDNN SDPA nested tensor support does not support backward",
        "Expected query, key and value to all be of dtype",
        "Flash Attention kernel failed",
    )

    for msg in attention_messages:
        warnings.filterwarnings(
            "ignore",
            message=f".*{msg}.*",
            category=UserWarning,
        )


suppress_noisy_logs()


class ChestTumorDataset(Dataset):
    """
    胸部CT腫瘤資料集
    
    載入NIfTI格式的CT和腫瘤遮罩，自動提取2D切片和bounding boxes
    """
    
    def __init__(self, 
                 data_dir: str,
                 patient_ids: List[str],
                 axis: int = 2,
                 transform=None,
                 cache_data: bool = False):
        """
        Args:
            data_dir: 患者資料目錄
            patient_ids: 患者ID列表
            axis: 切片軸向 (0, 1, 或 2)
            transform: 資料增強函數
            cache_data: 是否緩存資料到記憶體
        """
        self.data_dir = Path(data_dir)
        self.patient_ids = patient_ids
        self.axis = axis
        self.transform = transform
        self.cache_data = cache_data
        
        # 建立所有切片的索引
        self.samples = []
        self._build_sample_index()
        
        # 緩存資料
        if cache_data:
            logger.info("🔄 緩存資料到記憶體...")
            self.cached_data = {}
            for idx in tqdm(range(len(self.samples))):
                self.cached_data[idx] = self._load_sample(idx)
    
    def _build_sample_index(self):
        """建立所有有效切片的索引"""
        logger.info(f"📊 建立資料集索引 ({len(self.patient_ids)} 個患者)...")
        
        for patient_id in tqdm(self.patient_ids):
            ct_file = self.data_dir / patient_id / f"{patient_id}_CT.nii.gz"
            tumor_file = self.data_dir / patient_id / f"{patient_id}_tumor.nii.gz"
            
            if not ct_file.exists() or not tumor_file.exists():
                continue
            
            try:
                # 載入體積
                tumor_volume = nib.load(str(tumor_file)).get_fdata()
                
                # 檢查每個切片是否有腫瘤
                depth = tumor_volume.shape[self.axis]
                for slice_idx in range(depth):
                    if self.axis == 0:
                        tumor_slice = tumor_volume[slice_idx, :, :]
                    elif self.axis == 1:
                        tumor_slice = tumor_volume[:, slice_idx, :]
                    else:
                        tumor_slice = tumor_volume[:, :, slice_idx]
                    
                    if tumor_slice.sum() > 0:  # 有腫瘤
                        self.samples.append({
                            'patient_id': patient_id,
                            'ct_file': ct_file,
                            'tumor_file': tumor_file,
                            'slice_index': slice_idx
                        })
            
            except Exception as e:
                logger.warning(f"⚠️ 跳過患者 {patient_id}: {e}")
                continue
        
        logger.info(f"✅ 找到 {len(self.samples)} 個有效切片")
    
    def _load_sample(self, idx: int) -> Dict:
        """載入單個樣本"""
        sample_info = self.samples[idx]
        
        # 載入體積
        ct_volume = nib.load(str(sample_info['ct_file'])).get_fdata()
        tumor_volume = nib.load(str(sample_info['tumor_file'])).get_fdata()
        
        # 提取切片
        slice_idx = sample_info['slice_index']
        if self.axis == 0:
            ct_slice = ct_volume[slice_idx, :, :]
            tumor_slice = tumor_volume[slice_idx, :, :]
        elif self.axis == 1:
            ct_slice = ct_volume[:, slice_idx, :]
            tumor_slice = tumor_volume[:, slice_idx, :]
        else:
            ct_slice = ct_volume[:, :, slice_idx]
            tumor_slice = tumor_volume[:, :, slice_idx]
        
        # 標準化CT切片到0-255
        ct_min, ct_max = ct_slice.min(), ct_slice.max()
        if ct_max > ct_min:
            ct_normalized = ((ct_slice - ct_min) / (ct_max - ct_min) * 255).astype(np.uint8)
        else:
            ct_normalized = np.zeros_like(ct_slice, dtype=np.uint8)
        
        # 轉換為RGB (MedSAM2需要)
        ct_rgb = np.stack([ct_normalized] * 3, axis=-1)
        
        # 提取bounding boxes
        bboxes = self._extract_bboxes(tumor_slice)
        
        # 二值化遮罩
        mask = (tumor_slice > 0).astype(np.uint8)
        
        return {
            'image': ct_rgb,
            'mask': mask,
            'bboxes': bboxes,
            'patient_id': sample_info['patient_id'],
            'slice_index': slice_idx
        }
    
    def _extract_bboxes(self, mask: np.ndarray) -> np.ndarray:
        """從遮罩提取bounding boxes"""
        if mask.sum() == 0:
            return np.array([[0, 0, 1, 1]])  # 空遮罩
        
        labeled_mask = measure.label(mask > 0)
        regions = measure.regionprops(labeled_mask)
        
        bboxes = []
        for region in regions:
            minr, minc, maxr, maxc = region.bbox
            bboxes.append([minc, minr, maxc, maxr])  # [x1, y1, x2, y2]
        
        return np.array(bboxes)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        # 從緩存或磁碟載入
        if self.cache_data:
            data = self.cached_data[idx].copy()
        else:
            data = self._load_sample(idx)
        
        # 資料增強
        if self.transform:
            data = self.transform(data)
        
        # 轉換為tensor
        image = torch.from_numpy(data['image']).permute(2, 0, 1).float()  # [3, H, W]
        mask = torch.from_numpy(data['mask']).unsqueeze(0).float()  # [1, H, W]
        bboxes = torch.from_numpy(data['bboxes']).float()  # [N, 4]
        
        return {
            'image': image,
            'mask': mask,
            'bboxes': bboxes,
            'patient_id': data['patient_id'],
            'slice_index': data['slice_index']
        }


class DiceLoss(nn.Module):
    """Dice Loss用於分割任務"""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """結合Dice Loss和Binary Cross Entropy Loss"""
    
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 應用sigmoid到預測值
        pred_sigmoid = torch.sigmoid(pred)
        
        # 計算兩種損失
        dice = self.dice_loss(pred_sigmoid, target)
        bce = self.bce_loss(pred, target)
        
        return self.dice_weight * dice + self.bce_weight * bce


class MedSAM2Trainer:
    """MedSAM2訓練器"""
    
    def __init__(self,
                 model_config: str = "sam2.1_hiera_t512.yaml",
                 checkpoint_path: Optional[str] = None,
                 device: str = "cuda",
                 output_dir: str = "finetune_output"):
        """
        Args:
            model_config: MedSAM2配置檔案
            checkpoint_path: 預訓練模型路徑
            device: 計算設備
            output_dir: 輸出目錄
        """
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 載入模型
        logger.info(f"🔧 載入MedSAM2模型: {model_config}")
        self._load_model(model_config, checkpoint_path)
        
        # 損失函數
        self.criterion = CombinedLoss(dice_weight=0.5, bce_weight=0.5)
        
        # 訓練歷史 - 包含所有評估指標
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'val_dice': [],
            'val_iou': [],
            'val_precision': [],
            'val_recall': [],
            'val_specificity': [],
            'val_hausdorff_95': [],
            'learning_rate': []
        }
        
        self.best_val_dice = 0.0
        self.current_epoch = 0
    
    def _load_model(self, config: str, checkpoint: Optional[str]):
        """載入MedSAM2模型"""
        from hydra import initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        
        # 使用絕對路徑
        medsam2_path = Path(__file__).parent / "MedSAM2"
        config_dir = str((medsam2_path / "sam2" / "configs").absolute())
        
        if not Path(config_dir).exists():
            raise FileNotFoundError(f"Config directory not found: {config_dir}")
        
        initialize_config_dir(config_dir=config_dir, version_base="1.2")
        config_name = config.replace('.yaml', '')
        
        # 建立模型
        if checkpoint and Path(checkpoint).exists():
            logger.info(f"📥 從checkpoint載入: {checkpoint}")
            self.model = build_sam2(config_name, checkpoint, device=self.device)
        else:
            logger.info(f"🆕 建立新模型")
            self.model = build_sam2(config_name, device=self.device)
        
        self.predictor = SAM2ImagePredictor(self.model)
        
        # 只訓練mask decoder和prompt encoder
        for name, param in self.model.named_parameters():
            if "image_encoder" in name:
                param.requires_grad = False  # 凍結image encoder
            else:
                param.requires_grad = True
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"✅ 可訓練參數: {trainable_params:,}")

    def _prepare_image_features(self, image: torch.Tensor) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Compute MedSAM2 image embeddings once per slice."""

        with torch.no_grad():
            np_image = image.permute(1, 2, 0).cpu().numpy()
            self.predictor.set_image(np_image)

        features = self.predictor._features or {}
        image_embedding = features.get("image_embed")
        if image_embedding is None:
            raise RuntimeError("Predictor features unavailable after set_image call")

        high_res_feats = features.get("high_res_feats")
        return image_embedding, high_res_feats
    
    def train_epoch(self, train_loader: DataLoader, optimizer, scheduler) -> float:
        """訓練一個epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")
        for batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            bboxes = batch['bboxes']
            
            optimizer.zero_grad()
            batch_loss = 0.0
            batch_samples = 0
            
            # 處理batch中的每個樣本
            for i in range(len(images)):
                image = images[i]  # [3, H, W]
                gt_mask = masks[i]  # [1, H, W]
                bbox_tensor = bboxes[i]

                if len(bbox_tensor) == 0:
                    continue

                image_embedding, high_res_feats = self._prepare_image_features(image)
                bbox_tensor = bbox_tensor.to(self.device)

                sample_loss = 0.0
                valid_boxes = 0

                for bbox in bbox_tensor:
                    if bbox.sum() == 0:
                        continue

                    box_torch = bbox.unsqueeze(0)
                    sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
                        points=None,
                        boxes=box_torch,
                        masks=None,
                    )

                    low_res_masks, _, _, _ = self.model.sam_mask_decoder(
                        image_embeddings=image_embedding,
                        image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                        repeat_image=False,
                        high_res_features=high_res_feats,
                    )

                    pred_mask = F.interpolate(
                        low_res_masks,
                        size=(gt_mask.shape[-2], gt_mask.shape[-1]),
                        mode='bilinear',
                        align_corners=False
                    )

                    loss = self.criterion(pred_mask.squeeze(0), gt_mask)
                    sample_loss += loss
                    valid_boxes += 1

                if valid_boxes > 0:
                    sample_loss = sample_loss / valid_boxes
                    batch_loss += sample_loss
                    batch_samples += 1
            
            # 反向傳播
            if batch_samples > 0:
                batch_loss = batch_loss / batch_samples
                batch_loss.backward()
                optimizer.step()
                
                total_loss += batch_loss.item()
                num_batches += 1
                pbar.set_postfix({'loss': f'{batch_loss.item():.4f}'})
        
        scheduler.step()
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """驗證模型 - 返回loss和所有評估指標"""
        self.model.eval()
        total_loss = 0.0
        
        # 初始化指標累加器
        metrics_sum = {
            'dice': 0.0,
            'iou': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'specificity': 0.0,
            'hausdorff_95': 0.0
        }
        num_samples = 0
        
        pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch+1} [Val]")
        for batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            bboxes = batch['bboxes']
            
            for i in range(len(images)):
                image = images[i]
                gt_mask = masks[i]
                bbox_tensor = bboxes[i]

                if len(bbox_tensor) == 0:
                    continue

                self._prepare_image_features(image)

                sample_metrics = {k: 0.0 for k in metrics_sum.keys()}
                sample_loss = 0.0
                valid_boxes = 0

                for bbox in bbox_tensor:
                    if bbox.sum() == 0:
                        continue

                    masks_pred, _, _ = self.predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=bbox.cpu().numpy()[None, :],
                        multimask_output=False
                    )

                    pred_mask = torch.from_numpy(masks_pred[0]).unsqueeze(0).to(self.device)

                    loss = self.criterion(pred_mask, gt_mask)
                    sample_loss += loss.item()
                    valid_boxes += 1

                    batch_metrics = self.compute_all_metrics(pred_mask, gt_mask)
                    for key, value in batch_metrics.items():
                        sample_metrics[key] += value

                if valid_boxes > 0:
                    normalized_loss = sample_loss / valid_boxes
                    total_loss += normalized_loss

                    for key in metrics_sum.keys():
                        metrics_sum[key] += sample_metrics[key] / valid_boxes

                    num_samples += 1

                    pbar.set_postfix({
                        'loss': f'{normalized_loss:.4f}',
                        'dice': f'{sample_metrics["dice"] / valid_boxes:.4f}',
                        'iou': f'{sample_metrics["iou"] / valid_boxes:.4f}'
                    })
        
        # 計算平均值
        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        avg_metrics = {k: v / num_samples if num_samples > 0 else 0.0 
                      for k, v in metrics_sum.items()}
        
        return avg_loss, avg_metrics
    
    def _compute_dice(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> float:
        """計算Dice係數 (F1 Score)"""
        pred = (pred > 0.5).float()
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return dice.item()
    
    def _compute_iou(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> float:
        """計算IoU (Intersection over Union / Jaccard Index)"""
        pred = (pred > 0.5).float()
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        iou = (intersection + smooth) / (union + smooth)
        return iou.item()
    
    def _compute_precision(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
        """計算Precision (陽性預測值)"""
        pred = (pred > 0.5).float()
        true_positive = (pred * target).sum()
        predicted_positive = pred.sum()
        precision = (true_positive + smooth) / (predicted_positive + smooth)
        return precision.item()
    
    def _compute_recall(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
        """計算Recall (Sensitivity / True Positive Rate)"""
        pred = (pred > 0.5).float()
        true_positive = (pred * target).sum()
        actual_positive = target.sum()
        recall = (true_positive + smooth) / (actual_positive + smooth)
        return recall.item()
    
    def _compute_specificity(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
        """計算Specificity (True Negative Rate)"""
        pred = (pred > 0.5).float()
        true_negative = ((1 - pred) * (1 - target)).sum()
        actual_negative = (1 - target).sum()
        specificity = (true_negative + smooth) / (actual_negative + smooth)
        return specificity.item()
    
    def _compute_hausdorff_distance(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """計算95th percentile Hausdorff Distance"""
        try:
            from scipy.ndimage import distance_transform_edt
            
            pred_np = (pred > 0.5).cpu().numpy().astype(bool)
            target_np = target.cpu().numpy().astype(bool)
            
            # 如果其中一個是空的
            if not pred_np.any() or not target_np.any():
                return 100.0  # 返回一個大值
            
            # 計算距離變換
            pred_dt = distance_transform_edt(~pred_np)
            target_dt = distance_transform_edt(~target_np)
            
            # 計算表面距離
            pred_surface = pred_np ^ np.roll(pred_np, 1, axis=0) | pred_np ^ np.roll(pred_np, 1, axis=1)
            target_surface = target_np ^ np.roll(target_np, 1, axis=0) | target_np ^ np.roll(target_np, 1, axis=1)
            
            pred_distances = pred_dt[target_surface]
            target_distances = target_dt[pred_surface]
            
            all_distances = np.concatenate([pred_distances, target_distances])
            
            if len(all_distances) == 0:
                return 0.0
            
            # 返回95th percentile
            return float(np.percentile(all_distances, 95))
        
        except ImportError:
            logger.warning("scipy未安裝，無法計算Hausdorff Distance")
            return 0.0
        except Exception as e:
            logger.warning(f"Hausdorff Distance計算失敗: {e}")
            return 0.0
    
    def compute_all_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """計算所有評估指標"""
        pred_sigmoid = torch.sigmoid(pred) if pred.min() < 0 else pred
        
        metrics = {
            'dice': self._compute_dice(pred_sigmoid, target),
            'iou': self._compute_iou(pred_sigmoid, target),
            'precision': self._compute_precision(pred_sigmoid, target),
            'recall': self._compute_recall(pred_sigmoid, target),
            'specificity': self._compute_specificity(pred_sigmoid, target),
            'hausdorff_95': self._compute_hausdorff_distance(pred_sigmoid.squeeze(), target.squeeze())
        }
        
        return metrics
    
    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int = 50,
            learning_rate: float = 1e-5,
            weight_decay: float = 1e-4):
        """訓練模型"""
        
        # 優化器和學習率調度器
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 開始訓練")
        logger.info(f"{'='*80}")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Learning Rate: {learning_rate}")
        logger.info(f"Train samples: {len(train_loader.dataset)}")
        logger.info(f"Val samples: {len(val_loader.dataset)}")
        logger.info(f"{'='*80}\n")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # 訓練
            train_loss = self.train_epoch(train_loader, optimizer, scheduler)
            
            # 驗證 - 現在返回所有指標
            val_loss, val_metrics = self.validate(val_loader)
            
            # 記錄歷史
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_dice'].append(val_metrics['dice'])
            self.train_history['val_iou'].append(val_metrics['iou'])
            self.train_history['val_precision'].append(val_metrics['precision'])
            self.train_history['val_recall'].append(val_metrics['recall'])
            self.train_history['val_specificity'].append(val_metrics['specificity'])
            self.train_history['val_hausdorff_95'].append(val_metrics['hausdorff_95'])
            self.train_history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            # 輸出結果
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n"
                f"  Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}, "
                f"Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}\n"
                f"  Specificity: {val_metrics['specificity']:.4f}, "
                f"Hausdorff95: {val_metrics['hausdorff_95']:.2f}, "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )
            
            # 保存最佳模型（基於Dice）
            if val_metrics['dice'] > self.best_val_dice:
                self.best_val_dice = val_metrics['dice']
                self.save_checkpoint('best_model.pth', is_best=True)
                logger.info(f"✅ 保存最佳模型 (Dice: {val_metrics['dice']:.4f})")
            
            # 定期保存checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ 訓練完成！最佳Dice: {self.best_val_dice:.4f}")
        logger.info(f"{'='*80}\n")
        
        # 繪製訓練曲線
        self.plot_training_curves()
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """保存checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'best_val_dice': self.best_val_dice,
            'train_history': self.train_history
        }
        
        checkpoint_path = self.output_dir / filename
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            logger.info(f"💾 最佳模型已保存: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """載入checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_dice = checkpoint['best_val_dice']
        self.train_history = checkpoint['train_history']
        logger.info(f"✅ Checkpoint已載入: {checkpoint_path}")
    
    def plot_training_curves(self):
        """繪製訓練曲線（包含所有評估指標）"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
        epochs = range(1, len(self.train_history['train_loss']) + 1)
        
        # 1. Loss曲線
        axes[0, 0].plot(epochs, self.train_history['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(epochs, self.train_history['val_loss'], label='Val Loss', color='red')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. Dice Score曲線
        axes[0, 1].plot(epochs, self.train_history['val_dice'], label='Val Dice', color='green', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Score')
        axes[0, 1].set_title('Dice Score (F1)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        axes[0, 1].set_ylim([0, 1])
        
        # 3. IoU曲線
        axes[0, 2].plot(epochs, self.train_history['val_iou'], label='Val IoU', color='purple', linewidth=2)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('IoU')
        axes[0, 2].set_title('IoU/Jaccard Index')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        axes[0, 2].set_ylim([0, 1])
        
        # 4. Precision曲線
        axes[1, 0].plot(epochs, self.train_history['val_precision'], label='Val Precision', color='orange', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision (PPV)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        axes[1, 0].set_ylim([0, 1])
        
        # 5. Recall曲線
        axes[1, 1].plot(epochs, self.train_history['val_recall'], label='Val Recall', color='cyan', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].set_title('Recall/Sensitivity (TPR)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_ylim([0, 1])
        
        # 6. Specificity曲線
        axes[1, 2].plot(epochs, self.train_history['val_specificity'], label='Val Specificity', color='magenta', linewidth=2)
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Specificity')
        axes[1, 2].set_title('Specificity (TNR)')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        axes[1, 2].set_ylim([0, 1])
        
        # 7. Hausdorff Distance曲線
        axes[2, 0].plot(epochs, self.train_history['val_hausdorff_95'], label='Val HD95', color='brown', linewidth=2)
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('Hausdorff Distance (pixels)')
        axes[2, 0].set_title('Hausdorff Distance (95th percentile)')
        axes[2, 0].legend()
        axes[2, 0].grid(True)
        
        # 8. Learning Rate曲線
        axes[2, 1].plot(epochs, self.train_history['learning_rate'], color='darkblue', linewidth=2)
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('Learning Rate')
        axes[2, 1].set_title('Learning Rate Schedule')
        axes[2, 1].set_yscale('log')
        axes[2, 1].grid(True)
        
        # 9. 摘要統計
        axes[2, 2].axis('off')
        summary_text = f"""
Training Summary
================

Best Val Dice: {self.best_val_dice:.4f}
Final Metrics:
  Dice: {self.train_history['val_dice'][-1]:.4f}
  IoU: {self.train_history['val_iou'][-1]:.4f}
  Precision: {self.train_history['val_precision'][-1]:.4f}
  Recall: {self.train_history['val_recall'][-1]:.4f}
  Specificity: {self.train_history['val_specificity'][-1]:.4f}
  HD95: {self.train_history['val_hausdorff_95'][-1]:.2f}

Total Epochs: {len(epochs)}
        """
        axes[2, 2].text(0.1, 0.5, summary_text, fontsize=11, family='monospace', verticalalignment='center')
        
        plt.tight_layout()
        
        plot_path = self.output_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"📊 訓練曲線已保存: {plot_path}")
        plt.close()


def custom_collate_fn(batch):
    """
    自定義 collate function 處理不同數量的 bounding boxes
    
    Args:
        batch: List of dicts from dataset
    
    Returns:
        Dict with batched tensors (images, masks) and list of bboxes
    """
    images = torch.stack([item['image'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    bboxes = [item['bboxes'] for item in batch]  # 保持為 list，不 stack
    patient_ids = [item['patient_id'] for item in batch]
    slice_indices = [item['slice_index'] for item in batch]
    
    return {
        'image': images,
        'mask': masks,
        'bboxes': bboxes,
        'patient_id': patient_ids,
        'slice_index': slice_indices
    }


def split_dataset(patient_ids: List[str], 
                  train_ratio: float = 0.7,
                  val_ratio: float = 0.15,
                  test_ratio: float = 0.15,
                  seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """分割資料集"""
    import random
    random.seed(seed)
    
    shuffled = patient_ids.copy()
    random.shuffle(shuffled)
    
    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_ids = shuffled[:train_end]
    val_ids = shuffled[train_end:val_end]
    test_ids = shuffled[val_end:]
    
    return train_ids, val_ids, test_ids


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MedSAM2 for Chest Tumor Segmentation")
    
    # 資料參數
    parser.add_argument("--data_dir", type=str, default="../datasets/all_patient_data",
                       help="患者資料目錄")
    parser.add_argument("--axis", type=int, default=2, choices=[0, 1, 2],
                       help="切片軸向")
    
    # 訓練參數
    parser.add_argument("--epochs", type=int, default=50,
                       help="訓練輪數")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-5,
                       help="學習率")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                       help="權重衰減")
    
    # 模型參數
    parser.add_argument("--config", type=str, default="sam2.1_hiera_t512.yaml",
                       help="MedSAM2配置檔案")
    parser.add_argument("--checkpoint", type=str, default="MedSAM2/checkpoints/MedSAM2_latest.pt",
                       help="預訓練模型路徑")
    parser.add_argument("--resume", type=str, default=None,
                       help="從checkpoint繼續訓練")
    
    # 其他參數
    parser.add_argument("--output_dir", type=str, default="finetune_output",
                       help="輸出目錄")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="DataLoader工作進程數")
    parser.add_argument("--cache_data", action="store_true",
                       help="緩存資料到記憶體")
    parser.add_argument("--eval_only", action="store_true",
                       help="只進行評估")
    parser.add_argument("--seed", type=int, default=42,
                       help="隨機種子")
    
    args = parser.parse_args()
    
    # 設定隨機種子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 檢查CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🖥️ 使用設備: {device}")
    
    # 獲取所有患者ID
    data_dir = Path(args.data_dir)
    all_patients = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    logger.info(f"📊 找到 {len(all_patients)} 個患者")
    
    # 分割資料集
    train_ids, val_ids, test_ids = split_dataset(all_patients, seed=args.seed)
    logger.info(f"📊 資料集分割: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
    
    # 建立資料集
    logger.info("🔧 建立資料集...")
    train_dataset = ChestTumorDataset(
        args.data_dir, train_ids, axis=args.axis, cache_data=args.cache_data
    )
    val_dataset = ChestTumorDataset(
        args.data_dir, val_ids, axis=args.axis, cache_data=args.cache_data
    )
    
    # 建立DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device == "cuda" else False,
        collate_fn=custom_collate_fn  # 使用自定義 collate function
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device == "cuda" else False,
        collate_fn=custom_collate_fn  # 使用自定義 collate function
    )
    
    # 建立訓練器
    trainer = MedSAM2Trainer(
        model_config=args.config,
        checkpoint_path=args.checkpoint if not args.resume else None,
        device=device,
        output_dir=args.output_dir
    )
    
    # 載入checkpoint (如果需要)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 評估模式
    if args.eval_only:
        logger.info("🔍 評估模式")
        val_loss, val_metrics = trainer.validate(val_loader)
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ 評估結果:")
        logger.info(f"  Loss: {val_loss:.4f}")
        logger.info(f"  Dice Score: {val_metrics['dice']:.4f}")
        logger.info(f"  IoU/Jaccard: {val_metrics['iou']:.4f}")
        logger.info(f"  Precision: {val_metrics['precision']:.4f}")
        logger.info(f"  Recall/Sensitivity: {val_metrics['recall']:.4f}")
        logger.info(f"  Specificity: {val_metrics['specificity']:.4f}")
        logger.info(f"  Hausdorff Distance (95%): {val_metrics['hausdorff_95']:.2f} pixels")
        logger.info(f"{'='*80}\n")
        return
    
    # 開始訓練
    trainer.fit(
        train_loader,
        val_loader,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 保存訓練配置
    config_dict = vars(args)
    config_dict['best_val_dice'] = trainer.best_val_dice
    config_path = Path(args.output_dir) / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    logger.info(f"✅ 訓練配置已保存: {config_path}")


if __name__ == "__main__":
    main()
