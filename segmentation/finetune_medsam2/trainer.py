#!/usr/bin/env python3
"""
訓練器模組
提供 MedSAM2 模型的訓練與評估功能

重構說明：
- checkpoint_manager.py: Checkpoint 保存/載入
- llm_data_generator.py: LLM 訓練資料生成
- patient_analyzer.py: 患者分析與低分識別
- feature_saver.py: 特徵保存
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from tqdm import tqdm
import matplotlib.pyplot as plt

from .losses import (
    CombinedLoss, 
    EnhancedCombinedLoss, 
    TverskyLoss,
    FocalLoss,
    MedSAM2NativeLoss, 
    NATIVE_LOSS_AVAILABLE,
    PrecisionFocusedLoss
)
from .utils import compute_all_metrics, compute_lightweight_metrics, EarlyStopping, PatientMetricsTracker, convert_to_serializable
from .visualizer import SegmentationVisualizer
from .feature_extractor import LesionFeatureExtractor

# 重構後的模組
from .checkpoint_manager import CheckpointManager
from .patient_analyzer import PatientAnalyzer
from .feature_saver import FeatureSaver
from .llm_data_generator import LLMDataGenerator


class MedSAM2Trainer:
    """
    MedSAM2 訓練器
    
    負責模型載入、訓練、驗證、評估和模型保存
    
    Args:
        model_config: MedSAM2 配置檔案名稱
        checkpoint_path: 預訓練模型路徑
        device: 計算設備 ('cuda' 或 'cpu')
        output_dir: 輸出目錄
        loss_type: 損失函數類型 ('combined', 'enhanced', 'tversky', 'focal')
    """
    
    def __init__(
        self,
        model_config: str = "sam2.1_hiera_t512.yaml",
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        output_dir: str = "finetune_output",
        loss_type: str = "combined",
        use_amp: bool = True  # ✅ 新增 AMP 支援
    ):
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.loss_type = loss_type
        
        # ✅ AMP 混合精度設定
        self.use_amp = use_amp and device == "cuda"
        if self.use_amp:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        self.logger = logging.getLogger(__name__)
        
        # 載入模型
        self.logger.info(f"🔧 載入 MedSAM2 模型: {model_config}")
        self._load_model(model_config, checkpoint_path)
        
        # 損失函數（根據 loss_type 選擇）
        self.criterion = self._create_criterion(loss_type)
        
        # 訓練歷史
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'val_dice': [],
            'val_iou': [],
            'val_precision': [],
            'val_recall': [],
            'val_specificity': [],
            'val_accuracy': [],
            'val_hausdorff_95': [],
            'val_lesion_dice': [],  # ✅ 新增：只有病灶切片的 Dice
            'val_lesion_iou': [],   # ✅ 新增：只有病灶切片的 IoU
            'learning_rate': [],
            'epoch_time': [],
            'inference_time_per_sample': []
        }
        
        self.best_val_dice = 0.0
        self.best_val_metrics = {}  # 記錄所有 best validation 指標
        self.best_epoch = 0
        self.current_epoch = 0
        
        # ✅ 優化：用於緩存 image embeddings（每個 batch 清空避免 OOM）
        self._current_batch_cache = {}
        
        # ✅ 重構：初始化輔助模組
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.output_dir / "checkpoints",
            model_name="medsam2_finetuned",
            logger=self.logger
        )
        self.patient_analyzer = PatientAnalyzer(logger=self.logger)
        self.feature_saver = FeatureSaver(logger=self.logger)
        self.llm_generator = LLMDataGenerator()
        
        # 記錄 AMP 狀態
        if self.use_amp:
            self.logger.info("⚡ AMP 混合精度訓練已啟用")

    
    def _create_criterion(self, loss_type: str):
        """
        根據 loss_type 創建損失函數
        
        ✅ 重構：新增 'native' 選項使用 MedSAM2 原生損失
        
        Args:
            loss_type: 損失函數類型
                - 'combined': Dice + BCE (預設)
                - 'enhanced': Dice + Focal + Tversky + Boundary (推薦用於高 DSC)
                - 'native': MedSAM2 原生損失 (Dice + Focal，與 MedSAM2 訓練一致)
                - 'tversky': Tversky Loss (減少漏檢)
                - 'focal': Focal Loss (處理類別不平衡)
                - 'precision': Precision Focused Loss (減少過度分割)
        
        Returns:
            損失函數實例
        """
        if loss_type == 'precision':
            self.logger.info("📊 使用 Precision 聚焦損失: Tversky Loss (alpha=0.2, beta=0.8)")
            return PrecisionFocusedLoss(alpha=0.2, beta=0.8)
            
        if loss_type == 'native':
            if NATIVE_LOSS_AVAILABLE:
                self.logger.info("📊 使用 MedSAM2 原生損失函數: dice_loss + sigmoid_focal_loss")
                return MedSAM2NativeLoss(
                    dice_weight=1.0,
                    focal_weight=0.5,
                    focal_alpha=0.25,
                    focal_gamma=2.0
                )
            else:
                self.logger.warning("⚠️ MedSAM2 原生損失函數不可用，改用 enhanced")
                loss_type = 'enhanced'
        
        if loss_type == 'enhanced':
            self.logger.info("📊 使用增強損失函數: Dice + Focal + Tversky + Boundary (使用 MedSAM2 原生 Dice/Focal)")
            return EnhancedCombinedLoss(
                dice_weight=0.5,
                focal_weight=0.2,
                tversky_weight=0.2,
                boundary_weight=0.1
            )
        elif loss_type == 'tversky':
            self.logger.info("📊 使用 Tversky Loss (alpha=0.7, beta=0.3)")
            return TverskyLoss(alpha=0.7, beta=0.3)
        elif loss_type == 'focal':
            self.logger.info("📊 使用 Focal Loss (alpha=0.25, gamma=2.0，使用 MedSAM2 原生)")
            return FocalLoss(alpha=0.25, gamma=2.0, use_native=True)
        else:  # 'combined' 或其他
            self.logger.info("📊 使用組合損失函數: Dice + BCE (使用 MedSAM2 原生 Dice)")
            return CombinedLoss(dice_weight=0.8, bce_weight=0.2)
    
    def _load_model(self, config: str, checkpoint: Optional[str]):
        """
        載入 MedSAM2 模型
        
        ✅ 修正：移除重複的 Hydra 初始化
        """
        import sys
        from pathlib import Path
        
        # 添加 MedSAM2 路徑
        medsam2_path = Path(__file__).parent.parent / "MedSAM2"
        if str(medsam2_path) not in sys.path:
            sys.path.insert(0, str(medsam2_path))
        
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        # ✅ 修正：Hydra 初始化應該在外部完成，這裡只載入模型
        # 不再重複呼叫 initialize_config_dir
        
        config_name = config.replace('.yaml', '')
        
        # 建立模型
        if checkpoint and Path(checkpoint).exists():
            self.logger.info(f"📥 從 checkpoint 載入: {checkpoint}")
            self.model = build_sam2(config_name, checkpoint, device=self.device)
        else:
            self.logger.info(f"🆕 建立新模型（使用預設權重）")
            self.model = build_sam2(config_name, device=self.device)
        
        self.predictor = SAM2ImagePredictor(self.model)
        
        # 只訓練 mask decoder 和 prompt encoder
        for name, param in self.model.named_parameters():
            if "image_encoder" in name:
                param.requires_grad = False  # 凍結 image encoder
            else:
                param.requires_grad = True
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"✅ 可訓練參數: {trainable_params:,}")
    
    def _prepare_image_features(
        self, 
        image: torch.Tensor,
        use_cache: bool = False  # ✅ 預設禁用 cache，因為 cache key 計算成本高
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        計算 MedSAM2 image embeddings
        
        ✅ 優化：支援緩存避免重複計算（僅限當前 batch 內）
        ⚠️ 注意：預設禁用 cache，因為 cache key 計算成本可能超過 embedding 計算
        
        Args:
            image: 輸入影像 [3, H, W]，float tensor，範圍 0-255
            use_cache: 是否使用緩存（建議在同一張圖計算多個 bbox 時啟用）
            
        Returns:
            (image_embedding, high_res_feats)
        """
        # ✅ 使用簡化的 cache key（基於 tensor 的 data_ptr，避免 CPU 轉換）
        if use_cache:
            # 使用 data pointer + shape 作為 cache key（更快）
            cache_key = (image.data_ptr(), image.shape)
            if cache_key in self._current_batch_cache:
                return self._current_batch_cache[cache_key]
        
        with torch.no_grad():
            # 將 [3, H, W] 轉換為 [H, W, 3]
            np_image = image.permute(1, 2, 0).cpu().numpy()
            
            # ✅ 確保影像是 uint8 格式 (SAM2 predictor 期望 0-255 uint8)
            if np_image.dtype != np.uint8:
                # 如果是 float，確保範圍是 0-255 然後轉換
                if np_image.max() <= 1.0:
                    np_image = (np_image * 255).astype(np.uint8)
                else:
                    np_image = np.clip(np_image, 0, 255).astype(np.uint8)
            
            self.predictor.set_image(np_image)
        
        features = self.predictor._features or {}
        image_embedding = features.get("image_embed")
        if image_embedding is None:
            raise RuntimeError("Predictor features unavailable after set_image call")
        
        high_res_feats = features.get("high_res_feats")
        
        # ✅ 使用優化的 cache key
        if use_cache:
            cache_key = (image.data_ptr(), image.shape)
            self._current_batch_cache[cache_key] = (image_embedding, high_res_feats)
        
        return image_embedding, high_res_feats
    
    def train_epoch(
        self, 
        train_loader: DataLoader, 
        optimizer, 
        scheduler,
        accumulation_steps: int = 1
    ) -> Tuple[float, float]:
        """
        訓練一個 epoch
        
        ✅ 優化：支援梯度累積
        
        Args:
            train_loader: 訓練資料載入器
            optimizer: 優化器
            scheduler: 學習率調度器
            accumulation_steps: 梯度累積步數
            
        Returns:
            (平均訓練損失, 訓練耗時秒數)
        """
        start_time = time.time()
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # ✅ FIX: 清空 batch 緩存（每個 epoch 開始時）
        self._current_batch_cache.clear()
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # ✅ FIX: 每個 batch 開始時清空緩存，避免跨 batch 累積記憶體
            self._current_batch_cache.clear()
            
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            bboxes = batch['bboxes']
            
            batch_loss = 0.0
            batch_samples = 0
            
            # 處理 batch 中的每個樣本
            for i in range(len(images)):
                image = images[i]  # [3, H, W]
                gt_mask = masks[i]  # [1, H, W]
                bbox_tensor = bboxes[i]
                
                if len(bbox_tensor) == 0:
                    continue
                
                # ✅ 優化：使用 AMP autocast 包裝前向傳播
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    # ✅ 優化：同一張影像只計算一次 embedding
                    image_embedding, high_res_feats = self._prepare_image_features(image)
                    bbox_tensor = bbox_tensor.to(self.device)
                    
                    # ✅ 修正：收集所有 bbox 的預測，合併後再計算 loss
                    all_pred_logits = []  # 收集所有預測的 logits
                    
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
                        
                        all_pred_logits.append(pred_mask.squeeze())
                    
                    # ✅ 合併所有預測（取 max，因為是 logits）
                    if len(all_pred_logits) > 0:
                        if len(all_pred_logits) == 1:
                            combined_pred = all_pred_logits[0]
                        else:
                            # 多個 bbox：合併預測（取 logits 的 max）
                            stacked = torch.stack(all_pred_logits, dim=0)  # [N, H, W]
                            combined_pred = torch.max(stacked, dim=0)[0]   # [H, W]
                        
                        gt_mask_squeezed = gt_mask.squeeze()
                        
                        # ✅ 只計算一次 loss（使用合併後的預測）
                        sample_loss = self.criterion(combined_pred, gt_mask_squeezed)
                        batch_loss += sample_loss
                        batch_samples += 1
            
            # 反向傳播（支援梯度累積 + AMP）
            if batch_samples > 0:
                batch_loss = batch_loss / batch_samples
                
                # 梯度累積：除以累積步數
                loss_scaled = batch_loss / accumulation_steps
                
                # ✅ AMP: 使用 scaler 進行反向傳播
                if self.use_amp and self.scaler is not None:
                    self.scaler.scale(loss_scaled).backward()
                else:
                    loss_scaled.backward()
                
                # 每 accumulation_steps 步更新一次
                if (batch_idx + 1) % accumulation_steps == 0:
                    # ✅ 梯度裁剪防止梯度爆炸
                    if self.use_amp and self.scaler is not None:
                        self.scaler.unscale_(optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, self.model.parameters()), 
                        max_norm=1.0
                    )
                    
                    # ✅ AMP: 使用 scaler 更新參數
                    if self.use_amp and self.scaler is not None:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                
                total_loss += batch_loss.item()
                num_batches += 1
                
                # ✅ 修改：顯示累積平均 Loss，而非當前 Batch Loss
                current_avg_loss = total_loss / num_batches
                pbar.set_postfix({'loss': f'{current_avg_loss:.4f}'})
        
        # 清理最後可能剩餘的梯度
        if num_batches % accumulation_steps != 0:
            if self.use_amp and self.scaler is not None:
                self.scaler.unscale_(optimizer)
            
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()), 
                max_norm=1.0
            )
            
            if self.use_amp and self.scaler is not None:
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        scheduler.step()
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        epoch_time = time.time() - start_time
        return avg_loss, epoch_time
    
    def _save_first_epoch_samples(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        保存第一個 epoch 的訓練和驗證樣本以供檢視（只保存有病灶的切片）
        
        輸出到 output_dir/first_epoch_samples/
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        sample_dir = self.output_dir / "first_epoch_samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"📸 保存訓練樣本到: {sample_dir}（只保存有病灶的切片）")
        
        # 統計有病灶的切片數
        train_positive_count = 0
        val_positive_count = 0
        train_total_count = 0
        val_total_count = 0
        
        # 保存訓練樣本（只保存有病灶的）
        try:
            train_positive_samples = []
            for batch in train_loader:
                masks = batch['mask']  # [B, 1, H, W]
                for i in range(len(masks)):
                    train_total_count += 1
                    mask = masks[i].cpu().numpy()
                    if mask.sum() > 0:  # 有病灶
                        train_positive_count += 1
                        if len(train_positive_samples) < 8:  # 最多保存 8 個樣本
                            train_positive_samples.append({
                                'image': batch['image'][i],
                                'mask': batch['mask'][i],
                                'patient_id': batch['patient_id'][i],
                                'bboxes': batch['bboxes'][i] if 'bboxes' in batch else None
                            })
            
            # 保存訓練樣本
            if train_positive_samples:
                self._save_positive_samples(train_positive_samples, sample_dir, prefix="train")
            
            train_positive_ratio = (train_positive_count / train_total_count * 100) if train_total_count else 0.0
            self.logger.info(f"📊 訓練集統計: {train_positive_count}/{train_total_count} 個有病灶的切片 "
                           f"({train_positive_ratio:.1f}%)")
        except Exception as e:
            self.logger.warning(f"無法保存訓練樣本: {e}")
        
        # 保存驗證樣本（只保存有病灶的）
        try:
            if len(val_loader) > 0:
                val_positive_samples = []
                for batch in val_loader:
                    masks = batch['mask']  # [B, 1, H, W]
                    for i in range(len(masks)):
                        val_total_count += 1
                        mask = masks[i].cpu().numpy()
                        if mask.sum() > 0:  # 有病灶
                            val_positive_count += 1
                            if len(val_positive_samples) < 8:  # 最多保存 8 個樣本
                                val_positive_samples.append({
                                    'image': batch['image'][i],
                                    'mask': batch['mask'][i],
                                    'patient_id': batch['patient_id'][i],
                                    'bboxes': batch['bboxes'][i] if 'bboxes' in batch else None
                                })
                
                # 保存驗證樣本
                if val_positive_samples:
                    self._save_positive_samples(val_positive_samples, sample_dir, prefix="val")
                
                val_positive_ratio = (val_positive_count / val_total_count * 100) if val_total_count else 0.0
                self.logger.info(f"📊 驗證集統計: {val_positive_count}/{val_total_count} 個有病灶的切片 "
                               f"({val_positive_ratio:.1f}%)")
        except Exception as e:
            self.logger.warning(f"無法保存驗證樣本: {e}")
        
        # 保存統計資訊到文件
        stats_file = sample_dir / "lesion_statistics.txt"
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("有病灶切片統計\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"訓練集:\n")
            f.write(f"  - 總切片數: {train_total_count}\n")
            f.write(f"  - 有病灶切片數: {train_positive_count}\n")
            train_positive_ratio = (train_positive_count / train_total_count * 100) if train_total_count else 0.0
            val_positive_ratio = (val_positive_count / val_total_count * 100) if val_total_count else 0.0
            total_count = train_total_count + val_total_count
            total_positive_count = train_positive_count + val_positive_count
            total_positive_ratio = (total_positive_count / total_count * 100) if total_count else 0.0

            f.write(f"  - 比例: {train_positive_ratio:.2f}%\n\n")
            f.write(f"驗證集:\n")
            f.write(f"  - 總切片數: {val_total_count}\n")
            f.write(f"  - 有病灶切片數: {val_positive_count}\n")
            f.write(f"  - 比例: {val_positive_ratio:.2f}%\n\n")
            f.write(f"總計:\n")
            f.write(f"  - 總切片數: {total_count}\n")
            f.write(f"  - 有病灶切片數: {total_positive_count}\n")
            f.write(f"  - 比例: {total_positive_ratio:.2f}%\n")
        
        self.logger.info(f"✅ 第一 epoch 樣本已保存（只保存有病灶的切片）")
        self.logger.info(f"📄 統計資訊已保存到: {stats_file}")
    
    def _save_batch_samples(self, batch: dict, save_dir: Path, prefix: str, max_samples: int = 4):
        """
        保存 batch 中的樣本
        
        顯示：image, mask, bbox 位置
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        images = batch['image']  # [B, 3, H, W]
        masks = batch['mask']    # [B, 1, H, W]
        patient_ids = batch['patient_id']
        bboxes = batch.get('bboxes', None)
        
        num_samples = min(max_samples, len(images))
        
        for i in range(num_samples):
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            # 取得資料
            img = images[i].cpu().numpy()  # [3, H, W]
            mask = masks[i].cpu().numpy()  # [1, H, W]
            patient_id = str(patient_ids[i])
            
            # 中間通道 (2.5D 的 z 切片)
            img_center = img[1] if img.shape[0] == 3 else img[0]
            mask_2d = mask[0] if mask.ndim == 3 else mask
            
            # 1. 原始影像
            axes[0].imshow(img_center, cmap='gray')
            axes[0].set_title(f'{prefix.upper()} #{i}\n{patient_id}\nShape: {img.shape}')
            axes[0].axis('off')
            
            # 2. Mask (GT)
            axes[1].imshow(mask_2d, cmap='Reds', vmin=0, vmax=1)
            mask_area = (mask_2d > 0.5).sum()
            axes[1].set_title(f'Mask (GT)\nArea: {mask_area} px')
            axes[1].axis('off')
            
            # 3. 3-channel 視覺化 (RGB)
            if img.shape[0] == 3:
                # 歸一化每個通道到 0-1
                img_rgb = np.stack([
                    (img[0] - img[0].min()) / (img[0].max() - img[0].min() + 1e-6),
                    (img[1] - img[1].min()) / (img[1].max() - img[1].min() + 1e-6),
                    (img[2] - img[2].min()) / (img[2].max() - img[2].min() + 1e-6),
                ], axis=-1)
                axes[2].imshow(img_rgb)
                axes[2].set_title('3-Channel (Z-1, Z, Z+1)')
            else:
                axes[2].imshow(img_center, cmap='gray')
                axes[2].set_title('Single Channel')
            axes[2].axis('off')
            
            # 4. Overlay (影像 + Mask + BBox)
            overlay = np.stack([img_center, img_center, img_center], axis=-1)
            overlay = (overlay - overlay.min()) / (overlay.max() - overlay.min() + 1e-6)
            
            # 疊加 mask (紅色)
            mask_overlay = np.zeros_like(overlay)
            mask_overlay[:, :, 0] = (mask_2d > 0.5).astype(float) * 0.5
            overlay = np.clip(overlay + mask_overlay, 0, 1)
            
            axes[3].imshow(overlay)
            
            # 畫 BBox with size annotation
            if bboxes is not None and len(bboxes[i]) > 0:
                for bbox in bboxes[i]:
                    if bbox.sum() > 0:
                        x1, y1, x2, y2 = bbox.cpu().numpy()
                        width, height = x2 - x1, y2 - y1
                        rect = plt.Rectangle((x1, y1), width, height, 
                                             linewidth=2, edgecolor='lime', facecolor='none')
                        axes[3].add_patch(rect)
                        # Add size label
                        size_text = f'{int(width)}x{int(height)}px'
                        axes[3].text(x1, y1 - 3, size_text, fontsize=8, color='lime', 
                                    fontweight='bold', ha='left', va='bottom',
                                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7, edgecolor='none'))
            
            axes[3].set_title('Overlay + BBox')
            axes[3].axis('off')
            
            plt.tight_layout()
            save_path = save_dir / f"{prefix}_sample_{i}.png"
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
    
    def _save_positive_samples(self, samples: list, save_dir: Path, prefix: str):
        """
        保存有病灶的樣本
        
        Args:
            samples: 包含 image, mask, patient_id, bboxes 的樣本列表
            save_dir: 保存目錄
            prefix: 檔案名前綴 (train/val)
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        for i, sample in enumerate(samples):
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            # 取得資料
            img = sample['image'].cpu().numpy()  # [3, H, W]
            mask = sample['mask'].cpu().numpy()  # [1, H, W]
            patient_id = str(sample['patient_id'])
            bboxes = sample['bboxes']
            
            # 中間通道 (2.5D 的 z 切片)
            img_center = img[1] if img.shape[0] == 3 else img[0]
            mask_2d = mask[0] if mask.ndim == 3 else mask
            
            # 1. 原始影像
            axes[0].imshow(img_center, cmap='gray')
            axes[0].set_title(f'{prefix.upper()} #{i}\n{patient_id}\nShape: {img.shape}')
            axes[0].axis('off')
            
            # 2. Mask (GT)
            axes[1].imshow(mask_2d, cmap='Reds', vmin=0, vmax=1)
            mask_area = (mask_2d > 0.5).sum()
            axes[1].set_title(f'Mask (GT)\nArea: {mask_area} px')
            axes[1].axis('off')
            
            # 3. 3-channel 視覺化 (RGB)
            if img.shape[0] == 3:
                # 歸一化每個通道到 0-1
                img_rgb = np.stack([
                    (img[0] - img[0].min()) / (img[0].max() - img[0].min() + 1e-6),
                    (img[1] - img[1].min()) / (img[1].max() - img[1].min() + 1e-6),
                    (img[2] - img[2].min()) / (img[2].max() - img[2].min() + 1e-6),
                ], axis=-1)
                axes[2].imshow(img_rgb)
                axes[2].set_title('3-Channel (Z-1, Z, Z+1)')
            else:
                axes[2].imshow(img_center, cmap='gray')
                axes[2].set_title('Single Channel')
            axes[2].axis('off')
            
            # 4. Overlay (影像 + Mask + BBox)
            overlay = np.stack([img_center, img_center, img_center], axis=-1)
            overlay = (overlay - overlay.min()) / (overlay.max() - overlay.min() + 1e-6)
            
            # 疊加 mask (紅色)
            mask_overlay = np.zeros_like(overlay)
            mask_overlay[:, :, 0] = (mask_2d > 0.5).astype(float) * 0.5
            overlay = np.clip(overlay + mask_overlay, 0, 1)
            
            axes[3].imshow(overlay)
            
            # 畫 BBox with size annotation
            if bboxes is not None and len(bboxes) > 0:
                for bbox in bboxes:
                    if bbox.sum() > 0:
                        x1, y1, x2, y2 = bbox.cpu().numpy()
                        width, height = x2 - x1, y2 - y1
                        rect = plt.Rectangle((x1, y1), width, height, 
                                             linewidth=2, edgecolor='lime', facecolor='none')
                        axes[3].add_patch(rect)
                        # Add size label
                        size_text = f'{int(width)}x{int(height)}px'
                        axes[3].text(x1, y1 - 3, size_text, fontsize=8, color='lime', 
                                    fontweight='bold', ha='left', va='bottom',
                                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7, edgecolor='none'))
            
            axes[3].set_title('Overlay + BBox')
            axes[3].axis('off')
            
            plt.tight_layout()
            save_path = save_dir / f"{prefix}_positive_sample_{i}.png"
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
    
    @torch.no_grad()
    def validate(
        self, 
        val_loader: DataLoader, 
        metrics_tracker: Optional[PatientMetricsTracker] = None
    ) -> Tuple[float, Dict[str, float], float]:
        """
        驗證模型
        
        ✅ 修正：加入異常處理避免 tqdm 卡死
        ✅ 新增：支援患者指標追蹤
        
        Args:
            val_loader: 驗證資料載入器
            metrics_tracker: 患者指標追蹤器（可選）
        
        Returns:
            (平均損失, 評估指標字典, 驗證耗時秒數)
        """
        start_time = time.time()
        self.model.eval()
        total_loss = 0.0
        
        # 初始化指標累加器
        metrics_sum = {
            'dice': 0.0,
            'iou': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'specificity': 0.0,
            'accuracy': 0.0,
            'hausdorff_95': 0.0,
        }
        num_samples = 0
        
        # ✅ 新增：只統計有病灶切片的指標
        lesion_metrics_sum = {
            'dice': 0.0,
            'iou': 0.0,
            'precision': 0.0,
            'recall': 0.0,
        }
        num_lesion_samples = 0
        
        # ✅ FIX: 清空 batch 緩存
        self._current_batch_cache.clear()
        
        pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch+1} [Val]")
        
        try:
            for batch in pbar:
                # ✅ FIX: 每個 batch 開始時清空緩存
                self._current_batch_cache.clear()
                
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                patient_ids = batch['patient_id']
                slice_indices = batch['slice_index']
                bboxes = batch['bboxes']
                
                for i in range(len(images)):
                    image = images[i]
                    gt_mask = masks[i]
                    bbox_tensor = bboxes[i]
                    
                    if len(bbox_tensor) == 0:
                        continue
                    
                    # ✅ 優化：使用相同的 embedding 計算邏輯
                    image_embedding, high_res_feats = self._prepare_image_features(image)
                    bbox_tensor = bbox_tensor.to(self.device)
                    
                    # ✅ 修正：收集所有 bbox 的預測，合併後再計算指標
                    all_pred_logits = []
                    
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
                        
                        all_pred_logits.append(pred_mask.squeeze())
                    
                    # ✅ 合併所有預測（取 max）
                    if len(all_pred_logits) > 0:
                        if len(all_pred_logits) == 1:
                            combined_pred = all_pred_logits[0]
                        else:
                            stacked = torch.stack(all_pred_logits, dim=0)
                            combined_pred = torch.max(stacked, dim=0)[0]
                        
                        gt_mask_squeezed = gt_mask.squeeze()
                        
                        # ✅ 只計算一次 loss 和指標（使用合併後的預測）
                        loss = self.criterion(combined_pred, gt_mask_squeezed)
                        total_loss += loss.item()
                        
                        # ✅ 指標計算：使用合併後的預測
                        sample_metrics = compute_lightweight_metrics(combined_pred, gt_mask_squeezed)
                        
                        for key in metrics_sum.keys():
                            if key in sample_metrics:
                                metrics_sum[key] += sample_metrics[key]
                        
                        num_samples += 1
                        
                        # ✅ 新增：只統計有病灶切片的指標
                        # 檢查 gt_mask 是否有病灶
                        if gt_mask.sum().item() > 0:
                            for key in lesion_metrics_sum.keys():
                                if key in sample_metrics:
                                    lesion_metrics_sum[key] += sample_metrics[key]
                            num_lesion_samples += 1
                        
                        # ✅ 新增：記錄患者級別指標
                        if metrics_tracker is not None:
                            patient_id = patient_ids[i]
                            slice_idx = slice_indices[i]
                            
                            # 計算體積統計 (用於 Volume Dice)
                            pred_binary = (torch.sigmoid(combined_pred) > 0.5).float()
                            gt_binary = (gt_mask_squeezed > 0.5).float()
                            
                            intersection = (pred_binary * gt_binary).sum().item()
                            pred_area = pred_binary.sum().item()
                            gt_area = gt_binary.sum().item()
                            union = pred_area + gt_area - intersection
                            
                            vol_stats = {
                                'intersection': intersection,
                                'pred_area': pred_area,
                                'gt_area': gt_area,
                                'union': union
                            }
                            
                            metrics_tracker.add_slice_metrics(
                                patient_id=patient_id,
                                slice_idx=slice_idx,
                                metrics=sample_metrics,
                                vol_stats=vol_stats
                            )
                        
                        # ✅ 修改：顯示累積平均指標，而非當前 Batch 指標
                        current_avg_loss = total_loss / num_samples
                        current_avg_metrics = {k: v / num_samples for k, v in metrics_sum.items()}
                        
                        pbar.set_postfix({
                            'loss': f'{current_avg_loss:.4f}',
                            'dice': f'{current_avg_metrics["dice"]:.4f}',
                            'iou': f'{current_avg_metrics["iou"]:.4f}'
                        })
        
        except Exception as e:
            self.logger.error(f"❌ 驗證過程發生錯誤: {e}")
            raise
        
        # 計算平均值
        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        avg_metrics = {k: v / num_samples if num_samples > 0 else 0.0 
                      for k, v in metrics_sum.items()}
        
        # ✅ 新增：計算只有病灶切片的平均指標
        avg_lesion_metrics = {k: v / num_lesion_samples if num_lesion_samples > 0 else 0.0 
                              for k, v in lesion_metrics_sum.items()}
        
        # ✅ 計算 Global Volume Dice (全資料集)
        # 需在 loop 中累積，但為了避免修改 loop 太多，這裡暫時使用 slice average 的 dice 作為默認
        # 若要精確計算 Global Volume Dice，需要在 loop 中累積 global_vol_stats
        # 鑑於 metrics_tracker 已經有完整記錄，我們可以從 metrics_tracker 聚合（如果有傳入）
        
        global_vol_dice = avg_metrics['dice'] # fallback
        
        if metrics_tracker is not None:
             # 從 tracker 聚合所有患者的 Volume Stats
            t_inter = 0.0
            t_pred = 0.0
            t_gt = 0.0
            
            # 確保計算過 averages (雖然 add_slice 沒觸發，但可以直接存取 vol_stats)
            for pid, data in metrics_tracker.patient_metrics.items():
                v = data.get('vol_stats', {})
                t_inter += v.get('intersection', 0.0)
                t_pred += v.get('pred_area', 0.0)
                t_gt += v.get('gt_area', 0.0)
            
            if t_pred + t_gt > 0:
                global_vol_dice = (2.0 * t_inter + 1e-6) / (t_pred + t_gt + 1e-6)
                
                # ✅ 覆寫主要的 Dice 為 Global Volume Dice
                avg_metrics['slice_avg_dice'] = avg_metrics['dice']
                avg_metrics['dice'] = global_vol_dice
                
                # 更新 lesion_dice 也使用 volume dice (如果有辦法區分的話... 這裡簡單假設 lesion_dice 也可以用 global volume 近似，或者保留 slice avg)
                # 這裡保留 avg_lesion_metrics 為 slice avg 以呈現差異
        
        # ✅ 將有病灶切片指標加入結果（使用 lesion_ 前綴）
        avg_metrics['lesion_dice'] = avg_lesion_metrics['dice']
        avg_metrics['lesion_iou'] = avg_lesion_metrics['iou']
        avg_metrics['lesion_precision'] = avg_lesion_metrics['precision']
        avg_metrics['lesion_recall'] = avg_lesion_metrics['recall']
        avg_metrics['num_lesion_samples'] = num_lesion_samples
        avg_metrics['lesion_ratio'] = num_lesion_samples / num_samples if num_samples > 0 else 0.0
        
        val_time = time.time() - start_time
        return avg_loss, avg_metrics, val_time
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        learning_rate: float = 1e-5,
        weight_decay: float = 1e-4,
        early_stopping_patience: int = 7,
        accumulation_steps: int = 1,
        warmup_epochs: int = 5
    ):
        """
        訓練模型
        
        Args:
            train_loader: 訓練資料載入器
            val_loader: 驗證資料載入器
            epochs: 訓練輪數
            learning_rate: 學習率
            weight_decay: 權重衰減
            early_stopping_patience: 早停容忍 epoch 數
            accumulation_steps: 梯度累積步數
            warmup_epochs: Warmup epoch 數 (預設 5)
        """
        # 優化器
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 學習率調度器：Warmup + Cosine Annealing
        if warmup_epochs > 0 and epochs > warmup_epochs:
            warmup_scheduler = LinearLR(
                optimizer, 
                start_factor=0.1,  # 從 10% LR 開始
                end_factor=1.0, 
                total_iters=warmup_epochs
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer, 
                T_max=epochs - warmup_epochs,
                eta_min=learning_rate * 0.01  # 最小 LR 為初始的 1%
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )
            self.logger.info(f"📈 使用 Warmup ({warmup_epochs} epochs) + Cosine Annealing 調度器")
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
            self.logger.info(f"📈 使用 Cosine Annealing 調度器")
        
        # 早停機制
        early_stopping = EarlyStopping(
            patience=early_stopping_patience, 
            min_delta=0.001, 
            mode='max'
        )
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"🚀 開始訓練")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Epochs: {epochs}")
        self.logger.info(f"Warmup Epochs: {warmup_epochs}")
        self.logger.info(f"Learning Rate: {learning_rate}")
        self.logger.info(f"Gradient Accumulation Steps: {accumulation_steps}")
        self.logger.info(f"Early Stopping Patience: {early_stopping_patience}")
        self.logger.info(f"Loss Type: {self.loss_type}")
        self.logger.info(f"Train samples: {len(train_loader.dataset)}")
        self.logger.info(f"Val samples: {len(val_loader.dataset)}")
        self.logger.info(f"{'='*80}\n")
        
        # 保存第一個 epoch 的樣本以供檢視
        self._save_first_epoch_samples(train_loader, val_loader)
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # 訓練
            train_loss, epoch_time = self.train_epoch(
                train_loader, 
                optimizer, 
                scheduler,
                accumulation_steps
            )
            
            # 驗證
            val_loss, val_metrics, val_time = self.validate(val_loader)
            
            # 計算推理時間 (每樣本)
            inference_time_per_sample = (val_time / len(val_loader.dataset)) * 1000 if len(val_loader.dataset) > 0 else 0
            
            # 記錄歷史
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_dice'].append(val_metrics['dice'])
            self.train_history['val_iou'].append(val_metrics['iou'])
            self.train_history['val_precision'].append(val_metrics['precision'])
            self.train_history['val_recall'].append(val_metrics['recall'])
            self.train_history['val_specificity'].append(val_metrics['specificity'])
            self.train_history['val_accuracy'].append(val_metrics['accuracy'])
            self.train_history['val_hausdorff_95'].append(val_metrics['hausdorff_95'])
            self.train_history['val_lesion_dice'].append(val_metrics.get('lesion_dice', 0.0))  # ✅ 新增
            self.train_history['val_lesion_iou'].append(val_metrics.get('lesion_iou', 0.0))    # ✅ 新增
            self.train_history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            self.train_history['epoch_time'].append(epoch_time)
            self.train_history['inference_time_per_sample'].append(inference_time_per_sample)
            
            # 輸出結果
            lesion_ratio_pct = val_metrics.get('lesion_ratio', 0) * 100
            num_lesion = val_metrics.get('num_lesion_samples', 0)
            self.logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n"
                f"  Time: {epoch_time:.1f}s ({epoch_time/len(train_loader):.3f}s/batch), "
                f"Inference: {inference_time_per_sample:.1f}ms/sample\n"
                f"  📊 全切片 Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}, "
                f"Acc: {val_metrics['accuracy']:.4f}\n"
                f"  🎯 有病灶 Dice: {val_metrics.get('lesion_dice', 0):.4f}, "
                f"IoU: {val_metrics.get('lesion_iou', 0):.4f} "
                f"({num_lesion} 切片, {lesion_ratio_pct:.1f}%)\n"
                f"  Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, "
                f"Specificity: {val_metrics['specificity']:.4f}\n"
                f"  Hausdorff95: {val_metrics['hausdorff_95']:.2f}, "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )
            
            # 保存最佳模型（✅ 使用有病灶切片的 Dice 作為判斷標準）
            lesion_dice = val_metrics.get('lesion_dice', val_metrics['dice'])
            if lesion_dice > self.best_val_dice:
                self.best_val_dice = lesion_dice
                self.best_epoch = epoch + 1
                # 記錄所有 best validation 指標
                self.best_val_metrics = {
                    'epoch': epoch + 1,
                    'loss': val_loss,
                    'dice': val_metrics['dice'],
                    'lesion_dice': lesion_dice,  # ✅ 新增
                    'iou': val_metrics['iou'],
                    'lesion_iou': val_metrics.get('lesion_iou', 0),  # ✅ 新增
                    'recall': val_metrics['recall'],
                    'precision': val_metrics['precision'],
                    'specificity': val_metrics['specificity'],
                    'accuracy': val_metrics['accuracy'],
                    'hausdorff_95': val_metrics['hausdorff_95'],
                    'inference_time_ms': inference_time_per_sample
                }
                self.save_checkpoint('best_model.pth', is_best=True)
                self.logger.info(f"✅ 保存最佳模型 (有病灶 Dice: {lesion_dice:.4f})")
                # 保存最佳指標到 JSON
                self._save_best_metrics()
            
            # 早停檢查（✅ 使用有病灶切片的 Dice）
            if early_stopping(epoch, lesion_dice):
                self.logger.info(f"🛑 早停：訓練在 Epoch {epoch+1} 停止")
                break
            
            # 定期保存 checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
            
            # 每個 epoch 更新訓練曲線
            self.plot_training_curves()
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"✅ 訓練完成！")
        self.logger.info(f"{'='*80}")
        if self.best_val_metrics:
            self.logger.info(f"📊 Best Validation Metrics (Epoch {self.best_val_metrics['epoch']}):")
            self.logger.info(f"   Loss: {self.best_val_metrics['loss']:.4f}")
            self.logger.info(f"   Dice: {self.best_val_metrics['dice']:.4f}")
            self.logger.info(f"   IoU: {self.best_val_metrics['iou']:.4f}")
            self.logger.info(f"   Recall (Sensitivity): {self.best_val_metrics['recall']:.4f}")
            self.logger.info(f"   Precision (PPV): {self.best_val_metrics['precision']:.4f}")
            self.logger.info(f"   Specificity: {self.best_val_metrics['specificity']:.4f}")
            self.logger.info(f"   Accuracy: {self.best_val_metrics['accuracy']:.4f}")
            self.logger.info(f"   Hausdorff95: {self.best_val_metrics['hausdorff_95']:.2f} px")
            self.logger.info(f"   Inference Time: {self.best_val_metrics['inference_time_ms']:.1f} ms/sample")
        self.logger.info(f"{'='*80}\n")
        
        # 保存最終模型
        self.save_checkpoint('final_model.pth')
        self.logger.info(f"💾 最終模型已保存: {self.output_dir / 'final_model.pth'}")
        
        # 保存訓練歷史到 JSON
        self._save_training_history()
        
        # 繪製最終訓練曲線
        self.plot_training_curves()
    
    def _save_best_metrics(self):
        """保存最佳指標到 JSON 檔案"""
        if not self.best_val_metrics:
            return
        metrics_path = self.output_dir / 'best_metrics.json'
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(convert_to_serializable(self.best_val_metrics), f, indent=2, ensure_ascii=False)
    
    def _save_training_history(self):
        """保存訓練歷史到 JSON 檔案"""
        history_path = self.output_dir / 'history.json'
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(convert_to_serializable(self.train_history), f, indent=2, ensure_ascii=False)
        self.logger.info(f"💾 訓練歷史已保存: {history_path}")
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """保存 checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'best_val_dice': self.best_val_dice,
            'best_val_metrics': self.best_val_metrics,
            'best_epoch': self.best_epoch,
            'train_history': self.train_history
        }
        
        checkpoint_path = self.output_dir / filename
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            self.logger.info(f"💾 最佳模型已保存: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """載入 checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_dice = checkpoint['best_val_dice']
        self.best_val_metrics = checkpoint.get('best_val_metrics', {})
        self.best_epoch = checkpoint.get('best_epoch', 0)
        self.train_history = checkpoint['train_history']
        self.logger.info(f"✅ Checkpoint 已載入: {checkpoint_path}")
    
    def plot_training_curves(self):
        """繪製訓練曲線（包含所有評估指標）"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-display backend
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.warning("matplotlib not installed, cannot plot training curves")
            return
        
        fig, axes = plt.subplots(3, 4, figsize=(24, 15))  # 3x4 layout for more charts
        
        epochs = range(1, len(self.train_history['train_loss']) + 1)
        
        # 1. Loss Curves
        axes[0, 0].plot(epochs, self.train_history['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(epochs, self.train_history['val_loss'], label='Val Loss', color='red')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. All Slices Dice & IoU (including empty masks)
        axes[0, 1].plot(epochs, self.train_history['val_dice'], label='All Dice', color='green', alpha=0.5)
        axes[0, 1].plot(epochs, self.train_history['val_iou'], label='All IoU', color='orange', alpha=0.5)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('All Slices Dice & IoU (incl. empty)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. Lesion Slices Dice & IoU (key metrics)
        lesion_dice = self.train_history.get('val_lesion_dice', [0] * len(epochs))
        lesion_iou = self.train_history.get('val_lesion_iou', [0] * len(epochs))
        axes[0, 2].plot(epochs, lesion_dice, label='Lesion Dice', color='green', linewidth=2)
        axes[0, 2].plot(epochs, lesion_iou, label='Lesion IoU', color='orange', linewidth=2)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Score')
        axes[0, 2].set_title('Lesion Slices Dice & IoU (Key Metrics)')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # 4. Precision & Recall
        axes[0, 3].plot(epochs, self.train_history['val_precision'], label='Precision', color='purple')
        axes[0, 3].plot(epochs, self.train_history['val_recall'], label='Recall', color='brown')
        axes[0, 3].set_xlabel('Epoch')
        axes[0, 3].set_ylabel('Score')
        axes[0, 3].set_title('Precision & Recall')
        axes[0, 3].legend()
        axes[0, 3].grid(True)
        
        # 5. Specificity & Accuracy
        axes[1, 0].plot(epochs, self.train_history['val_specificity'], label='Specificity', color='cyan')
        axes[1, 0].plot(epochs, self.train_history['val_accuracy'], label='Accuracy', color='magenta')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Specificity & Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 6. Hausdorff Distance
        axes[1, 1].plot(epochs, self.train_history['val_hausdorff_95'], label='HD95', color='black')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Pixels')
        axes[1, 1].set_title('Hausdorff Distance (95%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # 7. Learning Rate
        axes[1, 2].plot(epochs, self.train_history['learning_rate'], label='LR', color='gray')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('LR')
        axes[1, 2].set_title('Learning Rate')
        axes[1, 2].set_yscale('log')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        # 8. All Slices vs Lesion Slices Dice comparison
        axes[1, 3].plot(epochs, self.train_history['val_dice'], label='All Dice', color='blue', linestyle='--', alpha=0.7)
        axes[1, 3].plot(epochs, lesion_dice, label='Lesion Dice', color='red', linewidth=2)
        axes[1, 3].set_xlabel('Epoch')
        axes[1, 3].set_ylabel('Dice Score')
        axes[1, 3].set_title('Dice: All vs Lesion Slices')
        axes[1, 3].legend()
        axes[1, 3].grid(True)
        
        # 9. Training Time
        axes[2, 0].plot(epochs, self.train_history['epoch_time'], label='Epoch Time (s)', color='blue')
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('Seconds')
        axes[2, 0].set_title('Training Efficiency')
        axes[2, 0].legend()
        axes[2, 0].grid(True)
        
        # 10. Inference Time
        axes[2, 1].plot(epochs, self.train_history['inference_time_per_sample'], label='Inference (ms)', color='red')
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('Milliseconds')
        axes[2, 1].set_title('Inference Speed')
        axes[2, 1].legend()
        axes[2, 1].grid(True)
        
        # 11-12. Summary Statistics (merged two cells)
        axes[2, 2].axis('off')
        axes[2, 3].axis('off')
        
        # Get lesion metrics
        final_lesion_dice = lesion_dice[-1] if lesion_dice else 0
        final_lesion_iou = lesion_iou[-1] if lesion_iou else 0
        
        summary_text = f"""
Training Summary
================

Best Lesion Dice: {self.best_val_dice:.4f}
   (for model selection & early stopping)

Final Metrics (Lesion Slices):
  Lesion Dice: {final_lesion_dice:.4f}
  Lesion IoU:  {final_lesion_iou:.4f}

Final Metrics (All Slices):
  All Dice:    {self.train_history['val_dice'][-1]:.4f}
  All IoU:     {self.train_history['val_iou'][-1]:.4f}
  Precision:   {self.train_history['val_precision'][-1]:.4f}
  Recall:      {self.train_history['val_recall'][-1]:.4f}
  Specificity: {self.train_history['val_specificity'][-1]:.4f}
  HD95:        {self.train_history['val_hausdorff_95'][-1]:.2f}

Efficiency:
  Avg Epoch Time: {np.mean(self.train_history['epoch_time']):.1f}s
  Avg Inference:  {np.mean(self.train_history['inference_time_per_sample']):.1f}ms

Total Epochs: {len(epochs)}
        """
        axes[2, 2].text(0.0, 0.5, summary_text, fontsize=11, family='monospace', verticalalignment='center')
        
        plt.tight_layout()
        
        plot_path = self.output_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        self.logger.info(f"Training curves saved: {plot_path}")
        plt.close()
    
    @torch.no_grad()
    def test_and_extract_features(
        self,
        test_loader: DataLoader,
        output_dir: Optional[str] = None,
        extract_deep_features: bool = True,
        save_predictions: bool = True,
        save_visualizations: bool = True,
        spacing: Tuple[float, float] = (1.0, 1.0),
        min_area: int = 0,
        min_confidence: float = 0.0,
        min_dice: float = 0.0,
        prompt_type: str = 'bbox'  # 'bbox' or 'point'
    ) -> Dict:
        """
        測試模型並提取病灶特徵用於 LLM Fine-Tuning
        
        Args:
            test_loader: 測試資料載入器
            output_dir: 特徵輸出目錄（預設為 self.output_dir/features）
            extract_deep_features: 是否提取深層特徵向量
            save_predictions: 是否保存預測遮罩
            save_visualizations: 是否保存可視化 PNG 圖片（GT mask、Pred mask、對比圖）
            spacing: 像素間距 (mm)
            min_area: 最小病灶面積閾值（像素），預設 0（不過濾）
            min_confidence: SAM2 IoU 預測的最小置信度閾值，預設 0（不過濾）
            min_dice: 預測與 GT 的最小 Dice 分數閾值，預設 0（不過濾）
            prompt_type: 提示類型 ('bbox' or 'point')，預設 'bbox'
        
        Returns:
            包含所有測試結果和特徵的字典
        """
        from datetime import datetime
        
        if output_dir is None:
            output_dir = self.output_dir / "features"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化特徵提取器
        feature_extractor = LesionFeatureExtractor(self.model, self.device)
        
        # 初始化可視化器
        visualizer = None
        if save_visualizations:
            vis_dir = output_dir / "visualizations"
            visualizer = SegmentationVisualizer(vis_dir)
            self.logger.info(f"📸 可視化輸出目錄: {vis_dir}")
        
        self.model.eval()
        self._current_batch_cache.clear()
        
        # 結果容器
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'best_val_dice': self.best_val_dice,
                'current_epoch': self.current_epoch,
                'prompt_type': prompt_type
            },
            'test_metrics': {
                'dice': [], 'iou': [], 'precision': [], 'recall': [],
                'specificity': [], 'accuracy': [], 'hausdorff_95': []
            },
            'patient_features': {},  # patient_id -> features
            'total_samples': 0,
            'total_lesions': 0,
            'filtering_stats': {
                'total_bboxes': 0,
                'filtered_by_area': 0,
                'filtered_by_confidence': 0,
                'filtered_by_dice': 0,
                'kept_lesions': 0
            }
        }
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"🔬 開始測試並提取特徵")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"測試樣本數: {len(test_loader.dataset)}")
        self.logger.info(f"輸出目錄: {output_dir}")
        self.logger.info(f"提取深層特徵: {extract_deep_features}")
        self.logger.info(f"提示類型 (Prompt Type): {prompt_type}")
        if prompt_type == 'point':
            self.logger.info(f"   - 模擬使用者點擊 (Click Simulation): 使用 BBox 中心點作為提示")
        self.logger.info(f"📊 過濾閾值:")
        self.logger.info(f"   - 最小面積: {min_area} px")
        self.logger.info(f"   - 最小置信度 (IoU prediction): {min_confidence:.2f}")
        self.logger.info(f"   - 最小 Dice 分數: {min_dice:.2f}")
        self.logger.info(f"{'='*80}\n")
        
        pbar = tqdm(test_loader, desc="Testing & Extracting Features")
        
        for batch in pbar:
            self._current_batch_cache.clear()
            
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            patient_ids = batch['patient_id']
            slice_indices = batch['slice_index']
            bboxes = batch['bboxes']
            
            for i in range(len(images)):
                image = images[i]  # [3, H, W]
                gt_mask = masks[i]  # [1, H, W]
                bbox_tensor = bboxes[i]
                patient_id = str(patient_ids[i])
                slice_idx = int(slice_indices[i])
                
                if len(bbox_tensor) == 0:
                    continue
                
                # 計算 image embedding
                image_embedding, high_res_feats = self._prepare_image_features(image)
                bbox_tensor = bbox_tensor.to(self.device)
                
                # 初始化患者特徵
                if patient_id not in all_results['patient_features']:
                    all_results['patient_features'][patient_id] = {
                        'patient_id': patient_id,
                        'slices': {},
                        'summary': {}
                    }
                
                slice_features = {
                    'slice_index': slice_idx,
                    'lesions': [],
                    'metrics': {}
                }
                
                # 處理每個 bbox（病灶）
                all_pred_masks = []
                # 收集可視化用的點
                vis_points = []
                vis_labels = []
                
                lesion_idx = 0
                
                for bbox in bbox_tensor:
                    all_results['filtering_stats']['total_bboxes'] += 1
                    
                    if bbox.sum() == 0:
                        continue
                    
                    # 準備 Prompt (支援 BBox 或 Point)
                    sparse_embeddings, dense_embeddings = None, None
                    
                    if prompt_type == 'point':
                        # 模擬使用者點擊：計算 BBox 中心點
                        x1, y1, x2, y2 = bbox
                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0
                        
                        # Point format: [B, N, 2] -> [1, 1, 2] (here B=1, N=1 point per lesion prompt)
                        # 注意：我們是逐個 lesion 處理，所以 batch size=1
                        point_coords = torch.tensor([[[cx, cy]]], device=self.device)
                        # Label format: [B, N] -> [1, 1] (1=positive)
                        point_labels = torch.tensor([[1]], device=self.device)
                        
                        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
                            points=(point_coords, point_labels),
                            boxes=None,
                            masks=None,
                        )
                        
                        # 記錄用於可視化
                        vis_points.append([cx.item(), cy.item()])
                        vis_labels.append(1)
                        
                    else: # 'bbox' (default)
                        box_torch = bbox.unsqueeze(0)
                        
                        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
                            points=None,
                            boxes=box_torch,
                            masks=None,
                        )
                    
                    # Mask Decoder
                    low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(
                        image_embeddings=image_embedding,
                        image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                        repeat_image=False,
                        high_res_features=high_res_feats,
                    )
                    
                    # 上採樣到原始大小
                    pred_mask = F.interpolate(
                        low_res_masks,
                        size=(gt_mask.shape[-2], gt_mask.shape[-1]),
                        mode='bilinear',
                        align_corners=False
                    )
                    
                    pred_mask_squeezed = pred_mask.squeeze()
                    gt_mask_squeezed = gt_mask.squeeze()
                    
                    # 計算評估指標
                    metrics = compute_all_metrics(pred_mask_squeezed, gt_mask_squeezed)
                    
                    # 取得二值化預測 mask
                    pred_binary = (torch.sigmoid(pred_mask_squeezed) > 0.5).cpu().numpy()
                    
                    # ✅ 過濾器 1: 面積過濾（噪音過濾）
                    min_pred_area_px = min_area  # 從參數讀取
                    pred_area = pred_binary.sum()
                    if pred_area < min_pred_area_px:
                        all_results['filtering_stats']['filtered_by_area'] += 1
                        continue  # 跳過太小的預測
                    
                    # ✅ 過濾器 2: SAM2 置信度過濾（IoU prediction）
                    confidence = float(iou_predictions.cpu().numpy().mean()) if iou_predictions is not None else 1.0
                    if confidence < min_confidence:
                        all_results['filtering_stats']['filtered_by_confidence'] += 1
                        continue  # 跳過低置信度預測
                    
                    # ✅ 過濾器 3: Dice 分數過濾（與 GT 的匹配度）
                    lesion_dice = metrics.get('dice', 0.0)
                    if lesion_dice < min_dice:
                        all_results['filtering_stats']['filtered_by_dice'] += 1
                        continue  # 跳過與 GT 不匹配的預測
                    
                    # ✅ 通過所有過濾器，保留此 lesion
                    all_results['filtering_stats']['kept_lesions'] += 1
                    all_pred_masks.append(pred_binary)

                    
                    # 原始影像（用於強度特徵計算）
                    original_image = image[0].cpu().numpy()  # 取第一個通道
                    
                    # 提取形態學特徵
                    morphological_features = feature_extractor.compute_morphological_features(
                        pred_binary, spacing
                    )
                    
                    # 提取強度特徵
                    intensity_features = feature_extractor.compute_intensity_features(
                        original_image, pred_binary
                    )
                    
                    # 結節類型分類（基於 HU 值分佈）
                    nodule_classification = feature_extractor.classify_nodule_type(
                        original_image, pred_binary
                    )
                    
                    # 提取深層特徵（可選）
                    deep_features = {}
                    if extract_deep_features:
                        deep_features = feature_extractor.extract_deep_features(
                            image_embedding,
                            sparse_embeddings,
                            dense_embeddings,
                            high_res_feats
                        )
                    
                    # 聚合病灶特徵（包含結節類型）
                    lesion_feature = feature_extractor.aggregate_lesion_features(
                        morphological_features,
                        intensity_features,
                        deep_features,
                        confidence,
                        nodule_classification
                    )
                    lesion_feature['lesion_id'] = lesion_idx
                    lesion_feature['bbox'] = bbox.cpu().numpy().tolist()
                    lesion_feature['metrics'] = metrics
                    # 如果是點擊模式，記錄點擊座標
                    lesion_feature['click_point'] = [cx.item(), cy.item()] if prompt_type == 'point' else None
                    
                    slice_features['lesions'].append(lesion_feature)
                    lesion_idx += 1
                    all_results['total_lesions'] += 1
                    
                    # 累積測試指標
                    for key in all_results['test_metrics'].keys():
                        if key in metrics:
                            all_results['test_metrics'][key].append(metrics[key])
                
                # 計算切片級別指標（平均）
                if slice_features['lesions']:
                    slice_metrics = {}
                    for key in ['dice', 'iou', 'precision', 'recall']:
                        values = [l['metrics'].get(key, 0) for l in slice_features['lesions']]
                        slice_metrics[key] = float(np.mean(values))
                    slice_features['metrics'] = slice_metrics
                
                # 保存切片特徵
                all_results['patient_features'][patient_id]['slices'][slice_idx] = slice_features
                all_results['total_samples'] += 1
                
                # 保存預測遮罩（可選）
                if save_predictions and all_pred_masks:
                    pred_save_dir = output_dir / "predictions" / patient_id
                    pred_save_dir.mkdir(parents=True, exist_ok=True)
                    
                    combined_mask = np.zeros_like(all_pred_masks[0], dtype=np.uint8)
                    for idx, pm in enumerate(all_pred_masks):
                        combined_mask = np.maximum(combined_mask, pm.astype(np.uint8) * (idx + 1))
                    
                    np.save(pred_save_dir / f"slice_{slice_idx:04d}_pred.npy", combined_mask)
                
                # 保存可視化圖片（可選 - 保存所有有 BBox 的切片，包括 False Negatives）
                # ✅ 優化：應使用者要求，保存所有有 predict 的結果 (即使是全黑的)
                if visualizer is not None:
                    # 準備原始影像（歸一化到 0-1）
                    original_image_np = image[0].cpu().numpy()  # 取第一個通道
                    if original_image_np.max() > 1.0:
                        original_image_np = (original_image_np - original_image_np.min()) / (original_image_np.max() - original_image_np.min() + 1e-8)
                    
                    # 合併所有預測遮罩
                    if all_pred_masks:
                        combined_pred = np.zeros_like(all_pred_masks[0], dtype=np.float32)
                        for pm in all_pred_masks:
                            combined_pred = np.maximum(combined_pred, pm.astype(np.float32))
                    else:
                        # 如果沒有預測結果（被過濾或無預測），則是全黑 mask
                        combined_pred = np.zeros((gt_mask.shape[-2], gt_mask.shape[-1]), dtype=np.float32)
                    
                    # GT mask
                    gt_mask_np = gt_mask.squeeze().cpu().numpy()
                    
                    # 計算切片級別的平均指標
                    slice_dice = slice_features['metrics'].get('dice', 0.0)
                    slice_iou = slice_features['metrics'].get('iou', 0.0)
                    
                    # 準備 bboxes 用於可視化
                    bboxes_np = bbox_tensor.cpu().numpy() if bbox_tensor is not None else None
                    
                    # 準備 points 用於可視化
                    points_np = np.array(vis_points) if vis_points else None
                    labels_np = np.array(vis_labels) if vis_labels else None
                    
                    visualizer.save_slice_comparison(
                        image=original_image_np,
                        gt_mask=gt_mask_np,
                        pred_mask=combined_pred,
                        patient_id=patient_id,
                        slice_idx=slice_idx,
                        dice_score=slice_dice,
                        iou_score=slice_iou,
                        bboxes=bboxes_np,
                        points=points_np,
                        point_labels=labels_np
                    )
                
                # 更新進度條 (slice_detections = 2D 切片級別檢測數，3D 結節聚合在測試結束後計算)
                pbar.set_postfix({
                    'patients': len(all_results['patient_features']),
                    'slice_detections': all_results['total_lesions']  # ✅ 更名為 slice_detections 避免混淆
                })

        
        # 計算患者級別摘要（使用 patient_analyzer 模組）
        for patient_id, patient_data in all_results['patient_features'].items():
            patient_summary = self.patient_analyzer.compute_patient_summary(patient_data)
            all_results['patient_features'][patient_id]['summary'] = patient_summary
        
        # 生成患者摘要可視化圖（可選）
        if visualizer is not None:
            self.logger.info("📸 正在生成患者摘要可視化圖...")
            for patient_id in tqdm(all_results['patient_features'].keys(), desc="生成患者摘要圖"):
                visualizer.create_patient_summary_grid(patient_id)
            
            # 輸出可視化統計
            vis_stats = visualizer.get_statistics()
            self.logger.info(f"📊 可視化統計: 已生成 {vis_stats['total_images']} 張圖片，{vis_stats['total_patients']} 個患者")
        
        # 計算總體測試指標
        test_summary = {}
        for key, values in all_results['test_metrics'].items():
            if values:
                test_summary[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                }
        all_results['test_summary'] = test_summary
        
        # ✅ 識別並保存預測結果不好的患者（使用 patient_analyzer 模組）
        poor_performers = self.patient_analyzer.identify_poor_performers(
            all_results['patient_features'],
            dice_threshold=0.5,
            iou_threshold=0.4
        )
        all_results['poor_performers'] = poor_performers
        
        # 保存低分患者報告（包含可視化圖片）
        self.patient_analyzer.save_poor_performers_report(poor_performers, output_dir, visualizer)
        
        # 保存結果（使用 feature_saver 模組）
        self.feature_saver.save_features(all_results, output_dir)
        
        # 輸出摘要
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"✅ 測試完成")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"總樣本數: {all_results['total_samples']}")
        self.logger.info(f"總病灶數: {all_results['total_lesions']}")
        self.logger.info(f"患者數: {len(all_results['patient_features'])}")
        
        # 輸出過濾統計
        fstats = all_results['filtering_stats']
        self.logger.info(f"\n📊 過濾統計:")
        self.logger.info(f"   總 bbox 數: {fstats['total_bboxes']}")
        self.logger.info(f"   過濾 (面積 < 50px): {fstats['filtered_by_area']} ({fstats['filtered_by_area']/max(fstats['total_bboxes'],1)*100:.1f}%)")
        self.logger.info(f"   過濾 (置信度 < {min_confidence:.2f}): {fstats['filtered_by_confidence']} ({fstats['filtered_by_confidence']/max(fstats['total_bboxes'],1)*100:.1f}%)")
        self.logger.info(f"   過濾 (Dice < {min_dice:.2f}): {fstats['filtered_by_dice']} ({fstats['filtered_by_dice']/max(fstats['total_bboxes'],1)*100:.1f}%)")
        self.logger.info(f"   ✅ 保留病灶: {fstats['kept_lesions']} ({fstats['kept_lesions']/max(fstats['total_bboxes'],1)*100:.1f}%)")
        
        if 'dice' in test_summary:
            self.logger.info(f"\n平均 Dice: {test_summary['dice']['mean']:.4f} ± {test_summary['dice']['std']:.4f}")
        if 'iou' in test_summary:
            self.logger.info(f"平均 IoU: {test_summary['iou']['mean']:.4f} ± {test_summary['iou']['std']:.4f}")
        
        # 輸出低分患者統計
        n_poor = poor_performers['summary']['total_poor_patients']
        if n_poor > 0:
            self.logger.info(f"⚠️ 低分患者: {n_poor} 個 (Dice < 0.5 或 IoU < 0.4)")
            worst_patients = poor_performers['patients'][:3]
            for wp in worst_patients:
                self.logger.info(f"   - {wp['patient_id'][:30]}...: Dice={wp['avg_dice']:.4f}")
        
        if visualizer is not None:
            self.logger.info(f"可視化圖片已保存至: {visualizer.output_dir}")
        self.logger.info(f"特徵已保存至: {output_dir}")
        self.logger.info(f"{'='*80}\n")
        
        return all_results
    
    # =========================================================================
    # 以下方法已移至獨立模組（重構說明）：
    # 
    # - _compute_patient_summary() -> patient_analyzer.py: PatientAnalyzer.compute_patient_summary()
    # - _identify_poor_performers() -> patient_analyzer.py: PatientAnalyzer.identify_poor_performers()
    # - _save_poor_performers_report() -> patient_analyzer.py: PatientAnalyzer.save_poor_performers_report()
    # - _save_features() -> feature_saver.py: FeatureSaver.save_features()
    # - _create_lite_results() -> feature_saver.py: FeatureSaver._create_lite_results()
    # - _save_patient_features() -> feature_saver.py: FeatureSaver._save_patient_features()
    # - _generate_llm_training_data() -> llm_data_generator.py: LLMDataGenerator.generate_training_data()
    # - _generate_patient_description() -> llm_data_generator.py: LLMDataGenerator.generate_patient_description()
    # =========================================================================
