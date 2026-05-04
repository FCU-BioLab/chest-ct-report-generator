#!/usr/bin/env python3
"""
RetinaNet 偵測訓練器 (Trainer)
=============================

基於 MONAI 的 3D RetinaNet 肺結節偵測訓練器。
對齊 bundles/lung_nodule_ct_detection 的配置與資料處理。
"""

import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score

from .config import RetinaNetConfig
from .dataset import (
    prepare_datalist, build_train_transform, build_val_transform,
    build_det_transform, build_rand_transform,
    get_full_split_info, get_monai_cache_dir,
)

logger = logging.getLogger(__name__)


# ─── MONAI 延遲載入 (Lazy Imports) ──────────────────────────────────
def _import_monai():
    """載入 MONAI 元件。若未安裝則拋出清楚的錯誤訊息。"""
    try:
        from monai.apps.detection.metrics.coco import COCOMetric
        from monai.apps.detection.metrics.matching import matching_batch
        from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
        from monai.apps.detection.networks.retinanet_network import (
            RetinaNet,
            resnet_fpn_feature_extractor,
        )
        from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
        from monai.data import box_utils, Dataset, PersistentDataset
        from monai.data.utils import no_collation
        from monai.networks.nets import resnet
        from monai.utils import set_determinism
        return {
            "COCOMetric": COCOMetric,
            "matching_batch": matching_batch,
            "RetinaNetDetector": RetinaNetDetector,
            "RetinaNet": RetinaNet,
            "resnet_fpn_feature_extractor": resnet_fpn_feature_extractor,
            "AnchorGeneratorWithAnchorShape": AnchorGeneratorWithAnchorShape,
            "box_utils": box_utils,
            "Dataset": Dataset,
            "PersistentDataset": PersistentDataset,
            "no_collation": no_collation,
            "resnet": resnet,
            "set_determinism": set_determinism,
        }
    except ImportError as e:
        raise ImportError(
            "未找到 MONAI detection 模組。請安裝:\n"
            "  pip install 'monai[all]>=1.3'\n"
            f"原始錯誤: {e}"
        ) from e


def _import_warmup_scheduler():
    """載入 GradualWarmupScheduler。"""
    # 嘗試從 bundle scripts 載入
    try:
        # 先加入 bundle 路徑
        bundle_scripts_dir = str(Path(__file__).resolve().parent.parent.parent / "bundles" / "lung_nodule_ct_detection")
        if bundle_scripts_dir not in sys.path:
            sys.path.insert(0, bundle_scripts_dir)
        from scripts.warmup_scheduler import GradualWarmupScheduler
        return GradualWarmupScheduler
    except ImportError:
        logger.warning("無法載入 GradualWarmupScheduler，使用手動 Warmup 替代。")
        return None


# ─── 訓練器 (Trainer) ────────────────────────────────────────────────
class RetinaNetTrainer:
    """
    遵循 MONAI LUNA16 教學模式的 3D RetinaNet 訓練器。
    """

    def __init__(self, config: RetinaNetConfig, inference_only: bool = False):
        self.config = config
        self.inference_only = inference_only
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # 輸出目錄（時間戳記）
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 設定 file logging
        self._setup_file_logging()

        # 匯入 MONAI
        self.monai = _import_monai()
        self.monai["set_determinism"](seed=config.seed)

        # 儲存 config.json
        self._save_config()

        # 建立模型
        self.detector = self._build_detector()
        self.detector.to(self.device)

        # 設定資料
        if not self.inference_only:
            self.train_loader, self.val_loader = self._setup_data()
            self._save_data_split_info()
            self._save_sample_visualization()
            self.optimizer, self.scheduler = self._setup_optimizer()
            self.scaler = torch.amp.GradScaler("cuda") if config.amp else None
            self.coco_metric = self.monai["COCOMetric"](
                classes=["nodule"],
                iou_list=config.iou_list,
                max_detection=[100],
            )
        else:
            self.train_loader = None
            self.val_loader = None
            self.optimizer = None
            self.scheduler = None
            self.scaler = None
            self.coco_metric = None

        # 歷史紀錄
        self.history = {
            "train_loss": [], "train_cls_loss": [], "train_box_loss": [],
            "val_mAP": [], "val_mAP_per_iou": {},
            "roc_fpr": [], "roc_tpr": [], "roc_auc": [],
            "pr_precision": [], "pr_recall": [], "pr_ap": [],
        }

        logger.info(f"🔧 RetinaNet Trainer 初始化完成")
        logger.info(f"  裝置: {self.device}")
        logger.info(f"  輸出路徑: {self.output_dir}")
        logger.info(f"  patch_size: {config.patch_size}")
        logger.info(f"  max_boxes_for_crop: {config.max_boxes_for_crop}")
        logger.info(f"  crop_pos_neg_ratio: {config.crop_pos_ratio}:{config.crop_neg_ratio}")
        logger.info(f"  train_pos_oversample_weight: {config.train_pos_oversample_weight}")
        logger.info(f"  train_epoch_samples: {config.train_epoch_samples}")
        logger.info(f"  feature_map_scales: {config.feature_map_scales}")
        logger.info(f"  iou_list: {config.iou_list}")

    def _setup_file_logging(self):
        """設定檔案 logging — 同時輸出到 console 和 train.log。"""
        log_file = self.output_dir / "train.log"
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        # 加到 root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        logger.info(f"📄 Log 檔案: {log_file}")

    def _save_config(self):
        """儲存完整 config 到 config.json。"""
        import dataclasses
        config_dict = dataclasses.asdict(self.config)
        config_path = self.output_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"📋 Config 已儲存至: {config_path}")

    def _save_data_split_info(self):
        """儲存資料分割資訊到 data.json。"""
        cfg = self.config
        try:
            split_info = get_full_split_info(
                cfg.data_path, cfg.train_ratio, cfg.val_ratio, cfg.test_ratio, cfg.split_seed,
            )
            data_path = self.output_dir / "data.json"
            with open(data_path, "w", encoding="utf-8") as f:
                json.dump(split_info, f, indent=2, ensure_ascii=False, default=str)
            stats = split_info["statistics"]
            logger.info(f"📊 資料分割已儲存至: {data_path}")
            logger.info(
                f"  Train: {stats['train_total']} ({stats['train_with_nodule']} with nodule) | "
                f"Val: {stats['val_total']} ({stats['val_with_nodule']} with nodule) | "
                f"Test: {stats['test_total']} ({stats['test_with_nodule']} with nodule)"
            )
        except Exception as e:
            logger.warning(f"無法儲存 data.json: {e}")

    def _save_sample_visualization(self, n_samples: int = 4):
        """儲存訓練樣本可視化 — 顯示影像切片與 bounding box。"""
        cfg = self.config
        try:
            import matplotlib.patches as mpatches
            from .dataset import build_val_transform
            val_transform = build_val_transform(
                spacing=cfg.spacing, hu_min=cfg.hu_min, hu_max=cfg.hu_max,
            )

            train_data = prepare_datalist(
                cfg.data_path, "train",
                cfg.train_ratio, cfg.val_ratio, cfg.test_ratio, cfg.split_seed,
            )

            # 選取有 box 的樣本
            samples_with_box = [it for it in train_data if len(it.get("box", [])) > 0]
            selected = samples_with_box[:n_samples]

            if len(selected) == 0:
                logger.info("  ⚠️ 無有效樣本可供可視化")
                return

            n_cols = min(n_samples, len(selected))
            fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10))
            if n_cols == 1:
                axes = axes[:, np.newaxis]

            for i, item in enumerate(selected):
                if i >= n_cols:
                    break

                processed = val_transform(item)
                # Orientationd("RAS") 後：
                #   image shape = (1, dim0, dim1, dim2) = (1, Y, X, Z)
                #   box = [dim0_min, dim1_min, dim2_min, dim0_max, dim1_max, dim2_max]
                #       = [Y_min,    X_min,    Z_min,    Y_max,    X_max,    Z_max]
                image = processed["image"].numpy()
                boxes = processed["box"].numpy()

                _, nY, nX, nZ = image.shape

                if len(boxes) > 0:
                    box0 = boxes[0]
                    center_z = int((box0[2] + box0[5]) / 2)  # Z 軸中心
                    center_y = int((box0[0] + box0[3]) / 2)  # Y 軸中心
                else:
                    center_z = nZ // 2
                    center_y = nY // 2

                center_z = max(0, min(center_z, nZ - 1))
                center_y = max(0, min(center_y, nY - 1))

                # ─── Axial (固定 Z，顯示 Y × X) ───
                # imshow(shape=(Y,X)) → rows=Y, cols=X
                # Rectangle(x=col, y=row) → (X_min=box[1], Y_min=box[0])
                ax_ax = axes[0, i]
                ax_ax.imshow(image[0, :, :, center_z], cmap="gray")
                for box in boxes:
                    if box[2] <= center_z <= box[5]:
                        rect = mpatches.Rectangle(
                            (box[1], box[0]),          # (col=X_min, row=Y_min)
                            box[4] - box[1],           # width  = X range
                            box[3] - box[0],           # height = Y range
                            linewidth=2, edgecolor='lime', facecolor='none',
                        )
                        ax_ax.add_patch(rect)
                sid = Path(item.get('image', '')).stem[:25]
                ax_ax.set_title(f"Axial (Z={center_z})\n{sid}…", fontsize=8)
                ax_ax.axis("off")

                # ─── Coronal (固定 Y，顯示 Z × X) ───
                # image[0, center_y, :, :] = (X, Z) → 但我們要 (Z rows, X cols)
                # → 取 image[0, center_y, :, :].T 或直接 slice 成 (Z, X)
                # image[0, :, :, :] 的 shape 是 (Y, X, Z)
                # image[0, y_idx, :, :] 的 shape 是 (X, Z) → .T = (Z, X)
                ax_cor = axes[1, i]
                coronal_slice = image[0, center_y, :, :].T  # (Z, X)
                ax_cor.imshow(coronal_slice, cmap="gray", origin="lower")
                for box in boxes:
                    if box[0] <= center_y <= box[3]:
                        rect = mpatches.Rectangle(
                            (box[1], box[2]),          # (col=X_min, row=Z_min)
                            box[4] - box[1],           # width  = X range
                            box[5] - box[2],           # height = Z range
                            linewidth=2, edgecolor='cyan', facecolor='none',
                        )
                        ax_cor.add_patch(rect)
                ax_cor.set_title(f"Coronal (Y={center_y})", fontsize=8)
                ax_cor.axis("off")

            plt.suptitle("Sample Visualization (Post-Transform)", fontsize=12)
            plt.tight_layout()
            viz_path = self.output_dir / "sample_visualization.png"
            plt.savefig(viz_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"🖼️ 樣本可視化已儲存至: {viz_path}")

        except Exception as e:
            logger.warning(f"樣本可視化失敗: {e}", exc_info=True)

    def _build_detector(self):
        """建立 MONAI RetinaNetDetector。對齊 Bundle train_luna16.json。"""
        cfg = self.config
        M = self.monai

        # 1) 錨點生成器 — 使用 per-axis feature_map_scales
        feature_map_scales = [
            torch.as_tensor(scale, dtype=torch.float32)
            for scale in cfg.feature_map_scales
        ]
        anchor_generator = M["AnchorGeneratorWithAnchorShape"](
            feature_map_scales=feature_map_scales,
            base_anchor_shapes=cfg.base_anchor_shapes,
        )

        # 2) 骨幹網路 (Backbone): ResNet50 + FPN
        conv1_t_size = [max(7, 2 * s + 1) for s in cfg.conv1_t_stride]
        backbone = M["resnet"].ResNet(
            block=M["resnet"].ResNetBottleneck,
            layers=[3, 4, 6, 3],  # ResNet50
            block_inplanes=M["resnet"].get_inplanes(),
            n_input_channels=cfg.n_input_channels,
            conv1_t_stride=cfg.conv1_t_stride,
            conv1_t_size=conv1_t_size,
        )
        feature_extractor = M["resnet_fpn_feature_extractor"](
            backbone=backbone,
            spatial_dims=cfg.spatial_dims,
            pretrained_backbone=False,
            trainable_backbone_layers=None,
            returned_layers=cfg.returned_layers,
        )

        # 3) RetinaNet — 使用固定 size_divisible
        num_anchors = anchor_generator.num_anchors_per_location()[0]

        net = M["RetinaNet"](
            spatial_dims=cfg.spatial_dims,
            num_classes=cfg.num_classes,
            num_anchors=num_anchors,
            feature_extractor=feature_extractor,
            size_divisible=cfg.size_divisible,
        )

        # 載入預訓練權重
        if cfg.pretrained_weights and os.path.exists(cfg.pretrained_weights):
            pretrained_state = None
            load_error = None

            try:
                loaded_obj = torch.load(
                    cfg.pretrained_weights, map_location="cpu", weights_only=False
                )
                if isinstance(loaded_obj, dict) and "model_state_dict" in loaded_obj:
                    loaded_obj = loaded_obj["model_state_dict"]
                if isinstance(loaded_obj, dict):
                    pretrained_state = loaded_obj
                else:
                    load_error = f"Unsupported checkpoint object: {type(loaded_obj)}"
            except Exception as exc:
                load_error = str(exc)

            if pretrained_state is None:
                try:
                    scripted = torch.jit.load(cfg.pretrained_weights, map_location="cpu")
                    pretrained_state = scripted.state_dict()
                except Exception as exc:
                    load_error = f"{load_error}; torch.jit.load failed: {exc}" if load_error else str(exc)

            if pretrained_state is None:
                logger.warning(
                    f"  ⚠️ 無法載入預訓練權重，將改用隨機初始化: {cfg.pretrained_weights} ({load_error})"
                )
            else:
                matched = net.load_state_dict(pretrained_state, strict=False)
                n_total = len(pretrained_state)
                n_loaded = n_total - len(matched.unexpected_keys)
                logger.info(
                    f"  ✅ 已載入預訓練權重: {cfg.pretrained_weights}"
                    f" ({n_loaded}/{n_total} keys matched)"
                )
                if matched.missing_keys:
                    logger.info(f"  ⚠️ Missing keys: {matched.missing_keys}")
        elif cfg.pretrained_weights:
            logger.warning(f"  ⚠️ 預訓練權重檔案不存在: {cfg.pretrained_weights}")

        net = torch.jit.script(net)

        # 4) 偵測器包裝
        detector = M["RetinaNetDetector"](
            network=net, anchor_generator=anchor_generator, debug=False
        )

        # 訓練元件設定
        detector.set_atss_matcher(
            num_candidates=cfg.atss_num_candidates, center_in_gt=False
        )
        detector.set_hard_negative_sampler(
            batch_size_per_image=cfg.hn_batch_size_per_image,
            positive_fraction=cfg.hn_positive_fraction,
            pool_size=cfg.hn_pool_size,
            min_neg=cfg.hn_min_neg,
        )
        detector.set_target_keys(box_key="box", label_key="label")

        # 推論元件設定
        detector.set_box_selector_parameters(
            score_thresh=cfg.proposal_score_thresh,
            topk_candidates_per_level=cfg.topk_candidates_per_level,
            nms_thresh=cfg.nms_thresh,
            detections_per_img=cfg.detections_per_img,
        )
        detector.set_sliding_window_inferer(
            roi_size=cfg.val_patch_size,
            overlap=0.25,
            sw_batch_size=1,
            mode="constant",
            device="cpu",
        )

        logger.info(f"  每位置錨點數: {num_anchors}")
        logger.info(f"  尺寸整除倍數: {cfg.size_divisible}")
        return detector

    def _setup_data(self):
        """準備訓練與驗證 DataLoader。使用 MONAI Transform Pipeline。"""
        cfg = self.config
        M = self.monai
        from monai.transforms import Compose

        train_data = prepare_datalist(
            cfg.data_path, "train",
            cfg.train_ratio, cfg.val_ratio, cfg.test_ratio, cfg.split_seed,
        )
        val_data = prepare_datalist(
            cfg.data_path, "val",
            cfg.train_ratio, cfg.val_ratio, cfg.test_ratio, cfg.split_seed,
        )

        # 過濾不存在的檔案
        def _filter_existing(data_list):
            valid = []
            missing = 0
            for item in data_list:
                img_path = item.get("image", "")
                if os.path.exists(img_path):
                    valid.append(item)
                else:
                    missing += 1
            if missing > 0:
                logger.warning(f"  ⚠️ 跳過 {missing} 筆不存在的檔案")
            return valid

        train_data = _filter_existing(train_data)
        val_data = _filter_existing(val_data)

        val_transform = build_val_transform(
            spacing=cfg.spacing,
            hu_min=cfg.hu_min,
            hu_max=cfg.hu_max,
        )

        if cfg.cache_dataset:
            # PersistentDataset: 快取到磁碟，避免 RAM 不足
            det_transform = build_det_transform(
                spacing=cfg.spacing,
                hu_min=cfg.hu_min,
                hu_max=cfg.hu_max,
            )
            rand_transform = build_rand_transform(
                patch_size=cfg.patch_size,
                batch_size=cfg.batch_size,
                max_boxes_for_crop=cfg.max_boxes_for_crop,
                crop_pos_ratio=cfg.crop_pos_ratio,
                crop_neg_ratio=cfg.crop_neg_ratio,
            )

            cache_dir = get_monai_cache_dir()
            cache_dir.mkdir(parents=True, exist_ok=True)

            # 組合 deterministic + random transforms
            # PersistentDataset 自動偵測 Randomizable transforms，
            # 只快取 deterministic 部分，每次載入後重新套用 random 部分
            full_train_transform = Compose([det_transform, rand_transform])

            logger.info(f"  📦 使用 PersistentDataset 磁碟快取: {cache_dir}")
            train_ds = M["PersistentDataset"](
                train_data,
                transform=full_train_transform,
                cache_dir=str(cache_dir / "train"),
            )

            val_ds = M["PersistentDataset"](
                val_data,
                transform=val_transform,
                cache_dir=str(cache_dir / "val"),
            )
        else:
            # 傳統 Dataset
            train_transform = build_train_transform(
                patch_size=cfg.patch_size,
                batch_size=cfg.batch_size,
                max_boxes_for_crop=cfg.max_boxes_for_crop,
                crop_pos_ratio=cfg.crop_pos_ratio,
                crop_neg_ratio=cfg.crop_neg_ratio,
                spacing=cfg.spacing,
                hu_min=cfg.hu_min,
                hu_max=cfg.hu_max,
            )
            train_ds = M["Dataset"](train_data, transform=train_transform)
            val_ds = M["Dataset"](val_data, transform=val_transform)

        # Optional case-level oversampling:
        # increase chance of drawing scans with nodules before patch-level RandCrop.
        train_sampler = None
        train_shuffle = True
        pos_flags = [1 if len((it.get("box", []) or [])) > 0 else 0 for it in train_data]
        if len(pos_flags) > 0 and (cfg.train_pos_oversample_weight > 1.0 or cfg.train_epoch_samples > 0):
            sample_weights = [
                float(cfg.train_pos_oversample_weight) if flag else 1.0
                for flag in pos_flags
            ]
            num_samples = int(cfg.train_epoch_samples) if cfg.train_epoch_samples > 0 else len(sample_weights)
            train_sampler = WeightedRandomSampler(
                weights=torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=num_samples,
                replacement=True,
            )
            train_shuffle = False

            pos_cnt = int(sum(pos_flags))
            neg_cnt = int(len(pos_flags) - pos_cnt)
            weighted_pos = float(cfg.train_pos_oversample_weight) * pos_cnt
            expected_pos_ratio = weighted_pos / max(weighted_pos + float(neg_cnt), 1.0)
            logger.info(
                "  啟用 case-level oversampling: pos_weight=%.3f, epoch_samples=%d",
                cfg.train_pos_oversample_weight,
                num_samples,
            )
            logger.info(
                "  預期被抽中正樣本比例約 %.2f%% (dataset pos=%d, neg=%d)",
                expected_pos_ratio * 100.0,
                pos_cnt,
                neg_cnt,
            )

        train_loader = DataLoader(
            train_ds,
            batch_size=1,  # MONAI detection 使用 batch_size=1 配合 no_collation
            shuffle=train_shuffle,
            sampler=train_sampler,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=M["no_collation"],
            persistent_workers=cfg.num_workers > 0,
        )
        # ⚠️ 驗證用 num_workers=0 (主進程載入)
        # 全尺寸 CT (512×512×600+) 在 Windows 上會耗盡共享記憶體 (error 1455)
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            collate_fn=M["no_collation"],
        )

        logger.info(f"  訓練集樣本數: {len(train_ds)}")
        logger.info(f"  驗證集樣本數: {len(val_ds)}")
        logger.info(f"  每個 epoch 訓練 steps: {len(train_loader)}")
        return train_loader, val_loader

    def _setup_optimizer(self):
        """設定 SGD 優化器 + GradualWarmupScheduler + StepLR。"""
        cfg = self.config

        optimizer = torch.optim.SGD(
            self.detector.network.parameters(),
            lr=cfg.lr,
            momentum=0.9,
            weight_decay=cfg.weight_decay,
            nesterov=True,
        )

        # 嘗試使用 GradualWarmupScheduler
        GradualWarmupScheduler = _import_warmup_scheduler()

        after_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma
        )

        if GradualWarmupScheduler is not None:
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,
                total_epoch=cfg.warmup_epochs,
                after_scheduler=after_scheduler,
            )
            self._use_manual_warmup = False
            logger.info(f"  使用 GradualWarmupScheduler (warmup={cfg.warmup_epochs})")
        else:
            scheduler = after_scheduler
            self._use_manual_warmup = True
            logger.info(f"  使用手動 Warmup (warmup={cfg.warmup_epochs})")

        return optimizer, scheduler

    # ─── 訓練流程 ────────────────────────────────────────────────
    def train(self, resume_checkpoint: Optional[str] = None):
        """主訓練迴圈。"""
        cfg = self.config
        best_metric = -1.0
        best_epoch = -1
        start_epoch = 0
        last_epoch = -1
        early_stop_patience = int(getattr(cfg, "early_stop_patience", 0) or 0)
        early_stop_min_delta = float(getattr(cfg, "early_stop_min_delta", 0.0) or 0.0)
        early_stop_enabled = early_stop_patience > 0

        if resume_checkpoint:
            resume_path = Path(resume_checkpoint)
            if resume_path.exists():
                start_epoch, best_metric, best_epoch = self._load_training_state(resume_path)
                logger.info(
                    "⏯️ Resume enabled: %s (start_epoch=%d, best_mAP=%.4f @ epoch %d)",
                    resume_path,
                    start_epoch + 1,
                    best_metric,
                    best_epoch,
                )
            else:
                logger.warning("⚠️ Resume checkpoint not found: %s. Training from scratch.", resume_path)

        logger.info(f"🚀 開始 RetinaNet 訓練，共 {cfg.epochs} Epochs")

        if early_stop_enabled:
            logger.info(
                "EarlyStopping enabled: patience=%d epochs, min_delta=%.6f (monitor=mAP)",
                early_stop_patience,
                early_stop_min_delta,
            )

        for epoch in range(start_epoch, cfg.epochs):
            last_epoch = epoch
            # 訓練一個 Epoch
            train_metrics = self._train_epoch(epoch)

            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_cls_loss"].append(train_metrics["cls_loss"])
            self.history["train_box_loss"].append(train_metrics["box_loss"])

            self.scheduler.step()

            # 儲存最新模型
            self._save_model(self.output_dir / "model_last.pt")
            self._save_training_state(
                self.output_dir / "train_state_last.pt",
                epoch=epoch,
                best_metric=best_metric,
                best_epoch=best_epoch,
            )

            # 驗證
            stop_training = False
            if (epoch + 1) % cfg.val_interval == 0:
                logger.info("  開始驗證...")
                try:
                    val_metrics = self._validate(epoch)
                    self.history["val_mAP"].append(val_metrics.get("mAP", 0))
                    self.history["val_mAP_per_iou"] = val_metrics.get("coco", {})

                    # ROC / PR curves
                    curves = val_metrics.get("_curves", {})
                    self.history["roc_fpr"] = curves.get("roc_fpr", [])
                    self.history["roc_tpr"] = curves.get("roc_tpr", [])
                    self.history["roc_auc"] = val_metrics.get("roc_auc", 0)
                    self.history["pr_precision"] = curves.get("pr_precision", [])
                    self.history["pr_recall"] = curves.get("pr_recall", [])
                    self.history["pr_ap"] = val_metrics.get("pr_ap", 0)

                    # FROC & F1
                    froc = val_metrics.get("froc", {})
                    self.history["froc_score"] = froc.get("froc_score", 0)
                    self.history["detection_f1"] = val_metrics.get("detection_f1", 0)
                    self.history["detection_precision"] = val_metrics.get("detection_precision", 0)
                    self.history["detection_recall"] = val_metrics.get("detection_recall", 0)

                    current_map = val_metrics.get("mAP", 0)
                    improved = current_map > (best_metric + early_stop_min_delta)

                    if improved:
                        best_metric = current_map
                        best_epoch = epoch + 1
                        self._save_model(self.output_dir / "model_best.pt")
                        self._save_training_state(
                            self.output_dir / "train_state_best.pt",
                            epoch=epoch,
                            best_metric=best_metric,
                            best_epoch=best_epoch,
                        )
                        logger.info(f"  💾 儲存新的最佳模型 (mAP={best_metric:.4f})")

                    elif early_stop_enabled and best_epoch > 0:
                        epochs_since_best = (epoch + 1) - best_epoch
                        logger.info(
                            "  EarlyStopping: no improvement for %d/%d epochs",
                            epochs_since_best,
                            early_stop_patience,
                        )
                        if epochs_since_best >= early_stop_patience:
                            logger.info(
                                "Early stopping triggered at epoch %d (best mAP=%.4f @ epoch %d)",
                                epoch + 1,
                                best_metric,
                                best_epoch,
                            )
                            stop_training = True

                    logger.info(
                        f"  Epoch {epoch+1}: mAP={current_map:.4f} | "
                        f"F1={val_metrics.get('detection_f1', 0):.4f} | "
                        f"FROC={froc.get('froc_score', 0):.4f} | "
                        f"歷史最佳: {best_metric:.4f} @ epoch {best_epoch}"
                    )
                except Exception as e:
                    logger.error(f"  ❌ 驗證失敗: {e}", exc_info=True)

            # 儲存歷程
            self._save_history()
            self._plot_curves()
            if stop_training:
                break

        logger.info(
            f"✅ 訓練完成。最佳 mAP: {best_metric:.4f} (Epoch {best_epoch})"
        )

        # ─── 訓練結束後自動執行測試集評估 ───
        final_epoch = last_epoch if last_epoch >= 0 else cfg.epochs - 1
        self._save_training_state(
            self.output_dir / "train_state_final.pt",
            epoch=final_epoch,
            best_metric=best_metric,
            best_epoch=best_epoch,
        )
        self._run_test_evaluation()

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """執行單一訓練 Epoch。"""
        cfg = self.config
        self.detector.train()

        epoch_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_box_loss = 0.0
        step = 0

        # 手動 Warmup (僅在無 GradualWarmupScheduler 時使用)
        if self._use_manual_warmup and epoch < cfg.warmup_epochs:
            warmup_factor = (epoch + 1) / cfg.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg["lr"] = cfg.lr * warmup_factor

        start_time = time.time()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}", leave=True, ncols=100)

        for batch_data in pbar:
            step += 1

            # RandCropBoxByPosNegLabeld (num_samples=batch_size) 會讓 DataLoader
            # 回傳 list of list。需展平為 list of dict。
            # 格式: batch_data = [ [sample_0_crop_0, sample_0_crop_1, ...] ]
            # (因為 DataLoader batch_size=1, collate_fn=no_collation)
            inputs = [
                batch_data_ii["image"].to(self.device)
                for batch_data_i in batch_data
                for batch_data_ii in batch_data_i
            ]
            targets = [
                {
                    "label": batch_data_ii["label"].to(self.device),
                    "box": batch_data_ii["box"].to(self.device),
                }
                for batch_data_i in batch_data
                for batch_data_ii in batch_data_i
            ]

            # 清除梯度
            for param in self.detector.network.parameters():
                param.grad = None

            if cfg.amp and self.scaler is not None:
                with torch.amp.autocast("cuda"):
                    outputs = self.detector(inputs, targets)
                    loss = (
                        cfg.w_cls * outputs[self.detector.cls_key]
                        + outputs[self.detector.box_reg_key]
                    )
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.detector(inputs, targets)
                loss = (
                    cfg.w_cls * outputs[self.detector.cls_key]
                    + outputs[self.detector.box_reg_key]
                )
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.detach().item()
            epoch_cls_loss += outputs[self.detector.cls_key].detach().item()
            epoch_box_loss += outputs[self.detector.box_reg_key].detach().item()

            pbar.set_postfix({
                "Loss": f"{loss.detach().item():.4f}",
                "Cls": f"{outputs[self.detector.cls_key].detach().item():.4f}",
                "Box": f"{outputs[self.detector.box_reg_key].detach().item():.4f}",
            })

        elapsed = time.time() - start_time
        avg_loss = epoch_loss / max(step, 1)
        avg_cls = epoch_cls_loss / max(step, 1)
        avg_box = epoch_box_loss / max(step, 1)

        logger.info(
            f"📊 Epoch {epoch+1} summary: AvgLoss={avg_loss:.4f} "
            f"(Cls={avg_cls:.4f}, Box={avg_box:.4f}) "
            f"[{elapsed:.1f}s] LR={self.optimizer.param_groups[0]['lr']:.6f}"
        )

        del inputs, targets
        torch.cuda.empty_cache()
        gc.collect()

        return {"loss": avg_loss, "cls_loss": avg_cls, "box_loss": avg_box}

    def _validate(self, epoch: int) -> Dict[str, float]:
        """執行驗證並計算 COCO mAP 及其他指標。"""
        cfg = self.config
        self.detector.eval()

        val_outputs_all = []
        val_targets_all = []

        start_time = time.time()

        pbar = tqdm(self.val_loader, desc=f"Validation {epoch+1}", leave=False, ncols=100)

        # 釋放訓練階段的 GPU 記憶體
        torch.cuda.empty_cache()

        with torch.no_grad():
            for val_data in pbar:
                # 驗證時無 RandCropBoxByPosNegLabeld，所以 val_data 不是 nested list
                # val_data = [dict] (因 no_collation + batch_size=1)

                # 檢查是否需要滑動視窗
                use_inferer = not all(
                    item["image"][0, ...].numel() < np.prod(cfg.val_patch_size)
                    for item in val_data
                )

                val_inputs = [item.pop("image").to(self.device) for item in val_data]

                if cfg.amp:
                    with torch.amp.autocast("cuda"):
                        val_outputs = self.detector(val_inputs, use_inferer=use_inferer)
                else:
                    val_outputs = self.detector(val_inputs, use_inferer=use_inferer)

                val_outputs_all += val_outputs
                val_targets_all += val_data

        elapsed = time.time() - start_time
        logger.info(f"  驗證耗時: {elapsed:.1f}s")

        # 計算所有指標
        del val_inputs
        torch.cuda.empty_cache()

        M = self.monai

        pred_boxes_list = [o[self.detector.target_box_key].cpu().numpy() for o in val_outputs_all]
        pred_scores_list = [o[self.detector.pred_score_key].cpu().numpy() for o in val_outputs_all]
        pred_labels_list = [o[self.detector.target_label_key].cpu().numpy() for o in val_outputs_all]
        gt_boxes_list = [t[self.detector.target_box_key].cpu().numpy() for t in val_targets_all]
        gt_labels_list = [t[self.detector.target_label_key].cpu().numpy() for t in val_targets_all]

        from .metrics import compute_all_metrics
        all_metrics = compute_all_metrics(
            pred_boxes_list=pred_boxes_list,
            pred_scores_list=pred_scores_list,
            pred_labels_list=pred_labels_list,
            gt_boxes_list=gt_boxes_list,
            gt_labels_list=gt_labels_list,
            iou_thresh_match=cfg.iou_list[0],
            coco_metric=self.coco_metric,
            matching_batch_fn=M["matching_batch"],
            box_iou_fn=M["box_utils"].box_iou,
        )

        return all_metrics

    def _apply_prediction_mask(self, output: Dict, keep_mask) -> int:
        pb = output[self.detector.target_box_key]
        ps = output[self.detector.pred_score_key]
        pl = output[self.detector.target_label_key]
        original_count = len(pb)
        if original_count == 0:
            return 0
        if not isinstance(keep_mask, torch.Tensor):
            keep_mask = torch.tensor(keep_mask, device=pb.device, dtype=torch.bool)
        output[self.detector.target_box_key] = pb[keep_mask]
        output[self.detector.pred_score_key] = ps[keep_mask]
        output[self.detector.target_label_key] = pl[keep_mask]
        return int(original_count - int(keep_mask.sum().item()))

    def _filter_predictions_by_lung_mask(self, output: Dict, image_tensor: torch.Tensor) -> int:
        from .postprocess import generate_lung_mask

        pb = output[self.detector.target_box_key]
        if len(pb) == 0:
            return 0

        img_np = image_tensor[0].cpu().numpy()
        lung_mask = generate_lung_mask(img_np, thresh_val=0.3)
        keep_mask = []
        for box in pb.cpu().numpy():
            cx = min(max(int((box[0] + box[3]) / 2), 0), img_np.shape[0] - 1)
            cy = min(max(int((box[1] + box[4]) / 2), 0), img_np.shape[1] - 1)
            cz = min(max(int((box[2] + box[5]) / 2), 0), img_np.shape[2] - 1)
            keep_mask.append(bool(lung_mask[cx, cy, cz]))
        return self._apply_prediction_mask(output, keep_mask)

    def _filter_predictions_by_morphology(
        self,
        output: Dict,
        image_tensor: torch.Tensor,
        spacing: np.ndarray,
        max_elongation: float,
        min_solidity: float,
        min_vol_mm3: float,
        max_vol_mm3: float,
    ) -> int:
        from skimage import measure, morphology

        pb = output[self.detector.target_box_key]
        if len(pb) == 0:
            return 0

        img_np = image_tensor[0].cpu().numpy()
        keep_mask = np.ones(len(pb), dtype=bool)
        for b_idx, box in enumerate(pb.cpu().numpy()):
            dx = (box[3] - box[0]) * spacing[0]
            dy = (box[4] - box[1]) * spacing[1]
            dz = (box[5] - box[2]) * spacing[2]
            vol_mm3 = dx * dy * dz
            if vol_mm3 < min_vol_mm3 or vol_mm3 > max_vol_mm3:
                keep_mask[b_idx] = False
                continue

            pad = 2
            x0 = max(int(box[0]) - pad, 0)
            y0 = max(int(box[1]) - pad, 0)
            z0 = max(int(box[2]) - pad, 0)
            x1 = min(int(box[3]) + pad + 1, img_np.shape[0])
            y1 = min(int(box[4]) + pad + 1, img_np.shape[1])
            z1 = min(int(box[5]) + pad + 1, img_np.shape[2])
            patch = img_np[x0:x1, y0:y1, z0:z1]
            patch_bin = morphology.binary_closing(patch > 0.333, morphology.ball(1))
            if not np.any(patch_bin):
                continue

            label_img = measure.label(patch_bin)
            props = measure.regionprops(label_img)
            if not props:
                continue

            largest_prop = max(props, key=lambda p: p.area)
            try:
                major_axis = largest_prop.major_axis_length
                minor_axis = largest_prop.minor_axis_length
            except ValueError:
                continue
            elongation = 0.0 if minor_axis == 0 else major_axis / minor_axis
            solidity = largest_prop.solidity
            if elongation > max_elongation or solidity < min_solidity:
                keep_mask[b_idx] = False

        return self._apply_prediction_mask(output, keep_mask)

    @staticmethod
    def _count_bad_morphology_planes(
        patch_bin: np.ndarray,
        max_elongation: float,
        min_solidity: float,
    ) -> int:
        from skimage import measure

        projections = (
            np.any(patch_bin, axis=2),  # axial: y-x
            np.any(patch_bin, axis=1),  # sagittal: y-z
            np.any(patch_bin, axis=0),  # coronal: x-z
        )
        bad_planes = 0
        for proj in projections:
            if not np.any(proj):
                continue
            label_img = measure.label(proj)
            props = measure.regionprops(label_img)
            if not props:
                continue
            largest_prop = max(props, key=lambda p: p.area)
            minor_axis = float(getattr(largest_prop, "minor_axis_length", 0.0) or 0.0)
            major_axis = float(getattr(largest_prop, "major_axis_length", 0.0) or 0.0)
            elongation = float("inf") if minor_axis <= 1e-6 and major_axis > 0 else major_axis / max(minor_axis, 1e-6)
            solidity = float(getattr(largest_prop, "solidity", 1.0) or 1.0)
            if elongation > float(max_elongation) or solidity < float(min_solidity):
                bad_planes += 1
        return bad_planes

    @staticmethod
    def _has_round_similar_morphology_planes(
        patch_bin: np.ndarray,
        max_round_elongation: float,
        min_solidity: float,
        min_area_ratio: float,
        min_round_planes: int,
    ) -> bool:
        from itertools import combinations
        from skimage import measure

        projections = (
            np.any(patch_bin, axis=2),  # axial: y-x
            np.any(patch_bin, axis=1),  # sagittal: y-z
            np.any(patch_bin, axis=0),  # coronal: x-z
        )
        round_planes = []
        for proj in projections:
            if not np.any(proj):
                continue
            label_img = measure.label(proj)
            props = measure.regionprops(label_img)
            if not props:
                continue
            largest_prop = max(props, key=lambda p: p.area)
            minor_axis = float(getattr(largest_prop, "minor_axis_length", 0.0) or 0.0)
            major_axis = float(getattr(largest_prop, "major_axis_length", 0.0) or 0.0)
            if major_axis <= 1e-6:
                continue
            elongation = float("inf") if minor_axis <= 1e-6 else major_axis / minor_axis
            solidity = float(getattr(largest_prop, "solidity", 1.0) or 1.0)
            if elongation <= float(max_round_elongation) and solidity >= float(min_solidity):
                round_planes.append(float(largest_prop.area))

        min_round_planes = max(1, int(min_round_planes))
        if len(round_planes) < min_round_planes:
            return False

        min_area_ratio = float(min_area_ratio)
        for areas in combinations(round_planes, min_round_planes):
            min_area = min(areas)
            max_area = max(areas)
            if max_area <= 0:
                continue
            if min_area / max_area >= min_area_ratio:
                return True
        return False

    @staticmethod
    def _projection_shape_features(proj: np.ndarray) -> Optional[Dict[str, float]]:
        from skimage import measure

        if not np.any(proj):
            return None
        label_img = measure.label(proj)
        props = measure.regionprops(label_img)
        if not props:
            return None
        largest_prop = max(props, key=lambda p: p.area)
        minor_axis = float(getattr(largest_prop, "minor_axis_length", 0.0) or 0.0)
        major_axis = float(getattr(largest_prop, "major_axis_length", 0.0) or 0.0)
        if major_axis <= 1e-6:
            return None
        elongation = float("inf") if minor_axis <= 1e-6 else major_axis / minor_axis
        solidity = float(getattr(largest_prop, "solidity", 1.0) or 1.0)
        minr, minc, maxr, maxc = largest_prop.bbox
        bbox_area = max(1.0, float((maxr - minr) * (maxc - minc)))
        fill = float(largest_prop.area) / bbox_area
        return {
            "area": float(largest_prop.area),
            "elongation": float(elongation),
            "solidity": float(solidity),
            "fill": float(fill),
        }

    @classmethod
    def _has_axial_similar_side_plane(
        cls,
        patch_bin: np.ndarray,
        max_elongation_delta: float,
        min_solidity_delta: float,
        min_fill_delta: float,
        min_area_ratio: float,
    ) -> bool:
        axial = cls._projection_shape_features(np.any(patch_bin, axis=2))
        if axial is None:
            return False
        side_planes = (
            cls._projection_shape_features(np.any(patch_bin, axis=1)),  # sagittal
            cls._projection_shape_features(np.any(patch_bin, axis=0)),  # coronal
        )
        for side in side_planes:
            if side is None:
                continue
            area_ratio = min(axial["area"], side["area"]) / max(axial["area"], side["area"], 1.0)
            if area_ratio < float(min_area_ratio):
                continue
            if abs(axial["elongation"] - side["elongation"]) > float(max_elongation_delta):
                continue
            if abs(axial["solidity"] - side["solidity"]) > float(min_solidity_delta):
                continue
            if abs(axial["fill"] - side["fill"]) > float(min_fill_delta):
                continue
            return True
        return False

    @staticmethod
    def _proposal_morphology_features(
        patch_arr: np.ndarray,
        max_elongation: float = 4.0,
        min_fill_ratio: float = 0.35,
    ) -> Dict[str, np.ndarray]:
        from skimage import measure, morphology

        n = int(len(patch_arr))
        plane_names = ("axial", "coronal", "sagittal")
        elongations = {name: np.ones(n, dtype=np.float32) for name in plane_names}
        fill_ratios = {name: np.ones(n, dtype=np.float32) for name in plane_names}
        bad_counts = np.zeros(n, dtype=np.float32)
        valid_counts = np.zeros(n, dtype=np.float32)

        for i, patch in enumerate(patch_arr):
            patch_bin = morphology.binary_closing(patch > 0.333, morphology.ball(1))
            projections = {
                "axial": np.any(patch_bin, axis=2),
                "coronal": np.any(patch_bin, axis=0),
                "sagittal": np.any(patch_bin, axis=1),
            }
            for name, proj in projections.items():
                if not np.any(proj):
                    elongations[name][i] = 999.0
                    fill_ratios[name][i] = 0.0
                    bad_counts[i] += 1.0
                    continue
                label_img = measure.label(proj)
                props = measure.regionprops(label_img)
                if not props:
                    elongations[name][i] = 999.0
                    fill_ratios[name][i] = 0.0
                    bad_counts[i] += 1.0
                    continue
                largest = max(props, key=lambda p: p.area)
                minor_axis = float(getattr(largest, "minor_axis_length", 0.0) or 0.0)
                major_axis = float(getattr(largest, "major_axis_length", 0.0) or 0.0)
                elong = 999.0 if minor_axis <= 1e-6 and major_axis > 0 else major_axis / max(minor_axis, 1e-6)
                minr, minc, maxr, maxc = largest.bbox
                bbox_area = max(1.0, float((maxr - minr) * (maxc - minc)))
                fill = float(largest.area) / bbox_area
                elongations[name][i] = np.float32(min(elong, 999.0))
                fill_ratios[name][i] = np.float32(fill)
                valid_counts[i] += 1.0
                if elong > float(max_elongation) or fill < float(min_fill_ratio):
                    bad_counts[i] += 1.0

        max_elong = np.maximum.reduce([elongations[name] for name in plane_names])
        mean_elong = np.mean(np.stack([elongations[name] for name in plane_names], axis=0), axis=0).astype(np.float32)
        min_fill = np.minimum.reduce([fill_ratios[name] for name in plane_names])
        mean_fill = np.mean(np.stack([fill_ratios[name] for name in plane_names], axis=0), axis=0).astype(np.float32)
        return {
            "morph_axial_elongation": elongations["axial"],
            "morph_coronal_elongation": elongations["coronal"],
            "morph_sagittal_elongation": elongations["sagittal"],
            "morph_axial_fill": fill_ratios["axial"],
            "morph_coronal_fill": fill_ratios["coronal"],
            "morph_sagittal_fill": fill_ratios["sagittal"],
            "morph_max_elongation": max_elong.astype(np.float32, copy=False),
            "morph_mean_elongation": mean_elong,
            "morph_min_fill": min_fill.astype(np.float32, copy=False),
            "morph_mean_fill": mean_fill,
            "morph_bad_plane_count": bad_counts.astype(np.float32, copy=False),
            "morph_good_plane_count": (3.0 - bad_counts).astype(np.float32, copy=False),
            "morph_valid_plane_count": valid_counts.astype(np.float32, copy=False),
        }

    def _fuse_fpr_scores(
        self,
        output: Dict,
        image_tensor: torch.Tensor,
        fpr_model,
        fpr_fuser,
        fpr_patch_size: int,
        fpr_weight: float,
        fpr_thresh: float,
        fpr_mode: str,
        fpr_score_aware: bool = False,
        fpr_det_high_thresh: float = 0.9,
        fpr_det_mid_thresh: float = 0.6,
        fpr_high_thresh: float = 0.15,
        fpr_mid_thresh: float = 0.25,
        fpr_apply_min_diam: float = None,
        fpr_apply_max_diam: float = None,
        size_aware_small_diam: float = 0.0,
        size_aware_fpr_thresh: float = None,
    ) -> Dict[str, Any]:
        from .collect_fpr_data import crop_patch
        from .fpr_fuser import predict_fused_prob
        from .fpr_model import (
            batch_patches_to_model_input,
            get_model_backbone_from_model,
            get_model_num_slices_per_view,
            get_model_type_from_model,
        )

        pb = output[self.detector.target_box_key]
        ps = output[self.detector.pred_score_key]
        if len(pb) == 0:
            return {"rescored": 0, "filtered": 0, "trace": []}

        img_np = image_tensor[0].cpu().numpy()
        patches = []
        boxes_np = pb.detach().cpu().numpy().astype(np.float32, copy=False)
        det_scores_np = ps.detach().cpu().numpy().astype(np.float32, copy=False)
        for box in boxes_np:
            center = ((box[0] + box[3]) / 2, (box[1] + box[4]) / 2, (box[2] + box[5]) / 2)
            patches.append(crop_patch(img_np, center, fpr_patch_size))

        fpr_model_type = get_model_type_from_model(fpr_model)
        fpr_backbone = get_model_backbone_from_model(fpr_model)
        fpr_num_slices_per_view = get_model_num_slices_per_view(fpr_model)
        patch_arr = np.stack(patches).astype(np.float32, copy=False)
        patch_batch = batch_patches_to_model_input(
            patch_arr,
            fpr_model_type,
            num_slices_per_view=fpr_num_slices_per_view,
        )
        patches_tensor = torch.from_numpy(patch_batch).to(self.device)
        with torch.no_grad():
            logits = fpr_model(patches_tensor)
            probs = torch.softmax(logits, dim=1)
            nodule_prob = probs[:, 1].to(ps.device)
        nodule_prob_np = nodule_prob.detach().cpu().numpy().astype(np.float32, copy=False)

        dx = np.clip(boxes_np[:, 3] - boxes_np[:, 0], a_min=0.0, a_max=None)
        dy = np.clip(boxes_np[:, 4] - boxes_np[:, 1], a_min=0.0, a_max=None)
        dz = np.clip(boxes_np[:, 5] - boxes_np[:, 2], a_min=0.0, a_max=None)
        vol = dx * dy * dz
        min_axis = np.maximum(np.minimum(np.minimum(dx, dy), dz), 1e-3)
        max_axis = np.maximum(np.maximum(dx, dy), dz)
        spacing_np = np.asarray(self.config.spacing, dtype=np.float32)
        dx_mm = dx * float(spacing_np[0])
        dy_mm = dy * float(spacing_np[1])
        dz_mm = dz * float(spacing_np[2])
        max_axis_mm = np.maximum(np.maximum(dx_mm, dy_mm), dz_mm)
        active_fpr_np = np.ones(len(boxes_np), dtype=bool)
        if fpr_apply_min_diam is not None:
            active_fpr_np &= max_axis_mm >= float(fpr_apply_min_diam)
        if fpr_apply_max_diam is not None:
            active_fpr_np &= max_axis_mm <= float(fpr_apply_max_diam)
        active_fpr = torch.as_tensor(active_fpr_np, device=ps.device, dtype=torch.bool)
        small_box_np = (max_axis_mm <= float(size_aware_small_diam)) if size_aware_small_diam and size_aware_small_diam > 0 else np.zeros(len(boxes_np), dtype=bool)
        small_box = torch.as_tensor(small_box_np, device=ps.device, dtype=torch.bool)
        patch_mean = patch_arr.mean(axis=(1, 2, 3))
        patch_std = patch_arr.std(axis=(1, 2, 3))
        patch_p90 = np.percentile(patch_arr, 90, axis=(1, 2, 3))
        fuser_extra = {
            "log_volume": torch.from_numpy(np.log1p(vol).astype(np.float32, copy=False)).to(ps.device),
            "elongation": torch.from_numpy((max_axis / min_axis).astype(np.float32, copy=False)).to(ps.device),
            "patch_mean": torch.from_numpy(patch_mean.astype(np.float32, copy=False)).to(ps.device),
            "patch_std": torch.from_numpy(patch_std.astype(np.float32, copy=False)).to(ps.device),
            "patch_p90": torch.from_numpy(patch_p90.astype(np.float32, copy=False)).to(ps.device),
        }
        morph_features_np = self._proposal_morphology_features(patch_arr)
        final_score_np = det_scores_np.copy()
        keep_mask_np = np.ones(len(boxes_np), dtype=bool)
        applied_thresh_np = np.full(len(boxes_np), np.nan, dtype=np.float32)
        det_band = np.array(["all"] * len(boxes_np), dtype=object)

        if fpr_mode == "learned":
            if fpr_fuser is None:
                raise ValueError("fpr_mode='learned' requires fpr_fuser_model_path.")
            fused_prob = predict_fused_prob(
                fpr_fuser,
                ps,
                nodule_prob,
                self.device,
                extra_features=fuser_extra,
            ).to(ps.device)
            final_score_all = torch.where(active_fpr, fused_prob, ps)
            final_score_np = final_score_all.detach().cpu().numpy().astype(np.float32, copy=False)
            filtered = 0
            if fpr_thresh is not None and fpr_thresh > 0:
                keep_mask = (~active_fpr) | (fused_prob >= fpr_thresh)
                keep_mask_np = keep_mask.detach().cpu().numpy().astype(bool, copy=False)
                applied_thresh_np[:] = float(fpr_thresh)
                applied_thresh_np = np.where(active_fpr_np, applied_thresh_np, np.nan).astype(np.float32, copy=False)
                det_band[:] = "learned"
                det_band = np.where(active_fpr_np, det_band, "inactive_size")
                filtered = self._apply_prediction_mask(output, keep_mask)
            else:
                det_band[:] = "learned"
                det_band = np.where(active_fpr_np, det_band, "inactive_size")
            trace = [
                {
                    "proposal_index": int(i),
                    "box": [float(v) for v in boxes_np[i].tolist()],
                    "det_score": float(det_scores_np[i]),
                    "fpr_prob": float(nodule_prob_np[i]),
                    "final_score_after_fpr": float(final_score_np[i]),
                    "keep_after_fpr": bool(keep_mask_np[i]),
                    "removed_by_fpr": bool(not keep_mask_np[i]),
                    "det_band": str(det_band[i]),
                    "applied_threshold": None if np.isnan(applied_thresh_np[i]) else float(applied_thresh_np[i]),
                    "fpr_mode": fpr_mode,
                    "max_diameter_mm": float(max_axis_mm[i]),
                    "fpr_active": bool(active_fpr_np[i]),
                    "size_aware_small": bool(small_box_np[i]),
                    **{name: float(values[i]) for name, values in morph_features_np.items()},
                }
                for i in range(len(boxes_np))
            ]
            if len(output[self.detector.target_box_key]) == 0:
                return {"rescored": int(len(pb)), "filtered": int(filtered), "trace": trace}
            if fpr_thresh is not None and fpr_thresh > 0:
                final_score_all = final_score_all[keep_mask]
            output[self.detector.pred_score_key] = final_score_all
            return {"rescored": int(len(pb)), "filtered": int(filtered), "trace": trace}

        filtered = 0
        if fpr_mode in {"gate", "hybrid"}:
            if fpr_score_aware:
                det_scores = ps
                keep_high_band = det_scores >= fpr_det_high_thresh
                keep_mid_band = (det_scores >= fpr_det_mid_thresh) & (det_scores < fpr_det_high_thresh)
                keep_low_band = det_scores < fpr_det_mid_thresh
                keep_high = nodule_prob >= fpr_high_thresh
                keep_mid = nodule_prob >= fpr_mid_thresh
                keep_low = nodule_prob >= fpr_thresh
                if size_aware_fpr_thresh is not None and size_aware_small_diam and size_aware_small_diam > 0:
                    keep_small = nodule_prob >= float(size_aware_fpr_thresh)
                    keep_high = torch.where(small_box, keep_small, keep_high)
                    keep_mid = torch.where(small_box, keep_small, keep_mid)
                    keep_low = torch.where(small_box, keep_small, keep_low)
                keep_mask = (~active_fpr) | (keep_high_band & keep_high) | (keep_mid_band & keep_mid) | (keep_low_band & keep_low)
                keep_high_band_np = keep_high_band.detach().cpu().numpy().astype(bool, copy=False)
                keep_mid_band_np = keep_mid_band.detach().cpu().numpy().astype(bool, copy=False)
                det_band = np.where(keep_high_band_np, "high", np.where(keep_mid_band_np, "mid", "low"))
                applied_thresh_np = np.where(
                    keep_high_band_np,
                    np.float32(fpr_high_thresh),
                    np.where(keep_mid_band_np, np.float32(fpr_mid_thresh), np.float32(fpr_thresh)),
                ).astype(np.float32, copy=False)
                if size_aware_fpr_thresh is not None and size_aware_small_diam and size_aware_small_diam > 0:
                    applied_thresh_np = np.where(small_box_np, np.float32(size_aware_fpr_thresh), applied_thresh_np).astype(np.float32, copy=False)
                    det_band = np.where(small_box_np, np.char.add(det_band.astype(str), "_small"), det_band)
                applied_thresh_np = np.where(active_fpr_np, applied_thresh_np, np.nan).astype(np.float32, copy=False)
                det_band = np.where(active_fpr_np, det_band, "inactive_size")
            else:
                fpr_thresholds = torch.full_like(nodule_prob, float(fpr_thresh))
                if size_aware_fpr_thresh is not None and size_aware_small_diam and size_aware_small_diam > 0:
                    fpr_thresholds = torch.where(small_box, torch.full_like(nodule_prob, float(size_aware_fpr_thresh)), fpr_thresholds)
                keep_mask = (~active_fpr) | (nodule_prob >= fpr_thresholds)
                applied_thresh_np[:] = float(fpr_thresh)
                if size_aware_fpr_thresh is not None and size_aware_small_diam and size_aware_small_diam > 0:
                    applied_thresh_np = np.where(small_box_np, np.float32(size_aware_fpr_thresh), applied_thresh_np).astype(np.float32, copy=False)
                det_band[:] = "all"
                if size_aware_fpr_thresh is not None and size_aware_small_diam and size_aware_small_diam > 0:
                    det_band = np.where(small_box_np, "all_small", det_band)
                applied_thresh_np = np.where(active_fpr_np, applied_thresh_np, np.nan).astype(np.float32, copy=False)
                det_band = np.where(active_fpr_np, det_band, "inactive_size")
            keep_mask_np = keep_mask.detach().cpu().numpy().astype(bool, copy=False)
            filtered = self._apply_prediction_mask(output, keep_mask)
            ps = output[self.detector.pred_score_key]
            if len(output[self.detector.target_box_key]) > 0:
                nodule_prob = nodule_prob[keep_mask]
                active_fpr = active_fpr[keep_mask]

        if fpr_mode in {"fuse", "hybrid"}:
            final_score_np = det_scores_np * (1.0 - float(fpr_weight)) + nodule_prob_np * float(fpr_weight)
            final_score_np = np.where(active_fpr_np, final_score_np, det_scores_np).astype(np.float32, copy=False)
            if len(output[self.detector.target_box_key]) > 0:
                fused_scores = ps * (1.0 - fpr_weight) + nodule_prob * fpr_weight
                output[self.detector.pred_score_key] = torch.where(active_fpr, fused_scores, ps)

        trace = [
            {
                "proposal_index": int(i),
                "box": [float(v) for v in boxes_np[i].tolist()],
                "det_score": float(det_scores_np[i]),
                "fpr_prob": float(nodule_prob_np[i]),
                "final_score_after_fpr": float(final_score_np[i]),
                "keep_after_fpr": bool(keep_mask_np[i]),
                "removed_by_fpr": bool(not keep_mask_np[i]),
                "max_diameter_mm": float(max_axis_mm[i]),
                "fpr_active": bool(active_fpr_np[i]),
                "size_aware_small": bool(small_box_np[i]),
                **{name: float(values[i]) for name, values in morph_features_np.items()},
                "det_band": str(det_band[i]),
                    "applied_threshold": None if np.isnan(applied_thresh_np[i]) else float(applied_thresh_np[i]),
                    "fpr_mode": fpr_mode,
                    "fpr_backbone": str(fpr_backbone),
                    "fpr_num_slices_per_view": int(fpr_num_slices_per_view),
                }
                for i in range(len(boxes_np))
            ]
        return {"rescored": int(len(pb)), "filtered": int(filtered), "trace": trace}

    def _apply_score_threshold_to_arrays(self, pred_boxes_list, pred_scores_list, pred_labels_list, score_thresh: float) -> int:
        n_filtered = 0
        for i in range(len(pred_scores_list)):
            scores = pred_scores_list[i]
            keep_mask = scores >= score_thresh
            n_filtered += int(np.sum(~keep_mask))
            pred_boxes_list[i] = pred_boxes_list[i][keep_mask]
            pred_scores_list[i] = scores[keep_mask]
            pred_labels_list[i] = pred_labels_list[i][keep_mask]
        return n_filtered

    def _apply_size_aware_score_threshold_to_arrays(
        self,
        pred_boxes_list,
        pred_scores_list,
        pred_labels_list,
        score_thresh: float,
        small_diam: float,
        small_score_thresh: float,
    ) -> int:
        n_filtered = 0
        for i in range(len(pred_scores_list)):
            boxes = pred_boxes_list[i]
            scores = pred_scores_list[i]
            if len(scores) == 0:
                continue
            dx = np.clip(boxes[:, 3] - boxes[:, 0], a_min=0.0, a_max=None)
            dy = np.clip(boxes[:, 4] - boxes[:, 1], a_min=0.0, a_max=None)
            dz = np.clip(boxes[:, 5] - boxes[:, 2], a_min=0.0, a_max=None)
            spacing_np = np.asarray(self.config.spacing, dtype=np.float32)
            max_diam = np.maximum(np.maximum(dx * float(spacing_np[0]), dy * float(spacing_np[1])), dz * float(spacing_np[2]))
            thresholds = np.where(max_diam <= float(small_diam), float(small_score_thresh), float(score_thresh))
            keep_mask = scores >= thresholds
            n_filtered += int(np.sum(~keep_mask))
            pred_boxes_list[i] = boxes[keep_mask]
            pred_scores_list[i] = scores[keep_mask]
            pred_labels_list[i] = pred_labels_list[i][keep_mask]
        return n_filtered

    def _apply_bbox_aspect_filter_to_output(
        self,
        output: Dict,
        max_aspect_ratio: float,
        skip_small_diam: float = 0.0,
    ) -> int:
        boxes = output[self.detector.target_box_key]
        scores = output[self.detector.pred_score_key]
        labels = output[self.detector.target_label_key]
        if len(boxes) == 0:
            return 0

        spacing = torch.as_tensor(self.config.spacing, device=boxes.device, dtype=boxes.dtype)
        dims_mm = torch.clamp(boxes[:, 3:6] - boxes[:, 0:3], min=0.0) * spacing
        max_axis = torch.max(dims_mm, dim=1).values
        min_axis = torch.clamp(torch.min(dims_mm, dim=1).values, min=1e-3)
        aspect = max_axis / min_axis
        keep = aspect <= float(max_aspect_ratio)
        if skip_small_diam and skip_small_diam > 0:
            keep = keep | (max_axis <= float(skip_small_diam))

        n_filtered = int((~keep).sum().item())
        if n_filtered > 0:
            output[self.detector.target_box_key] = boxes[keep]
            output[self.detector.pred_score_key] = scores[keep]
            output[self.detector.target_label_key] = labels[keep]
        return n_filtered

    @staticmethod
    def _box_iou_3d_single_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        if boxes.size == 0:
            return np.zeros((0,), dtype=np.float32)
        ixmin = np.maximum(box[0], boxes[:, 0])
        iymin = np.maximum(box[1], boxes[:, 1])
        izmin = np.maximum(box[2], boxes[:, 2])
        ixmax = np.minimum(box[3], boxes[:, 3])
        iymax = np.minimum(box[4], boxes[:, 4])
        izmax = np.minimum(box[5], boxes[:, 5])

        inter = np.clip(ixmax - ixmin, 0, None) * np.clip(iymax - iymin, 0, None) * np.clip(izmax - izmin, 0, None)
        vol_box = np.clip(box[3] - box[0], 0, None) * np.clip(box[4] - box[1], 0, None) * np.clip(box[5] - box[2], 0, None)
        vol_boxes = np.clip(boxes[:, 3] - boxes[:, 0], 0, None) * np.clip(boxes[:, 4] - boxes[:, 1], 0, None) * np.clip(boxes[:, 5] - boxes[:, 2], 0, None)
        union = vol_box + vol_boxes - inter
        return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0).astype(np.float32, copy=False)

    def _ensemble_merge_single_sample(
        self,
        sample_outputs: List[Dict[str, torch.Tensor]],
        num_models: int,
        iou_thresh: float,
        vote_power: float,
    ) -> Dict[str, torch.Tensor]:
        box_key = self.detector.target_box_key
        score_key = self.detector.pred_score_key
        label_key = self.detector.target_label_key

        base_device = sample_outputs[0][box_key].device
        base_box_dtype = sample_outputs[0][box_key].dtype
        base_score_dtype = sample_outputs[0][score_key].dtype
        base_label_dtype = sample_outputs[0][label_key].dtype

        boxes_chunks = []
        scores_chunks = []
        labels_chunks = []
        for output in sample_outputs:
            boxes = output[box_key].detach().cpu().numpy()
            scores = output[score_key].detach().cpu().numpy()
            labels = output[label_key].detach().cpu().numpy()
            if len(boxes) == 0:
                continue
            boxes_chunks.append(boxes.astype(np.float32, copy=False))
            scores_chunks.append(scores.astype(np.float32, copy=False))
            labels_chunks.append(labels.astype(np.int64, copy=False))

        if not boxes_chunks:
            return {
                box_key: torch.empty((0, 6), device=base_device, dtype=base_box_dtype),
                score_key: torch.empty((0,), device=base_device, dtype=base_score_dtype),
                label_key: torch.empty((0,), device=base_device, dtype=base_label_dtype),
            }

        all_boxes = np.concatenate(boxes_chunks, axis=0)
        all_scores = np.concatenate(scores_chunks, axis=0)
        all_labels = np.concatenate(labels_chunks, axis=0)

        fused_boxes = []
        fused_scores = []
        fused_labels = []

        for cls_id in np.unique(all_labels):
            cls_mask = all_labels == cls_id
            cls_boxes = all_boxes[cls_mask]
            cls_scores = all_scores[cls_mask]
            if len(cls_boxes) == 0:
                continue

            order = np.argsort(-cls_scores)
            while order.size > 0:
                ref_idx = int(order[0])
                ious = self._box_iou_3d_single_to_many(cls_boxes[ref_idx], cls_boxes[order])
                cluster_mask = ious >= float(iou_thresh)
                cluster_order = order[cluster_mask]

                cluster_boxes = cls_boxes[cluster_order]
                cluster_scores = cls_scores[cluster_order]
                weights = np.maximum(cluster_scores, 1e-6)

                fused_box = (cluster_boxes * weights[:, None]).sum(axis=0) / weights.sum()
                vote_factor = (len(cluster_order) / max(int(num_models), 1)) ** max(float(vote_power), 0.0)
                fused_score = float(cluster_scores.mean() * vote_factor)

                fused_boxes.append(fused_box.astype(np.float32, copy=False))
                fused_scores.append(fused_score)
                fused_labels.append(int(cls_id))
                order = order[~cluster_mask]

        if not fused_boxes:
            return {
                box_key: torch.empty((0, 6), device=base_device, dtype=base_box_dtype),
                score_key: torch.empty((0,), device=base_device, dtype=base_score_dtype),
                label_key: torch.empty((0,), device=base_device, dtype=base_label_dtype),
            }

        fused_boxes_np = np.stack(fused_boxes, axis=0).astype(np.float32, copy=False)
        fused_scores_np = np.asarray(fused_scores, dtype=np.float32)
        fused_labels_np = np.asarray(fused_labels, dtype=np.int64)
        keep_order = np.argsort(-fused_scores_np)

        return {
            box_key: torch.as_tensor(fused_boxes_np[keep_order], device=base_device, dtype=base_box_dtype),
            score_key: torch.as_tensor(fused_scores_np[keep_order], device=base_device, dtype=base_score_dtype),
            label_key: torch.as_tensor(fused_labels_np[keep_order], device=base_device, dtype=base_label_dtype),
        }

    def _run_test_evaluation(self, save_gifs: bool = False, gif_dir: str = None, filter_fp: bool = False, score_thresh: float = None, filter_lung_mask: bool = False, override_model_path: str = None, ensemble_model_paths: List[str] = None, ensemble_iou_thresh: float = None, ensemble_vote_power: float = 1.0, fpr_model_path: str = None, fpr_thresh: float = 0.5, fpr_patch_size: int = 32, fpr_weight: float = 0.5, fpr_mode: str = "hybrid", fp_max_elongation: float = 5.0, fp_min_solidity: float = 0.3, fp_min_vol: float = 4.2, fp_max_vol: float = 65450.0, morph_skip_small_diam: float = 0.0, morph_three_plane: bool = False, morph_min_bad_planes: int = 2, morph_require_round_planes: bool = False, morph_min_round_planes: int = 2, morph_max_round_elongation: float = 1.8, morph_min_plane_area_ratio: float = 0.5, morph_axial_similarity: bool = False, morph_max_elongation_delta: float = 1.5, morph_max_solidity_delta: float = 0.35, morph_max_fill_delta: float = 0.35, morph_min_axial_area_ratio: float = 0.35, bbox_filter: bool = False, bbox_max_aspect_ratio: float = 3.5, bbox_aspect_skip_small_diam: float = 0.0, eval_split: str = "test", max_samples: int = 0, fpr_score_aware: bool = False, fpr_det_high_thresh: float = 0.9, fpr_det_mid_thresh: float = 0.6, fpr_high_thresh: float = 0.15, fpr_mid_thresh: float = 0.25, fpr_apply_min_diam: float = None, fpr_apply_max_diam: float = None, fpr_fuser_model_path: str = None, size_aware_small_diam: float = 0.0, size_aware_fpr_thresh: float = None, size_aware_final_thresh: float = None, export_case_analysis: bool = False, case_analysis_dir: str = None):
        """訓練結束後，自動載入最佳模型並在測試集上評估。"""
        cfg = self.config
        M = self.monai
        if fpr_mode not in {"fuse", "gate", "hybrid", "learned"}:
            raise ValueError(f"Unsupported fpr_mode: {fpr_mode}")
        if fpr_mode == "learned" and not fpr_fuser_model_path:
            raise ValueError("fpr_mode='learned' requires --fpr_fuser_model.")
        if fpr_mode == "learned" and not fpr_model_path:
            raise ValueError("fpr_mode='learned' requires --fpr_model.")
        if fpr_score_aware:
            if fpr_det_mid_thresh > fpr_det_high_thresh:
                raise ValueError(
                    f"Invalid score-aware thresholds: det_mid({fpr_det_mid_thresh}) > det_high({fpr_det_high_thresh})"
                )
            if not (fpr_high_thresh <= fpr_mid_thresh <= fpr_thresh):
                logger.warning(
                    "  Score-aware FPR thresholds are non-monotonic (high=%.2f, mid=%.2f, low=%.2f).",
                    float(fpr_high_thresh),
                    float(fpr_mid_thresh),
                    float(fpr_thresh),
                )
            logger.info(
                "  Score-aware gate enabled: det>=%.2f use fpr>=%.2f | %.2f<=det<%.2f use fpr>=%.2f | det<%.2f use fpr>=%.2f",
                float(fpr_det_high_thresh),
                float(fpr_high_thresh),
                float(fpr_det_mid_thresh),
                float(fpr_det_high_thresh),
                float(fpr_mid_thresh),
                float(fpr_det_mid_thresh),
                float(fpr_thresh),
            )
        final_score_thresh = cfg.test_score_thresh if score_thresh is None else score_thresh
        logger.info(
            "  Two-stage thresholds: candidate=%.4f, final=%.4f",
            float(cfg.proposal_score_thresh),
            float(final_score_thresh),
        )

        ensemble_paths = [str(Path(p)) for p in (ensemble_model_paths or []) if p]
        if not ensemble_paths:
            best_model_path = Path(override_model_path) if override_model_path else self.output_dir / "model_best.pt"
            ensemble_paths = [str(best_model_path)]
        for model_path in ensemble_paths:
            if not Path(model_path).exists():
                logger.warning(f"  ⚠️ 找不到模型: {model_path}，跳過測試集評估")
                return

        ensemble_enabled = len(ensemble_paths) > 1
        ensemble_iou = float(ensemble_iou_thresh if ensemble_iou_thresh is not None else cfg.nms_thresh)
        if ensemble_iou <= 0:
            ensemble_iou = float(cfg.nms_thresh)

        logger.info("🧪 開始測試集評估...")

        ensemble_networks = []
        for model_path in ensemble_paths:
            try:
                net = torch.jit.load(str(model_path)).to(self.device)
                net.eval()
                ensemble_networks.append(net)
                logger.info("  📂 已載入模型: %s", model_path)
            except RuntimeError:
                if len(ensemble_paths) == 1:
                    ensemble_networks.append(self.detector.network)
                else:
                    raise
        self.detector.network = ensemble_networks[0]
        if ensemble_enabled:
            logger.info(
                "  🧠 Ensemble enabled: %d models | iou=%.3f | vote_power=%.2f",
                len(ensemble_networks),
                ensemble_iou,
                float(ensemble_vote_power),
            )

        # 準備測試資料
        test_data = prepare_datalist(
            cfg.data_path, eval_split,
            cfg.train_ratio, cfg.val_ratio, cfg.test_ratio, cfg.split_seed,
        )
        if max_samples and max_samples > 0:
            test_data = test_data[:max_samples]

        # 過濾不存在的檔案
        test_data = [d for d in test_data if os.path.exists(d.get("image", ""))]

        if len(test_data) == 0:
            logger.warning("  ⚠️ 測試集為空，跳過評估")
            return

        val_transform = build_val_transform(
            spacing=cfg.spacing,
            hu_min=cfg.hu_min,
            hu_max=cfg.hu_max,
        )

        if cfg.cache_dataset:
            cache_dir = get_monai_cache_dir("test")
            cache_dir.mkdir(parents=True, exist_ok=True)
            test_ds = M["PersistentDataset"](
                test_data,
                transform=val_transform,
                cache_dir=str(cache_dir),
            )
        else:
            test_ds = M["Dataset"](test_data, transform=val_transform)

        # ⚠️ 測試用 num_workers=0，避免 Windows 共享記憶體不足
        test_loader = DataLoader(
            test_ds,
            batch_size=1,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            collate_fn=M["no_collation"],
        )

        logger.info(f"  評估分割: {eval_split}")
        logger.info(f"  評估樣本數: {len(test_ds)}")

        # 執行推論
        self.detector.eval()
        test_outputs_all = []
        test_targets_all = []
        fpr_stats = {"rescored": 0, "filtered": 0}
        bbox_filter_stats = {"filtered": 0}

        start_time = time.time()
        torch.cuda.empty_cache()

        pbar = tqdm(test_loader, desc="Testing", leave=False, ncols=100)

        case_analysis_out_dir = None
        case_summaries = []
        if export_case_analysis:
            case_analysis_out_dir = Path(case_analysis_dir) if case_analysis_dir else (self.output_dir / "case_analysis")
            case_analysis_out_dir.mkdir(exist_ok=True, parents=True)
            logger.info("  Per-case analysis export enabled: %s", case_analysis_out_dir)

        # 載入 FPR 分類器 (如果有提供)
        _fpr_model = None
        if fpr_model_path is not None:
            from .fpr_model import (
                get_model_backbone_from_model,
                get_model_num_slices_per_view,
                get_model_type_from_model,
                load_fpr_model,
            )
            _fpr_model = load_fpr_model(fpr_model_path, device=str(self.device))
            logger.info(
                "  🧠 已載入 FPR 分類器: %s | type=%s backbone=%s slices/view=%d",
                fpr_model_path,
                get_model_type_from_model(_fpr_model),
                get_model_backbone_from_model(_fpr_model),
                get_model_num_slices_per_view(_fpr_model),
            )
        
        _fpr_fuser = None
        if fpr_fuser_model_path is not None:
            from .fpr_fuser import load_fpr_fuser
            _fpr_fuser = load_fpr_fuser(fpr_fuser_model_path, device=str(self.device))
            logger.info(f"  🧮 已載入 learned fuser: {fpr_fuser_model_path}")
            logger.info(
                "  🧮 learned fuser suggested threshold: %.2f",
                float(_fpr_fuser.get("meta", {}).get("best_threshold", 0.5)),
            )
        if save_gifs:
            if gif_dir:
                gif_out_dir = Path(gif_dir)
            else:
                gif_out_dir = self.output_dir / "test_gifs"
            gif_out_dir.mkdir(exist_ok=True, parents=True)
            from .visualize_predictions import create_prediction_gif
            logger.info(f"  🎞️ 測試集動態預測圖將儲存至: {gif_out_dir}")

        with torch.no_grad():
            for test_data_batch in pbar:
                import numpy as np
                use_inferer = not all(
                    item["image"][0, ...].numel() < np.prod(cfg.val_patch_size)
                    for item in test_data_batch
                )
                # 因為 image pop 出來會跑到 GPU 但原始檔案路徑通常還在 original_image 等 metadata
                test_inputs = []
                test_affines = []
                test_source_paths = []
                for item in test_data_batch:
                    image_mt = item.pop("image")
                    test_inputs.append(image_mt.to(self.device))
                    affine = getattr(image_mt, "affine", None)
                    if affine is None:
                        affine = np.eye(4, dtype=np.float32)
                    elif torch.is_tensor(affine):
                        affine = affine.detach().cpu().numpy()
                    else:
                        affine = np.asarray(affine)
                    test_affines.append(affine.astype(np.float32, copy=False))
                    meta = getattr(image_mt, "meta", {}) or {}
                    source_path = meta.get("filename_or_obj") or item.get("image_meta_dict", {}).get("filename_or_obj")
                    test_source_paths.append(str(source_path) if source_path else None)

                if ensemble_enabled:
                    outputs_per_model = []
                    for net in ensemble_networks:
                        self.detector.network = net
                        if cfg.amp:
                            with torch.amp.autocast("cuda"):
                                model_outputs = self.detector(test_inputs, use_inferer=use_inferer)
                        else:
                            model_outputs = self.detector(test_inputs, use_inferer=use_inferer)
                        outputs_per_model.append(model_outputs)

                    test_outputs = []
                    for sample_idx in range(len(test_inputs)):
                        merged_output = self._ensemble_merge_single_sample(
                            sample_outputs=[model_outputs[sample_idx] for model_outputs in outputs_per_model],
                            num_models=len(ensemble_networks),
                            iou_thresh=ensemble_iou,
                            vote_power=ensemble_vote_power,
                        )
                        test_outputs.append(merged_output)
                    self.detector.network = ensemble_networks[0]
                else:
                    if cfg.amp:
                        with torch.amp.autocast("cuda"):
                            test_outputs = self.detector(test_inputs, use_inferer=use_inferer)
                    else:
                        test_outputs = self.detector(test_inputs, use_inferer=use_inferer)

                # 3. 肺部遮罩過濾 (Lung Masking)
                if filter_lung_mask:
                    from .postprocess import generate_lung_mask
                    import numpy as np
                    
                    for i in range(len(test_inputs)):
                        # 生成遮罩 (需要原始的 CPU影像陣列)
                        img_np = test_inputs[i][0].cpu().numpy()
                        lung_mask = generate_lung_mask(img_np, thresh_val=0.3)
                        
                        # 取得當前掃描預測框
                        pb = test_outputs[i][self.detector.target_box_key]
                        ps = test_outputs[i][self.detector.pred_score_key]
                        pl = test_outputs[i][self.detector.target_label_key]
                        
                        if len(pb) > 0:
                            pb_np = pb.cpu().numpy()
                            keep_mask = []
                            for box in pb_np:
                                # 取預測框中心點，判斷中心點是否落在肺部遮罩內
                                cx = min(max(int((box[0] + box[3]) / 2), 0), img_np.shape[0] - 1)
                                cy = min(max(int((box[1] + box[4]) / 2), 0), img_np.shape[1] - 1)
                                cz = min(max(int((box[2] + box[5]) / 2), 0), img_np.shape[2] - 1)
                                keep_mask.append(bool(lung_mask[cx, cy, cz]))
                                
                            keep_mask_tensor = torch.tensor(keep_mask, device=pb.device, dtype=torch.bool)
                            
                            # 過濾掉肺部以外的框
                            test_outputs[i][self.detector.target_box_key] = pb[keep_mask_tensor]
                            test_outputs[i][self.detector.pred_score_key] = ps[keep_mask_tensor]
                            test_outputs[i][self.detector.target_label_key] = pl[keep_mask_tensor]
                # 4. 形態學過濾 (Morphological Voxel-level FP Reduction)
                if filter_fp:
                    from skimage import measure, morphology
                    import numpy as np
                    
                    min_vol_mm3 = fp_min_vol      # 預設 ~直徑 2mm 球體
                    max_vol_mm3 = fp_max_vol     # 預設 ~直徑 50mm 球體
                    max_elongation = fp_max_elongation  # 管狀物(血管)過濾
                    min_solidity = fp_min_solidity      # 實心度過濾

                    spacing_np = np.array(cfg.spacing) # [1.0, 1.0, 1.0]

                    for i in range(len(test_inputs)):
                        img_np = test_inputs[i][0].cpu().numpy()
                        pb = test_outputs[i][self.detector.target_box_key]
                        ps = test_outputs[i][self.detector.pred_score_key]
                        pl = test_outputs[i][self.detector.target_label_key]

                        if len(pb) == 0:
                            continue

                        pb_np = pb.cpu().numpy()
                        keep_mask = np.ones(len(pb_np), dtype=bool)

                        for b_idx, box in enumerate(pb_np):
                            # 1. 物理尺寸過濾
                            dx = (box[3] - box[0]) * spacing_np[0]
                            dy = (box[4] - box[1]) * spacing_np[1]
                            dz = (box[5] - box[2]) * spacing_np[2]
                            if morph_skip_small_diam and morph_skip_small_diam > 0 and max(dx, dy, dz) <= float(morph_skip_small_diam):
                                continue
                            vols = dx * dy * dz
                            
                            if vols < min_vol_mm3 or vols > max_vol_mm3:
                                keep_mask[b_idx] = False
                                continue
                            
                            # 2. 擷取 BBox 內的 3D Voxel 進行分析
                            # 擴大一點邊界 (padding 2 pixels) 來捕捉完整形狀
                            pad = 2
                            x0 = max(int(box[0]) - pad, 0)
                            y0 = max(int(box[1]) - pad, 0)
                            z0 = max(int(box[2]) - pad, 0)
                            x1 = min(int(box[3]) + pad + 1, img_np.shape[0])
                            y1 = min(int(box[4]) + pad + 1, img_np.shape[1])
                            z1 = min(int(box[5]) + pad + 1, img_np.shape[2])

                            patch = img_np[x0:x1, y0:y1, z0:z1]
                            # 簡單閾值分割: -600 HU 以上視為實體組織/血管/結節
                            # 因為前處理已經把 HU 轉成了 [0, 1] 之間 (基於 -1000 ~ 200)
                            # 換算: threshold_hu = -600 -> norm_val = (-600 - (-1000)) / 1200 = 400/1200 = 0.333
                            patch_bin = patch > 0.333
                            
                            # 填補小洞，讓結節/血管更完整
                            patch_bin = morphology.binary_closing(patch_bin, morphology.ball(1))
                            
                            # 若全空，則無法判斷，保守保留
                            if not np.any(patch_bin):
                                continue
                                
                            label_img = measure.label(patch_bin)
                            props = measure.regionprops(label_img)
                            
                            if not props:
                                continue
                                
                            # 取最大的連通區域 (假設中心點就是目標物)
                            largest_prop = max(props, key=lambda p: p.area)
                            
                            # 提取高階形狀特徵
                            # 極扁平區域 (如 z 軸僅 1 slice) 會導致 skimage
                            # minor_axis_length 內部 sqrt 負值，保守保留
                            try:
                                ma_len = largest_prop.major_axis_length
                                mi_len = largest_prop.minor_axis_length
                            except ValueError:
                                continue
                            # 3D solidity uses a convex hull and is unstable for thin/flat components.
                            # Use bbox fill ratio as a conservative fallback when not using three-plane checks.
                            bbox = largest_prop.bbox
                            bbox_vol = max(
                                1.0,
                                float((bbox[3] - bbox[0]) * (bbox[4] - bbox[1]) * (bbox[5] - bbox[2])),
                            )
                            solidity = float(largest_prop.area) / bbox_vol
                            
                            # 避免除以零
                            if mi_len == 0:
                                elongation = 0
                            else:
                                elongation = ma_len / mi_len
                                
                            # 判斷是否為血管 (長條狀) 或形狀極度不規則
                            if morph_axial_similarity:
                                should_remove = not self._has_axial_similar_side_plane(
                                    patch_bin=patch_bin,
                                    max_elongation_delta=morph_max_elongation_delta,
                                    min_solidity_delta=morph_max_solidity_delta,
                                    min_fill_delta=morph_max_fill_delta,
                                    min_area_ratio=morph_min_axial_area_ratio,
                                )
                            elif morph_require_round_planes:
                                should_remove = not self._has_round_similar_morphology_planes(
                                    patch_bin=patch_bin,
                                    max_round_elongation=morph_max_round_elongation,
                                    min_solidity=min_solidity,
                                    min_area_ratio=morph_min_plane_area_ratio,
                                    min_round_planes=morph_min_round_planes,
                                )
                            elif morph_three_plane:
                                bad_plane_count = self._count_bad_morphology_planes(
                                    patch_bin=patch_bin,
                                    max_elongation=max_elongation,
                                    min_solidity=min_solidity,
                                )
                                should_remove = bad_plane_count >= max(1, int(morph_min_bad_planes))
                            else:
                                should_remove = elongation > max_elongation or solidity < min_solidity

                            if should_remove:
                                keep_mask[b_idx] = False

                        # 更新過濾後的框
                        keep_mask_tensor = torch.tensor(keep_mask, device=pb.device, dtype=torch.bool)
                        test_outputs[i][self.detector.target_box_key] = pb[keep_mask_tensor]
                        test_outputs[i][self.detector.pred_score_key] = ps[keep_mask_tensor]
                        test_outputs[i][self.detector.target_label_key] = pl[keep_mask_tensor]

                # 5. FPR 3D CNN 分類器過濾
                if bbox_filter:
                    for i in range(len(test_inputs)):
                        bbox_filter_stats["filtered"] += self._apply_bbox_aspect_filter_to_output(
                            test_outputs[i],
                            max_aspect_ratio=bbox_max_aspect_ratio,
                            skip_small_diam=bbox_aspect_skip_small_diam,
                        )

                batch_fpr_traces = [None] * len(test_inputs)
                if (fpr_model_path is not None and _fpr_model is not None) or fpr_mode == "learned":
                    for i in range(len(test_inputs)):
                        stats = self._fuse_fpr_scores(
                            output=test_outputs[i],
                            image_tensor=test_inputs[i],
                            fpr_model=_fpr_model,
                            fpr_fuser=_fpr_fuser,
                            fpr_patch_size=fpr_patch_size,
                            fpr_weight=fpr_weight,
                            fpr_thresh=fpr_thresh,
                            fpr_mode=fpr_mode,
                            fpr_score_aware=fpr_score_aware,
                            fpr_det_high_thresh=fpr_det_high_thresh,
                            fpr_det_mid_thresh=fpr_det_mid_thresh,
                            fpr_high_thresh=fpr_high_thresh,
                            fpr_mid_thresh=fpr_mid_thresh,
                            fpr_apply_min_diam=fpr_apply_min_diam,
                            fpr_apply_max_diam=fpr_apply_max_diam,
                            size_aware_small_diam=size_aware_small_diam,
                            size_aware_fpr_thresh=size_aware_fpr_thresh,
                        )
                        fpr_stats["rescored"] += stats["rescored"]
                        fpr_stats["filtered"] += stats["filtered"]
                        batch_fpr_traces[i] = stats.get("trace", [])

                test_outputs_all += test_outputs
                test_targets_all += test_data_batch
                
                if save_gifs:
                    # 逐一為該 batch (通常 batch=1) 的推論結果生成 GIF
                    for i in range(len(test_inputs)):
                        idx = len(test_outputs_all) - len(test_inputs) + i
                        img_np = test_inputs[i][0].cpu().numpy()  # 取第一 channel
                        pb = test_outputs[i][self.detector.target_box_key].cpu().numpy()
                        ps = test_outputs[i][self.detector.pred_score_key].cpu().numpy()
                        gt = test_data_batch[i][self.detector.target_box_key].cpu().numpy()
                        
                        # 盡量從源檔案取得名稱
                        src_path = test_data_batch[i].get("image_meta_dict", {}).get("filename_or_obj", f"test_scan_{idx}")
                        name = Path(src_path).stem[:35] if isinstance(src_path, str) else f"test_scan_{idx}"
                        gif_path = str(gif_out_dir / f"{name}.gif")
                        # 決定動畫中要畫出來的最低閾值，若使用者有設定 score_thresh 則以此為準，否則畫出 >= 0.1 的所有框
                        disp_thresh = final_score_thresh if final_score_thresh is not None else 0.1
                        create_prediction_gif(
                            img_np, pb, ps, gt,
                            output_path=gif_path,
                            score_thresh=disp_thresh,
                            fps=8
                        )

                if export_case_analysis and case_analysis_out_dir is not None:
                    from .case_analysis import export_case_analysis as export_case_analysis_fn
                    for i in range(len(test_inputs)):
                        idx = len(test_outputs_all) - len(test_inputs) + i
                        img_np = test_inputs[i][0].detach().cpu().numpy()
                        pb = test_outputs[i][self.detector.target_box_key].detach().cpu().numpy()
                        ps = test_outputs[i][self.detector.pred_score_key].detach().cpu().numpy()
                        gt = test_data_batch[i][self.detector.target_box_key].detach().cpu().numpy()
                        pb_export = pb
                        ps_export = ps
                        if final_score_thresh is not None and final_score_thresh > 0.0 and len(ps_export) > 0:
                            if (
                                size_aware_small_diam
                                and size_aware_small_diam > 0
                                and size_aware_final_thresh is not None
                            ):
                                boxes_for_export = [pb_export.copy()]
                                scores_for_export = [ps_export.copy()]
                                labels_for_export = [np.ones(len(ps_export), dtype=np.int64)]
                                self._apply_size_aware_score_threshold_to_arrays(
                                    boxes_for_export,
                                    scores_for_export,
                                    labels_for_export,
                                    score_thresh=float(final_score_thresh),
                                    small_diam=float(size_aware_small_diam),
                                    small_score_thresh=float(size_aware_final_thresh),
                                )
                                pb_export = boxes_for_export[0]
                                ps_export = scores_for_export[0]
                            else:
                                keep_export = ps_export >= float(final_score_thresh)
                                pb_export = pb_export[keep_export]
                                ps_export = ps_export[keep_export]
                        summary = export_case_analysis_fn(
                            image_yxz=img_np,
                            affine=test_affines[i],
                            pred_boxes=pb_export,
                            pred_scores=ps_export,
                            gt_boxes=gt,
                            output_root=case_analysis_out_dir,
                            source_image=test_source_paths[i],
                            fallback_case_id=f"{eval_split}_{idx:04d}",
                            iou_thresh=float(cfg.iou_list[0]),
                            score_thresh=0.0,
                            fpr_trace=batch_fpr_traces[i],
                        )
                        case_summaries.append(summary)

        elapsed = time.time() - start_time
        logger.info(f"  測試推論耗時: {elapsed:.1f}s")

        # 計算所有指標
        del test_inputs
        torch.cuda.empty_cache()

        pred_boxes_list = [o[self.detector.target_box_key].cpu().numpy() for o in test_outputs_all]
        pred_scores_list = [o[self.detector.pred_score_key].cpu().numpy() for o in test_outputs_all]
        pred_labels_list = [o[self.detector.target_label_key].cpu().numpy() for o in test_outputs_all]
        gt_boxes_list = [t[self.detector.target_box_key].cpu().numpy() for t in test_targets_all]
        gt_labels_list = [t[self.detector.target_label_key].cpu().numpy() for t in test_targets_all]

        import numpy as np
        
        # 1. 信心分數過濾 (Score Thresholding)
        if (
            final_score_thresh is not None
            and final_score_thresh > 0.0
            and size_aware_small_diam
            and size_aware_small_diam > 0
            and size_aware_final_thresh is not None
        ):
            logger.info(
                "  Score filtering: base=%.4f | small<=%.2f use %.4f",
                float(final_score_thresh),
                float(size_aware_small_diam),
                float(size_aware_final_thresh),
            )
            n_filtered_score = self._apply_size_aware_score_threshold_to_arrays(
                pred_boxes_list,
                pred_scores_list,
                pred_labels_list,
                score_thresh=float(final_score_thresh),
                small_diam=float(size_aware_small_diam),
                small_score_thresh=float(size_aware_final_thresh),
            )
            logger.info("    filtered %d boxes by size-aware score threshold", n_filtered_score)

        if (
            final_score_thresh is not None
            and final_score_thresh > 0.0
            and not (
                size_aware_small_diam
                and size_aware_small_diam > 0
                and size_aware_final_thresh is not None
            )
        ):
            logger.info(f"  🔪 啟用信心分數過濾 (Score Thresh = {final_score_thresh})...")
            n_filtered_score = 0
            for i in range(len(pred_scores_list)):
                scores = pred_scores_list[i]
                keep_mask = scores >= final_score_thresh
                n_filtered_score += np.sum(~keep_mask)
                
                pred_boxes_list[i] = pred_boxes_list[i][keep_mask]
                pred_scores_list[i] = scores[keep_mask]
                pred_labels_list[i] = pred_labels_list[i][keep_mask]
            logger.info(f"    過濾了 {n_filtered_score} 個低於 {final_score_thresh} 的低分預測框。")

        # (形態學過濾移至上方迴圈內，以取得原圖影像矩陣)

        from .metrics import compute_all_metrics
        test_results = compute_all_metrics(
            pred_boxes_list=pred_boxes_list,
            pred_scores_list=pred_scores_list,
            pred_labels_list=pred_labels_list,
            gt_boxes_list=gt_boxes_list,
            gt_labels_list=gt_labels_list,
            iou_thresh_match=cfg.iou_list[0],
            coco_metric=self.coco_metric,
            matching_batch_fn=M["matching_batch"],
            box_iou_fn=M["box_utils"].box_iou,
        )

        test_results["num_samples"] = len(test_ds)
        test_results["inference_time_s"] = elapsed
        test_results["postprocess"] = {
            "candidate_score_thresh": float(cfg.proposal_score_thresh),
            "score_thresh": final_score_thresh,
            "filter_lung_mask": bool(filter_lung_mask),
            "filter_fp": bool(filter_fp),
            "morphology": {
                "enabled": bool(filter_fp),
                "max_elongation": float(fp_max_elongation),
                "min_solidity": float(fp_min_solidity),
                "min_vol_mm3": float(fp_min_vol),
                "max_vol_mm3": float(fp_max_vol),
                "skip_small_diam_mm": float(morph_skip_small_diam),
                "three_plane": bool(morph_three_plane),
                "min_bad_planes": int(morph_min_bad_planes),
                "require_round_planes": bool(morph_require_round_planes),
                "min_round_planes": int(morph_min_round_planes),
                "max_round_elongation": float(morph_max_round_elongation),
                "min_plane_area_ratio": float(morph_min_plane_area_ratio),
                "axial_similarity": bool(morph_axial_similarity),
                "max_elongation_delta": float(morph_max_elongation_delta),
                "max_solidity_delta": float(morph_max_solidity_delta),
                "max_fill_delta": float(morph_max_fill_delta),
                "min_axial_area_ratio": float(morph_min_axial_area_ratio),
            },
            "bbox_filter": {
                "enabled": bool(bbox_filter),
                "max_aspect_ratio": float(bbox_max_aspect_ratio),
                "skip_small_diam_mm": float(bbox_aspect_skip_small_diam),
                "filtered_boxes": int(bbox_filter_stats["filtered"]),
            },
            "ensemble": {
                "enabled": bool(ensemble_enabled),
                "num_models": int(len(ensemble_paths)),
                "model_paths": [str(p) for p in ensemble_paths],
                "iou_thresh": float(ensemble_iou),
                "vote_power": float(ensemble_vote_power),
            },
            "fpr": {
                "enabled": bool(fpr_model_path),
                "model_path": fpr_model_path,
                "mode": fpr_mode,
                "threshold": fpr_thresh,
                "fuser_model_path": fpr_fuser_model_path,
                "score_aware": bool(fpr_score_aware),
                "det_high_thresh": float(fpr_det_high_thresh),
                "det_mid_thresh": float(fpr_det_mid_thresh),
                "high_thresh": float(fpr_high_thresh),
                "mid_thresh": float(fpr_mid_thresh),
                "apply_min_diam_mm": fpr_apply_min_diam,
                "apply_max_diam_mm": fpr_apply_max_diam,
                "patch_size": fpr_patch_size,
                "weight": fpr_weight,
                "rescored_boxes": fpr_stats["rescored"],
                "filtered_boxes": fpr_stats["filtered"],
            },
        }

        # 儲存測試結果（排除曲線大型陣列）
        test_results["postprocess"]["case_analysis"] = {
            "enabled": bool(export_case_analysis),
            "output_dir": str(case_analysis_out_dir) if case_analysis_out_dir is not None else None,
            "n_cases": int(len(case_summaries)),
        }
        if export_case_analysis and case_analysis_out_dir is not None:
            from .case_analysis import write_case_analysis_index
            write_case_analysis_index(case_analysis_out_dir, case_summaries)

        import json
        import numpy as np
        
        def default_serializer(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        save_results = {k: v for k, v in test_results.items() if k != "_curves"}
        test_path = self.output_dir / "test_metrics.json"
        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(save_results, f, indent=2, ensure_ascii=False, default=default_serializer)

        logger.info(
            f"🧪 測試集結果: mAP={test_results.get('mAP', 0):.4f} | "
            f"F1={test_results.get('detection_f1', 0):.4f} | "
            f"FROC={test_results.get('froc', {}).get('froc_score', 0):.4f} | "
            f"ROC-AUC={test_results.get('roc_auc', 0):.4f}"
        )
        logger.info(f"  📁 已儲存至: {test_path}")

    # ─── 檢查點 (Checkpoint) ───────────────────────────────────────
    def _save_model(self, path: Path):
        """使用 TorchScript 儲存模型。"""
        torch.jit.save(self.detector.network, str(path))

    def _save_training_state(self, path: Path, epoch: int, best_metric: float, best_epoch: int) -> None:
        """Save a resumable training-state checkpoint."""
        state: Dict[str, Any] = {
            "epoch": int(epoch),
            "best_metric": float(best_metric),
            "best_epoch": int(best_epoch),
            "model_state_dict": self.detector.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer is not None else None,
            "scheduler_state_dict": (
                self.scheduler.state_dict()
                if self.scheduler is not None and hasattr(self.scheduler, "state_dict")
                else None
            ),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler is not None else None,
            "history": self.history,
        }
        torch.save(state, str(path))

    def _load_training_state(self, path: Path) -> Tuple[int, float, int]:
        """Load a resumable training-state checkpoint."""
        # PyTorch >=2.6 defaults to weights_only=True, which cannot restore
        # optimizer/scheduler/scaler objects in a full training-state checkpoint.
        try:
            checkpoint = torch.load(str(path), map_location=self.device, weights_only=False)
        except TypeError:
            # Backward compatibility for older torch versions without weights_only arg.
            checkpoint = torch.load(str(path), map_location=self.device)

        model_state = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
        if model_state is None and isinstance(checkpoint, dict):
            is_plain_state_dict = (
                len(checkpoint) > 0
                and all(isinstance(k, str) and "." in k for k in checkpoint.keys())
                and any(torch.is_tensor(v) for v in checkpoint.values())
            )
            if is_plain_state_dict:
                model_state = checkpoint

        if model_state is None:
            raise RuntimeError(f"Invalid training-state checkpoint: {path}")

        load_result = self.detector.network.load_state_dict(model_state, strict=False)
        missing_keys = list(getattr(load_result, "missing_keys", []))
        unexpected_keys = list(getattr(load_result, "unexpected_keys", []))
        if missing_keys:
            logger.warning("Resume checkpoint missing %d model keys.", len(missing_keys))
        if unexpected_keys:
            logger.warning("Resume checkpoint has %d unexpected model keys.", len(unexpected_keys))

        if isinstance(checkpoint, dict):
            opt_state = checkpoint.get("optimizer_state_dict")
            if opt_state is not None:
                try:
                    self.optimizer.load_state_dict(opt_state)
                except Exception as exc:
                    logger.warning("Failed to load optimizer state. Using fresh optimizer. (%s)", exc)

            sch_state = checkpoint.get("scheduler_state_dict")
            if sch_state is not None and self.scheduler is not None and hasattr(self.scheduler, "load_state_dict"):
                try:
                    self.scheduler.load_state_dict(sch_state)
                except Exception as exc:
                    logger.warning("Failed to load scheduler state. Using fresh scheduler. (%s)", exc)

            scaler_state = checkpoint.get("scaler_state_dict")
            if scaler_state is not None and self.scaler is not None:
                try:
                    self.scaler.load_state_dict(scaler_state)
                except Exception as exc:
                    logger.warning("Failed to load AMP scaler state. (%s)", exc)

            history = checkpoint.get("history")
            if isinstance(history, dict):
                self.history = history

            last_epoch = int(checkpoint.get("epoch", -1))
            best_metric = float(checkpoint.get("best_metric", 0.0))
            best_epoch = int(checkpoint.get("best_epoch", -1))
        else:
            last_epoch = -1
            best_metric = 0.0
            best_epoch = -1

        start_epoch = max(last_epoch + 1, 0)
        return start_epoch, best_metric, best_epoch

    def load_checkpoint(self, path: str):
        """載入 TorchScript 模型檢查點。"""
        self.detector.network = torch.jit.load(path).to(self.device)
        logger.info(f"📂 已讀取檢查點: {path}")

    def _save_history(self):
        """儲存訓練歷程為 JSON。"""
        with open(self.output_dir / "history.json", "w", encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)

    def _plot_curves(self):
        """繪製訓練曲線與驗證 mAP。"""
        try:
            # 1. Loss 曲線
            plt.figure(figsize=(10, 6))
            epochs = range(1, len(self.history["train_loss"]) + 1)
            plt.plot(epochs, self.history["train_loss"], label="Total Loss")
            plt.plot(epochs, self.history["train_cls_loss"], label="Cls Loss")
            plt.plot(epochs, self.history["train_box_loss"], label="Box Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig(self.output_dir / "loss.png")
            plt.close()

            # 2. mAP 曲線
            if len(self.history["val_mAP"]) > 0:
                plt.figure(figsize=(10, 6))
                val_epochs = [
                    (i + 1) * self.config.val_interval
                    for i in range(len(self.history["val_mAP"]))
                ]
                plt.plot(val_epochs, self.history["val_mAP"], marker="o", label="mAP")
                plt.xlabel("Epoch")
                plt.ylabel("mAP")
                plt.title("Validation mAP")
                plt.legend()
                plt.grid(True)
                plt.savefig(self.output_dir / "val_map.png")
                plt.close()

            # 3. ROC Curve
            if len(self.history.get("roc_fpr", [])) > 0:
                plt.figure(figsize=(8, 8))
                plt.plot(
                    self.history["roc_fpr"], self.history["roc_tpr"],
                    color='darkorange', lw=2,
                    label=f'ROC (AUC = {self.history["roc_auc"]:.2f})',
                )
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc="lower right")
                plt.grid(True)
                plt.savefig(self.output_dir / "roc_curve.png")
                plt.close()

            # 4. PR Curve
            if len(self.history.get("pr_recall", [])) > 0:
                plt.figure(figsize=(8, 8))
                plt.plot(
                    self.history["pr_recall"], self.history["pr_precision"],
                    color='blue', lw=2,
                    label=f'PR (AP = {self.history["pr_ap"]:.2f})',
                )
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve')
                plt.legend(loc="lower left")
                plt.grid(True)
                plt.savefig(self.output_dir / "pr_curve.png")
                plt.close()

            # 5. F1 Curve
            if len(self.history.get("pr_recall", [])) > 0:
                p = np.array(self.history["pr_precision"])
                r = np.array(self.history["pr_recall"])
                f1 = 2 * (p * r) / (p + r + 1e-8)

                plt.figure(figsize=(10, 6))
                plt.plot(r, f1, color='green', lw=2, label=f'Max F1 = {np.max(f1):.4f}')
                plt.xlabel('Recall')
                plt.ylabel('F1 Score')
                plt.title('F1 Score vs Recall')
                plt.legend(loc="best")
                plt.grid(True)
                plt.savefig(self.output_dir / "f1_curve.png")
                plt.close()

        except Exception as e:
            logger.error(f"繪圖失敗: {e}")

    # ─── 推論 (Inference) ──────────────────────────────────────────
    def predict(self, input_path: str) -> List[Dict]:
        """
        對單一檔案執行偵測。
        使用與驗證相同的 Transform Pipeline 載入影像。
        """
        self.detector.eval()
        cfg = self.config

        # 使用 val_transform 載入影像
        val_transform = build_val_transform(
            spacing=cfg.spacing,
            hu_min=cfg.hu_min,
            hu_max=cfg.hu_max,
        )

        # 建立 dummy data dict (只有 image，無 box/label)
        # 需要提供空 box/label 以通過 transform
        data = {
            "image": input_path,
            "box": np.zeros((0, 6), dtype=np.float32).tolist(),
            "label": np.zeros((0,), dtype=np.int64).tolist(),
        }
        processed = val_transform(data)
        image_tensor = processed["image"].to(self.device)

        with torch.no_grad():
            if cfg.amp:
                with torch.amp.autocast("cuda"):
                    outputs = self.detector([image_tensor], use_inferer=True)
            else:
                outputs = self.detector([image_tensor], use_inferer=True)

        results = []
        for output in outputs:
            results.append({
                "boxes": output[self.detector.target_box_key].cpu().numpy(),
                "labels": output[self.detector.target_label_key].cpu().numpy(),
                "scores": output[self.detector.pred_score_key].cpu().numpy(),
            })
        return results
