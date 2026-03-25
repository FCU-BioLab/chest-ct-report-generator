#!/usr/bin/env python3
"""
3D Volumetric Dataset
=====================

 NIfTI-only dataset loader for task folders:
- imagesTr/*_0000.nii.gz
- labelsTr/*.nii.gz
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    import SimpleITK as sitk
except ModuleNotFoundError:  # pragma: no cover - runtime guard
    sitk = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class VolumetricDataset(Dataset):
    """
    Volumetric lesion dataset for 3D U-Net.
    Reads imagesTr/*_0000.nii.gz + labelsTr/*.nii.gz from NIfTI task folders.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        split: str = "train",
        image_size: int = 256,
        max_depth: int = 32,
        augmentation: bool = False,
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        split_seed: int = 42,
        positive_ratio: float = 0.7,
        full_volume: bool = False,
    ):
        if data_dir is None:
            raise ValueError("`data_dir` is required (NIfTI task folder).")

        self.root_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.max_depth = max_depth
        self.augmentation = augmentation
        self.split_ratios = split_ratios
        self.split_seed = split_seed
        self.positive_ratio = positive_ratio
        self.full_volume = full_volume

        if self.split in {"val", "test"} and (not self.full_volume):
            logger.warning(
                "Split=%s is using depth cropping (max_depth=%d). "
                "For final evaluation, prefer full-volume inference.",
                self.split,
                self.max_depth,
            )

        if not (self.root_dir / "imagesTr").exists():
            raise ValueError(
                f"NIfTI task folder not found: {self.root_dir}. "
                "Expected imagesTr/labelsTr layout."
            )
        self.samples = self._load_file_list()
        logger.info(
            "Loading %d samples (split=%s, mode=nifti_task, root=%s)",
            len(self.samples),
            split,
            self.root_dir,
        )

    def _load_file_list(self) -> List[Dict[str, Any]]:
        return self._load_nifti_task_samples()

    def _load_nifti_task_samples(self) -> List[Dict[str, Any]]:
        images_tr = self.root_dir / "imagesTr"
        labels_tr = self.root_dir / "labelsTr"
        if not images_tr.exists():
            logger.warning("imagesTr not found in %s", self.root_dir)
            return []

        entries: List[Dict[str, Any]] = []
        for img_path in sorted(images_tr.glob("*_0000.nii.gz")):
            case_id = img_path.name.replace("_0000.nii.gz", "")
            label_path = labels_tr / f"{case_id}.nii.gz"
            entries.append(
                {
                    "type": "nifti",
                    "image_path": img_path,
                    "label_path": label_path if label_path.exists() else None,
                    "patient_id": case_id,
                    "lesion_id": 0,
                }
            )

        if not entries:
            logger.warning("No *_0000.nii.gz files found under %s", images_tr)
            return []

        # Patient-level split to avoid leakage
        patient_ids = sorted({e["patient_id"] for e in entries})
        rng = np.random.RandomState(self.split_seed)
        rng.shuffle(patient_ids)

        n = len(patient_ids)
        train_end = int(n * self.split_ratios[0])
        val_end = train_end + int(n * self.split_ratios[1])
        if self.split == "train":
            selected = set(patient_ids[:train_end])
        elif self.split == "val":
            selected = set(patient_ids[train_end:val_end])
        elif self.split == "test":
            selected = set(patient_ids[val_end:])
        else:
            selected = set(patient_ids)

        filtered = [e for e in entries if e["patient_id"] in selected]
        logger.info(
            "NIfTI split (%s): %d patients, %d samples",
            self.split,
            len(selected),
            len(filtered),
        )
        return filtered

    def __len__(self) -> int:
        return len(self.samples)

    def get_patient_ids(self) -> List[str]:
        return [str(s["patient_id"]) for s in self.samples]

    def get_dataset_info(self) -> Dict[str, Any]:
        patient_ids = self.get_patient_ids()
        unique_patients = sorted(set(patient_ids))
        sample_paths = [str(Path(s["image_path"]).name) for s in self.samples]
        return {
            "split": self.split,
            "mode": "nifti_task",
            "total_samples": len(self.samples),
            "unique_patients": len(unique_patients),
            "patient_ids": unique_patients,
            "sample_paths": sample_paths,
        }

    def _prepare_crop_range(self, depth: int, center_idx: int) -> Tuple[int, int]:
        start, end = 0, depth
        use_random_crop = False
        if self.split == "train" and depth > self.max_depth:
            if np.random.rand() > self.positive_ratio:
                use_random_crop = True

        if not self.full_volume:
            if use_random_crop:
                max_start = max(0, depth - self.max_depth)
                start = np.random.randint(0, max_start + 1)
                end = min(depth, start + self.max_depth)
            elif depth > self.max_depth:
                half = self.max_depth // 2
                start = max(0, center_idx - half)
                end = min(depth, start + self.max_depth)
                if end - start < self.max_depth:
                    if start == 0:
                        end = min(depth, self.max_depth)
                    elif end == depth:
                        start = max(0, depth - self.max_depth)
        return start, end

    @staticmethod
    def _normalize_ct_to_u8(image_zyx: np.ndarray) -> np.ndarray:
        # HU window [-1000, 400] -> [0, 255]
        clipped = np.clip(image_zyx.astype(np.float32), -1000.0, 400.0)
        norm = (clipped + 1000.0) / 1400.0
        return (norm * 255.0).astype(np.uint8)

    def _load_nifti_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if sitk is None:
            raise RuntimeError("SimpleITK is required for NIfTI loading.")

        img_path: Path = sample["image_path"]
        label_path: Optional[Path] = sample["label_path"]

        image = sitk.ReadImage(str(img_path))
        frames = sitk.GetArrayFromImage(image)  # zyx
        frames = self._normalize_ct_to_u8(frames)

        if label_path is not None and label_path.exists():
            label_img = sitk.ReadImage(str(label_path))
            masks = (sitk.GetArrayFromImage(label_img) > 0).astype(np.uint8)
        else:
            masks = np.zeros_like(frames, dtype=np.uint8)

        has_lesion = masks.sum() > 0
        lesion_center_idx = int(np.argmax(masks.sum(axis=(1, 2)))) if has_lesion else int(frames.shape[0] // 2)
        lesion_id = 1 if has_lesion else 0

        # Avoid label leakage:
        # - train: allow lesion-centered sampling for positive mining
        # - val/test: never use GT center for crop selection
        crop_center_idx = lesion_center_idx if self.split == "train" else int(frames.shape[0] // 2)

        slice_indices = np.arange(frames.shape[0], dtype=np.int64)
        spacing = np.array(image.GetSpacing(), dtype=np.float32)
        origin = np.array(image.GetOrigin(), dtype=np.float32)
        original_shape = np.array(frames.shape, dtype=np.int64)

        start, end = self._prepare_crop_range(frames.shape[0], crop_center_idx)
        frames = frames[start:end]
        masks = masks[start:end]
        slice_indices = slice_indices[start:end]
        lesion_center_idx = int(np.clip(lesion_center_idx - start, 0, max(0, frames.shape[0] - 1)))

        frames_t = torch.from_numpy(frames).float().unsqueeze(0) / 255.0
        masks_t = torch.from_numpy(masks).float().unsqueeze(0)
        frames_t, masks_t = self._resize_video_torch(frames_t, masks_t)

        if self.augmentation and self.split == "train":
            frames_t, masks_t = self._augment_torch(frames_t, masks_t)

        return {
            "image": frames_t,
            "mask": masks_t.long(),
            "patient_id": str(sample["patient_id"]),
            "lesion_id": lesion_id,
            "source_path": str(img_path),
            "slice_indices": torch.tensor(slice_indices, dtype=torch.long),
            "spacing": torch.tensor(spacing, dtype=torch.float32),
            "origin": torch.tensor(origin, dtype=torch.float32),
            "original_shape": torch.tensor(original_shape, dtype=torch.long),
            "lesion_center_idx": lesion_center_idx,
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._load_nifti_sample(self.samples[idx])

    def _resize_video_torch(self, frames: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        d, h, w = frames.shape[1], frames.shape[2], frames.shape[3]
        if h == self.image_size and w == self.image_size:
            return frames, masks

        f_in = frames.permute(1, 0, 2, 3)  # (D,1,H,W)
        m_in = masks.permute(1, 0, 2, 3)  # (D,1,H,W)
        f_out = F.interpolate(f_in, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        m_out = F.interpolate(m_in, size=(self.image_size, self.image_size), mode="nearest")
        return f_out.permute(1, 0, 2, 3), m_out.permute(1, 0, 2, 3)

    def _augment_torch(self, frames: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if np.random.rand() > 0.5:
            frames = torch.flip(frames, dims=[3])
            masks = torch.flip(masks, dims=[3])
        if np.random.rand() > 0.5:
            frames = torch.flip(frames, dims=[2])
            masks = torch.flip(masks, dims=[2])
        if np.random.rand() > 0.5:
            shift = (np.random.rand() * 0.2 - 0.1)
            frames = torch.clamp(frames + shift, 0.0, 1.0)
        if np.random.rand() > 0.5:
            k = int(np.random.randint(1, 4))
            frames = torch.rot90(frames, k, dims=[2, 3])
            masks = torch.rot90(masks, k, dims=[2, 3])
        return frames, masks


def collate_video_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate batch with variable depth D by end-padding to max depth.
    """
    if not batch:
        return {}

    max_d = max(s["image"].shape[1] for s in batch)
    images: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    slice_indices_list: List[torch.Tensor] = []
    has_slice_indices = "slice_indices" in batch[0]

    spacings: List[torch.Tensor] = []
    origins: List[torch.Tensor] = []
    original_shapes: List[torch.Tensor] = []

    for s in batch:
        img = s["image"]  # (1,D,H,W)
        msk = s["mask"]  # (1,D,H,W)
        d = img.shape[1]

        if d < max_d:
            pad_d = max_d - d
            img = torch.nn.functional.pad(img, (0, 0, 0, 0, 0, pad_d))
            msk = torch.nn.functional.pad(msk, (0, 0, 0, 0, 0, pad_d))

            if has_slice_indices and "slice_indices" in s:
                si = s["slice_indices"]
                si = torch.cat([si, torch.full((pad_d,), -1, dtype=si.dtype)])
                slice_indices_list.append(si)
        else:
            if has_slice_indices and "slice_indices" in s:
                slice_indices_list.append(s["slice_indices"])

        images.append(img)
        masks.append(msk)

        if "spacing" in s:
            spacings.append(s["spacing"])
        if "origin" in s:
            origins.append(s["origin"])
        if "original_shape" in s:
            original_shapes.append(s["original_shape"])

    result: Dict[str, Any] = {
        "image": torch.stack(images),  # (B,1,D,H,W)
        "mask": torch.stack(masks),  # (B,1,D,H,W)
        "patient_id": [s["patient_id"] for s in batch],
        "source_path": [s["source_path"] for s in batch],
    }
    if slice_indices_list:
        result["slice_indices"] = torch.stack(slice_indices_list)
    if spacings:
        result["spacing"] = torch.stack(spacings)
    if origins:
        result["origin"] = torch.stack(origins)
    if original_shapes:
        result["original_shape"] = torch.stack(original_shapes)
    return result
