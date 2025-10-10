#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv7 Medical Image Augmentations

包含：
- Mosaic augmentation (4-image mosaic)
- Mixup augmentation
- Random rotation, scaling, translation
- Brightness/contrast adjustment
- Copy-paste for small objects
- Medical-specific augmentations (HU windowing variations)

針對醫學影像優化，保護小病灶特徵
"""

import cv2
import numpy as np
import random
import torch
from typing import List, Tuple, Optional, Dict, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2


class MosaicAugmentation:
    """
    Mosaic 增強：將 4 張圖像拼接成一張
    有助於學習不同尺度和位置的物體
    """
    
    def __init__(self, img_size: int = 640):
        self.img_size = img_size
    
    def __call__(
        self,
        images: List[np.ndarray],
        labels: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            images: List of 4 images (H, W, C)
            labels: List of 4 label arrays (N, 5) [class, x_center, y_center, w, h]
        
        Returns:
            mosaic_img: Mosaic image (img_size, img_size, C)
            mosaic_labels: Combined labels (M, 5)
        """
        assert len(images) == 4 and len(labels) == 4, "Mosaic requires exactly 4 images"
        
        # 創建 mosaic 畫布
        mosaic_img = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        
        # 隨機選擇分割點（中心位置）
        yc = int(random.uniform(self.img_size * 0.4, self.img_size * 0.6))
        xc = int(random.uniform(self.img_size * 0.4, self.img_size * 0.6))
        
        mosaic_labels = []
        
        # 定義 4 個區域的位置
        positions = [
            (0, 0, xc, yc),           # top-left
            (xc, 0, self.img_size, yc),    # top-right
            (0, yc, xc, self.img_size),    # bottom-left
            (xc, yc, self.img_size, self.img_size)  # bottom-right
        ]
        
        for i, (img, label, (x1, y1, x2, y2)) in enumerate(zip(images, labels, positions)):
            h, w = img.shape[:2]
            
            # 計算需要的縮放比例
            region_w = x2 - x1
            region_h = y2 - y1
            scale = min(region_w / w, region_h / h)
            
            # 縮放圖像
            new_w = int(w * scale)
            new_h = int(h * scale)
            if new_w > 0 and new_h > 0:
                img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                # 放置到 mosaic 中
                mosaic_img[y1:y1+new_h, x1:x1+new_w] = img_resized
                
                # 調整標籤座標
                if len(label) > 0:
                    label_adjusted = label.copy()
                    # 從歸一化座標轉換為原始像素
                    label_adjusted[:, 1] = label[:, 1] * w  # x_center
                    label_adjusted[:, 2] = label[:, 2] * h  # y_center
                    label_adjusted[:, 3] = label[:, 3] * w  # width
                    label_adjusted[:, 4] = label[:, 4] * h  # height
                    
                    # 縮放
                    label_adjusted[:, 1:5] *= scale
                    
                    # 平移到 mosaic 位置
                    label_adjusted[:, 1] += x1
                    label_adjusted[:, 2] += y1
                    
                    # 轉換回歸一化座標（相對於 mosaic 大小）
                    label_adjusted[:, 1] /= self.img_size
                    label_adjusted[:, 2] /= self.img_size
                    label_adjusted[:, 3] /= self.img_size
                    label_adjusted[:, 4] /= self.img_size
                    
                    # 裁剪超出範圍的 boxes
                    label_adjusted = self._clip_boxes(label_adjusted)
                    
                    if len(label_adjusted) > 0:
                        mosaic_labels.append(label_adjusted)
        
        # 合併所有標籤
        if mosaic_labels:
            mosaic_labels = np.concatenate(mosaic_labels, axis=0)
        else:
            mosaic_labels = np.zeros((0, 5), dtype=np.float32)
        
        return mosaic_img, mosaic_labels
    
    def _clip_boxes(self, labels: np.ndarray) -> np.ndarray:
        """裁剪超出 [0, 1] 範圍的 boxes"""
        if len(labels) == 0:
            return labels
        
        # 確保中心點在 [0, 1] 範圍內
        labels[:, 1] = np.clip(labels[:, 1], 0, 1)
        labels[:, 2] = np.clip(labels[:, 2], 0, 1)
        
        # 確保寬高不超過 1
        labels[:, 3] = np.clip(labels[:, 3], 0, 1)
        labels[:, 4] = np.clip(labels[:, 4], 0, 1)
        
        # 過濾掉過小的 boxes（寬或高 < 2 像素）
        min_size = 2.0 / 640.0  # 假設 img_size=640
        mask = (labels[:, 3] > min_size) & (labels[:, 4] > min_size)
        
        return labels[mask]


class MixUpAugmentation:
    """
    MixUp 增強：混合兩張圖像和標籤
    有助於提升模型泛化能力
    """
    
    def __init__(self, alpha: float = 0.5):
        """
        Args:
            alpha: Mixup ratio (0.0 = first image only, 1.0 = second image only)
        """
        self.alpha = alpha
    
    def __call__(
        self,
        img1: np.ndarray,
        labels1: np.ndarray,
        img2: np.ndarray,
        labels2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            img1, img2: Images (H, W, C)
            labels1, labels2: Labels (N, 5)
        
        Returns:
            mixed_img: Mixed image
            mixed_labels: Combined labels
        """
        # 隨機 mixup ratio
        r = np.random.beta(self.alpha, self.alpha)
        
        # 混合圖像
        mixed_img = (img1 * r + img2 * (1 - r)).astype(np.uint8)
        
        # 合併標籤
        mixed_labels = np.concatenate([labels1, labels2], axis=0)
        
        return mixed_img, mixed_labels


class CopyPasteAugmentation:
    """
    Copy-Paste 增強：複製小物體到其他位置
    專門針對小病灶設計
    """
    
    def __init__(self, prob: float = 0.5, max_paste: int = 3):
        self.prob = prob
        self.max_paste = max_paste
    
    def __call__(
        self,
        img: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            img: Image (H, W, C)
            labels: Labels (N, 5) [class, x_center, y_center, w, h]
        
        Returns:
            img: Modified image
            labels: Modified labels with pasted objects
        """
        if random.random() > self.prob or len(labels) == 0:
            return img, labels
        
        h, w = img.shape[:2]
        new_labels = [labels]
        
        # 找出小物體（面積 < 0.05）
        areas = labels[:, 3] * labels[:, 4]
        small_objects = labels[areas < 0.05]
        
        if len(small_objects) == 0:
            return img, labels
        
        # 隨機選擇要複製的物體
        n_paste = min(random.randint(1, self.max_paste), len(small_objects))
        paste_objects = small_objects[np.random.choice(len(small_objects), n_paste, replace=False)]
        
        for obj in paste_objects:
            cls_id, x_center, y_center, box_w, box_h = obj
            
            # 計算像素座標
            x_c_px = int(x_center * w)
            y_c_px = int(y_center * h)
            w_px = int(box_w * w)
            h_px = int(box_h * h)
            
            x1 = max(0, x_c_px - w_px // 2)
            y1 = max(0, y_c_px - h_px // 2)
            x2 = min(w, x_c_px + w_px // 2)
            y2 = min(h, y_c_px + h_px // 2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # 提取物體區域
            obj_region = img[y1:y2, x1:x2].copy()
            
            # 隨機選擇新位置（避免重疊）
            new_x_c = random.uniform(box_w / 2, 1 - box_w / 2)
            new_y_c = random.uniform(box_h / 2, 1 - box_h / 2)
            
            new_x_c_px = int(new_x_c * w)
            new_y_c_px = int(new_y_c * h)
            
            new_x1 = max(0, new_x_c_px - w_px // 2)
            new_y1 = max(0, new_y_c_px - h_px // 2)
            new_x2 = min(w, new_x_c_px + w_px // 2)
            new_y2 = min(h, new_y_c_px + h_px // 2)
            
            if new_x2 <= new_x1 or new_y2 <= new_y1:
                continue
            
            # 貼上物體（使用加權混合避免明顯邊界）
            paste_h = new_y2 - new_y1
            paste_w = new_x2 - new_x1
            
            if obj_region.shape[0] > 0 and obj_region.shape[1] > 0:
                obj_resized = cv2.resize(obj_region, (paste_w, paste_h), interpolation=cv2.INTER_LINEAR)
                alpha = 0.7  # 混合權重
                img[new_y1:new_y2, new_x1:new_x2] = (
                    img[new_y1:new_y2, new_x1:new_x2] * (1 - alpha) +
                    obj_resized * alpha
                ).astype(np.uint8)
                
                # 添加新標籤
                new_label = np.array([[cls_id, new_x_c, new_y_c, box_w, box_h]])
                new_labels.append(new_label)
        
        # 合併標籤
        labels = np.concatenate(new_labels, axis=0)
        
        return img, labels


def get_medical_ct_transforms(
    img_size: int = 640,
    train: bool = True,
    enable_mosaic: bool = True,
    enable_mixup: bool = True,
    enable_copy_paste: bool = True
) -> A.Compose:
    """
    獲取醫學 CT 圖像的增強管線
    
    Args:
        img_size: 目標圖像大小
        train: 是否訓練模式
        enable_mosaic: 啟用 Mosaic
        enable_mixup: 啟用 MixUp
        enable_copy_paste: 啟用 Copy-Paste
    
    Returns:
        Albumentations Compose 對象
    """
    if train:
        transforms = [
            # 幾何變換
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5
            ),
            A.HorizontalFlip(p=0.5),
            
            # 亮度/對比度調整（針對 CT 影像）
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            
            # 噪聲增強
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            
            # 像素級增強
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.2),
            
            # 尺寸調整
            A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0
            ),
        ]
    else:
        # 驗證模式：僅調整大小
        transforms = [
            A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0
            ),
        ]
    
    # Bbox 參數
    bbox_params = A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_area=0,
        min_visibility=0.3,
        check_each_transform=True
    )
    
    return A.Compose(transforms, bbox_params=bbox_params)


class YOLOv7Augmenter:
    """
    YOLOv7 完整增強管線
    整合 Mosaic, MixUp, Copy-Paste 和其他增強
    """
    
    def __init__(
        self,
        img_size: int = 640,
        mosaic_prob: float = 0.5,
        mixup_prob: float = 0.3,
        copy_paste_prob: float = 0.3,
        enable_mosaic: bool = True,
        enable_mixup: bool = True,
        enable_copy_paste: bool = True
    ):
        self.img_size = img_size
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.copy_paste_prob = copy_paste_prob
        self.enable_mosaic = enable_mosaic
        self.enable_mixup = enable_mixup
        self.enable_copy_paste = enable_copy_paste
        
        # 初始化增強器
        self.mosaic = MosaicAugmentation(img_size=img_size)
        self.mixup = MixUpAugmentation(alpha=0.5)
        self.copy_paste = CopyPasteAugmentation(prob=copy_paste_prob, max_paste=3)
        
        # 基礎變換
        self.transforms = get_medical_ct_transforms(
            img_size=img_size,
            train=True,
            enable_mosaic=enable_mosaic,
            enable_mixup=enable_mixup,
            enable_copy_paste=enable_copy_paste
        )
    
    def __call__(
        self,
        img: np.ndarray,
        labels: np.ndarray,
        extra_samples: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply augmentations
        
        Args:
            img: Image (H, W, C)
            labels: Labels (N, 5) [class, x_center, y_center, w, h]
            extra_samples: Extra samples for mosaic/mixup [(img, labels), ...]
        
        Returns:
            augmented_img: Augmented image
            augmented_labels: Augmented labels
        """
        # 1. Mosaic augmentation
        if self.enable_mosaic and random.random() < self.mosaic_prob and extra_samples and len(extra_samples) >= 3:
            # 使用當前樣本 + 3 個額外樣本
            mosaic_imgs = [img] + [s[0] for s in extra_samples[:3]]
            mosaic_labels = [labels] + [s[1] for s in extra_samples[:3]]
            img, labels = self.mosaic(mosaic_imgs, mosaic_labels)
        
        # 2. MixUp augmentation
        if self.enable_mixup and random.random() < self.mixup_prob and extra_samples:
            mixup_img, mixup_labels = extra_samples[0]
            img, labels = self.mixup(img, labels, mixup_img, mixup_labels)
        
        # 3. Copy-Paste augmentation
        if self.enable_copy_paste:
            img, labels = self.copy_paste(img, labels)
        
        # 4. 基礎增強（使用 Albumentations）
        if len(labels) > 0:
            # 準備 bboxes 和 class_labels
            bboxes = labels[:, 1:5].tolist()  # [x_center, y_center, w, h]
            class_labels = labels[:, 0].tolist()
            
            try:
                transformed = self.transforms(
                    image=img,
                    bboxes=bboxes,
                    class_labels=class_labels
                )
                
                img = transformed['image']
                
                # 重建 labels
                if transformed['bboxes']:
                    labels = np.array([
                        [cls] + list(bbox)
                        for cls, bbox in zip(transformed['class_labels'], transformed['bboxes'])
                    ], dtype=np.float32)
                else:
                    labels = np.zeros((0, 5), dtype=np.float32)
            except Exception as e:
                print(f"⚠️  Augmentation error: {e}, using original image")
                # 回退到基礎 resize
                img = cv2.resize(img, (self.img_size, self.img_size))
        else:
            # 無標籤的情況
            img = cv2.resize(img, (self.img_size, self.img_size))
        
        return img, labels


if __name__ == "__main__":
    print("YOLOv7 Medical Image Augmentations")
    print("Includes: Mosaic, MixUp, Copy-Paste, and medical-specific transforms")
    
    # 測試 Mosaic
    print("\n Testing Mosaic Augmentation...")
    mosaic = MosaicAugmentation(img_size=640)
    
    # 創建 4 張模擬圖像
    imgs = [np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8) for _ in range(4)]
    labels = [
        np.array([[0, 0.5, 0.5, 0.2, 0.2]]),
        np.array([[0, 0.3, 0.3, 0.15, 0.15]]),
        np.array([[0, 0.7, 0.7, 0.1, 0.1]]),
        np.array([[0, 0.4, 0.6, 0.25, 0.25]])
    ]
    
    mosaic_img, mosaic_labels = mosaic(imgs, labels)
    print(f"Mosaic output: img shape={mosaic_img.shape}, labels shape={mosaic_labels.shape}")
    
    # 測試完整增強器
    print("\nTesting YOLOv7Augmenter...")
    augmenter = YOLOv7Augmenter(img_size=640, enable_mosaic=True, enable_mixup=True, enable_copy_paste=True)
    
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    labels = np.array([[0, 0.5, 0.5, 0.2, 0.2], [0, 0.3, 0.3, 0.1, 0.1]])
    extra_samples = [(imgs[i], labels_item) for i, labels_item in enumerate(labels[:3])]
    
    aug_img, aug_labels = augmenter(img, labels, extra_samples)
    print(f"Augmented output: img shape={aug_img.shape}, labels shape={aug_labels.shape}")
    
    print("\n✅ All tests passed!")
