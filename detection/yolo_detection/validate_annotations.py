#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv7 Annotation Validation and Visualization Tool

功能：
1. 隨機抽取訓練/驗證集影像
2. 將 YOLO 格式標註繪製在 CT slice 上
3. 檢查標註錯誤（超出範圍、過小、重疊等）
4. 輸出 grid 可視化圖與詳細報告

Usage:
    python validate_annotations.py --data_root ../../datasets/splited_dataset --num_samples 100
"""

import sys
import os
import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from detection_dataset import CTDetectionDataset
    DATASET_AVAILABLE = True
except ImportError:
    DATASET_AVAILABLE = False


@dataclass
class BBoxError:
    """Bounding box 錯誤記錄"""
    image_path: str
    bbox_idx: int
    error_type: str  # 'out_of_bounds', 'too_small', 'invalid_format', 'negative_coords'
    bbox: List[float]  # [x_center, y_center, w, h]
    details: str


@dataclass
class ValidationReport:
    """驗證報告"""
    total_images: int
    total_boxes: int
    error_boxes: int
    error_rate: float
    errors_by_type: Dict[str, int]
    too_small_boxes: int
    out_of_bounds_boxes: int
    avg_box_width: float
    avg_box_height: float
    min_box_size: float
    errors: List[BBoxError]


class AnnotationValidator:
    """標註驗證器"""
    
    def __init__(
        self,
        data_root: str,
        output_dir: str = "./yolov7_logs/annotation_validation",
        min_box_size_px: int = 3,
        img_size: int = 640
    ):
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_box_size_px = min_box_size_px
        self.img_size = img_size
        
        # 統計數據
        self.errors: List[BBoxError] = []
        self.box_sizes: List[float] = []
        self.box_widths: List[float] = []
        self.box_heights: List[float] = []
    
    def load_yolo_annotation(self, label_path: Path) -> List[List[float]]:
        """
        讀取 YOLO 格式標註
        
        Returns:
            List of [class_id, x_center, y_center, w, h]
        """
        if not label_path.exists():
            return []
        
        boxes = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    try:
                        box = [float(x) for x in parts]
                        boxes.append(box)
                    except ValueError:
                        continue
        return boxes
    
    def validate_bbox(
        self,
        bbox: List[float],
        image_path: str,
        bbox_idx: int,
        img_width: int,
        img_height: int
    ) -> Optional[BBoxError]:
        """
        驗證單個 bbox
        
        Args:
            bbox: [class_id, x_center, y_center, w, h] (normalized 0-1)
            image_path: 圖像路徑
            bbox_idx: bbox 索引
            img_width: 圖像寬度
            img_height: 圖像高度
        
        Returns:
            BBoxError if error found, else None
        """
        if len(bbox) != 5:
            return BBoxError(
                image_path=image_path,
                bbox_idx=bbox_idx,
                error_type='invalid_format',
                bbox=bbox,
                details=f"Invalid format: expected 5 values, got {len(bbox)}"
            )
        
        cls_id, x_center, y_center, w, h = bbox
        
        # 檢查是否為負數
        if any(v < 0 for v in [x_center, y_center, w, h]):
            return BBoxError(
                image_path=image_path,
                bbox_idx=bbox_idx,
                error_type='negative_coords',
                bbox=bbox,
                details=f"Negative coordinates: {bbox}"
            )
        
        # 檢查是否超出 [0, 1] 範圍
        if x_center > 1.0 or y_center > 1.0 or w > 1.0 or h > 1.0:
            return BBoxError(
                image_path=image_path,
                bbox_idx=bbox_idx,
                error_type='out_of_bounds',
                bbox=bbox,
                details=f"Coordinates > 1.0: x={x_center:.3f}, y={y_center:.3f}, w={w:.3f}, h={h:.3f}"
            )
        
        # 檢查框是否過小（轉換為像素）
        w_px = w * img_width
        h_px = h * img_height
        
        if w_px < self.min_box_size_px or h_px < self.min_box_size_px:
            return BBoxError(
                image_path=image_path,
                bbox_idx=bbox_idx,
                error_type='too_small',
                bbox=bbox,
                details=f"Box too small: {w_px:.1f}x{h_px:.1f} px (min: {self.min_box_size_px}px)"
            )
        
        # 統計 box 大小
        self.box_widths.append(w_px)
        self.box_heights.append(h_px)
        self.box_sizes.append(min(w_px, h_px))
        
        return None
    
    def draw_boxes_on_image(
        self,
        image: np.ndarray,
        boxes: List[List[float]],
        errors: List[Optional[BBoxError]]
    ) -> np.ndarray:
        """
        在圖像上繪製 bounding boxes
        
        Args:
            image: 圖像 (H, W, 3)
            boxes: YOLO 格式 boxes
            errors: 對應的錯誤列表
        
        Returns:
            繪製後的圖像
        """
        img_drawn = image.copy()
        h, w = img_drawn.shape[:2]
        
        for i, (box, error) in enumerate(zip(boxes, errors)):
            if len(box) != 5:
                continue
            
            cls_id, x_center, y_center, box_w, box_h = box
            
            # 轉換為像素座標
            x_center_px = int(x_center * w)
            y_center_px = int(y_center * h)
            w_px = int(box_w * w)
            h_px = int(box_h * h)
            
            x1 = int(x_center_px - w_px / 2)
            y1 = int(y_center_px - h_px / 2)
            x2 = int(x_center_px + w_px / 2)
            y2 = int(y_center_px + h_px / 2)
            
            # 根據是否有錯誤選擇顏色
            if error:
                color = (0, 0, 255)  # 紅色 (錯誤)
                thickness = 3
            else:
                color = (0, 255, 0)  # 綠色 (正常)
                thickness = 2
            
            # 繪製矩形
            cv2.rectangle(img_drawn, (x1, y1), (x2, y2), color, thickness)
            
            # 繪製中心點
            cv2.circle(img_drawn, (x_center_px, y_center_px), 3, color, -1)
            
            # 標註 box 索引
            label = f"{i}"
            if error:
                label += f" [{error.error_type}]"
            
            cv2.putText(
                img_drawn,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )
        
        return img_drawn
    
    def create_grid_visualization(
        self,
        images_with_boxes: List[Tuple[np.ndarray, str]],
        grid_size: Tuple[int, int] = (5, 5),
        output_path: Optional[Path] = None
    ):
        """
        創建 grid 可視化（最多 25 張圖）
        
        Args:
            images_with_boxes: List of (image, title)
            grid_size: Grid 大小 (rows, cols)
            output_path: 輸出路徑
        """
        rows, cols = grid_size
        max_images = rows * cols
        images_with_boxes = images_with_boxes[:max_images]
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        
        for i, (img, title) in enumerate(images_with_boxes):
            if i >= len(axes):
                break
            
            # 轉換 BGR -> RGB
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            axes[i].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
            axes[i].set_title(title, fontsize=8)
            axes[i].axis('off')
        
        # 隱藏多餘的 subplot
        for i in range(len(images_with_boxes), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Grid visualization saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def validate_dataset_samples(
        self,
        sample_paths: List[Tuple[Path, Path]],
        visualize: bool = True
    ) -> ValidationReport:
        """
        驗證數據集樣本
        
        Args:
            sample_paths: List of (image_path, label_path)
            visualize: 是否生成可視化
        
        Returns:
            ValidationReport
        """
        print(f"\n{'='*80}")
        print(f"Validating {len(sample_paths)} samples...")
        print(f"{'='*80}\n")
        
        total_boxes = 0
        error_boxes = 0
        images_for_viz = []
        
        for img_path, label_path in tqdm(sample_paths, desc="Validating"):
            # 讀取圖像
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            # 轉換為 3 通道用於可視化
            img_viz = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            h, w = img.shape
            
            # 讀取標註
            boxes = self.load_yolo_annotation(label_path)
            total_boxes += len(boxes)
            
            # 驗證每個 box
            box_errors = []
            for i, box in enumerate(boxes):
                error = self.validate_bbox(box, str(img_path), i, w, h)
                box_errors.append(error)
                
                if error:
                    self.errors.append(error)
                    error_boxes += 1
            
            # 繪製 boxes
            img_drawn = self.draw_boxes_on_image(img_viz, boxes, box_errors)
            
            # 準備可視化
            if visualize and len(images_for_viz) < 100:
                title = f"{img_path.name}\n{len(boxes)} boxes"
                if any(box_errors):
                    title += f" ({sum(1 for e in box_errors if e)} errors)"
                images_for_viz.append((img_drawn, title))
        
        # 生成可視化 grid
        if visualize and images_for_viz:
            # 分批創建 grid（每批 25 張）
            batch_size = 25
            for batch_idx in range(0, len(images_for_viz), batch_size):
                batch = images_for_viz[batch_idx:batch_idx + batch_size]
                output_path = self.output_dir / f"annotation_grid_{batch_idx // batch_size + 1}.png"
                self.create_grid_visualization(batch, grid_size=(5, 5), output_path=output_path)
        
        # 計算錯誤統計
        errors_by_type = {}
        for error in self.errors:
            errors_by_type[error.error_type] = errors_by_type.get(error.error_type, 0) + 1
        
        # 生成報告
        report = ValidationReport(
            total_images=len(sample_paths),
            total_boxes=total_boxes,
            error_boxes=error_boxes,
            error_rate=error_boxes / total_boxes if total_boxes > 0 else 0.0,
            errors_by_type=errors_by_type,
            too_small_boxes=errors_by_type.get('too_small', 0),
            out_of_bounds_boxes=errors_by_type.get('out_of_bounds', 0),
            avg_box_width=np.mean(self.box_widths) if self.box_widths else 0.0,
            avg_box_height=np.mean(self.box_heights) if self.box_heights else 0.0,
            min_box_size=np.min(self.box_sizes) if self.box_sizes else 0.0,
            errors=self.errors
        )
        
        return report
    
    def generate_report_markdown(self, report: ValidationReport) -> str:
        """生成 Markdown 格式報告"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        md = f"""# YOLOv7 Annotation Validation Report

**生成時間**: {timestamp}  
**數據集**: {self.data_root}

---

## 📊 總體統計

| 指標 | 數值 |
|------|------|
| 檢查影像數 | {report.total_images} |
| 總 Bounding Box 數 | {report.total_boxes} |
| 錯誤 Box 數 | {report.error_boxes} |
| **錯誤率** | **{report.error_rate * 100:.2f}%** |

---

## ⚠️ 錯誤類型分布

| 錯誤類型 | 數量 | 比例 |
|---------|------|------|
"""
        
        for error_type, count in report.errors_by_type.items():
            percentage = (count / report.error_boxes * 100) if report.error_boxes > 0 else 0
            md += f"| {error_type} | {count} | {percentage:.2f}% |\n"
        
        md += f"""
---

## 📏 Bounding Box 尺寸統計

| 指標 | 數值 (像素) |
|------|------------|
| 平均寬度 | {report.avg_box_width:.2f} px |
| 平均高度 | {report.avg_box_height:.2f} px |
| 最小邊長 | {report.min_box_size:.2f} px |
| 過小 Box 數 (< {self.min_box_size_px}px) | {report.too_small_boxes} |

---

## 🔍 詳細錯誤列表

共 {len(report.errors)} 個錯誤：

"""
        
        # 最多顯示前 50 個錯誤
        for i, error in enumerate(report.errors[:50]):
            md += f"""
### 錯誤 #{i+1}

- **文件**: `{error.image_path}`
- **Box 索引**: {error.bbox_idx}
- **錯誤類型**: `{error.error_type}`
- **座標**: `{error.bbox}`
- **詳情**: {error.details}

---
"""
        
        if len(report.errors) > 50:
            md += f"\n*（僅顯示前 50 個錯誤，總共 {len(report.errors)} 個）*\n"
        
        md += f"""
---

## 💡 建議

"""
        
        # 根據錯誤類型給出建議
        if report.too_small_boxes > 0:
            md += f"""
### 過小 Bounding Box ({report.too_small_boxes} 個)
- 建議檢查標註工具的設置，確保小病灶也能正確標註
- 考慮調整 `min_box_size_px` 參數或使用更高解析度的圖像
- 訓練時可使用 `small_object_augmentation` 增強小物體檢測
"""
        
        if report.out_of_bounds_boxes > 0:
            md += f"""
### 超出邊界 ({report.out_of_bounds_boxes} 個)
- 檢查標註轉換流程（COCO -> YOLO 等）
- 確保座標歸一化正確（範圍應在 0~1）
- 檢查圖像預處理是否改變了原始尺寸
"""
        
        if 'negative_coords' in report.errors_by_type:
            md += f"""
### 負座標 ({report.errors_by_type['negative_coords']} 個)
- 嚴重錯誤！需立即修復標註檔案
- 檢查標註工具或轉換腳本
"""
        
        md += f"""
---

## 📁 輸出文件

- 報告文件: `{self.output_dir}/validation_report.md`
- 可視化 Grid: `{self.output_dir}/annotation_grid_*.png`
- 詳細 JSON: `{self.output_dir}/validation_report.json`

---

**驗證完成** ✅
"""
        
        return md
    
    def save_report(self, report: ValidationReport):
        """保存報告"""
        # Markdown 報告
        md_report = self.generate_report_markdown(report)
        md_path = self.output_dir / "validation_report.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_report)
        print(f"\n✅ Markdown report saved to: {md_path}")
        
        # JSON 報告（詳細數據）
        json_data = {
            'summary': {
                'total_images': report.total_images,
                'total_boxes': report.total_boxes,
                'error_boxes': report.error_boxes,
                'error_rate': report.error_rate,
                'errors_by_type': report.errors_by_type,
                'avg_box_width': report.avg_box_width,
                'avg_box_height': report.avg_box_height,
                'min_box_size': report.min_box_size,
            },
            'errors': [asdict(e) for e in report.errors]
        }
        
        json_path = self.output_dir / "validation_report.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"✅ JSON report saved to: {json_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLOv7 Annotation Validator")
    parser.add_argument('--data_root', type=str, default='../../datasets/splited_dataset',
                        help='Dataset root directory')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to validate')
    parser.add_argument('--output_dir', type=str, default='./yolov7_logs/annotation_validation',
                        help='Output directory')
    parser.add_argument('--min_box_size_px', type=int, default=3,
                        help='Minimum box size in pixels')
    parser.add_argument('--img_size', type=int, default=640,
                        help='Image size for validation')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'],
                        help='Dataset splits to validate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 初始化驗證器
    validator = AnnotationValidator(
        data_root=args.data_root,
        output_dir=args.output_dir,
        min_box_size_px=args.min_box_size_px,
        img_size=args.img_size
    )
    
    # 收集樣本
    all_samples = []
    data_root = Path(args.data_root)
    
    for split in args.splits:
        images_dir = data_root / split / 'images'
        labels_dir = data_root / split / 'labels'
        
        if not images_dir.exists():
            print(f"⚠️  Warning: {images_dir} not found, skipping...")
            continue
        
        # 收集所有圖像和標註對
        image_files = list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg'))
        for img_path in image_files:
            label_path = labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                all_samples.append((img_path, label_path))
    
    print(f"\n Found {len(all_samples)} samples from {args.splits}")
    
    # 隨機抽樣
    if len(all_samples) > args.num_samples:
        all_samples = random.sample(all_samples, args.num_samples)
    
    print(f"Validating {len(all_samples)} samples...\n")
    
    # 執行驗證
    report = validator.validate_dataset_samples(all_samples, visualize=True)
    
    # 保存報告
    validator.save_report(report)
    
    # 打印總結
    print(f"\n{'='*80}")
    print("📊 Validation Summary")
    print(f"{'='*80}")
    print(f"Total images: {report.total_images}")
    print(f"Total boxes: {report.total_boxes}")
    print(f"Error boxes: {report.error_boxes} ({report.error_rate * 100:.2f}%)")
    print(f"\nError breakdown:")
    for error_type, count in report.errors_by_type.items():
        print(f"  - {error_type}: {count}")
    print(f"\nBox size stats:")
    print(f"  - Average width: {report.avg_box_width:.2f} px")
    print(f"  - Average height: {report.avg_box_height:.2f} px")
    print(f"  - Min size: {report.min_box_size:.2f} px")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
