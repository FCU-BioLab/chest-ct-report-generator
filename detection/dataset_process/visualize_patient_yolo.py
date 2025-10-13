#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
視覺化按患者分組的 YOLO 資料集
用於檢查 preprocessed_yolo_lesion 資料集

使用方法:
    # 視覺化單個患者
    python visualize_patient_yolo.py --data_dir ../../datasets/preprocessed_yolo_lesion --patient A0001 --num_samples 10
    
    # 互動模式檢查患者資料
    python visualize_patient_yolo.py --data_dir ../../datasets/preprocessed_yolo_lesion --patient A0001 --interactive
    
    # 視覺化所有患者（每位患者抽樣）
    python visualize_patient_yolo.py --data_dir ../../datasets/preprocessed_yolo_lesion --patient all --num_samples 5
    
    # 只顯示有標註的樣本
    python visualize_patient_yolo.py --data_dir ../../datasets/preprocessed_yolo_lesion --patient A0001 --only_annotated
"""

import os
import sys
import cv2
import numpy as np
import random
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc="", total=None):
        return iterable

try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  警告: matplotlib 未安装")


class PatientYOLOVisualizer:
    """按患者分組的 YOLO 標註視覺化工具"""
    
    def __init__(
        self,
        data_dir: str,
        class_names: List[str] = None,
        colors: List[Tuple[int, int, int]] = None
    ):
        """
        初始化視覺化工具
        
        Args:
            data_dir: 資料目錄 (包含 A0001, A0002... 等患者資料夾)
            class_names: 類別名稱列表
            colors: 每個類別的顏色（BGR 格式）
        """
        self.data_dir = Path(data_dir)
        self.class_names = class_names or ["lesion"]
        self.num_classes = len(self.class_names)
        
        # 設定顏色
        if colors is None:
            self.colors = self._generate_colors(self.num_classes)
        else:
            self.colors = colors
        
        # 載入處理報告
        self.processing_report = self._load_processing_report()
    
    def _load_processing_report(self) -> Optional[Dict]:
        """載入處理報告"""
        report_path = self.data_dir / "processing_report.json"
        if report_path.exists():
            with open(report_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def _generate_colors(self, num_classes: int) -> List[Tuple[int, int, int]]:
        """生成不同的顏色（BGR 格式）"""
        colors = []
        for i in range(num_classes):
            hue = int(180 * i / num_classes)
            color_hsv = np.uint8([[[hue, 255, 255]]])
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, color_bgr)))
        return colors
    
    def get_all_patients(self) -> List[str]:
        """獲取所有患者ID"""
        patients = []
        for d in sorted(self.data_dir.iterdir()):
            if d.is_dir() and d.name.startswith('A'):
                patients.append(d.name)
        return patients
    
    def load_image_and_labels(
        self, 
        patient_id: str,
        image_name: str
    ) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
        """
        載入圖像和對應的標註
        
        Args:
            patient_id: 患者ID
            image_name: 圖像檔名
            
        Returns:
            image: 圖像數組
            labels: 標註列表 [(class_id, cx, cy, w, h), ...]
        """
        # 讀取圖像
        image_path = self.data_dir / patient_id / "images" / image_name
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"無法讀取圖像: {image_path}")
        
        # 讀取標註
        label_name = image_path.stem + ".txt"
        label_path = self.data_dir / patient_id / "labels" / label_name
        labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            cx, cy, w, h = map(float, parts[1:5])
                            labels.append((class_id, cx, cy, w, h))
        
        return image, labels
    
    def draw_yolo_boxes(
        self,
        image: np.ndarray,
        labels: List[Tuple[int, float, float, float, float]],
        thickness: int = 2,
        font_scale: float = 0.6
    ) -> np.ndarray:
        """在圖像上繪製 YOLO 標註框"""
        img_draw = image.copy()
        img_h, img_w = image.shape[:2]
        
        for label_idx, (class_id, cx, cy, w, h) in enumerate(labels, 1):
            # 將歸一化座標轉換為像素座標
            cx_pixel = int(cx * img_w)
            cy_pixel = int(cy * img_h)
            w_pixel = int(w * img_w)
            h_pixel = int(h * img_h)
            
            # 計算左上角和右下角座標
            x1 = int(cx_pixel - w_pixel / 2)
            y1 = int(cy_pixel - h_pixel / 2)
            x2 = int(cx_pixel + w_pixel / 2)
            y2 = int(cy_pixel + h_pixel / 2)
            
            # 確保座標在圖像範圍內
            x1 = max(0, min(x1, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            x2 = max(0, min(x2, img_w - 1))
            y2 = max(0, min(y2, img_h - 1))
            
            # 選擇顏色
            color = self.colors[class_id % len(self.colors)]
            
            # 繪製邊界框
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, thickness)
            
            # 繪製中心點
            cv2.circle(img_draw, (cx_pixel, cy_pixel), 4, color, -1)
            
            # 準備標籤文字
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            label_text = f"#{label_idx} {class_name}"
            size_text = f"[{w_pixel}x{h_pixel}]"
            
            # 繪製標籤背景
            (text_w, text_h), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            cv2.rectangle(
                img_draw,
                (x1, y1 - text_h - baseline - 5),
                (x1 + text_w, y1),
                color,
                -1
            )
            
            # 繪製標籤文字
            cv2.putText(
                img_draw,
                label_text,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )
            
            # 顯示尺寸
            cv2.putText(
                img_draw,
                size_text,
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA
            )
        
        return img_draw
    
    def add_info_panel(
        self,
        image: np.ndarray,
        patient_id: str,
        image_name: str,
        num_boxes: int,
        image_size: Tuple[int, int]
    ) -> np.ndarray:
        """在圖像上添加資訊面板"""
        img_with_info = image.copy()
        h, w = img_with_info.shape[:2]
        
        # 創建頂部資訊條
        info_height = 100
        info_panel = np.zeros((info_height, w, 3), dtype=np.uint8)
        info_panel[:] = (40, 40, 40)  # 深灰色背景
        
        # 添加資訊文字
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        color = (255, 255, 255)
        
        info_texts = [
            f"患者: {patient_id}",
            f"檔案: {image_name}",
            f"尺寸: {image_size[1]} x {image_size[0]}",
            f"標註: {num_boxes} 個病灶"
        ]
        
        y_offset = 25
        for text in info_texts:
            cv2.putText(
                info_panel, text, (10, y_offset),
                font, font_scale, color, thickness, cv2.LINE_AA
            )
            y_offset += 25
        
        # 合併資訊面板和圖像
        result = np.vstack([info_panel, img_with_info])
        
        return result
    
    def show_with_matplotlib(
        self,
        image: np.ndarray,
        title: str = "YOLO Annotation"
    ) -> str:
        """使用 matplotlib 顯示圖像"""
        if not MATPLOTLIB_AVAILABLE:
            print("❌ matplotlib 不可用")
            return 'q'
        
        # 轉換 BGR 到 RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 創建圖形
        fig, ax = plt.subplots(figsize=(14, 12))
        ax.imshow(image_rgb)
        ax.set_title(title, fontsize=16, pad=15)
        ax.axis('off')
        
        # 添加操作說明
        instruction_text = (
            "操作: n/→=下一張 | p/←=上一張 | r=隨機 | s=儲存 | q/ESC=退出 | 關閉視窗=下一張"
        )
        fig.text(
            0.5, 0.02, instruction_text,
            ha='center', va='bottom', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8)
        )
        
        plt.tight_layout()
        
        # 按鍵事件處理
        pressed_key = {'key': 'n'}
        
        def on_key(event):
            if event.key in ['n', 'right']:
                pressed_key['key'] = 'n'
                plt.close(fig)
            elif event.key in ['p', 'left']:
                pressed_key['key'] = 'p'
                plt.close(fig)
            elif event.key == 'r':
                pressed_key['key'] = 'r'
                plt.close(fig)
            elif event.key == 's':
                pressed_key['key'] = 's'
                plt.close(fig)
            elif event.key in ['q', 'escape']:
                pressed_key['key'] = 'q'
                plt.close(fig)
        
        def on_close(event):
            pressed_key['key'] = 'n'
        
        fig.canvas.mpl_connect('key_press_event', on_key)
        fig.canvas.mpl_connect('close_event', on_close)
        
        plt.show()
        
        return pressed_key['key']
    
    def visualize_patient(
        self,
        patient_id: str,
        num_samples: int = 10,
        only_annotated: bool = False,
        random_sample: bool = True,
        output_dir: Optional[Path] = None,
        show: bool = True
    ) -> Dict:
        """視覺化單個患者的資料"""
        patient_dir = self.data_dir / patient_id
        images_dir = patient_dir / "images"
        labels_dir = patient_dir / "labels"
        
        if not images_dir.exists():
            print(f"❌ 找不到患者目錄: {patient_dir}")
            return None
        
        # 獲取所有圖像
        image_files = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))
        
        if not image_files:
            print(f"❌ 找不到圖像文件: {images_dir}")
            return None
        
        print(f"\n{'='*80}")
        print(f"👤 患者: {patient_id}")
        print(f"{'='*80}")
        print(f"圖像目錄: {images_dir}")
        print(f"總圖像數: {len(image_files)}")
        
        # 如果只顯示有標註的
        if only_annotated:
            annotated_files = []
            for img_file in image_files:
                label_file = labels_dir / f"{img_file.stem}.txt"
                if label_file.exists() and label_file.stat().st_size > 0:
                    annotated_files.append(img_file)
            image_files = annotated_files
            print(f"有標註的圖像: {len(image_files)}")
        
        # 選擇樣本
        if random_sample:
            selected_files = random.sample(image_files, min(num_samples, len(image_files)))
        else:
            selected_files = image_files[:num_samples]
        
        print(f"檢查樣本: {len(selected_files)} 個")
        
        # 統計資訊
        stats = {
            "patient_id": patient_id,
            "total_images": len(image_files),
            "checked_images": 0,
            "images_with_boxes": 0,
            "images_without_boxes": 0,
            "total_boxes": 0
        }
        
        # 視覺化樣本
        if output_dir:
            output_dir = Path(output_dir) / patient_id
            output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🔍 開始視覺化...")
        for img_path in tqdm(selected_files, desc=f"處理 {patient_id}"):
            try:
                # 載入標註統計
                image, labels = self.load_image_and_labels(patient_id, img_path.name)
                
                # 更新統計
                stats["checked_images"] += 1
                if labels:
                    stats["images_with_boxes"] += 1
                    stats["total_boxes"] += len(labels)
                else:
                    stats["images_without_boxes"] += 1
                
                # 繪製標註框
                img_with_boxes = self.draw_yolo_boxes(image, labels)
                
                # 添加資訊面板
                img_final = self.add_info_panel(
                    img_with_boxes,
                    patient_id,
                    img_path.name,
                    len(labels),
                    image.shape[:2]
                )
                
                # 儲存
                if output_dir:
                    save_path = output_dir / img_path.name
                    cv2.imwrite(str(save_path), img_final)
                
                # 顯示
                if show and not only_annotated:
                    if MATPLOTLIB_AVAILABLE:
                        self.show_with_matplotlib(img_final, f"{patient_id} - {img_path.name}")
                
            except Exception as e:
                print(f"❌ 處理 {img_path.name} 時出錯: {e}")
                continue
        
        # 打印統計
        self._print_stats(stats)
        
        return stats
    
    def interactive_mode(self, patient_id: str, only_annotated: bool = False):
        """互動式檢查模式"""
        patient_dir = self.data_dir / patient_id
        images_dir = patient_dir / "images"
        labels_dir = patient_dir / "labels"
        
        image_files = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))
        
        if not image_files:
            print(f"❌ 找不到圖像: {images_dir}")
            return
        
        # 如果只顯示有標註的
        if only_annotated:
            annotated_files = []
            for img_file in image_files:
                label_file = labels_dir / f"{img_file.stem}.txt"
                if label_file.exists() and label_file.stat().st_size > 0:
                    annotated_files.append(img_file)
            image_files = annotated_files
        
        print(f"\n{'='*80}")
        print(f"🎮 互動式檢查模式")
        print(f"{'='*80}")
        print(f"患者: {patient_id}")
        print(f"圖像數量: {len(image_files)}")
        if only_annotated:
            print(f"模式: 僅顯示有標註的切片")
        print(f"{'='*80}\n")
        
        current_idx = 0
        save_dir = Path("./visualization_saves") / patient_id
        save_dir.mkdir(parents=True, exist_ok=True)
        
        while True:
            img_path = image_files[current_idx]
            
            # 載入並顯示
            image, labels = self.load_image_and_labels(patient_id, img_path.name)
            img_with_boxes = self.draw_yolo_boxes(image, labels)
            img_final = self.add_info_panel(
                img_with_boxes,
                patient_id,
                f"{img_path.name} ({current_idx+1}/{len(image_files)})",
                len(labels),
                image.shape[:2]
            )
            
            if MATPLOTLIB_AVAILABLE:
                key = self.show_with_matplotlib(
                    img_final,
                    f"{patient_id} - {img_path.name} ({current_idx+1}/{len(image_files)})"
                )
                
                if key == 'q':
                    print("👋 退出互動模式")
                    break
                elif key == 'n':
                    current_idx = (current_idx + 1) % len(image_files)
                elif key == 'p':
                    current_idx = (current_idx - 1) % len(image_files)
                elif key == 'r':
                    current_idx = random.randint(0, len(image_files) - 1)
                    print(f"🎲 隨機跳轉到第 {current_idx + 1} 張")
                elif key == 's':
                    save_path = save_dir / f"saved_{img_path.name}"
                    cv2.imwrite(str(save_path), img_final)
                    print(f"💾 已儲存: {save_path}")
            else:
                print("❌ matplotlib 不可用，無法使用互動模式")
                break
    
    def _print_stats(self, stats: Dict):
        """打印統計資訊"""
        print(f"\n{'='*80}")
        print(f"📈 統計結果")
        print(f"{'='*80}")
        print(f"患者ID:         {stats['patient_id']}")
        print(f"總圖像數:       {stats['total_images']}")
        print(f"已檢查圖像:     {stats['checked_images']}")
        print(f"  - 有標註:     {stats['images_with_boxes']} ({stats['images_with_boxes']/max(stats['checked_images'],1)*100:.1f}%)")
        print(f"  - 無標註:     {stats['images_without_boxes']} ({stats['images_without_boxes']/max(stats['checked_images'],1)*100:.1f}%)")
        print(f"總標註框數:     {stats['total_boxes']}")
        if stats['images_with_boxes'] > 0:
            print(f"平均框數/圖:    {stats['total_boxes']/stats['images_with_boxes']:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="視覺化按患者分組的 YOLO 資料集",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="資料目錄 (包含 A0001, A0002... 等患者資料夾)"
    )
    parser.add_argument(
        "--patient", type=str, default="A0001",
        help="患者ID (例如: A0001) 或 'all' 查看所有患者"
    )
    parser.add_argument(
        "--num_samples", type=int, default=10,
        help="每位患者要視覺化的樣本數量"
    )
    parser.add_argument(
        "--only_annotated", action="store_true",
        help="只顯示有標註的切片"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="視覺化結果輸出目錄"
    )
    parser.add_argument(
        "--no_random", action="store_true",
        help="不隨機抽樣，按順序選擇"
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="啟用互動式檢查模式"
    )
    parser.add_argument(
        "--no_show", action="store_true",
        help="不顯示圖像，只儲存"
    )
    
    args = parser.parse_args()
    
    # 創建視覺化工具
    print(f"\n{'='*80}")
    print(f"🔍 患者YOLO標註視覺化工具")
    print(f"{'='*80}")
    
    visualizer = PatientYOLOVisualizer(args.data_dir)
    
    print(f"資料目錄: {args.data_dir}")
    print(f"類別名稱: {', '.join(visualizer.class_names)}")
    
    # 獲取所有患者
    all_patients = visualizer.get_all_patients()
    print(f"可用患者: {', '.join(all_patients)}")
    
    # 互動模式
    if args.interactive:
        if args.patient == "all":
            print("⚠️  互動模式不支援 'all'，將使用第一位患者")
            patient_id = all_patients[0] if all_patients else "A0001"
        else:
            patient_id = args.patient
        visualizer.interactive_mode(patient_id, only_annotated=args.only_annotated)
        return
    
    # 批次視覺化模式
    output_dir = Path(args.output_dir) if args.output_dir else None
    show = not args.no_show
    random_sample = not args.no_random
    
    if args.patient == "all":
        # 檢查所有患者
        all_stats = []
        for patient_id in all_patients:
            stats = visualizer.visualize_patient(
                patient_id,
                num_samples=args.num_samples,
                only_annotated=args.only_annotated,
                random_sample=random_sample,
                output_dir=output_dir,
                show=show
            )
            if stats:
                all_stats.append(stats)
        
        # 打印整體統計
        if all_stats:
            print(f"\n{'='*80}")
            print(f"📊 整體統計")
            print(f"{'='*80}")
            total_images = sum(s['total_images'] for s in all_stats)
            total_boxes = sum(s['total_boxes'] for s in all_stats)
            print(f"總患者數: {len(all_stats)}")
            print(f"總圖像數: {total_images}")
            print(f"總標註框: {total_boxes}")
    else:
        # 檢查單個患者
        visualizer.visualize_patient(
            args.patient,
            num_samples=args.num_samples,
            only_annotated=args.only_annotated,
            random_sample=random_sample,
            output_dir=output_dir,
            show=show
        )
    
    print(f"\n{'='*80}")
    print(f"✅ 檢查完成！")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
