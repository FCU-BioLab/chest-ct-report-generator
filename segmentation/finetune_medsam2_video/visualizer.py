#!/usr/bin/env python3
"""
視頻視覺化模組
==============

播放和視覺化轉換完成的 NPZ 視頻資料。

功能：
- 互動式播放 CT 切片序列
- 顯示 Ground Truth mask 疊加
- 保存為 GIF 動畫
- 批量預覽多個樣本
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


class VideoVisualizer:
    """
    NPZ 視頻視覺化器
    
    提供互動式播放和動畫保存功能。
    """
    
    def __init__(self, npz_dir: str = "video_npz"):
        """
        初始化視覺化器
        
        Args:
            npz_dir: NPZ 檔案目錄
        """
        self.npz_dir = Path(npz_dir)
        self.samples = self._load_sample_list()
        
        logger.info(f"📹 視頻視覺化器初始化")
        logger.info(f"  - NPZ 目錄: {self.npz_dir}")
        logger.info(f"  - 總樣本數: {len(self.samples)}")
    
    def _load_sample_list(self) -> List[Path]:
        """載入所有 NPZ 檔案列表"""
        samples = []
        
        for split in ['train', 'val', 'test']:
            split_dir = self.npz_dir / split
            if split_dir.exists():
                samples.extend(sorted(split_dir.glob("*.npz")))
        
        # 也檢查根目錄
        samples.extend(sorted(self.npz_dir.glob("*.npz")))
        
        return samples
    
    def load_video(self, npz_path: Union[str, Path]) -> Dict:
        """
        載入單個 NPZ 視頻
        
        Args:
            npz_path: NPZ 檔案路徑
            
        Returns:
            視頻資料字典
        """
        npz_path = Path(npz_path)
        data = np.load(npz_path, allow_pickle=True)
        
        result = {
            'frames': data['frames'],           # (D, H, W)
            'masks': data['masks'],             # (D, H, W)
            'center_idx': int(data['center_idx']),
            'patient_id': str(data.get('patient_id', 'Unknown')),
            'lesion_id': int(data.get('lesion_id', 0)),
            'diameter_mm': float(data.get('diameter_mm', 0)),
            'spacing': data.get('spacing', np.array([1, 1, 1])),
            'bbox': data.get('bbox', None),
            'path': str(npz_path),
        }
        
        # 如果檔案中沒有 bbox，則從 mask 計算 (fallback)
        if result['bbox'] is None:
            # 找出有標註的中心幀（或最大標註幀）
            center_idx = result['center_idx']
            masks = result['masks']
            if center_idx < len(masks):
                result['bbox'] = self._compute_bbox_from_mask(masks[center_idx])
        
        return result

    def _compute_bbox_from_mask(self, mask: np.ndarray) -> np.ndarray:
        """從 mask 計算 bounding box (與 video_dataset.py 邏輯一致)"""
        if mask.max() == 0:
            h, w = mask.shape
            cx, cy = w // 2, h // 2
            return np.array([cx - 10, cy - 10, cx + 10, cy + 10], dtype=np.float32)
        
        ys, xs = np.where(mask > 0)
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        
        # 稍微擴大 bbox
        pad = 5
        h, w = mask.shape
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        
        return np.array([x1, y1, x2, y2], dtype=np.float32)
    
    def play_interactive(self, npz_path: Union[str, Path, int] = 0):
        """
        互動式播放視頻
        
        使用 matplotlib 滑桿控制播放。
        
        Args:
            npz_path: NPZ 路徑或樣本索引
        """
        import matplotlib.patches as patches

        # 取得 NPZ 路徑
        if isinstance(npz_path, int):
            if npz_path >= len(self.samples):
                logger.error(f"❌ 索引超出範圍: {npz_path} >= {len(self.samples)}")
                return
            npz_path = self.samples[npz_path]
        
        # 載入資料
        video = self.load_video(npz_path)
        frames = video['frames']
        masks = video['masks']
        center_idx = video['center_idx']
        num_frames = len(frames)
        
        # 建立圖形
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        plt.subplots_adjust(bottom=0.25)
        
        # 標題
        fig.suptitle(
            f"Patient: {video['patient_id']} | Lesion: {video['lesion_id']} | "
            f"Diameter: {video['diameter_mm']:.1f}mm | Frames: {num_frames}",
            fontsize=12
        )
        
        # 初始顯示
        im_frame = axes[0].imshow(frames[0], cmap='gray', vmin=0, vmax=255)
        axes[0].set_title(f'CT Slice (Frame 0/{num_frames-1})')
        axes[0].axis('off')
        
        # Mask 疊加
        im_overlay = axes[1].imshow(frames[0], cmap='gray', vmin=0, vmax=255)
        
        # BBox patch (初始化為不可見)
        # 用於顯示每幀的 bbox
        bbox_patch = patches.Rectangle(
            (0, 0), 1, 1, linewidth=2, edgecolor='yellow', facecolor='none', linestyle='--'
        )
        axes[0].add_patch(bbox_patch)
        bbox_patch.set_visible(False)

        # 建立疊加 mask
        mask_overlay = self._create_mask_overlay(frames[0], masks[0])
        im_mask = axes[1].imshow(mask_overlay, alpha=0.5)
        axes[1].set_title(f'With Mask (Frame 0)')
        axes[1].axis('off')
        
        # 標記中心幀
        center_marker = axes[0].text(
            0.02, 0.98, '', transform=axes[0].transAxes,
            fontsize=12, color='yellow', fontweight='bold',
            verticalalignment='top', backgroundcolor='black'
        )
        
        # 滑桿
        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
        slider = Slider(ax_slider, 'Frame', 0, num_frames - 1, valinit=0, valstep=1)
        
        # 播放按鈕 (使用 ASCII 相容符號)
        ax_play = plt.axes([0.2, 0.02, 0.1, 0.04])
        ax_stop = plt.axes([0.35, 0.02, 0.1, 0.04])
        ax_prev = plt.axes([0.5, 0.02, 0.1, 0.04])
        ax_next = plt.axes([0.65, 0.02, 0.1, 0.04])
        
        btn_play = Button(ax_play, '> Play')
        btn_stop = Button(ax_stop, '[] Stop')
        btn_prev = Button(ax_prev, '< Prev')
        btn_next = Button(ax_next, 'Next >')
        
        # 狀態
        state = {'playing': False, 'timer': None, 'current_frame': 0}
        
        def update(frame_idx):
            """更新顯示"""
            frame_idx = int(frame_idx)
            state['current_frame'] = frame_idx
            
            # 更新影像
            im_frame.set_array(frames[frame_idx])
            axes[0].set_title(f'CT Slice (Frame {frame_idx}/{num_frames-1})')
            
            # 更新 BBox (每幀獨立計算)
            mask = masks[frame_idx]
            if mask.max() > 0:
                bbox = self._compute_bbox_from_mask(mask)
                x1, y1, x2, y2 = bbox
                bbox_patch.set_bounds(x1, y1, x2 - x1, y2 - y1)
                bbox_patch.set_visible(True)
            else:
                bbox_patch.set_visible(False)
            
            # 更新疊加
            im_overlay.set_array(frames[frame_idx])
            mask_overlay = self._create_mask_overlay(frames[frame_idx], masks[frame_idx])
            im_mask.set_array(mask_overlay)
            
            # 中心幀標記
            if frame_idx == center_idx:
                center_marker.set_text('* CENTER FRAME *')
                axes[1].set_title(f'With Mask (Frame {frame_idx}) * CENTER')
            else:
                center_marker.set_text('')
                axes[1].set_title(f'With Mask (Frame {frame_idx})')
            
            fig.canvas.draw_idle()
        
        def on_slider_change(val):
            update(val)
        
        def timer_callback():
            """Timer 回調 - 播放下一幀"""
            if state['playing']:
                next_frame = (state['current_frame'] + 1) % num_frames
                slider.set_val(next_frame)
        
        def on_play(event):
            if not state['playing']:
                state['playing'] = True
                state['timer'] = fig.canvas.new_timer(interval=200)
                state['timer'].add_callback(timer_callback)
                state['timer'].start()
                btn_play.label.set_text('Playing...')
                fig.canvas.draw_idle()
        
        def on_stop(event):
            state['playing'] = False
            if state['timer']:
                state['timer'].stop()
                state['timer'] = None
            btn_play.label.set_text('> Play')
            fig.canvas.draw_idle()
        
        def on_prev(event):
            current = int(slider.val)
            if current > 0:
                slider.set_val(current - 1)
            else:
                slider.set_val(num_frames - 1)
        
        def on_next(event):
            current = int(slider.val)
            if current < num_frames - 1:
                slider.set_val(current + 1)
            else:
                slider.set_val(0)
        
        def on_close(event):
            on_stop(None)
        
        slider.on_changed(on_slider_change)
        btn_play.on_clicked(on_play)
        btn_stop.on_clicked(on_stop)
        btn_prev.on_clicked(on_prev)
        btn_next.on_clicked(on_next)
        fig.canvas.mpl_connect('close_event', on_close)
        
        # 初始更新
        update(0)
        
        plt.show()
    
    def _create_mask_overlay(
        self, 
        frame: np.ndarray, 
        mask: np.ndarray,
        mask_color: Tuple[int, int, int] = (255, 0, 0),
        alpha: float = 0.5
    ) -> np.ndarray:
        """建立 mask 疊加影像"""
        if len(frame.shape) == 2:
            rgb = np.stack([frame, frame, frame], axis=-1)
        else:
            rgb = frame.copy()
        
        overlay = rgb.copy().astype(np.float32)
        
        if mask.max() > 0:
            mask_bool = mask > 0
            for c, color_val in enumerate(mask_color):
                overlay[mask_bool, c] = overlay[mask_bool, c] * (1 - alpha) + color_val * alpha
        
        return overlay.astype(np.uint8)

    def _draw_bbox_on_image(
        self,
        image: np.ndarray,
        bbox: np.ndarray,
        color: Tuple[int, int, int] = (255, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """在影像上繪製 BBox"""
        try:
            import cv2
            image = image.copy()
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            return image
        except ImportError:
            return image

    def save_gif(
        self,
        npz_path: Union[str, Path, int],
        output_path: Optional[str] = None,
        fps: int = 5,
        show_mask: bool = True,
        show_info: bool = True,
    ) -> str:
        """
        將視頻保存為 GIF
        
        Args:
            npz_path: NPZ 路徑或樣本索引
            output_path: 輸出路徑（可選）
            fps: 幀率
            show_mask: 是否顯示 mask
            show_info: 是否顯示資訊
            
        Returns:
            輸出檔案路徑
        """
        if isinstance(npz_path, int):
            npz_path = self.samples[npz_path]
        npz_path = Path(npz_path)
        
        video = self.load_video(npz_path)
        frames = video['frames']
        masks = video['masks']
        center_idx = video['center_idx']
        
        if output_path is None:
            output_path = npz_path.with_suffix('.gif')
        
        images = []
        
        for i in range(len(frames)):
            if show_mask:
                frame_rgb = self._create_mask_overlay(frames[i], masks[i])
            else:
                frame_rgb = np.stack([frames[i], frames[i], frames[i]], axis=-1)
            
            # 若每幀有 mask，則繪製 bbox
            if masks[i].max() > 0:
                frame_bbox = self._compute_bbox_from_mask(masks[i])
                frame_rgb = self._draw_bbox_on_image(frame_rgb, frame_bbox)

            if show_info:
                frame_rgb = self._add_text_overlay(
                    frame_rgb,
                    f"Frame {i}/{len(frames)-1}" + (" ⭐" if i == center_idx else ""),
                    position='top'
                )
                
                if masks[i].max() > 0:
                    mask_area = np.sum(masks[i] > 0)
                    frame_rgb = self._add_text_overlay(
                        frame_rgb,
                        f"Mask: {mask_area} px",
                        position='bottom'
                    )
            
            images.append(Image.fromarray(frame_rgb))
        
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=1000 // fps,
            loop=0,
        )
        
        logger.info(f"💾 GIF 已保存: {output_path}")
        return str(output_path)
    
    def _add_text_overlay(
        self,
        image: np.ndarray,
        text: str,
        position: str = 'top',
        font_scale: float = 0.7,
        color: Tuple[int, int, int] = (255, 255, 0),
    ) -> np.ndarray:
        """在影像上添加文字"""
        try:
            import cv2
            
            image = image.copy()
            h, w = image.shape[:2]
            
            (text_w, text_h), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
            )
            
            if position == 'top':
                x, y = 10, text_h + 10
            else:
                x, y = 10, h - 10
            
            cv2.rectangle(image, (x - 5, y - text_h - 5), (x + text_w + 5, y + 5), (0, 0, 0), -1)
            cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
            
            return image
            
        except ImportError:
            return image
    
    def save_mp4(
        self,
        npz_path: Union[str, Path, int],
        output_path: Optional[str] = None,
        fps: int = 10,
        show_mask: bool = True,
    ) -> str:
        """
        將視頻保存為 MP4
        
        Args:
            npz_path: NPZ 路徑或樣本索引
            output_path: 輸出路徑
            fps: 幀率
            show_mask: 是否顯示 mask
            
        Returns:
            輸出檔案路徑
        """
        try:
            import cv2
        except ImportError:
            logger.error("❌ 需要安裝 OpenCV: pip install opencv-python")
            return ""
        
        if isinstance(npz_path, int):
            npz_path = self.samples[npz_path]
        npz_path = Path(npz_path)
        
        video = self.load_video(npz_path)
        frames = video['frames']
        masks = video['masks']
        center_idx = video['center_idx']
        
        if output_path is None:
            output_path = str(npz_path.with_suffix('.mp4'))
        
        h, w = frames[0].shape
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        for i in range(len(frames)):
            if show_mask:
                frame_rgb = self._create_mask_overlay(frames[i], masks[i])
            else:
                frame_rgb = np.stack([frames[i], frames[i], frames[i]], axis=-1)
            
            # 若每幀有 mask，則繪製 bbox
            if masks[i].max() > 0:
                frame_bbox = self._compute_bbox_from_mask(masks[i])
                frame_rgb = self._draw_bbox_on_image(frame_rgb, frame_bbox)
            
            frame_rgb = self._add_text_overlay(
                frame_rgb,
                f"Frame {i}/{len(frames)-1}" + (" CENTER" if i == center_idx else ""),
                position='top'
            )
            
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
        
        writer.release()
        
        logger.info(f"💾 MP4 已保存: {output_path}")
        return output_path
    
    def preview_grid(
        self,
        num_samples: int = 9,
        split: str = 'train',
        output_path: Optional[str] = None,
    ):
        """
        預覽多個樣本的網格圖
        
        Args:
            num_samples: 顯示樣本數
            split: 資料分割
            output_path: 輸出路徑（可選）
        """
        import matplotlib.patches as patches

        # 過濾指定 split 的樣本
        split_samples = [s for s in self.samples if split in str(s)]
        
        if not split_samples:
            logger.warning(f"⚠️ 找不到 {split} 分割的樣本")
            return
        
        num_samples = min(num_samples, len(split_samples))
        cols = int(np.ceil(np.sqrt(num_samples)))
        rows = int(np.ceil(num_samples / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, sample_path in enumerate(split_samples[:num_samples]):
            row, col = idx // cols, idx % cols
            
            video = self.load_video(sample_path)
            center_idx = video['center_idx']
            bbox = video.get('bbox')
            
            # 顯示中心幀
            frame = video['frames'][center_idx]
            mask = video['masks'][center_idx]
            
            overlay = self._create_mask_overlay(frame, mask)
            
            axes[row, col].imshow(overlay)
            
            # 畫 BBox
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1
                # 建立 patch (黃色虛線框)
                bbox_patch = patches.Rectangle(
                    (x1, y1), w, h, linewidth=1.5, edgecolor='yellow', facecolor='none', linestyle='--'
                )
                axes[row, col].add_patch(bbox_patch)
            
            axes[row, col].set_title(
                f"{video['patient_id']}\n"
                f"Lesion {video['lesion_id']} | {video['diameter_mm']:.1f}mm\n"
                f"Frames: {len(video['frames'])}",
                fontsize=9
            )
            axes[row, col].axis('off')
        
        # 隱藏空白子圖
        for idx in range(num_samples, rows * cols):
            row, col = idx // cols, idx % cols
            axes[row, col].axis('off')
        
        plt.suptitle(f'{split.upper()} Split - Center Frames with Masks & BBox Prompts', fontsize=14)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"💾 網格圖已保存: {output_path}")
        else:
            plt.show()
    
    def batch_save_gifs(
        self,
        output_dir: str = "video_gifs",
        split: Optional[str] = None,
        max_samples: Optional[int] = None,
        fps: int = 5,
    ):
        """
        批量將所有樣本保存為 GIF
        
        Args:
            output_dir: 輸出目錄
            split: 指定分割（可選）
            max_samples: 最大樣本數（可選）
            fps: 幀率
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        samples = self.samples
        
        if split:
            samples = [s for s in samples if split in str(s)]
        
        if max_samples:
            samples = samples[:max_samples]
        
        logger.info(f"📦 批量生成 {len(samples)} 個 GIF...")
        
        for sample_path in tqdm(samples, desc="Generating GIFs"):
            try:
                output_path = output_dir / f"{sample_path.stem}.gif"
                self.save_gif(sample_path, str(output_path), fps=fps)
            except Exception as e:
                logger.warning(f"⚠️ 處理 {sample_path.name} 失敗: {e}")
        
        logger.info(f"✅ 完成! GIF 保存於: {output_dir}")
    
    def show_sample_animation(
        self,
        npz_path: Union[str, Path, int] = 0,
        interval: int = 200,
    ):
        """
        使用 matplotlib 動畫顯示樣本
        
        Args:
            npz_path: NPZ 路徑或索引
            interval: 幀間隔 (ms)
        """
        # 取得 NPZ 路徑
        if isinstance(npz_path, int):
            npz_path = self.samples[npz_path]
        
        video = self.load_video(npz_path)
        frames = video['frames']
        masks = video['masks']
        center_idx = video['center_idx']
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 初始幀
        overlay = self._create_mask_overlay(frames[0], masks[0])
        im = ax.imshow(overlay)
        ax.axis('off')
        
        title = ax.set_title(
            f"{video['patient_id']} | Lesion {video['lesion_id']} | "
            f"Diameter: {video['diameter_mm']:.1f}mm\nFrame 0/{len(frames)-1}"
        )
        
        def animate(i):
            overlay = self._create_mask_overlay(frames[i], masks[i])
            im.set_array(overlay)
            
            marker = " * CENTER *" if i == center_idx else ""
            title.set_text(
                f"{video['patient_id']} | Lesion {video['lesion_id']} | "
                f"Diameter: {video['diameter_mm']:.1f}mm\n"
                f"Frame {i}/{len(frames)-1}{marker}"
            )
            return [im, title]
        
        anim = animation.FuncAnimation(
            fig, animate, frames=len(frames),
            interval=interval, blit=True, repeat=True
        )
        
        plt.tight_layout()
        plt.show()


def main():
    """命令列工具"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='視頻視覺化工具')
    parser.add_argument('--npz_dir', type=str, default='cache/lndb_video_npz',
                       help='NPZ 資料目錄')
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # play 子命令
    play_parser = subparsers.add_parser('play', help='互動式播放')
    play_parser.add_argument('--index', type=int, default=0,
                            help='樣本索引')
    play_parser.add_argument('--path', type=str, default=None,
                            help='NPZ 檔案路徑')
    
    # animate 子命令
    anim_parser = subparsers.add_parser('animate', help='動畫播放')
    anim_parser.add_argument('--index', type=int, default=0)
    anim_parser.add_argument('--interval', type=int, default=200,
                            help='幀間隔 (ms)')
    
    # gif 子命令
    gif_parser = subparsers.add_parser('gif', help='保存為 GIF')
    gif_parser.add_argument('--index', type=int, default=0)
    gif_parser.add_argument('--output', type=str, default=None)
    gif_parser.add_argument('--fps', type=int, default=5)
    
    # batch 子命令
    batch_parser = subparsers.add_parser('batch', help='批量生成 GIF')
    batch_parser.add_argument('--output_dir', type=str, default='video_gifs')
    batch_parser.add_argument('--split', type=str, default=None)
    batch_parser.add_argument('--max', type=int, default=None)
    batch_parser.add_argument('--fps', type=int, default=5)
    
    # grid 子命令
    grid_parser = subparsers.add_parser('grid', help='網格預覽')
    grid_parser.add_argument('--num', type=int, default=9)
    grid_parser.add_argument('--split', type=str, default='train')
    grid_parser.add_argument('--output', type=str, default=None)
    
    # list 子命令
    list_parser = subparsers.add_parser('list', help='列出所有樣本')
    
    args = parser.parse_args()
    
    viz = VideoVisualizer(args.npz_dir)
    
    if args.command == 'play':
        if args.path:
            viz.play_interactive(args.path)
        else:
            viz.play_interactive(args.index)
    
    elif args.command == 'animate':
        viz.show_sample_animation(args.index, args.interval)
    
    elif args.command == 'gif':
        viz.save_gif(args.index, args.output, fps=args.fps)
    
    elif args.command == 'batch':
        viz.batch_save_gifs(args.output_dir, args.split, args.max, args.fps)
    
    elif args.command == 'grid':
        viz.preview_grid(args.num, args.split, args.output)
    
    elif args.command == 'list':
        print(f"\n📁 NPZ 目錄: {viz.npz_dir}")
        print(f"📊 總樣本數: {len(viz.samples)}\n")
        
        for i, sample in enumerate(viz.samples[:20]):
            video = viz.load_video(sample)
            print(f"  [{i:3d}] {sample.name}")
            print(f"        Patient: {video['patient_id']} | "
                  f"Frames: {len(video['frames'])} | "
                  f"Diameter: {video['diameter_mm']:.1f}mm")
        
        if len(viz.samples) > 20:
            print(f"\n  ... 還有 {len(viz.samples) - 20} 個樣本")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
