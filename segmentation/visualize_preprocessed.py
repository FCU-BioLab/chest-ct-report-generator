#!/usr/bin/env python3
"""
預處理切片檢視器
================

互動式 GUI 檢視預處理後的 slice npz 檔案
- 切片瀏覽
- Image/Mask/Lung Mask 疊加顯示
- 病人切換
- 跳至有結節的切片

使用方式:
    python visualize_preprocessed.py
    python visualize_preprocessed.py --patient LNDb-0001
"""

import argparse
import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 設定中文字型
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft JhengHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class SliceDataLoader:
    """切片資料載入器（支援 slice 和 4-patch 格式）"""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.patients = []
        self.patient_meta = {}
        self.mode = 'slice'  # 'slice' or '4patch'
        self._find_patients()
    
    def _find_patients(self):
        """尋找所有病人"""
        self.patients = []
        for patient_dir in sorted(self.cache_dir.iterdir()):
            if patient_dir.is_dir() and (patient_dir / "meta.json").exists():
                self.patients.append(patient_dir.name)
                
                # 載入 meta
                with open(patient_dir / "meta.json", 'r') as f:
                    self.patient_meta[patient_dir.name] = json.load(f)
        
        # 檢測模式
        if self.patients:
            meta = self.patient_meta[self.patients[0]]
            preprocessing = meta.get('preprocessing', {})
            if preprocessing.get('mode') == '4-patch':
                self.mode = '4patch'
            else:
                self.mode = 'slice'
        
        print(f"找到 {len(self.patients)} 個病人 (模式: {self.mode})")
    
    def get_patient_list(self):
        """取得病人列表"""
        return self.patients
    
    def get_meta(self, patient_id: str):
        """取得病人 meta"""
        return self.patient_meta.get(patient_id, {})
    
    def get_available_slices(self, patient_id: str):
        """取得可用的切片索引列表"""
        patient_dir = self.cache_dir / patient_id
        if self.mode == '4patch':
            # 從 patch 檔案名稱解析切片索引
            slices = set()
            for f in patient_dir.glob('slice_*_patch_*.npz'):
                # slice_0044_patch_0.npz -> 44
                parts = f.stem.split('_')
                if len(parts) >= 2:
                    slices.add(int(parts[1]))
            return sorted(slices)
        else:
            # 從 slice 檔案名稱解析
            slices = []
            for f in patient_dir.glob('slice_*.npz'):
                # slice_0044.npz -> 44
                parts = f.stem.split('_')
                if len(parts) == 2:
                    slices.append(int(parts[1]))
            return sorted(slices)
    
    def load_slice(self, patient_id: str, slice_idx: int):
        """載入切片（slice 模式）"""
        slice_path = self.cache_dir / patient_id / f"slice_{slice_idx:04d}.npz"
        if not slice_path.exists():
            return None
        
        data = np.load(slice_path, allow_pickle=True)
        return {
            'image': data['image'].astype(np.float32),
            'mask': data['mask'].astype(np.float32),
            'lung_mask': data['lung_mask']
        }
    
    def load_patches(self, patient_id: str, slice_idx: int):
        """載入 4-patch（4patch 模式）"""
        patient_dir = self.cache_dir / patient_id
        patches = []
        
        for patch_idx in range(4):
            patch_path = patient_dir / f"slice_{slice_idx:04d}_patch_{patch_idx}.npz"
            if patch_path.exists():
                data = np.load(patch_path, allow_pickle=True)
                patches.append({
                    'image': data['image'].astype(np.float32),
                    'mask': data['mask'].astype(np.float32),
                    'lung_mask': data['lung_mask'],
                    'patch_idx': int(data['patch_idx']),
                    'patch_pos': data['patch_pos']
                })
        
        return patches if patches else None


class SliceViewer:
    """切片互動式檢視器（支援 slice 和 4-patch 格式）"""
    
    def __init__(self, data_loader: SliceDataLoader, initial_patient: str = None):
        self.loader = data_loader
        self.current_patient = None
        self.current_slice = None
        self.meta = None
        self.available_slices = []  # 可用切片索引列表
        
        self._create_gui()
        
        # 載入初始病人
        if initial_patient and initial_patient in self.loader.patients:
            idx = self.loader.patients.index(initial_patient)
            self.patient_combo.current(idx)
            self._load_patient()
        elif self.loader.patients:
            self.patient_combo.current(0)
            self._load_patient()
    
    def _create_gui(self):
        """建立 GUI"""
        self.root = tk.Tk()
        mode_str = "4-Patch" if self.loader.mode == '4patch' else "Slice"
        self.root.title(f'預處理檢視器 - {mode_str} 模式')
        self.root.geometry('1400x900')
        
        # 主框架
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左側控制面板
        control_frame = tk.Frame(main_frame, width=280)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        control_frame.pack_propagate(False)
        
        # === 病人選擇 ===
        tk.Label(control_frame, text="選擇病人:", font=('Arial', 10, 'bold')).pack(pady=5)
        
        self.patient_var = tk.StringVar()
        self.patient_combo = ttk.Combobox(control_frame, textvariable=self.patient_var,
                                          values=self.loader.get_patient_list(),
                                          width=25, state='readonly')
        self.patient_combo.pack(fill=tk.X, pady=5)
        self.patient_combo.bind('<<ComboboxSelected>>', lambda e: self._load_patient())
        
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # === 切片控制 ===
        tk.Label(control_frame, text="切片控制:", font=('Arial', 10, 'bold')).pack(pady=5)
        
        slice_frame = tk.Frame(control_frame)
        slice_frame.pack(fill=tk.X, pady=5)
        tk.Label(slice_frame, text="Slice Z:").pack(side=tk.LEFT)
        self.z_var = tk.IntVar(value=0)
        self.z_scale = tk.Scale(slice_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                variable=self.z_var, command=self._on_slice_change)
        self.z_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 切片資訊
        self.slice_info_var = tk.StringVar(value="Slice: 0 / 0")
        tk.Label(control_frame, textvariable=self.slice_info_var).pack(pady=5)
        
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # === 顯示選項 ===
        tk.Label(control_frame, text="顯示選項:", font=('Arial', 10, 'bold')).pack(pady=5)
        
        self.show_mask_var = tk.BooleanVar(value=True)
        tk.Checkbutton(control_frame, text="顯示結節遮罩 (紅)", variable=self.show_mask_var,
                       command=self._update_display).pack(anchor=tk.W)
        
        self.show_lung_var = tk.BooleanVar(value=True)
        tk.Checkbutton(control_frame, text="顯示肺部遮罩 (藍)", variable=self.show_lung_var,
                       command=self._update_display).pack(anchor=tk.W)
        
        # 透明度
        alpha_frame = tk.Frame(control_frame)
        alpha_frame.pack(fill=tk.X, pady=5)
        tk.Label(alpha_frame, text="遮罩透明度:").pack(side=tk.LEFT)
        self.alpha_var = tk.DoubleVar(value=0.4)
        self.alpha_scale = tk.Scale(alpha_frame, from_=0.1, to=1.0, resolution=0.1,
                                    orient=tk.HORIZONTAL, variable=self.alpha_var,
                                    command=lambda e: self._update_display())
        self.alpha_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # === 正樣本切片列表 ===
        tk.Label(control_frame, text="有結節的切片:", font=('Arial', 10, 'bold')).pack(pady=5)
        
        self.positive_listbox = tk.Listbox(control_frame, height=10, font=('Courier', 9))
        self.positive_listbox.pack(fill=tk.X, pady=5)
        self.positive_listbox.bind('<<ListboxSelect>>', self._on_positive_selected)
        
        tk.Button(control_frame, text="跳至選定切片", command=self._goto_slice,
                  bg='#2196F3', fg='white').pack(fill=tk.X, pady=5)
        
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # === 統計資訊 ===
        self.stats_label = tk.Label(control_frame, text="", justify=tk.LEFT,
                                    font=('Courier', 9), anchor='w')
        self.stats_label.pack(fill=tk.X, pady=5)
        
        # === 右側影像顯示區 ===
        image_frame = tk.Frame(main_frame)
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 建立 matplotlib 圖形
        self.fig = plt.figure(figsize=(14, 10))
        
        # 根據模式選擇佈局
        if self.loader.mode == '4patch':
            # 4-patch 模式: 2x4 佈局 (4 patches, 每個 patch 顯示 image+overlay)
            self.patch_axes = []
            for i in range(4):
                ax_img = self.fig.add_subplot(2, 4, i + 1)
                ax_overlay = self.fig.add_subplot(2, 4, i + 5)
                self.patch_axes.append((ax_img, ax_overlay))
        else:
            # slice 模式: 2x2 佈局
            self.ax_image = self.fig.add_subplot(221)
            self.ax_mask = self.fig.add_subplot(222)
            self.ax_lung = self.fig.add_subplot(223)
            self.ax_overlay = self.fig.add_subplot(224)
            
            self.ax_image.set_title('Image (CT)')
            self.ax_mask.set_title('Mask (Nodule GT)')
            self.ax_lung.set_title('Lung Mask')
            self.ax_overlay.set_title('Overlay')
        
        plt.tight_layout()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=image_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 綁定滑鼠滾輪
        self.canvas.get_tk_widget().bind('<MouseWheel>', self._on_mousewheel)
        
        # 狀態列
        self.status_var = tk.StringVar(value="請選擇病人")
        status_bar = tk.Label(self.root, textvariable=self.status_var,
                              bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _load_patient(self):
        """載入病人"""
        patient_id = self.patient_var.get()
        if not patient_id:
            return
        
        self.current_patient = patient_id
        self.meta = self.loader.get_meta(patient_id)
        
        # 取得可用切片列表
        self.available_slices = self.loader.get_available_slices(patient_id)
        
        if not self.available_slices:
            self.status_var.set(f"無可用切片: {patient_id}")
            return
        
        # 更新切片滑桿（使用索引而非切片號）
        self.z_scale.configure(to=len(self.available_slices) - 1)
        self.z_var.set(len(self.available_slices) // 2)
        
        # 更新正樣本列表
        self.positive_listbox.delete(0, tk.END)
        positive = self.meta.get('positive_slices', [])
        # 只顯示有存檔的正樣本
        for z in positive:
            if z in self.available_slices:
                self.positive_listbox.insert(tk.END, f"Slice {z:04d}")
        
        # 更新統計
        mode = self.meta.get('preprocessing', {}).get('mode', 'slice')
        stats_text = f"病人: {patient_id}\n"
        stats_text += f"模式: {mode}\n"
        stats_text += f"可用切片: {len(self.available_slices)}\n"
        stats_text += f"正樣本: {len(positive)} slices\n"
        if mode == '4-patch':
            stats_text += f"Patches: {self.meta.get('saved_patches', 'N/A')}\n"
        stats_text += f"Spacing: {self.meta.get('spacing', 'N/A')}"
        self.stats_label.config(text=stats_text)
        
        self.status_var.set(f"已載入: {patient_id} ({len(self.available_slices)} slices available)")
        
        self._update_display()
    
    def _on_slice_change(self, event=None):
        """切片改變"""
        if self.current_patient:
            self._update_display()
    
    def _on_positive_selected(self, event=None):
        """選擇正樣本切片"""
        pass  # 雙擊或按按鈕才跳轉
    
    def _goto_slice(self):
        """跳至選定切片"""
        selection = self.positive_listbox.curselection()
        if not selection:
            return
        
        text = self.positive_listbox.get(selection[0])
        # 解析 "Slice 0044" -> 44
        z = int(text.split()[-1])
        
        # 找到對應的索引
        if z in self.available_slices:
            idx = self.available_slices.index(z)
            self.z_var.set(idx)
            self._update_display()
    
    def _on_mousewheel(self, event):
        """滑鼠滾輪"""
        if not self.current_patient or not self.available_slices:
            return
        
        delta = -1 if event.delta > 0 else 1
        new_idx = self.z_var.get() + delta
        new_idx = max(0, min(new_idx, len(self.available_slices) - 1))
        self.z_var.set(new_idx)
        self._update_display()
    
    def _update_display(self):
        """更新顯示"""
        if not self.current_patient or not self.available_slices:
            return
        
        # 從索引取得實際切片號
        idx = self.z_var.get()
        if idx >= len(self.available_slices):
            idx = len(self.available_slices) - 1
        z = self.available_slices[idx]
        
        alpha = self.alpha_var.get()
        
        if self.loader.mode == '4patch':
            self._update_display_4patch(z, alpha)
        else:
            self._update_display_slice(z, alpha)
    
    def _update_display_4patch(self, z: int, alpha: float):
        """更新 4-patch 顯示"""
        patches = self.loader.load_patches(self.current_patient, z)
        
        if not patches:
            self.status_var.set(f"無法載入切片 {z} 的 patches")
            return
        
        # 更新切片資訊
        is_positive = z in self.meta.get('positive_slices', [])
        is_2_5d = self.meta.get('is_2_5d', False)
        format_str = "(2.5D)" if is_2_5d else ""
        self.slice_info_var.set(f"Slice: {z} ({len(patches)} patches) {format_str} {'(+)' if is_positive else ''}")
        
        # 清除並重繪
        patch_names = ['Top-Left', 'Top-Right', 'Bottom-Left', 'Bottom-Right']
        
        for i, (ax_img, ax_overlay) in enumerate(self.patch_axes):
            ax_img.clear()
            ax_overlay.clear()
            
            if i < len(patches):
                patch = patches[i]
                image_raw = patch['image']
                mask = patch['mask']
                lung_mask = patch['lung_mask']
                
                # Handle 2.5D format: (d, H, W) -> extract middle slice
                if image_raw.ndim == 3:
                    # 2.5D: use middle slice (index // 2)
                    mid_idx = image_raw.shape[0] // 2
                    image = image_raw[mid_idx]  # (H, W)
                else:
                    image = image_raw  # Already (H, W)
                
                # Image
                ax_img.imshow(image, cmap='gray')
                mask_area = (mask > 0).sum()
                title = f'{patch_names[i]}\n[{image.min():.2f}, {image.max():.2f}]'
                if mask_area > 0:
                    title += f' ⚫{mask_area}px'
                ax_img.set_title(title, fontsize=9)
                ax_img.axis('off')
                
                # Overlay - for 2.5D, show as pseudo-RGB using middle 3 channels
                if image_raw.ndim == 3 and image_raw.shape[0] >= 3:
                    # Select middle 3 channels for RGB
                    mid_idx = image_raw.shape[0] // 2
                    rgb_indices = [mid_idx-1, mid_idx, mid_idx+1]
                    
                    # Normalize each channel
                    overlay = np.stack([image_raw[k] for k in rgb_indices], axis=-1)  # (H, W, 3)
                    for c in range(3):
                        ch = overlay[:, :, c]
                        overlay[:, :, c] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-6)
                else:
                    overlay = np.stack([image, image, image], axis=-1)
                    overlay = (overlay - overlay.min()) / (overlay.max() - overlay.min() + 1e-6)
                
                if self.show_lung_var.get():
                    lung_overlay = np.zeros_like(overlay)
                    lung_overlay[:, :, 2] = (lung_mask > 0).astype(float) * 0.3
                    overlay = np.clip(overlay + lung_overlay, 0, 1)
                
                if self.show_mask_var.get():
                    nodule_overlay = np.zeros_like(overlay)
                    nodule_overlay[:, :, 0] = (mask > 0).astype(float) * alpha
                    overlay = np.clip(overlay + nodule_overlay, 0, 1)
                
                ax_overlay.imshow(overlay)
                overlay_title = f'Overlay {patch_names[i]}'
                if image_raw.ndim == 3 and image_raw.shape[0] == 3:
                    overlay_title += '\n(R=z-1, G=z, B=z+1)'
                ax_overlay.set_title(overlay_title, fontsize=9)
                ax_overlay.axis('off')
            else:
                ax_img.set_title(f'{patch_names[i]}\n(N/A)', fontsize=9)
                ax_img.axis('off')
                ax_overlay.axis('off')
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def _update_display_slice(self, z: int, alpha: float):
        """更新 slice 顯示"""
        data = self.loader.load_slice(self.current_patient, z)
        
        if data is None:
            self.status_var.set(f"無法載入切片 {z}")
            return
        
        image = data['image']
        mask = data['mask']
        lung_mask = data['lung_mask']
        
        # 更新切片資訊
        is_positive = z in self.meta.get('positive_slices', [])
        self.slice_info_var.set(f"Slice: {z} / {self.meta.get('num_slices', 0) - 1} {'(+)' if is_positive else ''}")
        
        # 清除所有圖
        for ax in [self.ax_image, self.ax_mask, self.ax_lung, self.ax_overlay]:
            ax.clear()
        
        # Image
        self.ax_image.imshow(image, cmap='gray')
        self.ax_image.set_title(f'Image [{image.min():.2f}, {image.max():.2f}]')
        self.ax_image.axis('off')
        
        # Mask
        self.ax_mask.imshow(mask, cmap='Reds', vmin=0, vmax=1)
        mask_area = (mask > 0).sum()
        self.ax_mask.set_title(f'Nodule Mask (Area: {mask_area} px)')
        self.ax_mask.axis('off')
        
        # Lung Mask
        self.ax_lung.imshow(lung_mask, cmap='Blues')
        lung_area = (lung_mask > 0).sum()
        self.ax_lung.set_title(f'Lung Mask (Area: {lung_area} px)')
        self.ax_lung.axis('off')
        
        # Overlay
        overlay = np.stack([image, image, image], axis=-1)
        overlay = (overlay - overlay.min()) / (overlay.max() - overlay.min() + 1e-6)
        
        if self.show_lung_var.get():
            lung_overlay = np.zeros_like(overlay)
            lung_overlay[:, :, 2] = (lung_mask > 0).astype(float) * 0.3
            overlay = np.clip(overlay + lung_overlay, 0, 1)
        
        if self.show_mask_var.get():
            nodule_overlay = np.zeros_like(overlay)
            nodule_overlay[:, :, 0] = (mask > 0).astype(float) * alpha
            overlay = np.clip(overlay + nodule_overlay, 0, 1)
        
        self.ax_overlay.imshow(overlay)
        self.ax_overlay.set_title('Overlay (Blue=Lung, Red=Nodule)')
        self.ax_overlay.axis('off')
        
        self.canvas.draw()
    
    def run(self):
        """執行"""
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description='預處理切片檢視器')
    parser.add_argument('--cache_dir', type=str,
                        # default='C:/GitHub/chest-ct-report-generator/segmentation/cache/msd_lung_slices',
                        default='C:/GitHub/chest-ct-report-generator/segmentation/cache/lndb_patches',
                        help='切片快取目錄')
    parser.add_argument('--patient', type=str, default=None,
                        help='初始載入的病人 ID')
    
    args = parser.parse_args()
    
    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        print(f"Error: Cache directory not found: {cache_dir}")
        sys.exit(1)
    
    loader = SliceDataLoader(str(cache_dir))
    viewer = SliceViewer(loader, initial_patient=args.patient)
    viewer.run()


if __name__ == "__main__":
    main()
