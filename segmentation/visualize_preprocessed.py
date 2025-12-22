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
    """切片資料載入器"""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.patients = []
        self.patient_meta = {}
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
        
        print(f"找到 {len(self.patients)} 個病人")
    
    def get_patient_list(self):
        """取得病人列表"""
        return self.patients
    
    def get_meta(self, patient_id: str):
        """取得病人 meta"""
        return self.patient_meta.get(patient_id, {})
    
    def load_slice(self, patient_id: str, slice_idx: int):
        """載入切片"""
        slice_path = self.cache_dir / patient_id / f"slice_{slice_idx:04d}.npz"
        if not slice_path.exists():
            return None
        
        data = np.load(slice_path, allow_pickle=True)
        return {
            'image': data['image'].astype(np.float32),
            'mask': data['mask'].astype(np.float32),
            'lung_mask': data['lung_mask']
        }


class SliceViewer:
    """切片互動式檢視器"""
    
    def __init__(self, data_loader: SliceDataLoader, initial_patient: str = None):
        self.loader = data_loader
        self.current_patient = None
        self.current_slice = None
        self.meta = None
        
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
        self.root.title('預處理切片檢視器 - Lungmask + 2.5D Pipeline')
        self.root.geometry('1200x800')
        
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
        self.fig = plt.figure(figsize=(12, 8))
        
        # 2x2 佈局
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
        
        # 更新切片滑桿
        num_slices = self.meta.get('num_slices', 1)
        self.z_scale.configure(to=num_slices - 1)
        self.z_var.set(num_slices // 2)
        
        # 更新正樣本列表
        self.positive_listbox.delete(0, tk.END)
        positive = self.meta.get('positive_slices', [])
        for z in positive:
            self.positive_listbox.insert(tk.END, f"Slice {z:04d}")
        
        # 更新統計
        stats_text = f"病人: {patient_id}\n"
        stats_text += f"切片數: {num_slices}\n"
        stats_text += f"正樣本: {len(positive)} slices\n"
        stats_text += f"Spacing: {self.meta.get('spacing', 'N/A')}"
        self.stats_label.config(text=stats_text)
        
        self.status_var.set(f"已載入: {patient_id} ({num_slices} slices, {len(positive)} positive)")
        
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
        self.z_var.set(z)
        self._update_display()
    
    def _on_mousewheel(self, event):
        """滑鼠滾輪"""
        if not self.current_patient:
            return
        
        delta = -1 if event.delta > 0 else 1
        new_z = self.z_var.get() + delta
        num_slices = self.meta.get('num_slices', 1)
        new_z = max(0, min(new_z, num_slices - 1))
        self.z_var.set(new_z)
        self._update_display()
    
    def _update_display(self):
        """更新顯示"""
        if not self.current_patient:
            return
        
        z = self.z_var.get()
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
        
        alpha = self.alpha_var.get()
        
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
                        default='C:/GitHub/chest-ct-report-generator/segmentation/cache/msd_lung_slices',
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
