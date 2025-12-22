#!/usr/bin/env python3
"""
MSD Lung Tumours 資料集專用檢視器
================================

支援 Medical Segmentation Decathlon Task06 (Lung Tumours) 格式的 CT 影像瀏覽
- 2D 切片瀏覽（軸向、冠狀、矢狀）
- 3D 多平面重建
- 腫瘤分割遮罩顯示
- 腫瘤資訊統計

MSD Lung Tumours 資料集特點：
- 96 個 3D CT 掃描 (64 training + 32 testing)
- NIfTI 格式 (.nii.gz)
- 標註為肺部腫瘤分割遮罩
- 來源: The Cancer Imaging Archive

使用方式:
    python msd_lung_viewer.py
    python msd_lung_viewer.py --path /path/to/MSD_Lung_Tumours
    python msd_lung_viewer.py --case_id 1  # 直接載入 lung_001
"""

import argparse
import json
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import ListedColormap
from scipy import ndimage

# 設定中文字型
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft JhengHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def get_msd_lung_path():
    """取得 MSD Lung Tumours 資料集路徑"""
    return Path(r'E:\lung_ct_lesion_dataset\MSD Lung Tumours\Task06_Lung')


class MSDLungDataLoader:
    """MSD Lung Tumours 資料載入器"""
    
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.dataset_info = None
        self.scan_files = []  # [(case_id, image_path, label_path or None)]
        self.scan_dict = {}  # {case_id: (image_path, label_path or None)}
        
        self._load_dataset_info()
        self._find_scans()
    
    def _load_dataset_info(self):
        """載入資料集資訊 (dataset.json)"""
        json_path = self.base_path / 'dataset.json'
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                self.dataset_info = json.load(f)
            print(f"載入資料集資訊: {self.dataset_info.get('name', 'MSD Lung')}")
            print(f"  - 訓練樣本: {self.dataset_info.get('numTraining', 'N/A')}")
            print(f"  - 測試樣本: {self.dataset_info.get('numTest', 'N/A')}")
            print(f"  - 模態: {self.dataset_info.get('modality', 'N/A')}")
        else:
            print("警告: 找不到 dataset.json")
    
    def _find_scans(self):
        """尋找所有 CT 掃描檔案"""
        self.scan_files = []
        self.scan_dict = {}
        
        # 搜尋 imagesTr 資料夾（訓練影像）
        images_tr_dir = self.base_path / 'imagesTr'
        labels_tr_dir = self.base_path / 'labelsTr'
        
        if images_tr_dir.exists():
            for nii_file in sorted(images_tr_dir.glob('*.nii.gz')):
                # 跳過 macOS 元資料檔案 (以 ._ 開頭)
                if nii_file.name.startswith('._'):
                    continue
                    
                # 檔名格式: lung_001.nii.gz
                case_name = nii_file.stem.replace('.nii', '')  # 移除 .nii
                case_id = case_name  # e.g., 'lung_001'
                
                # 尋找對應的標註
                label_path = labels_tr_dir / f'{case_name}.nii.gz'
                if not label_path.exists():
                    label_path = None
                
                self.scan_files.append((case_id, nii_file, label_path))
                self.scan_dict[case_id] = (nii_file, label_path)
        
        # 搜尋 imagesTs 資料夾（測試影像，無標註）
        images_ts_dir = self.base_path / 'imagesTs'
        
        if images_ts_dir.exists():
            for nii_file in sorted(images_ts_dir.glob('*.nii.gz')):
                # 跳過 macOS 元資料檔案 (以 ._ 開頭)
                if nii_file.name.startswith('._'):
                    continue
                    
                case_name = nii_file.stem.replace('.nii', '')
                case_id = f'{case_name}_test'  # 標記為測試樣本
                
                self.scan_files.append((case_id, nii_file, None))
                self.scan_dict[case_id] = (nii_file, None)
        
        print(f"找到 {len(self.scan_files)} 個 CT 掃描")
        
        # 統計有標註的數量
        with_label = sum(1 for _, _, label in self.scan_files if label is not None)
        print(f"  - 有標註: {with_label}")
        print(f"  - 無標註: {len(self.scan_files) - with_label}")
    
    def get_scan_list(self):
        """取得所有掃描的 case ID 清單"""
        return [case_id for case_id, _, _ in self.scan_files]
    
    def load_scan(self, case_id):
        """
        載入指定的 CT 掃描
        
        Parameters:
        -----------
        case_id : str
            Case ID (如 'lung_001')
        
        Returns:
        --------
        dict : 包含 volume, affine, header, case_id, has_label
        """
        if case_id not in self.scan_dict:
            raise FileNotFoundError(f"找不到 Case ID: {case_id}")
        
        image_path, label_path = self.scan_dict[case_id]
        
        # 使用 nibabel 載入 NIfTI
        nii_img = nib.load(str(image_path))
        
        # 取得體積資料
        volume = nii_img.get_fdata()  # shape: (X, Y, Z) 或 (X, Y, Z, 1)
        
        # 如果有第四維度，移除它
        if volume.ndim == 4:
            volume = volume[:, :, :, 0]
        
        # 轉置為 (Z, Y, X) 以配合顯示
        volume = np.transpose(volume, (2, 1, 0))
        
        # 取得空間資訊
        affine = nii_img.affine
        header = nii_img.header
        
        # 計算 spacing
        spacing = header.get_zooms()[:3]  # (X, Y, Z)
        
        return {
            'volume': volume,
            'affine': affine,
            'header': header,
            'spacing': np.array(spacing),
            'case_id': case_id,
            'has_label': label_path is not None,
            'filename': image_path.name
        }
    
    def load_label(self, case_id):
        """
        載入指定 CT 的腫瘤分割遮罩
        
        Parameters:
        -----------
        case_id : str
            Case ID
        
        Returns:
        --------
        numpy.ndarray or None : 腫瘤遮罩
        """
        if case_id not in self.scan_dict:
            return None
        
        _, label_path = self.scan_dict[case_id]
        
        if label_path is None or not label_path.exists():
            return None
        
        try:
            nii_label = nib.load(str(label_path))
            label = nii_label.get_fdata()
            
            # 如果有第四維度，移除它
            if label.ndim == 4:
                label = label[:, :, :, 0]
            
            # 轉置為 (Z, Y, X)
            label = np.transpose(label, (2, 1, 0))
            
            return label.astype(np.int32)
        except Exception as e:
            print(f"載入標註失敗 {label_path}: {e}")
            return None
    
    def get_tumor_statistics(self, label_volume, spacing):
        """
        計算腫瘤統計資訊
        
        Parameters:
        -----------
        label_volume : numpy.ndarray
            腫瘤遮罩
        spacing : numpy.ndarray
            像素間距 (X, Y, Z)
        
        Returns:
        --------
        dict : 包含腫瘤統計資訊
        """
        if label_volume is None:
            return None
        
        # 計算體積 (mm³)
        voxel_volume = spacing[0] * spacing[1] * spacing[2]
        tumor_voxels = np.sum(label_volume > 0)
        tumor_volume_mm3 = tumor_voxels * voxel_volume
        
        # 尋找連通區域
        labeled_array, num_tumors = ndimage.label(label_volume > 0)
        
        # 計算每個腫瘤的資訊
        tumors = []
        for i in range(1, num_tumors + 1):
            tumor_mask = labeled_array == i
            voxels = np.sum(tumor_mask)
            volume_mm3 = voxels * voxel_volume
            
            # 計算等效球體直徑
            diameter_mm = (6 * volume_mm3 / np.pi) ** (1/3)
            
            # 找到腫瘤中心
            coords = np.where(tumor_mask)
            center_z = np.mean(coords[0])
            center_y = np.mean(coords[1])
            center_x = np.mean(coords[2])
            
            # 計算邊界框
            z_min, z_max = coords[0].min(), coords[0].max()
            y_min, y_max = coords[1].min(), coords[1].max()
            x_min, x_max = coords[2].min(), coords[2].max()
            
            tumors.append({
                'id': i,
                'voxels': voxels,
                'volume_mm3': volume_mm3,
                'diameter_mm': diameter_mm,
                'center': (center_z, center_y, center_x),
                'bbox': ((z_min, z_max), (y_min, y_max), (x_min, x_max))
            })
        
        return {
            'total_volume_mm3': tumor_volume_mm3,
            'total_voxels': tumor_voxels,
            'num_tumors': num_tumors,
            'tumors': sorted(tumors, key=lambda x: x['volume_mm3'], reverse=True)
        }


class MSDLungViewer:
    """MSD Lung Tumours 互動式檢視器"""
    
    def __init__(self, data_loader):
        self.loader = data_loader
        self.current_scan = None
        self.current_label = None
        self.current_tumor_stats = None
        
        # 視窗設定
        self.window_center = -600  # 肺窗
        self.window_width = 1500
        
        self._create_gui()
    
    def _create_gui(self):
        """建立 GUI 介面"""
        self.root = tk.Tk()
        self.root.title('MSD Lung Tumours CT 影像檢視器')
        self.root.geometry('1500x950')
        
        # 主框架
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左側控制面板
        control_frame = tk.Frame(main_frame, width=320)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        control_frame.pack_propagate(False)
        
        # === 掃描選擇 ===
        tk.Label(control_frame, text="選擇 CT 掃描:", font=('Arial', 10, 'bold')).pack(pady=5)
        
        scan_frame = tk.Frame(control_frame)
        scan_frame.pack(fill=tk.X, pady=5)
        
        self.scan_var = tk.StringVar()
        self.scan_combo = ttk.Combobox(scan_frame, textvariable=self.scan_var, 
                                        width=38, state='readonly')
        scan_list = self.loader.get_scan_list()
        self.scan_combo['values'] = scan_list
        if scan_list:
            self.scan_combo.current(0)
        self.scan_combo.pack(fill=tk.X)
        
        # 載入按鈕
        tk.Button(control_frame, text="載入掃描", command=self._load_selected_scan,
                 bg='#4CAF50', fg='white', font=('Arial', 10)).pack(pady=10, fill=tk.X)
        
        # 分隔線
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # === 切片控制 ===
        tk.Label(control_frame, text="切片控制:", font=('Arial', 10, 'bold')).pack(pady=5)
        
        # 軸向切片 (Z)
        slice_frame = tk.Frame(control_frame)
        slice_frame.pack(fill=tk.X, pady=5)
        tk.Label(slice_frame, text="軸向 (Z):").pack(side=tk.LEFT)
        self.z_var = tk.IntVar(value=0)
        self.z_scale = tk.Scale(slice_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                variable=self.z_var, command=self._on_slice_change)
        self.z_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 冠狀切片 (Y)
        coronal_frame = tk.Frame(control_frame)
        coronal_frame.pack(fill=tk.X, pady=5)
        tk.Label(coronal_frame, text="冠狀 (Y):").pack(side=tk.LEFT)
        self.y_var = tk.IntVar(value=0)
        self.y_scale = tk.Scale(coronal_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                variable=self.y_var, command=self._on_slice_change)
        self.y_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 矢狀切片 (X)
        sagittal_frame = tk.Frame(control_frame)
        sagittal_frame.pack(fill=tk.X, pady=5)
        tk.Label(sagittal_frame, text="矢狀 (X):").pack(side=tk.LEFT)
        self.x_var = tk.IntVar(value=0)
        self.x_scale = tk.Scale(sagittal_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                variable=self.x_var, command=self._on_slice_change)
        self.x_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 分隔線
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # === 窗寬窗位控制 ===
        tk.Label(control_frame, text="窗寬窗位:", font=('Arial', 10, 'bold')).pack(pady=5)
        
        # 預設值按鈕
        preset_frame = tk.Frame(control_frame)
        preset_frame.pack(fill=tk.X, pady=5)
        tk.Button(preset_frame, text="肺窗", command=lambda: self._set_window(-600, 1500),
                 width=8).pack(side=tk.LEFT, padx=2)
        tk.Button(preset_frame, text="縱隔窗", command=lambda: self._set_window(40, 400),
                 width=8).pack(side=tk.LEFT, padx=2)
        tk.Button(preset_frame, text="軟組織", command=lambda: self._set_window(50, 350),
                 width=8).pack(side=tk.LEFT, padx=2)
        
        # 窗位
        wl_frame = tk.Frame(control_frame)
        wl_frame.pack(fill=tk.X, pady=5)
        tk.Label(wl_frame, text="窗位 (WL):").pack(side=tk.LEFT)
        self.wl_var = tk.IntVar(value=self.window_center)
        self.wl_scale = tk.Scale(wl_frame, from_=-1024, to=1024, orient=tk.HORIZONTAL,
                                 variable=self.wl_var, command=self._on_window_change)
        self.wl_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 窗寬
        ww_frame = tk.Frame(control_frame)
        ww_frame.pack(fill=tk.X, pady=5)
        tk.Label(ww_frame, text="窗寬 (WW):").pack(side=tk.LEFT)
        self.ww_var = tk.IntVar(value=self.window_width)
        self.ww_scale = tk.Scale(ww_frame, from_=1, to=4000, orient=tk.HORIZONTAL,
                                 variable=self.ww_var, command=self._on_window_change)
        self.ww_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 分隔線
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # === 顯示選項 ===
        tk.Label(control_frame, text="顯示選項:", font=('Arial', 10, 'bold')).pack(pady=5)
        
        self.show_mask_var = tk.BooleanVar(value=True)
        tk.Checkbutton(control_frame, text="顯示腫瘤遮罩 (紅色)", variable=self.show_mask_var,
                      command=self._update_display).pack(anchor=tk.W)
        
        self.show_crosshair_var = tk.BooleanVar(value=True)
        tk.Checkbutton(control_frame, text="顯示十字線", variable=self.show_crosshair_var,
                      command=self._update_display).pack(anchor=tk.W)
        
        self.show_tumor_center_var = tk.BooleanVar(value=True)
        tk.Checkbutton(control_frame, text="顯示腫瘤中心點", variable=self.show_tumor_center_var,
                      command=self._update_display).pack(anchor=tk.W)
        
        # 遮罩透明度
        alpha_frame = tk.Frame(control_frame)
        alpha_frame.pack(fill=tk.X, pady=5)
        tk.Label(alpha_frame, text="遮罩透明度:").pack(side=tk.LEFT)
        self.alpha_var = tk.DoubleVar(value=0.4)
        self.alpha_scale = tk.Scale(alpha_frame, from_=0.1, to=1.0, resolution=0.1,
                                    orient=tk.HORIZONTAL, variable=self.alpha_var,
                                    command=self._on_alpha_change)
        self.alpha_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 分隔線
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # === 腫瘤資訊 ===
        tk.Label(control_frame, text="腫瘤列表:", font=('Arial', 10, 'bold')).pack(pady=5)
        
        self.tumor_listbox = tk.Listbox(control_frame, height=6, font=('Courier', 9))
        self.tumor_listbox.pack(fill=tk.X, pady=5)
        self.tumor_listbox.bind('<<ListboxSelect>>', self._on_tumor_selected)
        
        # 跳至腫瘤按鈕
        tk.Button(control_frame, text="跳至選定腫瘤", command=self._goto_tumor,
                 bg='#2196F3', fg='white').pack(fill=tk.X, pady=5)
        
        # 腫瘤詳細資訊
        self.tumor_info_label = tk.Label(control_frame, text="", justify=tk.LEFT, 
                                          font=('Courier', 9), anchor='w', wraplength=300)
        self.tumor_info_label.pack(fill=tk.X, pady=5)
        
        # 掃描資訊
        self.info_label = tk.Label(control_frame, text="", justify=tk.LEFT, 
                                   font=('Courier', 9), anchor='w')
        self.info_label.pack(fill=tk.X, pady=10)
        
        # === 右側影像顯示區 ===
        image_frame = tk.Frame(main_frame)
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 建立 matplotlib 圖形
        self.fig = plt.figure(figsize=(13, 10))
        
        # 2x2 佈局
        self.ax_axial = self.fig.add_subplot(221)
        self.ax_coronal = self.fig.add_subplot(222)
        self.ax_sagittal = self.fig.add_subplot(223)
        self.ax_3d = self.fig.add_subplot(224, projection='3d')
        
        self.ax_axial.set_title('軸向 (Axial)')
        self.ax_coronal.set_title('冠狀 (Coronal)')
        self.ax_sagittal.set_title('矢狀 (Sagittal)')
        self.ax_3d.set_title('3D 腫瘤位置')
        
        plt.tight_layout()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=image_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 綁定滑鼠滾輪
        self.canvas.get_tk_widget().bind('<MouseWheel>', self._on_mousewheel)
        
        # 狀態列
        self.status_var = tk.StringVar(value="請選擇並載入 CT 掃描")
        status_bar = tk.Label(self.root, textvariable=self.status_var, 
                             bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _load_selected_scan(self):
        """載入選定的掃描"""
        scan_name = self.scan_var.get()
        if not scan_name:
            messagebox.showwarning("警告", "請先選擇一個 CT 掃描")
            return
        
        self.status_var.set(f"正在載入 {scan_name}...")
        self.root.update()
        
        def load_thread():
            try:
                self.current_scan = self.loader.load_scan(scan_name)
                
                # 載入腫瘤標註
                self.current_label = self.loader.load_label(scan_name)
                
                # 計算腫瘤統計
                if self.current_label is not None:
                    self.current_tumor_stats = self.loader.get_tumor_statistics(
                        self.current_label, self.current_scan['spacing'])
                else:
                    self.current_tumor_stats = None
                
                self.root.after(0, self._on_scan_loaded)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: messagebox.showerror("錯誤", f"載入失敗: {e}"))
                self.root.after(0, lambda: self.status_var.set("載入失敗"))
        
        thread = threading.Thread(target=load_thread)
        thread.daemon = True
        thread.start()
    
    def _on_scan_loaded(self):
        """掃描載入完成後"""
        volume = self.current_scan['volume']
        
        # 更新切片滑桿範圍
        self.z_scale.configure(to=volume.shape[0] - 1)
        self.y_scale.configure(to=volume.shape[1] - 1)
        self.x_scale.configure(to=volume.shape[2] - 1)
        
        # 設定到中間切片
        self.z_var.set(volume.shape[0] // 2)
        self.y_var.set(volume.shape[1] // 2)
        self.x_var.set(volume.shape[2] // 2)
        
        # 更新腫瘤列表
        self.tumor_listbox.delete(0, tk.END)
        if self.current_tumor_stats and self.current_tumor_stats['tumors']:
            for tumor in self.current_tumor_stats['tumors']:
                self.tumor_listbox.insert(tk.END, 
                    f"#{tumor['id']:2d}: {tumor['diameter_mm']:5.1f}mm, "
                    f"{tumor['volume_mm3']:8.1f}mm³")
        
        # 更新資訊
        spacing = self.current_scan['spacing']
        has_label = "有" if self.current_label is not None else "無"
        
        info_text = f"Case ID: {self.current_scan['case_id']}\n"
        info_text += f"體積大小: {volume.shape}\n"
        info_text += f"像素間距: {spacing[0]:.2f}x{spacing[1]:.2f}x{spacing[2]:.2f} mm\n"
        info_text += f"腫瘤標註: {has_label}\n"
        
        if self.current_tumor_stats:
            info_text += f"腫瘤數量: {self.current_tumor_stats['num_tumors']}\n"
            info_text += f"總體積: {self.current_tumor_stats['total_volume_mm3']:.1f} mm³\n"
        
        info_text += f"HU範圍: [{volume.min():.0f}, {volume.max():.0f}]"
        self.info_label.config(text=info_text)
        
        num_tumors = self.current_tumor_stats['num_tumors'] if self.current_tumor_stats else 0
        self.status_var.set(f"已載入: {self.current_scan['case_id']} "
                           f"({num_tumors} 個腫瘤)")
        
        self._update_display()
    
    def _on_slice_change(self, event=None):
        """切片改變時更新顯示"""
        if self.current_scan is not None:
            self._update_display()
    
    def _on_window_change(self, event=None):
        """窗寬窗位改變時更新顯示"""
        self.window_center = self.wl_var.get()
        self.window_width = self.ww_var.get()
        if self.current_scan is not None:
            self._update_display()
    
    def _on_alpha_change(self, event=None):
        """透明度改變時"""
        if self.current_scan is not None:
            self._update_display()
    
    def _set_window(self, wl, ww):
        """設定窗寬窗位"""
        self.wl_var.set(wl)
        self.ww_var.set(ww)
        self._on_window_change()
    
    def _apply_window(self, image):
        """套用窗寬窗位"""
        wl = self.window_center
        ww = self.window_width
        
        img_min = wl - ww / 2
        img_max = wl + ww / 2
        
        windowed = np.clip(image, img_min, img_max)
        windowed = (windowed - img_min) / (img_max - img_min)
        return windowed
    
    def _on_tumor_selected(self, event=None):
        """腫瘤選擇改變時"""
        selection = self.tumor_listbox.curselection()
        if selection and self.current_tumor_stats:
            idx = selection[0]
            tumors = self.current_tumor_stats['tumors']
            if idx < len(tumors):
                tumor = tumors[idx]
                info_text = f"腫瘤 #{tumor['id']}\n"
                info_text += f"體積: {tumor['volume_mm3']:.1f} mm³\n"
                info_text += f"直徑: {tumor['diameter_mm']:.1f} mm\n"
                info_text += f"體素數: {tumor['voxels']}\n"
                center = tumor['center']
                info_text += f"中心: ({center[2]:.0f}, {center[1]:.0f}, {center[0]:.0f})"
                self.tumor_info_label.config(text=info_text)
    
    def _goto_tumor(self):
        """跳至選定的腫瘤位置"""
        selection = self.tumor_listbox.curselection()
        if not selection:
            messagebox.showinfo("提示", "請先選擇一個腫瘤")
            return
        
        if not self.current_tumor_stats:
            return
        
        idx = selection[0]
        tumors = self.current_tumor_stats['tumors']
        if idx < len(tumors):
            tumor = tumors[idx]
            center = tumor['center']
            
            # 設定切片位置
            volume = self.current_scan['volume']
            self.z_var.set(int(np.clip(center[0], 0, volume.shape[0] - 1)))
            self.y_var.set(int(np.clip(center[1], 0, volume.shape[1] - 1)))
            self.x_var.set(int(np.clip(center[2], 0, volume.shape[2] - 1)))
            
            self._update_display()
    
    def _on_mousewheel(self, event):
        """滑鼠滾輪控制切片"""
        if self.current_scan is None:
            return
        
        delta = -1 if event.delta > 0 else 1
        new_z = self.z_var.get() + delta
        new_z = max(0, min(new_z, self.current_scan['volume'].shape[0] - 1))
        self.z_var.set(new_z)
        self._update_display()
    
    def _update_display(self):
        """更新所有顯示"""
        if self.current_scan is None:
            return
        
        volume = self.current_scan['volume']
        z_idx = self.z_var.get()
        y_idx = self.y_var.get()
        x_idx = self.x_var.get()
        
        # 清除所有圖
        for ax in [self.ax_axial, self.ax_coronal, self.ax_sagittal]:
            ax.clear()
        self.ax_3d.clear()
        
        alpha = self.alpha_var.get()
        
        # 腫瘤遮罩顏色映射
        tumor_cmap = ListedColormap(['none', 'red'])
        
        # === 軸向切片 (Axial) - Z 平面 ===
        axial_slice = volume[z_idx, :, :]
        axial_slice = np.flipud(axial_slice)  # 垂直翻轉
        axial_windowed = self._apply_window(axial_slice)
        self.ax_axial.imshow(axial_windowed, cmap='gray', origin='lower')
        
        if self.show_mask_var.get() and self.current_label is not None:
            mask_slice = self.current_label[z_idx, :, :]
            mask_slice = np.flipud(mask_slice)  # 垂直翻轉
            if np.any(mask_slice > 0):
                self.ax_axial.imshow(mask_slice, cmap=tumor_cmap, alpha=alpha, 
                                     origin='lower', vmin=0, vmax=1)
        
        self.ax_axial.set_title(f'軸向 (Axial) - Z={z_idx}')
        self.ax_axial.axis('off')
        
        # === 冠狀切片 (Coronal) - Y 平面 ===
        coronal_slice = volume[:, y_idx, :]
        coronal_windowed = self._apply_window(coronal_slice)
        self.ax_coronal.imshow(coronal_windowed, cmap='gray', aspect='auto', origin='lower')
        
        if self.show_mask_var.get() and self.current_label is not None:
            mask_slice = self.current_label[:, y_idx, :]
            if np.any(mask_slice > 0):
                self.ax_coronal.imshow(mask_slice, cmap=tumor_cmap, alpha=alpha, 
                                       aspect='auto', origin='lower', vmin=0, vmax=1)
        
        self.ax_coronal.set_title(f'冠狀 (Coronal) - Y={y_idx}')
        self.ax_coronal.axis('off')
        
        # === 矢狀切片 (Sagittal) - X 平面 ===
        sagittal_slice = volume[:, :, x_idx]
        sagittal_windowed = self._apply_window(sagittal_slice)
        self.ax_sagittal.imshow(sagittal_windowed, cmap='gray', aspect='auto', origin='lower')
        
        if self.show_mask_var.get() and self.current_label is not None:
            mask_slice = self.current_label[:, :, x_idx]
            if np.any(mask_slice > 0):
                self.ax_sagittal.imshow(mask_slice, cmap=tumor_cmap, alpha=alpha, 
                                        aspect='auto', origin='lower', vmin=0, vmax=1)
        
        self.ax_sagittal.set_title(f'矢狀 (Sagittal) - X={x_idx}')
        self.ax_sagittal.axis('off')
        
        # 顯示十字線
        if self.show_crosshair_var.get():
            # 軸向 (座標需要翻轉以配合垂直翻轉)
            axial_h = volume.shape[1]
            self.ax_axial.axhline(y=axial_h - 1 - y_idx, color='yellow', linewidth=0.5, alpha=0.7)
            self.ax_axial.axvline(x=x_idx, color='yellow', linewidth=0.5, alpha=0.7)
            # 冠狀
            self.ax_coronal.axhline(y=z_idx, color='yellow', linewidth=0.5, alpha=0.7)
            self.ax_coronal.axvline(x=x_idx, color='yellow', linewidth=0.5, alpha=0.7)
            # 矢狀
            self.ax_sagittal.axhline(y=z_idx, color='yellow', linewidth=0.5, alpha=0.7)
            self.ax_sagittal.axvline(x=y_idx, color='yellow', linewidth=0.5, alpha=0.7)
        
        # 顯示腫瘤中心點
        if self.show_tumor_center_var.get():
            self._draw_tumor_centers(z_idx, y_idx, x_idx)
        
        # 3D 腫瘤視圖
        self._draw_3d_tumors()
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def _draw_tumor_centers(self, z_idx, y_idx, x_idx):
        """繪製腫瘤中心點"""
        if not self.current_tumor_stats or not self.current_tumor_stats['tumors']:
            return
        
        volume = self.current_scan['volume']
        
        for tumor in self.current_tumor_stats['tumors']:
            center = tumor['center']
            vz, vy, vx = center
            
            # 計算顯示容許範圍（基於腫瘤大小）
            diameter = tumor['diameter_mm']
            spacing = self.current_scan['spacing']
            tolerance = max(diameter / 2 / spacing[2], 3)
            
            # 軸向視圖 (座標需要翻轉以配合垂直翻轉)
            if abs(vz - z_idx) < tolerance:
                display_vy = volume.shape[1] - 1 - vy
                self.ax_axial.plot(vx, display_vy, 'c+', markersize=10, markeredgewidth=2)
                self.ax_axial.annotate(f"#{tumor['id']}", 
                                       (vx, display_vy), color='cyan', fontsize=8,
                                       xytext=(5, 5), textcoords='offset points')
            
            # 冠狀視圖
            if abs(vy - y_idx) < tolerance:
                self.ax_coronal.plot(vx, vz, 'c+', markersize=10, markeredgewidth=2)
            
            # 矢狀視圖
            if abs(vx - x_idx) < tolerance:
                self.ax_sagittal.plot(vy, vz, 'c+', markersize=10, markeredgewidth=2)
    
    def _draw_3d_tumors(self):
        """繪製 3D 腫瘤位置圖"""
        if not self.current_tumor_stats or not self.current_tumor_stats['tumors']:
            self.ax_3d.set_title('3D 腫瘤位置 (無腫瘤)')
            return
        
        spacing = self.current_scan['spacing']
        tumors = self.current_tumor_stats['tumors']
        
        # 繪製所有腫瘤
        xs, ys, zs, sizes = [], [], [], []
        for tumor in tumors:
            center = tumor['center']
            xs.append(center[2] * spacing[0])  # X
            ys.append(center[1] * spacing[1])  # Y
            zs.append(center[0] * spacing[2])  # Z
            
            # 根據體積決定標記大小
            diameter = tumor['diameter_mm']
            sizes.append(max(diameter * 5, 30))
        
        # 繪製散點
        self.ax_3d.scatter(xs, ys, zs, c='red', s=sizes, alpha=0.6, 
                          edgecolors='darkred', linewidth=1)
        
        # 繪製當前位置
        volume = self.current_scan['volume']
        x_world = self.x_var.get() * spacing[0]
        y_world = self.y_var.get() * spacing[1]
        z_world = self.z_var.get() * spacing[2]
        
        self.ax_3d.scatter([x_world], [y_world], [z_world], 
                          c='blue', s=100, marker='^', label='當前位置')
        
        self.ax_3d.set_xlabel('X (mm)')
        self.ax_3d.set_ylabel('Y (mm)')
        self.ax_3d.set_zlabel('Z (mm)')
        self.ax_3d.set_title(f'3D 腫瘤位置 (共 {len(tumors)} 個)')
    
    def run(self):
        """執行檢視器"""
        self.root.mainloop()


def main():
    """主函數"""
    print("=" * 60)
    print("MSD Lung Tumours CT 影像檢視器")
    print("=" * 60)
    
    parser = argparse.ArgumentParser(description='MSD Lung Tumours CT 影像檢視器')
    parser.add_argument('--path', type=str, help='MSD Lung Tumours 資料集路徑')
    parser.add_argument('--case_id', type=str, help='直接載入指定的 Case ID (如 lung_001)')
    args = parser.parse_args()
    
    # 取得資料集路徑
    if args.path:
        data_path = Path(args.path)
    else:
        data_path = get_msd_lung_path()
    
    if not data_path.exists():
        print(f"錯誤: 找不到 MSD Lung Tumours 資料集路徑: {data_path}")
        print("請使用 --path 參數指定正確的路徑")
        sys.exit(1)
    
    print(f"MSD Lung Tumours 資料集路徑: {data_path}")
    
    # 建立資料載入器
    loader = MSDLungDataLoader(data_path)
    
    if not loader.scan_files:
        print("錯誤: 找不到任何 CT 掃描檔案")
        print("請確認 imagesTr 或 imagesTs 資料夾中包含 .nii.gz 檔案")
        sys.exit(1)
    
    # 建立並執行檢視器
    viewer = MSDLungViewer(loader)
    
    # 如果指定了 case_id，自動載入
    if args.case_id:
        if args.case_id in loader.get_scan_list():
            viewer.scan_var.set(args.case_id)
            viewer.root.after(100, viewer._load_selected_scan)
        else:
            print(f"警告: 找不到 Case ID {args.case_id}")
    
    viewer.run()


if __name__ == "__main__":
    main()
