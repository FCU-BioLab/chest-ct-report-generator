#!/usr/bin/env python3
"""
LNDb 資料集專用檢視器
=====================

支援 LNDb (Lung Nodule Database) 格式的 CT 影像瀏覽
- 2D 切片瀏覽（軸向、冠狀、矢狀）
- 3D 多平面重建
- **專家標註的分割遮罩顯示**（非圓形，真實分割）
- 多位放射科醫師標註比較
- 結節資訊顯示

LNDb 資料集特點：
- 294 個 CT 掃描
- 每個掃描由 3 位放射科醫師獨立標註
- 提供真實的分割遮罩（.mhd 格式）
- 標註包含結節位置、體積、惡性度評分

使用方式:
    python lndb_viewer.py
    python lndb_viewer.py --path /path/to/LNDb
    python lndb_viewer.py --lndb_id 1  # 直接載入 LNDb-0001
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
import pandas as pd
import SimpleITK as sitk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import ListedColormap
from scipy import ndimage

# 設定中文字型
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft JhengHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def find_config():
    """尋找 config.json 設定檔"""
    for up in range(4):
        candidate = Path(__file__).parent
        for _ in range(up):
            candidate = candidate.parent
        test_path = candidate / 'config.json'
        if test_path.exists():
            return test_path
    return None


def get_lndb_path():
    """取得 LNDb 資料集路徑"""
    # 直接使用指定的路徑
    return Path(r'E:\lung_ct_lesion_dataset\LNDb')


class LNDbDataLoader:
    """LNDb 資料載入器"""
    
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.nodules_df = None
        self.nodules_gt_df = None
        self.cts_df = None
        self.scan_files = []
        self.mask_files = {}  # {lndb_id: {rad_id: mask_path}}
        self.lung_mask_files = {}  # {lndb_id: lung_mask_path}
        
        self._load_annotations()
        self._find_scans()
        self._find_masks()
        self._find_lung_masks()
    
    def _load_annotations(self):
        """載入標註資料"""
        csv_dir = self.base_path / 'trainset_csv'
        
        # 載入結節標註
        nodules_path = csv_dir / 'trainNodules.csv'
        if nodules_path.exists():
            self.nodules_df = pd.read_csv(nodules_path)
            print(f"載入 {len(self.nodules_df)} 個結節標註 (trainNodules.csv)")
        
        # 載入 GT 標註（融合多位醫師的結果）
        nodules_gt_path = csv_dir / 'trainNodules_gt.csv'
        if nodules_gt_path.exists():
            self.nodules_gt_df = pd.read_csv(nodules_gt_path)
            print(f"載入 {len(self.nodules_gt_df)} 個融合標註 (trainNodules_gt.csv)")
        
        # 載入 CT 資訊
        cts_path = csv_dir / 'trainCTs.csv'
        if cts_path.exists():
            self.cts_df = pd.read_csv(cts_path)
            print(f"載入 {len(self.cts_df)} 個 CT 資訊")
    
    def _find_scans(self):
        """尋找所有 CT 掃描檔案"""
        self.scan_files = []
        self.scan_dict = {}  # {lndb_id: mhd_path}
        
        # 搜尋 data0 ~ data5
        for i in range(6):
            data_dir = self.base_path / f'data{i}'
            if data_dir.exists():
                for mhd_file in sorted(data_dir.glob('LNDb-*.mhd')):
                    self.scan_files.append(mhd_file)
                    # 解析 LNDb ID
                    # 檔名格式: LNDb-0001.mhd
                    lndb_id = int(mhd_file.stem.split('-')[1])
                    self.scan_dict[lndb_id] = mhd_file
        
        print(f"找到 {len(self.scan_files)} 個 CT 掃描")
    
    def _find_masks(self):
        """尋找所有分割遮罩檔案"""
        self.mask_files = {}
        
        # 遮罩在 mask/masks 資料夾
        mask_dir = self.base_path / 'mask' / 'masks'
        if not mask_dir.exists():
            mask_dir = self.base_path / 'mask'
        
        if mask_dir.exists():
            for mhd_file in sorted(mask_dir.glob('LNDb-*_rad*.mhd')):
                # 檔名格式: LNDb-0001_rad1.mhd
                parts = mhd_file.stem.split('_')
                lndb_id = int(parts[0].split('-')[1])
                rad_id = int(parts[1].replace('rad', ''))
                
                if lndb_id not in self.mask_files:
                    self.mask_files[lndb_id] = {}
                self.mask_files[lndb_id][rad_id] = mhd_file
        
        total_masks = sum(len(v) for v in self.mask_files.values())
        print(f"找到 {total_masks} 個分割遮罩 ({len(self.mask_files)} 個 CT)")
    
    def _find_lung_masks(self):
        """尋找所有肺部分割遮罩檔案"""
        self.lung_mask_files = {}
        
        # 肺部遮罩在 lung_masks 資料夾
        lung_mask_dir = self.base_path / 'lung_masks'
        
        if lung_mask_dir.exists():
            for mhd_file in sorted(lung_mask_dir.glob('LNDb-*_lung.mhd')):
                # 檔名格式: LNDb-0001_lung.mhd
                lndb_id = int(mhd_file.stem.split('-')[1].split('_')[0])
                self.lung_mask_files[lndb_id] = mhd_file
        
        print(f"找到 {len(self.lung_mask_files)} 個肺部遮罩")
    
    def get_scan_list(self):
        """取得所有掃描的 LNDb ID 清單"""
        return [f.stem for f in self.scan_files]
    
    def get_lndb_ids(self):
        """取得所有 LNDb ID 清單（整數）"""
        return sorted(self.scan_dict.keys())
    
    def load_scan(self, lndb_id):
        """
        載入指定的 CT 掃描
        
        Parameters:
        -----------
        lndb_id : int or str
            LNDb ID (如 1 或 'LNDb-0001')
        
        Returns:
        --------
        dict : 包含 volume, origin, spacing, direction, lndb_id
        """
        # 解析 lndb_id
        if isinstance(lndb_id, str):
            if lndb_id.startswith('LNDb-'):
                lndb_id = int(lndb_id.split('-')[1])
            else:
                lndb_id = int(lndb_id)
        
        if lndb_id not in self.scan_dict:
            raise FileNotFoundError(f"找不到 LNDb ID: {lndb_id}")
        
        mhd_file = self.scan_dict[lndb_id]
        
        # 使用 SimpleITK 載入
        itk_img = sitk.ReadImage(str(mhd_file))
        
        # 轉換為 numpy array
        volume = sitk.GetArrayFromImage(itk_img)  # shape: (Z, Y, X)
        
        # 取得空間資訊
        origin = np.array(itk_img.GetOrigin())      # (X, Y, Z)
        spacing = np.array(itk_img.GetSpacing())    # (X, Y, Z)
        direction = np.array(itk_img.GetDirection()).reshape(3, 3)
        
        return {
            'volume': volume,
            'origin': origin,
            'spacing': spacing,
            'direction': direction,
            'lndb_id': lndb_id,
            'filename': mhd_file.stem
        }
    
    def load_masks(self, lndb_id):
        """
        載入指定 CT 的所有分割遮罩
        
        Parameters:
        -----------
        lndb_id : int
            LNDb ID
        
        Returns:
        --------
        dict : {rad_id: mask_volume}
        """
        if isinstance(lndb_id, str):
            if lndb_id.startswith('LNDb-'):
                lndb_id = int(lndb_id.split('-')[1])
            else:
                lndb_id = int(lndb_id)
        
        if lndb_id not in self.mask_files:
            return {}
        
        masks = {}
        for rad_id, mask_path in self.mask_files[lndb_id].items():
            try:
                itk_mask = sitk.ReadImage(str(mask_path))
                mask_volume = sitk.GetArrayFromImage(itk_mask)
                masks[rad_id] = mask_volume
            except Exception as e:
                print(f"載入遮罩失敗 {mask_path}: {e}")
        
        return masks
    
    def load_lung_mask(self, lndb_id):
        """
        載入指定 CT 的肺部分割遮罩
        
        Parameters:
        -----------
        lndb_id : int
            LNDb ID
        
        Returns:
        --------
        numpy.ndarray or None : 肺部遮罩 (1=右肺, 2=左肺)
        """
        if isinstance(lndb_id, str):
            if lndb_id.startswith('LNDb-'):
                lndb_id = int(lndb_id.split('-')[1])
            else:
                lndb_id = int(lndb_id)
        
        if lndb_id not in self.lung_mask_files:
            return None
        
        try:
            mask_path = self.lung_mask_files[lndb_id]
            itk_mask = sitk.ReadImage(str(mask_path))
            lung_mask = sitk.GetArrayFromImage(itk_mask)
            return lung_mask
        except Exception as e:
            print(f"載入肺部遮罩失敗 {self.lung_mask_files[lndb_id]}: {e}")
            return None
    
    def get_nodules_for_scan(self, lndb_id):
        """取得指定掃描的結節標註"""
        if isinstance(lndb_id, str):
            if lndb_id.startswith('LNDb-'):
                lndb_id = int(lndb_id.split('-')[1])
            else:
                lndb_id = int(lndb_id)
        
        if self.nodules_df is None:
            return []
        
        nodules = self.nodules_df[self.nodules_df['LNDbID'] == lndb_id]
        return nodules.to_dict('records')
    
    def get_nodules_gt_for_scan(self, lndb_id):
        """取得指定掃描的融合標註（GT）"""
        if isinstance(lndb_id, str):
            if lndb_id.startswith('LNDb-'):
                lndb_id = int(lndb_id.split('-')[1])
            else:
                lndb_id = int(lndb_id)
        
        if self.nodules_gt_df is None:
            return []
        
        nodules = self.nodules_gt_df[self.nodules_gt_df['LNDbID'] == lndb_id]
        return nodules.to_dict('records')
    
    def world_to_voxel(self, world_coord, origin, spacing):
        """將世界座標轉換為體素座標"""
        voxel_coord = (world_coord - origin) / spacing
        return voxel_coord.astype(int)
    
    def get_nodule_info_text(self, nodule, is_gt=False):
        """生成結節資訊文字"""
        if is_gt:
            text = f"FindingID: {nodule.get('FindingID', 'N/A')}\n"
            text += f"位置: ({nodule['x']:.1f}, {nodule['y']:.1f}, {nodule['z']:.1f})\n"
            text += f"體積: {nodule.get('Volume', 0):.1f} mm³\n"
            text += f"AgrLevel: {nodule.get('AgrLevel', 'N/A')}\n"
            text += f"惡性度: {nodule.get('Text', 'N/A')}"
        else:
            text = f"RadID: {nodule.get('RadID', 'N/A')}, FindingID: {nodule.get('FindingID', 'N/A')}\n"
            text += f"位置: ({nodule['x']:.1f}, {nodule['y']:.1f}, {nodule['z']:.1f})\n"
            text += f"體積: {nodule.get('Volume', 0):.1f} mm³\n"
            text += f"惡性度: {nodule.get('Text', 'N/A')}"
        return text


class LNDbViewer:
    """LNDb 互動式檢視器"""
    
    def __init__(self, data_loader):
        self.loader = data_loader
        self.current_scan = None
        self.current_nodules = []
        self.current_nodules_gt = []
        self.current_masks = {}  # {rad_id: mask_volume}
        self.current_lung_mask = None  # 肺部遮罩
        
        # 視窗設定
        self.window_center = -600  # 肺窗
        self.window_width = 1500
        
        # 顯示選項
        self.show_rad_id = 1  # 預設顯示第一位醫師的標註
        
        self._create_gui()
    
    def _create_gui(self):
        """建立 GUI 介面"""
        self.root = tk.Tk()
        self.root.title('LNDb CT 影像檢視器 - 專家分割遮罩')
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
        tk.Button(preset_frame, text="骨窗", command=lambda: self._set_window(400, 1800),
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
        tk.Checkbutton(control_frame, text="顯示分割遮罩", variable=self.show_mask_var,
                      command=self._update_display).pack(anchor=tk.W)
        
        self.show_crosshair_var = tk.BooleanVar(value=True)
        tk.Checkbutton(control_frame, text="顯示十字線", variable=self.show_crosshair_var,
                      command=self._update_display).pack(anchor=tk.W)
        
        self.show_nodule_center_var = tk.BooleanVar(value=True)
        tk.Checkbutton(control_frame, text="顯示結節中心點", variable=self.show_nodule_center_var,
                      command=self._update_display).pack(anchor=tk.W)
        
        self.show_lung_mask_var = tk.BooleanVar(value=True)
        tk.Checkbutton(control_frame, text="顯示肺部遮罩 (藍色)", variable=self.show_lung_mask_var,
                      command=self._update_display).pack(anchor=tk.W)
        
        # 醫師選擇
        rad_frame = tk.Frame(control_frame)
        rad_frame.pack(fill=tk.X, pady=5)
        tk.Label(rad_frame, text="放射科醫師:").pack(side=tk.LEFT)
        self.rad_var = tk.StringVar(value='1')
        self.rad_combo = ttk.Combobox(rad_frame, textvariable=self.rad_var, 
                                       values=['1', '2', '3', '所有', '融合'],
                                       width=10, state='readonly')
        self.rad_combo.pack(side=tk.LEFT, padx=5)
        self.rad_combo.bind('<<ComboboxSelected>>', self._on_rad_change)
        
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
        
        # === 結節資訊 ===
        tk.Label(control_frame, text="結節列表 (GT):", font=('Arial', 10, 'bold')).pack(pady=5)
        
        self.nodule_listbox = tk.Listbox(control_frame, height=6, font=('Courier', 9))
        self.nodule_listbox.pack(fill=tk.X, pady=5)
        self.nodule_listbox.bind('<<ListboxSelect>>', self._on_nodule_selected)
        
        # 跳至結節按鈕
        tk.Button(control_frame, text="跳至選定結節", command=self._goto_nodule,
                 bg='#2196F3', fg='white').pack(fill=tk.X, pady=5)
        
        # 結節詳細資訊
        self.nodule_info_label = tk.Label(control_frame, text="", justify=tk.LEFT, 
                                          font=('Courier', 9), anchor='w', wraplength=300)
        self.nodule_info_label.pack(fill=tk.X, pady=5)
        
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
        self.ax_3d.set_title('3D 結節位置')
        
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
                lndb_id = self.current_scan['lndb_id']
                
                # 載入結節標註
                self.current_nodules = self.loader.get_nodules_for_scan(lndb_id)
                self.current_nodules_gt = self.loader.get_nodules_gt_for_scan(lndb_id)
                
                # 載入分割遮罩
                self.current_masks = self.loader.load_masks(lndb_id)
                
                # 載入肺部遮罩
                self.current_lung_mask = self.loader.load_lung_mask(lndb_id)
                
                self.root.after(0, self._on_scan_loaded)
            except Exception as e:
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
        
        # 更新結節列表（使用 GT）
        self.nodule_listbox.delete(0, tk.END)
        for i, nodule in enumerate(self.current_nodules_gt):
            volume_mm3 = nodule.get('Volume', 0)
            diameter_mm = (6 * volume_mm3 / np.pi) ** (1/3)  # 等效球體直徑
            text_score = nodule.get('Text', 'N/A')
            self.nodule_listbox.insert(tk.END, 
                f"#{nodule.get('FindingID', i+1):2d}: {diameter_mm:5.1f}mm, 惡性:{text_score}")
        
        # 更新資訊
        spacing = self.current_scan['spacing']
        lung_mask_status = "有" if self.current_lung_mask is not None else "無"
        info_text = f"LNDb ID: {self.current_scan['lndb_id']}\n"
        info_text += f"體積大小: {volume.shape}\n"
        info_text += f"像素間距: {spacing[0]:.2f}x{spacing[1]:.2f}x{spacing[2]:.2f} mm\n"
        info_text += f"結節數量: {len(self.current_nodules_gt)} (GT)\n"
        info_text += f"遮罩數量: {len(self.current_masks)} 位醫師\n"
        info_text += f"肺部遮罩: {lung_mask_status}\n"
        info_text += f"HU範圍: [{volume.min():.0f}, {volume.max():.0f}]"
        self.info_label.config(text=info_text)
        
        # 更新醫師選擇下拉選單
        available_rads = ['融合'] + [str(r) for r in sorted(self.current_masks.keys())]
        if len(self.current_masks) > 1:
            available_rads.append('所有')
        self.rad_combo['values'] = available_rads
        if available_rads:
            self.rad_combo.current(0)
        
        self.status_var.set(f"已載入: LNDb-{self.current_scan['lndb_id']:04d} "
                           f"({len(self.current_nodules_gt)} 個結節, "
                           f"{len(self.current_masks)} 位醫師標註)")
        
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
    
    def _on_rad_change(self, event=None):
        """醫師選擇改變時"""
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
    
    def _on_nodule_selected(self, event=None):
        """結節選擇改變時"""
        selection = self.nodule_listbox.curselection()
        if selection and self.current_nodules_gt:
            idx = selection[0]
            if idx < len(self.current_nodules_gt):
                nodule = self.current_nodules_gt[idx]
                info_text = self.loader.get_nodule_info_text(nodule, is_gt=True)
                self.nodule_info_label.config(text=info_text)
    
    def _goto_nodule(self):
        """跳至選定的結節位置"""
        selection = self.nodule_listbox.curselection()
        if not selection:
            messagebox.showinfo("提示", "請先選擇一個結節")
            return
        
        idx = selection[0]
        if idx < len(self.current_nodules_gt):
            nodule = self.current_nodules_gt[idx]
            origin = self.current_scan['origin']
            spacing = self.current_scan['spacing']
            
            # 世界座標轉體素座標
            world_coord = np.array([nodule['x'], nodule['y'], nodule['z']])
            voxel_coord = self.loader.world_to_voxel(world_coord, origin, spacing)
            
            # 設定切片位置
            self.x_var.set(int(np.clip(voxel_coord[0], 0, self.current_scan['volume'].shape[2] - 1)))
            self.y_var.set(int(np.clip(voxel_coord[1], 0, self.current_scan['volume'].shape[1] - 1)))
            self.z_var.set(int(np.clip(voxel_coord[2], 0, self.current_scan['volume'].shape[0] - 1)))
            
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
    
    def _get_combined_mask(self):
        """
        取得合併的遮罩
        
        根據選擇的醫師返回對應的遮罩
        """
        if not self.current_masks:
            return None
        
        rad_choice = self.rad_var.get()
        
        if rad_choice == '融合':
            # 取所有醫師遮罩的交集或聯集（這裡用聯集）
            combined = None
            for mask in self.current_masks.values():
                if combined is None:
                    combined = mask > 0
                else:
                    combined = combined | (mask > 0)
            return combined.astype(np.int32) if combined is not None else None
        
        elif rad_choice == '所有':
            # 返回所有醫師的遮罩（用不同數值標記）
            combined = np.zeros_like(list(self.current_masks.values())[0], dtype=np.int32)
            for i, (rad_id, mask) in enumerate(sorted(self.current_masks.items())):
                combined[mask > 0] = rad_id
            return combined
        
        else:
            # 返回指定醫師的遮罩
            rad_id = int(rad_choice)
            if rad_id in self.current_masks:
                return self.current_masks[rad_id]
            return None
    
    def _update_display(self):
        """更新所有顯示"""
        if self.current_scan is None:
            return
        
        volume = self.current_scan['volume']
        z_idx = self.z_var.get()
        y_idx = self.y_var.get()
        x_idx = self.x_var.get()
        
        origin = self.current_scan['origin']
        spacing = self.current_scan['spacing']
        
        # 取得遮罩
        mask_volume = self._get_combined_mask()
        
        # 清除所有圖
        for ax in [self.ax_axial, self.ax_coronal, self.ax_sagittal]:
            ax.clear()
        self.ax_3d.clear()
        
        alpha = self.alpha_var.get()
        
        # 創建顏色映射
        rad_choice = self.rad_var.get()
        if rad_choice == '所有':
            # 多醫師：不同顏色
            colors = ['none', 'red', 'blue', 'green', 'orange']
            cmap = ListedColormap(colors[:len(self.current_masks) + 1])
        else:
            # 單一醫師或融合：紅色
            cmap = ListedColormap(['none', 'red'])
        
        # 肺部遮罩顏色映射 (1=右肺:藍色, 2=左肺:青色)
        lung_cmap = ListedColormap(['none', '#0066CC', '#00CCCC'])
        
        # === 軸向切片 (Axial) - Z 平面 ===
        axial_slice = volume[z_idx, :, :]
        axial_slice = np.flipud(axial_slice)  # 垂直翻轉
        axial_windowed = self._apply_window(axial_slice)
        self.ax_axial.imshow(axial_windowed, cmap='gray', origin='lower')
        
        if self.show_mask_var.get() and mask_volume is not None:
            mask_slice = mask_volume[z_idx, :, :]
            mask_slice = np.flipud(mask_slice)  # 垂直翻轉
            if np.any(mask_slice > 0):
                self.ax_axial.imshow(mask_slice, cmap=cmap, alpha=alpha, 
                                     origin='lower', vmin=0, 
                                     vmax=max(3, mask_slice.max()))
        
        # 肺部遮罩 (藍色)
        if self.show_lung_mask_var.get() and self.current_lung_mask is not None:
            lung_slice = self.current_lung_mask[z_idx, :, :]
            lung_slice = np.flipud(lung_slice)
            if np.any(lung_slice > 0):
                self.ax_axial.imshow(lung_slice, cmap=lung_cmap, alpha=alpha * 0.5, 
                                     origin='lower', vmin=0, vmax=2)
        
        self.ax_axial.set_title(f'軸向 (Axial) - Z={z_idx}')
        self.ax_axial.axis('off')
        
        # === 冠狀切片 (Coronal) - Y 平面 ===
        coronal_slice = volume[:, y_idx, :]
        coronal_windowed = self._apply_window(coronal_slice)
        self.ax_coronal.imshow(coronal_windowed, cmap='gray', aspect='auto', origin='lower')
        
        if self.show_mask_var.get() and mask_volume is not None:
            mask_slice = mask_volume[:, y_idx, :]
            if np.any(mask_slice > 0):
                self.ax_coronal.imshow(mask_slice, cmap=cmap, alpha=alpha, 
                                       aspect='auto', origin='lower', vmin=0,
                                       vmax=max(3, mask_slice.max()))
        
        # 肺部遮罩 (藍色)
        if self.show_lung_mask_var.get() and self.current_lung_mask is not None:
            lung_slice = self.current_lung_mask[:, y_idx, :]
            if np.any(lung_slice > 0):
                self.ax_coronal.imshow(lung_slice, cmap=lung_cmap, alpha=alpha * 0.5, 
                                       aspect='auto', origin='lower', vmin=0, vmax=2)
        
        self.ax_coronal.set_title(f'冠狀 (Coronal) - Y={y_idx}')
        self.ax_coronal.axis('off')
        
        # === 矢狀切片 (Sagittal) - X 平面 ===
        sagittal_slice = volume[:, :, x_idx]
        sagittal_windowed = self._apply_window(sagittal_slice)
        self.ax_sagittal.imshow(sagittal_windowed, cmap='gray', aspect='auto', origin='lower')
        
        if self.show_mask_var.get() and mask_volume is not None:
            mask_slice = mask_volume[:, :, x_idx]
            if np.any(mask_slice > 0):
                self.ax_sagittal.imshow(mask_slice, cmap=cmap, alpha=alpha, 
                                        aspect='auto', origin='lower', vmin=0,
                                        vmax=max(3, mask_slice.max()))
        
        # 肺部遮罩 (藍色)
        if self.show_lung_mask_var.get() and self.current_lung_mask is not None:
            lung_slice = self.current_lung_mask[:, :, x_idx]
            if np.any(lung_slice > 0):
                self.ax_sagittal.imshow(lung_slice, cmap=lung_cmap, alpha=alpha * 0.5, 
                                        aspect='auto', origin='lower', vmin=0, vmax=2)
        
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
        
        # 顯示結節中心點
        if self.show_nodule_center_var.get():
            self._draw_nodule_centers(z_idx, y_idx, x_idx, origin, spacing)
        
        # 3D 結節視圖
        self._draw_3d_nodules(origin, spacing)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def _draw_nodule_centers(self, z_idx, y_idx, x_idx, origin, spacing):
        """繪製結節中心點"""
        for nodule in self.current_nodules_gt:
            world_coord = np.array([nodule['x'], nodule['y'], nodule['z']])
            voxel_coord = self.loader.world_to_voxel(world_coord, origin, spacing)
            vx, vy, vz = voxel_coord
            
            # 計算等效直徑（用於決定標記大小）
            volume_mm3 = nodule.get('Volume', 100)
            diameter_voxel = (6 * volume_mm3 / np.pi) ** (1/3) / spacing[0]
            
            # 在每個視圖中標記結節中心
            tolerance = max(diameter_voxel / 2, 3)
            
            # 軸向視圖 (座標需要翻轉以配合垂直翻轉)
            if abs(vz - z_idx) < tolerance:
                shape = self.current_scan['volume'].shape
                ax_vy = shape[1] - 1 - vy  # 只翻轉 Y
                self.ax_axial.plot(vx, ax_vy, 'c+', markersize=10, markeredgewidth=2)
                self.ax_axial.annotate(f"#{nodule.get('FindingID', '?')}", 
                                       (vx, ax_vy), color='cyan', fontsize=8,
                                       xytext=(5, 5), textcoords='offset points')
            
            # 冠狀視圖
            if abs(vy - y_idx) < tolerance:
                self.ax_coronal.plot(vx, vz, 'c+', markersize=10, markeredgewidth=2)
            
            # 矢狀視圖
            if abs(vx - x_idx) < tolerance:
                self.ax_sagittal.plot(vy, vz, 'c+', markersize=10, markeredgewidth=2)
    
    def _draw_3d_nodules(self, origin, spacing):
        """繪製 3D 結節位置圖"""
        if not self.current_nodules_gt:
            self.ax_3d.set_title('3D 結節位置 (無結節)')
            return
        
        # 繪製所有結節
        xs, ys, zs, sizes, colors = [], [], [], [], []
        for nodule in self.current_nodules_gt:
            xs.append(nodule['x'])
            ys.append(nodule['y'])
            zs.append(nodule['z'])
            
            volume_mm3 = nodule.get('Volume', 100)
            diameter = (6 * volume_mm3 / np.pi) ** (1/3)
            sizes.append(max(diameter * 5, 30))
            
            # 根據惡性度著色
            text_score = nodule.get('Text', 3)
            if text_score >= 4:
                colors.append('red')
            elif text_score >= 3:
                colors.append('orange')
            else:
                colors.append('green')
        
        # 繪製散點
        self.ax_3d.scatter(xs, ys, zs, c=colors, s=sizes, alpha=0.6, 
                          edgecolors='darkred', linewidth=1)
        
        # 繪製當前位置
        volume = self.current_scan['volume']
        x_world = origin[0] + self.x_var.get() * spacing[0]
        y_world = origin[1] + self.y_var.get() * spacing[1]
        z_world = origin[2] + self.z_var.get() * spacing[2]
        
        self.ax_3d.scatter([x_world], [y_world], [z_world], 
                          c='blue', s=100, marker='^', label='當前位置')
        
        self.ax_3d.set_xlabel('X (mm)')
        self.ax_3d.set_ylabel('Y (mm)')
        self.ax_3d.set_zlabel('Z (mm)')
        self.ax_3d.set_title(f'3D 結節位置 (共 {len(self.current_nodules_gt)} 個)')
        
        # 添加圖例說明顏色
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                   markersize=10, label='高風險 (>=4)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                   markersize=10, label='中風險 (3)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                   markersize=10, label='低風險 (<3)')
        ]
        self.ax_3d.legend(handles=legend_elements, loc='upper left', fontsize=8)
    
    def run(self):
        """執行檢視器"""
        self.root.mainloop()


def main():
    """主函數"""
    print("=" * 60)
    print("LNDb CT 影像檢視器 - 專家分割遮罩")
    print("=" * 60)
    
    parser = argparse.ArgumentParser(description='LNDb CT 影像檢視器')
    parser.add_argument('--path', type=str, help='LNDb 資料集路徑')
    parser.add_argument('--lndb_id', type=int, help='直接載入指定的 LNDb ID')
    args = parser.parse_args()
    
    # 取得資料集路徑
    if args.path:
        data_path = Path(args.path)
    else:
        data_path = get_lndb_path()
    
    if not data_path.exists():
        print(f"錯誤: 找不到 LNDb 資料集路徑: {data_path}")
        print("請使用 --path 參數指定正確的路徑")
        sys.exit(1)
    
    print(f"LNDb 資料集路徑: {data_path}")
    
    # 建立資料載入器
    loader = LNDbDataLoader(data_path)
    
    if not loader.scan_files:
        print("錯誤: 找不到任何 CT 掃描檔案")
        print("請確認 data0 ~ data5 資料夾中包含 .mhd 檔案")
        sys.exit(1)
    
    # 建立並執行檢視器
    viewer = LNDbViewer(loader)
    
    # 如果指定了 lndb_id，自動載入
    if args.lndb_id:
        lndb_name = f"LNDb-{args.lndb_id:04d}"
        if lndb_name in loader.get_scan_list():
            viewer.scan_var.set(lndb_name)
            viewer.root.after(100, viewer._load_selected_scan)
        else:
            print(f"警告: 找不到 LNDb ID {args.lndb_id}")
    
    viewer.run()


if __name__ == "__main__":
    main()
