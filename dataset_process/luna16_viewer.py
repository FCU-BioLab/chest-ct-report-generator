# luna16_viewer.py
# LUNA16 資料集專用檢視器 - 支援 MHD/RAW 格式的 CT 影像瀏覽
# 功能：2D 切片瀏覽、3D 多平面重建、結節標註顯示

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Slider
import pandas as pd
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import sys
import threading

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


def get_luna16_path():
    """取得 LUNA16 資料集路徑"""
    config_path = find_config()
    if config_path:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        data_path = config.get('all_patient_data', 'datasets/aLL_patients_data')
        if not Path(data_path).is_absolute():
            data_path = (config_path.parent / data_path).resolve()
        return Path(data_path)
    # 預設路徑
    return Path(__file__).parent.parent / 'datasets' / 'aLL_patients_data'


class LUNA16DataLoader:
    """LUNA16 資料載入器"""
    
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.annotations = None
        self.candidates = None
        self.scan_files = []
        self._load_annotations()
        self._find_scans()
    
    def _load_annotations(self):
        """載入標註資料"""
        ann_path = self.base_path / 'annotations.csv'
        if ann_path.exists():
            self.annotations = pd.read_csv(ann_path)
            print(f"載入 {len(self.annotations)} 個結節標註")
        
        # 嘗試載入 candidates (可能檔案太大)
        cand_path = self.base_path / 'candidates.csv'
        if cand_path.exists():
            try:
                self.candidates = pd.read_csv(cand_path)
                print(f"載入 {len(self.candidates)} 個候選結節")
            except Exception as e:
                print(f"無法載入 candidates.csv: {e}")
    
    def _find_scans(self):
        """尋找所有 CT 掃描檔案"""
        self.scan_files = []
        
        # 搜尋 subset0 ~ subset9
        for i in range(10):
            subset_dir = self.base_path / f'subset{i}'
            if subset_dir.exists():
                mhd_files = list(subset_dir.glob('*.mhd'))
                self.scan_files.extend(mhd_files)
        
        # 也搜尋 seg-lungs-LUNA16 (肺部分割遮罩)
        seg_dir = self.base_path / 'seg-lungs-LUNA16'
        if seg_dir.exists():
            self.seg_files = list(seg_dir.glob('*.mhd'))
        else:
            self.seg_files = []
        
        print(f"找到 {len(self.scan_files)} 個 CT 掃描")
        print(f"找到 {len(self.seg_files)} 個肺部分割遮罩")
    
    def get_scan_list(self):
        """取得所有掃描的 seriesuid 清單"""
        return [f.stem for f in self.scan_files]
    
    def load_scan(self, seriesuid):
        """載入指定的 CT 掃描"""
        # 尋找對應的 mhd 檔案
        mhd_file = None
        for f in self.scan_files:
            if f.stem == seriesuid:
                mhd_file = f
                break
        
        if mhd_file is None:
            raise FileNotFoundError(f"找不到 seriesuid: {seriesuid}")
        
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
            'seriesuid': seriesuid
        }
    
    def load_lung_mask(self, seriesuid):
        """載入肺部分割遮罩"""
        for f in self.seg_files:
            if f.stem == seriesuid:
                itk_img = sitk.ReadImage(str(f))
                return sitk.GetArrayFromImage(itk_img)
        return None
    
    def get_nodules_for_scan(self, seriesuid):
        """取得指定掃描的結節標註"""
        if self.annotations is None:
            return []
        
        nodules = self.annotations[self.annotations['seriesuid'] == seriesuid]
        return nodules.to_dict('records')
    
    def world_to_voxel(self, world_coord, origin, spacing):
        """將世界座標轉換為體素座標"""
        voxel_coord = (world_coord - origin) / spacing
        return voxel_coord.astype(int)


class LUNA16Viewer:
    """LUNA16 互動式檢視器"""
    
    def __init__(self, data_loader):
        self.loader = data_loader
        self.current_scan = None
        self.current_nodules = []
        self.lung_mask = None
        
        # 視窗設定
        self.window_center = -600  # 肺窗
        self.window_width = 1500
        
        self._create_gui()
    
    def _create_gui(self):
        """建立 GUI 介面"""
        self.root = tk.Tk()
        self.root.title('LUNA16 CT 影像檢視器')
        self.root.geometry('1400x900')
        
        # 主框架
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左側控制面板
        control_frame = tk.Frame(main_frame, width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        control_frame.pack_propagate(False)
        
        # 掃描選擇
        tk.Label(control_frame, text="選擇 CT 掃描:", font=('Arial', 10, 'bold')).pack(pady=5)
        
        scan_frame = tk.Frame(control_frame)
        scan_frame.pack(fill=tk.X, pady=5)
        
        self.scan_var = tk.StringVar()
        self.scan_combo = ttk.Combobox(scan_frame, textvariable=self.scan_var, 
                                        width=35, state='readonly')
        scan_list = self.loader.get_scan_list()
        self.scan_combo['values'] = scan_list
        if scan_list:
            self.scan_combo.current(0)
        self.scan_combo.pack(fill=tk.X)
        self.scan_combo.bind('<<ComboboxSelected>>', self._on_scan_selected)
        
        # 載入按鈕
        tk.Button(control_frame, text="載入掃描", command=self._load_selected_scan,
                 bg='#4CAF50', fg='white', font=('Arial', 10)).pack(pady=10, fill=tk.X)
        
        # 分隔線
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # 切片控制
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
        
        # 窗寬窗位控制
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
        
        # 顯示選項
        tk.Label(control_frame, text="顯示選項:", font=('Arial', 10, 'bold')).pack(pady=5)
        
        self.show_nodules_var = tk.BooleanVar(value=True)
        tk.Checkbutton(control_frame, text="顯示結節標註", variable=self.show_nodules_var,
                      command=self._update_display).pack(anchor=tk.W)
        
        self.show_crosshair_var = tk.BooleanVar(value=True)
        tk.Checkbutton(control_frame, text="顯示十字線", variable=self.show_crosshair_var,
                      command=self._update_display).pack(anchor=tk.W)
        
        self.show_lung_mask_var = tk.BooleanVar(value=False)
        tk.Checkbutton(control_frame, text="顯示肺部遮罩", variable=self.show_lung_mask_var,
                      command=self._update_display).pack(anchor=tk.W)
        
        # 分隔線
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # 結節資訊
        tk.Label(control_frame, text="結節資訊:", font=('Arial', 10, 'bold')).pack(pady=5)
        
        self.nodule_listbox = tk.Listbox(control_frame, height=8, font=('Courier', 9))
        self.nodule_listbox.pack(fill=tk.X, pady=5)
        self.nodule_listbox.bind('<<ListboxSelect>>', self._on_nodule_selected)
        
        # 跳至結節按鈕
        tk.Button(control_frame, text="跳至選定結節", command=self._goto_nodule,
                 bg='#2196F3', fg='white').pack(fill=tk.X, pady=5)
        
        # 掃描資訊
        self.info_label = tk.Label(control_frame, text="", justify=tk.LEFT, 
                                   font=('Courier', 9), anchor='w')
        self.info_label.pack(fill=tk.X, pady=10)
        
        # 右側影像顯示區
        image_frame = tk.Frame(main_frame)
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 建立 matplotlib 圖形
        self.fig = plt.figure(figsize=(12, 9))
        
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
    
    def _on_scan_selected(self, event=None):
        """掃描選擇改變時"""
        pass
    
    def _load_selected_scan(self):
        """載入選定的掃描"""
        seriesuid = self.scan_var.get()
        if not seriesuid:
            messagebox.showwarning("警告", "請先選擇一個掃描")
            return
        
        self.status_var.set(f"正在載入 {seriesuid[:30]}...")
        self.root.update()
        
        def load_thread():
            try:
                self.current_scan = self.loader.load_scan(seriesuid)
                self.current_nodules = self.loader.get_nodules_for_scan(seriesuid)
                self.lung_mask = self.loader.load_lung_mask(seriesuid)
                
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
        
        # 更新結節列表
        self.nodule_listbox.delete(0, tk.END)
        for i, nodule in enumerate(self.current_nodules):
            self.nodule_listbox.insert(tk.END, 
                f"#{i+1}: D={nodule['diameter_mm']:.1f}mm")
        
        # 更新資訊
        spacing = self.current_scan['spacing']
        info_text = f"體積大小: {volume.shape}\n"
        info_text += f"像素間距: {spacing[0]:.2f}x{spacing[1]:.2f}x{spacing[2]:.2f} mm\n"
        info_text += f"結節數量: {len(self.current_nodules)}\n"
        info_text += f"HU範圍: [{volume.min():.0f}, {volume.max():.0f}]"
        self.info_label.config(text=info_text)
        
        self.status_var.set(f"已載入: {self.current_scan['seriesuid'][:40]}...")
        
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
        pass
    
    def _goto_nodule(self):
        """跳至選定的結節位置"""
        selection = self.nodule_listbox.curselection()
        if not selection:
            messagebox.showinfo("提示", "請先選擇一個結節")
            return
        
        idx = selection[0]
        if idx < len(self.current_nodules):
            nodule = self.current_nodules[idx]
            
            # 將世界座標轉換為體素座標
            world_coord = np.array([nodule['coordX'], nodule['coordY'], nodule['coordZ']])
            origin = self.current_scan['origin']
            spacing = self.current_scan['spacing']
            
            voxel_coord = self.loader.world_to_voxel(world_coord, origin, spacing)
            
            # 設定切片位置 (注意座標順序)
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
        
        # 清除所有圖
        for ax in [self.ax_axial, self.ax_coronal, self.ax_sagittal]:
            ax.clear()
        self.ax_3d.clear()
        
        # 軸向切片 (Axial) - Z 平面
        axial_slice = volume[z_idx, :, :]
        axial_windowed = self._apply_window(axial_slice)
        self.ax_axial.imshow(axial_windowed, cmap='gray', origin='lower')
        self.ax_axial.set_title(f'軸向 (Axial) - Z={z_idx}')
        self.ax_axial.axis('off')
        
        # 冠狀切片 (Coronal) - Y 平面
        coronal_slice = volume[:, y_idx, :]
        coronal_windowed = self._apply_window(coronal_slice)
        self.ax_coronal.imshow(coronal_windowed, cmap='gray', aspect='auto', origin='lower')
        self.ax_coronal.set_title(f'冠狀 (Coronal) - Y={y_idx}')
        self.ax_coronal.axis('off')
        
        # 矢狀切片 (Sagittal) - X 平面
        sagittal_slice = volume[:, :, x_idx]
        sagittal_windowed = self._apply_window(sagittal_slice)
        self.ax_sagittal.imshow(sagittal_windowed, cmap='gray', aspect='auto', origin='lower')
        self.ax_sagittal.set_title(f'矢狀 (Sagittal) - X={x_idx}')
        self.ax_sagittal.axis('off')
        
        # 顯示十字線
        if self.show_crosshair_var.get():
            # Axial
            self.ax_axial.axhline(y_idx, color='yellow', linestyle='--', alpha=0.5, linewidth=1)
            self.ax_axial.axvline(x_idx, color='cyan', linestyle='--', alpha=0.5, linewidth=1)
            # Coronal
            self.ax_coronal.axhline(z_idx, color='red', linestyle='--', alpha=0.5, linewidth=1)
            self.ax_coronal.axvline(x_idx, color='cyan', linestyle='--', alpha=0.5, linewidth=1)
            # Sagittal
            self.ax_sagittal.axhline(z_idx, color='red', linestyle='--', alpha=0.5, linewidth=1)
            self.ax_sagittal.axvline(y_idx, color='yellow', linestyle='--', alpha=0.5, linewidth=1)
        
        # 顯示肺部遮罩
        if self.show_lung_mask_var.get() and self.lung_mask is not None:
            mask_slice = self.lung_mask[z_idx, :, :]
            self.ax_axial.contour(mask_slice, levels=[0.5], colors='green', linewidths=1, alpha=0.7)
        
        # 顯示結節標註
        if self.show_nodules_var.get():
            self._draw_nodules(z_idx, y_idx, x_idx, origin, spacing)
        
        # 3D 結節視圖
        self._draw_3d_nodules(origin, spacing)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def _draw_nodules(self, z_idx, y_idx, x_idx, origin, spacing):
        """繪製結節標註"""
        for nodule in self.current_nodules:
            # 將世界座標轉換為體素座標
            world_coord = np.array([nodule['coordX'], nodule['coordY'], nodule['coordZ']])
            voxel_coord = self.loader.world_to_voxel(world_coord, origin, spacing)
            
            # 計算體素半徑
            radius_mm = nodule['diameter_mm'] / 2
            radius_voxel = radius_mm / spacing  # (X, Y, Z)
            
            vx, vy, vz = voxel_coord
            rx, ry, rz = radius_voxel
            
            # 在軸向切片上繪製 (如果結節在當前切片附近)
            if abs(vz - z_idx) < rz + 2:
                # 計算在當前切片上的顯示半徑
                if abs(vz - z_idx) < rz:
                    # 球體在當前z的截面半徑
                    dz = abs(vz - z_idx)
                    slice_radius = np.sqrt(max(0, rz**2 - dz**2)) * (rx / rz)
                else:
                    slice_radius = 5
                
                circle = plt.Circle((vx, vy), slice_radius, 
                                    fill=False, color='red', linewidth=2)
                self.ax_axial.add_patch(circle)
                self.ax_axial.plot(vx, vy, 'r+', markersize=10, markeredgewidth=2)
            
            # 在冠狀切片上繪製
            if abs(vy - y_idx) < ry + 2:
                if abs(vy - y_idx) < ry:
                    dy = abs(vy - y_idx)
                    slice_radius = np.sqrt(max(0, ry**2 - dy**2)) * (rx / ry)
                else:
                    slice_radius = 5
                    
                circle = plt.Circle((vx, vz), slice_radius, 
                                    fill=False, color='red', linewidth=2)
                self.ax_coronal.add_patch(circle)
                self.ax_coronal.plot(vx, vz, 'r+', markersize=10, markeredgewidth=2)
            
            # 在矢狀切片上繪製
            if abs(vx - x_idx) < rx + 2:
                if abs(vx - x_idx) < rx:
                    dx = abs(vx - x_idx)
                    slice_radius = np.sqrt(max(0, rx**2 - dx**2)) * (ry / rx)
                else:
                    slice_radius = 5
                    
                circle = plt.Circle((vy, vz), slice_radius, 
                                    fill=False, color='red', linewidth=2)
                self.ax_sagittal.add_patch(circle)
                self.ax_sagittal.plot(vy, vz, 'r+', markersize=10, markeredgewidth=2)
    
    def _draw_3d_nodules(self, origin, spacing):
        """繪製 3D 結節位置圖"""
        if not self.current_nodules:
            self.ax_3d.text(0.5, 0.5, 0.5, '無結節', transform=self.ax_3d.transAxes,
                          ha='center', va='center', fontsize=12)
            return
        
        # 繪製所有結節
        xs, ys, zs, sizes = [], [], [], []
        for nodule in self.current_nodules:
            xs.append(nodule['coordX'])
            ys.append(nodule['coordY'])
            zs.append(nodule['coordZ'])
            sizes.append(nodule['diameter_mm'] * 10)  # 放大顯示
        
        # 繪製散點
        scatter = self.ax_3d.scatter(xs, ys, zs, c='red', s=sizes, alpha=0.6, 
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
        self.ax_3d.set_title(f'3D 結節位置 (共 {len(self.current_nodules)} 個)')
        self.ax_3d.legend()
    
    def run(self):
        """執行檢視器"""
        self.root.mainloop()


def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LUNA16 CT 影像檢視器')
    parser.add_argument('--path', type=str, help='LUNA16 資料集路徑')
    parser.add_argument('--seriesuid', type=str, help='直接載入指定的 seriesuid')
    args = parser.parse_args()
    
    # 取得資料集路徑
    if args.path:
        data_path = Path(args.path)
    else:
        data_path = get_luna16_path()
    
    if not data_path.exists():
        print(f"錯誤: 找不到 LUNA16 資料集路徑: {data_path}")
        print("請使用 --path 參數指定正確的路徑")
        sys.exit(1)
    
    print(f"LUNA16 資料集路徑: {data_path}")
    
    # 建立資料載入器
    loader = LUNA16DataLoader(data_path)
    
    if not loader.scan_files:
        print("錯誤: 找不到任何 CT 掃描檔案")
        print("請確認 subset0 ~ subset9 資料夾中包含 .mhd 檔案")
        sys.exit(1)
    
    # 建立並執行檢視器
    viewer = LUNA16Viewer(loader)
    
    # 如果指定了 seriesuid，自動載入
    if args.seriesuid:
        viewer.scan_var.set(args.seriesuid)
        viewer.root.after(100, viewer._load_selected_scan)
    
    viewer.run()


if __name__ == "__main__":
    main()
