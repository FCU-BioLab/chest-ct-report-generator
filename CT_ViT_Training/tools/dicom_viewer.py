# dicom_viewer.py
# 用於瀏覽 DICOM 檔案和對應的 XML 標記
import pydicom
import matplotlib.pyplot as plt
import matplotlib.patches
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
import matplotlib
import tkinter as tk
from tkinter import ttk, messagebox
import json
import sys
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

# 設定字型以解決中文顯示問題
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft JhengHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# 自動往上多層尋找 config.json
def find_config():
    for up in range(4):
        candidate = Path(__file__).parent
        for _ in range(up):
            candidate = candidate.parent
        test_path = candidate / 'config.json'
        if test_path.exists():
            return test_path
    print('找不到 config.json，請確認檔案位置。')
    sys.exit(1)

config_path = find_config()
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

matched_data_path = config.get('all_patient_data', 'CT_ViT_Training/all_patient_data')
if not Path(matched_data_path).is_absolute():
    matched_data_path = (config_path.parent / matched_data_path).resolve()
BASE_DIR = Path(matched_data_path)

def parse_xml_bboxes(xml_path):
    """解析XML標記檔，提取bounding box資訊"""
    bboxes = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('.//object'):
            bbox = obj.find('bndbox')
            if bbox is not None:
                try:
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)
                    bboxes.append((xmin, ymin, xmax, ymax))
                except (AttributeError, ValueError):
                    continue
    except Exception as e:
        print(f"解析XML失敗: {e}")
    return bboxes


def load_dicom_volume(dicom_files):
    """載入DICOM檔案並建構3D體積"""
    slices = []
    for dicom_file in dicom_files:
        try:
            ds = pydicom.dcmread(str(dicom_file))
            slices.append(ds)
        except Exception as e:
            print(f"讀取DICOM檔案失敗 {dicom_file}: {e}")
            continue
    
    if not slices:
        return None, None, None
    
    # 根據切片位置排序
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except:
        try:
            slices.sort(key=lambda x: int(x.InstanceNumber))
        except:
            print("警告：無法確定切片順序")
    
    # 建構3D陣列
    img_shape = list(slices[0].pixel_array.shape)
    img_shape.append(len(slices))
    volume = np.zeros(img_shape, dtype=np.float32)
    
    for i, s in enumerate(slices):
        try:
            volume[:, :, i] = s.pixel_array.astype(np.float32)
        except Exception as e:
            print(f"載入切片 {i} 時發生錯誤: {e}")
            continue
    
    # 獲取像素間距
    try:
        pixel_spacing = slices[0].PixelSpacing
        slice_thickness = float(slices[0].SliceThickness) if hasattr(slices[0], 'SliceThickness') else 1.0
        spacing = [pixel_spacing[0], pixel_spacing[1], slice_thickness]
    except:
        spacing = [1.0, 1.0, 1.0]
    
    return volume, slices, spacing


def create_3d_viewer_window(patient_id, base_dir):
    """建立3D檢視器視窗"""
    patient_folder = Path(base_dir) / patient_id
    dicom_files_dir = patient_folder / "dicom_files"
    xml_ann_dir = patient_folder / "xml_annotations"
    
    if not dicom_files_dir.exists():
        messagebox.showerror("錯誤", f"找不到 {patient_id} 的DICOM檔案目錄")
        return
    
    dicom_files = sorted(dicom_files_dir.glob("*.dcm"))
    if not dicom_files:
        messagebox.showerror("錯誤", f"{patient_id} 沒有DICOM檔案")
        return
    
    viewer_window = tk.Toplevel()
    viewer_window.title(f'3D DICOM Viewer - {patient_id}')
    viewer_window.geometry('1200x800')
    
    loading_label = tk.Label(viewer_window, text="正在載入3D數據，請稍候...", font=('Arial', 12))
    loading_label.pack(pady=50)
    
    def load_and_display():
        try:
            volume, slices, spacing = load_dicom_volume(dicom_files)
            if volume is None:
                messagebox.showerror("錯誤", "無法載入3D體積資料")
                viewer_window.destroy()
                return
            viewer_window.after(0, lambda: setup_3d_gui(viewer_window, volume, slices, spacing, patient_id, xml_ann_dir, loading_label))
        except Exception as e:
            messagebox.showerror("錯誤", f"載入3D資料時發生錯誤: {e}")
            viewer_window.destroy()
    
    thread = threading.Thread(target=load_and_display)
    thread.daemon = True
    thread.start()


def setup_3d_gui(window, volume, slices, spacing, patient_id, xml_ann_dir, loading_label):
    """設置3D檢視器的GUI介面"""
    loading_label.destroy()
    
    # 建立控制面板
    control_frame = tk.Frame(window)
    control_frame.pack(pady=10)
    
    # 切片選擇控制
    slice_frame = tk.Frame(control_frame)
    slice_frame.pack(side=tk.LEFT, padx=10)
    
    tk.Label(slice_frame, text="軸向切片:").pack()
    axial_var = tk.IntVar(value=volume.shape[2]//2)
    axial_scale = tk.Scale(slice_frame, from_=0, to=volume.shape[2]-1, 
                          orient=tk.HORIZONTAL, variable=axial_var, length=200)
    axial_scale.pack()
    
    tk.Label(slice_frame, text="冠狀切片:").pack()
    coronal_var = tk.IntVar(value=volume.shape[1]//2)
    coronal_scale = tk.Scale(slice_frame, from_=0, to=volume.shape[1]-1, 
                            orient=tk.HORIZONTAL, variable=coronal_var, length=200)
    coronal_scale.pack()
    
    tk.Label(slice_frame, text="矢狀切片:").pack()
    sagittal_var = tk.IntVar(value=volume.shape[0]//2)
    sagittal_scale = tk.Scale(slice_frame, from_=0, to=volume.shape[0]-1, 
                             orient=tk.HORIZONTAL, variable=sagittal_var, length=200)
    sagittal_scale.pack()
    
    # 顯示參數控制
    param_frame = tk.Frame(control_frame)
    param_frame.pack(side=tk.LEFT, padx=20)
    
    tk.Label(param_frame, text="窗寬:").pack()
    ww_var = tk.IntVar(value=1500)
    ww_scale = tk.Scale(param_frame, from_=100, to=4000, 
                       orient=tk.HORIZONTAL, variable=ww_var, length=200)
    ww_scale.pack()
    
    tk.Label(param_frame, text="窗位:").pack()
    wl_var = tk.IntVar(value=-600)
    wl_scale = tk.Scale(param_frame, from_=-1000, to=1000, 
                       orient=tk.HORIZONTAL, variable=wl_var, length=200)
    wl_scale.pack()
    
    # 顯示選項
    option_frame = tk.Frame(control_frame)
    option_frame.pack(side=tk.LEFT, padx=20)
    
    show_bbox_var = tk.BooleanVar(value=True)
    tk.Checkbutton(option_frame, text="顯示標記框", variable=show_bbox_var).pack()
    
    show_crosshair_var = tk.BooleanVar(value=True)
    tk.Checkbutton(option_frame, text="顯示十字線", variable=show_crosshair_var).pack()
    
    # 建立圖形顯示區域
    fig = plt.figure(figsize=(15, 10))
    
    ax1 = fig.add_subplot(221)
    ax1.set_title('軸向切片')
    ax2 = fig.add_subplot(222)
    ax2.set_title('冠狀切片')
    ax3 = fig.add_subplot(223)
    ax3.set_title('矢狀切片')
    ax4 = fig.add_subplot(224, projection='3d')
    ax4.set_title('3D體積渲染')
    
    plt.tight_layout()
    
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
    
    def apply_window_level(image, ww, wl):
        """應用窗寬窗位"""
        img_min = wl - ww // 2
        img_max = wl + ww // 2
        windowed = np.clip(image, img_min, img_max)
        windowed = (windowed - img_min) / (img_max - img_min)
        return windowed
    
    def get_bboxes_for_slice(slice_idx):
        """獲取特定切片的bounding boxes"""
        if not xml_ann_dir or not xml_ann_dir.exists() or slice_idx >= len(slices):
            return []
        try:
            slice_obj = slices[slice_idx]
            sop_uid = getattr(slice_obj, 'SOPInstanceUID', None)
            if sop_uid:
                xml_file = xml_ann_dir / f"{sop_uid}.xml"
                if xml_file.exists():
                    return parse_xml_bboxes(str(xml_file))
        except:
            pass
        return []
    
    def update_display(*args):
        axial_idx = axial_var.get()
        coronal_idx = coronal_var.get()
        sagittal_idx = sagittal_var.get()
        ww = ww_var.get()
        wl = wl_var.get()
        show_bbox = show_bbox_var.get()
        show_crosshair = show_crosshair_var.get()
        
        # 軸向切片
        ax1.clear()
        if axial_idx < volume.shape[2]:
            axial_slice = volume[:, :, axial_idx]
            windowed_axial = apply_window_level(axial_slice, ww, wl)
            ax1.imshow(windowed_axial, cmap='bone', origin='lower')
            ax1.set_title(f'軸向切片 {axial_idx}')
            
            if show_crosshair:
                ax1.axhline(coronal_idx, color='yellow', linestyle='--', alpha=0.7, linewidth=1)
                ax1.axvline(sagittal_idx, color='green', linestyle='--', alpha=0.7, linewidth=1)
            
            if show_bbox:
                bboxes = get_bboxes_for_slice(axial_idx)
                for bbox in bboxes:
                    xmin, ymin, xmax, ymax = bbox
                    ax1.add_patch(
                        matplotlib.patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, 
                                                   edgecolor='red', facecolor='none', linewidth=2)
                    )
        
        # 冠狀切片
        ax2.clear()
        if coronal_idx < volume.shape[1]:
            coronal_slice = volume[:, coronal_idx, :]
            windowed_coronal = apply_window_level(coronal_slice, ww, wl)
            ax2.imshow(windowed_coronal, cmap='bone', origin='lower', aspect='auto')
            ax2.set_title(f'冠狀切片 {coronal_idx}')
            
            if show_crosshair:
                ax2.axhline(sagittal_idx, color='green', linestyle='--', alpha=0.7, linewidth=1)
                ax2.axvline(axial_idx, color='yellow', linestyle='--', alpha=0.7, linewidth=1)
        
        # 矢狀切片
        ax3.clear()
        if sagittal_idx < volume.shape[0]:
            sagittal_slice = volume[sagittal_idx, :, :]
            ax3.imshow(sagittal_slice.T, cmap='bone', origin='lower', aspect='auto')
            ax3.set_title(f'矢狀切片 {sagittal_idx}')
            
            if show_crosshair:
                ax3.axhline(axial_idx, color='yellow', linestyle='--', alpha=0.7, linewidth=1)
                ax3.axvline(coronal_idx, color='red', linestyle='--', alpha=0.7, linewidth=1)
        
        # 3D體積渲染
        ax4.clear()
        try:
            step = 4
            sampled_volume = volume[::step, ::step, ::step]
            x_coords, y_coords, z_coords = np.mgrid[0:sampled_volume.shape[0], 
                                                    0:sampled_volume.shape[1], 
                                                    0:sampled_volume.shape[2]]
            
            threshold = np.percentile(sampled_volume, 85)
            mask = sampled_volume > threshold
            
            if np.any(mask):
                ax4.scatter(x_coords[mask] * step, y_coords[mask] * step, z_coords[mask] * step, 
                           c=sampled_volume[mask], cmap='bone', alpha=0.1, s=1)
            
            ax4.set_title('3D體積渲染')
            ax4.set_xlim(0, volume.shape[0])
            ax4.set_ylim(0, volume.shape[1])
            ax4.set_zlim(0, volume.shape[2])
            
        except Exception as e:
            ax4.text(0.5, 0.5, 0.5, f'3D渲染錯誤\n{type(e).__name__}', 
                    transform=ax4.transAxes, ha='center', va='center', fontsize=10)
        
        canvas.draw()
    
    # 綁定事件
    for var in [axial_var, coronal_var, sagittal_var, ww_var, wl_var, show_bbox_var, show_crosshair_var]:
        var.trace('w', update_display)
    
    update_display()


def interactive_patient_viewer(patient_id, base_dir, save_image=False):
    """單一視窗左右切換預覽病人所有 DICOM+XML 標記，並可即時切換病人"""
    patient_ids = sorted([d.name for d in Path(base_dir).iterdir() if d.is_dir()])
    default_id = patient_id if patient_id in patient_ids else (patient_ids[0] if patient_ids else '')

    root = tk.Tk()
    root.title('DICOM Viewer')
    var = tk.StringVar(value=default_id)
    combo = ttk.Combobox(root, textvariable=var, values=patient_ids, state='readonly', width=10)
    combo.pack(padx=10, pady=5)

    fig, ax = plt.subplots(figsize=(8, 8))
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
    
    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=5)
    btn_prev = tk.Button(btn_frame, text='上一張')
    btn_prev.pack(side=tk.LEFT, padx=10)
    btn_next = tk.Button(btn_frame, text='下一張')
    btn_next.pack(side=tk.LEFT, padx=10)
    btn_3d = tk.Button(btn_frame, text='3D檢視')
    btn_3d.pack(side=tk.LEFT, padx=10)

    state = {'dicom_files': [], 'xml_ann_dir': None, 'idx': 0, 'patient_id': default_id}

    def load_patient_data(pid):
        patient_folder = Path(base_dir) / pid
        dicom_files_dir = patient_folder / "dicom_files"
        xml_ann_dir = patient_folder / "xml_annotations"
        if not dicom_files_dir.exists() or not xml_ann_dir.exists():
            messagebox.showerror("錯誤", f"{pid} 缺少目錄")
            return [], None
        return sorted(dicom_files_dir.glob("*.dcm")), xml_ann_dir

    def show_img():
        ax.clear()
        dicom_files = state['dicom_files']
        idx = state['idx']
        if not dicom_files:
            ax.set_title('無 DICOM 檔案')
            canvas.draw()
            return
        
        try:
            dicom_data = pydicom.dcmread(str(dicom_files[idx]))
            img_array = dicom_data.pixel_array
            ax.imshow(img_array, cmap=plt.cm.bone)
            ax.set_title(f"{dicom_files[idx].name}")
            ax.axis('off')
            
            # 顯示標記框
            sop_uid = getattr(dicom_data, 'SOPInstanceUID', None)
            xml_ann_dir = state['xml_ann_dir']
            if sop_uid and xml_ann_dir:
                xml_file = xml_ann_dir / f"{sop_uid}.xml"
                if xml_file.exists():
                    bboxes = parse_xml_bboxes(str(xml_file))
                    for bbox in bboxes:
                        xmin, ymin, xmax, ymax = bbox
                        ax.add_patch(
                            matplotlib.patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, 
                                                       edgecolor='red', facecolor='none', linewidth=2)
                        )
        except Exception as e:
            ax.set_title(f'載入失敗: {e}')
        canvas.draw()

    def prev():
        if state['idx'] > 0:
            state['idx'] -= 1
            show_img()

    def next():
        if state['dicom_files'] and state['idx'] < len(state['dicom_files']) - 1:
            state['idx'] += 1
            show_img()

    def show_3d_viewer():
        try:
            create_3d_viewer_window(state['patient_id'], base_dir)
        except Exception as e:
            messagebox.showerror("錯誤", f"開啟3D檢視器時發生錯誤: {e}")

    def on_patient_change(event=None):
        pid = var.get()
        dicom_files, xml_ann_dir = load_patient_data(pid)
        state.update({'dicom_files': dicom_files, 'xml_ann_dir': xml_ann_dir, 'idx': 0, 'patient_id': pid})
        show_img()

    btn_prev.config(command=prev)
    btn_next.config(command=next)
    btn_3d.config(command=show_3d_viewer)
    combo.bind('<<ComboboxSelected>>', on_patient_change)
    on_patient_change()
    root.mainloop()

def select_patient_id(base_dir):
    """直接返回預設病人編號 A0001"""
    return 'A0001'

def main():
    """主函數：同時支援命令列參數與互動式選單"""
    parser = argparse.ArgumentParser(description="DICOM檔案訪問工具")
    parser.add_argument("--patient-id", type=str, help="指定病人ID預覽所有標記資料")
    parser.add_argument("--save", action="store_true", help="保存影像預覽")
    args = parser.parse_args()
    
    base_dir = BASE_DIR
    patient_id = args.patient_id if args.patient_id else select_patient_id(base_dir)
    if not patient_id:
        print('未選擇病人編號，程式結束。')
        return
    interactive_patient_viewer(patient_id, str(base_dir), save_image=args.save)

if __name__ == "__main__":
    main()
