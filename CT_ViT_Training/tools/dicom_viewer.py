# dicom_viewer.py
# 用於瀏覽 DICOM 檔案和對應的 XML 標記
import pydicom
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
import matplotlib
import tkinter as tk
from tkinter import ttk
import json
import sys
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import messagebox
import threading

# 設定字型以解決中文顯示問題
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft JhengHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# 自動往上多層尋找 config.json（最多三層）
config_path = None
for up in range(4):
    candidate = Path(__file__).parent
    for _ in range(up):
        candidate = candidate.parent
    test_path = candidate / 'config.json'
    if test_path.exists():
        config_path = test_path
        break
if config_path is None:
    print('找不到 config.json，請確認檔案位置。')
    sys.exit(1)
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)
# 取得 matched_data_by_patient 路徑（支援絕對與相對路徑）
matched_data_path = config.get('matched_data_by_patient', 'CT_ViT_Training/matched_data_by_patient')
if not Path(matched_data_path).is_absolute():
    matched_data_path = (config_path.parent / matched_data_path).resolve()
BASE_DIR = Path(matched_data_path)

def parse_xml_bboxes(xml_path):
    """
    解析XML標記檔，提取bounding box資訊
    Args:
        xml_path: XML檔案路徑
    Returns:
        bboxes: List of bounding boxes, each as (xmin, ymin, xmax, ymax)
    """
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
                except Exception:
                    continue
    except Exception as e:
        print(f"解析XML失敗: {e}")
    return bboxes


def load_dicom_volume(dicom_files):
    """
    載入DICOM檔案並建構3D體積
    Args:
        dicom_files: DICOM檔案列表
    Returns:
        volume: 3D numpy array
        slices: 排序後的DICOM slice物件列表
        spacing: 像素間距資訊
    """
    print(f"開始載入DICOM體積，檔案數量: {len(dicom_files)}")
    slices = []
    for i, dicom_file in enumerate(dicom_files):
        try:
            print(f"載入第 {i+1}/{len(dicom_files)} 個檔案: {dicom_file.name}")
            ds = pydicom.dcmread(str(dicom_file))
            slices.append(ds)
        except Exception as e:
            print(f"讀取DICOM檔案失敗 {dicom_file}: {e}")
            continue
    
    if not slices:
        print("錯誤：沒有成功載入任何DICOM檔案")
        return None, None, None
    
    print(f"成功載入 {len(slices)} 個DICOM切片")
    
    # 根據切片位置排序
    try:
        print("嘗試使用ImagePositionPatient排序...")
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        print("使用ImagePositionPatient排序成功")
    except:
        # 如果沒有ImagePositionPatient，使用InstanceNumber排序
        try:
            print("ImagePositionPatient排序失敗，嘗試使用InstanceNumber排序...")
            slices.sort(key=lambda x: int(x.InstanceNumber))
            print("使用InstanceNumber排序成功")
        except:
            print("警告：無法確定切片順序，使用原始順序")
    
    # 建構3D陣列
    if len(slices) == 0:
        print("錯誤：排序後沒有切片")
        return None, None, None
    
    # 檢查第一個切片的信息
    first_slice = slices[0]
    print(f"第一個切片形狀: {first_slice.pixel_array.shape}")
    print(f"第一個切片數據類型: {first_slice.pixel_array.dtype}")
    print(f"第一個切片數據範圍: {first_slice.pixel_array.min()} ~ {first_slice.pixel_array.max()}")
    
    img_shape = list(slices[0].pixel_array.shape)
    img_shape.append(len(slices))
    print(f"準備建構3D陣列，形狀: {img_shape}")
    
    volume = np.zeros(img_shape, dtype=np.float32)
    
    for i, s in enumerate(slices):
        try:
            volume[:, :, i] = s.pixel_array.astype(np.float32)
            if i % 5 == 0:  # 每5個切片印一次進度
                print(f"載入切片進度: {i+1}/{len(slices)}")
        except Exception as e:
            print(f"載入切片 {i} 時發生錯誤: {e}")
            continue
    
    print(f"3D體積建構完成，最終形狀: {volume.shape}")
    print(f"3D體積數據範圍: {volume.min()} ~ {volume.max()}")
    
    # 獲取像素間距資訊
    try:
        pixel_spacing = slices[0].PixelSpacing
        slice_thickness = float(slices[0].SliceThickness) if hasattr(slices[0], 'SliceThickness') else 1.0
        spacing = [pixel_spacing[0], pixel_spacing[1], slice_thickness]
        print(f"像素間距: {spacing}")
    except Exception as e:
        print(f"獲取像素間距失敗: {e}，使用預設值")
        spacing = [1.0, 1.0, 1.0]
    return volume, slices, spacing


def create_3d_viewer_window(patient_id, base_dir):
    """
    建立3D檢視器視窗
    Args:
        patient_id: 病人ID
        base_dir: 基礎目錄路徑
    """
    # 載入DICOM資料
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
    
    # 建立3D檢視器視窗
    viewer_window = tk.Toplevel()
    viewer_window.title(f'3D DICOM Viewer - {patient_id}')
    viewer_window.geometry('1200x800')
    
    # 建立載入提示
    loading_label = tk.Label(viewer_window, text="正在載入3D數據，請稍候...", font=('Arial', 12))
    loading_label.pack(pady=50)
    
    def load_and_display():
        try:
            # 載入3D體積資料
            volume, slices, spacing = load_dicom_volume(dicom_files)
            
            if volume is None:
                messagebox.showerror("錯誤", "無法載入3D體積資料")
                viewer_window.destroy()
                return
            
            # 在主線程中更新GUI
            viewer_window.after(0, lambda: setup_3d_gui(viewer_window, volume, slices, spacing, patient_id, xml_ann_dir, loading_label))
            
        except Exception as e:
            messagebox.showerror("錯誤", f"載入3D資料時發生錯誤: {e}")
            viewer_window.destroy()
    
    # 在背景線程中載入資料
    thread = threading.Thread(target=load_and_display)
    thread.daemon = True
    thread.start()


def setup_3d_gui(window, volume, slices, spacing, patient_id, xml_ann_dir, loading_label):
    """
    設置3D檢視器的GUI介面
    """
    # 移除載入提示
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
    
    tk.Label(param_frame, text="窗寬 (Window Width):").pack()
    ww_var = tk.IntVar(value=1500)
    ww_scale = tk.Scale(param_frame, from_=100, to=4000, 
                       orient=tk.HORIZONTAL, variable=ww_var, length=200)
    ww_scale.pack()
    
    tk.Label(param_frame, text="窗位 (Window Level):").pack()
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
    
    # 軸向切片 (XY平面)
    ax1 = fig.add_subplot(221)
    ax1.set_title('軸向切片 (Axial)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    # 冠狀切片 (XZ平面)
    ax2 = fig.add_subplot(222)
    ax2.set_title('冠狀切片 (Coronal)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    
    # 矢狀切片 (YZ平面)
    ax3 = fig.add_subplot(223)
    ax3.set_title('矢狀切片 (Sagittal)')
    ax3.set_xlabel('Y')
    ax3.set_ylabel('Z')
    
    # 3D體積渲染
    ax4 = fig.add_subplot(224, projection='3d')
    ax4.set_title('3D體積渲染')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    
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
    
    def get_bboxes_for_slice(slice_idx, plane='axial'):
        """獲取特定切片的bounding boxes"""
        if not xml_ann_dir or not xml_ann_dir.exists():
            return []
        
        try:
            if plane == 'axial' and slice_idx < len(slices):
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
        
        # 軸向切片 (XY)
        ax1.clear()
        if axial_idx < volume.shape[2]:
            axial_slice = volume[:, :, axial_idx]
            windowed_axial = apply_window_level(axial_slice, ww, wl)
            ax1.imshow(windowed_axial, cmap='bone', origin='lower')
            ax1.set_title(f'軸向切片 {axial_idx}')
            
            # 顯示十字線
            if show_crosshair:
                ax1.axhline(coronal_idx, color='yellow', linestyle='--', alpha=0.7, linewidth=1)
                ax1.axvline(sagittal_idx, color='green', linestyle='--', alpha=0.7, linewidth=1)
            
            # 顯示bounding boxes
            if show_bbox:
                bboxes = get_bboxes_for_slice(axial_idx, 'axial')
                for bbox in bboxes:
                    xmin, ymin, xmax, ymax = bbox
                    ax1.add_patch(
                        matplotlib.patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, 
                                                   edgecolor='red', facecolor='none', linewidth=2)
                    )
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        
        # 冠狀切片 (XZ)
        ax2.clear()
        if coronal_idx < volume.shape[1]:
            coronal_slice = volume[:, coronal_idx, :]
            windowed_coronal = apply_window_level(coronal_slice, ww, wl)
            ax2.imshow(windowed_coronal, cmap='bone', origin='lower', aspect='auto')
            ax2.set_title(f'冠狀切片 {coronal_idx}')
            
            if show_crosshair:
                ax2.axhline(sagittal_idx, color='green', linestyle='--', alpha=0.7, linewidth=1)
                ax2.axvline(axial_idx, color='yellow', linestyle='--', alpha=0.7, linewidth=1)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z')
        
        # 矢狀切片 (YZ)
        ax3.clear()
        if sagittal_idx < volume.shape[0]:
            sagittal_slice = volume[sagittal_idx, :, :]
            windowed_sagittal = apply_window_level(sagittal_slice, ww, wl)
            ax3.imshow(sagittal_slice.T, cmap='bone', origin='lower', aspect='auto')
            ax3.set_title(f'矢狀切片 {sagittal_idx}')
            
            if show_crosshair:
                ax3.axhline(axial_idx, color='yellow', linestyle='--', alpha=0.7, linewidth=1)
                ax3.axvline(coronal_idx, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax3.set_xlabel('Y')
        ax3.set_ylabel('Z')        # 3D體積渲染（簡化版）
        ax4.clear()
        try:
            print(f"開始3D渲染，體積形狀: {volume.shape}")
            print(f"體積數據類型: {volume.dtype}")
            print(f"體積數據範圍: {volume.min()} ~ {volume.max()}")
            
            # 創建簡化的3D可視化 - 修正版本
            step = 4
            sampled_volume = volume[::step, ::step, ::step]
            
            # 創建對應的座標網格 - 維度順序要匹配
            x_coords, y_coords, z_coords = np.mgrid[0:sampled_volume.shape[0], 
                                                    0:sampled_volume.shape[1], 
                                                    0:sampled_volume.shape[2]]
            
            print(f"下採樣後的體積形狀: {sampled_volume.shape}")
            print(f"下採樣後的數據範圍: {sampled_volume.min()} ~ {sampled_volume.max()}")
            print(f"座標網格形狀: x={x_coords.shape}, y={y_coords.shape}, z={z_coords.shape}")
            
            # 只顯示高密度區域
            threshold = np.percentile(sampled_volume, 85)
            mask = sampled_volume > threshold
            
            print(f"閾值: {threshold}")
            print(f"符合閾值的體素數量: {np.sum(mask)}")
            print(f"mask形狀: {mask.shape}")
            
            if np.any(mask):
                print("開始繪製3D散點圖...")
                # 使用原始座標系統來顯示，乘以步長來映射回原始尺寸
                ax4.scatter(x_coords[mask] * step, y_coords[mask] * step, z_coords[mask] * step, 
                           c=sampled_volume[mask], cmap='bone', alpha=0.1, s=1)
                print("3D散點圖繪製完成")
            else:
                print("警告：沒有符合閾值的體素，顯示低閾值版本")
                threshold_low = np.percentile(sampled_volume, 70)
                mask_low = sampled_volume > threshold_low
                if np.any(mask_low):
                    ax4.scatter(x_coords[mask_low] * step, y_coords[mask_low] * step, z_coords[mask_low] * step, 
                               c=sampled_volume[mask_low], cmap='bone', alpha=0.05, s=0.5)
            
            ax4.set_title('3D體積渲染')
            ax4.set_xlabel('X')
            ax4.set_ylabel('Y')
            ax4.set_zlabel('Z')
            
            # 設置坐標軸範圍
            ax4.set_xlim(0, volume.shape[0])
            ax4.set_ylim(0, volume.shape[1])
            ax4.set_zlim(0, volume.shape[2])
            
            print("3D渲染設置完成")
            
        except Exception as e:
            print(f"3D渲染發生錯誤: {type(e).__name__}: {str(e)}")
            print(f"錯誤詳細信息:")
            import traceback
            traceback.print_exc()
            
            # 在圖上顯示錯誤
            ax4.text(0.5, 0.5, 0.5, f'3D渲染錯誤\n{type(e).__name__}\n點擊查看控制台', 
                    transform=ax4.transAxes, ha='center', va='center', fontsize=10)
        
        canvas.draw()
    
    # 綁定事件
    axial_var.trace('w', update_display)
    coronal_var.trace('w', update_display)
    sagittal_var.trace('w', update_display)
    ww_var.trace('w', update_display)
    wl_var.trace('w', update_display)
    show_bbox_var.trace('w', update_display)
    show_crosshair_var.trace('w', update_display)
    
    # 初始顯示
    update_display()


def interactive_patient_viewer(patient_id: str, base_dir: str, save_image: bool = False):
    """
    單一視窗左右切換預覽病人所有 DICOM+XML 標記，並可即時切換病人
    Args:
        patient_id: 初始病人ID (如 A00001)
        base_dir: matched_data_by_patient 根目錄
        save_image: 是否保存影像
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import tkinter as tk
    from tkinter import ttk, messagebox
    from pathlib import Path
    import pydicom

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
            messagebox.showerror("錯誤", f"{pid} 缺少 dicom_files 或 xml_annotations 目錄")
            return [], None
        dicom_files = sorted(dicom_files_dir.glob("*.dcm"))
        return dicom_files, xml_ann_dir

    def show_img():
        ax.clear()
        dicom_files = state['dicom_files']
        idx = state['idx']
        if not dicom_files:
            ax.set_title('無 DICOM 檔案')
            canvas.draw()
            return
        dicom_file = dicom_files[idx]
        try:
            dicom_data = pydicom.dcmread(str(dicom_file))
            img_array = dicom_data.pixel_array
            ax.imshow(img_array, cmap=plt.cm.bone)
            ax.set_title(f"{dicom_file.name}\nUID: {getattr(dicom_data, 'SOPInstanceUID', 'Unknown')}")
            ax.axis('off')
            sop_uid = getattr(dicom_data, 'SOPInstanceUID', None)
            xml_ann_dir = state['xml_ann_dir']
            if sop_uid and xml_ann_dir:
                xml_file = xml_ann_dir / f"{sop_uid}.xml"
                if xml_file.exists():
                    bboxes = parse_xml_bboxes(str(xml_file))
                    for bbox in bboxes:
                        xmin, ymin, xmax, ymax = bbox
                        ax.add_patch(
                            matplotlib.patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, edgecolor='red', facecolor='none', linewidth=2)
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
        """開啟3D檢視器"""
        try:
            create_3d_viewer_window(state['patient_id'], base_dir)
        except Exception as e:
            messagebox.showerror("錯誤", f"開啟3D檢視器時發生錯誤: {e}")

    def on_patient_change(event=None):
        pid = var.get()
        dicom_files, xml_ann_dir = load_patient_data(pid)
        state['dicom_files'] = dicom_files
        state['xml_ann_dir'] = xml_ann_dir
        state['idx'] = 0
        state['patient_id'] = pid
        show_img()

    btn_prev.config(command=prev)
    btn_next.config(command=next)
    btn_3d.config(command=show_3d_viewer)
    combo.bind('<<ComboboxSelected>>', on_patient_change)
    on_patient_change()
    root.mainloop()

def select_patient_id(base_dir):
    """
    直接返回預設病人編號 A0001，不彈出選單
    Returns: patient_id (str)
    """
    return 'A0001'

def main():
    """主函數：同時支援命令列參數與互動式選單"""
    parser = argparse.ArgumentParser(description="DICOM檔案訪問工具")
    parser.add_argument("--patient-id", type=str, help="指定病人ID預覽所有標記資料")
    parser.add_argument("--save", action="store_true", help="保存影像預覽")
    args = parser.parse_args()
    # 使用 config.json 的 BASE_DIR
    base_dir = BASE_DIR
    patient_id = args.patient_id if args.patient_id else select_patient_id(base_dir)
    if not patient_id:
        print('未選擇病人編號，程式結束。')
        return
    interactive_patient_viewer(patient_id, str(base_dir), save_image=args.save)

if __name__ == "__main__":
    main()
