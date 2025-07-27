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
