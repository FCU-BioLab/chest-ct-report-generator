# 顯示DICOM圖片和XML標記的可視化程式
import pydicom
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xml.etree.ElementTree as ET
import numpy as np
import os
from matplotlib.widgets import Button
import json

class DicomAnnotationViewer:
    def __init__(self, patient_id='G0001'):
        self.patient_id = patient_id
        self.base_dir = rf'D:\GitHub\chest-ct-report-generator\matched_data_by_patient\{patient_id}'
        self.dicom_dir = os.path.join(self.base_dir, 'dicom_files')
        self.xml_dir = os.path.join(self.base_dir, 'xml_annotations')
        
        # 讀取文件清單
        self.load_file_list()
        
        # 當前顯示的圖片索引
        self.current_index = 0
        
        # 設置圖形
        self.setup_plot()
        
    def load_file_list(self):
        """讀取文件清單"""
        json_file = os.path.join(self.base_dir, f'{self.patient_id}_file_list.json')
        
        if os.path.exists(json_file):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.file_pairs = data['copied_files']
        else:
            print(f"找不到文件清單: {json_file}")
            return
            
        print(f"載入了 {len(self.file_pairs)} 個文件對")
    
    def setup_plot(self):
        """設置matplotlib圖形"""
        # 設置中文字體
        try:
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            print("警告: 無法設置中文字體，可能會有顯示問題")
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 8))
        self.fig.suptitle(f'Patient {self.patient_id} - DICOM Image with XML Annotations', fontsize=16)
        
        # 調整子圖間距
        plt.subplots_adjust(bottom=0.15, left=0.05, right=0.95, top=0.9, wspace=0.3)
        
        # 添加控制按鈕
        self.add_navigation_buttons()
        
        # 顯示第一張圖片
        self.show_current_image()
    
    def add_navigation_buttons(self):
        """添加導航按鈕"""
        # 上一張按鈕
        ax_prev = plt.axes([0.2, 0.05, 0.1, 0.04])
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_prev.on_clicked(self.prev_image)
        
        # 下一張按鈕
        ax_next = plt.axes([0.7, 0.05, 0.1, 0.04])
        self.btn_next = Button(ax_next, 'Next')
        self.btn_next.on_clicked(self.next_image)
        
        # 顯示當前圖片信息的文字
        self.info_text = self.fig.text(0.5, 0.02, '', ha='center', fontsize=10)
    
    def prev_image(self, event):
        """顯示上一張圖片"""
        if self.current_index > 0:
            self.current_index -= 1
            self.show_current_image()
    
    def next_image(self, event):
        """顯示下一張圖片"""
        if self.current_index < len(self.file_pairs) - 1:
            self.current_index += 1
            self.show_current_image()
    
    def load_dicom_image(self, dicom_path):
        """載入DICOM圖片"""
        try:
            ds = pydicom.dcmread(dicom_path)
            img_array = ds.pixel_array
            
            # 正規化圖片數據
            img_array = img_array.astype(float)
            img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())
            
            return img_array, ds
        except Exception as e:
            print(f"載入DICOM文件錯誤: {e}")
            return None, None
    
    def parse_xml_annotations(self, xml_path):
        """解析XML標記文件"""
        annotations = []
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # 查找所有物件標記 (標準的annotation格式)
            for obj in root.findall('.//object'):
                name = obj.find('name')
                bndbox = obj.find('bndbox')
                
                if bndbox is not None:
                    # 提取邊界框座標
                    xmin_elem = bndbox.find('xmin')
                    ymin_elem = bndbox.find('ymin')
                    xmax_elem = bndbox.find('xmax')
                    ymax_elem = bndbox.find('ymax')
                    
                    if all(elem is not None for elem in [xmin_elem, ymin_elem, xmax_elem, ymax_elem]):
                        xmin = float(xmin_elem.text)
                        ymin = float(ymin_elem.text)
                        xmax = float(xmax_elem.text)
                        ymax = float(ymax_elem.text)
                        
                        annotation = {
                            'type': 'bounding_box',
                            'name': name.text if name is not None else 'Unknown',
                            'coords': {
                                'xmin': xmin,
                                'ymin': ymin,
                                'xmax': xmax,
                                'ymax': ymax
                            }
                        }
                        annotations.append(annotation)
            
            # 如果沒有找到標準格式，嘗試查找其他格式
            if not annotations:
                # 查找結節標記 (原本的代碼)
                for marking in root.iter():
                    if 'Nodule' in marking.tag or 'nodule' in marking.tag.lower():
                        # 提取結節信息
                        nodule_info = {'type': 'nodule', 'regions': []}
                        
                        # 查找所有標記區域
                        for roi in marking.iter():
                            if 'roi' in roi.tag.lower() or 'region' in roi.tag.lower():
                                # 提取座標信息
                                coords = []
                                for coord in roi.iter():
                                    if 'edgeMap' in coord.tag:
                                        x = coord.get('xCoord')
                                        y = coord.get('yCoord')
                                        if x and y:
                                            coords.append((float(x), float(y)))
                                
                                if coords:
                                    nodule_info['regions'].append(coords)
                        
                        if nodule_info['regions']:
                            annotations.append(nodule_info)
            
            return annotations
            
        except Exception as e:
            print(f"解析XML文件錯誤: {e}")
            return []
    
    def show_current_image(self):
        """顯示當前圖片和標記"""
        if not hasattr(self, 'file_pairs') or not self.file_pairs:
            print("沒有可顯示的文件")
            return
            
        current_pair = self.file_pairs[self.current_index]
        
        # 載入DICOM圖片
        dicom_path = current_pair['copied_dcm']
        xml_path = current_pair['copied_xml']
        
        img_array, dicom_data = self.load_dicom_image(dicom_path)
        
        if img_array is None:
            print(f"無法載入圖片: {dicom_path}")
            return
        
        # 清除之前的圖片
        self.ax1.clear()
        self.ax2.clear()
        
        # 顯示原始DICOM圖片
        self.ax1.imshow(img_array, cmap='gray')
        self.ax1.set_title('Original DICOM Image')
        self.ax1.axis('off')
        
        # 顯示帶標記的圖片
        self.ax2.imshow(img_array, cmap='gray')
        self.ax2.set_title('DICOM with XML Annotations')
        self.ax2.axis('off')
        
        # 解析並顯示XML標記
        annotations = self.parse_xml_annotations(xml_path)
        
        annotation_count = 0
        for annotation in annotations:
            if annotation['type'] == 'bounding_box':
                # 繪製邊界框
                coords = annotation['coords']
                width = coords['xmax'] - coords['xmin']
                height = coords['ymax'] - coords['ymin']
                
                # 創建矩形
                rect = patches.Rectangle((coords['xmin'], coords['ymin']), 
                                       width, height,
                                       linewidth=2, edgecolor='red', 
                                       facecolor='none', alpha=0.8)
                self.ax2.add_patch(rect)
                
                # 添加標籤
                label = annotation['name']
                self.ax2.text(coords['xmin'], coords['ymin']-5, label, 
                            color='red', fontweight='bold', fontsize=10,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                
                annotation_count += 1
                
            elif annotation['type'] == 'nodule':
                # 原本的多邊形繪製
                for region in annotation['regions']:
                    if len(region) > 2:  # 確保有足夠的點來繪製區域
                        # 將座標轉換為numpy數組
                        coords = np.array(region)
                        
                        # 創建多邊形
                        polygon = patches.Polygon(coords, linewidth=2, 
                                                edgecolor='yellow', facecolor='none', 
                                                alpha=0.8)
                        self.ax2.add_patch(polygon)
                        annotation_count += 1
        
        # 更新信息文字
        info = f"Image {self.current_index + 1}/{len(self.file_pairs)} | "
        info += f"UID: {current_pair['uid'][:20]}... | "
        info += f"Annotations: {annotation_count} | "
        info += f"Size: {img_array.shape}"
        
        self.info_text.set_text(info)
        
        # 如果沒有找到標記，嘗試顯示一些XML內容
        if annotation_count == 0:
            self.show_xml_content(xml_path)
        
        # 重新繪製
        self.fig.canvas.draw()
        
        print(f"Displaying image {self.current_index + 1}: {os.path.basename(dicom_path)}")
        print(f"Corresponding XML: {os.path.basename(xml_path)}")
        print(f"Found {annotation_count} annotation regions")
        print("-" * 50)
    
    def show_xml_content(self, xml_path):
        """顯示XML文件的部分內容（當無法解析標記時）"""
        try:
            with open(xml_path, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"XML file content preview ({os.path.basename(xml_path)}):")
                print(content[:500] + "..." if len(content) > 500 else content)
                print("-" * 30)
        except Exception as e:
            print(f"Error reading XML file: {e}")
    
    def show(self):
        """顯示視窗"""
        plt.show()

def create_simple_example():
    """創建一個簡單的示例，顯示單張圖片"""
    
    # 使用第一個可用的文件對
    patient_id = 'A0177'
    base_dir = rf'D:\GitHub\chest-ct-report-generator\matched_data_by_patient\{patient_id}'
    json_file = os.path.join(base_dir, f'{patient_id}_file_list.json')
    
    if not os.path.exists(json_file):
        print(f"File not found: {json_file}")
        return
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        first_pair = data['copied_files'][0]  # 使用第一個文件對
    
    # 載入DICOM圖片
    dicom_path = first_pair['copied_dcm']
    xml_path = first_pair['copied_xml']
    
    print(f"Loading DICOM file: {dicom_path}")
    print(f"Loading XML file: {xml_path}")
    
    # 讀取DICOM
    ds = pydicom.dcmread(dicom_path)
    img_array = ds.pixel_array
    
    # 正規化圖片
    img_array = img_array.astype(float)
    img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())
    
    # 創建圖形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'Patient {patient_id} - Sample Image with Annotations', fontsize=14)
    
    # 顯示原始圖片
    ax1.imshow(img_array, cmap='gray')
    ax1.set_title('Original DICOM Image')
    ax1.axis('off')
    
    # 顯示圖片信息
    info_text = f"UID: {first_pair['uid'][:30]}...\n"
    info_text += f"Image size: {img_array.shape}\n"
    info_text += f"File size: {first_pair['dcm_size']} bytes"
    
    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
    
    # 顯示帶標記的圖片
    ax2.imshow(img_array, cmap='gray')
    ax2.set_title('XML Annotation Region')
    ax2.axis('off')    # 讀取並顯示XML內容
    try:
        with open(xml_path, 'r', encoding='utf-8') as f:
            xml_content = f.read()
            
        # 在圖片上顯示XML文件的基本信息
        xml_info = f"XML file: {os.path.basename(xml_path)}\n"
        xml_info += f"File size: {first_pair['xml_size']} bytes\n"
        xml_info += f"Contains annotation info"
        
        # 如果XML很短，顯示部分內容
        if len(xml_content) < 1000:
            xml_info += f"\n\nXML content preview:\n{xml_content[:200]}..."
        
        ax2.text(0.02, 0.98, xml_info, transform=ax2.transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle="round", facecolor='yellow', alpha=0.8),
                 fontsize=8, wrap=True)
        
        # 嘗試解析XML並繪製標記（簡化版本）
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # 在圖片中心繪製一個示例標記框
            height, width = img_array.shape
            
            # 創建一個示例標記框（因為XML格式可能很複雜）
            rect = patches.Rectangle((width*0.4, height*0.4), width*0.2, height*0.2, 
                                   linewidth=2, edgecolor='red', facecolor='none', alpha=0.8)
            ax2.add_patch(rect)
            
            # 添加標記文字
            ax2.text(width*0.5, height*0.3, 'XML Annotation\nRegion (Example)', 
                    ha='center', va='center', color='red', fontweight='bold',
                    bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
            
        except Exception as e:
            print(f"XML parsing error: {e}")
            
    except Exception as e:
        print(f"Error reading XML file: {e}")
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n=== Example Information ===")
    print(f"Patient ID: {patient_id}")
    print(f"DICOM file: {os.path.basename(dicom_path)}")
    print(f"XML file: {os.path.basename(xml_path)}")
    print(f"Image dimensions: {img_array.shape}")
    print(f"SOP Instance UID: {first_pair['uid']}")

if __name__ == "__main__":
    print("DICOM Image with XML Annotation Visualization")
    print("="*50)
    
    # 選擇運行模式
    mode = input("Select mode (1: Simple example, 2: Interactive browser): ").strip()
    
    if mode == "1":
        print("Loading simple example...")
        create_simple_example()
    else:
        print("Loading interactive browser...")
        viewer = DicomAnnotationViewer()
        viewer.show()
