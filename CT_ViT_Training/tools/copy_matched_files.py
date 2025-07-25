# 複製有標記的DICOM文件和XML文件到新資料夾
import os
import shutil
import json
from datetime import datetime

def create_directory_if_not_exists(path):
    """如果目錄不存在則創建"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"創建目錄: {path}")

def copy_matched_files():
    """複製匹配的DICOM和XML文件"""
    
    # 從之前的腳本結果讀取匹配信息
    # 這裡我們重新運行匹配邏輯，因為我們需要matched_pairs數據
    
    # === 重新獲取匹配數據 ===
    import pydicom
    
    # DICOM文件搜索
    dcm_file = []
    search_path = r'D:\GitHub\chest-ct-report-generator\datasets\Lung-PET-CT-Dx\manifest-1608669183333\Lung-PET-CT-Dx\Lung_Dx-A0001'
    
    for root, dirs, files in os.walk(search_path):
        for file in files:
            file_path = os.path.join(root, file)
            if 'dcm' in file_path:
                dcm_file.append(file_path)
    
    # 提取DICOM文件的SOP Instance UID
    dcm_file_uid = []
    for dcm in dcm_file:
        try:
            im = pydicom.dcmread(dcm)
            dcm_uid = im.SOPInstanceUID
            dcm_file_uid.append(dcm_uid)
        except Exception as e:
            print(f"Error reading {dcm}: {e}")
    
    # XML文件搜索
    xml_file = []
    xml_search_path = r'D:\GitHub\chest-ct-report-generator\datasets\Lung-PET-CT-Dx\Lung-PET-CT-Dx-Annotations-XML-Files-rev12222020\Annotation\A0001'
    
    for root, dirs, files in os.walk(xml_search_path):
        for file in files:
            file_path = os.path.join(root, file)      
            if 'xml' in file_path:
                xml_file.append(file_path)
    
    # 從XML文件名提取UID
    xml_uids = []
    for xml_path in xml_file:
        filename = os.path.basename(xml_path)
        uid = filename.replace('.xml', '')
        xml_uids.append(uid)
    
    # 匹配DICOM和XML
    matched_pairs = []
    xml_uid_set = set(xml_uids)
    
    for i, dcm_uid in enumerate(dcm_file_uid):
        if dcm_uid in xml_uid_set:
            xml_index = xml_uids.index(dcm_uid)
            matched_pairs.append({
                'dcm_file': dcm_file[i],
                'dcm_uid': dcm_uid,
                'xml_file': xml_file[xml_index]
            })
    
    print(f"找到 {len(matched_pairs)} 個匹配的DICOM-XML對")
    
    # === 創建目標目錄 ===
    patient_id = 'A0001'  # 當前處理的病例ID
    base_output_dir = r'D:\GitHub\chest-ct-report-generator\matched_data'
    patient_output_dir = os.path.join(base_output_dir, patient_id)
    dcm_output_dir = os.path.join(patient_output_dir, 'dicom_files')
    xml_output_dir = os.path.join(patient_output_dir, 'xml_annotations')
    
    create_directory_if_not_exists(base_output_dir)
    create_directory_if_not_exists(patient_output_dir)
    create_directory_if_not_exists(dcm_output_dir)
    create_directory_if_not_exists(xml_output_dir)
    
    # === 複製文件並記錄 ===
    copied_files = []
    copy_errors = []
    
    print(f"\n開始複製文件...")
    
    for i, pair in enumerate(matched_pairs):
        try:
            # 複製DICOM文件
            dcm_filename = os.path.basename(pair['dcm_file'])
            dcm_dest = os.path.join(dcm_output_dir, f"{pair['dcm_uid']}.dcm")
            shutil.copy2(pair['dcm_file'], dcm_dest)
            
            # 複製XML文件
            xml_filename = os.path.basename(pair['xml_file'])
            xml_dest = os.path.join(xml_output_dir, xml_filename)
            shutil.copy2(pair['xml_file'], xml_dest)
            
            # 記錄複製的文件信息
            copied_files.append({
                'index': i + 1,
                'uid': pair['dcm_uid'],
                'original_dcm': pair['dcm_file'],
                'copied_dcm': dcm_dest,
                'original_xml': pair['xml_file'],
                'copied_xml': xml_dest,
                'dcm_size': os.path.getsize(pair['dcm_file']),
                'xml_size': os.path.getsize(pair['xml_file'])
            })
            
            if (i + 1) % 5 == 0:
                print(f"已複製 {i + 1}/{len(matched_pairs)} 個文件對...")
                
        except Exception as e:
            error_info = {
                'index': i + 1,
                'uid': pair['dcm_uid'],
                'error': str(e),
                'dcm_file': pair['dcm_file'],
                'xml_file': pair['xml_file']
            }
            copy_errors.append(error_info)
            print(f"複製第 {i + 1} 個文件對時出錯: {e}")
    
    print(f"\n複製完成!")
    print(f"成功複製: {len(copied_files)} 個文件對")
    print(f"複製失敗: {len(copy_errors)} 個文件對")
    
    # === 生成文件清單 ===
    
    # 1. 生成JSON格式的詳細清單
    file_list = {
        'summary': {
            'patient_id': patient_id,
            'total_pairs': len(matched_pairs),
            'successfully_copied': len(copied_files),
            'copy_errors': len(copy_errors),
            'creation_time': datetime.now().isoformat(),
            'source_patient': f'Lung_Dx-{patient_id}'
        },
        'copied_files': copied_files,
        'copy_errors': copy_errors
    }
    
    json_list_path = os.path.join(patient_output_dir, f'{patient_id}_file_list.json')
    with open(json_list_path, 'w', encoding='utf-8') as f:
        json.dump(file_list, f, indent=2, ensure_ascii=False)
    
    # 2. 生成可讀的文本清單
    txt_list_path = os.path.join(patient_output_dir, f'{patient_id}_file_list.txt')
    with open(txt_list_path, 'w', encoding='utf-8') as f:
        f.write(f"=== 病例 {patient_id} 的DICOM和XML文件清單 ===\n\n")
        f.write(f"患者ID: {patient_id}\n")
        f.write(f"原始資料夾: Lung_Dx-{patient_id}\n")
        f.write(f"創建時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"總匹配對數: {len(matched_pairs)}\n")
        f.write(f"成功複製: {len(copied_files)}\n")
        f.write(f"複製失敗: {len(copy_errors)}\n\n")
        
        f.write("=== 成功複製的文件 ===\n")
        for file_info in copied_files:
            f.write(f"\n文件對 {file_info['index']}:\n")
            f.write(f"  UID: {file_info['uid']}\n")
            f.write(f"  DICOM: {os.path.basename(file_info['copied_dcm'])} ({file_info['dcm_size']} bytes)\n")
            f.write(f"  XML: {os.path.basename(file_info['copied_xml'])} ({file_info['xml_size']} bytes)\n")
        
        if copy_errors:
            f.write(f"\n=== 複製失敗的文件 ===\n")
            for error in copy_errors:
                f.write(f"\n文件對 {error['index']}:\n")
                f.write(f"  UID: {error['uid']}\n")
                f.write(f"  錯誤: {error['error']}\n")
    
    # 3. 生成簡單的CSV清單
    csv_list_path = os.path.join(patient_output_dir, f'{patient_id}_file_list.csv')
    with open(csv_list_path, 'w', encoding='utf-8') as f:
        f.write("Index,Patient_ID,UID,DICOM_File,XML_File,DICOM_Size,XML_Size\n")
        for file_info in copied_files:
            f.write(f"{file_info['index']},{patient_id},{file_info['uid']},{os.path.basename(file_info['copied_dcm'])},{os.path.basename(file_info['copied_xml'])},{file_info['dcm_size']},{file_info['xml_size']}\n")
    
    # 4. 生成總體統計清單（在根目錄）
    summary_path = os.path.join(base_output_dir, 'patients_summary.txt')
    with open(summary_path, 'a', encoding='utf-8') as f:  # 使用 'a' 模式追加
        f.write(f"病例 {patient_id}:\n")
        f.write(f"  - 匹配的文件對: {len(matched_pairs)}\n")
        f.write(f"  - 成功複製: {len(copied_files)}\n")
        f.write(f"  - 複製失敗: {len(copy_errors)}\n")
        f.write(f"  - 處理時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  - 資料夾: {patient_output_dir}\n\n")
    
    print(f"\n=== 文件清單已生成 ===")
    print(f"病例目錄: {patient_output_dir}")
    print(f"詳細清單(JSON): {json_list_path}")
    print(f"可讀清單(TXT): {txt_list_path}")
    print(f"簡單清單(CSV): {csv_list_path}")
    print(f"總體統計: {summary_path}")
    print(f"\n=== 複製的文件位置 ===")
    print(f"病例 {patient_id} DICOM文件: {dcm_output_dir}")
    print(f"病例 {patient_id} XML文件: {xml_output_dir}")
    
    return copied_files, copy_errors

if __name__ == "__main__":
    copied_files, copy_errors = copy_matched_files()
