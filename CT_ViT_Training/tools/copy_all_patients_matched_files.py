# 複製所有病例的有標記DICOM文件和XML文件，按病例分類
import os
import shutil
import json
from datetime import datetime

def create_directory_if_not_exists(path):
    """如果目錄不存在則創建"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"創建目錄: {path}")

def copy_patient_matched_files(patient_id):
    """複製指定病例的匹配DICOM和XML文件"""
    
    import pydicom
    
    print(f"\n=== 處理病例 {patient_id} ===")
    
    # DICOM文件搜索
    dcm_file = []
    search_path = rf'D:\GitHub\chest-ct-report-generator\datasets\Lung-PET-CT-Dx\manifest-1608669183333\Lung-PET-CT-Dx\Lung_Dx-{patient_id}'
    
    if not os.path.exists(search_path):
        print(f"警告: 病例 {patient_id} 的DICOM資料夾不存在: {search_path}")
        return None
    
    for root, dirs, files in os.walk(search_path):
        for file in files:
            file_path = os.path.join(root, file)
            if 'dcm' in file_path:
                dcm_file.append(file_path)
    
    print(f"找到 {len(dcm_file)} 個DICOM文件")
    
    if len(dcm_file) == 0:
        print(f"病例 {patient_id} 沒有DICOM文件")
        return None
    
    # 提取DICOM文件的SOP Instance UID
    dcm_file_uid = []
    for dcm in dcm_file:
        try:
            im = pydicom.dcmread(dcm)
            dcm_uid = im.SOPInstanceUID
            dcm_file_uid.append(dcm_uid)
        except Exception as e:
            print(f"讀取DICOM文件錯誤 {dcm}: {e}")
    
    print(f"成功讀取 {len(dcm_file_uid)} 個DICOM文件的UID")
    
    # XML文件搜索
    xml_file = []
    xml_search_path = rf'D:\GitHub\chest-ct-report-generator\datasets\Lung-PET-CT-Dx\Lung-PET-CT-Dx-Annotations-XML-Files-rev12222020\Annotation\{patient_id}'
    
    if not os.path.exists(xml_search_path):
        print(f"警告: 病例 {patient_id} 的XML標注資料夾不存在: {xml_search_path}")
        return None
    
    for root, dirs, files in os.walk(xml_search_path):
        for file in files:
            file_path = os.path.join(root, file)      
            if 'xml' in file_path:
                xml_file.append(file_path)
    
    print(f"找到 {len(xml_file)} 個XML文件")
    
    if len(xml_file) == 0:
        print(f"病例 {patient_id} 沒有XML標注文件")
        return None
    
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
    
    print(f"匹配成功: {len(matched_pairs)} 個DICOM-XML對")
    
    if len(matched_pairs) == 0:
        print(f"病例 {patient_id} 沒有匹配的DICOM-XML對")
        return None
    
    # === 創建目標目錄 ===
    base_output_dir = r'D:\GitHub\chest-ct-report-generator\matched_data_by_patient'
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
    
    print(f"開始複製病例 {patient_id} 的文件...")
    
    for i, pair in enumerate(matched_pairs):
        try:
            # 複製DICOM文件
            dcm_filename = os.path.basename(pair['dcm_file'])
            # 使用原始文件名加上序號，避免重名
            dcm_dest = os.path.join(dcm_output_dir, f"{patient_id}_{i+1:03d}_{dcm_filename}")
            shutil.copy2(pair['dcm_file'], dcm_dest)
            
            # 複製XML文件
            xml_filename = os.path.basename(pair['xml_file'])
            xml_dest = os.path.join(xml_output_dir, xml_filename)
            shutil.copy2(pair['xml_file'], xml_dest)
            
            # 記錄複製的文件信息
            copied_files.append({
                'index': i + 1,
                'patient_id': patient_id,
                'uid': pair['dcm_uid'],
                'original_dcm': pair['dcm_file'],
                'copied_dcm': dcm_dest,
                'original_xml': pair['xml_file'],
                'copied_xml': xml_dest,
                'dcm_size': os.path.getsize(pair['dcm_file']),
                'xml_size': os.path.getsize(pair['xml_file'])
            })
            
            if (i + 1) % 5 == 0:
                print(f"  已複製 {i + 1}/{len(matched_pairs)} 個文件對...")
                
        except Exception as e:
            error_info = {
                'index': i + 1,
                'patient_id': patient_id,
                'uid': pair['dcm_uid'],
                'error': str(e),
                'dcm_file': pair['dcm_file'],
                'xml_file': pair['xml_file']
            }
            copy_errors.append(error_info)
            print(f"複製第 {i + 1} 個文件對時出錯: {e}")
    
    print(f"病例 {patient_id} 複製完成!")
    print(f"  成功複製: {len(copied_files)} 個文件對")
    print(f"  複製失敗: {len(copy_errors)} 個文件對")
    
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
    
    return {
        'patient_id': patient_id,
        'matched_pairs': len(matched_pairs),
        'copied_files': len(copied_files),
        'copy_errors': len(copy_errors),
        'patient_dir': patient_output_dir
    }

def copy_all_patients_matched_files():
    """處理所有病例的匹配文件"""
    
    # 獲取所有可用的病例ID
    patients_dir = r'D:\GitHub\chest-ct-report-generator\datasets\Lung-PET-CT-Dx\manifest-1608669183333\Lung-PET-CT-Dx'
    patient_ids = []
    
    if os.path.exists(patients_dir):
        for item in os.listdir(patients_dir):
            if item.startswith('Lung_Dx-') and os.path.isdir(os.path.join(patients_dir, item)):
                patient_id = item.replace('Lung_Dx-', '')
                patient_ids.append(patient_id)
    
    patient_ids.sort()
    print(f"找到 {len(patient_ids)} 個病例: {patient_ids[:10]}..." if len(patient_ids) > 10 else f"找到 {len(patient_ids)} 個病例: {patient_ids}")
    
    # 處理所有病例
    results = []
    total_copied = 0
    total_errors = 0
    
    base_output_dir = r'D:\GitHub\chest-ct-report-generator\matched_data_by_patient'
    
    # 清除之前的總體統計
    summary_path = os.path.join(base_output_dir, 'all_patients_summary.txt')
    if os.path.exists(summary_path):
        os.remove(summary_path)
    
    for i, patient_id in enumerate(patient_ids):
        print(f"\n>>> 處理進度: {i+1}/{len(patient_ids)} <<<")
        result = copy_patient_matched_files(patient_id)
        
        if result:
            results.append(result)
            total_copied += result['copied_files']
            total_errors += result['copy_errors']
            
            # 追加到總體統計
            with open(summary_path, 'a', encoding='utf-8') as f:
                f.write(f"病例 {patient_id}:\n")
                f.write(f"  - 匹配的文件對: {result['matched_pairs']}\n")
                f.write(f"  - 成功複製: {result['copied_files']}\n")
                f.write(f"  - 複製失敗: {result['copy_errors']}\n")
                f.write(f"  - 處理時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"  - 資料夾: {result['patient_dir']}\n\n")
    
    # 生成總體統計
    with open(summary_path, 'a', encoding='utf-8') as f:
        f.write("="*50 + "\n")
        f.write("總體統計:\n")
        f.write(f"  - 處理的病例數: {len(results)}\n")
        f.write(f"  - 總成功複製文件對: {total_copied}\n")
        f.write(f"  - 總複製失敗: {total_errors}\n")
        f.write(f"  - 完成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"\n=== 所有病例處理完成 ===")
    print(f"成功處理病例數: {len(results)}")
    print(f"總複製文件對: {total_copied}")
    print(f"總複製失敗: {total_errors}")
    print(f"總體統計文件: {summary_path}")
    print(f"所有病例資料夾: {base_output_dir}")

if __name__ == "__main__":
    # 可以選擇處理單個病例或所有病例
    import sys
    
    if len(sys.argv) > 1:
        # 處理指定病例
        patient_id = sys.argv[1]
        result = copy_patient_matched_files(patient_id)
        if result:
            print(f"\n病例 {patient_id} 處理完成:")
            print(f"  複製文件對: {result['copied_files']}")
            print(f"  資料夾: {result['patient_dir']}")
    else:
        # 處理所有病例
        copy_all_patients_matched_files()
