# 複製所有病例的DICOM文件和XML文件，按病例分類整理（不篩選配對）
import os
import shutil
import json
from datetime import datetime

# 讀取 config 設定
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.json'))
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

dicom_root = config['data']['dicom_root']
xml_root = config['data']['xml_root']
base_output_dir = config['data']['all_patient_data_dir']

def create_directory_if_not_exists(path):
    """如果目錄不存在則創建"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"創建目錄: {path}")

def copy_patient_all_files(patient_id):
    """複製指定病例的所有DICOM和XML文件"""
    
    import pydicom
    
    print(f"\n=== 處理病例 {patient_id} ===")
    
    # DICOM文件搜索
    dcm_files = []
    search_path = os.path.join(dicom_root, f'Lung_Dx-{patient_id}')
    
    if os.path.exists(search_path):
        for root, dirs, files in os.walk(search_path):
            for file in files:
                file_path = os.path.join(root, file)
                if 'dcm' in file_path or file.lower().endswith('.dcm'):
                    dcm_files.append(file_path)
        print(f"找到 {len(dcm_files)} 個DICOM文件")
    else:
        print(f"警告: 病例 {patient_id} 的DICOM資料夾不存在: {search_path}")
    
    # XML文件搜索
    xml_files = []
    xml_search_path = os.path.join(xml_root, patient_id)
    
    if os.path.exists(xml_search_path):
        for root, dirs, files in os.walk(xml_search_path):
            for file in files:
                file_path = os.path.join(root, file)      
                if 'xml' in file_path or file.lower().endswith('.xml'):
                    xml_files.append(file_path)
        print(f"找到 {len(xml_files)} 個XML文件")
    else:
        print(f"警告: 病例 {patient_id} 的XML標注資料夾不存在: {xml_search_path}")
    
    # 如果DICOM和XML文件都沒有找到，跳過此病例
    if len(dcm_files) == 0 and len(xml_files) == 0:
        print(f"病例 {patient_id} 沒有找到任何DICOM或XML文件")
        return None
    
    # === 創建目標目錄 ===
    patient_output_dir = os.path.join(base_output_dir, patient_id)
    dcm_output_dir = os.path.join(patient_output_dir, 'dicom_files')
    xml_output_dir = os.path.join(patient_output_dir, 'xml_annotations')
    
    create_directory_if_not_exists(base_output_dir)
    create_directory_if_not_exists(patient_output_dir)
    create_directory_if_not_exists(dcm_output_dir)
    create_directory_if_not_exists(xml_output_dir)
    
    # === 複製文件並記錄 ===
    copied_dcm_files = []
    copied_xml_files = []
    copy_errors = []
    
    print(f"開始複製病例 {patient_id} 的文件...")
    
    # 複製DICOM文件
    for i, dcm_file in enumerate(dcm_files):
        try:
            dcm_filename = os.path.basename(dcm_file)
            # 使用原始文件名加上序號，避免重名
            dcm_dest = os.path.join(dcm_output_dir, f"{patient_id}_DCM_{i+1:03d}_{dcm_filename}")
            shutil.copy2(dcm_file, dcm_dest)
            
            # 嘗試讀取DICOM文件的UID信息
            dcm_uid = None
            try:
                im = pydicom.dcmread(dcm_file)
                dcm_uid = im.SOPInstanceUID
            except:
                dcm_uid = f"Unknown_UID_{i+1}"
            
            # 記錄複製的DICOM文件信息
            copied_dcm_files.append({
                'index': i + 1,
                'patient_id': patient_id,
                'uid': dcm_uid,
                'original_file': dcm_file,
                'copied_file': dcm_dest,
                'file_size': os.path.getsize(dcm_file)
            })
            
            if (i + 1) % 10 == 0:
                print(f"  已複製 {i + 1}/{len(dcm_files)} 個DICOM文件...")
                
        except Exception as e:
            error_info = {
                'file_type': 'DICOM',
                'index': i + 1,
                'patient_id': patient_id,
                'error': str(e),
                'original_file': dcm_file
            }
            copy_errors.append(error_info)
            print(f"複製第 {i + 1} 個DICOM文件時出錯: {e}")
    
    # 複製XML文件
    for i, xml_file in enumerate(xml_files):
        try:
            xml_filename = os.path.basename(xml_file)
            xml_dest = os.path.join(xml_output_dir, xml_filename)
            
            # 如果目標文件已存在，添加序號
            if os.path.exists(xml_dest):
                name, ext = os.path.splitext(xml_filename)
                xml_dest = os.path.join(xml_output_dir, f"{name}_{i+1:03d}{ext}")
            
            shutil.copy2(xml_file, xml_dest)
            
            # 記錄複製的XML文件信息
            copied_xml_files.append({
                'index': i + 1,
                'patient_id': patient_id,
                'original_file': xml_file,
                'copied_file': xml_dest,
                'file_size': os.path.getsize(xml_file)
            })
            
            if (i + 1) % 10 == 0:
                print(f"  已複製 {i + 1}/{len(xml_files)} 個XML文件...")
                
        except Exception as e:
            error_info = {
                'file_type': 'XML',
                'index': i + 1,  
                'patient_id': patient_id,
                'error': str(e),
                'original_file': xml_file
            }
            copy_errors.append(error_info)
            print(f"複製第 {i + 1} 個XML文件時出錯: {e}")
    
    print(f"病例 {patient_id} 複製完成!")
    print(f"  成功複製DICOM文件: {len(copied_dcm_files)} 個")
    print(f"  成功複製XML文件: {len(copied_xml_files)} 個")
    print(f"  複製失敗: {len(copy_errors)} 個文件")
    
    # === 生成文件清單 ===
    
    # 1. 生成JSON格式的詳細清單
    file_list = {
        'summary': {
            'patient_id': patient_id,
            'total_dcm_files': len(dcm_files),
            'total_xml_files': len(xml_files),
            'successfully_copied_dcm': len(copied_dcm_files),
            'successfully_copied_xml': len(copied_xml_files),
            'copy_errors': len(copy_errors),
            'creation_time': datetime.now().isoformat(),
            'source_patient': f'Lung_Dx-{patient_id}'
        },
        'copied_dcm_files': copied_dcm_files,
        'copied_xml_files': copied_xml_files,
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
        f.write(f"原始DICOM文件數: {len(dcm_files)}\n")
        f.write(f"原始XML文件數: {len(xml_files)}\n")
        f.write(f"成功複製DICOM: {len(copied_dcm_files)}\n")
        f.write(f"成功複製XML: {len(copied_xml_files)}\n")
        f.write(f"複製失敗: {len(copy_errors)}\n\n")
        
        f.write("=== 成功複製的DICOM文件 ===\n")
        for file_info in copied_dcm_files:
            f.write(f"\nDICOM文件 {file_info['index']}:\n")
            f.write(f"  UID: {file_info['uid']}\n")
            f.write(f"  文件名: {os.path.basename(file_info['copied_file'])}\n")
            f.write(f"  大小: {file_info['file_size']} bytes\n")
        
        f.write(f"\n=== 成功複製的XML文件 ===\n")
        for file_info in copied_xml_files:
            f.write(f"\nXML文件 {file_info['index']}:\n")
            f.write(f"  文件名: {os.path.basename(file_info['copied_file'])}\n")
            f.write(f"  大小: {file_info['file_size']} bytes\n")
        
        if copy_errors:
            f.write(f"\n=== 複製失敗的文件 ===\n")
            for error in copy_errors:
                f.write(f"\n{error['file_type']}文件 {error['index']}:\n")
                f.write(f"  錯誤: {error['error']}\n")
                f.write(f"  原始文件: {os.path.basename(error['original_file'])}\n")
    
    # 3. 生成簡單的CSV清單
    csv_list_path = os.path.join(patient_output_dir, f'{patient_id}_file_list.csv')
    with open(csv_list_path, 'w', encoding='utf-8') as f:
        f.write("File_Type,Index,Patient_ID,UID,File_Name,File_Size\n")
        for file_info in copied_dcm_files:
            f.write(f"DICOM,{file_info['index']},{patient_id},{file_info['uid']},{os.path.basename(file_info['copied_file'])},{file_info['file_size']}\n")
        for file_info in copied_xml_files:
            f.write(f"XML,{file_info['index']},{patient_id},N/A,{os.path.basename(file_info['copied_file'])},{file_info['file_size']}\n")
    
    return {
        'patient_id': patient_id,
        'total_dcm_files': len(dcm_files),
        'total_xml_files': len(xml_files),
        'copied_dcm_files': len(copied_dcm_files),
        'copied_xml_files': len(copied_xml_files),
        'copy_errors': len(copy_errors),
        'patient_dir': patient_output_dir
    }

def copy_all_patients_files():
    """處理所有病例的文件複製"""
    # 獲取所有可用的病例ID
    patients_dir = dicom_root
    patient_ids = []
    
    base_output_dir = config['data']['all_patient_data_dir']
    create_directory_if_not_exists(base_output_dir)
    
    if os.path.exists(patients_dir):
        for item in os.listdir(patients_dir):
            if item.startswith('Lung_Dx-') and os.path.isdir(os.path.join(patients_dir, item)):
                patient_id = item.replace('Lung_Dx-', '')
                patient_ids.append(patient_id)
    
    patient_ids.sort()
    print(f"找到 {len(patient_ids)} 個病例: {patient_ids[:10]}..." if len(patient_ids) > 10 else f"找到 {len(patient_ids)} 個病例: {patient_ids}")
    
    if len(patient_ids) == 0:
        print("未找到任何病例，請確認 patients_dir 路徑及資料夾內容。")
        return
    
    # 處理所有病例
    results = []
    total_dcm_copied = 0
    total_xml_copied = 0
    total_errors = 0
    
    # 清除之前的總體統計
    summary_path = os.path.join(base_output_dir, 'all_patients_summary.txt')
    if os.path.exists(summary_path):
        os.remove(summary_path)
    
    for i, patient_id in enumerate(patient_ids):
        print(f"\n>>> 處理進度: {i+1}/{len(patient_ids)} <<<")
        result = copy_patient_all_files(patient_id)
        
        if result:
            results.append(result)
            total_dcm_copied += result['copied_dcm_files']
            total_xml_copied += result['copied_xml_files']
            total_errors += result['copy_errors']
            
            # 追加到總體統計
            with open(summary_path, 'a', encoding='utf-8') as f:
                f.write(f"病例 {patient_id}:\n")
                f.write(f"  - 原始DICOM文件: {result['total_dcm_files']}\n")
                f.write(f"  - 原始XML文件: {result['total_xml_files']}\n")
                f.write(f"  - 成功複製DICOM: {result['copied_dcm_files']}\n")
                f.write(f"  - 成功複製XML: {result['copied_xml_files']}\n")
                f.write(f"  - 複製失敗: {result['copy_errors']}\n")
                f.write(f"  - 處理時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"  - 資料夾: {result['patient_dir']}\n\n")
    
    # 生成總體統計
    with open(summary_path, 'a', encoding='utf-8') as f:
        f.write("="*50 + "\n")
        f.write("總體統計:\n")
        f.write(f"  - 處理的病例數: {len(results)}\n")
        f.write(f"  - 總成功複製DICOM文件: {total_dcm_copied}\n")
        f.write(f"  - 總成功複製XML文件: {total_xml_copied}\n")
        f.write(f"  - 總複製失敗: {total_errors}\n")
        f.write(f"  - 完成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"\n=== 所有病例處理完成 ===")
    print(f"成功處理病例數: {len(results)}")
    print(f"總複製DICOM文件: {total_dcm_copied}")
    print(f"總複製XML文件: {total_xml_copied}")
    print(f"總複製失敗: {total_errors}")
    print(f"總體統計文件: {summary_path}")
    print(f"所有病例資料夾: {base_output_dir}")

if __name__ == "__main__":
    # 可以選擇處理單個病例或所有病例
    import sys
    
    if len(sys.argv) > 1:
        # 處理指定病例
        patient_id = sys.argv[1]
        result = copy_patient_all_files(patient_id)
        if result:
            print(f"\n病例 {patient_id} 處理完成:")
            print(f"  複製DICOM文件: {result['copied_dcm_files']}")
            print(f"  複製XML文件: {result['copied_xml_files']}")
            print(f"  資料夾: {result['patient_dir']}")
    else:
        # 處理所有病例
        copy_all_patients_files()
