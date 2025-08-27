#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
檢查資料集大小的腳本
"""

import os
import json
from pathlib import Path

def check_data_size():
    data_root = os.path.join('..', 'datasets', 'splited_dataset')
    train_dir = os.path.join(data_root, 'train')
    
    print(f'檢查資料目錄: {train_dir}')
    print(f'目錄存在: {os.path.exists(train_dir)}')
    
    if not os.path.exists(train_dir):
        print("資料目錄不存在！")
        return
    
    patients = [f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))]
    print(f'患者數量: {len(patients)}')
    
    # 檢查前5個患者的檔案數量
    total_files = 0
    sample_count = min(5, len(patients))
    
    for i, patient in enumerate(patients[:sample_count]):
        patient_path = os.path.join(train_dir, patient)
        json_file = os.path.join(patient_path, f'{patient}_file_list.json')
        
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                dcm_count = len(json_data.get('copied_dcm_files', []))
                xml_count = len(json_data.get('copied_xml_files', []))
                matched_count = min(dcm_count, xml_count)
                
                print(f'患者 {patient}: DICOM={dcm_count}, XML={xml_count}, 配對={matched_count}')
                total_files += matched_count
                
            except Exception as e:
                print(f'讀取 {patient} 的JSON檔案失敗: {e}')
        else:
            print(f'患者 {patient}: 找不到JSON檔案')
    
    if sample_count > 0:
        # 估算總樣本數
        avg_per_patient = total_files / sample_count
        estimated_total = int(avg_per_patient * len(patients))
        
        print(f'\n統計結果:')
        print(f'前{sample_count}位患者平均檔案數: {avg_per_patient:.1f}')
        print(f'估算總樣本數: {estimated_total}')
        print(f'移除限制前（每患者15張）: {len(patients) * 15}')
        print(f'移除限制前（每患者10張）: {len(patients) * 10}')
        print(f'相比10張/患者增加倍數: {estimated_total / (len(patients) * 10):.1f}x')
        print(f'相比15張/患者增加倍數: {estimated_total / (len(patients) * 15):.1f}x')
        
        # 記憶體使用估算
        estimated_memory_gb = estimated_total * 512 * 512 * 4 / (1024**3)  # 假設float32
        print(f'預估記憶體需求: {estimated_memory_gb:.1f} GB')

if __name__ == "__main__":
    check_data_size()
