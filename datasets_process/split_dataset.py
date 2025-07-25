#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
資料集劃分工具
將胸部CT資料集按照訓練/驗證/測試比例進行劃分

功能:
- 支援多種患者系列 (A, B, E, G)
- 確保相同患者的所有資料在同一集合中
- 生成詳細的劃分報告
- 支援自訂劃分比例和隨機種子

作者: GitHub Copilot
日期: 2025-07-24
"""

import os
import shutil
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import argparse
from datetime import datetime

class DatasetSplitter:
    """資料集劃分器"""
    
    def __init__(self, 
                 source_dir: str,
                 output_dir: str,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 random_seed: int = 42):
        """
        初始化資料集劃分器
        
        Args:
            source_dir: 源資料目錄 (matched_data_by_patient)
            output_dir: 輸出目錄
            train_ratio: 訓練集比例
            val_ratio: 驗證集比例
            test_ratio: 測試集比例
            random_seed: 隨機種子
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        
        # 驗證比例總和為1
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"比例總和必須為1.0，當前為: {total_ratio}")
        
        # 設置隨機種子
        random.seed(random_seed)
        
        # 統計資訊
        self.stats = {
            'total_patients': 0,
            'total_dicom_files': 0,
            'total_xml_files': 0,
            'series_distribution': defaultdict(int),
            'train_stats': defaultdict(int),
            'val_stats': defaultdict(int),
            'test_stats': defaultdict(int)
        }
    
    def scan_patients(self) -> Dict[str, Dict]:
        """
        掃描所有患者資料
        
        Returns:
            患者資訊字典 {patient_id: {'series': str, 'dicom_count': int, 'xml_count': int}}
        """
        patients_info = {}
        
        print("🔍 掃描患者資料...")
        for patient_dir in self.source_dir.iterdir():
            if not patient_dir.is_dir():
                continue
            
            patient_id = patient_dir.name
            series = patient_id[0]  # A, B, E, G
            
            # 統計DICOM和XML檔案數量
            dicom_dir = patient_dir / "dicom_files"
            xml_dir = patient_dir / "xml_annotations"
            
            dicom_count = len(list(dicom_dir.glob("*.dcm"))) if dicom_dir.exists() else 0
            xml_count = len(list(xml_dir.glob("*.xml"))) if xml_dir.exists() else 0
            
            if dicom_count > 0 and xml_count > 0:
                patients_info[patient_id] = {
                    'series': series,
                    'dicom_count': dicom_count,
                    'xml_count': xml_count,
                    'path': str(patient_dir)
                }
                
                # 更新統計
                self.stats['total_patients'] += 1
                self.stats['total_dicom_files'] += dicom_count
                self.stats['total_xml_files'] += xml_count
                self.stats['series_distribution'][series] += 1
        
        print(f"✅ 掃描完成，共找到 {len(patients_info)} 個有效患者")
        return patients_info
    
    def stratified_split(self, patients_info: Dict[str, Dict]) -> Tuple[List[str], List[str], List[str]]:
        """
        按系列進行分層劃分
        
        Args:
            patients_info: 患者資訊字典
            
        Returns:
            (train_patients, val_patients, test_patients)
        """
        # 按系列分組
        series_groups = defaultdict(list)
        for patient_id, info in patients_info.items():
            series_groups[info['series']].append(patient_id)
        
        train_patients = []
        val_patients = []
        test_patients = []
        
        print("📊 進行分層劃分...")
        for series, patients in series_groups.items():
            # 打亂患者順序
            patients_shuffled = patients.copy()
            random.shuffle(patients_shuffled)
            
            n_patients = len(patients_shuffled)
            n_train = int(n_patients * self.train_ratio)
            n_val = int(n_patients * self.val_ratio)
            n_test = n_patients - n_train - n_val
            
            # 劃分
            train_series = patients_shuffled[:n_train]
            val_series = patients_shuffled[n_train:n_train + n_val]
            test_series = patients_shuffled[n_train + n_val:]
            
            train_patients.extend(train_series)
            val_patients.extend(val_series)
            test_patients.extend(test_series)
            
            print(f"  {series}系列: 總計={n_patients:3d}, 訓練={len(train_series):3d}, 驗證={len(val_series):2d}, 測試={len(test_series):2d}")
        
        return train_patients, val_patients, test_patients
    
    def copy_patient_data(self, patient_ids: List[str], split_name: str) -> None:
        """
        複製患者資料到對應的劃分目錄
        
        Args:
            patient_ids: 患者ID列表
            split_name: 劃分名稱 (train/validation/test)
        """
        split_dir = self.output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📂 複製 {split_name} 資料 ({len(patient_ids)} 個患者)...")
        
        for i, patient_id in enumerate(patient_ids, 1):
            source_path = self.source_dir / patient_id
            target_path = split_dir / patient_id
            
            if source_path.exists():
                if target_path.exists():
                    shutil.rmtree(target_path)
                shutil.copytree(source_path, target_path)
                
                if i % 50 == 0 or i == len(patient_ids):
                    print(f"  進度: {i}/{len(patient_ids)} ({i/len(patient_ids)*100:.1f}%)")
    
    def save_patient_lists(self, train_patients: List[str], val_patients: List[str], test_patients: List[str]) -> None:
        """保存患者列表文件"""
        lists_to_save = {
            'train_patients.txt': train_patients,
            'validation_patients.txt': val_patients,
            'test_patients.txt': test_patients
        }
        
        for filename, patient_list in lists_to_save.items():
            file_path = self.output_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                for patient_id in sorted(patient_list):
                    f.write(f"{patient_id}\n")
    
    def calculate_split_stats(self, patients_info: Dict[str, Dict], 
                            train_patients: List[str], val_patients: List[str], test_patients: List[str]) -> None:
        """計算各劃分的統計資訊"""
        splits = {
            'train': (train_patients, self.stats['train_stats']),
            'val': (val_patients, self.stats['val_stats']),
            'test': (test_patients, self.stats['test_stats'])
        }
        
        for split_name, (patient_list, stats_dict) in splits.items():
            stats_dict['patient_count'] = len(patient_list)
            stats_dict['dicom_count'] = sum(patients_info[p]['dicom_count'] for p in patient_list)
            stats_dict['xml_count'] = sum(patients_info[p]['xml_count'] for p in patient_list)
            
            # 按系列統計
            series_count = defaultdict(int)
            series_patients = defaultdict(list)
            for patient_id in patient_list:
                series = patients_info[patient_id]['series']
                series_count[series] += 1
                series_patients[series].append(patient_id)
            
            stats_dict['series_count'] = dict(series_count)
            stats_dict['series_patients'] = dict(series_patients)
    
    def generate_report(self, train_patients: List[str], val_patients: List[str], test_patients: List[str]) -> None:
        """生成詳細報告"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 生成摘要報告
        summary_path = self.output_dir / "dataset_split_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=== 胸部CT報告生成器 - 資料集劃分報告 ===\n\n")
            f.write(f"生成時間: {timestamp}\n")
            f.write(f"隨機種子: {self.random_seed}\n")
            f.write(f"劃分比例: 訓練 {self.train_ratio*100:.1f}% | 驗證 {self.val_ratio*100:.1f}% | 測試 {self.test_ratio*100:.1f}%\n\n")
            
            f.write("=== 資料集統計 ===\n")
            f.write(f"總病例數: {self.stats['total_patients']}\n")
            f.write(f"總DICOM文件: {self.stats['total_dicom_files']}\n")
            f.write(f"總XML標注文件: {self.stats['total_xml_files']}\n\n")
            
            # 各劃分詳細資訊
            splits_info = [
                ("訓練集", train_patients, self.stats['train_stats']),
                ("驗證集", val_patients, self.stats['val_stats']),
                ("測試集", test_patients, self.stats['test_stats'])
            ]
            
            for split_name, patient_list, stats in splits_info:
                f.write(f"=== {split_name} ===\n")
                f.write(f"病例數: {stats['patient_count']}\n")
                f.write(f"DICOM文件: {stats['dicom_count']}\n")
                f.write(f"XML文件: {stats['xml_count']}\n")
                
                # 按系列列出患者
                for series in sorted(stats['series_patients'].keys()):
                    patients = stats['series_patients'][series]
                    f.write(f"{series}系列病例 ({len(patients)}個): {', '.join(sorted(patients))}\n")
                f.write("\n")
        
        # 生成JSON格式報告
        json_report = {
            'timestamp': timestamp,
            'random_seed': self.random_seed,
            'split_ratios': {
                'train': self.train_ratio,
                'validation': self.val_ratio,
                'test': self.test_ratio
            },
            'overall_stats': {
                'total_patients': self.stats['total_patients'],
                'total_dicom_files': self.stats['total_dicom_files'],
                'total_xml_files': self.stats['total_xml_files'],
                'series_distribution': dict(self.stats['series_distribution'])
            },
            'splits': {
                'train': {
                    'patients': sorted(train_patients),
                    'stats': dict(self.stats['train_stats'])
                },
                'validation': {
                    'patients': sorted(val_patients),
                    'stats': dict(self.stats['val_stats'])
                },
                'test': {
                    'patients': sorted(test_patients),
                    'stats': dict(self.stats['test_stats'])
                }
            }
        }
        
        json_path = self.output_dir / "dataset_split_report.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 報告已生成: {summary_path}")
        print(f"✅ JSON報告已生成: {json_path}")
    
    def split_dataset(self) -> None:
        """執行資料集劃分的主要函數"""
        print("🚀 開始資料集劃分...")
        print(f"源目錄: {self.source_dir}")
        print(f"輸出目錄: {self.output_dir}")
        print(f"劃分比例: 訓練={self.train_ratio:.1%}, 驗證={self.val_ratio:.1%}, 測試={self.test_ratio:.1%}")
        print(f"隨機種子: {self.random_seed}")
        print("-" * 60)
        
        # 1. 掃描患者資料
        patients_info = self.scan_patients()
        
        if not patients_info:
            raise ValueError("沒有找到有效的患者資料")
        
        # 2. 分層劃分
        train_patients, val_patients, test_patients = self.stratified_split(patients_info)
        
        # 3. 計算統計資訊
        self.calculate_split_stats(patients_info, train_patients, val_patients, test_patients)
        
        # 4. 創建輸出目錄
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 5. 複製資料
        self.copy_patient_data(train_patients, "train")
        self.copy_patient_data(val_patients, "validation") 
        self.copy_patient_data(test_patients, "test")
        
        # 6. 保存患者列表
        self.save_patient_lists(train_patients, val_patients, test_patients)
        
        # 7. 生成報告
        self.generate_report(train_patients, val_patients, test_patients)
        
        print("-" * 60)
        print("🎉 資料集劃分完成!")
        print(f"  訓練集: {len(train_patients)} 個患者")
        print(f"  驗證集: {len(val_patients)} 個患者") 
        print(f"  測試集: {len(test_patients)} 個患者")
        print(f"  總計: {len(train_patients) + len(val_patients) + len(test_patients)} 個患者")

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="胸部CT資料集劃分工具")
    parser.add_argument("--source_dir", type=str, 
                       default="d:/GitHub/chest-ct-report-generator/matched_data_by_patient",
                       help="源資料目錄路徑")
    parser.add_argument("--output_dir", type=str,
                       default="d:/GitHub/chest-ct-report-generator/dataset_splits", 
                       help="輸出目錄路徑")
    parser.add_argument("--train_ratio", type=float, default=0.7,
                       help="訓練集比例 (預設: 0.7)")
    parser.add_argument("--val_ratio", type=float, default=0.15,
                       help="驗證集比例 (預設: 0.15)")
    parser.add_argument("--test_ratio", type=float, default=0.15,
                       help="測試集比例 (預設: 0.15)")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="隨機種子 (預設: 42)")
    
    args = parser.parse_args()
    
    try:
        # 創建劃分器並執行
        splitter = DatasetSplitter(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_seed=args.random_seed
        )
        
        splitter.split_dataset()
        
    except Exception as e:
        print(f"❌ 錯誤: {str(e)}")
        raise

if __name__ == "__main__":
    main()
