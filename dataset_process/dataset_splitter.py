#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
資料集劃分工具 (K-Fold交叉驗證版本)
將胸部CT資料集按照訓練/測試比例進行劃分，適用於K-Fold交叉驗證
支援預處理後的 YOLO 格式資料 (images/ 和 labels/ 結構)

作者: GitHub Copilot
日期: 2025-07-28
"""

import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import argparse

def load_config(config_path: str = "../config.json") -> Dict:
    """載入配置文件"""
    config_file = Path(__file__).parent / config_path
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_file}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)


class DatasetSplitter:
    """資料集劃分器 (K-Fold版本)"""
    
    def __init__(self, 
                 source_dir: str = None,
                 output_dir: str = None,
                 train_ratio: float = 0.8,
                 test_ratio: float = 0.2,
                 random_seed: int = 42,
                 config_path: str = "../config.json"):
        """初始化資料集劃分器"""
        # 載入配置
        config = load_config(config_path)
        
        # 設置路徑
        if source_dir is None:
            source_dir = Path(__file__).parent / ".." / config["data"]["all_patient_data_dir"]
        if output_dir is None:
            output_dir = Path(__file__).parent / ".." / config["data"]["dataset_splits_dir"]
        
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        
        # 驗證比例
        if abs(train_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError(f"比例總和必須為1.0，當前為: {train_ratio + test_ratio}")
        
        random.seed(random_seed)
        
        # 統計資訊
        self.stats = {
            'total_patients': 0,
            'series_distribution': defaultdict(int),
            'train_stats': {'patient_count': 0, 'series_count': {}},
            'test_stats': {'patient_count': 0, 'series_count': {}}
        }
    
    def scan_patients(self) -> Dict[str, Dict]:
        """掃描所有患者資料"""
        patients_info = {}
        
        print("🔍 掃描患者資料...")
        for patient_dir in self.source_dir.iterdir():
            if not patient_dir.is_dir():
                continue
            
            patient_id = patient_dir.name
            series = patient_id[0]  # A, B, E, G
            
            # 檢查必要文件夾 (YOLO 格式: images/ 和 labels/)
            images_dir = patient_dir / "images"
            labels_dir = patient_dir / "labels"
            
            if images_dir.exists() and labels_dir.exists():
                patients_info[patient_id] = {
                    'series': series,
                    'path': str(patient_dir)
                }
                
                self.stats['total_patients'] += 1
                self.stats['series_distribution'][series] += 1
        
        print(f"✅ 掃描完成，共找到 {len(patients_info)} 個有效患者")
        return patients_info
    
    def stratified_split(self, patients_info: Dict[str, Dict]) -> Tuple[List[str], List[str]]:
        """按系列進行分層劃分"""
        # 按系列分組
        series_groups = defaultdict(list)
        for patient_id, info in patients_info.items():
            series_groups[info['series']].append(patient_id)
        
        train_patients = []
        test_patients = []
        
        print("📊 進行分層劃分...")
        for series, patients in series_groups.items():
            patients_shuffled = patients.copy()
            random.shuffle(patients_shuffled)
            
            n_patients = len(patients_shuffled)
            n_train = int(n_patients * self.train_ratio)
            
            train_series = patients_shuffled[:n_train]
            test_series = patients_shuffled[n_train:]
            
            train_patients.extend(train_series)
            test_patients.extend(test_series)
            
            print(f"  {series}系列: 總計={n_patients:3d}, 訓練={len(train_series):3d}, 測試={len(test_series):3d}")
        
        return train_patients, test_patients
    
    def copy_patient_data(self, patient_ids: List[str], split_name: str) -> None:
        """複製患者資料到對應的劃分目錄"""
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
    
    def save_patient_lists(self, train_patients: List[str], test_patients: List[str]) -> None:
        """保存患者列表文件"""
        lists_to_save = {
            'train_patients.txt': train_patients,
            'test_patients.txt': test_patients
        }
        
        for filename, patient_list in lists_to_save.items():
            file_path = self.output_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                for patient_id in sorted(patient_list):
                    f.write(f"{patient_id}\n")
    
    def calculate_split_stats(self, patients_info: Dict[str, Dict], 
                            train_patients: List[str], test_patients: List[str]) -> None:
        """計算各劃分的統計資訊"""
        # 統計訓練集
        train_series_count = defaultdict(int)
        for patient_id in train_patients:
            series = patients_info[patient_id]['series']
            train_series_count[series] += 1
        
        # 統計測試集
        test_series_count = defaultdict(int)
        for patient_id in test_patients:
            series = patients_info[patient_id]['series']
            test_series_count[series] += 1
        
        self.stats['train_stats']['patient_count'] = len(train_patients)
        self.stats['train_stats']['series_count'] = dict(train_series_count)
        self.stats['test_stats']['patient_count'] = len(test_patients)
        self.stats['test_stats']['series_count'] = dict(test_series_count)
    
    def generate_report(self, train_patients: List[str], test_patients: List[str]) -> None:
        """生成簡化報告"""
        # 生成簡化的JSON報告
        json_report = {
            'random_seed': self.random_seed,
            'split_ratios': {
                'train': self.train_ratio,
                'test': self.test_ratio
            },
            'total_patients': self.stats['total_patients'],
            'series_distribution': dict(self.stats['series_distribution']),
            'splits': {
                'train': {
                    'patients': sorted(train_patients),
                    'count': len(train_patients),
                    'series_count': self.stats['train_stats']['series_count']
                },
                'test': {
                    'patients': sorted(test_patients),
                    'count': len(test_patients),
                    'series_count': self.stats['test_stats']['series_count']
                }
            }
        }
        
        json_path = self.output_dir / "dataset_split_report.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 報告已生成: {json_path}")
    
    def split_dataset(self) -> None:
        """執行資料集劃分的主要函數"""
        print("🚀 開始資料集劃分 (K-Fold版本)...")
        print("📋 資料格式: YOLO 格式 (每個患者包含 images/ 和 labels/ 目錄)")
        print(f"源目錄: {self.source_dir}")
        print(f"輸出目錄: {self.output_dir}")
        print(f"劃分比例: 訓練={self.train_ratio:.1%}, 測試={self.test_ratio:.1%}")
        print(f"隨機種子: {self.random_seed}")
        print("-" * 60)
        
        # 掃描患者資料
        patients_info = self.scan_patients()
        if not patients_info:
            raise ValueError("沒有找到有效的患者資料")
        
        # 分層劃分
        train_patients, test_patients = self.stratified_split(patients_info)
        
        # 計算統計資訊
        self.calculate_split_stats(patients_info, train_patients, test_patients)
        
        # 創建輸出目錄
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 複製資料
        self.copy_patient_data(train_patients, "train")
        self.copy_patient_data(test_patients, "test")
        
        # 保存患者列表
        self.save_patient_lists(train_patients, test_patients)
        
        # 生成報告
        self.generate_report(train_patients, test_patients)
        
        print("-" * 60)
        print("🎉 資料集劃分完成!")
        print(f"  訓練集: {len(train_patients)} 個患者")
        print(f"  測試集: {len(test_patients)} 個患者")
        print(f"  總計: {len(train_patients) + len(test_patients)} 個患者")

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="胸部CT資料集劃分工具 (K-Fold版本) - 支援 YOLO 格式")
    parser.add_argument("--source_dir", type=str, default=None,
                       help="源資料目錄路徑 (預處理後的 YOLO 格式資料)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="輸出目錄路徑")
    parser.add_argument("--train_ratio", type=float, default=0.9,
                       help="訓練集比例 (預設: 90% )")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                       help="測試集比例 (預設: 10% )")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="隨機種子 (預設: 42)")
    parser.add_argument("--config", type=str, default="../config.json",
                       help="配置文件路徑")
    
    args = parser.parse_args()
    
    try:
        splitter = DatasetSplitter(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            test_ratio=args.test_ratio,
            random_seed=args.random_seed,
            config_path=args.config
        )
        
        splitter.split_dataset()
        
    except Exception as e:
        print(f"❌ 錯誤: {str(e)}")
        raise


if __name__ == "__main__":
    main()
