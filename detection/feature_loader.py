#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深層特徵加載和處理工具
用於加載提取的深層特徵並為LLM生成報告提供結構化數據

主要功能：
1. 加載病例深層特徵
2. 特徵向量處理和歸一化
3. 為LLM生成結構化的特徵描述
4. 特徵可視化和分析

作者: GitHub Copilot  
日期: 2025-09-03
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path

class FeatureLoader:
    """深層特徵加載器"""
    
    def __init__(self, features_dir: str):
        """
        初始化特徵加載器
        
        Args:
            features_dir: 特徵文件目錄
        """
        self.features_dir = Path(features_dir)
        if not self.features_dir.exists():
            raise FileNotFoundError(f"特徵目錄不存在: {features_dir}")
        
        self.features_cache = {}
        self._load_all_features()
    
    def _load_all_features(self):
        """加載所有特徵文件"""
        logging.info(f"加載特徵文件從: {self.features_dir}")
        
        # 查找所有特徵文件 - 支持兩種結構
        pkl_files = []
        json_files = []
        
        # 新結構：每個病例一個資料夾
        for patient_dir in self.features_dir.iterdir():
            if patient_dir.is_dir():
                # 檢查資料夾內的特徵文件
                patient_pkl = list(patient_dir.glob("*_features.pkl"))
                patient_json = list(patient_dir.glob("*_features.json"))
                pkl_files.extend(patient_pkl)
                json_files.extend(patient_json)
        
        # 舊結構：所有文件在同一目錄
        if not pkl_files:
            pkl_files = list(self.features_dir.glob("*_features.pkl"))
            json_files = list(self.features_dir.glob("*_features.json"))
        
        logging.info(f"找到 {len(pkl_files)} 個pkl文件，{len(json_files)} 個json文件")
        
        # 優先使用pkl文件（包含完整的numpy數組）
        for pkl_file in pkl_files:
            patient_id = pkl_file.stem.replace("_features", "")
            try:
                with open(pkl_file, 'rb') as f:
                    features = pickle.load(f)
                self.features_cache[patient_id] = features
            except Exception as e:
                logging.warning(f"加載特徵文件失敗 {pkl_file}: {str(e)}")
        
        # 如果沒有pkl文件，嘗試加載json文件
        if not self.features_cache:
            for json_file in json_files:
                patient_id = json_file.stem.replace("_features", "")
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        features = json.load(f)
                    self.features_cache[patient_id] = features
                except Exception as e:
                    logging.warning(f"加載特徵文件失敗 {json_file}: {str(e)}")
        
        logging.info(f"成功加載 {len(self.features_cache)} 個病例的特徵")
    
    def get_patient_features(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """獲取指定病例的特徵"""
        return self.features_cache.get(patient_id)
    
    def get_all_patient_ids(self) -> List[str]:
        """獲取所有病例ID"""
        return list(self.features_cache.keys())
    
    def get_detection_summary(self, patient_id: str) -> Dict[str, Any]:
        """獲取病例的檢測結果摘要"""
        features = self.get_patient_features(patient_id)
        if not features:
            return {}
        
        detection_features = features.get('detection_features', {})
        
        summary = {
            'patient_id': patient_id,
            'has_lesions': detection_features.get('total_detections', 0) > 0,
            'total_detections': detection_features.get('total_detections', 0),
            'slices_with_lesions': detection_features.get('slices_with_lesions', 0),
            'lesion_ratio': detection_features.get('lesion_ratio', 0),
            'total_lesion_volume': detection_features.get('total_lesion_volume', 0),
            'avg_confidence': detection_features.get('avg_confidence', 0),
            'max_confidence': detection_features.get('max_confidence', 0),
        }
        
        # 添加嚴重程度評估
        summary['severity_level'] = self._assess_severity(detection_features)
        
        return summary
    
    def _assess_severity(self, detection_features: Dict[str, Any]) -> str:
        """評估病灶嚴重程度"""
        total_detections = detection_features.get('total_detections', 0)
        lesion_ratio = detection_features.get('lesion_ratio', 0)
        total_volume = detection_features.get('total_lesion_volume', 0)
        avg_confidence = detection_features.get('avg_confidence', 0)
        
        if total_detections == 0:
            return "無病灶"
        
        # 基於多個指標評估嚴重程度
        severity_score = 0
        
        # 檢測數量評分
        if total_detections >= 10:
            severity_score += 3
        elif total_detections >= 5:
            severity_score += 2
        elif total_detections >= 1:
            severity_score += 1
        
        # 病灶比例評分
        if lesion_ratio >= 0.3:
            severity_score += 3
        elif lesion_ratio >= 0.15:
            severity_score += 2
        elif lesion_ratio >= 0.05:
            severity_score += 1
        
        # 病灶體積評分
        if total_volume >= 10000:
            severity_score += 3
        elif total_volume >= 5000:
            severity_score += 2
        elif total_volume >= 1000:
            severity_score += 1
        
        # 置信度評分
        if avg_confidence >= 0.8:
            severity_score += 2
        elif avg_confidence >= 0.6:
            severity_score += 1
        
        # 根據總分評估嚴重程度
        if severity_score >= 8:
            return "重度"
        elif severity_score >= 5:
            return "中度"
        elif severity_score >= 2:
            return "輕度"
        else:
            return "疑似"
    
    def generate_llm_prompt_features(self, patient_id: str) -> str:
        """為LLM生成結構化的特徵描述"""
        features = self.get_patient_features(patient_id)
        if not features:
            return f"患者 {patient_id} 的特徵數據不可用。"
        
        # 基本信息
        num_slices = features.get('num_slices', 0)
        
        # 檢測特徵
        detection_features = features.get('detection_features', {})
        detection_summary = self.get_detection_summary(patient_id)
        
        # 構建結構化描述
        prompt_parts = []
        
        # 基本掃描信息
        prompt_parts.append(f"患者ID: {patient_id}")
        prompt_parts.append(f"CT掃描切片數: {num_slices}")
        
        # 病灶檢測結果
        if detection_summary['has_lesions']:
            prompt_parts.append(f"檢測結果: 發現病灶")
            prompt_parts.append(f"病灶總數: {detection_summary['total_detections']}")
            prompt_parts.append(f"含病灶切片數: {detection_summary['slices_with_lesions']}")
            prompt_parts.append(f"病灶分布比例: {detection_summary['lesion_ratio']:.1%}")
            prompt_parts.append(f"病灶總體積(近似): {detection_summary['total_lesion_volume']:.0f}")
            prompt_parts.append(f"平均檢測置信度: {detection_summary['avg_confidence']:.3f}")
            prompt_parts.append(f"最高檢測置信度: {detection_summary['max_confidence']:.3f}")
            prompt_parts.append(f"嚴重程度評估: {detection_summary['severity_level']}")
            
            # 詳細的切片分析
            slice_features = features.get('slice_features', [])
            if slice_features:
                lesion_slices = [s for s in slice_features if s['num_detections'] > 0]
                if lesion_slices:
                    prompt_parts.append(f"病灶分布詳情:")
                    
                    # 按切片位置分組
                    total_slices = len(slice_features)
                    upper_third = [s for s in lesion_slices if s['slice_idx'] < total_slices // 3]
                    middle_third = [s for s in lesion_slices if total_slices // 3 <= s['slice_idx'] < 2 * total_slices // 3]
                    lower_third = [s for s in lesion_slices if s['slice_idx'] >= 2 * total_slices // 3]
                    
                    if upper_third:
                        prompt_parts.append(f"  上段(肺尖): {len(upper_third)} 個切片有病灶")
                    if middle_third:
                        prompt_parts.append(f"  中段: {len(middle_third)} 個切片有病灶")
                    if lower_third:
                        prompt_parts.append(f"  下段(肺底): {len(lower_third)} 個切片有病灶")
                    
                    # 最大病灶信息
                    max_lesion_slice = max(lesion_slices, key=lambda x: x['total_lesion_area'])
                    prompt_parts.append(f"  最大病灶位於第 {max_lesion_slice['slice_idx']+1} 切片，面積: {max_lesion_slice['total_lesion_area']:.0f}")
        else:
            prompt_parts.append(f"檢測結果: 未發現明顯病灶")
            prompt_parts.append(f"嚴重程度評估: {detection_summary['severity_level']}")
        
        # 影像特徵總結
        prompt_parts.append(f"\n深層特徵分析:")
        global_features = features.get('global_features', {})
        
        # 簡化的特徵描述（避免過多技術細節）
        available_features = []
        if 'backbone_layer4' in global_features:
            available_features.append("高級語義特徵")
        if 'fpn' in global_features:
            available_features.append("多尺度特徵")
        if 'roi' in global_features:
            available_features.append("區域特徵")
        
        if available_features:
            prompt_parts.append(f"  可用特徵類型: {', '.join(available_features)}")
        
        return '\n'.join(prompt_parts)
    
    def get_statistical_summary(self) -> Dict[str, Any]:
        """獲取所有病例的統計摘要"""
        if not self.features_cache:
            return {}
        
        summaries = []
        for patient_id in self.features_cache.keys():
            summary = self.get_detection_summary(patient_id)
            summaries.append(summary)
        
        # 統計分析
        df = pd.DataFrame(summaries)
        
        stats = {
            'total_patients': len(summaries),
            'patients_with_lesions': df['has_lesions'].sum(),
            'lesion_detection_rate': df['has_lesions'].mean(),
            'avg_detections_per_patient': df['total_detections'].mean(),
            'avg_lesion_ratio': df['lesion_ratio'].mean(),
            'avg_confidence': df['avg_confidence'].mean(),
        }
        
        # 嚴重程度分布
        severity_counts = df['severity_level'].value_counts().to_dict()
        stats['severity_distribution'] = severity_counts
        
        return stats

class FeatureVisualizer:
    """特徵可視化工具"""
    
    def __init__(self, feature_loader: FeatureLoader):
        self.loader = feature_loader
    
    def create_patient_report(self, patient_id: str, save_path: Optional[str] = None) -> str:
        """創建病例特徵報告"""
        features = self.loader.get_patient_features(patient_id)
        if not features:
            return f"患者 {patient_id} 的特徵數據不可用。"
        
        # 生成詳細報告
        report_lines = []
        report_lines.append(f"# 患者 {patient_id} 深層特徵報告")
        report_lines.append(f"生成時間: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 基本信息
        report_lines.append("## 基本信息")
        report_lines.append(f"- CT切片數量: {features.get('num_slices', 0)}")
        
        # 檢測結果
        detection_summary = self.loader.get_detection_summary(patient_id)
        report_lines.append("")
        report_lines.append("## 病灶檢測結果")
        
        if detection_summary['has_lesions']:
            report_lines.append(f"- 病灶狀態: ✅ 檢測到病灶")
            report_lines.append(f"- 病灶總數: {detection_summary['total_detections']}")
            report_lines.append(f"- 含病灶切片: {detection_summary['slices_with_lesions']}")
            report_lines.append(f"- 病灶分布比例: {detection_summary['lesion_ratio']:.1%}")
            report_lines.append(f"- 估計總體積: {detection_summary['total_lesion_volume']:.0f}")
            report_lines.append(f"- 平均置信度: {detection_summary['avg_confidence']:.3f}")
            report_lines.append(f"- 最高置信度: {detection_summary['max_confidence']:.3f}")
            report_lines.append(f"- 嚴重程度: **{detection_summary['severity_level']}**")
        else:
            report_lines.append(f"- 病灶狀態: ❌ 未檢測到明顯病灶")
            report_lines.append(f"- 嚴重程度: **{detection_summary['severity_level']}**")
        
        # 切片詳情
        slice_features = features.get('slice_features', [])
        if slice_features:
            report_lines.append("")
            report_lines.append("## 切片分析詳情")
            
            lesion_slices = [s for s in slice_features if s['num_detections'] > 0]
            if lesion_slices:
                report_lines.append(f"含病灶的切片數: {len(lesion_slices)}")
                report_lines.append("")
                report_lines.append("### 主要病灶切片")
                
                # 顯示前5個最嚴重的切片
                top_slices = sorted(lesion_slices, key=lambda x: x['total_lesion_area'], reverse=True)[:5]
                for i, slice_info in enumerate(top_slices, 1):
                    report_lines.append(f"{i}. 切片 {slice_info['slice_idx']+1}:")
                    report_lines.append(f"   - 病灶數量: {slice_info['num_detections']}")
                    report_lines.append(f"   - 病灶面積: {slice_info['total_lesion_area']:.0f}")
                    report_lines.append(f"   - 最高置信度: {slice_info['max_confidence']:.3f}")
        
        # LLM特徵摘要
        report_lines.append("")
        report_lines.append("## LLM特徵摘要")
        llm_features = self.loader.generate_llm_prompt_features(patient_id)
        report_lines.append("```")
        report_lines.append(llm_features)
        report_lines.append("```")
        
        report_content = '\n'.join(report_lines)
        
        # 保存報告
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
        
        return report_content
    
    def create_dataset_summary(self, save_path: Optional[str] = None) -> str:
        """創建數據集統計摘要"""
        stats = self.loader.get_statistical_summary()
        
        report_lines = []
        report_lines.append("# 數據集深層特徵統計摘要")
        report_lines.append(f"生成時間: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        report_lines.append("## 總體統計")
        report_lines.append(f"- 總病例數: {stats.get('total_patients', 0)}")
        report_lines.append(f"- 含病灶病例: {stats.get('patients_with_lesions', 0)}")
        report_lines.append(f"- 病灶檢出率: {stats.get('lesion_detection_rate', 0):.1%}")
        report_lines.append(f"- 平均病灶數/病例: {stats.get('avg_detections_per_patient', 0):.1f}")
        report_lines.append(f"- 平均病灶比例: {stats.get('avg_lesion_ratio', 0):.1%}")
        report_lines.append(f"- 平均檢測置信度: {stats.get('avg_confidence', 0):.3f}")
        
        # 嚴重程度分布
        severity_dist = stats.get('severity_distribution', {})
        if severity_dist:
            report_lines.append("")
            report_lines.append("## 嚴重程度分布")
            for severity, count in severity_dist.items():
                percentage = count / stats.get('total_patients', 1) * 100
                report_lines.append(f"- {severity}: {count} 例 ({percentage:.1f}%)")
        
        report_content = '\n'.join(report_lines)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
        
        return report_content

def main():
    """主函數 - 演示用法"""
    import argparse
    
    parser = argparse.ArgumentParser(description="深層特徵加載和分析工具")
    parser.add_argument('--features_dir', type=str, required=True,
                       help="特徵文件目錄")
    parser.add_argument('--patient_id', type=str, default=None,
                       help="指定病例ID生成報告")
    parser.add_argument('--output_dir', type=str, default='./feature_reports',
                       help="報告輸出目錄")
    
    args = parser.parse_args()
    
    # 設置日誌
    logging.basicConfig(level=logging.INFO)
    
    # 載入特徵
    try:
        loader = FeatureLoader(args.features_dir)
        visualizer = FeatureVisualizer(loader)
        
        # 創建輸出目錄
        os.makedirs(args.output_dir, exist_ok=True)
        
        if args.patient_id:
            # 生成指定病例報告
            if args.patient_id in loader.get_all_patient_ids():
                report_path = os.path.join(args.output_dir, f"{args.patient_id}_report.md")
                report = visualizer.create_patient_report(args.patient_id, report_path)
                print(f"病例報告已保存: {report_path}")
                
                # 顯示LLM特徵
                print("\n=== LLM特徵摘要 ===")
                print(loader.generate_llm_prompt_features(args.patient_id))
            else:
                print(f"錯誤: 找不到病例 {args.patient_id}")
                print(f"可用病例: {loader.get_all_patient_ids()[:10]}...")
        else:
            # 生成數據集摘要
            summary_path = os.path.join(args.output_dir, "dataset_summary.md")
            summary = visualizer.create_dataset_summary(summary_path)
            print(f"數據集摘要已保存: {summary_path}")
            
            # 顯示統計信息
            stats = loader.get_statistical_summary()
            print("\n=== 數據集統計 ===")
            print(f"總病例數: {stats.get('total_patients', 0)}")
            print(f"含病灶病例: {stats.get('patients_with_lesions', 0)}")
            print(f"病灶檢出率: {stats.get('lesion_detection_rate', 0):.1%}")
            
    except Exception as e:
        print(f"錯誤: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
