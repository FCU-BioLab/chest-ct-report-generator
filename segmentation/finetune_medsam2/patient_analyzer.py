"""
患者分析模組

用於分析患者級別的特徵摘要和識別低分患者
"""

import numpy as np
import json
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .visualizer import SegmentationVisualizer


class PatientAnalyzer:
    """
    患者分析器
    
    負責:
    - 計算患者級別特徵摘要
    - 識別低分患者
    - 生成低分患者報告
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        初始化患者分析器
        
        Args:
            logger: 日誌記錄器
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def compute_patient_summary(self, patient_data: Dict) -> Dict:
        """
        計算患者級別的特徵摘要（包含結節類型統計和 3D 結節聚合）
        
        Args:
            patient_data: 患者資料字典
            
        Returns:
            患者摘要字典
        """
        slices = patient_data.get('slices', {})
        
        # 計算 3D 結節數量
        try:
            from .utils import aggregate_3d_nodules
            unique_nodules, nodule_mapping = aggregate_3d_nodules(patient_data)
        except ImportError:
            unique_nodules = 0
            nodule_mapping = {}
        
        summary = {
            'total_slices': len(slices),
            'total_lesions': 0,  # 2D 切片級別計數
            'unique_nodules_3d': unique_nodules,  # 真實 3D 結節數量
            'avg_lesion_area_mm2': 0.0,
            'max_lesion_area_mm2': 0.0,
            'avg_lesion_diameter_mm': 0.0,
            'max_lesion_diameter_mm': 0.0,
            'avg_circularity': 0.0,
            'avg_solidity': 0.0,
            'avg_confidence': 0.0,
            'metrics': {'dice': 0.0, 'iou': 0.0, 'precision': 0.0, 'recall': 0.0},
            # 結節類型統計
            'nodule_type_counts': {
                'solid': 0,
                'part_solid': 0,
                'ground_glass': 0,
                'calcified': 0,
                'unknown': 0
            },
            'nodule_type_distribution': {}
        }
        
        all_areas = []
        all_diameters = []
        all_circularities = []
        all_solidities = []
        all_confidences = []
        all_metrics = {k: [] for k in summary['metrics'].keys()}
        
        for slice_data in slices.values():
            for lesion in slice_data.get('lesions', []):
                summary['total_lesions'] += 1
                
                morph = lesion.get('morphological', {})
                all_areas.append(morph.get('area_mm2', 0))
                all_diameters.append(morph.get('equivalent_diameter_mm', 0))
                all_circularities.append(morph.get('circularity', 0))
                all_solidities.append(morph.get('solidity', 0))
                all_confidences.append(lesion.get('confidence', 0))
                
                # 統計結節類型
                nodule_class = lesion.get('nodule_classification', {})
                nodule_type = nodule_class.get('nodule_type', 'unknown')
                if nodule_type in summary['nodule_type_counts']:
                    summary['nodule_type_counts'][nodule_type] += 1
                else:
                    summary['nodule_type_counts']['unknown'] += 1
                
                for key in all_metrics.keys():
                    all_metrics[key].append(lesion.get('metrics', {}).get(key, 0))
        
        if all_areas:
            summary['avg_lesion_area_mm2'] = float(np.mean(all_areas))
            summary['max_lesion_area_mm2'] = float(np.max(all_areas))
            summary['avg_lesion_diameter_mm'] = float(np.mean(all_diameters))
            summary['max_lesion_diameter_mm'] = float(np.max(all_diameters))
            summary['avg_circularity'] = float(np.mean(all_circularities))
            summary['avg_solidity'] = float(np.mean(all_solidities))
            summary['avg_confidence'] = float(np.mean(all_confidences))
            
            for key, values in all_metrics.items():
                summary['metrics'][key] = float(np.mean(values))
        
        # 計算結節類型百分比分佈
        if summary['total_lesions'] > 0:
            for ntype, count in summary['nodule_type_counts'].items():
                summary['nodule_type_distribution'][ntype] = round(
                    count / summary['total_lesions'] * 100, 1
                )
        
        return summary
    
    def identify_poor_performers(
        self,
        patient_features: Dict,
        dice_threshold: float = 0.5,
        iou_threshold: float = 0.4
    ) -> Dict:
        """
        識別預測結果不好的患者
        
        Args:
            patient_features: 所有患者的特徵資料
            dice_threshold: Dice 分數低於此閾值視為低分
            iou_threshold: IoU 分數低於此閾值視為低分
            
        Returns:
            包含低分患者資訊的字典
        """
        poor_performers = {
            'thresholds': {
                'dice': dice_threshold,
                'iou': iou_threshold
            },
            'patients': [],
            'summary': {
                'total_poor_patients': 0,
                'avg_dice': 0.0,
                'avg_iou': 0.0,
                'worst_dice': 1.0,
                'worst_iou': 1.0
            }
        }
        
        all_poor_dice = []
        all_poor_iou = []
        
        for patient_id, patient_data in patient_features.items():
            summary = patient_data.get('summary', {})
            metrics = summary.get('metrics', {})
            
            patient_dice = metrics.get('dice', 0.0)
            patient_iou = metrics.get('iou', 0.0)
            
            # 判斷是否為低分患者
            is_poor = patient_dice < dice_threshold or patient_iou < iou_threshold
            
            if is_poor:
                # 收集切片級別的詳細資訊
                slice_details = []
                for slice_key, slice_data in patient_data.get('slices', {}).items():
                    for lesion in slice_data.get('lesions', []):
                        lesion_metrics = lesion.get('metrics', {})
                        slice_details.append({
                            'slice_idx': slice_data.get('slice_idx', slice_key),
                            'dice': lesion_metrics.get('dice', 0.0),
                            'iou': lesion_metrics.get('iou', 0.0),
                            'precision': lesion_metrics.get('precision', 0.0),
                            'recall': lesion_metrics.get('recall', 0.0),
                            'lesion_area_mm2': lesion.get('morphological', {}).get('area_mm2', 0.0)
                        })
                
                # 按 dice 分數排序（最差的在前）
                slice_details.sort(key=lambda x: x['dice'])
                
                poor_patient_info = {
                    'patient_id': patient_id,
                    'avg_dice': patient_dice,
                    'avg_iou': patient_iou,
                    'total_slices': summary.get('total_slices', 0),
                    'total_lesions': summary.get('total_lesions', 0),
                    'worst_slices': slice_details[:5],  # 保存最差的 5 個切片
                    'reason': []
                }
                
                if patient_dice < dice_threshold:
                    poor_patient_info['reason'].append(
                        f'Dice ({patient_dice:.4f}) < {dice_threshold}'
                    )
                if patient_iou < iou_threshold:
                    poor_patient_info['reason'].append(
                        f'IoU ({patient_iou:.4f}) < {iou_threshold}'
                    )
                
                poor_performers['patients'].append(poor_patient_info)
                all_poor_dice.append(patient_dice)
                all_poor_iou.append(patient_iou)
        
        # 計算摘要統計
        if poor_performers['patients']:
            poor_performers['summary']['total_poor_patients'] = len(poor_performers['patients'])
            poor_performers['summary']['avg_dice'] = float(np.mean(all_poor_dice))
            poor_performers['summary']['avg_iou'] = float(np.mean(all_poor_iou))
            poor_performers['summary']['worst_dice'] = float(np.min(all_poor_dice))
            poor_performers['summary']['worst_iou'] = float(np.min(all_poor_iou))
            
            # 按平均 dice 分數排序（最差的在前）
            poor_performers['patients'].sort(key=lambda x: x['avg_dice'])
        
        return poor_performers
    
    def save_poor_performers_report(
        self, 
        poor_performers: Dict, 
        output_dir: Path,
        visualizer: Optional['SegmentationVisualizer'] = None
    ):
        """
        保存低分患者報告
        
        Args:
            poor_performers: 低分患者資訊
            output_dir: 輸出目錄
            visualizer: 可視化器（用於複製低分患者的可視化圖片）
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        thresholds = poor_performers['thresholds']
        
        # 1. 保存完整 JSON 報告
        report_path = output_dir / 'poor_performers_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(poor_performers, f, indent=2, ensure_ascii=False)
        
        # 2. 生成可讀的文字報告
        self._save_text_report(poor_performers, output_dir)
        
        # 3. 生成低分患者 ID 列表（方便後續處理）
        patient_ids_path = output_dir / 'poor_performers_ids.txt'
        with open(patient_ids_path, 'w', encoding='utf-8') as f:
            for patient in poor_performers['patients']:
                f.write(f"{patient['patient_id']}\n")
        
        # 4. 複製低分患者的可視化圖片到獨立資料夾
        copied_count = self._copy_poor_performer_visualizations(
            poor_performers, output_dir, visualizer
        )
        
        # 記錄日誌
        n_poor = poor_performers['summary']['total_poor_patients']
        if n_poor > 0:
            self.logger.warning(
                f"⚠️ 發現 {n_poor} 個低分患者 "
                f"(Dice < {thresholds['dice']} 或 IoU < {thresholds['iou']})"
            )
            self.logger.info(f"   低分患者報告已保存: {report_path}")
            if copied_count > 0:
                self.logger.info(f"   複製 {copied_count} 張可視化圖片")
        else:
            self.logger.info(f"✅ 所有患者預測結果良好，無低分患者")
    
    def _save_text_report(self, poor_performers: Dict, output_dir: Path):
        """保存文字格式的報告"""
        txt_report_path = output_dir / 'poor_performers_report.txt'
        
        with open(txt_report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("低分患者報告 (Poor Performers Report)\n")
            f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # 摘要
            summary = poor_performers['summary']
            thresholds = poor_performers['thresholds']
            f.write(f"📊 摘要統計\n")
            f.write(f"   閾值: Dice < {thresholds['dice']}, IoU < {thresholds['iou']}\n")
            f.write(f"   低分患者數: {summary['total_poor_patients']}\n")
            if summary['total_poor_patients'] > 0:
                f.write(f"   平均 Dice: {summary['avg_dice']:.4f}\n")
                f.write(f"   平均 IoU: {summary['avg_iou']:.4f}\n")
                f.write(f"   最差 Dice: {summary['worst_dice']:.4f}\n")
                f.write(f"   最差 IoU: {summary['worst_iou']:.4f}\n")
            f.write("\n")
            
            # 患者詳情
            if poor_performers['patients']:
                f.write("=" * 80 + "\n")
                f.write("📋 低分患者詳情（按 Dice 分數排序，由低至高）\n")
                f.write("=" * 80 + "\n\n")
                
                for i, patient in enumerate(poor_performers['patients'], 1):
                    f.write(f"[{i:3d}] 患者 ID: {patient['patient_id']}\n")
                    f.write(f"      平均 Dice: {patient['avg_dice']:.4f}\n")
                    f.write(f"      平均 IoU: {patient['avg_iou']:.4f}\n")
                    f.write(f"      切片數: {patient['total_slices']}, "
                           f"病灶數: {patient['total_lesions']}\n")
                    f.write(f"      原因: {', '.join(patient['reason'])}\n")
                    
                    if patient.get('worst_slices'):
                        f.write(f"      最差切片:\n")
                        for ws in patient['worst_slices'][:3]:
                            f.write(f"        - Slice {ws['slice_idx']}: "
                                   f"Dice={ws['dice']:.4f}, IoU={ws['iou']:.4f}\n")
                    f.write("\n")
            else:
                f.write("🎉 沒有低分患者！所有預測結果都達到閾值標準。\n")
    
    def _copy_poor_performer_visualizations(
        self,
        poor_performers: Dict,
        output_dir: Path,
        visualizer: Optional['SegmentationVisualizer']
    ) -> int:
        """複製低分患者的可視化圖片"""
        copied_count = 0
        
        if visualizer is None or not poor_performers['patients']:
            return copied_count
        
        poor_vis_dir = output_dir / 'poor_performers_visualizations'
        poor_vis_dir.mkdir(parents=True, exist_ok=True)
        
        for patient in poor_performers['patients']:
            patient_id = patient['patient_id']
            # 轉換為安全的檔名格式
            safe_patient_id = str(patient_id).replace('.', '_').replace('/', '_')[:50]
            
            # 來源資料夾
            src_patient_dir = visualizer.vis_dir / safe_patient_id
            
            if src_patient_dir.exists():
                # 目標資料夾（包含 Dice 分數以便排序）
                dice_score = patient['avg_dice']
                dst_patient_dir = poor_vis_dir / f"dice_{dice_score:.4f}_{safe_patient_id}"
                dst_patient_dir.mkdir(parents=True, exist_ok=True)
                
                # 複製所有圖片
                for img_file in src_patient_dir.glob('*.png'):
                    shutil.copy2(img_file, dst_patient_dir / img_file.name)
                    copied_count += 1
                
                # 也複製患者摘要圖（如果存在）
                summary_img = src_patient_dir / 'patient_summary.png'
                if summary_img.exists():
                    shutil.copy2(summary_img, dst_patient_dir / 'patient_summary.png')
        
        return copied_count
