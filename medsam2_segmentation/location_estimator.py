#!/usr/bin/env python3
"""
肺部病灶解剖學位置估算器

根據病灶在 CT 影像中的相對位置和切片位置，推斷病灶所在的解剖學區域。

肺葉解剖學對照：
- RUL (Right Upper Lobe): 右上葉
- RML (Right Middle Lobe): 右中葉
- RLL (Right Lower Lobe): 右下葉
- LUL (Left Upper Lobe): 左上葉
- LLL (Left Lower Lobe): 左下葉
- Lingula: 左肺舌段 (左肺中葉等效區域)

注意：此估算基於統計學分布，準確定位需要結合肺葉分割 (lobe segmentation)
"""

from typing import Dict, List, Tuple, Optional
import numpy as np


class LungLocationEstimator:
    """
    基於座標和切片位置估算肺部病灶的解剖學位置
    """
    
    # 肺葉縮寫
    LOBES = {
        'RUL': 'Right Upper Lobe (右上葉)',
        'RML': 'Right Middle Lobe (右中葉)', 
        'RLL': 'Right Lower Lobe (右下葉)',
        'LUL': 'Left Upper Lobe (左上葉)',
        'Lingula': 'Left Lingula (左肺舌段)',
        'LLL': 'Left Lower Lobe (左下葉)',
    }
    
    def __init__(self, total_slices: int = 100):
        """
        初始化估算器
        
        Args:
            total_slices: 總切片數（用於正規化切片位置）
        """
        self.total_slices = total_slices
    
    def estimate_location(
        self, 
        relative_x: float,          # 0=左, 1=右
        relative_y: float,          # 0=前(ventral), 1=後(dorsal)
        slice_ratio: float,         # 0=頭側, 1=足側 (normalized slice position)
        slice_index: Optional[int] = None,
        slice_location_mm: Optional[float] = None
    ) -> Dict[str, any]:
        """
        估算病灶的解剖學位置
        
        座標系統 (axial view, 仰臥位 supine):
        - X軸: 病人左→右 (relative_x: 0→1)
        - Y軸: 前(腹側)→後(背側) (relative_y: 0→1)  
        - Z軸: 頭側→足側 (slice_ratio: 0→1)
        
        Args:
            relative_x: 水平相對位置 (0=左側, 0.5=中線, 1=右側)
            relative_y: 前後相對位置 (0=前/腹側, 1=後/背側)
            slice_ratio: 切片相對位置 (0=頭側/上, 1=足側/下)
            slice_index: 原始切片索引
            slice_location_mm: 切片在 Z 軸的絕對位置 (mm)
        
        Returns:
            Dict containing:
            - lobe: 肺葉縮寫 (RUL/RML/RLL/LUL/LLL/Lingula)
            - lobe_full: 完整名稱
            - side: 'right' or 'left'
            - vertical_zone: 'upper', 'middle', 'lower'
            - confidence: 估算可信度 (0-1)
            - description: 中文描述
        """
        result = {
            'lobe': 'unknown',
            'lobe_full': 'Unknown',
            'side': 'unknown',
            'vertical_zone': 'unknown',
            'confidence': 0.0,
            'description': '',
            'coordinates': {
                'relative_x': relative_x,
                'relative_y': relative_y,
                'slice_ratio': slice_ratio
            }
        }
        
        # 1. 判斷左右側
        # 中線約在 0.45-0.55 (考慮心臟偏左)
        if relative_x < 0.42:
            side = 'left'
            result['side'] = 'left'
        elif relative_x > 0.58:
            side = 'right'
            result['side'] = 'right'
        else:
            # 中間區域，根據輕微偏向判斷
            side = 'left' if relative_x < 0.5 else 'right'
            result['side'] = side
            result['confidence'] = max(0.3, result['confidence'])  # 低可信度
        
        # 2. 判斷上中下區域 (基於切片位置)
        # 典型胸部 CT：上1/3為上葉，中1/3過渡區，下1/3為下葉
        if slice_ratio < 0.35:
            vertical_zone = 'upper'
        elif slice_ratio < 0.65:
            vertical_zone = 'middle'
        else:
            vertical_zone = 'lower'
        result['vertical_zone'] = vertical_zone
        
        # 3. 綜合判斷肺葉
        lobe, confidence = self._determine_lobe(side, vertical_zone, relative_y, slice_ratio)
        result['lobe'] = lobe
        result['lobe_full'] = self.LOBES.get(lobe, lobe)
        result['confidence'] = confidence
        
        # 4. 生成描述
        result['description'] = self._generate_description(result)
        
        return result
    
    def _determine_lobe(
        self, 
        side: str, 
        vertical_zone: str, 
        relative_y: float,
        slice_ratio: float
    ) -> Tuple[str, float]:
        """
        根據多個因素判斷具體肺葉
        
        右肺：3葉 (RUL, RML, RLL)
        - 水平裂 (horizontal fissure) 分隔 RUL/RML
        - 斜裂 (oblique fissure) 分隔 RUL+RML/RLL
        
        左肺：2葉 (LUL, LLL) + Lingula
        - 斜裂分隔 LUL/LLL
        - Lingula 是 LUL 的一部分，位於前下方
        """
        confidence = 0.7  # 基礎可信度
        
        if side == 'right':
            if vertical_zone == 'upper':
                lobe = 'RUL'
                confidence = 0.85
            elif vertical_zone == 'lower':
                lobe = 'RLL'
                confidence = 0.85
            else:  # middle zone - 需要更細緻判斷
                # RML 位於前方，RUL/RLL 需要根據前後位置判斷
                if relative_y < 0.5:  # 前方
                    lobe = 'RML'
                    confidence = 0.75
                else:  # 後方
                    # 根據切片位置傾向上或下葉
                    if slice_ratio < 0.5:
                        lobe = 'RUL'
                    else:
                        lobe = 'RLL'
                    confidence = 0.6
        
        else:  # left side
            if vertical_zone == 'upper':
                # 判斷是否為 Lingula (前下方區域)
                if slice_ratio > 0.25 and relative_y < 0.45:
                    lobe = 'Lingula'
                    confidence = 0.7
                else:
                    lobe = 'LUL'
                    confidence = 0.85
            elif vertical_zone == 'lower':
                lobe = 'LLL'
                confidence = 0.85
            else:  # middle zone
                if relative_y < 0.45:  # 前方
                    lobe = 'Lingula'
                    confidence = 0.7
                else:
                    if slice_ratio < 0.5:
                        lobe = 'LUL'
                    else:
                        lobe = 'LLL'
                    confidence = 0.6
        
        return lobe, confidence
    
    def _generate_description(self, result: Dict) -> str:
        """生成中文描述"""
        lobe = result['lobe']
        side = '右肺' if result['side'] == 'right' else '左肺'
        
        lobe_names = {
            'RUL': '上葉',
            'RML': '中葉',
            'RLL': '下葉',
            'LUL': '上葉',
            'Lingula': '舌段',
            'LLL': '下葉',
        }
        
        lobe_name = lobe_names.get(lobe, '未知區域')
        
        confidence_desc = ''
        if result['confidence'] < 0.6:
            confidence_desc = '（位置估算可信度較低）'
        elif result['confidence'] < 0.75:
            confidence_desc = '（位置估算）'
        
        return f"{side}{lobe_name} ({lobe}){confidence_desc}"
    
    def estimate_from_features(self, lesion_features: Dict, total_slices: int = None) -> Dict:
        """
        從現有特徵資料估算位置
        
        Args:
            lesion_features: 包含 relative_position_x, relative_position_y, slice_location 等
            total_slices: 總切片數
        """
        if total_slices:
            self.total_slices = total_slices
        
        relative_x = lesion_features.get('relative_position_x', 0.5)
        relative_y = lesion_features.get('relative_position_y', 0.5)
        
        # 計算 slice_ratio
        slice_location = lesion_features.get('slice_location', 0)
        slice_index = lesion_features.get('slice_index', 0)
        
        # 如果有 slice_index，使用它來計算比例
        if slice_index and self.total_slices:
            slice_ratio = slice_index / self.total_slices
        else:
            # 否則使用 slice_location 估算（假設肺部範圍約 -400 到 100 mm）
            # 這是一個粗略估計，實際應根據具體數據調整
            slice_ratio = (slice_location + 400) / 500 if slice_location else 0.5
            slice_ratio = np.clip(slice_ratio, 0, 1)
        
        return self.estimate_location(
            relative_x=relative_x,
            relative_y=relative_y,
            slice_ratio=slice_ratio,
            slice_index=slice_index,
            slice_location_mm=slice_location
        )


def add_location_to_features(features_dict: Dict, total_slices: int = None) -> Dict:
    """
    為特徵字典添加位置資訊
    
    Args:
        features_dict: LLM 特徵資料（包含 slices 列表）
        total_slices: 總切片數
    
    Returns:
        增強後的特徵字典，包含 anatomical_location
    """
    estimator = LungLocationEstimator(total_slices or features_dict.get('metadata', {}).get('total_slices', 100))
    
    # 收集所有病灶的位置估算
    lesion_locations = []
    
    # 如果有 numerical_features，添加代表性位置
    if 'numerical_features' in features_dict:
        # 這裡可以添加主要病灶的位置估算
        pass
    
    # 添加到特徵中
    features_dict['anatomical_location'] = {
        'lesion_locations': lesion_locations,
        'estimation_method': 'coordinate_based',
        'note': '位置為基於座標的估算，精確位置需結合肺葉分割'
    }
    
    return features_dict


def get_location_for_report(
    relative_x: float, 
    relative_y: float, 
    slice_ratio: float,
    total_slices: int = 100
) -> str:
    """
    快速獲取位置描述（用於報告生成）
    
    Returns:
        位置縮寫 (如 'RUL', 'LLL')
    """
    estimator = LungLocationEstimator(total_slices)
    result = estimator.estimate_location(relative_x, relative_y, slice_ratio)
    return result['lobe']


# 測試
if __name__ == "__main__":
    estimator = LungLocationEstimator(total_slices=100)
    
    # 測試案例
    test_cases = [
        # (x, y, z_ratio, expected_description)
        (0.7, 0.3, 0.2, "右上葉前段"),
        (0.8, 0.7, 0.8, "右下葉後段"),
        (0.7, 0.3, 0.5, "右中葉"),
        (0.3, 0.3, 0.2, "左上葉"),
        (0.2, 0.7, 0.8, "左下葉"),
        (0.3, 0.3, 0.45, "左肺舌段"),
    ]
    
    print("=" * 60)
    print("肺部病灶位置估算測試")
    print("=" * 60)
    
    for x, y, z, expected in test_cases:
        result = estimator.estimate_location(x, y, z)
        print(f"\n輸入: x={x:.1f}, y={y:.1f}, z_ratio={z:.1f}")
        print(f"估算結果: {result['lobe']} ({result['lobe_full']})")
        print(f"描述: {result['description']}")
        print(f"可信度: {result['confidence']:.2f}")
