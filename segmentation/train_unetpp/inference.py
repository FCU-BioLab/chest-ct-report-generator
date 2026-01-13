#!/usr/bin/env python3
"""
UNet++ 肺結節分割訓練 - 推論模組
===================================

提供推論和後處理功能：
1. 3D Volume 分割
2. 後處理（連通區域過濾、肺野遮罩、形態學）
3. 結節屬性提取
4. 輸出 NIfTI 和 JSON
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from scipy import ndimage
from skimage import measure, morphology
import SimpleITK as sitk
from tqdm import tqdm

import sys

# 支援直接執行和模組執行
try:
    from .config import Config
    from .preprocess import CTPreprocessor
    from .postprocess import MaskPostProcessor, PostProcessConfig
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from train_unetpp.config import Config
    from train_unetpp.preprocess import CTPreprocessor
    from train_unetpp.postprocess import MaskPostProcessor, PostProcessConfig


logger = logging.getLogger(__name__)


class NoduleExtractor:
    """結節屬性提取器"""
    
    def __init__(self, spacing: np.ndarray):
        """
        初始化提取器
        
        Args:
            spacing: 體素間距 (x, y, z)
        """
        self.spacing = spacing
    
    def extract_nodule_attributes(
        self,
        mask: np.ndarray,
        image: Optional[np.ndarray] = None,
        min_volume_mm3: float = 8.0
    ) -> List[Dict]:
        """
        從分割遮罩提取結節屬性
        
        Args:
            mask: 二值分割遮罩 (Z, Y, X)
            image: CT 影像（HU 值），用於提取強度特徵
            min_volume_mm3: 最小結節體積
            
        Returns:
            結節屬性列表
        """
        labels = measure.label(mask > 0.5)
        regions = measure.regionprops(labels, intensity_image=image)
        
        voxel_volume = np.prod(self.spacing)
        nodules = []
        
        for region in regions:
            volume_mm3 = region.area * voxel_volume
            
            if volume_mm3 < min_volume_mm3:
                continue
            
            # 計算中心座標（轉為毫米）
            centroid_voxel = np.array(region.centroid)  # (z, y, x)
            centroid_mm = centroid_voxel * self.spacing[::-1]  # 轉為 (x, y, z) mm
            
            # 計算直徑
            bbox = region.bbox
            dimensions_voxel = np.array([
                bbox[3] - bbox[0],  # z
                bbox[4] - bbox[1],  # y
                bbox[5] - bbox[2]   # x
            ])
            dimensions_mm = dimensions_voxel * self.spacing[::-1]
            max_diameter_mm = dimensions_mm.max()
            
            nodule = {
                'id': region.label,
                'centroid_voxel': centroid_voxel.tolist(),
                'centroid_mm': centroid_mm.tolist(),
                'volume_mm3': float(volume_mm3),
                'max_diameter_mm': float(max_diameter_mm),
                'dimensions_mm': dimensions_mm.tolist(),
                'bbox': list(bbox),
                'solidity': float(region.solidity) if hasattr(region, 'solidity') else None
            }
            
            # 添加強度特徵（如果有 HU 影像）
            if image is not None:
                nodule['mean_hu'] = float(region.mean_intensity)
                nodule['min_hu'] = float(region.min_intensity)
                nodule['max_hu'] = float(region.max_intensity)
            
            nodules.append(nodule)
        
        # 按體積排序
        nodules.sort(key=lambda x: x['volume_mm3'], reverse=True)
        
        return nodules


class Inferencer:
    """推論器"""
    
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        device: Optional[str] = None
    ):
        """
        初始化推論器
        
        Args:
            model: 訓練好的模型
            config: 配置物件
            device: 設備
        """
        self.model = model
        self.config = config
        self.device = device or config.device
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.threshold = config.inference.prediction_threshold
        self.preprocessor = CTPreprocessor(
            target_spacing=config.data.target_spacing,
            hu_window_center=config.data.hu_window_center,
            hu_window_width=config.data.hu_window_width
        )
        
        # 初始化後處理器
        self.postprocessor = self._create_postprocessor()
    
    def _create_postprocessor(self) -> MaskPostProcessor:
        """創建後處理器"""
        inf_cfg = self.config.inference
        pp_config = PostProcessConfig(
            threshold=inf_cfg.prediction_threshold,
            use_lung_mask=inf_cfg.use_lung_mask,
            lung_mask_dilate_mm=inf_cfg.lung_mask_dilate_mm,
            min_size_mm3=inf_cfg.min_volume_mm3,
            max_size_mm3=inf_cfg.max_volume_mm3,
            closing_radius_mm=inf_cfg.closing_radius_mm,
            fill_holes=inf_cfg.fill_holes,
            fill_holes_3d=inf_cfg.fill_holes_3d,
            remove_edge_nodules=inf_cfg.remove_edge_nodules,
            min_solidity=inf_cfg.min_solidity
        )
        return MaskPostProcessor(pp_config)
    
    @torch.no_grad()
    def predict_volume(
        self,
        volume: np.ndarray,
        spacing: np.ndarray,
        batch_size: int = 8
    ) -> np.ndarray:
        """
        預測 3D Volume
        
        Args:
            volume: 3D 影像 (Z, Y, X)，已歸一化
            spacing: 體素間距
            batch_size: 批次大小
            
        Returns:
            預測遮罩 (Z, Y, X)
        """
        num_slices = volume.shape[0]
        predictions = np.zeros_like(volume, dtype=np.float32)
        
        # 準備 2.5D 切片
        slice_distance = self.config.data.slice_distance_mm
        offset = max(1, int(round(slice_distance / spacing[2])))
        
        # 批次處理
        for batch_start in range(0, num_slices, batch_size):
            batch_end = min(batch_start + batch_size, num_slices)
            batch_slices = []
            
            for z in range(batch_start, batch_end):
                z_prev = max(0, z - offset)
                z_next = min(num_slices - 1, z + offset)
                
                slice_25d = np.stack([
                    volume[z_prev],
                    volume[z],
                    volume[z_next]
                ], axis=0)  # (3, H, W)
                batch_slices.append(slice_25d)
            
            # 轉為 tensor
            batch_tensor = torch.from_numpy(np.stack(batch_slices, axis=0)).float()
            batch_tensor = batch_tensor.to(self.device)
            
            # 預測
            with autocast(enabled=self.config.training.use_amp):
                outputs = self.model(batch_tensor)
            
            # 處理輸出
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            for i, z in enumerate(range(batch_start, batch_end)):
                predictions[z] = probs[i, 0]  # 取第一個通道
        
        return predictions
    
    def postprocess(
        self,
        pred_mask: np.ndarray,
        lung_mask: Optional[np.ndarray] = None,
        spacing: np.ndarray = None
    ) -> np.ndarray:
        """
        後處理預測結果
        
        流程：
        1. Model output (probability) → Threshold (0.5)
        2. Lung mask 限制
        3. Connected component filtering (min/max size)
        4. Small closing / fill holes
        5. Final mask
        
        Args:
            pred_mask: 預測機率遮罩
            lung_mask: 肺野遮罩
            spacing: 體素間距
            
        Returns:
            後處理後的二值遮罩
        """
        # 使用新的後處理模組
        self.postprocessor.set_spacing(spacing if spacing is not None else np.array([1.0, 1.0, 1.0]))
        return self.postprocessor(pred_mask, lung_mask)
    
    def postprocess_with_details(
        self,
        pred_mask: np.ndarray,
        lung_mask: Optional[np.ndarray] = None,
        spacing: np.ndarray = None
    ) -> Dict[str, np.ndarray]:
        """
        後處理並返回中間結果（用於調試/視覺化）
        
        Args:
            pred_mask: 預測機率遮罩
            lung_mask: 肺野遮罩
            spacing: 體素間距
            
        Returns:
            包含各步驟結果的字典
        """
        self.postprocessor.set_spacing(spacing if spacing is not None else np.array([1.0, 1.0, 1.0]))
        return self.postprocessor(pred_mask, lung_mask, return_intermediate=True)
    
    def run_inference(
        self,
        data: Dict,
        save_dir: Optional[str] = None
    ) -> Dict:
        """
        執行完整推論流程
        
        Args:
            data: 預處理後的資料字典
            save_dir: 輸出目錄
            
        Returns:
            推論結果
        """
        volume = data['volume']
        spacing = data['spacing']
        lung_mask = data.get('lung_mask')
        volume_hu = data.get('volume_hu')
        
        # 預測
        pred_probs = self.predict_volume(volume, spacing)
        
        # 後處理
        pred_mask = self.postprocess(pred_probs, lung_mask, spacing)
        
        # 提取結節屬性
        extractor = NoduleExtractor(spacing)
        nodules = extractor.extract_nodule_attributes(
            pred_mask,
            volume_hu,
            self.config.inference.min_volume_mm3
        )
        
        result = {
            'pred_probs': pred_probs,
            'pred_mask': pred_mask,
            'nodules': nodules,
            'spacing': spacing,
            'original_shape': data.get('original_shape'),
            'bbox': data.get('bbox')
        }
        
        # 保存結果
        if save_dir:
            self._save_results(result, data.get('patient_id', 'unknown'), save_dir)
        
        return result
    
    def _save_results(
        self,
        result: Dict,
        patient_id: str,
        save_dir: str
    ):
        """保存推論結果"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存 JSON（結節屬性，供 LLM 使用）
        if self.config.inference.save_json:
            json_path = save_dir / f"{patient_id}_nodules.json"
            output_data = {
                'patient_id': patient_id,
                'nodules': result['nodules'],
                'spacing': result['spacing'].tolist() if hasattr(result['spacing'], 'tolist') else result['spacing'],
                'num_nodules': len(result['nodules'])
            }
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            logger.info(f"JSON 已保存: {json_path}")
        
        # 保存 NIfTI
        if self.config.inference.save_nifti:
            nifti_path = save_dir / f"{patient_id}_pred.nii.gz"
            spacing = result['spacing']
            
            # 創建 SimpleITK 影像
            pred_image = sitk.GetImageFromArray(result['pred_mask'].astype(np.uint8))
            pred_image.SetSpacing(spacing.tolist() if hasattr(spacing, 'tolist') else list(spacing))
            
            sitk.WriteImage(pred_image, str(nifti_path))
            logger.info(f"NIfTI 已保存: {nifti_path}")


def load_model_for_inference(
    checkpoint_path: str,
    config: Config,
    device: Optional[str] = None
) -> nn.Module:
    """
    載入模型用於推論
    
    Args:
        checkpoint_path: 檢查點路徑
        config: 配置物件
        device: 設備
        
    Returns:
        載入的模型
    """
    try:
        from .model import get_model
    except ImportError:
        from train_unetpp.model import get_model
    
    device = device or config.device
    logger.info(f"Loading model from {checkpoint_path}")
    model = get_model(config)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"模型已載入: {checkpoint_path}")
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="UNet++ 推論")
    parser.add_argument("--model_path", type=str, required=True, help="模型路徑")
    parser.add_argument("--input_path", type=str, required=True, help="輸入檔案路徑")
    parser.add_argument("--output_dir", type=str, required=True, help="輸出目錄")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    try:
        from .config import get_default_config
    except ImportError:
        from train_unetpp.config import get_default_config
    config = get_default_config()
    
    # 載入模型
    model = load_model_for_inference(args.model_path, config)
    
    # 創建推論器
    inferencer = Inferencer(model, config)
    
    # 載入資料
    preprocessor = CTPreprocessor()
    data = preprocessor.load_preprocessed(args.input_path)
    
    # 推論
    result = inferencer.run_inference(data, args.output_dir)
    
    print(f"找到 {len(result['nodules'])} 個結節")
    for nodule in result['nodules']:
        print(f"  - Volume: {nodule['volume_mm3']:.1f} mm³, "
              f"Diameter: {nodule['max_diameter_mm']:.1f} mm")
