#!/usr/bin/env python3
"""
UNet++ 肺結節分割 - 後處理模組
================================

後處理流程：
1. Model output (probability) → Threshold
2. Lung mask 限制
3. Connected component filtering (min size)
4. Small closing / fill holes
5. Final mask

可配置參數：
- threshold: 機率閾值
- min_size_mm3: 最小連通區域體積（mm³）
- closing_radius: 閉運算半徑
- fill_holes: 是否填充孔洞
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import ndimage
from skimage import measure, morphology

logger = logging.getLogger(__name__)


@dataclass
class PostProcessConfig:
    """後處理配置"""
    # Step 1: Threshold
    threshold: float = 0.5
    
    # Step 2: Lung mask
    use_lung_mask: bool = True
    lung_mask_dilate_mm: float = 0.0  # 擴張肺遮罩（mm），0表示不擴張
    
    # Step 3: Connected component filtering
    min_size_mm3: float = 14.0  # 最小體積（約 3mm 直徑球體 = 14.14 mm³）
    max_size_mm3: float = 113097.0  # 最大體積（約 60mm 直徑球體）
    
    # Step 4: Morphology
    closing_radius_mm: float = 1.0  # 閉運算半徑 (mm)
    fill_holes: bool = True  # 填充 2D 孔洞
    fill_holes_3d: bool = False  # 填充 3D 孔洞（較慢）
    
    # 額外過濾
    remove_edge_nodules: bool = False  # 移除接觸邊緣的結節
    min_solidity: float = 0.0  # 最小 solidity（0-1，0表示不過濾）


class MaskPostProcessor:
    """
    分割遮罩後處理器
    
    處理流程：
    probability → threshold → lung mask → CC filter → morphology → final
    """
    
    def __init__(
        self,
        config: Optional[PostProcessConfig] = None,
        spacing: Optional[np.ndarray] = None
    ):
        """
        初始化後處理器
        
        Args:
            config: 後處理配置
            spacing: 體素間距 (x, y, z) 或 (y, x) for 2D，單位 mm
        """
        self.config = config or PostProcessConfig()
        self.spacing = spacing if spacing is not None else np.array([1.0, 1.0, 1.0])
        
    def set_spacing(self, spacing: np.ndarray):
        """更新體素間距"""
        self.spacing = spacing
        
    def __call__(
        self,
        prob_mask: np.ndarray,
        lung_mask: Optional[np.ndarray] = None,
        return_intermediate: bool = False
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        執行完整後處理流程
        
        Args:
            prob_mask: 模型輸出的機率圖 (H, W) 或 (D, H, W)
            lung_mask: 肺部遮罩（二值）
            return_intermediate: 是否返回中間結果
            
        Returns:
            最終二值遮罩，或包含中間結果的字典
        """
        intermediate = {}
        
        # Step 1: Threshold
        binary_mask = self.apply_threshold(prob_mask)
        intermediate['after_threshold'] = binary_mask.copy()
        logger.debug(f"After threshold: {binary_mask.sum()} positive voxels")
        
        # Step 2: Lung mask restriction
        if self.config.use_lung_mask and lung_mask is not None:
            binary_mask = self.apply_lung_mask(binary_mask, lung_mask)
            intermediate['after_lung_mask'] = binary_mask.copy()
            logger.debug(f"After lung mask: {binary_mask.sum()} positive voxels")
        
        # Step 3: Connected component filtering
        binary_mask = self.filter_connected_components(binary_mask)
        intermediate['after_cc_filter'] = binary_mask.copy()
        logger.debug(f"After CC filter: {binary_mask.sum()} positive voxels")
        
        # Step 4: Morphological operations
        binary_mask = self.apply_morphology(binary_mask)
        intermediate['final'] = binary_mask
        logger.debug(f"Final: {binary_mask.sum()} positive voxels")
        
        if return_intermediate:
            return intermediate
        return binary_mask
    
    def apply_threshold(self, prob_mask: np.ndarray) -> np.ndarray:
        """
        Step 1: 應用閾值
        
        Args:
            prob_mask: 機率圖 [0, 1]
            
        Returns:
            二值遮罩
        """
        return (prob_mask > self.config.threshold).astype(np.uint8)
    
    def apply_lung_mask(
        self,
        binary_mask: np.ndarray,
        lung_mask: np.ndarray
    ) -> np.ndarray:
        """
        Step 2: 肺部遮罩限制
        
        Args:
            binary_mask: 二值分割遮罩
            lung_mask: 肺部遮罩
            
        Returns:
            限制在肺部區域內的遮罩
        """
        # 確保 lung_mask 是二值的
        lung_binary = (lung_mask > 0).astype(np.uint8)
        
        # 可選：擴張肺遮罩
        if self.config.lung_mask_dilate_mm > 0:
            # 計算擴張半徑（體素）
            dilate_radius = int(np.ceil(
                self.config.lung_mask_dilate_mm / np.min(self.spacing[:binary_mask.ndim])
            ))
            if binary_mask.ndim == 2:
                struct = morphology.disk(dilate_radius)
            else:
                struct = morphology.ball(dilate_radius)
            lung_binary = morphology.binary_dilation(lung_binary, struct).astype(np.uint8)
        
        return binary_mask * lung_binary
    
    def filter_connected_components(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Step 3: 連通區域過濾
        
        過濾掉太小或太大的連通區域
        
        Args:
            binary_mask: 二值遮罩
            
        Returns:
            過濾後的遮罩
        """
        if binary_mask.sum() == 0:
            return binary_mask
        
        # 計算體素體積
        voxel_volume = np.prod(self.spacing[:binary_mask.ndim])
        
        # 計算最小/最大體素數
        min_voxels = int(np.ceil(self.config.min_size_mm3 / voxel_volume))
        max_voxels = int(np.floor(self.config.max_size_mm3 / voxel_volume))
        
        # 標記連通區域
        labels = measure.label(binary_mask, connectivity=binary_mask.ndim)
        regions = measure.regionprops(labels)
        
        filtered_mask = np.zeros_like(binary_mask)
        
        for region in regions:
            # 大小過濾
            if region.area < min_voxels:
                continue
            if region.area > max_voxels:
                continue
            
            # Solidity 過濾（可選）
            if self.config.min_solidity > 0:
                if hasattr(region, 'solidity') and region.solidity < self.config.min_solidity:
                    continue
            
            # 邊緣過濾（可選）
            if self.config.remove_edge_nodules:
                bbox = region.bbox
                # 檢查是否接觸邊緣
                if binary_mask.ndim == 2:
                    at_edge = (
                        bbox[0] == 0 or bbox[1] == 0 or
                        bbox[2] == binary_mask.shape[0] or
                        bbox[3] == binary_mask.shape[1]
                    )
                else:
                    at_edge = (
                        bbox[0] == 0 or bbox[1] == 0 or bbox[2] == 0 or
                        bbox[3] == binary_mask.shape[0] or
                        bbox[4] == binary_mask.shape[1] or
                        bbox[5] == binary_mask.shape[2]
                    )
                if at_edge:
                    continue
            
            filtered_mask[labels == region.label] = 1
        
        return filtered_mask.astype(np.uint8)
    
    def apply_morphology(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Step 4: 形態學操作
        
        - 閉運算：連接相鄰區域、平滑邊緣
        - 填充孔洞
        
        Args:
            binary_mask: 二值遮罩
            
        Returns:
            形態學處理後的遮罩
        """
        if binary_mask.sum() == 0:
            return binary_mask
        
        result = binary_mask.copy()
        
        # 計算結構元素半徑
        radius = int(np.ceil(
            self.config.closing_radius_mm / np.min(self.spacing[:binary_mask.ndim])
        ))
        
        if radius > 0:
            # 閉運算
            if binary_mask.ndim == 2:
                struct = morphology.disk(radius)
            else:
                struct = morphology.ball(radius)
            
            result = morphology.binary_closing(result, struct)
        
        # 填充孔洞
        if self.config.fill_holes:
            if binary_mask.ndim == 2:
                result = ndimage.binary_fill_holes(result)
            elif self.config.fill_holes_3d:
                result = ndimage.binary_fill_holes(result)
            else:
                # 逐切片填充 2D 孔洞（更快）
                for z in range(result.shape[0]):
                    result[z] = ndimage.binary_fill_holes(result[z])
        
        return result.astype(np.uint8)


class PatchPostProcessor(MaskPostProcessor):
    """
    Patch 級別的後處理器
    
    適用於 UNet++ 訓練時使用的小 patch
    """
    
    def __init__(
        self,
        config: Optional[PostProcessConfig] = None,
        spacing: Optional[np.ndarray] = None
    ):
        # 對於 patch，使用更保守的預設值
        if config is None:
            config = PostProcessConfig(
                threshold=0.5,
                min_size_mm3=5.0,  # patch 內結節可能被截斷，降低閾值
                max_size_mm3=50000.0,
                closing_radius_mm=0.5,  # 較小的閉運算
                fill_holes=True,
                fill_holes_3d=False
            )
        super().__init__(config, spacing)
    
    def process_patch(
        self,
        prob_mask: np.ndarray,
        lung_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        處理單個 patch
        
        Args:
            prob_mask: patch 機率圖 (H, W)
            lung_mask: patch 肺遮罩 (H, W)
            
        Returns:
            後處理後的二值遮罩
        """
        return self(prob_mask, lung_mask)


def create_postprocessor(
    config_dict: Optional[Dict] = None,
    spacing: Optional[np.ndarray] = None,
    mode: str = 'volume'
) -> MaskPostProcessor:
    """
    工廠函數：創建後處理器
    
    Args:
        config_dict: 配置字典
        spacing: 體素間距
        mode: 'volume' 或 'patch'
        
    Returns:
        後處理器實例
    """
    if config_dict is not None:
        config = PostProcessConfig(**config_dict)
    else:
        config = PostProcessConfig()
    
    if mode == 'patch':
        return PatchPostProcessor(config, spacing)
    else:
        return MaskPostProcessor(config, spacing)


# ============ 便捷函數 ============

def postprocess_prediction(
    prob_mask: np.ndarray,
    lung_mask: Optional[np.ndarray] = None,
    spacing: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    min_size_mm3: float = 14.0,
    closing_radius_mm: float = 1.0,
    fill_holes: bool = True
) -> np.ndarray:
    """
    便捷函數：後處理預測結果
    
    Args:
        prob_mask: 模型輸出機率圖
        lung_mask: 肺部遮罩
        spacing: 體素間距 (mm)
        threshold: 機率閾值
        min_size_mm3: 最小連通區域體積
        closing_radius_mm: 閉運算半徑
        fill_holes: 是否填充孔洞
        
    Returns:
        後處理後的二值遮罩
    """
    config = PostProcessConfig(
        threshold=threshold,
        min_size_mm3=min_size_mm3,
        closing_radius_mm=closing_radius_mm,
        fill_holes=fill_holes,
        use_lung_mask=lung_mask is not None
    )
    
    processor = MaskPostProcessor(config, spacing)
    return processor(prob_mask, lung_mask)


def visualize_postprocess_steps(
    prob_mask: np.ndarray,
    lung_mask: Optional[np.ndarray] = None,
    spacing: Optional[np.ndarray] = None,
    config: Optional[PostProcessConfig] = None,
    slice_idx: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    視覺化後處理各步驟結果
    
    Args:
        prob_mask: 機率圖
        lung_mask: 肺遮罩
        spacing: 體素間距
        config: 後處理配置
        slice_idx: 要視覺化的切片索引（僅 3D）
        
    Returns:
        各步驟結果的字典
    """
    processor = MaskPostProcessor(config or PostProcessConfig(), spacing)
    results = processor(prob_mask, lung_mask, return_intermediate=True)
    
    # 如果是 3D，提取指定切片
    if prob_mask.ndim == 3 and slice_idx is not None:
        for key in results:
            results[key] = results[key][slice_idx]
    
    return results


# ============ 測試 ============

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # 創建測試數據
    print("Creating test data...")
    
    # 模擬 256x256 patch
    prob_mask = np.zeros((256, 256), dtype=np.float32)
    
    # 添加一些模擬的結節（高機率區域）
    # 結節 1: 有效結節
    yy, xx = np.ogrid[:256, :256]
    center1 = (100, 100)
    mask1 = ((yy - center1[0])**2 + (xx - center1[1])**2) < 15**2
    prob_mask[mask1] = 0.8
    
    # 結節 2: 較小的結節
    center2 = (150, 180)
    mask2 = ((yy - center2[0])**2 + (xx - center2[1])**2) < 8**2
    prob_mask[mask2] = 0.7
    
    # 雜訊：一些小的假陽性
    for _ in range(20):
        cx, cy = np.random.randint(0, 256, 2)
        r = np.random.randint(1, 4)
        noise_mask = ((yy - cy)**2 + (xx - cx)**2) < r**2
        prob_mask[noise_mask] = np.random.uniform(0.3, 0.6)
    
    # 模擬肺遮罩
    lung_mask = np.zeros((256, 256), dtype=np.uint8)
    lung_center = (128, 128)
    lung_r = 100
    lung_region = ((yy - lung_center[0])**2 + (xx - lung_center[1])**2) < lung_r**2
    lung_mask[lung_region] = 1
    
    # 執行後處理
    print("Running postprocessing...")
    spacing = np.array([1.0, 1.0])  # 1mm x 1mm
    
    results = visualize_postprocess_steps(
        prob_mask,
        lung_mask,
        spacing,
        PostProcessConfig(threshold=0.5, min_size_mm3=50.0)
    )
    
    # 視覺化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(prob_mask, cmap='hot')
    axes[0, 0].set_title('1. Model Probability Output')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(results['after_threshold'], cmap='gray')
    axes[0, 1].set_title(f'2. After Threshold (>0.5)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(results['after_lung_mask'], cmap='gray')
    axes[0, 2].set_title('3. After Lung Mask')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(results['after_cc_filter'], cmap='gray')
    axes[1, 0].set_title('4. After CC Filter (min 50mm³)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(results['final'], cmap='gray')
    axes[1, 1].set_title('5. Final (closing + fill holes)')
    axes[1, 1].axis('off')
    
    # Overlay
    overlay = np.stack([prob_mask, results['final'].astype(float), lung_mask * 0.3], axis=-1)
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title('Overlay (R=prob, G=final, B=lung)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('postprocess_demo.png', dpi=150)
    plt.show()
    
    print("Done! Saved to postprocess_demo.png")
