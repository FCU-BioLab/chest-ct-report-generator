import time
import numpy as np
import scipy.ndimage as ndimage
import logging

logger = logging.getLogger(__name__)

def generate_lung_mask(image_np: np.ndarray, thresh_val: float = 0.3) -> np.ndarray:
    """
    從胸腔 CT 影像中萃取出肺部區域的 3D 遮罩 (Lung Mask)。
    
    Args:
        image_np: 3D numpy array [H, W, D], 數值已經過預處理並歸一化至 [0, 1] 範圍。
                  (通常這對應到 -1024 到 300 HU 之間的映射)。
        thresh_val: 空氣與組織的二值化閾值。0.3 對應約 -600 HU。
        
    Returns:
        lung_mask: 3D bool numpy array，肺部區域為 1，其他為 0。
    """
    start_t = time.time()
    
    # 1. 將所有近似空氣的暗部切出來 (包含肺部內部的空氣，以及身體外部的背景空氣)
    binary = image_np < thresh_val
    
    # 2. 標記所有獨立的連通區塊 (Connected Components)
    labeled_array, num_features = ndimage.label(binary)
    if num_features == 0:
        return np.zeros_like(binary, dtype=bool)
        
    # 3. 找出「外部背景空氣」: 取得與影像 8 個角落相連通的區塊標籤
    border_labels = set()
    border_labels.add(labeled_array[0, 0, 0])
    border_labels.add(labeled_array[0, 0, -1])
    border_labels.add(labeled_array[0, -1, 0])
    border_labels.add(labeled_array[0, -1, -1])
    border_labels.add(labeled_array[-1, 0, 0])
    border_labels.add(labeled_array[-1, 0, -1])
    border_labels.add(labeled_array[-1, -1, 0])
    border_labels.add(labeled_array[-1, -1, -1])
    
    # 移除這些背景空氣區塊，剩下的就是「身體內部的空氣 (包含肺葉、氣管、腸胃氣體)」
    internal_air_mask = np.copy(binary)
    for bg_label in border_labels:
        if bg_label != 0:
            internal_air_mask[labeled_array == bg_label] = 0
            
    # 4. 再次標記身體內部的空氣區塊
    labeled_internal, num_internal = ndimage.label(internal_air_mask)
    if num_internal == 0:
        return np.zeros_like(binary, dtype=bool)
        
    component_sizes = np.bincount(labeled_internal.ravel())
    component_sizes[0] = 0 # 忽略非空氣組織的背景
    
    # 取出體積最大的前 2 名區塊 (對應左肺與右肺，或者相連的整個雙肺)
    top_2_labels = np.argsort(component_sizes)[::-1][:2]
    
    lung_mask = np.zeros_like(binary, dtype=bool)
    for l in top_2_labels:
        # 如果該區塊體積太小 (如腸胃氣體或氣管的一小段)，則忽略
        if component_sizes[l] > 5000: 
            lung_mask[labeled_internal == l] = True
            
    # 5. 形態學擴張與閉合 (Morphological Closing & Dilation)
    # 目的是要把肺部裡面那些因為密度太高(呈現亮色)被剔除的「血管」和「貼壁結節」給包容進肺部遮罩中
    # iterations=7 約擴張 7 voxels (~5.6mm at 0.8mm spacing)，確保胸膜下結節不被遺漏
    struct = ndimage.generate_binary_structure(3, 1)
    lung_mask = ndimage.binary_dilation(lung_mask, structure=struct, iterations=7)
    lung_mask = ndimage.binary_closing(lung_mask, structure=struct, iterations=3)
    
    elapsed = time.time() - start_t
    logger.debug(f"Lung mask generated in {elapsed*1000:.1f}ms. Tissue fraction: {lung_mask.sum()/lung_mask.size:.1%}")
    return lung_mask
