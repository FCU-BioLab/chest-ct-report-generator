"""
結合 Segmentation Features 生成 CT 影像報告

此腳本會讀取 MedSAM2 分割後的特徵資料，並使用 LLM 生成結構化醫學報告。
"""

import torch
import time
import json
import os
import sys
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM

# 嘗試導入圖片處理庫
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("警告: matplotlib 未安裝，將無法生成視覺化圖片")

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    print("警告: pydicom 未安裝，將無法讀取 DICOM 檔案")

# ===== 設定 =====
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
USE_ADAPTER = False  # 設為 True 以載入微調後的 adapter
ADAPTER_PATH = "output/ct_report_adapter"

# 使用絕對路徑
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent  # chest-ct-report-generator
FEATURES_DIR = PROJECT_ROOT / "medsam2_segmentation" / "result" / "segmentation_20251204_152809" / "features" / "llm_data"

# 添加 medsam2_segmentation 到 path 以導入 location_estimator
try:
    from detection.common.location_estimator import LungLocationEstimator
    LOCATION_ESTIMATOR_AVAILABLE = True
except ImportError:
    LOCATION_ESTIMATOR_AVAILABLE = False
    print("警告: location_estimator 未載入，將無法估算解剖學位置")


class CTReportGenerator:
    """CT 影像報告生成器"""
    
    def __init__(self, model_id: str = MODEL_ID, use_adapter: bool = USE_ADAPTER, adapter_path: str = ADAPTER_PATH):
        self.model_id = model_id
        self.use_adapter = use_adapter
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        self.location_estimator = LungLocationEstimator() if LOCATION_ESTIMATOR_AVAILABLE else None
        self._load_model()
    
    def _load_model(self):
        """載入模型"""
        print(f"載入模型: {self.model_id}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
            torch.cuda.empty_cache()
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map="cuda:0",
                low_cpu_mem_usage=True,
            )
            
            if self.use_adapter:
                from peft import PeftModel
                print(f"載入 Adapter: {self.adapter_path}")
                self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
            
            self.model = self.model.eval()
            
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            print(f"GPU 記憶體使用: {allocated:.2f} GB")
        else:
            print("使用 CPU 運行...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,
                device_map="cpu",
            ).eval()
    
    def estimate_anatomical_location(self, centroid_x: float, centroid_y: float, 
                                       slice_index: int, total_slices: int,
                                       image_width: int = 512, image_height: int = 512) -> str:
        """
        估算病灶的解剖學位置
        
        Args:
            centroid_x: 病灶中心 X 座標
            centroid_y: 病灶中心 Y 座標  
            slice_index: 切片索引
            total_slices: 總切片數
            image_width: 影像寬度
            image_height: 影像高度
        
        Returns:
            解剖學位置縮寫 (如 RUL, RLL, LUL 等)
        """
        # 計算相對位置 (0-1)
        relative_x = centroid_x / image_width  # 0=左, 1=右
        relative_y = centroid_y / image_height  # 0=前, 1=後  
        slice_ratio = slice_index / total_slices if total_slices > 0 else 0.5
        
        # 判斷左右側 (病人座標系: 影像右邊=病人左側)
        # CT 影像通常是從病人足側看向頭側，所以影像右邊是病人左側
        if relative_x < 0.45:
            side = 'left'  # 影像左側 = 病人右側? 需要確認
        elif relative_x > 0.55:
            side = 'right'
        else:
            side = 'right' if relative_x >= 0.5 else 'left'
        
        # 判斷上中下區域
        if slice_ratio < 0.35:
            vertical = 'upper'
        elif slice_ratio < 0.65:
            vertical = 'middle'
        else:
            vertical = 'lower'
        
        # 綜合判斷肺葉
        if side == 'right':
            if vertical == 'upper':
                return 'RUL'
            elif vertical == 'lower':
                return 'RLL'
            else:  # middle
                if relative_y < 0.5:  # 前方
                    return 'RML'
                else:
                    return 'RUL' if slice_ratio < 0.5 else 'RLL'
        else:  # left
            if vertical == 'upper':
                if slice_ratio > 0.25 and relative_y < 0.45:
                    return 'Lingula'
                return 'LUL'
            elif vertical == 'lower':
                return 'LLL'
            else:  # middle
                if relative_y < 0.45:
                    return 'Lingula'
                return 'LUL' if slice_ratio < 0.5 else 'LLL'
    
    def build_prompt_from_features(self, features: Dict[str, Any], 
                                    detailed_features: Dict[str, Any] = None) -> str:
        """
        從分割特徵建立 prompt
        
        Args:
            features: LLM 特徵資料 (從 *_llm.json 讀取)
            detailed_features: 詳細特徵資料 (從 features.json 讀取，包含座標)
        
        Returns:
            格式化的 prompt
        """
        patient_id = features.get("patient_id", "未知")
        numerical = features.get("numerical_features", {})
        
        # 獲取病灶資訊
        total_lesions = numerical.get('total_lesions', 0)
        max_diameter = numerical.get('max_diameter_mm', 0)
        avg_diameter = numerical.get('avg_diameter_mm', 0)
        max_area = numerical.get('max_area_mm2', 0)
        avg_circularity = numerical.get('avg_circularity', 0)
        
        # 估算病灶位置
        lesion_locations = []
        if detailed_features and 'slices' in detailed_features:
            slices_data = detailed_features['slices']
            total_slices_count = len(slices_data)
            slice_indices = sorted([int(k) for k in slices_data.keys() if slices_data[k].get('lesions')])
            
            # 找出主要病灶 (最大的)
            max_lesion = None
            max_lesion_slice = None
            for slice_idx in slice_indices:
                slice_str = str(slice_idx)
                if slice_str in slices_data:
                    for lesion in slices_data[slice_str].get('lesions', []):
                        morph = lesion.get('morphological', {})
                        area = morph.get('area_mm2', 0)
                        if max_lesion is None or area > max_lesion.get('morphological', {}).get('area_mm2', 0):
                            max_lesion = lesion
                            max_lesion_slice = slice_idx
            
            if max_lesion and max_lesion_slice:
                morph = max_lesion.get('morphological', {})
                centroid_x = morph.get('centroid_x', 256)
                centroid_y = morph.get('centroid_y', 256)
                
                # 估算最大病灶的位置
                # 計算該切片在所有腫瘤切片中的相對位置
                if slice_indices:
                    min_slice = min(slice_indices)
                    max_slice = max(slice_indices)
                    slice_range = max_slice - min_slice if max_slice > min_slice else 1
                    relative_slice = (max_lesion_slice - min_slice) / slice_range
                else:
                    relative_slice = 0.5
                
                location = self.estimate_anatomical_location(
                    centroid_x, centroid_y, 
                    int(relative_slice * 100), 100
                )
                lesion_locations.append({
                    'location': location,
                    'diameter_mm': morph.get('equivalent_diameter_mm', max_diameter),
                    'area_mm2': morph.get('area_mm2', max_area)
                })
        
        # 構建位置描述
        if lesion_locations:
            main_lesion = lesion_locations[0]
            location_str = main_lesion['location']
            diameter_str = f"{main_lesion['diameter_mm']:.1f}"
        else:
            # 如果無法估算位置，使用預設值
            location_str = "lung"
            diameter_str = f"{max_diameter:.1f}"
        
        # 判斷病灶類型
        if avg_circularity > 0.8:
            lesion_type = "nodule"
            margin_desc = "smooth margins"
        elif avg_circularity > 0.6:
            lesion_type = "nodule"
            margin_desc = "moderately smooth margins"
        else:
            lesion_type = "opacity"
            margin_desc = "irregular margins"
        
        # 取得當前日期
        current_date = datetime.now().strftime("%Y/%m/%d")
        
        # 建立符合醫院格式的提示 (使用 few-shot 範例)
        prompt = f"""Generate a chest CT report. Output ONLY the report starting with "Study Date".

Example:
Study Date: 2025/12/05

Technique: 
Axial imaging of the chest was obtained without contrast. 

FINDINGS:

1.Lungs:
  Nodule in RUL, size 18.6mm with smooth margins.

2.Mediastinum:
  Lymph Nodes: no enlarged lymph nodes noted.

3.Vessels and Heart: 
  No cardiomegaly.

4.Pleural Spaces: 
  No pleural effusion.

5.Bones & Soft Tissue: 
  No significant findings.

IMPRESSION:
1.Nodule in RUL, size 18.6mm. Suggest follow-up.

---
Generate report for:

Study Date: {current_date}
Lesion: {lesion_type} in {location_str}
Size: {diameter_str} mm
Margins: {margin_desc}
Lesion count: {total_lesions}

Report:"""
        
        return prompt
    
    def generate_report(self, prompt: str, max_new_tokens: int = 1024) -> tuple[str, Dict[str, Any]]:
        """
        生成報告
        
        Args:
            prompt: 輸入提示
            max_new_tokens: 最大生成 token 數
        
        Returns:
            (生成的報告, 統計資訊)
        """
        # 使用 Llama 3.2 的對話格式
        formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        
        start_time = time.time()
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        # 解碼輸出
        full_response = self.tokenizer.decode(out[0], skip_special_tokens=True)
        
        # 提取模型回應部分 (移除 prompt)
        # 方法1: 找到 "Study Date" 開頭的報告內容
        if "Report:" in full_response:
            # 找到最後一個 "Report:" 後的內容
            parts = full_response.split("Report:")
            if len(parts) > 1:
                response = parts[-1].strip()
            else:
                response = full_response
        elif "Study Date" in full_response:
            # 找到最後一個 "Study Date" (模型生成的部分)
            parts = full_response.rsplit("Study Date", 1)
            if len(parts) > 1:
                response = "Study Date" + parts[-1].strip()
            else:
                response = full_response
        elif "model\n" in full_response:
            response = full_response.split("model\n")[-1].strip()
        elif "model" in full_response:
            response = full_response.split("model")[-1].strip()
        else:
            response = full_response
        
        # 計算統計
        input_tokens = inputs["input_ids"].shape[1]
        output_tokens = out.shape[1]
        generated_tokens = output_tokens - input_tokens
        inference_time = end_time - start_time
        
        stats = {
            "inference_time": inference_time,
            "input_tokens": input_tokens,
            "generated_tokens": generated_tokens,
            "tokens_per_sec": generated_tokens / inference_time if inference_time > 0 else 0
        }
        
        return response, stats
    
    def generate_from_features_file(self, features_path: str) -> tuple[str, Dict[str, Any]]:
        """
        從特徵檔案生成報告
        
        Args:
            features_path: LLM 特徵 JSON 檔案路徑
        
        Returns:
            (生成的報告, 統計資訊)
        """
        with open(features_path, 'r', encoding='utf-8') as f:
            features = json.load(f)
        
        prompt = self.build_prompt_from_features(features)
        return self.generate_report(prompt)


def load_all_features(features_dir: Path) -> list[Dict[str, Any]]:
    """載入所有 LLM 特徵檔案"""
    features_list = []
    for json_file in features_dir.glob("*_llm.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            features = json.load(f)
            features['_file_path'] = str(json_file)
            features_list.append(features)
    return features_list


def generate_slice_visualizations(patient_folder: Path, output_dir: Path, 
                                   detailed_features: Dict[str, Any]) -> int:
    """
    從特徵資料生成 slice 視覺化圖片
    
    由於原始 DICOM 可能不在同一位置，此函數生成簡化的視覺化，
    顯示病灶的邊界框和基本資訊。
    
    Args:
        patient_folder: 病患特徵資料夾
        output_dir: 輸出資料夾
        detailed_features: 詳細特徵資料
    
    Returns:
        生成的圖片數量
    """
    if not MATPLOTLIB_AVAILABLE:
        print("警告: matplotlib 未安裝，無法生成視覺化")
        return 0
    
    slices_data = detailed_features.get('slices', {})
    if not slices_data:
        return 0
    
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_count = 0
    
    for slice_idx, slice_info in slices_data.items():
        lesions = slice_info.get('lesions', [])
        if not lesions:
            continue
        
        # 創建視覺化圖
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # 設定黑色背景模擬 CT 影像
        ax.set_facecolor('black')
        ax.set_xlim(0, 512)
        ax.set_ylim(512, 0)  # Y 軸翻轉以符合影像座標
        
        # 繪製每個病灶的邊界框
        for lesion in lesions:
            bbox = lesion.get('bbox', [])
            morph = lesion.get('morphological', {})
            
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                
                # 繪製邊界框
                rect = patches.Rectangle(
                    (x1, y1), width, height,
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                ax.add_patch(rect)
                
                # 添加標籤
                diameter = morph.get('equivalent_diameter_mm', 0)
                area = morph.get('area_mm2', 0)
                label = f"D:{diameter:.1f}mm\nA:{area:.0f}mm²"
                ax.text(x1, y1 - 5, label, color='yellow', fontsize=10,
                       verticalalignment='bottom', fontweight='bold')
                
                # 繪製中心點
                cx = morph.get('centroid_x', (x1 + x2) / 2)
                cy = morph.get('centroid_y', (y1 + y2) / 2)
                ax.plot(cx, cy, 'r+', markersize=10, markeredgewidth=2)
        
        # 設定標題
        patient_id = detailed_features.get('patient_id', 'Unknown')[:30]
        metrics = slice_info.get('metrics', {})
        dice = metrics.get('dice', 0)
        ax.set_title(f"Slice {slice_idx} - {patient_id}\nDice: {dice:.3f}", 
                    color='white', fontsize=12, pad=10)
        ax.set_xlabel('X (pixels)', color='white')
        ax.set_ylabel('Y (pixels)', color='white')
        ax.tick_params(colors='white')
        
        # 添加網格
        ax.grid(True, alpha=0.3, color='gray')
        
        # 儲存圖片
        fig.patch.set_facecolor('black')
        output_path = output_dir / f"slice_{int(slice_idx):04d}_lesions.png"
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight',
                   facecolor='black', edgecolor='none')
        plt.close()
        generated_count += 1
    
    return generated_count


def main():
    """主程式"""
    # 初始化報告生成器
    generator = CTReportGenerator()
    
    # 載入特徵資料
    print(f"\n載入特徵資料: {FEATURES_DIR}")
    
    if not FEATURES_DIR.exists():
        print(f"錯誤: 特徵目錄不存在: {FEATURES_DIR}")
        return
    
    # 取得第一個特徵檔案作為範例
    feature_files = list(FEATURES_DIR.glob("*_llm.json"))
    if not feature_files:
        print("錯誤: 找不到特徵檔案")
        return
    
    print(f"找到 {len(feature_files)} 個特徵檔案")
    
    # 生成第一個病患的報告作為範例
    sample_file = feature_files[0]
    print(f"\n使用範例檔案: {sample_file.name}")
    
    with open(sample_file, 'r', encoding='utf-8') as f:
        features = json.load(f)
    
    # 嘗試載入詳細特徵 (包含座標資訊)
    detailed_features = None
    patient_folder_name = sample_file.stem.replace('_llm', '')  # 移除 _llm 後綴
    # 詳細特徵在 features/patients/ 資料夾下
    detailed_features_path = FEATURES_DIR.parent / "patients" / patient_folder_name / "features.json"
    
    if detailed_features_path.exists():
        print(f"載入詳細特徵: {detailed_features_path.name}")
        with open(detailed_features_path, 'r', encoding='utf-8') as f:
            detailed_features = json.load(f)
    else:
        print(f"詳細特徵檔案不存在: {detailed_features_path}")
    
    # 顯示特徵摘要
    print("\n" + "="*60)
    print("病患 ID:", features.get("patient_id", "未知"))
    print("病灶數量:", features.get("numerical_features", {}).get("total_lesions", "N/A"))
    
    # 顯示位置估算
    if detailed_features and 'slices' in detailed_features:
        slices = detailed_features['slices']
        print(f"分析切片數: {len(slices)}")
        
        # 找出並顯示主要病灶位置
        for slice_idx, slice_data in slices.items():
            for lesion in slice_data.get('lesions', []):
                morph = lesion.get('morphological', {})
                if morph.get('area_mm2', 0) > 50:  # 只顯示較大的病灶
                    centroid_x = morph.get('centroid_x', 256)
                    centroid_y = morph.get('centroid_y', 256)
                    diameter = morph.get('equivalent_diameter_mm', 0)
                    
                    # 估算位置
                    slice_indices = sorted([int(k) for k in slices.keys()])
                    relative_slice = (int(slice_idx) - min(slice_indices)) / max(len(slice_indices), 1)
                    location = generator.estimate_anatomical_location(
                        centroid_x, centroid_y, 
                        int(relative_slice * 100), 100
                    )
                    print(f"  切片 {slice_idx}: {location}, 直徑 {diameter:.1f}mm, 座標 ({centroid_x:.0f}, {centroid_y:.0f})")
    
    print("="*60)
    
    # 建立 prompt (使用詳細特徵)
    prompt = generator.build_prompt_from_features(
        features, 
        detailed_features=detailed_features
    )
    print("\n[Prompt 預覽]")
    print(prompt[:600] + "..." if len(prompt) > 600 else prompt)
    
    # 生成報告
    print("\n" + "="*60)
    print("生成報告中...")
    print("="*60 + "\n")
    
    response, stats = generator.generate_report(prompt)
    
    print(response)
    print("\n" + "="*60)
    print(f"推理時間: {stats['inference_time']:.2f} 秒")
    print(f"輸入 tokens: {stats['input_tokens']}")
    print(f"生成 tokens: {stats['generated_tokens']}")
    print(f"生成速度: {stats['tokens_per_sec']:.2f} tokens/秒")
    print("="*60)
    
    # 儲存結果
    output_dir = Path("output/generated_reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    patient_id = features.get('patient_id', 'unknown')[:50]
    patient_output_dir = output_dir / patient_id
    patient_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成 slice 視覺化圖片到 output 資料夾
    patient_features_folder = FEATURES_DIR.parent / "patients" / patient_folder_name
    slices_output_dir = patient_output_dir / "slices"
    
    print(f"病患特徵資料夾: {patient_features_folder}")
    
    if detailed_features:
        generated_count = generate_slice_visualizations(
            patient_features_folder, slices_output_dir, detailed_features
        )
        if generated_count > 0:
            print(f"已生成 {generated_count} 張 slice 視覺化圖片到: {slices_output_dir}")
        else:
            print("警告: 無法生成視覺化圖片")
    else:
        print("警告: 無詳細特徵資料，無法生成視覺化圖片")
    
    output_file = patient_output_dir / "report.json"
    result = {
        "patient_id": features.get("patient_id"),
        "generated_date": datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
        "input_features": features.get("numerical_features"),
        "generated_report": response,
        "stats": stats
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    # 同時儲存純文字報告
    report_txt_file = patient_output_dir / "report.txt"
    with open(report_txt_file, 'w', encoding='utf-8') as f:
        f.write(response)
    
    print(f"\n報告已儲存至: {output_file}")
    print(f"純文字報告: {report_txt_file}")


if __name__ == "__main__":
    main()
