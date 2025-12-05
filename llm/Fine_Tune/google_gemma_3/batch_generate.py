"""
批次生成所有病患的 CT 影像報告

此腳本會讀取所有 LLM 特徵檔案，並批次生成結構化醫學報告。
"""

import torch
import json
import os
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from generate_with_features import CTReportGenerator, load_all_features

# ===== 設定 =====
FEATURES_DIR = Path("../../medsam2_segmentation/result/segmentation_20251204_152809/features/llm_data")
OUTPUT_DIR = Path("output/batch_reports")
DOCTOR_NAME = "葉偉成醫師"
MAX_NEW_TOKENS = 1024


def batch_generate_reports(
    features_dir: Path = FEATURES_DIR,
    output_dir: Path = OUTPUT_DIR,
    doctor_name: str = DOCTOR_NAME,
    max_patients: int = None,
    skip_existing: bool = True
):
    """
    批次生成所有病患的報告
    
    Args:
        features_dir: LLM 特徵目錄
        output_dir: 輸出目錄
        doctor_name: 報告醫師名稱
        max_patients: 最大處理病患數 (None = 全部)
        skip_existing: 是否跳過已存在的報告
    """
    # 建立輸出目錄
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 載入所有特徵檔案
    print(f"載入特徵資料: {features_dir}")
    feature_files = list(features_dir.glob("*_llm.json"))
    
    if max_patients:
        feature_files = feature_files[:max_patients]
    
    print(f"共 {len(feature_files)} 個病患待處理")
    
    # 初始化報告生成器
    generator = CTReportGenerator()
    
    # 統計
    results = []
    total_time = 0
    total_tokens = 0
    success_count = 0
    skip_count = 0
    error_count = 0
    
    # 批次處理
    for feature_file in tqdm(feature_files, desc="生成報告"):
        patient_id = feature_file.stem.replace("_llm", "")
        output_file = output_dir / f"{patient_id}_report.json"
        
        # 檢查是否已存在
        if skip_existing and output_file.exists():
            skip_count += 1
            continue
        
        try:
            # 載入特徵
            with open(feature_file, 'r', encoding='utf-8') as f:
                features = json.load(f)
            
            # 生成報告
            prompt = generator.build_prompt_from_features(features, doctor_name)
            response, stats = generator.generate_report(prompt, MAX_NEW_TOKENS)
            
            # 儲存結果
            result = {
                "patient_id": features.get("patient_id"),
                "source_file": str(feature_file),
                "doctor_name": doctor_name,
                "generated_at": datetime.now().isoformat(),
                "input_features": {
                    "total_lesions": features.get("numerical_features", {}).get("total_lesions"),
                    "avg_area_mm2": features.get("numerical_features", {}).get("avg_area_mm2"),
                    "max_diameter_mm": features.get("numerical_features", {}).get("max_diameter_mm"),
                    "dice": features.get("numerical_features", {}).get("dice"),
                },
                "generated_report": response,
                "stats": stats
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            results.append(result)
            total_time += stats['inference_time']
            total_tokens += stats['generated_tokens']
            success_count += 1
            
        except Exception as e:
            print(f"\n錯誤處理 {patient_id}: {e}")
            error_count += 1
            continue
    
    # 輸出統計
    print("\n" + "="*60)
    print("批次處理完成")
    print("="*60)
    print(f"成功: {success_count}")
    print(f"跳過: {skip_count}")
    print(f"錯誤: {error_count}")
    if success_count > 0:
        print(f"總推理時間: {total_time:.2f} 秒")
        print(f"總生成 tokens: {total_tokens}")
        print(f"平均每報告時間: {total_time / success_count:.2f} 秒")
        print(f"平均生成速度: {total_tokens / total_time:.2f} tokens/秒")
    print(f"輸出目錄: {output_dir}")
    
    # 儲存批次統計
    summary = {
        "generated_at": datetime.now().isoformat(),
        "features_dir": str(features_dir),
        "output_dir": str(output_dir),
        "doctor_name": doctor_name,
        "statistics": {
            "total_patients": len(feature_files),
            "success": success_count,
            "skipped": skip_count,
            "errors": error_count,
            "total_time_sec": total_time,
            "total_tokens": total_tokens,
            "avg_time_per_report": total_time / success_count if success_count > 0 else 0,
            "avg_tokens_per_sec": total_tokens / total_time if total_time > 0 else 0
        }
    }
    
    summary_file = output_dir / "batch_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"批次統計已儲存至: {summary_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="批次生成 CT 影像報告")
    parser.add_argument("--features-dir", type=str, default=str(FEATURES_DIR),
                       help="LLM 特徵目錄路徑")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                       help="輸出目錄路徑")
    parser.add_argument("--doctor", type=str, default=DOCTOR_NAME,
                       help="報告醫師名稱")
    parser.add_argument("--max-patients", type=int, default=None,
                       help="最大處理病患數")
    parser.add_argument("--no-skip", action="store_true",
                       help="不跳過已存在的報告")
    
    args = parser.parse_args()
    
    batch_generate_reports(
        features_dir=Path(args.features_dir),
        output_dir=Path(args.output_dir),
        doctor_name=args.doctor,
        max_patients=args.max_patients,
        skip_existing=not args.no_skip
    )
