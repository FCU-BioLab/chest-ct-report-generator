#!/usr/bin/env python3
"""
MedSAM2 模型測試腳本
==================

用於測試 MedSAM2 安裝和模型可用性
"""

import sys
from pathlib import Path

def test_imports():
    """測試必要的模組導入"""
    print("測試模組導入...")
    
    # 測試基本依賴
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name()}")
    except ImportError:
        print("✗ PyTorch 未安裝")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy: {np.__version__}")
    except ImportError:
        print("✗ NumPy 未安裝")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV: {cv2.__version__}")
    except ImportError:
        print("✗ OpenCV 未安裝")
        return False
    
    try:
        import nibabel as nib
        print(f"✓ NiBabel: {nib.__version__}")
    except ImportError:
        print("✗ NiBabel 未安裝")
        return False
    
    try:
        import pydicom
        print(f"✓ PyDICOM: {pydicom.__version__}")
    except ImportError:
        print("✗ PyDICOM 未安裝")
        return False
    
    # 測試 MedSAM2
    try:
        import sys
        from pathlib import Path
        
        # Add MedSAM2 path to sys.path
        medsam2_path = Path("MedSAM2")
        if medsam2_path.exists():
            sys.path.insert(0, str(medsam2_path))
        
        from sam2.build_sam import build_sam2_video_predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        print("✓ MedSAM2 模組可用")
        return True
    except ImportError as e:
        print(f"✗ MedSAM2 模組不可用: {e}")
        print("  請確保 MedSAM2 目錄存在並包含 sam2 模組")
        
        # 測試原始 SAM 作為備用
        try:
            from transformers import SamModel, SamProcessor
            print("✓ 原始 SAM (Transformers) 可用作為備用")
            return True
        except ImportError:
            print("✗ 原始 SAM 也不可用")
            return False

def test_model_files():
    """測試模型檔案是否存在"""
    print("\n測試模型檔案...")
    
    # 檢查MedSAM2目錄中的checkpoints
    checkpoint_dir = Path("MedSAM2/checkpoints")
    
    if not checkpoint_dir.exists():
        print(f"✗ 模型目錄不存在: {checkpoint_dir}")
        print("  請確保 MedSAM2 已正確克隆並下載模型")
        return False
    
    models = [
        "MedSAM2_latest.pt",
        "MedSAM2_2411.pt",
        "MedSAM2_CTLesion.pt",
        "sam2.1_hiera_tiny.pt"
    ]
    
    found_models = []
    for model in models:
        model_path = checkpoint_dir / model
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"✓ {model} ({size_mb:.1f} MB)")
            found_models.append(model)
        else:
            print(f"✗ {model} 不存在")
    
    if found_models:
        print(f"找到 {len(found_models)} 個模型檔案")
        return True
    else:
        print("未找到任何模型檔案")
        return False

def test_patient_data():
    """測試患者數據是否存在"""
    print("\n測試患者數據...")
    
    # 先嘗試從項目根目錄的datasets目錄查找
    data_dir = Path("../datasets/all_patient_data")
    if not data_dir.exists():
        # 嘗試當前目錄
        data_dir = Path("all_patient_data")
        if not data_dir.exists():
            print(f"✗ 患者數據目錄不存在: {data_dir}")
            print("  請確保患者數據位於 datasets/all_patient_data 目錄中")
            return False
    
    patients = [d for d in data_dir.iterdir() if d.is_dir()]
    if patients:
        print(f"✓ 找到 {len(patients)} 個患者數據")
        
        # 檢查第一個患者的結構
        sample_patient = patients[0]
        print(f"  檢查樣本患者: {sample_patient.name}")
        
        dicom_found = False
        xml_found = False
        
        for subdir in ["dicom", "dicom_files"]:
            dicom_dir = sample_patient / subdir
            if dicom_dir.exists():
                dicom_files = list(dicom_dir.glob("*.dcm"))
                if dicom_files:
                    print(f"    ✓ DICOM 檔案: {len(dicom_files)} 個")
                    dicom_found = True
                    break
        
        for subdir in ["xml", "xml_annotations"]:
            xml_dir = sample_patient / subdir
            if xml_dir.exists():
                xml_files = list(xml_dir.glob("*.xml"))
                if xml_files:
                    print(f"    ✓ XML 標註: {len(xml_files)} 個")
                    xml_found = True
                    break
        
        if not dicom_found:
            print("    ✗ 未找到 DICOM 檔案")
        if not xml_found:
            print("    ✗ 未找到 XML 標註檔案")
            
        return dicom_found and xml_found
    else:
        print("✗ 未找到患者數據")
        return False

def main():
    """主測試函數"""
    print("MedSAM2 環境測試")
    print("=" * 50)
    
    # 測試導入
    imports_ok = test_imports()
    
    # 測試模型檔案
    models_ok = test_model_files()
    
    # 測試患者數據
    data_ok = test_patient_data()
    
    print("\n" + "=" * 50)
    print("測試結果總結:")
    print(f"  模組導入: {'✓ 通過' if imports_ok else '✗ 失敗'}")
    print(f"  模型檔案: {'✓ 通過' if models_ok else '✗ 失敗'}")
    print(f"  患者數據: {'✓ 通過' if data_ok else '✗ 失敗'}")
    
    if imports_ok:
        print("\n建議的執行命令:")
        if models_ok:
            print("  python sam_seg.py --patient_id A0001")
        else:
            print("  python sam_seg.py --patient_id A0001")
        print("\n註意：如果從項目根目錄執行，請使用:")
        print("  python medsam2_segmentation/sam_seg.py --patient_id A0001")
    else:
        print("\n請先解決導入問題，然後重新測試")
    
    return imports_ok and (models_ok or data_ok)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
