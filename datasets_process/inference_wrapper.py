#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CT-ViT 推理包裝器
方便從 datasets_process 目錄調用 CT_ViT_Training 的推理功能

作者: GitHub Copilot
日期: 2025-07-24
"""

import os
import sys
from pathlib import Path

# 添加 CT_ViT_Training 路徑
ct_vit_path = Path(__file__).parent.parent / "CT_ViT_Training"
sys.path.insert(0, str(ct_vit_path))

# 導入現有的推理模組
try:
    from inference import CTViTInference, main as inference_main
    print("✅ 成功導入 CT_ViT_Training/inference.py")
except ImportError as e:
    print(f"❌ 無法導入推理模組: {e}")
    print("請確保 CT_ViT_Training/inference.py 存在且可用")
    sys.exit(1)

def main():
    """包裝器主函數"""
    print("🔄 調用 CT_ViT_Training 的推理功能...")
    print(f"   CT-ViT 路徑: {ct_vit_path}")
    print("-" * 50)
    
    # 調用原始的推理主函數
    inference_main()

if __name__ == "__main__":
    main()
