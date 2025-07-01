#!/usr/bin/env python3
"""
模型下載腳本
支援下載多種模型到本地
"""

import os
import sys
from pathlib import Path
import argparse

def download_sentence_transformer():
    """下載sentence-transformers模型"""
    print("正在下載 Sentence Transformer 模型...")
    
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "sentence_transformer"
    
    try:
        from sentence_transformers import SentenceTransformer
        
        if model_path.exists():
            print(f"模型已存在: {model_path}")
            response = input("是否重新下載 Sentence Transformer? (y/N): ").strip().lower()
            if response != 'y':
                return True
        
        print("開始下載 all-MiniLM-L6-v2 模型...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        model.save(str(model_path))
        
        total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
        size_mb = total_size / (1024 * 1024)
        print(f"✅ Sentence Transformer 下載完成! 大小: {size_mb:.1f} MB")
        return True
        
    except ImportError:
        print("❌ sentence-transformers 未安裝")
        print("請執行: pip install sentence-transformers")
        return False
    except Exception as e:
        print(f"❌ Sentence Transformer 下載失敗: {str(e)}")
        return False

def download_gemma_model():
    """下載Gemma 3 4B模型"""
    print("正在下載 Google Gemma 3 4B 模型...")
    
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "gemma-3-4b-it"
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        if model_path.exists():
            print(f"模型已存在: {model_path}")
            response = input("是否重新下載 Gemma 3 4B? (y/N): ").strip().lower()
            if response != 'y':
                return True
        
        # 檢查Hugging Face登入狀態
        try:
            from huggingface_hub import whoami
            user_info = whoami()
            print(f"已登入 Hugging Face: {user_info['name']}")
        except Exception:
            print("❌ 未登入 Hugging Face")
            print("請先執行: huggingface-cli login")
            return False
        
        model_name = "google/gemma-3-4b-it"
        
        print("開始下載 Gemma 3 4B 模型...")
        print("⚠️  注意: 此模型約 8.6GB，下載時間較長")
        
        # 下載 tokenizer
        print("正在下載 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.save_pretrained(str(model_path))
        
        # 下載模型
        print("正在下載模型文件...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        model.save_pretrained(str(model_path))
        
        # 計算大小
        total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
        size_gb = total_size / (1024 * 1024 * 1024)
        print(f"✅ Gemma 3 4B 下載完成! 大小: {size_gb:.1f} GB")
        return True
        
    except ImportError:
        print("❌ transformers 或 huggingface_hub 未安裝")
        print("請執行: pip install transformers huggingface_hub")
        return False
    except Exception as e:
        print(f"❌ Gemma 3 4B 下載失敗: {str(e)}")
        print("可能的原因:")
        print("1. 未登入 Hugging Face")
        print("2. 沒有 Gemma 模型訪問權限")
        print("3. 網路連接問題")
        print("4. 磁碟空間不足")
        return False

def main():
    parser = argparse.ArgumentParser(description='下載模型到本地')
    parser.add_argument('--model', choices=['sentence', 'gemma', 'all'], 
                       default='all', help='選擇要下載的模型')
    parser.add_argument('model_type', nargs='?', choices=['sentence', 'gemma', 'all'],
                       help='模型類型 (可直接指定，無需--model)')
    args = parser.parse_args()
    
    # 支援兩種方式：python script.py all 或 python script.py --model all
    model_choice = args.model_type if args.model_type else args.model
    
    print("=" * 60)
    print("模型下載工具")
    print("=" * 60)
    print(f"選擇的模型: {model_choice}")
    print()
    
    success_count = 0
    total_count = 0
    
    if model_choice in ['sentence', 'all']:
        total_count += 1
        if download_sentence_transformer():
            success_count += 1
        print()
    
    if model_choice in ['gemma', 'all']:
        total_count += 1
        if download_gemma_model():
            success_count += 1
        print()
    
    print("=" * 60)
    print(f"下載完成: {success_count}/{total_count} 個模型成功")
    
    if success_count > 0:
        print("\n下載的模型:")
        model_dir = Path("model")
        if model_dir.exists():
            for item in model_dir.iterdir():
                if item.is_dir():
                    total_size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                    if total_size > 1024**3:  # > 1GB
                        size_str = f"{total_size / (1024**3):.1f} GB"
                    else:
                        size_str = f"{total_size / (1024**2):.1f} MB"
                    print(f"  📁 {item.name} ({size_str})")
        
        print(f"\n模型位置: {model_dir.absolute()}")
        print("現在可以運行程序使用本地模型!")
    
    if success_count < total_count:
        sys.exit(1)

if __name__ == "__main__":
    main()
