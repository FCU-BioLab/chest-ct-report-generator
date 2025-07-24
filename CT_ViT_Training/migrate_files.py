#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CT-ViT 訓練系統遷移腳本
將datasets_process中的CT-ViT相關文件移動到新的CT_ViT_Training目錄

功能：
1. 備份現有文件
2. 移動CT-ViT相關文件
3. 更新導入路徑
4. 清理舊文件

作者: GitHub Copilot
日期: 2025-07-22
"""

import os
import shutil
import json
import time
from pathlib import Path
from typing import List, Dict

class CTViTMigrator:
    """CT-ViT文件遷移器"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.old_dir = self.base_dir / "datasets_process"
        self.new_dir = self.base_dir / "CT_ViT_Training"
        
        # 需要遷移的文件映射
        self.file_migrations = {
            # 源文件 -> 目標位置
            "train_ct_vit.py": "legacy/train_ct_vit_original.py",
            "inference_ct_vit.py": "legacy/inference_ct_vit_original.py", 
            "config_ct_vit.yaml": "legacy/config_ct_vit_original.yaml",
            "requirements_ct_vit.txt": "legacy/requirements_ct_vit_original.txt",
            "run_ct_vit.bat": "legacy/run_ct_vit_original.bat",
            "README_CT_ViT.md": "legacy/README_CT_ViT_original.md"
        }
        
        # 需要更新導入的文件
        self.import_updates = {
            # 舊導入 -> 新導入
            "from datasets_process.": "from CT_ViT_Training.src.",
            "import datasets_process": "import CT_ViT_Training.src",
            "datasets_process/": "CT_ViT_Training/",
        }
    
    def backup_files(self) -> str:
        """備份現有文件"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_dir = self.base_dir / f"backup_ct_vit_{timestamp}"
        backup_dir.mkdir(exist_ok=True)
        
        print(f"📦 創建備份目錄: {backup_dir}")
        
        # 備份舊文件
        for source_file in self.file_migrations.keys():
            source_path = self.old_dir / source_file
            if source_path.exists():
                backup_path = backup_dir / f"old_{source_file}"
                shutil.copy2(source_path, backup_path)
                print(f"   ✓ 備份 {source_file}")
        
        # 備份新目錄（如果存在）
        if self.new_dir.exists():
            new_backup = backup_dir / "CT_ViT_Training_current"
            try:
                shutil.copytree(self.new_dir, new_backup)
                print(f"   ✓ 備份當前 CT_ViT_Training 目錄")
            except Exception as e:
                print(f"   ⚠ 備份當前目錄失敗: {e}")
        
        return str(backup_dir)
    
    def migrate_files(self):
        """遷移文件到新目錄"""
        print(f"📁 遷移文件到 {self.new_dir}")
        
        # 確保新目錄結構存在
        (self.new_dir / "legacy").mkdir(parents=True, exist_ok=True)
        
        # 移動文件
        for source_file, target_relative in self.file_migrations.items():
            source_path = self.old_dir / source_file
            target_path = self.new_dir / target_relative
            
            if source_path.exists():
                # 確保目標目錄存在
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 移動文件
                shutil.move(str(source_path), str(target_path))
                print(f"   ✓ 移動 {source_file} -> {target_relative}")
            else:
                print(f"   ⚠ 文件不存在: {source_file}")
    
    def update_imports(self):
        """更新Python文件中的導入路徑"""
        print(f"🔧 更新導入路徑...")
        
        # 尋找所有Python文件
        python_files = []
        for root, dirs, files in os.walk(self.base_dir):
            # 跳過備份目錄
            if "backup_ct_vit" in root:
                continue
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
        
        updated_count = 0
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # 應用導入更新
                for old_import, new_import in self.import_updates.items():
                    content = content.replace(old_import, new_import)
                
                # 如果內容有變化，寫回文件
                if content != original_content:
                    with open(py_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"   ✓ 更新 {py_file.relative_to(self.base_dir)}")
                    updated_count += 1
                    
            except Exception as e:
                print(f"   ⚠ 更新失敗 {py_file}: {e}")
        
        print(f"   📝 共更新 {updated_count} 個文件")
    
    def create_migration_report(self, backup_dir: str):
        """創建遷移報告"""
        report = {
            "migration_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backup_location": backup_dir,
            "migrated_files": {},
            "new_structure": {},
            "notes": []
        }
        
        # 記錄遷移的文件
        for source_file, target_relative in self.file_migrations.items():
            source_path = self.old_dir / source_file
            target_path = self.new_dir / target_relative
            
            report["migrated_files"][source_file] = {
                "source_existed": source_path.exists(),
                "target_location": str(target_relative),
                "target_exists": target_path.exists()
            }
        
        # 記錄新結構
        if self.new_dir.exists():
            for root, dirs, files in os.walk(self.new_dir):
                rel_root = Path(root).relative_to(self.new_dir)
                if str(rel_root) not in report["new_structure"]:
                    report["new_structure"][str(rel_root)] = []
                report["new_structure"][str(rel_root)] = files
        
        # 添加說明
        report["notes"] = [
            "所有CT-ViT相關文件已移動到CT_ViT_Training目錄",
            "舊文件已備份在指定的備份目錄中", 
            "新的模組化結構位於CT_ViT_Training/src/",
            "使用scripts/run_ct_vit.bat或run_ct_vit.sh啟動系統",
            "配置文件位於CT_ViT_Training/configs/"
        ]
        
        # 保存報告
        report_path = self.new_dir / "migration_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📋 遷移報告已保存: {report_path}")
        return report
    
    def cleanup_old_files(self):
        """清理舊目錄中的CT-ViT文件"""
        print(f"🧹 清理舊目錄...")
        
        # 檢查是否還有CT-ViT相關文件
        remaining_files = []
        for file_pattern in ["*ct_vit*", "*CT_ViT*", "*CT-ViT*"]:
            remaining_files.extend(self.old_dir.glob(file_pattern))
        
        if remaining_files:
            print(f"   ⚠ 發現 {len(remaining_files)} 個可能的相關文件:")
            for f in remaining_files:
                print(f"     - {f.name}")
            
            response = input("   是否要刪除這些文件? (y/N): ")
            if response.lower() == 'y':
                for f in remaining_files:
                    try:
                        if f.is_file():
                            f.unlink()
                        elif f.is_dir():
                            shutil.rmtree(f)
                        print(f"     ✓ 已刪除 {f.name}")
                    except Exception as e:
                        print(f"     ✗ 刪除失敗 {f.name}: {e}")
        else:
            print("   ✓ 沒有發現需要清理的文件")
    
    def run_migration(self):
        """執行完整的遷移流程"""
        print("🚀 開始 CT-ViT 文件遷移")
        print("=" * 50)
        
        try:
            # 1. 備份文件
            backup_dir = self.backup_files()
            
            # 2. 遷移文件
            self.migrate_files()
            
            # 3. 更新導入
            self.update_imports()
            
            # 4. 創建遷移報告
            report = self.create_migration_report(backup_dir)
            
            # 5. 清理舊文件（可選）
            self.cleanup_old_files()
            
            print("\n" + "=" * 50)
            print("✅ 遷移完成！")
            print(f"📦 備份位置: {backup_dir}")
            print(f"📁 新位置: {self.new_dir}")
            print(f"📋 詳細報告: {self.new_dir}/migration_report.json")
            print("\n🎯 下一步:")
            print("1. 檢查新的CT_ViT_Training目錄結構")
            print("2. 運行 scripts/run_ct_vit.bat 測試系統")
            print("3. 安裝依賴: pip install -r CT_ViT_Training/requirements.txt")
            
        except Exception as e:
            print(f"\n❌ 遷移過程中出現錯誤: {e}")
            print(f"請檢查備份: {backup_dir if 'backup_dir' in locals() else '未創建'}")
            raise

def main():
    """主函數"""
    print("CT-ViT 訓練系統文件遷移工具")
    print("此工具將把datasets_process中的CT-ViT相關文件移動到新的模組化結構中")
    print()
    
    response = input("是否要繼續進行遷移? (y/N): ")
    if response.lower() != 'y':
        print("遷移取消")
        return
    
    migrator = CTViTMigrator()
    migrator.run_migration()

if __name__ == "__main__":
    main()
