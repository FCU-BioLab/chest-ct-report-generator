#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UNet++ Training Runner with Balanced Patient-Based Data Split
為層次化數據結構運行 UNet++ 訓練的包裝腳本，實現按病例分割和數據平衡

該腳本：
1. 檢查扁平化數據集是否存在，如果不存在則創建
2. 按病例進行數據分割，確保同一病例的數據不會分散
3. 平衡每個病例的病灶和正常數據，避免數據偏斜
4. 運行 UNet++ 訓練

作者: GitHub Copilot
日期: 2025-09-19
"""

import os
import sys
import shutil
from pathlib import Path
import argparse
import logging
import random
from typing import List, Tuple
import xml.etree.ElementTree as ET
import torch

# 添加當前目錄到路徑
sys.path.append(str(Path(__file__).parent))

from train_unetpp import train_unetpp_detector

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 默認配置常量
DEFAULT_CONFIG = {
    'batch_size': 2,
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'num_workers': 2,
    'save_dir': './checkpoints',
    'log_dir': './logs',
    'resume_from': None,
    'multi_class_segmentation': False,
    'train_val_split_ratio': 0.8,
    'max_lesion_ratio': 0.7,
    'max_files_per_patient': 50
}


def create_empty_xml(xml_path: Path) -> None:
    """
    創建一個空的XML標註文件（表示無病灶）
    
    Args:
        xml_path: XML文件路徑
    """
    root = ET.Element("annotation")
    
    # 添加基本結構
    folder = ET.SubElement(root, "folder")
    folder.text = "images"
    
    filename = ET.SubElement(root, "filename")
    filename.text = xml_path.stem + ".dcm"
    
    # 添加圖像尺寸信息（使用默認值）
    size = ET.SubElement(root, "size")
    width = ET.SubElement(size, "width")
    width.text = "512"
    height = ET.SubElement(size, "height")
    height.text = "512"
    depth = ET.SubElement(size, "depth")
    depth.text = "1"
    
    # 創建XML樹並保存
    tree = ET.ElementTree(root)
    tree.write(xml_path, encoding='utf-8', xml_declaration=True)


def copy_file_with_fallback(source_path: Path, target_path: Path) -> None:
    """
    複製文件，優先使用硬鏈接，失敗時使用常規複製
    
    Args:
        source_path: 源文件路徑
        target_path: 目標文件路徑
    """
    if os.name == 'nt':  # Windows
        try:
            os.link(source_path, target_path)
        except (OSError, NotImplementedError):
            shutil.copy2(source_path, target_path)
    else:  # Unix/Linux
        os.symlink(source_path, target_path)


def check_and_create_flat_dataset(split_dataset_dir: str, flat_dataset_dir: str, config: dict = None):
    """
    檢查扁平化數據集是否存在，如果不存在則創建，如果存在則驗證格式
    
    Args:
        split_dataset_dir: 原始分層數據集目錄
        flat_dataset_dir: 目標扁平化數據集目錄
        config: 配置參數
    
    Returns:
        (train_data_dir, train_xml_dir, val_data_dir, val_xml_dir): 訓練和驗證數據目錄路徑
    """
    flat_path = Path(flat_dataset_dir)
    
    # 定義預期的目錄結構
    expected_dirs = {
        'train_data': flat_path / 'train_data',
        'train_xml': flat_path / 'train_xml', 
        'val_data': flat_path / 'val_data',
        'val_xml': flat_path / 'val_xml'
    }
    
    # 檢查是否所有必要的目錄都存在
    all_dirs_exist = all(dir_path.exists() for dir_path in expected_dirs.values())
    
    if all_dirs_exist:
        logger.info(f"找到現有的扁平化數據集: {flat_dataset_dir}")
        
        # 驗證數據集格式
        if validate_flat_dataset_format(expected_dirs):
            logger.info("數據集格式驗證通過，使用現有數據集")
            return (str(expected_dirs['train_data']), str(expected_dirs['train_xml']),
                   str(expected_dirs['val_data']), str(expected_dirs['val_xml']))
        else:
            logger.warning("數據集格式驗證失敗，將重新創建數據集")
            # 清理現有目錄
            if flat_path.exists():
                shutil.rmtree(flat_path)
    
    # 創建新的扁平化數據集
    logger.info(f"創建新的扁平化數據集: {flat_dataset_dir}")
    return create_balanced_flat_dataset(split_dataset_dir, flat_dataset_dir, config)


def validate_flat_dataset_format(expected_dirs: dict) -> bool:
    """
    驗證扁平化數據集的格式是否正確
    
    Args:
        expected_dirs: 預期的目錄結構字典
    
    Returns:
        是否格式正確
    """
    try:
        # 檢查每個目錄是否存在且包含文件
        for dir_name, dir_path in expected_dirs.items():
            if not dir_path.exists():
                logger.warning(f"目錄不存在: {dir_path}")
                return False
            
            # 檢查目錄是否包含預期的文件類型
            if 'data' in dir_name:
                files = list(dir_path.glob("*.dcm"))
                if len(files) == 0:
                    logger.warning(f"數據目錄 {dir_path} 中沒有 DICOM 文件")
                    return False
            elif 'xml' in dir_name:
                files = list(dir_path.glob("*.xml"))
                if len(files) == 0:
                    logger.warning(f"XML目錄 {dir_path} 中沒有 XML 文件")
                    return False
        
        # 檢查訓練和驗證數據的匹配性
        train_dcm_files = set(f.stem for f in expected_dirs['train_data'].glob("*.dcm"))
        train_xml_files = set(f.stem for f in expected_dirs['train_xml'].glob("*.xml"))
        val_dcm_files = set(f.stem for f in expected_dirs['val_data'].glob("*.dcm"))
        val_xml_files = set(f.stem for f in expected_dirs['val_xml'].glob("*.xml"))
        
        # 檢查訓練集和驗證集的文件匹配
        if train_dcm_files != train_xml_files:
            logger.warning("訓練集中 DICOM 和 XML 文件不匹配")
            return False
        
        if val_dcm_files != val_xml_files:
            logger.warning("驗證集中 DICOM 和 XML 文件不匹配")
            return False
        
        # 檢查是否有重疊
        if train_dcm_files & val_dcm_files:
            logger.warning("訓練集和驗證集存在重疊文件")
            return False
        
        logger.info(f"數據集驗證通過: 訓練集 {len(train_dcm_files)} 對, 驗證集 {len(val_dcm_files)} 對")
        return True
        
    except Exception as e:
        logger.error(f"驗證數據集格式時出錯: {e}")
        return False


def analyze_patient_lesion_distribution(patient_dir: Path) -> Tuple[int, int]:
    """
    分析單個患者的病灶分布
    
    Args:
        patient_dir: 患者目錄路徑
    
    Returns:
        (有病灶的文件數, 無病灶的文件數)
    """
    xml_dir = patient_dir / "xml_annotations"
    if not xml_dir.exists():
        return 0, 0
    
    lesion_count = 0
    no_lesion_count = 0
    
    xml_files = list(xml_dir.glob("*.xml"))
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # 檢查是否有標註框（病灶）
            objects = root.findall('.//object')
            if objects:
                lesion_count += 1
            else:
                no_lesion_count += 1
                
        except Exception as e:
            logger.warning(f"無法解析 XML 文件 {xml_file}: {e}")
            continue
    
    return lesion_count, no_lesion_count


def split_patients_by_case(split_dataset_dir: str, train_ratio: float = 0.8, seed: int = 42) -> Tuple[List[str], List[str]]:
    """
    按病例分割患者，確保同一病例的所有數據都在同一集合中
    
    Args:
        split_dataset_dir: 分層數據集目錄
        train_ratio: 訓練集比例
        seed: 隨機種子
    
    Returns:
        (train_patients, val_patients): 訓練和驗證病例列表
    """
    train_dir = Path(split_dataset_dir) / 'train'
    if not train_dir.exists():
        raise FileNotFoundError(f"訓練數據目錄不存在: {train_dir}")
    
    # 獲取所有患者目錄
    patient_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
    patient_ids = [d.name for d in patient_dirs]
    
    logger.info(f"找到 {len(patient_ids)} 個患者")
    
    # 分析每個患者的病灶分布
    patient_stats = {}
    total_lesion_files = 0
    total_no_lesion_files = 0
    
    for patient_dir in patient_dirs:
        lesion_count, no_lesion_count = analyze_patient_lesion_distribution(patient_dir)
        patient_stats[patient_dir.name] = {
            'lesion_count': lesion_count,
            'no_lesion_count': no_lesion_count,
            'total_count': lesion_count + no_lesion_count
        }
        total_lesion_files += lesion_count
        total_no_lesion_files += no_lesion_count
    
    logger.info(f"總計: {total_lesion_files} 個有病灶文件, {total_no_lesion_files} 個無病灶文件")
    
    # 設置隨機種子
    random.seed(seed)
    
    # 根據病灶分布進行分層抽樣
    # 將患者按病灶比例分組
    patients_with_lesions = []
    patients_without_lesions = []
    patients_mixed = []
    
    for patient_id, stats in patient_stats.items():
        if stats['lesion_count'] > 0 and stats['no_lesion_count'] > 0:
            patients_mixed.append(patient_id)
        elif stats['lesion_count'] > 0:
            patients_with_lesions.append(patient_id)
        elif stats['no_lesion_count'] > 0:
            patients_without_lesions.append(patient_id)
    
    logger.info(f"患者分布: {len(patients_mixed)} 混合型, {len(patients_with_lesions)} 純病灶型, {len(patients_without_lesions)} 純正常型")
    
    # 對每組進行隨機打亂
    random.shuffle(patients_mixed)
    random.shuffle(patients_with_lesions)
    random.shuffle(patients_without_lesions)
    
    # 按比例分割每組
    def split_group(group, ratio):
        split_point = int(len(group) * ratio)
        return group[:split_point], group[split_point:]
    
    train_mixed, val_mixed = split_group(patients_mixed, train_ratio)
    train_lesions, val_lesions = split_group(patients_with_lesions, train_ratio)
    train_normal, val_normal = split_group(patients_without_lesions, train_ratio)
    
    # 合併結果
    train_patients = train_mixed + train_lesions + train_normal
    val_patients = val_mixed + val_lesions + val_normal
    
    # 打亂最終列表
    random.shuffle(train_patients)
    random.shuffle(val_patients)
    
    logger.info(f"分割結果: 訓練集 {len(train_patients)} 患者, 驗證集 {len(val_patients)} 患者")
    
    # 統計分割後的病灶分布
    train_lesion_count = sum(patient_stats[p]['lesion_count'] for p in train_patients)
    train_no_lesion_count = sum(patient_stats[p]['no_lesion_count'] for p in train_patients)
    val_lesion_count = sum(patient_stats[p]['lesion_count'] for p in val_patients)
    val_no_lesion_count = sum(patient_stats[p]['no_lesion_count'] for p in val_patients)
    
    logger.info(f"訓練集病灶分布: {train_lesion_count} 有病灶, {train_no_lesion_count} 無病灶")
    logger.info(f"驗證集病灶分布: {val_lesion_count} 有病灶, {val_no_lesion_count} 無病灶")
    
    return train_patients, val_patients


def balance_patient_data(patient_dir: Path, max_lesion_ratio: float = 0.7, max_files_per_patient: int = 50) -> Tuple[List[Path], List[Path]]:
    """
    平衡單個患者的病灶和正常數據
    
    Args:
        patient_dir: 患者目錄
        max_lesion_ratio: 最大病灶比例
        max_files_per_patient: 每個患者最大文件數
    
    Returns:
        (selected_dicom_files, selected_xml_files): 選中的DICOM和XML文件列表
    """
    dicom_dir = patient_dir / "dicom_files"
    xml_dir = patient_dir / "xml_annotations"
    
    if not dicom_dir.exists():
        return [], []
    
    # 獲取所有DICOM文件
    dicom_files = list(dicom_dir.glob("*.dcm"))
    xml_files = list(xml_dir.glob("*.xml")) if xml_dir.exists() else []
    
    # 讀取文件對應關係
    file_list_path = patient_dir / f"{patient_dir.name}_file_list.csv"
    uid_to_dicom = {}
    
    if file_list_path.exists():
        try:
            import csv
            with open(file_list_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['File_Type'] == 'DICOM':
                        uid = row['UID']
                        file_name = row['File_Name']
                        dicom_path = dicom_dir / file_name
                        if dicom_path.exists():
                            uid_to_dicom[uid] = dicom_path
        except Exception as e:
            logger.warning(f"無法讀取文件列表 {file_list_path}: {e}")
    
    # 分析所有DICOM文件的病灶情況
    lesion_files = []
    normal_files = []
    
    # 首先處理有XML配對的文件
    matched_dicom_files = set()
    for xml_file in xml_files:
        uid = xml_file.stem  # XML文件名就是UID
        if uid in uid_to_dicom:
            dicom_file = uid_to_dicom[uid]
            matched_dicom_files.add(dicom_file)
            
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # 檢查是否有標註框（病灶）
                objects = root.findall('.//object')
                if objects:
                    lesion_files.append((dicom_file, xml_file))
                else:
                    normal_files.append((dicom_file, xml_file))
                    
            except Exception as e:
                logger.warning(f"無法解析 XML 文件 {xml_file}: {e}")
                # 解析失敗的話也當作正常文件
                normal_files.append((dicom_file, xml_file))
                continue
    
    # 將所有沒有配對到XML的DICOM文件當作正常文件（沒有病灶）
    for dicom_file in dicom_files:
        if dicom_file not in matched_dicom_files:
            # 創建一個空的XML文件路徑作為佔位符，表示這是正常文件
            normal_files.append((dicom_file, None))
    
    logger.info(f"患者 {patient_dir.name}: 找到 {len(dicom_files)} 個DICOM文件, {len(xml_files)} 個XML文件")
    logger.info(f"患者 {patient_dir.name}: {len(lesion_files)} 病灶文件, {len(normal_files)} 正常文件 (包含 {len(dicom_files) - len(matched_dicom_files)} 個無標註文件)")
    
    # 平衡數據
    total_lesion = len(lesion_files)
    total_normal = len(normal_files)
    
    logger.info(f"患者 {patient_dir.name}: {total_lesion} 病灶文件, {total_normal} 正常文件")
    
    # 如果病灶文件過多，進行下采樣
    if total_lesion > 0 and total_normal > 0:
        # 計算理想的病灶比例
        if total_lesion / (total_lesion + total_normal) > max_lesion_ratio:
            # 病灶比例過高，減少病灶文件
            target_lesion_count = int(total_normal * max_lesion_ratio / (1 - max_lesion_ratio))
            target_lesion_count = min(target_lesion_count, total_lesion)
            
            # 隨機選擇病灶文件
            random.shuffle(lesion_files)
            selected_lesion_files = lesion_files[:target_lesion_count]
        else:
            selected_lesion_files = lesion_files
        
        # 如果正常文件過多，進行下采樣
        selected_normal_files = normal_files
        if total_normal > len(selected_lesion_files) * 2:  # 正常文件不超過病灶文件的2倍
            target_normal_count = len(selected_lesion_files) * 2
            random.shuffle(normal_files)
            selected_normal_files = normal_files[:target_normal_count]
    else:
        # 如果只有一種類型的文件，直接使用
        selected_lesion_files = lesion_files
        selected_normal_files = normal_files
    
    # 合併選中的文件
    selected_pairs = selected_lesion_files + selected_normal_files
    
    # 限制每個患者的最大文件數
    if len(selected_pairs) > max_files_per_patient:
        random.shuffle(selected_pairs)
        selected_pairs = selected_pairs[:max_files_per_patient]
    
    # 分離DICOM和XML文件
    selected_dicom = [pair[0] for pair in selected_pairs]
    selected_xml = [pair[1] for pair in selected_pairs]
    
    final_lesion_count = sum(1 for _, xml_file in selected_pairs 
                           if has_lesion_annotations(xml_file))
    final_normal_count = len(selected_pairs) - final_lesion_count
    
    logger.info(f"患者 {patient_dir.name} 平衡後: 選擇了 {len(selected_dicom)} 個文件 ({final_lesion_count} 病灶, {final_normal_count} 正常)")
    
    return selected_dicom, selected_xml


def has_lesion_annotations(xml_file: Path) -> bool:
    """
    檢查XML文件是否包含病灶標註
    
    Args:
        xml_file: XML文件路徑
    
    Returns:
        是否包含病灶標註
    """
    if xml_file is None:
        return False
        
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        objects = root.findall('.//object')
        return len(objects) > 0
    except:
        return False


def create_balanced_flat_dataset(split_dataset_dir: str, flat_dataset_dir: str, config: dict = None) -> Tuple[str, str, str, str]:
    """
    創建平衡的扁平化數據集
    
    Args:
        split_dataset_dir: 原始分層數據集目錄
        flat_dataset_dir: 目標扁平化數據集目錄
        config: 配置參數
    
    Returns:
        (train_data_dir, train_xml_dir, val_data_dir, val_xml_dir): 訓練和驗證數據目錄路徑
    """
    logger.info("創建平衡的扁平化數據集...")
    
    # 從配置中獲取參數
    if config is None:
        config = {}
    
    max_lesion_ratio = config.get('max_lesion_ratio', 0.7)
    max_files_per_patient = config.get('max_files_per_patient', 50)
    train_val_split_ratio = config.get('train_val_split_ratio', 0.8)
    
    logger.info(f"數據平衡配置: 最大病灶比例={max_lesion_ratio}, 每患者最大文件數={max_files_per_patient}")
    
    # 按病例分割患者
    train_patients, val_patients = split_patients_by_case(split_dataset_dir, train_val_split_ratio)
    
    # 創建目錄結構
    flat_path = Path(flat_dataset_dir)
    train_data_dir = flat_path / 'train_data'
    train_xml_dir = flat_path / 'train_xml'
    val_data_dir = flat_path / 'val_data'
    val_xml_dir = flat_path / 'val_xml'
    
    for dir_path in [train_data_dir, train_xml_dir, val_data_dir, val_xml_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # 處理訓練集
    logger.info("處理訓練集數據...")
    train_lesion_count = 0
    train_normal_count = 0
    
    for i, patient_id in enumerate(train_patients):
        patient_dir = Path(split_dataset_dir) / 'train' / patient_id
        
        # 平衡患者數據
        selected_dicom, selected_xml = balance_patient_data(patient_dir, max_lesion_ratio, max_files_per_patient)
        
        # 複製文件到扁平結構
        for j, (dicom_file, xml_file) in enumerate(zip(selected_dicom, selected_xml)):
            new_name = f"{patient_id}_{j:03d}"
            
            target_dicom = train_data_dir / f"{new_name}.dcm"
            target_xml = train_xml_dir / f"{new_name}.xml"
            
            try:
                # 複製DICOM文件
                copy_file_with_fallback(dicom_file, target_dicom)
                
                # 處理XML文件
                if xml_file is not None:
                    # 有對應的XML文件
                    copy_file_with_fallback(xml_file, target_xml)
                    
                    # 統計病灶情況
                    if has_lesion_annotations(xml_file):
                        train_lesion_count += 1
                    else:
                        train_normal_count += 1
                else:
                    # 沒有對應的XML文件，創建空的XML文件
                    create_empty_xml(target_xml)
                    train_normal_count += 1
                    
            except Exception as e:
                logger.warning(f"複製文件失敗 {dicom_file}: {e}")
        
        if (i + 1) % 20 == 0:
            logger.info(f"已處理 {i + 1}/{len(train_patients)} 個訓練患者")
    
    # 處理驗證集
    logger.info("處理驗證集數據...")
    val_lesion_count = 0
    val_normal_count = 0
    
    for i, patient_id in enumerate(val_patients):
        patient_dir = Path(split_dataset_dir) / 'train' / patient_id
        
        # 平衡患者數據
        selected_dicom, selected_xml = balance_patient_data(patient_dir, max_lesion_ratio, max_files_per_patient)
        
        # 複製文件到扁平結構
        for j, (dicom_file, xml_file) in enumerate(zip(selected_dicom, selected_xml)):
            new_name = f"{patient_id}_{j:03d}"
            
            target_dicom = val_data_dir / f"{new_name}.dcm"
            target_xml = val_xml_dir / f"{new_name}.xml"
            
            try:
                # 複製DICOM文件
                copy_file_with_fallback(dicom_file, target_dicom)
                
                # 處理XML文件
                if xml_file is not None:
                    # 有對應的XML文件
                    copy_file_with_fallback(xml_file, target_xml)
                    
                    # 統計病灶情況
                    if has_lesion_annotations(xml_file):
                        val_lesion_count += 1
                    else:
                        val_normal_count += 1
                else:
                    # 沒有對應的XML文件，創建空的XML文件
                    create_empty_xml(target_xml)
                    val_normal_count += 1
                    
            except Exception as e:
                logger.warning(f"複製文件失敗 {dicom_file}: {e}")
        
        if (i + 1) % 20 == 0:
            logger.info(f"已處理 {i + 1}/{len(val_patients)} 個驗證患者")
    
    logger.info(f"扁平化數據集創建完成:")
    logger.info(f"訓練集: {train_lesion_count} 病灶文件, {train_normal_count} 正常文件")
    logger.info(f"驗證集: {val_lesion_count} 病灶文件, {val_normal_count} 正常文件")
    
    return str(train_data_dir), str(train_xml_dir), str(val_data_dir), str(val_xml_dir)


def _train_with_metrics(trainer, train_loader, val_loader, train_metrics, val_metrics, config):
    """
    包含詳細評估指標的訓練流程
    
    Args:
        trainer: UNetPPTrainer 實例
        train_loader: 訓練數據載入器
        val_loader: 驗證數據載入器
        train_metrics: 訓練評估指標
        val_metrics: 驗證評估指標
        config: 訓練配置
    """
    import torch.optim as optim
    from train_unetpp import CombinedLoss
    
    # 設置優化器和損失函數
    optimizer = optim.AdamW(
        trainer.model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config.get('weight_decay', 1e-5)
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    criterion = CombinedLoss(
        seg_weight=1.0,
        det_weight=1.0,
        classification_weight=1.0,
        use_focal_loss=True
    )
    
    # 恢復訓練
    if config.get('resume_from'):
        trainer.load_checkpoint(config['resume_from'], optimizer, scheduler)
    
    logger.info(f"開始增強訓練，共 {config['num_epochs']} 個 epoch")
    logger.info(f"設備: {trainer.device}")
    logger.info(f"模型參數數量: {sum(p.numel() for p in trainer.model.parameters()):,}")
    
    for epoch in range(trainer.current_epoch, config['num_epochs']):
        trainer.current_epoch = epoch
        
        # 重置指標
        train_metrics.reset()
        val_metrics.reset()
        
        # 訓練階段
        train_losses = trainer.train_epoch(train_loader, optimizer, criterion)
        
        # 驗證階段
        val_losses = trainer.validate_epoch(val_loader, criterion)
        
        # 計算詳細評估指標（在驗證階段）
        try:
            logger.info("開始計算詳細評估指標...")
            _calculate_detailed_metrics(trainer, val_loader, val_metrics)
            logger.info("評估指標計算完成")
        except Exception as e:
            logger.warning(f"無法計算詳細評估指標: {e}")
            import traceback
            logger.debug(f"詳細錯誤: {traceback.format_exc()}")
            # 繼續執行，不讓指標計算失敗影響訓練
        
        # 更新學習率
        scheduler.step(val_losses['total_loss'])
        
        # 獲取評估指標結果
        try:
            val_metrics_results = val_metrics.compute_metrics()
        except Exception as e:
            logger.warning(f"計算評估指標時出錯: {e}")
            val_metrics_results = {
                'mean_dice': 0.0,
                'mean_iou': 0.0,
                'pixel_accuracy': 0.0,
                'class_1_recall': 0.0,
                'class_0_recall': 0.0,
                'class_1_precision': 0.0,
                'class_1_f1_score': 0.0
            }
        
        # 記錄歷史
        epoch_history = {
            'epoch': epoch,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_metrics': val_metrics_results,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        trainer.training_history.append(epoch_history)
        
        # TensorBoard 記錄
        for key, value in train_losses.items():
            trainer.writer.add_scalar(f'Train/{key}', value, epoch)
        
        for key, value in val_losses.items():
            trainer.writer.add_scalar(f'Validation/{key}', value, epoch)
            
        # 記錄評估指標
        for key, value in val_metrics_results.items():
            if isinstance(value, (int, float)):
                trainer.writer.add_scalar(f'Metrics/{key}', value, epoch)
        
        trainer.writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # 檢查是否是最佳模型
        is_best = val_losses['total_loss'] < trainer.best_loss
        if is_best:
            trainer.best_loss = val_losses['total_loss']
        
        # 保存檢查點
        trainer.save_checkpoint(optimizer, scheduler, val_losses, is_best)
        
        # 詳細日誌輸出
        logger.info(
            f"Epoch {epoch}: Train Loss: {train_losses['total_loss']:.4f}, "
            f"Val Loss: {val_losses['total_loss']:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}"
        )
        
        # 輸出關鍵評估指標
        if 'mean_dice' in val_metrics_results:
            logger.info(f"  Dice Score: {val_metrics_results['mean_dice']:.4f}")
        if 'mean_iou' in val_metrics_results:
            logger.info(f"  IoU Score: {val_metrics_results['mean_iou']:.4f}")
        if 'pixel_accuracy' in val_metrics_results:
            logger.info(f"  Pixel Accuracy: {val_metrics_results['pixel_accuracy']:.4f}")
        if 'class_1_recall' in val_metrics_results:
            logger.info(f"  Sensitivity (Lesion): {val_metrics_results['class_1_recall']:.4f}")
        if 'class_0_recall' in val_metrics_results:
            logger.info(f"  Specificity (Background): {val_metrics_results['class_0_recall']:.4f}")
        if 'class_1_precision' in val_metrics_results:
            logger.info(f"  Precision (Lesion): {val_metrics_results['class_1_precision']:.4f}")
        if 'class_1_f1_score' in val_metrics_results:
            logger.info(f"  F1-Score (Lesion): {val_metrics_results['class_1_f1_score']:.4f}")
    
    trainer.writer.close()
    logger.info("增強訓練完成！")


def _calculate_detailed_metrics(trainer, val_loader, metrics):
    """
    計算詳細的評估指標
    
    Args:
        trainer: 訓練器實例
        val_loader: 驗證數據載入器
        metrics: 評估指標實例
    """
    trainer.model.eval()
    
    batch_count = 0
    processed_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # 調試第一個批次的結構
            if batch_idx == 0:
                logger.info(f"DEBUG: batch type: {type(batch)}")
                if isinstance(batch, dict):
                    logger.info(f"DEBUG: batch keys: {list(batch.keys())}")
                    for key, value in batch.items():
                        logger.info(f"DEBUG: batch['{key}'] type: {type(value)}")
                        if isinstance(value, torch.Tensor):
                            logger.info(f"DEBUG: batch['{key}'] shape: {value.shape}")
                        elif isinstance(value, (list, tuple)):
                            logger.info(f"DEBUG: batch['{key}'] length: {len(value)}")
                            if len(value) > 0:
                                logger.info(f"DEBUG: batch['{key}'][0] type: {type(value[0])}")
                                if isinstance(value[0], torch.Tensor):
                                    logger.info(f"DEBUG: batch['{key}'][0] shape: {value[0].shape}")
                elif isinstance(batch, (list, tuple)):
                    logger.info(f"DEBUG: batch length: {len(batch)}")
                    for i, item in enumerate(batch):
                        logger.info(f"DEBUG: batch[{i}] type: {type(item)}")
                        if isinstance(item, torch.Tensor):
                            logger.info(f"DEBUG: batch[{i}] shape: {item.shape}")
                        elif isinstance(item, (list, tuple)):
                            logger.info(f"DEBUG: batch[{i}] length: {len(item)}")
            
            # 嘗試不同的批次格式
            images = None
            targets = None
            
            try:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    images, targets = batch
                elif isinstance(batch, dict):
                    # 嘗試多種可能的鍵名來提取 images
                    for img_key in ['images', 'image', 'input', 'data']:
                        if img_key in batch:
                            images = batch[img_key]
                            if batch_idx == 0:
                                logger.info(f"DEBUG: 使用圖像鍵 '{img_key}'")
                            break
                    
                    # 嘗試多種可能的鍵名來提取 targets
                    for target_key in ['masks', 'segmentation', 'targets', 'target', 'labels', 'mask']:
                        if target_key in batch:
                            targets = batch[target_key]
                            if batch_idx == 0:
                                logger.info(f"DEBUG: 使用目標鍵 '{target_key}'")
                            break
                else:
                    if batch_idx == 0:
                        logger.warning(f"未知的批次格式: {type(batch)}")
                    continue
                
                # 檢查是否成功提取了 images 和 targets (避免直接布尔判断Tensor)
                images_valid = images is not None
                targets_valid = targets is not None
                
                # 如果是tensor，检查是否为空
                if isinstance(images, torch.Tensor) and images.numel() == 0:
                    images_valid = False
                if isinstance(targets, torch.Tensor) and targets.numel() == 0:
                    targets_valid = False
                
                if not images_valid or not targets_valid:
                    if batch_idx == 0:
                        logger.warning(f"無法提取 images 或 targets (images_valid: {images_valid}, targets_valid: {targets_valid})")
                        if images is not None:
                            logger.info(f"DEBUG: images type: {type(images)}")
                            if isinstance(images, torch.Tensor):
                                logger.info(f"DEBUG: images shape: {images.shape}")
                        if targets is not None:
                            logger.info(f"DEBUG: targets type: {type(targets)}")
                            if isinstance(targets, torch.Tensor):
                                logger.info(f"DEBUG: targets shape: {targets.shape}")
                    continue
                
                # 檢查張量是否有效
                if not hasattr(images, 'shape'):
                    if batch_idx == 0:
                        logger.warning(f"images 沒有 shape 屬性，type: {type(images)}")
                    continue
                
                # 檢查 targets 的有效性
                targets_has_len = hasattr(targets, '__len__')
                targets_has_shape = hasattr(targets, 'shape')
                
                if not (targets_has_len or targets_has_shape):
                    if batch_idx == 0:
                        logger.warning(f"targets 格式無效，type: {type(targets)}")
                    continue
                
            except Exception as e:
                if batch_idx == 0:
                    logger.warning(f"解析批次失敗: {e}")
                    import traceback
                    logger.debug(f"詳細錯誤: {traceback.format_exc()}")
                continue
            
            
            images = images.to(trainer.device)
            
            # 模型預測
            predictions = trainer.model(images)
            
            # 調試第一個批次的預測結構
            if batch_idx == 0:
                logger.info(f"DEBUG: predictions type: {type(predictions)}")
                if isinstance(predictions, dict):
                    logger.info(f"DEBUG: predictions keys: {list(predictions.keys())}")
                    for key, value in predictions.items():
                        if isinstance(value, torch.Tensor):
                            logger.info(f"DEBUG: predictions['{key}'] shape: {value.shape}")
                        elif isinstance(value, list):
                            logger.info(f"DEBUG: predictions['{key}'] list length: {len(value)}")
                            if len(value) > 0 and isinstance(value[0], torch.Tensor):
                                logger.info(f"DEBUG: predictions['{key}'][0] shape: {value[0].shape}")
                elif isinstance(predictions, torch.Tensor):
                    logger.info(f"DEBUG: predictions is tensor with shape: {predictions.shape}")
                elif isinstance(predictions, (list, tuple)):
                    logger.info(f"DEBUG: predictions is {type(predictions)} with length: {len(predictions)}")
                    if len(predictions) > 0 and isinstance(predictions[0], torch.Tensor):
                        logger.info(f"DEBUG: predictions[0] shape: {predictions[0].shape}")
            
            # 提取分割預測
            seg_preds = None
            if isinstance(predictions, dict):
                # 檢查可能的鍵名
                for key in ['segmentation', 'seg_outputs', 'outputs', 'seg']:
                    if key in predictions:
                        seg_preds = predictions[key]
                        if batch_idx == 0:
                            logger.info(f"DEBUG: 使用預測鍵 '{key}'")
                        break
            elif isinstance(predictions, torch.Tensor):
                seg_preds = predictions
                if batch_idx == 0:
                    logger.info("DEBUG: 使用tensor預測")
            elif isinstance(predictions, (list, tuple)) and len(predictions) > 0:
                seg_preds = predictions[0]  # 使用第一個輸出作為分割結果
                if batch_idx == 0:
                    logger.info("DEBUG: 使用list/tuple的第一個元素")
            
            if seg_preds is None:
                if batch_idx == 0:
                    logger.warning(f"無法提取分割預測")
                continue
            
            # 如果是列表（深度監督），使用最後一個輸出
            if isinstance(seg_preds, list):
                seg_preds = seg_preds[-1]
                if batch_idx == 0:
                    logger.info(f"DEBUG: 使用深度監督的最後一個輸出，shape: {seg_preds.shape}")
            
            try:
                # 收集分割目標
                if isinstance(targets, list):
                    # 檢查列表中是否包含字典格式的目標
                    if len(targets) > 0 and isinstance(targets[0], dict):
                        # 嘗試提取分割目標
                        seg_targets_list = []
                        for target in targets:
                            if 'segmentation' in target:
                                seg_targets_list.append(target['segmentation'])
                            elif 'mask' in target:
                                seg_targets_list.append(target['mask'])
                            elif 'masks' in target:
                                seg_targets_list.append(target['masks'])
                            else:
                                # 如果沒有找到分割目標，使用整個字典的值（假設只有一個值）
                                target_values = list(target.values())
                                if len(target_values) > 0:
                                    seg_targets_list.append(target_values[0])
                        
                        if seg_targets_list:
                            seg_targets = torch.stack(seg_targets_list).to(trainer.device)
                        else:
                            if batch_idx == 0:
                                logger.warning("無法從列表目標中提取分割標註")
                            continue
                    else:
                        # 假設是張量列表
                        seg_targets = torch.stack(targets).to(trainer.device)
                else:
                    # 直接是張量
                    seg_targets = targets.to(trainer.device)
                
                if batch_idx == 0:
                    logger.info(f"DEBUG: seg_targets shape: {seg_targets.shape}")
                    logger.info(f"DEBUG: seg_preds shape: {seg_preds.shape}")
                
                # 確保預測結果是概率分布
                if seg_preds.dim() == 4 and seg_preds.size(1) > 1:
                    # 多類別分割，使用 softmax
                    seg_preds = torch.softmax(seg_preds, dim=1)
                elif seg_preds.dim() == 4 and seg_preds.size(1) == 1:
                    # 二元分割，使用 sigmoid
                    seg_preds = torch.sigmoid(seg_preds)
                
                # 調整目標張量的尺寸以匹配預測
                if seg_targets.dim() == 3:
                    seg_targets = seg_targets.long()
                
                # 如果尺寸不匹配，調整目標尺寸
                if seg_preds.shape[-2:] != seg_targets.shape[-2:]:
                    import torch.nn.functional as F
                    seg_targets = F.interpolate(
                        seg_targets.float().unsqueeze(1), 
                        size=seg_preds.shape[-2:], 
                        mode='nearest'
                    ).squeeze(1).long()
                
                # 更新評估指標
                metrics.update(seg_preds, seg_targets)
                
                batch_count += 1
                # 安全計算樣本數，避免對Tensor做布爾判斷
                if isinstance(targets, list):
                    processed_samples += len(targets)
                elif isinstance(targets, torch.Tensor):
                    processed_samples += targets.size(0)
                else:
                    processed_samples += 1  # 假設單個樣本
                
                if batch_idx == 0:
                    logger.info(f"DEBUG: 成功處理批次 {batch_idx}")
                
            except Exception as e:
                if batch_idx < 5:  # 只記錄前幾個批次的錯誤
                    logger.warning(f"處理批次 {batch_idx} 時出錯: {e}")
                    if batch_idx == 0:
                        import traceback
                        logger.debug(f"詳細錯誤: {traceback.format_exc()}")
                continue
            
            # 只處理前幾個批次進行測試
            if batch_idx >= 10:  # 只處理前10個批次以節省時間
                break
    
    logger.info(f"評估指標計算完成：處理了 {batch_count} 個批次，{processed_samples} 個樣本")


def run_training_with_hierarchical_data(split_dataset_dir: str, flat_dataset_dir: str = None, config: dict = None):
    """
    使用層次化數據結構運行訓練
    
    Args:
        split_dataset_dir: 分割數據集目錄路徑
        flat_dataset_dir: 扁平化數據集目錄路徑
        config: 訓練配置
    """
    
    # 使用預設配置並更新用戶配置
    final_config = DEFAULT_CONFIG.copy()
    if config:
        final_config.update(config)
    
    # 設置設備
    if 'device' not in final_config:
        import torch as torch_module
        final_config['device'] = 'cuda' if torch_module.cuda.is_available() else 'cpu'
    
    # 設置扁平化數據集目錄
    if flat_dataset_dir is None:
        flat_dataset_dir = str(Path(split_dataset_dir).parent / 'flat_dataset')
    
    logger.info(f"使用原始數據集: {split_dataset_dir}")
    logger.info(f"使用扁平化數據集: {flat_dataset_dir}")
    
    try:
        # 檢查並創建扁平化數據集
        train_data_dir, train_xml_dir, val_data_dir, val_xml_dir = check_and_create_flat_dataset(
            split_dataset_dir, flat_dataset_dir, final_config
        )
        
        # 檢查數據
        train_dicom_files = list(Path(train_data_dir).glob("*.dcm"))
        train_xml_files = list(Path(train_xml_dir).glob("*.xml"))
        val_dicom_files = list(Path(val_data_dir).glob("*.dcm"))
        val_xml_files = list(Path(val_xml_dir).glob("*.xml"))
        
        logger.info(f"最終數據統計:")
        logger.info(f"訓練數據: {len(train_dicom_files)} DICOM, {len(train_xml_files)} XML")
        logger.info(f"驗證數據: {len(val_dicom_files)} DICOM, {len(val_xml_files)} XML")
        
        if len(train_dicom_files) == 0 or len(train_xml_files) == 0:
            raise ValueError("沒有找到訓練數據文件")
        
        if len(val_dicom_files) == 0 or len(val_xml_files) == 0:
            raise ValueError("沒有找到驗證數據文件")
        
        # 使用 train_unetpp.py 中的訓練函數
        logger.info("開始 UNet++ 訓練...")
        
        # 創建臨時的統一配置，為每個目錄單獨訓練
        # 由於原始函數期望單一目錄，我們需要直接使用 UNetPPTrainer
        from unetpp_model import UNetPPDetector
        from unetpp_dataset import UNetPPDetectionDataset, collate_fn
        from train_unetpp import UNetPPTrainer
        from segmentation_metrics import SegmentationMetrics
        from torch.utils.data import DataLoader
        
        # 設置設備
        device = torch.device(final_config['device'])
        logger.info(f"使用設備: {device}")
        
        # 創建數據變換
        try:
            import albumentations as A
            from albumentations.pytorch import ToTensorV2
            
            train_transform = A.Compose([
                A.Resize(height=512, width=512),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.OneOf([
                    A.GaussNoise(noise_scale_factor=0.1),
                    A.GaussianBlur(blur_limit=(1, 3)),
                    A.MotionBlur(blur_limit=(3, 7)),
                ], p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.Normalize(mean=[0.485], std=[0.229]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
            
            val_transform = A.Compose([
                A.Resize(height=512, width=512),
                A.Normalize(mean=[0.485], std=[0.229]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
            
        except ImportError:
            logger.error("albumentations 未安裝，請安裝: pip install albumentations")
            raise
        
        # 創建數據集
        logger.info("創建訓練數據集...")
        train_dataset = UNetPPDetectionDataset(
            data_dir=train_data_dir,
            xml_dir=train_xml_dir,
            transform=train_transform,
            multi_class_segmentation=final_config.get('multi_class_segmentation', False)
        )
        
        logger.info("創建驗證數據集...")
        val_dataset = UNetPPDetectionDataset(
            data_dir=val_data_dir,
            xml_dir=val_xml_dir,
            transform=val_transform,
            multi_class_segmentation=final_config.get('multi_class_segmentation', False)
        )
        
        # 創建數據載入器
        train_loader = DataLoader(
            train_dataset,
            batch_size=final_config['batch_size'],
            shuffle=True,
            num_workers=final_config.get('num_workers', 2),
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=final_config['batch_size'],
            shuffle=False,
            num_workers=final_config.get('num_workers', 2),
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        logger.info(f"訓練樣本: {len(train_dataset)}")
        logger.info(f"驗證樣本: {len(val_dataset)}")
        
        # 分析數據平衡情況
        train_lesion_samples = sum(1 for item in train_dataset.data_list if has_lesion_annotations(Path(item['xml_path'])))
        train_normal_samples = len(train_dataset) - train_lesion_samples
        val_lesion_samples = sum(1 for item in val_dataset.data_list if has_lesion_annotations(Path(item['xml_path'])))
        val_normal_samples = len(val_dataset) - val_lesion_samples
        
        logger.info(f"訓練集分布: {train_lesion_samples} 病灶樣本, {train_normal_samples} 正常樣本")
        logger.info(f"驗證集分布: {val_lesion_samples} 病灶樣本, {val_normal_samples} 正常樣本")
        logger.info(f"訓練集病灶比例: {train_lesion_samples/len(train_dataset)*100:.1f}%")
        logger.info(f"驗證集病灶比例: {val_lesion_samples/len(val_dataset)*100:.1f}%")
        
        # 創建模型
        logger.info("創建模型...")
        model = UNetPPDetector(
            in_channels=1,
            num_classes=2,  # 背景 + 病灶
            segmentation_classes=2 if not final_config.get('multi_class_segmentation', False) else 5,
            feature_scale=4
        )
        
        # 創建訓練器
        trainer = UNetPPTrainer(
            model=model,
            device=device,
            save_dir=final_config.get('save_dir', './checkpoints'),
            log_dir=final_config.get('log_dir', './logs')
        )
        
        # 創建評估指標
        train_metrics = SegmentationMetrics(num_classes=2, ignore_background=True)
        val_metrics = SegmentationMetrics(num_classes=2, ignore_background=True)
        
        # 開始增強的訓練流程
        _train_with_metrics(
            trainer=trainer,
            train_loader=train_loader,
            val_loader=val_loader,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            config=final_config
        )
        
    except Exception as e:
        logger.error(f"訓練失敗: {e}")
        raise


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='UNet++ Training with Balanced Patient-Based Data Split')
    parser.add_argument(
        '--split_dataset_dir', 
        type=str, 
        default='../../datasets/splited_dataset',
        help='原始分層數據集目錄路徑 (默認: ../../datasets/splited_dataset)'
    )
    parser.add_argument(
        '--flat_dataset_dir',
        type=str,
        default=None,
        help='扁平化數據集目錄路徑 (默認: 與原始數據集同級的 flat_dataset 目錄)'
    )
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=10, help='訓練輪數')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='學習率')
    parser.add_argument('--device', type=str, default=None, help='設備 (cuda/cpu)')
    parser.add_argument('--resume_from', type=str, help='恢復訓練的檢查點路徑')
    parser.add_argument('--train_val_split_ratio', type=float, default=0.8, 
                       help='訓練集比例 (默認: 0.8，即80%%訓練20%%驗證)')
    parser.add_argument('--max_lesion_ratio', type=float, default=0.7,
                       help='每個患者的最大病灶比例 (默認: 0.7)')
    parser.add_argument('--max_files_per_patient', type=int, default=50,
                       help='每個患者的最大文件數 (默認: 50)')
    
    args = parser.parse_args()
    
    # 構建配置
    config = {
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'resume_from': args.resume_from,
        'train_val_split_ratio': args.train_val_split_ratio,
        'max_lesion_ratio': args.max_lesion_ratio,
        'max_files_per_patient': args.max_files_per_patient
    }
    
    if args.device:
        config['device'] = args.device
    
    # 檢查數據集目錄
    split_dataset_dir = Path(args.split_dataset_dir)
    if not split_dataset_dir.exists():
        logger.error(f"數據集目錄不存在: {split_dataset_dir}")
        return
    
    train_dir = split_dataset_dir / 'train'
    if not train_dir.exists():
        logger.error(f"訓練數據目錄不存在: {train_dir}")
        return
    
    # 設置扁平化數據集目錄
    if args.flat_dataset_dir:
        flat_dataset_dir = args.flat_dataset_dir
    else:
        flat_dataset_dir = str(split_dataset_dir.parent / 'flat_dataset')
    
    logger.info(f"使用原始數據集: {split_dataset_dir}")
    logger.info(f"使用扁平化數據集: {flat_dataset_dir}")
    
    # 運行訓練
    run_training_with_hierarchical_data(
        split_dataset_dir=str(split_dataset_dir),
        flat_dataset_dir=flat_dataset_dir,
        config=config
    )


if __name__ == "__main__":
    main()