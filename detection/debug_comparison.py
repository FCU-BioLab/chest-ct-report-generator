#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
調試腳本：比較兩個測試腳本的差異
用於找出為什麼同樣的模型在不同測試腳本中表現差異很大
"""

import os
import sys
import logging
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# 添加detection目錄到路徑
detection_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(detection_dir)

from faster_rcnn_detection.faster_rcnn_dataset import CTDetectionDataset

def setup_logging():
    """設置日誌"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def create_model(device):
    """創建模型架構"""
    model = fasterrcnn_resnet50_fpn(weights=None)
    num_classes = 2  # 背景 + 病灶
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)
    model.eval()
    return model

def load_model_weights(model, model_path, device):
    """載入模型權重"""
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logging.info(f"成功載入檢查點，包含的鍵: {list(checkpoint.keys())}")
            if 'best_f1' in checkpoint:
                logging.info(f"訓練時最佳F1分數: {checkpoint['best_f1']:.4f}")
        else:
            model.load_state_dict(checkpoint)
            logging.info("成功載入模型權重（直接格式）")
        return True
    except Exception as e:
        logging.error(f"載入模型失敗: {e}")
        return False

def create_dataset_original_style(data_dir, split='test'):
    """使用原始腳本的方式創建數據集"""
    logging.info("=== 創建數據集（原始方式）===")
    
    # 模擬 test_detection.py 中的 create_test_dataset 函數
    dataset = CTDetectionDataset(
        data_root=data_dir,
        split=split,
        target_size=512,
        specific_patients=None,
        transforms=transforms.Compose([transforms.ToTensor()]),
        include_negative_samples=True,  # 原始腳本默認包含負樣本
        max_negative_per_patient=0      # 原始腳本默認無限制
    )
    
    logging.info(f"原始方式數據集大小: {len(dataset)}")
    return dataset

def create_dataset_refactored_style(data_dir, split='test'):
    """使用重構腳本的方式創建數據集"""
    logging.info("=== 創建數據集（重構方式）===")
    
    # 模擬 test_detection_refactored.py 中的數據集創建
    dataset = CTDetectionDataset(
        data_root=data_dir,
        split=split,
        target_size=512,
        transforms=transforms.Compose([transforms.ToTensor()])
        # 注意：重構版本缺少負樣本相關參數！
    )
    
    logging.info(f"重構方式數據集大小: {len(dataset)}")
    return dataset

def collate_fn_original(batch):
    """原始腳本的collate函數"""
    images = []
    targets = []
    
    for item in batch:
        if isinstance(item, dict) and 'image' in item and 'target' in item:
            images.append(item['image'])
            targets.append(item['target'])
        else:
            logging.warning(f"Unexpected batch item format: {type(item)}")
    
    return images, targets

def simple_evaluation(model, dataloader, device, dataset_name):
    """簡單的模型評估"""
    model.eval()
    total_predictions = 0
    total_targets = 0
    confidence_scores = []
    
    logging.info(f"=== 評估 {dataset_name} ===")
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            if batch_idx >= 5:  # 只測試前5個批次
                break
                
            images = [img.to(device) for img in images]
            predictions = model(images)
            
            for pred, target in zip(predictions, targets):
                total_predictions += len(pred['boxes'])
                total_targets += len(target['boxes'])
                confidence_scores.extend(pred['scores'].cpu().tolist())
        
        logging.info(f"  總預測數: {total_predictions}")
        logging.info(f"  總目標數: {total_targets}")
        logging.info(f"  平均置信度: {sum(confidence_scores)/len(confidence_scores):.4f}")
        logging.info(f"  置信度範圍: {min(confidence_scores):.4f} - {max(confidence_scores):.4f}")
        logging.info(f"  置信度>0.5的預測: {sum(1 for s in confidence_scores if s > 0.5)}")

def main():
    setup_logging()
    
    # 配置路徑
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'Simple_Training_Refactored_20250905_142020/models/best_model.pth')
    data_dir = os.path.join(os.path.dirname(script_dir), 'datasets', 'splited_dataset')
    
    logging.info("=== 開始調試分析 ===")
    logging.info(f"模型路徑: {model_path}")
    logging.info(f"數據路徑: {data_dir}")
    
    # 檢查文件存在性
    if not os.path.exists(model_path):
        logging.error(f"模型文件不存在: {model_path}")
        return
    
    if not os.path.exists(data_dir):
        logging.error(f"數據目錄不存在: {data_dir}")
        return
    
    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"使用設備: {device}")
    
    # 創建並載入模型
    model = create_model(device)
    if not load_model_weights(model, model_path, device):
        return
    
    # 創建兩種方式的數據集
    try:
        dataset_original = create_dataset_original_style(data_dir, 'test')
        dataset_refactored = create_dataset_refactored_style(data_dir, 'test')
    except Exception as e:
        logging.error(f"創建數據集失敗: {e}")
        return
    
    # 比較數據集
    logging.info(f"\n=== 數據集比較 ===")
    logging.info(f"原始方式數據集大小: {len(dataset_original)}")
    logging.info(f"重構方式數據集大小: {len(dataset_refactored)}")
    
    if len(dataset_original) != len(dataset_refactored):
        logging.warning("⚠️  數據集大小不一致！這可能是性能差異的主要原因")
    
    # 檢查數據集內容的差異
    if len(dataset_original) > 0 and len(dataset_refactored) > 0:
        try:
            sample_orig = dataset_original[0]
            sample_ref = dataset_refactored[0]
            
            logging.info(f"原始數據集第一個樣本目標數: {len(sample_orig['target']['boxes'])}")
            logging.info(f"重構數據集第一個樣本目標數: {len(sample_ref['target']['boxes'])}")
        except Exception as e:
            logging.error(f"檢查數據集樣本失敗: {e}")
    
    # 創建數據加載器
    loader_original = DataLoader(
        dataset_original, batch_size=4, shuffle=False, 
        collate_fn=collate_fn_original, num_workers=0
    )
    
    loader_refactored = DataLoader(
        dataset_refactored, batch_size=4, shuffle=False, 
        collate_fn=collate_fn_original, num_workers=0
    )
    
    # 簡單評估
    simple_evaluation(model, loader_original, device, "原始數據集")
    simple_evaluation(model, loader_refactored, device, "重構數據集")
    
    logging.info("\n=== 調試分析完成 ===")
    logging.info("如果數據集大小或預測結果有顯著差異，這就是性能差異的原因")

if __name__ == "__main__":
    main()
