"""
整合 EAPIoU Loss 和 Deep Supervision 的 YOLO11 訓練腳本

關鍵功能：
1. ✅ 使用 EAPIoU 替代預設 CIoU
2. ✅ 支援 Deep Supervision (可選)
3. ✅ 保留原有的資料處理和樣本平衡邏輯
4. ✅ 相容 Ultralytics 訓練框架

使用方法：
python train_with_eapiou.py --model models/yolo11_sse_eapiou_s.yaml --epochs 250 --batch_size 16
"""

import os
import sys
import argparse
import json
import logging
import time
from pathlib import Path
from datetime import datetime
import torch
from ultralytics import YOLO

# 新增當前目錄到 sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 匯入自訂損失函數
try:
    from custom_loss_integration import create_custom_loss
    EAPIOU_AVAILABLE = True
except ImportError as e:
    logging.warning(f"⚠️ 無法匯入 custom_loss_integration: {e}")
    logging.warning("⚠️ 將使用預設的 CIoU 損失")
    EAPIOU_AVAILABLE = False

# 匯入原有的資料處理函數
try:
    from train_custom_yolo import (
        collect_patient_data, 
        create_yolo_dataset,
        split_patients,
        ensure_val_has_positive_samples,
        setup_logging,
        set_seed,
        select_device
    )
    logging.info("✅ 成功匯入原有資料處理模組")
except ImportError as e:
    print(f"❌ 錯誤：無法匯入 train_custom_yolo: {e}")
    print("請確保 train_custom_yolo.py 在同一目錄下")
    sys.exit(1)


def patch_model_with_eapiou(trainer):
    """
    訓練後鉤子：替換預設損失函數為 EAPIoU Loss
    
    Args:
        trainer: Ultralytics Trainer 實例
    """
    if not EAPIOU_AVAILABLE:
        logging.warning("⚠️ EAPIoU 不可用，跳過損失函數替換")
        return
    
    logging.info("\n" + "=" * 70)
    logging.info("🔧 正在替換損失函數：CIoU -> EAPIoU")
    logging.info("=" * 70)
    
    try:
        # 檢查模型是否已初始化
        if not hasattr(trainer, 'model') or trainer.model is None:
            logging.warning("⚠️ 模型尚未初始化，跳過損失函數替換")
            return
        
        # 🔧 修復：從模型物件讀取自訂參數（而非 trainer.args）
        eapiou_beta = getattr(trainer.model, '_custom_eapiou_beta', 0.15)
        deep_supervision = getattr(trainer.model, '_custom_deep_supervision', True)
        
        # 建立自訂損失函數
        custom_loss = create_custom_loss(
            trainer.model,
            tal_topk=10,
            eapiou_beta=eapiou_beta,
            deep_supervision=deep_supervision
        )
        
        # 替換 trainer 的損失函數
        trainer.loss = custom_loss
        
        logging.info(f"✅ 成功啟用 EAPIoU Loss (beta={eapiou_beta})")
        if deep_supervision:
            logging.info(f"✅ 已啟用 Deep Supervision")
        logging.info("=" * 70 + "\n")
        
    except Exception as e:
        logging.error(f"❌ 錯誤：損失函數替換失敗: {e}")
        logging.warning("⚠️ 將使用預設損失函數繼續訓練")
        import traceback
        logging.error(traceback.format_exc())


def train_yolo_with_eapiou(
    model_path: str,
    data_dir: str,
    output_dir: str = "yolo_runs/training",
    epochs: int = 300,  # 🛡️ 超保守版300 epochs（防止檢測失效）
    batch_size: int = 8,
    imgsz: int = 640,
    lr: float = 0.003,  # 🛡️ 極低學習率（防止過快收斂）
    lrf: float = 0.1,  # 🛡️ 更高最終學習率（保持學習能力）
    val_ratio: float = 0.15,
    max_negative_ratio: float = 0.1,  # 🛡️ 進一步增加負樣本（平衡類別）
    oversample_positive: float = 2.0,  # 🛡️ 降低過採樣（避免過擬合重複樣本）
    warmup_epochs: int = 40,  # 🛡️ 適中預熱
    optimizer: str = "AdamW",
    weight_decay: float = 0.0005,  # 🛡️ 標準正則化
    mosaic: float = 0.0,
    mixup: float = 0.0,
    copy_paste: float = 0.0,
    degrees: float = 3.0,
    translate: float = 0.02,
    scale: float = 0.1,
    fliplr: float = 0.5,
    flipud: float = 0.0,
    hsv_h: float = 0.0,
    hsv_s: float = 0.0,
    hsv_v: float = 0.3,
    patience: int = 75,  # 🛡️ 適中耐心值（快速止損）
    workers: int = 8,
    save_period: int = 5,  # 🛡️ 每5 epochs保存
    random_seed: int = 42,
    # EAPIoU 相關參數
    eapiou_beta: float = 0.15,  # 🛡️ 保持已驗證的beta值
    deep_supervision: bool = True,
    # 損失權重（平衡版）
    box: float = 7.5,
    cls: float = 0.8,  # 🛡️ 提高cls權重（防止cls_loss過低）
    dfl: float = 1.5,
    # 正則化參數
    dropout: float = 0.1,  # 🛡️ 降低dropout（避免欠擬合）
    label_smoothing: float = 0.0,  # 🛡️ 移除label smoothing（避免預測抑制）
    # NMS 和推理阈值
    iou: float = 0.5,
    conf: float = 0.1,  # 🛡️ 提高conf閾值（防止sigmoid飽和）
    # 多尺度訓練
    multi_scale: bool = False,
    # 學習率調度
    cos_lr: bool = True,
    close_mosaic: int = 0,  # 🎯 從不使用Mosaic
):
    """
    使用 EAPIoU Loss 訓練 YOLO11 模型（極致優化版 - 目標mAP@0.5≥0.8）
    
    🚀 極致改進策略（基於Epoch 86深度分析）：
    
    【核心問題診斷】
    1. Recall過低(0.305) → 69.5%漏檢率 ❌
    2. mAP長期停滯(0.389) → 55輪無提升 ❌
    3. F1 Score不足 → 精確率與召回率失衡 ❌
    
    【突破性優化方案】
    A. 數據層面（解決樣本不平衡）
       - 負樣本比例：0.15 → 0.05（減少67%，消除假陽性干擾）
       - 正樣本過採樣：4.0x → 8.0x（加倍，強化病灶學習）
       - Copy-Paste增強：0.15 → 0.3（翻倍，生成更多病灶變體）
    
    B. 訓練策略（避免過早收斂）
       - 初始學習率：0.005 → 0.01（激進跳出局部最優）
       - 最終學習率：0.05 → 0.1（保持後期學習能力）
       - 訓練輪次：150 → 300（充分收斂）
       - Early Stopping：25 → 50（給予更多機會）
    
    C. 損失權重（極致Recall優化）
       - cls損失：0.3 → 0.2（進一步降低33%）
       - box損失：7.5 → 8.5（提高定位重要性）
       - dfl損失：1.5 → 2.0（強化分布焦點）
       - EAPIoU beta：0.1 → 0.15（更嚴格的寬高比約束）
    
    D. 推理閾值（極低過濾）
       - conf閾值：0.15 → 0.001（幾乎不過濾，消除漏檢）
       - NMS IoU：0.5 → 0.4（保留更多重疊框）
    
    E. 正則化（降低約束，避免欠擬合）
       - weight_decay：0.003 → 0.0005（降低83%）
       - dropout：0.4 → 0.2（降低50%）
       - label_smoothing：0.1 → 0.05（降低50%）
    
    F. 數據增強（CT醫學影像專用優化）
       - Mosaic/Mixup：保持1.0/0.2（增加樣本多樣性）
       - Copy-Paste：0.3 → 0.1（避免違反解剖學約束）
       - 旋轉角度：20° → 5°（CT已標準化，僅模擬體位偏差）
       - 平移/縮放：降低至0.1/0.5（模擬掃描協議差異）
       - 垂直翻轉：0.3 → 0.0（❌ 禁用，破壞解剖結構）
       - HSV增強：禁用色調/飽和度（CT是灰階），僅保留亮度0.4（模擬窗寬窗位）
       - 水平翻轉：保持0.5（✅ 左右肺對稱，醫學合理）
    
    G. 架構增強
       - 深度監督：啟用（多層級損失）
       - 多尺度訓練：啟用（0.5-1.5x）
    
    【預期性能提升】
    當前 Epoch 85：
      - mAP@0.5: 0.389
      - Precision: 0.487
      - Recall: 0.305
      - F1 Score: ~0.375
    
    目標 Epoch 300：
      - mAP@0.5: ≥ 0.80 (+106%)
      - Precision: ≥ 0.75 (+54%)
      - Recall: ≥ 0.85 (+179%)
      - F1 Score: ≥ 0.80 (+113%)
    
    【關鍵突破點】
    1. conf=0.001 + 負樣本0.05 → Recall預期達0.85+
    2. 正樣本8x過採樣 + Copy-Paste 0.3 → 消除小目標漏檢
    3. 深度監督 + EAPIoU β=0.15 → 定位精度顯著提升
    4. 300 epochs充分訓練 → 突破停滯平台
    
    Args:
        model_path: 模型設定檔案路徑
        data_dir: 訓練資料目錄
        output_dir: 輸出目錄
        epochs: 訓練輪數 (預設300，充分收斂至0.8目標)
        batch_size: 批次大小
        imgsz: 輸入影像大小
        lr: 初始學習率 (預設0.01，激進策略)
        lrf: 最終學習率倍數 (預設0.1，保持學習能力)
        val_ratio: 驗證集比例
        max_negative_ratio: 最大负樣本比例 (預設0.05，大幅降低)
        oversample_positive: 正樣本過採樣倍數 (預設8.0，加倍)
        warmup_epochs: 預熱輪數 (預設30)
        optimizer: 最佳化器類型
        weight_decay: 權重衰減 (預設0.0005，大幅降低)
        mosaic: Mosaic 增強機率
        mixup: Mixup 增強機率 (預設0.2)
        copy_paste: Copy-Paste 增強機率 (預設0.3，翻倍)
        degrees: 旋转角度範圍 (預設20.0)
        translate: 平移範圍 (預設0.2)
        scale: 縮放範圍 (預設0.9)
        fliplr: 水平翻轉機率
        flipud: 垂直翻轉機率 (預設0.3，新增)
        hsv_h: 色調變化 (預設0.03)
        hsv_s: 飽和度變化 (預設0.9)
        hsv_v: 亮度變化 (預設0.6)
        patience: 早停耐心值 (預設50)
        workers: 資料載入執行緒數
        save_period: 儲存週期 (預設5)
        random_seed: 隨機種子
        eapiou_beta: EAPIoU beta參數 (預設0.15)
        deep_supervision: 是否啟用深度監督 (預設True)
        box: 邊界框損失權重 (預設8.5)
        cls: 分類損失權重 (預設0.2，極低)
        dfl: DFL 損失權重 (預設2.0)
        dropout: Dropout 機率 (預設0.2)
        label_smoothing: 標籤平滑值 (預設0.05)
        iou: NMS IoU 阈值 (預設0.4)
        conf: 推理置信度閾值 (預設0.001，極低)
        multi_scale: 多尺度訓練 (預設True)
        cos_lr: Cosine學習率調度 (預設True)
        close_mosaic: 關閉Mosaic輪數 (預設50)
    """
    
    # 設定隨機種子
    set_seed(random_seed)
    
    # 產生時間戳記
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 建立輸出目錄
    save_dir = Path(output_dir) / f"train_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 設定日誌
    log_file = setup_logging(save_dir / "logs", timestamp)
    
    logging.info("=" * 80)
    logging.info("🛡️ YOLO11-SSE-EAPIoU 訓練啟動 (超保守版 - 防止檢測失效)")
    logging.info("=" * 80)
    logging.info(f"📂 模型：{model_path}")
    logging.info(f"📂 資料：{data_dir}")
    logging.info(f"📂 輸出：{save_dir}")
    logging.info("")
    logging.info(f"🎯 性能目標：mAP@0.5≥0.60, Recall≥0.55, Precision≥0.60")
    logging.info(f"📊 訓練配置：Epochs={epochs}, Batch={batch_size}, LR={lr}→{lr*lrf}, Patience={patience}")
    logging.info(f"🛡️ 超保守策略：負樣本{max_negative_ratio}(大幅增加), 正樣本×{oversample_positive}(降低), Warmup={warmup_epochs}")
    logging.info(f"🫁 CT增強：rotation=±{degrees}°, fliplr={fliplr}, hsv_v={hsv_v}")
    logging.info(f"🔧 EAPIoU：beta={eapiou_beta}, cls={cls}(提高), conf={conf}(提高), dropout={dropout}(降低)")
    logging.info("=" * 80)
    logging.info("")
    
    # 1. 資料準備
    logging.info("📊 步驟 1/3: 資料處理")
    
    # 收集患者資料
    data_path = Path(data_dir)
    patient_data = collect_patient_data(data_path)
    
    if not patient_data:
        logging.error("❌ 未找到任何患者資料，終止訓練")
        return {"success": False, "error": "No patient data found"}
    
    # 劃分訓練集和驗證集
    patient_ids = list(patient_data.keys())
    train_patients, val_patients = split_patients(patient_ids, val_ratio, random_seed)
    train_patients, val_patients = ensure_val_has_positive_samples(
        patient_data, val_patients, train_patients
    )
    
    logging.info(f"訓練集: {len(train_patients)} 患者, 驗證集: {len(val_patients)} 患者")
    
    # 建立 YOLO 資料集
    dataset_dir = save_dir / f"dataset_{timestamp}"
    dataset_yaml = create_yolo_dataset(
        patient_data,
        train_patients,
        val_patients,
        dataset_dir,
        max_negative_ratio=max_negative_ratio,
        oversample_positive=oversample_positive,
        random_seed=random_seed
    )
    
    logging.info(f"✅ 資料集建立完成：{dataset_yaml}")
    
    # 2. 模型初始化
    logging.info("🔧 步驟 2/3: 模型初始化")
    
    # 預先下載 AMP 檢查所需的 yolo11n.pt（避免訓練中途下載）
    try:
        _ = YOLO("yolo11n.pt")
    except Exception as e:
        pass
    
    model = YOLO(model_path)
    logging.info(f"✅ 模型載入完成")
    
    # 🔧 儲存 EAPIoU 參數到模型物件（稍後 callback 使用）
    model._custom_eapiou_beta = eapiou_beta
    model._custom_deep_supervision = deep_supervision
    
    # 新增訓練後回調，替换損失函數（在模型完全初始化後）
    if EAPIOU_AVAILABLE:
        model.add_callback('on_train_start', patch_model_with_eapiou)
    
    # 3. 訓練
    logging.info("🚀 步驟 3/3: 開始訓練")
    
    # 🔧 修改：使用相同的輸出目錄（不再創建子資料夾）
    project_name = f"train_{timestamp}"
    
    # 訓練參數
    train_args = {
        'data': str(dataset_yaml),
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': imgsz,
        'lr0': lr,
        'lrf': lrf,
        'momentum': 0.937,
        'weight_decay': weight_decay,
        'warmup_epochs': warmup_epochs,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': box,
        'cls': cls,
        'dfl': dfl,
        'dropout': dropout,
        'label_smoothing': label_smoothing,
        'iou': iou,
        'conf': conf,
        'optimizer': optimizer,
        'verbose': True,
        'seed': random_seed,
        'deterministic': False,
        'single_cls': True,
        'rect': False,
        'cos_lr': cos_lr,  # 🚀 使用傳入的參數
        'close_mosaic': close_mosaic,  # 🚀 延後關閉
        'hsv_h': hsv_h,  # 🚀 增強色調
        'hsv_s': hsv_s,  # 🚀 增強飽和度
        'hsv_v': hsv_v,  # 🚀 增強亮度
        'degrees': degrees,
        'translate': translate,
        'scale': scale,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': flipud,  # 🚀 新增垂直翻轉
        'fliplr': fliplr,
        'bgr': 0.0,
        'mosaic': mosaic,
        'mixup': mixup,
        'copy_paste': copy_paste,
        'auto_augment': None,
        'erasing': 0.0,
        'crop_fraction': 1.0,
        'project': str(save_dir.parent),  # 使用 yolo_runs 作為 project
        'name': project_name,             # 使用 train_{timestamp} 作為 name
        'exist_ok': True,                 # 🔧 允許覆寫（因為已經創建了 dataset）
        'pretrained': False,
        'resume': False,
        'amp': True,
        'fraction': 1.0,
        'profile': False,
        'freeze': None,
        'multi_scale': multi_scale,  # 🚀 啟用多尺度
        'overlap_mask': True,
        'mask_ratio': 4,
        'val': True,
        'split': 'val',
        'save': True,
        'save_period': save_period,
        'cache': False,
        'device': None,
        'workers': workers,
        'patience': patience,
        'plots': True,
        'nbs': 64,
        'save_json': False,
        'save_hybrid': False,
    }
    
    # 開始計時
    start_time = time.time()
    logging.info(f"⏱️  訓練開始：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 開始訓練
        results = model.train(**train_args)
        
        # 結束計時
        end_time = time.time()
        total_time_seconds = end_time - start_time
        hours = int(total_time_seconds // 3600)
        minutes = int((total_time_seconds % 3600) // 60)
        seconds = int(total_time_seconds % 60)
        
        logging.info("")
        logging.info("=" * 80)
        logging.info("✅ 訓練完成！")
        logging.info(f"⏱️  訓練時間：{hours}h {minutes}m {seconds}s")
        logging.info("=" * 80)
        
    except Exception as e:
        logging.error("")
        logging.error("=" * 80)
        logging.error("❌ 訓練失敗！")
        logging.error(f"錯誤：{str(e)}")
        logging.error("=" * 80)
        import traceback
        logging.error(traceback.format_exc())
        return {"success": False, "error": str(e)}
    
    # 4. 儲存訓練設定
    # 🔧 修改：直接使用 save_dir，不再創建新路徑
    output_path = save_dir
    config_path = output_path / "train_config.json"
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({
                'model': model_path,
                'data_dir': data_dir,
                'dataset_yaml': str(dataset_yaml),
                'timestamp': timestamp,
                'training_params': {
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'imgsz': imgsz,
                    'lr': lr,
                    'lrf': lrf,
                    'val_ratio': val_ratio,
                    'max_negative_ratio': max_negative_ratio,
                    'oversample_positive': oversample_positive,
                    'warmup_epochs': warmup_epochs,
                    'optimizer': optimizer,
                    'weight_decay': weight_decay,
                    'random_seed': random_seed,
                    'conf': conf,
                    'multi_scale': multi_scale,
                    'cos_lr': cos_lr,
                    'close_mosaic': close_mosaic,
                },
                'eapiou_config': {
                    'eapiou_beta': eapiou_beta,
                    'deep_supervision': deep_supervision,
                    'eapiou_enabled': EAPIOU_AVAILABLE,
                },
                'loss_weights': {
                    'box': box,
                    'cls': cls,
                    'dfl': dfl,
                    'dropout': dropout,
                    'label_smoothing': label_smoothing,
                    'iou': iou,
                },
                'data_augmentation': {
                    'mosaic': mosaic,
                    'mixup': mixup,
                    'copy_paste': copy_paste,
                    'degrees': degrees,
                    'translate': translate,
                    'scale': scale,
                    'fliplr': fliplr,
                    'flipud': flipud,
                    'hsv_h': hsv_h,
                    'hsv_s': hsv_s,
                    'hsv_v': hsv_v,
                },
                'training_time': {
                    'total_seconds': total_time_seconds,
                },
            }, f, indent=2, ensure_ascii=False)
        
        logging.info(f"✅ 訓練設定已儲存")
    except Exception as e:
        logging.warning(f"⚠️ 設定儲存失敗：{e}")
    
    logging.info(f"📁 輸出目錄：{output_path}")
    logging.info(f"📊 最佳模型：{output_path / 'weights' / 'best.pt'}")
    
    return {
        "success": True, 
        "results": results, 
        "training_time": total_time_seconds,
        "output_path": str(output_path),
        "config_path": str(config_path)
    }


def main():
    parser = argparse.ArgumentParser(
        description="使用 EAPIoU Loss 訓練 YOLO11 模型",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 基本參數
    parser.add_argument('--model', type=str, required=True, help='模型設定檔案路徑')
    parser.add_argument('--data_dir', type=str, required=True, help='訓練資料目錄')
    parser.add_argument('--output_dir', type=str, default='yolo_runs', help='輸出目錄')
    
    # 訓練參數
    parser.add_argument('--epochs', type=int, default=300, help='訓練輪數（超保守版300）')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小（適用8GB顯存）')
    parser.add_argument('--imgsz', type=int, default=640, help='輸入影像大小')
    parser.add_argument('--lr', type=float, default=0.003, help='初始學習率（超保守版）')
    parser.add_argument('--lrf', type=float, default=0.1, help='最終學習率倍數（超保守版）')
    parser.add_argument('--patience', type=int, default=75, help='早停耐心值（適中）')
    parser.add_argument('--workers', type=int, default=8, help='資料載入執行緒數')
    parser.add_argument('--seed', type=int, default=42, help='隨機種子')
    
    # EAPIoU 參數
    parser.add_argument('--eapiou_beta', type=float, default=0.15, help='EAPIoU beta 參數（已驗證）')
    parser.add_argument('--deep_supervision', action='store_true', default=True, help='啟用深度監督')
    
    # 損失權重
    parser.add_argument('--box', type=float, default=7.5, help='邊界框損失權重')
    parser.add_argument('--cls', type=float, default=0.8, help='分類損失權重（提高）')
    parser.add_argument('--dfl', type=float, default=1.5, help='DFL 損失權重')
    
    # 正則化
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout 機率（降低）')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='標籤平滑值（移除）')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='權重衰減（標準）')
    parser.add_argument('--warmup_epochs', type=int, default=40, help='預熱輪數（適中）')
    
    # 資料增強
    parser.add_argument('--mosaic', type=float, default=0.0, help='Mosaic 機率（肺部CT禁用）')
    parser.add_argument('--mixup', type=float, default=0.0, help='Mixup 機率（肺部CT禁用）')
    parser.add_argument('--copy_paste', type=float, default=0.0, help='Copy-Paste 機率（肺部CT禁用）')
    parser.add_argument('--degrees', type=float, default=3.0, help='旋转角度（肺部CT僅±3°）')
    parser.add_argument('--translate', type=float, default=0.02, help='平移範圍（肺部CT極小2%）')
    parser.add_argument('--scale', type=float, default=0.1, help='縮放範圍（肺部CT極小±10%）')
    parser.add_argument('--fliplr', type=float, default=0.5, help='水平翻轉機率')
    parser.add_argument('--flipud', type=float, default=0.0, help='垂直翻轉機率（肺部CT禁用）')
    parser.add_argument('--hsv_h', type=float, default=0.0, help='HSV色調變化（CT禁用）')
    parser.add_argument('--hsv_s', type=float, default=0.0, help='HSV飽和度變化（CT禁用）')
    parser.add_argument('--hsv_v', type=float, default=0.3, help='HSV亮度變化（窗寬窗位調整）')
    
    # 樣本平衡
    parser.add_argument('--val_ratio', type=float, default=0.15, help='驗證集比例')
    parser.add_argument('--max_negative_ratio', type=float, default=0.1, help='最大負樣本比例（超保守版）')
    parser.add_argument('--oversample_positive', type=float, default=2.0, help='正樣本過採樣倍數（降低×2）')
    
    # NMS 和推理閾值
    parser.add_argument('--iou', type=float, default=0.5, help='NMS IoU 閾值（保持0.5）')
    parser.add_argument('--conf', type=float, default=0.1, help='推理置信度閾值（提高防止飽和）')
    
    # 訓練策略
    parser.add_argument('--multi_scale', type=bool, default=False, help='多尺度訓練（禁用節省顯存）')
    parser.add_argument('--cos_lr', type=bool, default=True, help='Cosine學習率調度')
    parser.add_argument('--close_mosaic', type=int, default=0, help='關閉Mosaic輪數（從不使用）')
    
    # 其他
    parser.add_argument('--optimizer', type=str, default='AdamW', help='最佳化器類型')
    parser.add_argument('--save_period', type=int, default=5, help='儲存週期（每5 epochs）')
    
    args = parser.parse_args()
    
    # 執行訓練
    result = train_yolo_with_eapiou(
        model_path=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        lr=args.lr,
        lrf=args.lrf,
        val_ratio=args.val_ratio,
        max_negative_ratio=args.max_negative_ratio,
        oversample_positive=args.oversample_positive,
        warmup_epochs=args.warmup_epochs,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        fliplr=args.fliplr,
        flipud=args.flipud,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        patience=args.patience,
        workers=args.workers,
        save_period=args.save_period,
        random_seed=args.seed,
        eapiou_beta=args.eapiou_beta,
        deep_supervision=args.deep_supervision,
        box=args.box,
        cls=args.cls,
        dfl=args.dfl,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        iou=args.iou,
        conf=args.conf,
        multi_scale=args.multi_scale,
        cos_lr=args.cos_lr,
        close_mosaic=args.close_mosaic,
    )
    
    # 返回結果
    if result["success"]:
        sys.exit(0)
    else:
        logging.error("❌ 訓練失敗")
        sys.exit(1)


if __name__ == '__main__':
    main()
