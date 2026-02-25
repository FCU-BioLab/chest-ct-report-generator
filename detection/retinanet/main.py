#!/usr/bin/env python3
"""
RetinaNet 偵測 CLI 工具
========================

用於 MONAI 3D RetinaNet 肺結節偵測的命令列介面 (CLI)。
支援訓練 (train)、測試 (test) 與單檔推論 (predict)。
"""

import argparse
import logging
import sys
from pathlib import Path

# 加入父目錄以支援模組匯入
sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_logging(log_level: str = "INFO"):
    """設定日誌格式與層級。"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cmd_train(args):
    """執行 RetinaNet 偵測器訓練。"""
    from .config import RetinaNetConfig
    from .trainer import RetinaNetTrainer

    config = RetinaNetConfig(
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
        val_interval=args.val_interval,
        amp=not args.no_amp,
        device=args.device,
        num_workers=args.num_workers,
        seed=args.seed,
        cache_dataset=not args.no_cache,
    )

    if args.output_dir:
        config.output_dir = args.output_dir
    if args.patch_size:
        config.patch_size = args.patch_size
    if args.pretrained_weights is not None:
        config.pretrained_weights = args.pretrained_weights
    if args.no_pretrained:
        config.pretrained_weights = None

    trainer = RetinaNetTrainer(config)
    trainer.train()


def cmd_eval(args):
    """載入預訓練模型並在驗證集上評估（不訓練）。"""
    from .config import RetinaNetConfig
    from .trainer import RetinaNetTrainer

    config = RetinaNetConfig(
        data_path=args.data_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
        amp=not args.no_amp,
        device=args.device,
        num_workers=args.num_workers,
        cache_dataset=not args.no_cache,
    )

    if args.output_dir:
        config.output_dir = args.output_dir
    if args.pretrained_weights is not None:
        config.pretrained_weights = args.pretrained_weights

    trainer = RetinaNetTrainer(config)

    # 直接執行驗證
    logging.info("🔍 使用預訓練模型評估驗證集...")
    val_metrics = trainer._validate(epoch=0)
    logging.info("📊 評估結果:")
    for k, v in sorted(val_metrics.items()):
        if k.startswith("_") or k == "f1_per_threshold":
            continue  # 跳過內部資料
        if isinstance(v, (int, float)):
            logging.info(f"  {k}: {v:.4f}")
        elif isinstance(v, dict):
            logging.info(f"  {k}:")
            for dk, dv in sorted(v.items()):
                if isinstance(dv, (int, float)):
                    logging.info(f"    {dk}: {dv:.4f}")
        # 跳過大型陣列


def cmd_test(args):
    """執行 RetinaNet 偵測器測試集評估。"""
    from .config import RetinaNetConfig
    from .trainer import RetinaNetTrainer
    from .dataset import prepare_datalist

    config = RetinaNetConfig(
        data_path=args.data_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
        amp=not args.no_amp,
        device=args.device,
        num_workers=args.num_workers,
    )

    trainer = RetinaNetTrainer(config)
    trainer.load_checkpoint(args.checkpoint)

    # 準備測試資料
    test_data = prepare_datalist(
        config.data_path, "test",
        config.train_ratio, config.val_ratio, config.test_ratio, config.split_seed,
    )

    logging.info(f"🔍 開始測試 {len(test_data)} 筆樣本...")
    results = []

    for i, item in enumerate(test_data):
        if i % 10 == 0:
            logging.info(f"  處理中: {i+1}/{len(test_data)}")

        preds = trainer.predict(item["image"])
        results.append({
            "image": item["image"],
            "gt_boxes": item["box"],
            "predictions": [
                {
                    "boxes": p["boxes"].tolist(),
                    "scores": p["scores"].tolist(),
                }
                for p in preds
            ],
        })

    # 儲存測試結果
    import json
    output_path = Path(args.checkpoint).parent / "test_results.json"
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logging.info(f"📁 測試結果已儲存至: {output_path}")


def cmd_predict(args):
    """執行單一檔案的偵測。"""
    from .config import RetinaNetConfig
    from .trainer import RetinaNetTrainer

    config = RetinaNetConfig(
        amp=not args.no_amp,
        device=args.device,
    )

    trainer = RetinaNetTrainer(config)
    trainer.load_checkpoint(args.checkpoint)

    results = trainer.predict(args.input)
    for i, r in enumerate(results):
        logging.info(f"偵測結果 {i}: 找到 {len(r['boxes'])} 個結節")
        for j, (box, score) in enumerate(zip(r["boxes"], r["scores"])):
            logging.info(f"  [{j}] box={box}, score={score:.4f}")


def main():
    parser = argparse.ArgumentParser(description="MONAI 3D RetinaNet 偵測工具")
    parser.add_argument("--log_level", default="INFO", help="日誌等級 (DEBUG, INFO, WARN, ERROR)")

    subparsers = parser.add_subparsers(dest="command", help="可用指令")

    # 訓練指令 (Train)
    train_p = subparsers.add_parser("train", help="訓練模型")
    train_p.add_argument("--data_path", default="cache/lndb_volume_npz_agr1", help="資料路徑 (dataset.json)")
    train_p.add_argument("--output_dir", default=None, help="輸出目錄")
    train_p.add_argument("--epochs", type=int, default=300, help="訓練 Epochs 數")
    train_p.add_argument("--batch_size", type=int, default=2, help="Batch Size")
    train_p.add_argument("--lr", type=float, default=0.001, help="學習率")
    train_p.add_argument("--val_interval", type=int, default=5, help="驗證間隔 (Epochs)")
    train_p.add_argument("--train_ratio", type=float, default=0.8, help="訓練集比例")
    train_p.add_argument("--val_ratio", type=float, default=0.1, help="驗證集比例")
    train_p.add_argument("--test_ratio", type=float, default=0.1, help="測試集比例")
    train_p.add_argument("--split_seed", type=int, default=42, help="資料分割隨機種子")
    train_p.add_argument("--device", default="cuda", help="運算裝置 (cuda/cpu)")
    train_p.add_argument("--seed", type=int, default=42, help="全域隨機種子")
    train_p.add_argument("--num_workers", type=int, default=4, help="DataLoader 工作執行緒")
    train_p.add_argument("--no_amp", action="store_true", help="停用自動混合精度 (AMP)")
    train_p.add_argument("--patch_size", type=int, nargs=3, default=None,
                         help="訓練 patch 大小 [H W D]，預設 192 192 80")
    train_p.add_argument("--pretrained_weights", default=None, help="預訓練權重路徑")
    train_p.add_argument("--no_pretrained", action="store_true", help="不使用預訓練權重")
    train_p.add_argument("--no_cache", action="store_true", help="停用 CacheDataset")

    # 評估指令 (Eval) — 用預訓練模型直接跑驗證
    eval_p = subparsers.add_parser("eval", help="用預訓練模型評估驗證集")
    eval_p.add_argument("--data_path", default="dataset_luna16.json", help="資料路徑")
    eval_p.add_argument("--output_dir", default=None, help="輸出目錄")
    eval_p.add_argument("--pretrained_weights", default=None, help="預訓練權重路徑")
    eval_p.add_argument("--train_ratio", type=float, default=0.8)
    eval_p.add_argument("--val_ratio", type=float, default=0.1)
    eval_p.add_argument("--test_ratio", type=float, default=0.1)
    eval_p.add_argument("--split_seed", type=int, default=42)
    eval_p.add_argument("--device", default="cuda")
    eval_p.add_argument("--num_workers", type=int, default=4)
    eval_p.add_argument("--no_amp", action="store_true")
    eval_p.add_argument("--no_cache", action="store_true", help="停用 CacheDataset")

    # 測試指令 (Test)
    test_p = subparsers.add_parser("test", help="在測試集上評估模型")
    test_p.add_argument("--checkpoint", required=True, help="模型檢查點路徑 (.pt)")
    test_p.add_argument("--data_path", default="cache/lndb_volume_npz_agr1", help="資料路徑")
    test_p.add_argument("--train_ratio", type=float, default=0.8)
    test_p.add_argument("--val_ratio", type=float, default=0.1)
    test_p.add_argument("--test_ratio", type=float, default=0.1)
    test_p.add_argument("--split_seed", type=int, default=42)
    test_p.add_argument("--device", default="cuda")
    test_p.add_argument("--num_workers", type=int, default=4)
    test_p.add_argument("--no_amp", action="store_true")

    # 單檔預測指令 (Predict)
    pred_p = subparsers.add_parser("predict", help="單一檔案預測")
    pred_p.add_argument("--checkpoint", required=True, help="模型檢查點路徑")
    pred_p.add_argument("--input", required=True, help="輸入檔案路徑")
    pred_p.add_argument("--device", default="cuda")
    pred_p.add_argument("--no_amp", action="store_true")

    args = parser.parse_args()
    setup_logging(args.log_level)

    if args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "test":
        cmd_test(args)
    elif args.command == "predict":
        cmd_predict(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
