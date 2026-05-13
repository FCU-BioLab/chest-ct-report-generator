#!/usr/bin/env python3
"""
CLI for the RetinaNet detection pipeline.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _apply_train_overrides(config, args) -> None:
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.patch_size:
        config.patch_size = args.patch_size
    if args.val_patch_size:
        config.val_patch_size = args.val_patch_size
    if args.spacing:
        config.spacing = args.spacing
    if args.proposal_score_thresh is not None:
        config.proposal_score_thresh = args.proposal_score_thresh
    if args.test_score_thresh is not None:
        config.test_score_thresh = args.test_score_thresh
    # Backward compatibility: keep old flag behavior for training.
    if args.score_thresh is not None:
        config.proposal_score_thresh = args.score_thresh
    if args.nms_thresh is not None:
        config.nms_thresh = args.nms_thresh
    if args.anchor_shapes:
        config.base_anchor_shapes = [args.anchor_shapes[i:i + 3] for i in range(0, len(args.anchor_shapes), 3)]
    if args.anchor_scales:
        config.feature_map_scales = [args.anchor_scales[i:i + 3] for i in range(0, len(args.anchor_scales), 3)]
    if args.pretrained_weights is not None:
        config.pretrained_weights = args.pretrained_weights
    if args.no_pretrained:
        config.pretrained_weights = None
    if args.early_stop_patience is not None:
        config.early_stop_patience = args.early_stop_patience
    if args.early_stop_min_delta is not None:
        config.early_stop_min_delta = args.early_stop_min_delta
    if args.max_boxes_for_crop is not None:
        config.max_boxes_for_crop = args.max_boxes_for_crop
    if args.crop_pos_ratio is not None:
        config.crop_pos_ratio = args.crop_pos_ratio
    if args.crop_neg_ratio is not None:
        config.crop_neg_ratio = args.crop_neg_ratio
    if args.train_pos_oversample_weight is not None:
        config.train_pos_oversample_weight = args.train_pos_oversample_weight
    if args.train_epoch_samples is not None:
        config.train_epoch_samples = args.train_epoch_samples
    config.validate()


def cmd_train(args):
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
    _apply_train_overrides(config, args)
    resume_checkpoint = args.resume_checkpoint
    if args.resume and not resume_checkpoint:
        resume_checkpoint = str(Path(config.output_dir) / "train_state_last.pt")
    RetinaNetTrainer(config).train(resume_checkpoint=resume_checkpoint)


def cmd_eval(args):
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
    config.validate()

    trainer = RetinaNetTrainer(config)
    logging.info("Running one-pass validation...")
    val_metrics = trainer._validate(epoch=0)
    for key, value in sorted(val_metrics.items()):
        if key.startswith("_") or key == "f1_per_threshold":
            continue
        logging.info("%s: %s", key, value)


def cmd_test(args):
    from .config import RetinaNetConfig
    from .trainer import RetinaNetTrainer

    final_thresh = args.final_thresh if args.final_thresh is not None else args.score_thresh
    ensemble_paths = [str(Path(p)) for p in (args.ensemble_checkpoints or []) if p]
    if args.checkpoint:
        cp = str(Path(args.checkpoint))
        if cp not in ensemble_paths:
            ensemble_paths.insert(0, cp)
    if not ensemble_paths:
        raise ValueError("test mode requires --checkpoint or --ensemble_checkpoints")
    primary_checkpoint = ensemble_paths[0]

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
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.val_patch_size:
        config.val_patch_size = args.val_patch_size
    else:
        config.output_dir = str(Path(primary_checkpoint).parent)
    if args.spacing:
        config.spacing = args.spacing
    if args.candidate_thresh is not None:
        config.proposal_score_thresh = args.candidate_thresh
    if final_thresh is not None:
        config.test_score_thresh = final_thresh
    if args.nms_thresh is not None:
        config.nms_thresh = args.nms_thresh
    config.validate()
    fpr_thresh = 0.0 if args.fpr_no_hard_filter else args.fpr_thresh

    trainer = RetinaNetTrainer(config)
    trainer.output_dir = Path(config.output_dir)
    trainer.output_dir.mkdir(parents=True, exist_ok=True)
    trainer._run_test_evaluation(
        save_gifs=args.save_gifs,
        gif_dir=args.gif_dir,
        filter_fp=args.filter_fp,
        score_thresh=final_thresh,
        preprocess_lung_mask=args.preprocess_lung_mask,
        lung_mask_preprocess_dilate=args.lung_mask_preprocess_dilate,
        lung_mask_preprocess_fill=args.lung_mask_preprocess_fill,
        filter_lung_mask=args.filter_lung_mask,
        lung_mask_thresh=args.lung_mask_thresh,
        lung_mask_min_overlap_ratio=args.lung_mask_min_overlap_ratio,
        lung_mask_method=args.lung_mask_method,
        override_model_path=primary_checkpoint,
        ensemble_model_paths=ensemble_paths if len(ensemble_paths) > 1 else None,
        ensemble_iou_thresh=args.ensemble_iou_thresh,
        ensemble_vote_power=args.ensemble_vote_power,
        fpr_model_path=args.fpr_model,
        fpr_thresh=fpr_thresh,
        fpr_patch_size=args.fpr_patch_size,
        fpr_weight=args.fpr_weight,
        fpr_mode=args.fpr_mode,
        fpr_fuser_model_path=args.fpr_fuser_model,
        fp_max_elongation=args.fp_max_elongation,
        fp_min_solidity=args.fp_min_solidity,
        fp_min_vol=args.fp_min_vol,
        fp_max_vol=args.fp_max_vol,
        morph_skip_small_diam=args.morph_skip_small_diam,
        morph_three_plane=args.morph_three_plane,
        morph_min_bad_planes=args.morph_min_bad_planes,
        morph_require_round_planes=args.morph_require_round_planes,
        morph_min_round_planes=args.morph_min_round_planes,
        morph_max_round_elongation=args.morph_max_round_elongation,
        morph_min_plane_area_ratio=args.morph_min_plane_area_ratio,
        morph_axial_similarity=args.morph_axial_similarity,
        morph_max_elongation_delta=args.morph_max_elongation_delta,
        morph_max_solidity_delta=args.morph_max_solidity_delta,
        morph_max_fill_delta=args.morph_max_fill_delta,
        morph_min_axial_area_ratio=args.morph_min_axial_area_ratio,
        bbox_filter=args.bbox_filter,
        bbox_max_aspect_ratio=args.bbox_max_aspect_ratio,
        bbox_aspect_skip_small_diam=args.bbox_aspect_skip_small_diam,
        eval_split=args.eval_split,
        max_samples=args.max_samples,
        fpr_score_aware=args.fpr_score_aware,
        fpr_det_high_thresh=args.fpr_det_high_thresh,
        fpr_det_mid_thresh=args.fpr_det_mid_thresh,
        fpr_high_thresh=args.fpr_high_thresh,
        fpr_mid_thresh=args.fpr_mid_thresh,
        fpr_apply_min_diam=args.fpr_apply_min_diam,
        fpr_apply_max_diam=args.fpr_apply_max_diam,
        size_aware_small_diam=args.size_aware_small_diam,
        size_aware_fpr_thresh=args.size_aware_fpr_thresh,
        size_aware_final_thresh=args.size_aware_final_thresh,
        export_case_analysis=args.export_case_analysis,
        case_analysis_dir=args.case_analysis_dir,
    )


def cmd_predict(args):
    from .config import RetinaNetConfig
    from .trainer import RetinaNetTrainer

    config = RetinaNetConfig(
        amp=not args.no_amp,
        device=args.device,
    )
    trainer = RetinaNetTrainer(config)
    trainer.load_checkpoint(args.checkpoint)
    results = trainer.predict(args.input)
    for i, result in enumerate(results):
        logging.info("result %d: %d boxes", i, len(result["boxes"]))
        for j, (box, score) in enumerate(zip(result["boxes"], result["scores"])):
            logging.info("  [%d] box=%s score=%.4f", j, box, score)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MONAI 3D RetinaNet CLI")
    parser.add_argument("--log_level", default="INFO", help="DEBUG, INFO, WARN, ERROR")
    subparsers = parser.add_subparsers(dest="command", help="subcommands")

    train_p = subparsers.add_parser("train", help="train RetinaNet")
    train_p.add_argument("--data_path", default="detection/manifests/dataset_lndb.json", help="dataset JSON path")
    train_p.add_argument("--output_dir", default=None, help="output directory")
    train_p.add_argument("--epochs", type=int, default=300)
    train_p.add_argument("--batch_size", type=int, default=1)
    train_p.add_argument("--lr", type=float, default=0.001)
    train_p.add_argument("--val_interval", type=int, default=5)
    train_p.add_argument("--early_stop_patience", type=int, default=None, help="early stop patience in epochs (0=disabled)")
    train_p.add_argument("--early_stop_min_delta", type=float, default=None, help="minimum mAP improvement to reset early stop")
    train_p.add_argument("--max_boxes_for_crop", type=int, default=None, help="cap boxes per scan before random crop to reduce RAM")
    train_p.add_argument("--crop_pos_ratio", type=int, default=None, help="RandCrop positive sampling weight")
    train_p.add_argument("--crop_neg_ratio", type=int, default=None, help="RandCrop negative sampling weight")
    train_p.add_argument("--train_pos_oversample_weight", type=float, default=None, help="case-level weight for scans with nodules")
    train_p.add_argument("--train_epoch_samples", type=int, default=None, help="number of scan-samples per epoch (0=dataset size)")
    train_p.add_argument("--train_ratio", type=float, default=0.8)
    train_p.add_argument("--val_ratio", type=float, default=0.1)
    train_p.add_argument("--test_ratio", type=float, default=0.1)
    train_p.add_argument("--split_seed", type=int, default=42)
    train_p.add_argument("--device", default="cuda")
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--num_workers", type=int, default=4)
    train_p.add_argument("--no_amp", action="store_true")
    train_p.add_argument("--no_cache", action="store_true")
    train_p.add_argument("--patch_size", type=int, nargs=3, default=None, metavar=("H", "W", "D"))
    train_p.add_argument("--val_patch_size", type=int, nargs=3, default=None, metavar=("H", "W", "D"))
    train_p.add_argument("--spacing", type=float, nargs=3, default=None, metavar=("SX", "SY", "SZ"))
    train_p.add_argument("--proposal_score_thresh", type=float, default=None, help="candidate proposal threshold (stage 1)")
    train_p.add_argument("--test_score_thresh", type=float, default=None, help="default final threshold for evaluation (stage 2)")
    train_p.add_argument("--score_thresh", type=float, default=None)
    train_p.add_argument("--nms_thresh", type=float, default=None)
    train_p.add_argument("--anchor_shapes", type=int, nargs=9, default=None,
                         metavar=("A1X", "A1Y", "A1Z", "A2X", "A2Y", "A2Z", "A3X", "A3Y", "A3Z"))
    train_p.add_argument("--anchor_scales", type=int, nargs=9, default=None,
                         metavar=("S1X", "S1Y", "S1Z", "S2X", "S2Y", "S2Z", "S3X", "S3Y", "S3Z"))
    train_p.add_argument("--pretrained_weights", default=None)
    train_p.add_argument("--no_pretrained", action="store_true")
    train_p.add_argument("--resume", action="store_true", help="resume training from output_dir/train_state_last.pt")
    train_p.add_argument("--resume_checkpoint", default=None, help="explicit training-state checkpoint path (.pt)")

    eval_p = subparsers.add_parser("eval", help="run validation once")
    eval_p.add_argument("--data_path", default="detection/manifests/dataset_lndb.json")
    eval_p.add_argument("--output_dir", default=None)
    eval_p.add_argument("--pretrained_weights", default=None)
    eval_p.add_argument("--train_ratio", type=float, default=0.8)
    eval_p.add_argument("--val_ratio", type=float, default=0.1)
    eval_p.add_argument("--test_ratio", type=float, default=0.1)
    eval_p.add_argument("--split_seed", type=int, default=42)
    eval_p.add_argument("--device", default="cuda")
    eval_p.add_argument("--num_workers", type=int, default=4)
    eval_p.add_argument("--no_amp", action="store_true")
    eval_p.add_argument("--no_cache", action="store_true")

    test_p = subparsers.add_parser("test", help="evaluate test split from checkpoint")
    test_p.add_argument("--checkpoint", default=None)
    test_p.add_argument("--ensemble_checkpoints", nargs="+", default=None, help="optional multi-checkpoint ensemble list")
    test_p.add_argument("--ensemble_iou_thresh", type=float, default=None, help="IoU threshold for ensemble box fusion")
    test_p.add_argument("--ensemble_vote_power", type=float, default=1.0, help="score vote factor exponent for ensemble (>=0)")
    test_p.add_argument("--data_path", default="detection/manifests/dataset_lndb.json")
    test_p.add_argument("--output_dir", default=None)
    test_p.add_argument("--val_patch_size", type=int, nargs=3, default=None, metavar=("H", "W", "D"))
    test_p.add_argument("--train_ratio", type=float, default=0.8)
    test_p.add_argument("--val_ratio", type=float, default=0.1)
    test_p.add_argument("--test_ratio", type=float, default=0.1)
    test_p.add_argument("--split_seed", type=int, default=42)
    test_p.add_argument("--device", default="cuda")
    test_p.add_argument("--num_workers", type=int, default=4)
    test_p.add_argument("--no_amp", action="store_true")
    test_p.add_argument("--spacing", type=float, nargs=3, default=None, metavar=("SX", "SY", "SZ"))
    test_p.add_argument("--candidate_thresh", type=float, default=None, help="candidate proposal threshold before FP filtering (stage 1)")
    test_p.add_argument("--final_thresh", type=float, default=None, help="final score threshold after filtering (stage 2)")
    test_p.add_argument("--score_thresh", type=float, default=None)
    test_p.add_argument("--nms_thresh", type=float, default=None)
    test_p.add_argument("--eval_split", choices=["val", "test"], default="test", help="which split to evaluate")
    test_p.add_argument("--max_samples", type=int, default=0, help="limit number of scans for quick tuning (0=all)")
    test_p.add_argument("--save_gifs", action="store_true")
    test_p.add_argument("--gif_dir", default=None)
    test_p.add_argument("--filter_fp", action="store_true")
    test_p.add_argument("--preprocess_lung_mask", action="store_true", help="apply a coarse lung mask to the CT before detector inference")
    test_p.add_argument("--lung_mask_preprocess_dilate", type=int, default=5, help="dilation iterations for the preprocessing lung mask")
    test_p.add_argument("--lung_mask_preprocess_fill", type=float, default=0.0, help="normalized value used outside the preprocessing lung mask")
    test_p.add_argument("--filter_lung_mask", action="store_true")
    test_p.add_argument("--lung_mask_thresh", type=float, default=0.47, help="air threshold for generated lung mask in normalized CT scale")
    test_p.add_argument("--lung_mask_min_overlap_ratio", type=float, default=0.01, help="keep boxes with at least this lung-mask overlap ratio")
    test_p.add_argument("--lung_mask_method", choices=["slice", "volume"], default="slice", help="lung mask generation method: slice is n8n-style, volume is legacy")
    test_p.add_argument("--fpr_model", default=None)
    test_p.add_argument("--fpr_mode", choices=["fuse", "gate", "hybrid", "learned"], default="hybrid")
    test_p.add_argument("--fpr_thresh", type=float, default=0.5)
    test_p.add_argument("--fpr_patch_size", type=int, default=32)
    test_p.add_argument("--fpr_weight", type=float, default=0.5)
    test_p.add_argument("--fpr_fuser_model", default=None, help="optional learned fuser checkpoint (model_best.pt)")
    test_p.add_argument("--fpr_no_hard_filter", action="store_true", help="use FPR/fuser only to rescore proposals; do not remove proposals before final score thresholding")
    test_p.add_argument("--fpr_score_aware", action="store_true", help="enable score-aware gate policy for FPR filtering")
    test_p.add_argument("--fpr_det_high_thresh", type=float, default=0.9, help="detector high-score boundary")
    test_p.add_argument("--fpr_det_mid_thresh", type=float, default=0.6, help="detector mid-score lower bound")
    test_p.add_argument("--fpr_high_thresh", type=float, default=0.15, help="FPR gate threshold for high-score proposals")
    test_p.add_argument("--fpr_mid_thresh", type=float, default=0.25, help="FPR gate threshold for mid-score proposals")
    test_p.add_argument("--fpr_apply_min_diam", type=float, default=None, help="apply FPR only to proposals with max diameter >= this value in mm")
    test_p.add_argument("--fpr_apply_max_diam", type=float, default=None, help="apply FPR only to proposals with max diameter <= this value in mm")
    test_p.add_argument("--size_aware_small_diam", type=float, default=0.0, help="enable size-aware postprocess for proposals with max diameter <= this value")
    test_p.add_argument("--size_aware_fpr_thresh", type=float, default=None, help="FPR gate threshold for small proposals")
    test_p.add_argument("--size_aware_final_thresh", type=float, default=None, help="final score threshold for small proposals")
    test_p.add_argument("--fp_max_elongation", type=float, default=5.0)
    test_p.add_argument("--fp_min_solidity", type=float, default=0.3)
    test_p.add_argument("--fp_min_vol", type=float, default=4.2)
    test_p.add_argument("--fp_max_vol", type=float, default=65450.0)
    test_p.add_argument("--morph_skip_small_diam", type=float, default=0.0, help="skip morphology FP filtering for boxes with max diameter <= this value")
    test_p.add_argument("--morph_three_plane", action="store_true", help="apply morphology FP filtering on axial/coronal/sagittal projections")
    test_p.add_argument("--morph_min_bad_planes", type=int, default=2, help="remove a box only if at least this many planes fail morphology checks")
    test_p.add_argument("--morph_require_round_planes", action="store_true", help="keep morphology candidates only when enough projections are round and similarly sized")
    test_p.add_argument("--morph_min_round_planes", type=int, default=2, help="minimum round/similar projection planes required to keep a candidate")
    test_p.add_argument("--morph_max_round_elongation", type=float, default=1.8, help="maximum major/minor axis ratio for a projection to count as round")
    test_p.add_argument("--morph_min_plane_area_ratio", type=float, default=0.5, help="minimum smaller/larger area ratio among selected round planes")
    test_p.add_argument("--morph_axial_similarity", action="store_true", help="keep candidates when sagittal or coronal shape is similar to axial projection")
    test_p.add_argument("--morph_max_elongation_delta", type=float, default=1.5, help="maximum elongation difference from axial projection")
    test_p.add_argument("--morph_max_solidity_delta", type=float, default=0.35, help="maximum solidity difference from axial projection")
    test_p.add_argument("--morph_max_fill_delta", type=float, default=0.35, help="maximum bbox fill-ratio difference from axial projection")
    test_p.add_argument("--morph_min_axial_area_ratio", type=float, default=0.35, help="minimum smaller/larger area ratio between axial and side projection")
    test_p.add_argument("--bbox_filter", action="store_true", help="filter elongated prediction boxes by bbox axis aspect ratio")
    test_p.add_argument("--bbox_max_aspect_ratio", type=float, default=3.5, help="maximum max_axis/min_axis ratio for bbox filtering")
    test_p.add_argument("--bbox_aspect_skip_small_diam", type=float, default=0.0, help="skip bbox aspect filtering for boxes with max diameter <= this value in mm")
    test_p.add_argument("--export_case_analysis", action="store_true", help="export per-case CT/masks/HTML for TP/FP/FN analysis")
    test_p.add_argument("--case_analysis_dir", default=None, help="output directory for per-case analysis exports")

    pred_p = subparsers.add_parser("predict", help="predict from a single input")
    pred_p.add_argument("--checkpoint", required=True)
    pred_p.add_argument("--input", required=True)
    pred_p.add_argument("--device", default="cuda")
    pred_p.add_argument("--no_amp", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
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
