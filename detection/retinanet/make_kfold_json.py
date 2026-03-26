#!/usr/bin/env python3
"""
Build strict group-aware K-fold JSON files for RetinaNet.

Input JSON is expected to follow the usual schema:
{
  "training": [...],
  "validation": [...],
  "testing": [...]
}

Behavior:
- K-fold split is applied to training + validation samples.
- testing split is kept unchanged in every fold JSON.
- Group-aware split avoids leaking the same series/patient across train/val.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


TRAIN_KEYS = ("training", "train")
VAL_KEYS = ("validation", "val")
TEST_KEYS = ("testing", "test")
DEFAULT_GROUP_KEYS = ("seriesuid", "lndb_id", "patient_id")


def _default_output_dir() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "detection" / "manifests"


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict JSON at: {path}")
    return obj


def _pick_section(data: Dict, keys: Sequence[str]) -> List[Dict]:
    for key in keys:
        value = data.get(key)
        if isinstance(value, list):
            return value
    return []


def _sample_group_id(sample: Dict, group_keys: Sequence[str]) -> str:
    for key in group_keys:
        value = sample.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    image_path = str(sample.get("image", "")).strip()
    if image_path:
        return Path(image_path).stem
    return "__missing_group__"


def _group_samples(samples: Iterable[Dict], group_keys: Sequence[str]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = {}
    for sample in samples:
        gid = _sample_group_id(sample, group_keys)
        grouped.setdefault(gid, []).append(sample)
    return grouped


def _split_group_ids(group_ids: List[str], k: int, seed: int) -> List[List[str]]:
    rng = random.Random(seed)
    shuffled = list(group_ids)
    rng.shuffle(shuffled)
    # Round-robin assignment for balanced fold sizes.
    return [shuffled[i::k] for i in range(k)]


def _write_json(path: Path, payload: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create strict group-aware K-fold JSON files")
    parser.add_argument("--input_json", required=True, help="source dataset json path")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="where to write fold json files (default: detection/manifests)",
    )
    parser.add_argument("--num_folds", type=int, default=5, help="number of folds")
    parser.add_argument("--seed", type=int, default=42, help="split seed")
    parser.add_argument(
        "--group_keys",
        nargs="+",
        default=list(DEFAULT_GROUP_KEYS),
        help="candidate keys to group by (fallback: image stem)",
    )
    parser.add_argument(
        "--output_prefix",
        default=None,
        help="output file prefix (default: input stem)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_json = Path(args.input_json).resolve()
    if not input_json.exists():
        raise FileNotFoundError(f"input_json not found: {input_json}")

    if args.num_folds < 2:
        raise ValueError("--num_folds must be >= 2")

    data = _load_json(input_json)
    train_samples = _pick_section(data, TRAIN_KEYS)
    val_samples = _pick_section(data, VAL_KEYS)
    test_samples = _pick_section(data, TEST_KEYS)

    pool_samples = list(train_samples) + list(val_samples)
    if not pool_samples:
        raise ValueError("No training/validation samples found in input JSON")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else _default_output_dir().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix if args.output_prefix else input_json.stem

    grouped = _group_samples(pool_samples, args.group_keys)
    group_ids = sorted(grouped.keys())
    fold_group_ids = _split_group_ids(group_ids, args.num_folds, args.seed)

    manifest = {
        "input_json": str(input_json),
        "output_dir": str(output_dir),
        "num_folds": args.num_folds,
        "seed": args.seed,
        "group_keys": args.group_keys,
        "num_groups": len(group_ids),
        "num_pool_samples": len(pool_samples),
        "num_test_samples": len(test_samples),
        "folds": [],
    }

    for fold_idx in range(args.num_folds):
        val_group_set = set(fold_group_ids[fold_idx])
        fold_train: List[Dict] = []
        fold_val: List[Dict] = []
        for gid, samples in grouped.items():
            if gid in val_group_set:
                fold_val.extend(samples)
            else:
                fold_train.extend(samples)

        fold_json = {
            "training": fold_train,
            "validation": fold_val,
            "testing": test_samples,
        }
        fold_path = output_dir / f"{prefix}_fold{fold_idx}.json"
        _write_json(fold_path, fold_json)

        manifest["folds"].append(
            {
                "fold": fold_idx,
                "path": str(fold_path),
                "train_samples": len(fold_train),
                "val_samples": len(fold_val),
                "test_samples": len(test_samples),
                "val_groups": len(val_group_set),
            }
        )
        print(
            f"[fold {fold_idx}] train={len(fold_train)} val={len(fold_val)} "
            f"test={len(test_samples)} -> {fold_path}"
        )

    manifest_path = output_dir / f"{prefix}_kfold_manifest.json"
    _write_json(manifest_path, manifest)
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
