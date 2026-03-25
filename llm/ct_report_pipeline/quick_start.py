"""
Quick Start Script

Test configuration and verify all paths are set up correctly.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    load_config,
    validate_paths,
    get_medsam2_checkpoint,
    get_lndb_root,
    get_msd_root,
    get_medsam2_root,
    get_output_paths,
    get_llm_config,
)


def _exists_mark(path: Path) -> str:
    return "[OK]" if path.exists() else "[MISSING]"


def main() -> int:
    print("=" * 60)
    print("CT Report Pipeline - Configuration Test")
    print("=" * 60)

    print("\n1. Loading configuration...")
    try:
        config = load_config()
        print("   [OK] Configuration loaded successfully (config/config.yaml)")
    except Exception as e:
        print(f"   [ERROR] Error loading configuration: {e}")
        return 1

    print("\n2. Dataset paths:")
    try:
        lndb_root = get_lndb_root(config)
        print(f"   LNDb dataset:      {lndb_root}")
        print(f"      exists: {_exists_mark(lndb_root)}")
    except Exception as e:
        print(f"   LNDb dataset:      Error - {e}")

    try:
        msd_root = get_msd_root(config)
        print(f"   MSD Lung dataset:  {msd_root}")
        suffix = "" if msd_root.exists() else " (optional)"
        print(f"      exists: {_exists_mark(msd_root)}{suffix}")
    except Exception as e:
        print(f"   MSD Lung dataset:  Error - {e}")

    print("\n3. MedSAM2 configuration:")
    try:
        medsam2_root = get_medsam2_root(config)
        print(f"   Root directory:    {medsam2_root}")
        print(f"      exists: {_exists_mark(medsam2_root)}")
    except Exception as e:
        print(f"   Root directory:    Error - {e}")

    try:
        checkpoint_path = get_medsam2_checkpoint(config)
        print(f"   Checkpoint:        {checkpoint_path}")
        if checkpoint_path.exists():
            size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
            print(f"      exists: [OK] ({size_mb:.1f} MB)")
        else:
            print("      exists: [MISSING]")
    except FileNotFoundError as e:
        print(f"   Checkpoint:        {e}")

    print("\n4. Output paths:")
    outputs = get_output_paths(config)
    for name, path in outputs.items():
        print(f"   {name}: {path} [{_exists_mark(path)}]")

    print("\n5. LLM configuration:")
    llm_config = get_llm_config(config)
    print(f"   Model:       {llm_config.get('model_name', 'N/A')}")
    lora_weights = llm_config.get("lora_weights", {})
    print(f"   LoRA latest: {lora_weights.get('latest', 'N/A')}")

    print("\n6. Checking LNDb dataset structure...")
    try:
        lndb_root = get_lndb_root(config)
        if lndb_root.exists():
            data_dirs = list(lndb_root.glob("data*"))
            trainset_dir = lndb_root / "trainset_csv"
            masks_dir = lndb_root / "masks"

            print(f"   Found {len(data_dirs)} data directories")
            print(
                "   "
                + ("[OK] Found trainset_csv directory" if trainset_dir.exists() else "[MISSING] trainset_csv directory not found")
            )
            print(
                "   "
                + ("[OK] Found masks directory" if masks_dir.exists() else "[MISSING] masks directory not found")
            )
        else:
            print("   [MISSING] LNDb root directory not found")
    except Exception as e:
        print(f"   [ERROR] LNDb structure check failed: {e}")

    print("\n7. Full path validation:")
    validate_paths(config)

    print("\n" + "=" * 60)
    print("Configuration test complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run: python scripts/prepare_dataset.py")
    print("2. Run: python scripts/interactive_segmentation.py")
    print("3. Run: python scripts/finetune_llama.py (optional)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
