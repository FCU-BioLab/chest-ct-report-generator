"""
Configuration loader utility.

Loads and validates pipeline configuration from YAML file.
Provides convenience functions to access common paths and settings.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Cache for loaded config
_config_cache: Optional[Dict[str, Any]] = None
_CONFIG_DIR = Path(__file__).resolve().parent
_PIPELINE_ROOT = _CONFIG_DIR.parent
_REPO_ROOT = _PIPELINE_ROOT.parent.parent
_PLACEHOLDER_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _as_posix(path: Path) -> str:
    return str(path).replace("\\", "/")


def _placeholder_context() -> Dict[str, str]:
    return {
        "CONFIG_DIR": _as_posix(_CONFIG_DIR),
        "PIPELINE_ROOT": _as_posix(_PIPELINE_ROOT),
        "REPO_ROOT": _as_posix(_REPO_ROOT),
        "DATASET_ROOT": _as_posix(_REPO_ROOT / "dataset"),
        "N8N_ROOT": _as_posix(_REPO_ROOT / "n8n"),
    }


def _expand_placeholders(value: str, context: Dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        return os.environ.get(key, context.get(key, match.group(0)))

    expanded = _PLACEHOLDER_PATTERN.sub(replace, value)
    return os.path.expandvars(expanded)


def _resolve_config_values(value: Any, context: Dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {k: _resolve_config_values(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_config_values(v, context) for v in value]
    if isinstance(value, str):
        return _expand_placeholders(value, context)
    return value


def _contains_placeholder(value: Any) -> bool:
    return isinstance(value, str) and "${" in value


def _choose_path(configured: Any, env_var: str, default: Path) -> str:
    env_value = os.environ.get(env_var)
    if env_value:
        return env_value.replace("\\", "/")
    if isinstance(configured, str) and configured and not _contains_placeholder(configured):
        return configured.replace("\\", "/")
    return _as_posix(default)


def _apply_default_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    datasets = config.setdefault("datasets", {})
    lndb = datasets.setdefault("lndb", {})
    msd_lung = datasets.setdefault("msd_lung", {})
    luna16 = datasets.setdefault("luna16", {})
    medsam2 = config.setdefault("medsam2", {})
    checkpoints = medsam2.setdefault("checkpoints", {})
    llm = config.setdefault("llm", {})
    lora_weights = llm.setdefault("lora_weights", {})
    outputs = config.setdefault("outputs", {})
    reports = config.setdefault("reports", {})
    dataset_split = config.get("dataset_split", {})

    config["project_root"] = _choose_path(config.get("project_root"), "PIPELINE_ROOT", _PIPELINE_ROOT)

    lndb["root"] = _choose_path(lndb.get("root"), "LNDB_ROOT", _REPO_ROOT / "dataset" / "LNDb")
    msd_lung["root"] = _choose_path(
        msd_lung.get("root"),
        "MSD_LUNG_ROOT",
        _REPO_ROOT / "dataset" / "MSD" / "Task06_Lung",
    )
    luna16["root"] = _choose_path(luna16.get("root"), "LUNA16_ROOT", _REPO_ROOT / "dataset" / "LUNA16")

    medsam2["root"] = _choose_path(medsam2.get("root"), "MEDSAM2_ROOT", _REPO_ROOT / "segmentation" / "MedSAM2")
    checkpoints["pretrained"] = _choose_path(
        checkpoints.get("pretrained"),
        "MEDSAM2_PRETRAINED_CKPT",
        Path(medsam2["root"]) / "checkpoints" / "MedSAM2_CTLesion.pt",
    )
    checkpoints["finetuned"] = _choose_path(
        checkpoints.get("finetuned"),
        "MEDSAM2_FINETUNED_CKPT",
        _PIPELINE_ROOT / "segmentation" / "MedSAM2_best_model.pth",
    )
    checkpoints["latest"] = _choose_path(
        checkpoints.get("latest"),
        "MEDSAM2_LATEST_CKPT",
        Path(medsam2["root"]) / "checkpoints" / "MedSAM2_latest.pt",
    )

    lora_weights["base_dir"] = _choose_path(
        lora_weights.get("base_dir"),
        "LLM_LORA_BASE_DIR",
        _PIPELINE_ROOT / "assets" / "models" / "lora_ct_report",
    )
    lora_weights["latest"] = _choose_path(
        lora_weights.get("latest"),
        "LLM_LORA_LATEST",
        Path(lora_weights["base_dir"]) / "latest",
    )

    outputs["root"] = _choose_path(outputs.get("root"), "PIPELINE_OUTPUT_ROOT", _PIPELINE_ROOT / "outputs")
    outputs["processed_data"] = _choose_path(
        outputs.get("processed_data"),
        "PIPELINE_PROCESSED_DATA_DIR",
        _PIPELINE_ROOT / "processed_data",
    )
    outputs["segmentation_results"] = _choose_path(
        outputs.get("segmentation_results"),
        "SEGMENTATION_RESULTS_DIR",
        _REPO_ROOT / "segmentation" / "result",
    )
    outputs["training_data"] = _choose_path(
        outputs.get("training_data"),
        "PIPELINE_TRAINING_DATA_DIR",
        _PIPELINE_ROOT / "assets" / "data",
    )

    if isinstance(dataset_split, dict):
        dataset_split["saved_split"] = _choose_path(
            dataset_split.get("saved_split"),
            "DATASET_SPLIT_PATH",
            _REPO_ROOT / "segmentation" / "result" / "dataset_split.json",
        )

    reports["raw_reports"] = _choose_path(
        reports.get("raw_reports"),
        "RAW_REPORTS_DIR",
        _REPO_ROOT / "report_data" / "splited_reports",
    )
    reports["processed_reports"] = _choose_path(
        reports.get("processed_reports"),
        "PROCESSED_REPORTS_PATH",
        _PIPELINE_ROOT / "assets" / "data" / "finetune_real_reports.jsonl",
    )

    if "checkpoint" in medsam2:
        medsam2["checkpoint"] = _choose_path(
            medsam2.get("checkpoint"),
            "MEDSAM2_FINETUNED_CKPT",
            _PIPELINE_ROOT / "segmentation" / "MedSAM2_best_model.pth",
        )
    if "lora_path" in llm:
        llm["lora_path"] = _choose_path(
            llm.get("lora_path"),
            "LLM_LORA_LATEST",
            _PIPELINE_ROOT / "assets" / "models" / "lora_ct_report" / "latest",
        )
    if "lndb_root" in config:
        config["lndb_root"] = _choose_path(config.get("lndb_root"), "LNDB_ROOT", _REPO_ROOT / "dataset" / "LNDb")
    if "output_dir" in config:
        config["output_dir"] = _choose_path(
            config.get("output_dir"),
            "PIPELINE_OUTPUT_ROOT",
            _PIPELINE_ROOT / "outputs",
        )
    if "processed_data_dir" in config:
        config["processed_data_dir"] = _choose_path(
            config.get("processed_data_dir"),
            "PIPELINE_PROCESSED_DATA_DIR",
            _PIPELINE_ROOT / "processed_data",
        )
    if "dataset_split" in config and isinstance(config["dataset_split"], str):
        config["dataset_split"] = _choose_path(
            config.get("dataset_split"),
            "DATASET_SPLIT_PATH",
            _REPO_ROOT / "segmentation" / "result" / "dataset_split.json",
        )

    return config


def load_config(config_path: str = None, force_reload: bool = False) -> Dict[str, Any]:
    """
    Load pipeline configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses default location.
        force_reload: If True, reload config even if cached.

    Returns:
        Configuration dictionary
    """
    global _config_cache
    using_default_config = config_path is None

    if _config_cache is not None and not force_reload and using_default_config:
        return _config_cache

    if using_default_config:
        config_path = _CONFIG_DIR / "config.yaml"

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    config = _resolve_config_values(config, _placeholder_context())
    config = _apply_default_paths(config)

    if using_default_config:
        _config_cache = config

    return config


# =============================================================================
# Convenience functions for common paths
# =============================================================================

def get_project_root(config: Dict[str, Any] = None) -> Path:
    """Get the project root directory."""
    if config is None:
        config = load_config()
    default_root = _PIPELINE_ROOT
    return Path(config.get("project_root", _as_posix(default_root)))


def get_lndb_root(config: Dict[str, Any] = None) -> Path:
    """Get LNDb dataset root directory."""
    if config is None:
        config = load_config()
    return Path(config["datasets"]["lndb"]["root"])


def get_msd_root(config: Dict[str, Any] = None) -> Path:
    """Get MSD Lung dataset root directory."""
    if config is None:
        config = load_config()
    return Path(config["datasets"]["msd_lung"]["root"])


def get_luna16_root(config: Dict[str, Any] = None) -> Path:
    """Get LUNA16 dataset root directory."""
    if config is None:
        config = load_config()
    return Path(config["datasets"]["luna16"]["root"])


# =============================================================================
# MedSAM2 configuration
# =============================================================================

def get_medsam2_config(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Get MedSAM2 model configuration."""
    if config is None:
        config = load_config()
    return config.get("medsam2", {})


def get_medsam2_root(config: Dict[str, Any] = None) -> Path:
    """Get MedSAM2 root directory."""
    if config is None:
        config = load_config()
    return Path(config["medsam2"]["root"])


def get_medsam2_checkpoint(config: Dict[str, Any] = None, checkpoint_type: str = None) -> Path:
    """
    Get MedSAM2 checkpoint path.

    Args:
        config: Configuration dictionary
        checkpoint_type: One of 'pretrained', 'finetuned', 'latest', or None for default

    Returns:
        Path to checkpoint file
    """
    if config is None:
        config = load_config()

    medsam2_config = config.get("medsam2", {})
    checkpoints = medsam2_config.get("checkpoints", {})

    if checkpoint_type is None:
        checkpoint_type = medsam2_config.get("default_checkpoint", "finetuned")

    checkpoint_path = checkpoints.get(checkpoint_type)
    if checkpoint_path:
        return Path(checkpoint_path)

    for ckpt_type in ["finetuned", "pretrained", "latest"]:
        if ckpt_type in checkpoints:
            return Path(checkpoints[ckpt_type])

    raise FileNotFoundError("No MedSAM2 checkpoint found in config")


# =============================================================================
# LLM configuration
# =============================================================================

def get_llm_config(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Get LLM model configuration."""
    if config is None:
        config = load_config()
    return config.get("llm", {})


# =============================================================================
# Output paths
# =============================================================================

def get_output_paths(config: Dict[str, Any] = None) -> Dict[str, Path]:
    """Get all output directory paths."""
    if config is None:
        config = load_config()

    outputs = config.get("outputs", {})
    return {
        "root": Path(outputs.get("root", "outputs")),
        "processed_data": Path(outputs.get("processed_data", "processed_data")),
        "segmentation_results": Path(outputs.get("segmentation_results", "segmentation/result")),
        "training_data": Path(outputs.get("training_data", "data")),
    }


def get_reports_config(config: Dict[str, Any] = None) -> Dict[str, Path]:
    """Get report paths configuration."""
    if config is None:
        config = load_config()

    reports = config.get("reports", {})
    return {
        "raw_reports": Path(reports.get("raw_reports", "")),
        "processed_reports": Path(reports.get("processed_reports", "")),
    }


# =============================================================================
# Training configuration
# =============================================================================

def get_training_config(config: Dict[str, Any] = None, model_type: str = "segmentation") -> Dict[str, Any]:
    """
    Get training configuration for a specific model type.

    Args:
        config: Configuration dictionary
        model_type: 'segmentation' or 'llm_finetune'

    Returns:
        Training configuration dictionary
    """
    if config is None:
        config = load_config()

    training = config.get("training", {})
    return training.get(model_type, {})


def get_preprocessing_config(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Get preprocessing configuration."""
    if config is None:
        config = load_config()
    return config.get("preprocessing", {})


def get_ct_window(config: Dict[str, Any] = None, window_type: str = "lung") -> Dict[str, int]:
    """
    Get CT window settings.

    Args:
        config: Configuration dictionary
        window_type: 'lung' or 'mediastinum'

    Returns:
        Dictionary with 'center' and 'width' keys
    """
    if config is None:
        config = load_config()

    preprocessing = config.get("preprocessing", {})
    window = preprocessing.get("window", {})
    return window.get(window_type, {"center": -600, "width": 1500})


# =============================================================================
# Path validation
# =============================================================================

def validate_paths(config: Dict[str, Any] = None) -> bool:
    """
    Validate that required paths exist.

    Args:
        config: Configuration dictionary

    Returns:
        True if all paths are valid
    """
    if config is None:
        config = load_config()

    required_paths = {
        "MedSAM2 root": get_medsam2_root(config),
        "LNDb dataset": get_lndb_root(config),
    }

    all_valid = True

    for description, path in required_paths.items():
        if path.exists():
            print(f"[OK] {description}: {path}")
        else:
            print(f"[MISSING] {description}: {path}")
            all_valid = False

    optional_paths = {
        "MSD Lung dataset": get_msd_root(config),
        "Output directory": get_output_paths(config)["root"],
    }

    for description, path in optional_paths.items():
        if path.exists():
            print(f"[OK] {description}: {path}")
        else:
            print(f"[OPTIONAL] Missing {description}: {path}")

    return all_valid


def get_device(config: Dict[str, Any] = None) -> str:
    """Get device setting (cuda or cpu)."""
    if config is None:
        config = load_config()
    return config.get("device", "cuda")


# =============================================================================
# Main entry point for testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CT Report Pipeline - Configuration Test")
    print("=" * 60)

    try:
        config = load_config()
        print("Configuration loaded successfully.\n")
    except Exception as e:
        print(f"Failed to load config: {e}")
        raise SystemExit(1)

    print("Project Root:", get_project_root(config))
    print("\n--- Datasets ---")
    print(f"  LNDb:   {get_lndb_root(config)}")
    print(f"  MSD:    {get_msd_root(config)}")
    print(f"  LUNA16: {get_luna16_root(config)}")

    print("\n--- MedSAM2 ---")
    medsam2 = get_medsam2_config(config)
    print(f"  Root:   {medsam2.get('root')}")
    print(f"  Config: {medsam2.get('config')}")
    try:
        print(f"  Checkpoint: {get_medsam2_checkpoint(config)}")
    except FileNotFoundError as e:
        print(f"  Checkpoint: {e}")

    print("\n--- LLM ---")
    llm = get_llm_config(config)
    print(f"  Model: {llm.get('model_name')}")
    print(f"  LoRA:  {llm.get('lora_weights', {}).get('latest', 'N/A')}")

    print("\n--- Outputs ---")
    outputs = get_output_paths(config)
    for name, path in outputs.items():
        print(f"  {name}: {path}")

    print("\n--- Validating Paths ---")
    validate_paths(config)

    print("\n" + "=" * 60)
