"""
Configuration loader utility.

Loads and validates pipeline configuration from YAML file.
Provides convenience functions to access common paths and settings.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# Cache for loaded config
_config_cache: Optional[Dict[str, Any]] = None


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
    
    if _config_cache is not None and not force_reload and config_path is None:
        return _config_cache
    
    if config_path is None:
        # Default to config/config.yaml relative to this file
        config_path = Path(__file__).parent / "config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if config_path is None:
        _config_cache = config
    
    return config


# =============================================================================
# Convenience functions for common paths
# =============================================================================

def get_project_root(config: Dict[str, Any] = None) -> Path:
    """Get the project root directory."""
    if config is None:
        config = load_config()
    default_root = Path(__file__).resolve().parent.parent
    return Path(config.get('project_root', str(default_root).replace('\\', '/')))


def get_lndb_root(config: Dict[str, Any] = None) -> Path:
    """Get LNDb dataset root directory."""
    if config is None:
        config = load_config()
    return Path(config['datasets']['lndb']['root'])


def get_msd_root(config: Dict[str, Any] = None) -> Path:
    """Get MSD Lung dataset root directory."""
    if config is None:
        config = load_config()
    return Path(config['datasets']['msd_lung']['root'])


def get_luna16_root(config: Dict[str, Any] = None) -> Path:
    """Get LUNA16 dataset root directory."""
    if config is None:
        config = load_config()
    return Path(config['datasets']['luna16']['root'])


# =============================================================================
# MedSAM2 configuration
# =============================================================================

def get_medsam2_config(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Get MedSAM2 model configuration."""
    if config is None:
        config = load_config()
    return config.get('medsam2', {})


def get_medsam2_root(config: Dict[str, Any] = None) -> Path:
    """Get MedSAM2 root directory."""
    if config is None:
        config = load_config()
    return Path(config['medsam2']['root'])


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
    
    medsam2_config = config.get('medsam2', {})
    checkpoints = medsam2_config.get('checkpoints', {})
    
    if checkpoint_type is None:
        checkpoint_type = medsam2_config.get('default_checkpoint', 'finetuned')
    
    checkpoint_path = checkpoints.get(checkpoint_type)
    if checkpoint_path:
        return Path(checkpoint_path)
    
    # Fallback: try to find any available checkpoint
    for ckpt_type in ['finetuned', 'pretrained', 'latest']:
        if ckpt_type in checkpoints:
            return Path(checkpoints[ckpt_type])
    
    raise FileNotFoundError(f"No MedSAM2 checkpoint found in config")


# =============================================================================
# LLM configuration
# =============================================================================

def get_llm_config(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Get LLM model configuration."""
    if config is None:
        config = load_config()
    return config.get('llm', {})


# =============================================================================
# Output paths
# =============================================================================

def get_output_paths(config: Dict[str, Any] = None) -> Dict[str, Path]:
    """Get all output directory paths."""
    if config is None:
        config = load_config()
    
    outputs = config.get('outputs', {})
    return {
        'root': Path(outputs.get('root', 'outputs')),
        'processed_data': Path(outputs.get('processed_data', 'processed_data')),
        'segmentation_results': Path(outputs.get('segmentation_results', 'segmentation/result')),
        'training_data': Path(outputs.get('training_data', 'data')),
    }


def get_reports_config(config: Dict[str, Any] = None) -> Dict[str, Path]:
    """Get report paths configuration."""
    if config is None:
        config = load_config()
    
    reports = config.get('reports', {})
    return {
        'raw_reports': Path(reports.get('raw_reports', '')),
        'processed_reports': Path(reports.get('processed_reports', '')),
    }


# =============================================================================
# Training configuration
# =============================================================================

def get_training_config(config: Dict[str, Any] = None, model_type: str = 'segmentation') -> Dict[str, Any]:
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
    
    training = config.get('training', {})
    return training.get(model_type, {})


def get_preprocessing_config(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Get preprocessing configuration."""
    if config is None:
        config = load_config()
    return config.get('preprocessing', {})


def get_ct_window(config: Dict[str, Any] = None, window_type: str = 'lung') -> Dict[str, int]:
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
    
    preprocessing = config.get('preprocessing', {})
    window = preprocessing.get('window', {})
    return window.get(window_type, {'center': -600, 'width': 1500})


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
        'MedSAM2 root': get_medsam2_root(config),
        'LNDb dataset': get_lndb_root(config),
    }
    
    all_valid = True
    
    for description, path in required_paths.items():
        if path.exists():
            print(f"??{description}: {path}")
        else:
            print(f"??{description} not found: {path}")
            all_valid = False
    
    # Check optional paths
    optional_paths = {
        'MSD Lung dataset': get_msd_root(config),
        'Output directory': get_output_paths(config)['root'],
    }
    
    for description, path in optional_paths.items():
        if path.exists():
            print(f"??{description}: {path}")
        else:
            print(f"??{description} not found (optional): {path}")
    
    return all_valid


def get_device(config: Dict[str, Any] = None) -> str:
    """Get device setting (cuda or cpu)."""
    if config is None:
        config = load_config()
    return config.get('device', 'cuda')


# =============================================================================
# Main entry point for testing
# =============================================================================

if __name__ == "__main__":
    # Test configuration loading
    print("=" * 60)
    print("CT Report Pipeline - Configuration Test")
    print("=" * 60)
    
    try:
        config = load_config()
        print("??Configuration loaded successfully!\n")
    except Exception as e:
        print(f"??Failed to load config: {e}")
        exit(1)
    
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

