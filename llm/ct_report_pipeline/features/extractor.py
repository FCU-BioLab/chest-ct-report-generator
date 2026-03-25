"""
Feature Extraction Module

Implements Local Feature Decoupling (LFD) and Global Feature extraction
as described in the Reg2RG paper.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict

# Add parent directory to path for config access
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from config import load_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False


def _get_feature_dims():
    """Get default feature dimensions from config."""
    if _CONFIG_AVAILABLE:
        try:
            config = load_config()
            features = config.get('features', {})
            return features.get('encoder_dim', 512), features.get('llm_dim', 4096)
        except Exception:
            pass
    return 512, 4096


class Simple3DCNN(nn.Module):
    """Simple 3D CNN encoder for volume/mask encoding."""
    
    def __init__(self, in_channels: int = 1, feature_dim: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((4, 4, 4)),
            
            nn.Flatten(),
            nn.Linear(256 * 4 * 4 * 4, feature_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)


class FeatureAdapter(nn.Module):
    """Adapter to project encoder features to LLM embedding space."""
    
    def __init__(self, encoder_dim: int = 512, llm_dim: int = 4096):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(encoder_dim, llm_dim * 2),
            nn.GELU(),
            nn.Linear(llm_dim * 2, llm_dim)
        )
    
    def forward(self, x):
        return self.projection(x)


class FeatureExtractor:
    """
    Extract local and global features from CT volumes and region masks.
    
    Implements:
    - Texture features: high-resolution local features from cropped regions
    - Geometry features: spatial information from full masks
    - Global features: overall CT volume features
    """
    
    def __init__(
        self,
        encoder_dim: int = None,
        llm_dim: int = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize feature extractor.
        
        Args:
            encoder_dim: Dimension of encoder output (default from config)
            llm_dim: Dimension of LLM embeddings (default from config)
            device: Computing device
        """
        # Get defaults from config if not provided
        default_encoder_dim, default_llm_dim = _get_feature_dims()
        
        self.device = device
        self.encoder_dim = encoder_dim or default_encoder_dim
        self.llm_dim = llm_dim or default_llm_dim

        # Volume encoder for texture features
        self.volume_encoder = Simple3DCNN(in_channels=1, feature_dim=self.encoder_dim).to(device)
        
        # Mask encoder for geometry features
        self.mask_encoder = Simple3DCNN(in_channels=1, feature_dim=self.encoder_dim).to(device)
        
        # Shared adapter
        self.adapter = FeatureAdapter(self.encoder_dim, self.llm_dim).to(device)

        
        # Set to eval mode (for inference)
        self.volume_encoder.eval()
        self.mask_encoder.eval()
        self.adapter.eval()
    
    def extract_texture_feature(
        self,
        ct_volume: np.ndarray,
        mask: np.ndarray
    ) -> torch.Tensor:
        """
        Extract texture feature from masked and cropped region.
        
        Args:
            ct_volume: Full CT volume (D, H, W)
            mask: Binary mask for the region (D, H, W)
        
        Returns:
            Texture feature tensor (llm_dim,)
        """
        # Apply mask
        masked_volume = ct_volume * mask
        
        # Find bounding box
        coords = np.argwhere(mask > 0)
        if len(coords) == 0:
            # Empty mask, return zero feature
            return torch.zeros(self.llm_dim, device=self.device)
        
        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0)
        
        # Crop to bounding box
        cropped = masked_volume[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
        
        # Convert to tensor and add batch + channel dims
        cropped_tensor = torch.from_numpy(cropped).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Encode
        with torch.no_grad():
            encoded = self.volume_encoder(cropped_tensor)
            texture_feat = self.adapter(encoded)
        
        return texture_feat.squeeze(0)
    
    def extract_geometry_feature(
        self,
        mask: np.ndarray
    ) -> torch.Tensor:
        """
        Extract geometry feature from full (uncropped) mask.
        
        Args:
            mask: Binary mask (D, H, W)
        
        Returns:
            Geometry feature tensor (llm_dim,)
        """
        # Convert to tensor
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Encode
        with torch.no_grad():
            encoded = self.mask_encoder(mask_tensor)
            # Project to LLM space (separate projection for geometry)
            geometry_feat = nn.Linear(self.encoder_dim, self.llm_dim).to(self.device)(encoded)
        
        return geometry_feat.squeeze(0)
    
    def extract_local_feature(
        self,
        ct_volume: np.ndarray,
        mask: np.ndarray
    ) -> torch.Tensor:
        """
        Extract combined local feature (texture + geometry).
        
        Args:
            ct_volume: Full CT volume
            mask: Binary mask for the region
        
        Returns:
            Combined local feature (llm_dim * 2,)
        """
        texture_feat = self.extract_texture_feature(ct_volume, mask)
        geometry_feat = self.extract_geometry_feature(mask)
        
        # Concatenate
        local_feat = torch.cat([texture_feat, geometry_feat], dim=0)
        return local_feat
    
    def extract_global_feature(
        self,
        ct_volume: np.ndarray
    ) -> torch.Tensor:
        """
        Extract global feature from entire CT volume.
        
        Args:
            ct_volume: Full CT volume (D, H, W)
        
        Returns:
            Global feature tensor (llm_dim,)
        """
        # Convert to tensor
        volume_tensor = torch.from_numpy(ct_volume).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Encode using shared volume encoder and adapter
        with torch.no_grad():
            encoded = self.volume_encoder(volume_tensor)
            global_feat = self.adapter(encoded)
        
        return global_feat.squeeze(0)
    
    def extract_all_features(
        self,
        ct_volume: np.ndarray,
        masks: list[np.ndarray]
    ) -> Dict[str, torch.Tensor]:
        """
        Extract all features for a CT scan with multiple regions.
        
        Args:
            ct_volume: Full CT volume
            masks: List of binary masks, one per region
        
        Returns:
            Dictionary with:
                - 'global': global feature
                - 'local': list of local features (one per region)
        """
        global_feat = self.extract_global_feature(ct_volume)
        
        local_feats = []
        for mask in masks:
            local_feat = self.extract_local_feature(ct_volume, mask)
            local_feats.append(local_feat)
        
        return {
            'global': global_feat,
            'local': local_feats
        }
    
    def save_weights(self, path: str):
        """Save model weights."""
        torch.save({
            'volume_encoder': self.volume_encoder.state_dict(),
            'mask_encoder': self.mask_encoder.state_dict(),
            'adapter': self.adapter.state_dict()
        }, path)
    
    def load_weights(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.volume_encoder.load_state_dict(checkpoint['volume_encoder'])
        self.mask_encoder.load_state_dict(checkpoint['mask_encoder'])
        self.adapter.load_state_dict(checkpoint['adapter'])
