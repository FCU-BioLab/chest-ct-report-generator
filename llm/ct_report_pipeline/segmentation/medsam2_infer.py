"""
MedSAM2 Segmentation Wrapper (Fixed Version)

This module provides a wrapper around MedSAM2 for medical image segmentation
using point or box prompts.

NOTE: This module MUST be run from the MedSAM2 directory or with MedSAM2 properly installed.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import sys
import os

# Add parent directory to path for config access
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from config import load_config, get_medsam2_root, get_ct_window
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False


class MedSAM2Segmenter:
    """
    Wrapper for MedSAM2 model to perform lung nodule segmentation.
    
    Supports:
    - Point prompts (click on lesion)
    - Bounding box prompts
    - 3D volume segmentation with propagation
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        medsam2_root: str = None,
        config_file: str = "sam2.1_hiera_t512.yaml",
        device: str = None
    ):
        """
        Initialize MedSAM2 segmenter.
        
        Args:
            checkpoint_path: Path to MedSAM2 checkpoint file
            medsam2_root: Path to MedSAM2 repository root
            config_file: Config file name (in sam2/configs/)
            device: Device for inference (cuda/cpu)
        """
        import torch
        
        self.checkpoint_path = Path(checkpoint_path)
        self.config_file = config_file
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = None
        
        # Set MedSAM2 root - try config first, then use local copy
        if medsam2_root is None:
            if _CONFIG_AVAILABLE:
                try:
                    config = load_config()
                    self.medsam2_root = get_medsam2_root(config)
                except Exception:
                    # Fallback to local MedSAM2 in segmentation folder
                    self.medsam2_root = Path(__file__).parent / "MedSAM2"
            else:
                # Default to local MedSAM2 in segmentation folder
                self.medsam2_root = Path(__file__).parent / "MedSAM2"
        else:
            self.medsam2_root = Path(medsam2_root)
        
        # Verify checkpoint exists
        if not self.checkpoint_path.exists():
            print(f"Warning: checkpoint not found at {checkpoint_path}")
    
    def load_model(self):
        """Load MedSAM2 model from checkpoint."""
        import torch
        
        # Save original state
        original_cwd = os.getcwd()
        original_path = sys.path.copy()
        
        try:
            # Change to MedSAM2 directory (CRITICAL for proper imports)
            os.chdir(str(self.medsam2_root))
            
            # Add MedSAM2 to path FIRST
            medsam2_path = str(self.medsam2_root)
            if medsam2_path not in sys.path:
                sys.path.insert(0, medsam2_path)
            
            print(f"Loading MedSAM2 model from {self.checkpoint_path}")
            print(f"  Working directory: {os.getcwd()}")
            
            # Force import sam2 modules BEFORE Hydra instantiate
            # This ensures they're in sys.modules when Hydra tries to locate them
            import sam2
            import sam2.modeling
            import sam2.modeling.backbones
            import sam2.modeling.backbones.hieradet
            import sam2.modeling.backbones.image_encoder
            import sam2.modeling.sam
            import sam2.modeling.sam.mask_decoder
            import sam2.modeling.sam.prompt_encoder
            import sam2.modeling.sam.transformer
            import sam2.modeling.memory_attention
            import sam2.modeling.memory_encoder
            import sam2.modeling.position_encoding
            import sam2.modeling.sam2_base
            print("  [OK] Pre-imported sam2 modules")
            
            # Clear any existing Hydra configuration
            from hydra.core.global_hydra import GlobalHydra
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
            
            # Initialize Hydra with absolute path to configs directory
            from hydra import initialize_config_dir
            config_dir = os.path.abspath(os.path.join(str(self.medsam2_root), "sam2", "configs"))
            print(f"  Config dir: {config_dir}")
            initialize_config_dir(config_dir=config_dir, version_base="1.2")
            
            # Import necessary modules
            from sam2.build_sam import build_sam2_video_predictor_npz
            from hydra import compose
            from hydra.utils import instantiate
            from omegaconf import OmegaConf
            
            # Config name - just the filename without .yaml extension
            config_name = self.config_file.replace('.yaml', '')
            print(f"  Config name: {config_name}")
            
            # Build model with custom checkpoint loading
            hydra_overrides = [
                "++model._target_=sam2.sam2_video_predictor_npz.SAM2VideoPredictorNPZ",
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
                "++model.binarize_mask_from_pts_for_mem_enc=true",
                "++model.fill_hole_area=8",
            ]
            
            cfg = compose(config_name=config_name, overrides=hydra_overrides)
            OmegaConf.resolve(cfg)
            model = instantiate(cfg.model, _recursive_=True)
            
            # Custom checkpoint loading that handles fine-tuned format
            print(f"  Loading checkpoint...")
            sd = torch.load(str(self.checkpoint_path), map_location="cpu", weights_only=True)
            
            # Handle different checkpoint formats
            if "model" in sd:
                state_dict = sd["model"]
            elif "model_state_dict" in sd:
                state_dict = sd["model_state_dict"]
                print("  [OK] Using fine-tuned checkpoint format (model_state_dict)")
                print(f"    Best val dice: {sd.get('best_val_dice', 'N/A')}")
                print(f"    Epoch: {sd.get('epoch', 'N/A')}")
            else:
                # Assume it's just the state dict directly
                state_dict = sd
            
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"  [WARN] Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                print(f"  [WARN] Unexpected keys: {len(unexpected_keys)}")
            
            model = model.to(self.device)
            model.eval()
            
            self.predictor = model
            
            print("OK: MedSAM2 model loaded successfully")
            print(f"  Device: {self.device}")
            
        except Exception as e:
            print(f"ERROR: MedSAM2 loading failed: {e}")
            raise
        finally:
            # Restore original state
            os.chdir(original_cwd)
    
    def preprocess_volume(self, ct_volume: np.ndarray, 
                          window_center: float = -600, 
                          window_width: float = 1500) -> np.ndarray:
        """
        Preprocess CT volume for MedSAM2.
        
        Args:
            ct_volume: 3D CT array (D, H, W)
            window_center: CT window center for lung
            window_width: CT window width
        
        Returns:
            Preprocessed volume (D, 3, 512, 512) normalized to [0, 1]
        """
        import torch
        from PIL import Image
        
        # Apply window/level
        lower = window_center - window_width / 2
        upper = window_center + window_width / 2
        
        volume = np.clip(ct_volume, lower, upper)
        volume = (volume - lower) / (upper - lower) * 255.0
        volume = volume.astype(np.uint8)
        
        # Resize to 512x512 and convert to RGB
        d, h, w = volume.shape
        resized = np.zeros((d, 3, 512, 512), dtype=np.float32)
        
        for i in range(d):
            img = Image.fromarray(volume[i])
            img_rgb = img.convert("RGB")
            img_resized = img_rgb.resize((512, 512))
            img_array = np.array(img_resized).transpose(2, 0, 1) / 255.0
            resized[i] = img_array
        
        # Normalize with ImageNet stats
        resized = torch.from_numpy(resized).to(self.device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        resized = (resized - mean) / std
        
        return resized, h, w
    
    def segment_from_points(
        self,
        ct_volume: np.ndarray,
        point_prompts: List[Dict],
        propagate: bool = True
    ) -> List[np.ndarray]:
        """
        Segment using point prompts.
        
        Args:
            ct_volume: 3D CT volume (D, H, W)
            point_prompts: List of {'coords': (z, y, x), 'label': 0 or 1}
            propagate: Whether to propagate through volume
        
        Returns:
            List of 3D masks
        """
        import torch
        
        if self.predictor is None:
            self.load_model()
        
        # Preprocess
        img_tensor, orig_h, orig_w = self.preprocess_volume(ct_volume)
        
        # Group points by z-slice
        points_by_slice = {}
        for p in point_prompts:
            z = p['coords'][0]
            if z not in points_by_slice:
                points_by_slice[z] = {'points': [], 'labels': []}
            
            # Scale coordinates to 512x512
            y = p['coords'][1] * 512 / orig_h
            x = p['coords'][2] * 512 / orig_w
            points_by_slice[z]['points'].append([x, y])
            points_by_slice[z]['labels'].append(p['label'])
        
        masks = []
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for z, prompts in points_by_slice.items():
                # Initialize inference state
                inference_state = self.predictor.init_state(
                    img_tensor, 
                    orig_h, 
                    orig_w
                )
                
                # Add points
                points = np.array(prompts['points'])
                labels = np.array(prompts['labels'])
                
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=z,
                    obj_id=1,
                    points=points,
                    labels=labels
                )
                
                # Create 3D mask
                mask_3d = np.zeros(ct_volume.shape, dtype=np.uint8)
                
                if propagate:
                    # Forward propagation
                    for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
                        mask_slice = (out_mask_logits[0] > 0.0).cpu().numpy()[0]
                        mask_3d[out_frame_idx] = mask_slice
                    
                    self.predictor.reset_state(inference_state)
                    
                    # Re-add points
                    _, _, _ = self.predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=z,
                        obj_id=1,
                        points=points,
                        labels=labels
                    )
                    
                    # Backward propagation
                    for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state, reverse=True):
                        mask_slice = (out_mask_logits[0] > 0.0).cpu().numpy()[0]
                        mask_3d[out_frame_idx] = np.maximum(mask_3d[out_frame_idx], mask_slice)
                else:
                    # Single slice only
                    mask_3d[z] = (out_mask_logits[0] > 0.0).cpu().numpy()[0]
                
                self.predictor.reset_state(inference_state)
                masks.append(mask_3d)
        
        return masks
    
    def segment_from_boxes(
        self,
        ct_volume: np.ndarray,
        bounding_boxes: List[Dict],
        propagate: bool = True
    ) -> List[np.ndarray]:
        """
        Segment using bounding box prompts.
        
        Args:
            ct_volume: 3D CT volume (D, H, W)
            bounding_boxes: List of boxes with 'x_min', 'x_max', 'y_min', 'y_max', 'z_center'
            propagate: Whether to propagate through volume
        
        Returns:
            List of 3D masks
        """
        import torch
        
        if self.predictor is None:
            self.load_model()
        
        # Preprocess
        img_tensor, orig_h, orig_w = self.preprocess_volume(ct_volume)
        
        masks = []
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for box in bounding_boxes:
                # Scale box to 512x512
                x_min = box['x_min'] * 512 / orig_w
                x_max = box['x_max'] * 512 / orig_w
                y_min = box['y_min'] * 512 / orig_h
                y_max = box['y_max'] * 512 / orig_h
                z = box.get('z_center', ct_volume.shape[0] // 2)
                
                bbox = np.array([x_min, y_min, x_max, y_max])
                
                # Initialize inference state
                inference_state = self.predictor.init_state(
                    img_tensor,
                    orig_h,
                    orig_w
                )
                
                # Add box
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=z,
                    obj_id=1,
                    box=bbox
                )
                
                # Create 3D mask
                mask_3d = np.zeros(ct_volume.shape, dtype=np.uint8)
                
                if propagate:
                    # Forward
                    for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
                        mask_slice = (out_mask_logits[0] > 0.0).cpu().numpy()[0]
                        mask_3d[out_frame_idx] = mask_slice
                    
                    self.predictor.reset_state(inference_state)
                    
                    # Re-add box
                    _, _, _ = self.predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=z,
                        obj_id=1,
                        box=bbox
                    )
                    
                    # Backward
                    for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state, reverse=True):
                        mask_slice = (out_mask_logits[0] > 0.0).cpu().numpy()[0]
                        mask_3d[out_frame_idx] = np.maximum(mask_3d[out_frame_idx], mask_slice)
                else:
                    mask_3d[z] = (out_mask_logits[0] > 0.0).cpu().numpy()[0]
                
                self.predictor.reset_state(inference_state)
                masks.append(mask_3d)
        
        return masks
    
    @staticmethod
    def load_ct_volume(ct_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load CT volume from file.
        
        Args:
            ct_path: Path to CT file (.nii.gz or .mhd)
        
        Returns:
            Tuple of (volume array, affine matrix)
        """
        ct_path = Path(ct_path)
        
        if ct_path.suffix == '.mhd':
            import SimpleITK as sitk
            sitk_img = sitk.ReadImage(str(ct_path))
            volume = sitk.GetArrayFromImage(sitk_img)
            
            spacing = sitk_img.GetSpacing()
            origin = sitk_img.GetOrigin()
            affine = np.eye(4)
            affine[0, 0] = spacing[0]
            affine[1, 1] = spacing[1]
            affine[2, 2] = spacing[2]
            affine[:3, 3] = origin
            
            return volume, affine
        else:
            import nibabel as nib
            nii_img = nib.load(str(ct_path))
            volume = nii_img.get_fdata()
            affine = nii_img.affine
            return volume, affine
