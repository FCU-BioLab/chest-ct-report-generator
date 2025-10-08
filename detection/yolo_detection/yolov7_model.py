#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv7 Model Loader with Medical Module Integration

Handles:
- Loading YOLOv7 model from YAML config
- Integrating medical modules (CBAM, SimAM, Swin, BiFPN)
- Model initialization and weight loading
- FLOPs calculation
"""

import sys
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging

# Add models directory to path
MODELS_DIR = Path(__file__).parent / "models"
sys.path.insert(0, str(MODELS_DIR))

try:
    from custom_layers import CBAM, SimAM, SwinTransformerBlock, BiFPN
except ImportError:
    CBAM = SimAM = SwinTransformerBlock = BiFPN = None

LOGGER = logging.getLogger(__name__)


# ==================== Basic YOLOv7 Layers ====================
class Conv(nn.Module):
    """Standard convolution with BatchNorm and activation"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p if p is not None else k // 2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MP(nn.Module):
    """Max pooling"""
    def __init__(self, k=2):
        super().__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)
    
    def forward(self, x):
        return self.m(x)


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension"""
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
    
    def forward(self, x):
        return torch.cat(x, self.d)


class SPPCSPC(nn.Module):
    """Spatial Pyramid Pooling - Cross Stage Partial Channel"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super().__init__()
        c_ = int(2 * c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)
    
    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))


class RepConv(nn.Module):
    """RepConv for YOLOv7"""
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2
        
        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s, p if p is not None else k // 2, groups=g, bias=True)
        else:
            self.rbr_identity = nn.BatchNorm2d(c1) if c2 == c1 and s == 1 else None
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, p if p is not None else k // 2, groups=g, bias=False),
                nn.BatchNorm2d(c2)
            )
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(c1, c2, 1, s, padding=0, groups=g, bias=False),
                nn.BatchNorm2d(c2)
            ) if k > 1 else None
        
        self.act = nn.SiLU() if act else nn.Identity()
    
    def forward(self, x):
        if self.deploy:
            return self.act(self.rbr_reparam(x))
        
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(x)
        
        if self.rbr_1x1 is None:
            return self.act(self.rbr_dense(x) + id_out)
        
        return self.act(self.rbr_dense(x) + self.rbr_1x1(x) + id_out)


class Detect(nn.Module):
    """YOLOv7 Detect head"""
    def __init__(self, nc=80, anchors=(), ch=()):
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl
        
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
        self.export = False
    
    def forward(self, x):
        z = []
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            
            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                
                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                z.append(y.view(bs, -1, self.no))
        
        return x if self.training else (torch.cat(z, 1), x)
    
    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


# ==================== Model Parser ====================
class YOLOv7Model(nn.Module):
    """
    YOLOv7 Model with optional medical modules
    
    Supports:
    - CBAM attention after ELAN stages
    - Swin Transformer in backbone
    - BiFPN neck
    - SimAM before detection heads
    """
    
    def __init__(self, cfg_path: str, ch: int = 3, nc: Optional[int] = None, use_medical_modules: bool = True):
        """
        Args:
            cfg_path: Path to model YAML config
            ch: Input channels
            nc: Number of classes (overrides config)
            use_medical_modules: Enable medical modules
        """
        super().__init__()
        
        # Load config
        with open(cfg_path, 'r', encoding='utf-8') as f:
            self.yaml = yaml.safe_load(f)
        
        self.nc = nc or self.yaml['nc']
        self.use_medical_modules = use_medical_modules
        
        # Parse model
        self.model, self.save = self._parse_model(ch)
        
        # Initialize weights
        self._initialize_weights()
        
        LOGGER.info(f"YOLOv7Model created with {self.nc} classes, medical_modules={use_medical_modules}")
    
    def _parse_model(self, ch: int) -> Tuple[nn.Sequential, List[int]]:
        """Parse model from YAML config"""
        
        # Available modules
        module_dict = {
            'Conv': Conv,
            'MP': MP,
            'Concat': Concat,
            'SPPCSPC': SPPCSPC,
            'RepConv': RepConv,
            'Detect': Detect,
            'nn.Upsample': nn.Upsample,
            'nn.Conv2d': nn.Conv2d,
        }
        
        # Add medical modules if available and enabled
        if self.use_medical_modules:
            if CBAM:
                module_dict['CBAM'] = CBAM
            if SimAM:
                module_dict['SimAM'] = SimAM
            if SwinTransformerBlock:
                module_dict['SwinTransformerBlock'] = SwinTransformerBlock
            if BiFPN:
                module_dict['BiFPN'] = BiFPN
        
        anchors = self.yaml.get('anchors', [])
        backbone_cfg = self.yaml.get('backbone', [])
        head_cfg = self.yaml.get('head', [])
        
        layers = []
        save = []
        c2 = ch  # output channels
        
        # Parse backbone + head
        for i, (f, n, m, args) in enumerate(backbone_cfg + head_cfg):
            m_str = m
            m = module_dict.get(m, eval(m) if isinstance(m, str) else m)
            
            # Parse args
            if isinstance(args, list):
                args = [eval(str(a)) if isinstance(a, str) else a for a in args]
            
            # Handle different module types
            if m in [Conv, RepConv]:
                c1, c2 = ch if i == 0 else layers[f if f != -1 else i - 1].shape[1], args[0]
                args = [c1, c2, *args[1:]]
            elif m == Concat:
                c2 = sum([layers[x].shape[1] for x in f])
            elif m in [CBAM, SimAM] and self.use_medical_modules:
                c2 = layers[f].shape[1] if f != -1 else c2
                args = [c2, *args] if m == CBAM else args
            elif m == SwinTransformerBlock and self.use_medical_modules:
                c2 = layers[f].shape[1] if f != -1 else c2
                args = [c2, *args]
            elif m == BiFPN and self.use_medical_modules:
                c2 = args[0]
            elif m == Detect:
                args.append([layers[x].shape[1] for x in f])
            
            # Create module
            module = m(*args) if isinstance(args, list) else m(**args) if isinstance(args, dict) else m()
            
            # Store module
            module.i = i
            module.f = f
            module.type = m_str
            
            layers.append(module)
            
            if i in save or i == len(backbone_cfg + head_cfg) - 1:
                save.append(i)
        
        return nn.Sequential(*layers), save
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass"""
        y = []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Count medical module parameters
        medical_params = 0
        if self.use_medical_modules:
            for m in self.modules():
                if isinstance(m, (CBAM, SimAM, SwinTransformerBlock, BiFPN)):
                    medical_params += sum(p.numel() for p in m.parameters())
        
        return {
            'total': total,
            'trainable': trainable,
            'medical_modules': medical_params,
        }


def load_yolov7_model(
    cfg_path: str,
    weights_path: Optional[str] = None,
    nc: Optional[int] = None,
    use_medical_modules: bool = True,
    device: str = 'cuda'
) -> YOLOv7Model:
    """
    Load YOLOv7 model with optional medical modules
    
    Args:
        cfg_path: Path to model YAML config
        weights_path: Path to pretrained weights (optional)
        nc: Number of classes
        use_medical_modules: Enable medical modules
        device: Device to load model
    
    Returns:
        YOLOv7Model instance
    """
    model = YOLOv7Model(cfg_path, ch=3, nc=nc, use_medical_modules=use_medical_modules)
    
    if weights_path and Path(weights_path).exists():
        LOGGER.info(f"Loading weights from {weights_path}")
        ckpt = torch.load(weights_path, map_location=device)
        state_dict = ckpt.get('model', ckpt)
        model.load_state_dict(state_dict, strict=False)
    
    model = model.to(device)
    
    # Log parameters
    params = model.count_parameters()
    LOGGER.info(f"Model parameters: Total={params['total']:,}, Trainable={params['trainable']:,}, Medical={params['medical_modules']:,}")
    
    return model


if __name__ == "__main__":
    print("YOLOv7 Model Loader with Medical Modules")
    
    # Test model loading
    cfg_path = MODELS_DIR / "yolov7_medical.yaml"
    if cfg_path.exists():
        print(f"\nLoading model from {cfg_path}")
        model = load_yolov7_model(str(cfg_path), use_medical_modules=True, device='cpu')
        
        # Test forward pass
        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            y = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output: {type(y)}")
        
        params = model.count_parameters()
        print(f"\nModel Statistics:")
        print(f"  Total parameters: {params['total']:,}")
        print(f"  Trainable parameters: {params['trainable']:,}")
        print(f"  Medical module parameters: {params['medical_modules']:,}")
