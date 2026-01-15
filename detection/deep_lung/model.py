"""
CenterNet 3D - Anchor-Free Detection for Lung Nodule Detection

This module implements a 3D anchor-free object detection model based on CenterNet.
Key advantages over anchor-based methods:
- No anchor-GT IoU matching issues for small objects
- Direct center point prediction via heatmap
- More stable training with Gaussian ground truth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional

# -----------------------------------------------------------------------------
# 1. Backbone with FPN-like Decoder
# -----------------------------------------------------------------------------

class ResNetBackbone3D(nn.Module):
    """
    Encoder-Decoder backbone for CenterNet.
    Outputs feature map at stride 4 for better small object detection.
    """
    def __init__(self, in_channels=1, base_filters=32):
        super().__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, base_filters)      # /2
        self.enc2 = self._conv_block(base_filters, base_filters*2)   # /4
        self.enc3 = self._conv_block(base_filters*2, base_filters*4) # /8
        
        # Decoder (upsample back to /4)
        self.up1 = nn.ConvTranspose3d(base_filters*4, base_filters*2, kernel_size=2, stride=2)
        self.dec1 = self._conv_block_no_pool(base_filters*4, base_filters*2)  # cat with enc2
        
        self.out_channels = base_filters*2  # 64 channels at stride 4
        self.stride = 4
        
    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2)
        )
    
    def _conv_block_no_pool(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)   # /2
        e2 = self.enc2(e1)  # /4
        e3 = self.enc3(e2)  # /8
        
        # Decoder
        d1 = self.up1(e3)   # /4
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)
        
        return d1  # (B, 64, D/4, H/4, W/4)


# -----------------------------------------------------------------------------
# 2. CenterNet Detection Head
# -----------------------------------------------------------------------------

class CenterNetHead3D(nn.Module):
    """
    Detection head for CenterNet 3D.
    Outputs:
    - heatmap: (B, 1, D, H, W) - nodule center probability
    - size: (B, 3, D, H, W) - (dx, dy, dz) size at each location
    - offset: (B, 3, D, H, W) - sub-voxel offset (optional)
    """
    def __init__(self, in_channels, num_classes=1, use_offset=True):
        super().__init__()
        
        self.use_offset = use_offset
        hidden = 64
        
        # Shared conv
        self.shared = nn.Sequential(
            nn.Conv3d(in_channels, hidden, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden),
            nn.ReLU(inplace=True),
        )
        
        # Heatmap head (center point probability)
        self.heatmap = nn.Sequential(
            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, num_classes, kernel_size=1),
        )
        
        # Size head (predict d, h, w)
        self.size = nn.Sequential(
            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, 3, kernel_size=1),
        )
        
        # Offset head (sub-voxel offset)
        if use_offset:
            self.offset = nn.Sequential(
                nn.Conv3d(hidden, hidden, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(hidden, 3, kernel_size=1),
            )
        
        # Initialize heatmap bias to -2.19 (for focal loss, prevents early training instability)
        self.heatmap[-1].bias.data.fill_(-2.19)
        
    def forward(self, x):
        x = self.shared(x)
        
        hm = self.heatmap(x)
        hm = torch.sigmoid(hm)  # (B, 1, D, H, W)
        
        size = self.size(x)  # (B, 3, D, H, W)
        size = F.relu(size)  # Size must be positive
        
        if self.use_offset:
            offset = self.offset(x)  # (B, 3, D, H, W)
            return hm, size, offset
        else:
            return hm, size, None


# -----------------------------------------------------------------------------
# 3. Loss Functions
# -----------------------------------------------------------------------------

def focal_loss_heatmap(pred, target, alpha=2, beta=4):
    """
    Focal loss for heatmap.
    pred: (B, 1, D, H, W) predicted heatmap
    target: (B, 1, D, H, W) ground truth heatmap with Gaussian peaks
    """
    pos_mask = target.eq(1).float()
    neg_mask = target.lt(1).float()
    
    # Clamp for numerical stability
    pred = torch.clamp(pred, min=1e-6, max=1-1e-6)
    
    # Positive locations (center points)
    pos_loss = -torch.pow(1 - pred, alpha) * torch.log(pred) * pos_mask
    
    # Negative locations (apply penalty based on distance from center)
    neg_loss = -torch.pow(1 - target, beta) * torch.pow(pred, alpha) * torch.log(1 - pred) * neg_mask
    
    num_pos = pos_mask.sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    
    if num_pos == 0:
        return neg_loss
    else:
        return (pos_loss + neg_loss) / num_pos


def size_loss(pred_size, target_size, mask):
    """
    L1 loss for size regression.
    pred_size: (B, 3, D, H, W)
    target_size: (B, 3, D, H, W)
    mask: (B, 1, D, H, W) - 1 at center points
    """
    mask = mask.expand_as(pred_size)
    loss = F.l1_loss(pred_size * mask, target_size * mask, reduction='sum')
    num_pos = mask.sum() / 3  # Divide by 3 since mask is expanded
    if num_pos > 0:
        return loss / num_pos
    return loss * 0


def offset_loss(pred_offset, target_offset, mask):
    """L1 loss for offset regression."""
    mask = mask.expand_as(pred_offset)
    loss = F.l1_loss(pred_offset * mask, target_offset * mask, reduction='sum')
    num_pos = mask.sum() / 3
    if num_pos > 0:
        return loss / num_pos
    return loss * 0


# -----------------------------------------------------------------------------
# 4. Ground Truth Generation
# -----------------------------------------------------------------------------

def generate_heatmap_target(gt_boxes, output_size, stride, min_overlap=0.3):
    """
    Generate ground truth heatmap and size targets.
    
    Args:
        gt_boxes: (N, 6) tensor - (x1, y1, z1, x2, y2, z2)
        output_size: (D, H, W) of the output feature map
        stride: backbone stride
        min_overlap: minimum fraction of Gaussian to cover object
        
    Returns:
        heatmap: (1, D, H, W) with Gaussian peaks at centers
        size_map: (3, D, H, W) with (dx, dy, dz) at center locations
        offset_map: (3, D, H, W) with sub-voxel offsets
        mask: (1, D, H, W) binary mask of center points
    """
    D, H, W = output_size
    device = gt_boxes.device if len(gt_boxes) > 0 else 'cpu'
    
    heatmap = torch.zeros((1, D, H, W), dtype=torch.float32, device=device)
    size_map = torch.zeros((3, D, H, W), dtype=torch.float32, device=device)
    offset_map = torch.zeros((3, D, H, W), dtype=torch.float32, device=device)
    mask = torch.zeros((1, D, H, W), dtype=torch.float32, device=device)
    
    if len(gt_boxes) == 0:
        return heatmap, size_map, offset_map, mask
    
    for box in gt_boxes:
        x1, y1, z1, x2, y2, z2 = box.tolist()
        
        # Box center and size in original coordinates
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        cz = (z1 + z2) / 2
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1
        
        # Map to feature map coordinates
        cx_fm = cx / stride
        cy_fm = cy / stride
        cz_fm = cz / stride
        
        # Integer center position
        cx_int = int(cx_fm)
        cy_int = int(cy_fm)
        cz_int = int(cz_fm)
        
        # Clamp to valid range
        cx_int = max(0, min(cx_int, W - 1))
        cy_int = max(0, min(cy_int, H - 1))
        cz_int = max(0, min(cz_int, D - 1))
        
        # Gaussian radius based on object size
        # Smaller objects get smaller radius
        radius = max(1, int(min(dx, dy, dz) / stride / 2))
        
        # Generate Gaussian kernel
        diameter = 2 * radius + 1
        x = torch.arange(diameter, dtype=torch.float32, device=device)
        y = torch.arange(diameter, dtype=torch.float32, device=device)
        z = torch.arange(diameter, dtype=torch.float32, device=device)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
        
        sigma = diameter / 6  # 3-sigma covers the diameter
        gaussian = torch.exp(-((xx - radius)**2 + (yy - radius)**2 + (zz - radius)**2) / (2 * sigma**2))
        
        # Paste Gaussian onto heatmap
        z_start = max(0, cz_int - radius)
        z_end = min(D, cz_int + radius + 1)
        y_start = max(0, cy_int - radius)
        y_end = min(H, cy_int + radius + 1)
        x_start = max(0, cx_int - radius)
        x_end = min(W, cx_int + radius + 1)
        
        g_z_start = max(0, radius - cz_int)
        g_z_end = g_z_start + (z_end - z_start)
        g_y_start = max(0, radius - cy_int)
        g_y_end = g_y_start + (y_end - y_start)
        g_x_start = max(0, radius - cx_int)
        g_x_end = g_x_start + (x_end - x_start)
        
        # Take max with existing values (for overlapping objects)
        heatmap[0, z_start:z_end, y_start:y_end, x_start:x_end] = torch.max(
            heatmap[0, z_start:z_end, y_start:y_end, x_start:x_end],
            gaussian[g_z_start:g_z_end, g_y_start:g_y_end, g_x_start:g_x_end]
        )
        
        # Set size and offset at center point
        size_map[0, cz_int, cy_int, cx_int] = dx
        size_map[1, cz_int, cy_int, cx_int] = dy
        size_map[2, cz_int, cy_int, cx_int] = dz
        
        offset_map[0, cz_int, cy_int, cx_int] = cx_fm - cx_int
        offset_map[1, cz_int, cy_int, cx_int] = cy_fm - cy_int
        offset_map[2, cz_int, cy_int, cx_int] = cz_fm - cz_int
        
        mask[0, cz_int, cy_int, cx_int] = 1.0
    
    return heatmap, size_map, offset_map, mask


# -----------------------------------------------------------------------------
# 5. Inference Utilities
# -----------------------------------------------------------------------------

def decode_detections(heatmap, size, offset=None, stride=4, score_thresh=0.1, max_detections=100):
    """
    Decode heatmap predictions to bounding boxes.
    
    Args:
        heatmap: (B, 1, D, H, W) predicted heatmap
        size: (B, 3, D, H, W) predicted sizes
        offset: (B, 3, D, H, W) predicted offsets (optional)
        stride: backbone stride
        score_thresh: minimum score for detection
        max_detections: maximum number of detections per image
        
    Returns:
        List of dicts with 'boxes' (N, 6) and 'scores' (N,)
    """
    batch_size = heatmap.size(0)
    results = []
    
    for b in range(batch_size):
        hm = heatmap[b, 0]  # (D, H, W)
        sz = size[b]  # (3, D, H, W)
        
        # Find local maxima (simple 3x3x3 max pooling NMS)
        hm_max = F.max_pool3d(hm.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1)
        hm_max = hm_max.squeeze(0).squeeze(0)
        
        # Keep peaks
        peaks = (hm == hm_max) & (hm > score_thresh)
        
        # Get peak coordinates
        peak_coords = torch.nonzero(peaks, as_tuple=False)  # (N, 3) - z, y, x
        
        if len(peak_coords) == 0:
            results.append({'boxes': torch.zeros((0, 6), device=heatmap.device),
                           'scores': torch.zeros((0,), device=heatmap.device),
                           'labels': torch.zeros((0,), dtype=torch.int64, device=heatmap.device)})
            continue
        
        # Get scores
        scores = hm[peak_coords[:, 0], peak_coords[:, 1], peak_coords[:, 2]]
        
        # Sort by score and keep top N
        if len(scores) > max_detections:
            topk_scores, topk_inds = scores.topk(max_detections)
            peak_coords = peak_coords[topk_inds]
            scores = topk_scores
        
        # Get sizes at peaks
        sizes = sz[:, peak_coords[:, 0], peak_coords[:, 1], peak_coords[:, 2]]  # (3, N)
        sizes = sizes.t()  # (N, 3) - dx, dy, dz
        
        # Get offsets
        if offset is not None:
            off = offset[b]
            offsets = off[:, peak_coords[:, 0], peak_coords[:, 1], peak_coords[:, 2]].t()  # (N, 3)
        else:
            offsets = torch.zeros_like(sizes)
        
        # Convert to boxes
        # peak_coords is (z, y, x), convert to (x, y, z)
        cx = (peak_coords[:, 2].float() + offsets[:, 0]) * stride
        cy = (peak_coords[:, 1].float() + offsets[:, 1]) * stride
        cz = (peak_coords[:, 0].float() + offsets[:, 2]) * stride
        
        dx = sizes[:, 0]
        dy = sizes[:, 1]
        dz = sizes[:, 2]
        
        x1 = cx - dx / 2
        y1 = cy - dy / 2
        z1 = cz - dz / 2
        x2 = cx + dx / 2
        y2 = cy + dy / 2
        z2 = cz + dz / 2
        
        boxes = torch.stack([x1, y1, z1, x2, y2, z2], dim=1)
        labels = torch.ones(len(boxes), dtype=torch.int64, device=heatmap.device)
        
        results.append({'boxes': boxes, 'scores': scores, 'labels': labels})
    
    return results


# -----------------------------------------------------------------------------
# 6. Full Model
# -----------------------------------------------------------------------------

class CenterNet3D(nn.Module):
    """
    CenterNet 3D for lung nodule detection.
    """
    def __init__(self, num_classes=1, score_thresh=0.1):
        super().__init__()
        
        self.backbone = ResNetBackbone3D(in_channels=1, base_filters=32)
        self.head = CenterNetHead3D(self.backbone.out_channels, num_classes=num_classes)
        self.stride = self.backbone.stride
        self.score_thresh = score_thresh
        
    def forward(self, images, targets=None):
        """
        Args:
            images: (B, 1, D, H, W) input volumes
            targets: List of dicts with 'boxes' (N, 6) - only needed for training
            
        Returns:
            Training: dict of losses
            Inference: List of dicts with 'boxes', 'scores', 'labels'
        """
        features = self.backbone(images)
        hm, size, offset = self.head(features)
        
        if self.training:
            # Generate targets
            B = images.size(0)
            D_fm, H_fm, W_fm = features.shape[-3:]
            
            hm_targets = []
            size_targets = []
            offset_targets = []
            masks = []
            
            for i in range(B):
                gt_boxes = targets[i]['boxes']
                hm_t, size_t, offset_t, mask = generate_heatmap_target(
                    gt_boxes, (D_fm, H_fm, W_fm), self.stride
                )
                hm_targets.append(hm_t)
                size_targets.append(size_t)
                offset_targets.append(offset_t)
                masks.append(mask)
            
            hm_targets = torch.stack(hm_targets)
            size_targets = torch.stack(size_targets)
            offset_targets = torch.stack(offset_targets)
            masks = torch.stack(masks)
            
            # Compute losses
            loss_hm = focal_loss_heatmap(hm, hm_targets)
            loss_size = size_loss(size, size_targets, masks)
            loss_offset = offset_loss(offset, offset_targets, masks) if offset is not None else 0
            
            total_loss = loss_hm + 0.1 * loss_size + 0.1 * loss_offset
            
            return {
                'loss_hm': loss_hm,
                'loss_size': loss_size,
                'loss_offset': loss_offset if offset is not None else torch.tensor(0.0),
                'loss': total_loss
            }
        else:
            # Inference
            return decode_detections(hm, size, offset, self.stride, self.score_thresh)


def get_model(num_classes=2, score_thresh=0.1):
    """Factory function to get the detection model."""
    return CenterNet3D(num_classes=1, score_thresh=score_thresh)
