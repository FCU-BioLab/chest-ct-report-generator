import torch
import math
from typing import List, Tuple, Dict, Optional

def box_area_3d(boxes):
    """
    Computes the volume of 3D boxes.
    Arguments:
        boxes (Tensor[N, 6]): boxes in (x1, y1, z1, x2, y2, z2) format
    Returns:
        area (Tensor[N]): areas of the boxes
    """
    return (boxes[:, 3] - boxes[:, 0]) * (boxes[:, 4] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 2])

def box_iou_3d(boxes1, boxes2):
    """
    Compute 3D IoU between two sets of boxes.
    Arguments:
        boxes1 (Tensor[N, 6]): (x1, y1, z1, x2, y2, z2)
        boxes2 (Tensor[M, 6]): (x1, y1, z1, x2, y2, z2)
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise IoU values
    """
    area1 = box_area_3d(boxes1)
    area2 = box_area_3d(boxes2)

    lt = torch.max(boxes1[:, None, :3], boxes2[:, :3])  # [N,M,3]
    rb = torch.min(boxes1[:, None, 3:], boxes2[:, 3:])  # [N,M,3]

    whd = (rb - lt).clamp(min=0)  # [N,M,3]
    inter = whd[:, :, 0] * whd[:, :, 1] * whd[:, :, 2]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou

def nms_3d(boxes, scores, iou_threshold):
    """
    Performs Non-Maximum Suppression (NMS) on 3D boxes.
    Arguments:
        boxes (Tensor[N, 6]): (x1, y1, z1, x2, y2, z2)
        scores (Tensor[N]): scores
        iou_threshold (float): IoU threshold
    Returns:
        keep (Tensor): indices of boxes to keep
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    
    # Sort by score
    _, idx = scores.sort(descending=True)
    keep = []
    
    while idx.numel() > 0:
        current = idx[0]
        keep.append(current.item())
        
        if idx.numel() == 1:
            break
            
        current_box = boxes[current, :].unsqueeze(0)
        other_boxes = boxes[idx[1:], :]
        
        ious = box_iou_3d(current_box, other_boxes).squeeze(0)
        
        # Keep indices where IoU is less than threshold
        idx = idx[1:][ious < iou_threshold]
        
    return torch.tensor(keep, dtype=torch.int64, device=boxes.device)

class BoxCoder3D:
    """
    Encodes/Decodes 3D bounding boxes.
    Format: (x1, y1, z1, x2, y2, z2) <-> (dx, dy, dz, dw, dh, dd)
    """
    def __init__(self, weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)):
        self.weights = weights

    def encode(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some reference boxes
        Arguments:
            reference_boxes (Tensor[N, 6]): reference boxes (e.g., anchors)
            proposals (Tensor[N, 6]): boxes to be encoded (e.g., GT)
        """
        dtype = reference_boxes.dtype
        device = reference_boxes.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
        
        wx, wy, wz, ww, wh, wd = weights
        
        # Convert to center/size
        ex_lengths = reference_boxes[:, 3:] - reference_boxes[:, :3]
        ex_ctr = reference_boxes[:, :3] + 0.5 * ex_lengths
        
        gt_lengths = proposals[:, 3:] - proposals[:, :3]
        gt_ctr = proposals[:, :3] + 0.5 * gt_lengths
        
        targets_dx = wx * (gt_ctr[:, 0] - ex_ctr[:, 0]) / (ex_lengths[:, 0] + 1e-6)
        targets_dy = wy * (gt_ctr[:, 1] - ex_ctr[:, 1]) / (ex_lengths[:, 1] + 1e-6)
        targets_dz = wz * (gt_ctr[:, 2] - ex_ctr[:, 2]) / (ex_lengths[:, 2] + 1e-6)
        targets_dw = ww * torch.log((gt_lengths[:, 0] + 1e-6) / (ex_lengths[:, 0] + 1e-6))
        targets_dh = wh * torch.log((gt_lengths[:, 1] + 1e-6) / (ex_lengths[:, 1] + 1e-6))
        targets_dd = wd * torch.log((gt_lengths[:, 2] + 1e-6) / (ex_lengths[:, 2] + 1e-6))
        
        targets = torch.stack((targets_dx, targets_dy, targets_dz, targets_dw, targets_dh, targets_dd), dim=1)
        
        # Aggressive clamping for stability
        targets = torch.clamp(targets, min=-10.0, max=10.0)
        return targets

    def decode(self, rel_codes, boxes):
        """
        Decode relative codes to boxes
        """
        dtype = rel_codes.dtype
        device = rel_codes.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
        
        wx, wy, wz, ww, wh, wd = weights
        
        dx, dy, dz, dw, dh, dd = rel_codes.unbind(dim=-1)
        
        widths = boxes[:, 3] - boxes[:, 0]
        heights = boxes[:, 4] - boxes[:, 1]
        depths = boxes[:, 5] - boxes[:, 2]
        
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights
        ctr_z = boxes[:, 2] + 0.5 * depths
        
        # Prevent huge values
        dw = torch.clamp(dw / ww, max=math.log(1000.0/16))
        dh = torch.clamp(dh / wh, max=math.log(1000.0/16))
        dd = torch.clamp(dd / wd, max=math.log(1000.0/16))
        
        pred_ctr_x = dx / wx * widths + ctr_x
        pred_ctr_y = dy / wy * heights + ctr_y
        pred_ctr_z = dz / wz * depths + ctr_z
        
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights
        pred_d = torch.exp(dd) * depths
        
        # Guard against NaN/Inf
        pred_w = torch.nan_to_num(pred_w, nan=1.0, posinf=1000.0, neginf=1.0)
        pred_h = torch.nan_to_num(pred_h, nan=1.0, posinf=1000.0, neginf=1.0)
        pred_d = torch.nan_to_num(pred_d, nan=1.0, posinf=1000.0, neginf=1.0)
        
        pred_boxes1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes2 = pred_ctr_y - 0.5 * pred_h
        pred_boxes3 = pred_ctr_z - 0.5 * pred_d
        pred_boxes4 = pred_ctr_x + 0.5 * pred_w
        pred_boxes5 = pred_ctr_y + 0.5 * pred_h
        pred_boxes6 = pred_ctr_z + 0.5 * pred_d
        
        boxes_out = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4, pred_boxes5, pred_boxes6), dim=-1)
        
        # Final NaN check - replace whole box with 0 if any coord is weird (or just clamp)
        boxes_out = torch.nan_to_num(boxes_out, nan=0.0, posinf=10000.0, neginf=-10000.0)
        
        return boxes_out

class Matcher:
    """
    Simple Matcher for Anchors and GT
    """
    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=True):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """
        Arguments:
            match_quality_matrix (Tensor[M, N]): IoU between M anchors and N gt
        Returns:
            matches (Tensor[M]): Index of matched GT, -1 for ignore, -2 for background
        """
        if match_quality_matrix.numel() == 0:
             if match_quality_matrix.shape[0] == 0:
                 return torch.empty((0,), dtype=torch.int64, device=match_quality_matrix.device)
             else:
                 return torch.full((match_quality_matrix.shape[0],), -2, dtype=torch.int64, device=match_quality_matrix.device)

        # Max over GT
        matched_vals, matches = match_quality_matrix.max(dim=1)
        
        # Assign Background (< low_threshold)
        labels = torch.full_like(matches, -2) # Ignore by default
        
        # Low quality (Background)
        labels[matched_vals < self.low_threshold] = -1 # Background
        # Wait, usually -1 is ignore in some impl, 0 is bg. Let's stick to: -1 ignore, -2 bg?
        # Torchvision: -1 ignore, >=0 gt index.
        # Let's say: -1 Neg, -2 Ignore, >=0 Pos
        
        labels[matched_vals < self.low_threshold] = -1 # Neg
        labels[(matched_vals >= self.low_threshold) & (matched_vals < self.high_threshold)] = -2 # Ignore
        labels[matched_vals >= self.high_threshold] = matches[matched_vals >= self.high_threshold] # Pos
        
        if self.allow_low_quality_matches:
            # For each GT, find anchor with highest IoU
            highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=0)
            # Find all anchors that have this highest quality (can be multiple)
            gt_pred_pairs_of_highest_quality = torch.where(
                match_quality_matrix == highest_quality_foreach_gt[None, :]
            )
            pred_inds_to_update = gt_pred_pairs_of_highest_quality[0]
            gt_inds_to_update = gt_pred_pairs_of_highest_quality[1]
            labels[pred_inds_to_update] = gt_inds_to_update
            
        return labels

def balanced_positive_negative_sampler(labels, batch_size_per_image, positive_fraction):
    """
    Sample positive and negative examples
    """
    pos_idx = torch.where(labels == 1)[0]
    neg_idx = torch.where(labels == 0)[0]
    
    num_pos = int(batch_size_per_image * positive_fraction)
    num_pos = min(pos_idx.numel(), num_pos)
    num_neg = batch_size_per_image - num_pos
    num_neg = min(neg_idx.numel(), num_neg)
    
    # Random select
    perm_pos = torch.randperm(pos_idx.numel(), device=pos_idx.device)[:num_pos]
    perm_neg = torch.randperm(neg_idx.numel(), device=neg_idx.device)[:num_neg]
    
    pos_idx = pos_idx[perm_pos]
    neg_idx = neg_idx[perm_neg]
    
    return pos_idx, neg_idx
