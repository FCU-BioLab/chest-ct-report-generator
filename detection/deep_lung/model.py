#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Faster R-CNN Model Architecture (Full Implementation)
======================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Dict, Optional, Tuple

from detection.deep_lung.utils import (
    box_iou_3d, nms_3d, BoxCoder3D, Matcher, 
    balanced_positive_negative_sampler
)

# -----------------------------------------------------------------------------
# 1. Backbone
# -----------------------------------------------------------------------------

class ResNet3DBackbone(nn.Module):
    """
    Simple 3D CNN Backbone.
    Input: (B, 1, D, H, W)
    Output: (B, out_channels, D/16, H/16, W/16)
    
    Total stride = 2^4 = 16 (4x MaxPool3d with stride 2)
    """
    STRIDE = 16  # Expose backbone stride explicitly
    def __init__(self, in_channels=1, base_filters=16):
        super().__init__()
        self.enc1 = self._conv_block(in_channels, base_filters)
        self.enc2 = self._conv_block(base_filters, base_filters*2)
        self.enc3 = self._conv_block(base_filters*2, base_filters*4)
        self.enc4 = self._conv_block(base_filters*4, base_filters*8) # Stride 8? No, let's check strides.
        
        # 4 Poolings of stride 2 => Total stride 16.
        # Channels: 1 -> 16 -> 32 -> 64 -> 128
        self.out_channels = base_filters*8
        
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
        
    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        return x

# -----------------------------------------------------------------------------
# 2. Anchor Generator
# -----------------------------------------------------------------------------

class AnchorGenerator3D(nn.Module):
    def __init__(self, sizes=((10, 20, 30),), aspect_ratios=((0.5, 1.0, 2.0),)):
        super().__init__()
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self._cache = {}

    def _generate_anchors(self, size, aspect_ratios, dtype, device):
        # Generate generic 3D anchors at origin (0,0,0)
        # Assuming isotropic ratios apply to (w/h) and (d/w)? 
        # For simplicity in nodule detection, we often use [size, size, size] * scale
        # Let's assume aspect_ratios apply to W/H, and D is relative to W?
        # Or simpler: Isotropic Sphere/Cube approximation for nodules.
        # DeepLung uses spherical anchors. Let's use cubes of fixed sizes.
        
        anchors = []
        for s in size:
            # Cube anchors (common for nodules)
            # If we want specific AR, implement here. Use cubes for now.
            d, h, w = s, s, s
            anchors.append([-w/2, -h/2, -d/2, w/2, h/2, d/2])
            
        return torch.tensor(anchors, dtype=dtype, device=device)

    def grid_anchors(self, grid_sizes: List[List[int]], strides: List[List[Tensor]]):
        anchors = []
        cell_anchors = self.cell_anchors
        
        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            grid_depth, grid_height, grid_width = size
            stride_depth, stride_height, stride_width = stride
            device = base_anchors.device
            
            # Meshgrid
            shift_x = torch.arange(0, grid_width, dtype=torch.float32, device=device) * stride_width
            shift_y = torch.arange(0, grid_height, dtype=torch.float32, device=device) * stride_height
            shift_z = torch.arange(0, grid_depth, dtype=torch.float32, device=device) * stride_depth
            
            shift_z, shift_y, shift_x = torch.meshgrid(shift_z, shift_y, shift_x, indexing='ij')
            shift_z = shift_z.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shift_x = shift_x.reshape(-1)
            
            shifts = torch.stack((shift_x, shift_y, shift_z, shift_x, shift_y, shift_z), dim=1) # (N, 6)
            
            # Add shifts to base anchors
            # base: (A, 6), shifts: (N, 6) -> (N, A, 6)
            final_anchors = (shifts.view(-1, 1, 6) + base_anchors.view(1, -1, 6)).reshape(-1, 6)
            anchors.append(final_anchors)
            
        return anchors

    def forward(self, image_list, feature_maps):
        # image_list not used strictly if we assume fixed stride
        grid_sizes = [list(feature.shape[-3:]) for feature in feature_maps]
        image_size = image_list.shape[-3:] # (D, H, W)
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        
        # FIX: Use float stride to avoid truncation issues
        strides = [[
            torch.tensor(image_size[0] / g[0], dtype=torch.float32, device=device),
            torch.tensor(image_size[1] / g[1], dtype=torch.float32, device=device),
            torch.tensor(image_size[2] / g[2], dtype=torch.float32, device=device)
        ] for g in grid_sizes]
        
        # Cache cell anchors
        if self.cell_anchors is None:
            self.cell_anchors = [
                self._generate_anchors(self.sizes[0], self.aspect_ratios[0], dtype, device)
                for _ in feature_maps
            ]
            
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes, strides)
        
        anchors = []
        for i in range(len(image_list)): # Per Image
            anchors_in_image = []
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                anchors_in_image.append(anchors_per_feature_map)
            anchors.append(torch.cat(anchors_in_image))
            
        return anchors

# -----------------------------------------------------------------------------
# 3. Region Proposal Network (Full)
# -----------------------------------------------------------------------------

class RPNHead3D(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.cls_score = nn.Conv3d(in_channels, num_anchors, kernel_size=1)
        self.bbox_pred = nn.Conv3d(in_channels, num_anchors * 6, kernel_size=1)
        
    def forward(self, x):
        logits = []
        bbox_reg = []
        for feature in x:
            t = F.relu(self.conv(feature))
            logits.append(self.cls_score(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg

class RegionProposalNetwork3D(nn.Module):
    def __init__(self, anchor_generator, head, 
                 fg_iou_thresh, bg_iou_thresh, 
                 batch_size_per_image, positive_fraction,
                 pre_nms_top_n, post_nms_top_n, nms_thresh):
        super().__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = BoxCoder3D(weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
        
        # Params
        self.min_size = 1e-3
        self.proposal_matcher = Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=True)
        
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh

    def assign_targets_to_anchors(self, anchors, targets):
        labels = []
        matched_gt_boxes = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image['boxes']
            
            if gt_boxes.numel() == 0:
                device = anchors_per_image.device
                matched_gt_boxes.append(torch.zeros_like(anchors_per_image))
                labels.append(torch.full((anchors_per_image.shape[0],), 0, dtype=torch.float32, device=device))
                continue
            
            match_quality_matrix = box_iou_3d(gt_boxes, anchors_per_image)
            # Matcher expects (Anchors, GT), iou is (GT, Anchors)
            match_quality_matrix = match_quality_matrix.transpose(0, 1)
            matched_idxs = self.proposal_matcher(match_quality_matrix) # Shape (N_anchors)
            
            clamped_matched_idxs = matched_idxs.clamp(min=0)
            matched_gt_boxes_i = gt_boxes[clamped_matched_idxs]
            
            labels_i = matched_idxs >= 0
            labels_i = labels_i.to(dtype=torch.float32)
            
            # Background
            bg_inds = matched_idxs == -1
            labels_i[bg_inds] = 0.0
            
            # Ignore
            ignore_inds = matched_idxs == -2
            labels_i[ignore_inds] = -1.0 
            
            labels.append(labels_i)
            matched_gt_boxes.append(matched_gt_boxes_i)
            
        return labels, matched_gt_boxes

    def _compute_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):
        """
        objectness: list of (B, A, D, H, W) -> flatten
        labels: list of (N_anchors)
        """
        sampled_pos_inds, sampled_neg_inds = self._sample_proposals(labels)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]
        
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        
        objectness = objectness.flatten()
        labels = torch.cat(labels, dim=0)
        
        loss_objectness = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds]
        )
        
        pred_bbox_deltas = pred_bbox_deltas.reshape(-1, 6)
        regression_targets = regression_targets.reshape(-1, 6)
        
        loss_box_reg = F.smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1.0 / 9,
            reduction='sum'
        ) / (sampled_pos_inds.numel() + 1e-5)  # FIX: Divide by num positives only
        
        return loss_objectness, loss_box_reg

    def _sample_proposals(self, labels):
        pos_fraction = self.positive_fraction
        batch_size = self.batch_size_per_image
        
        pos_masks = []
        neg_masks = []
        for l in labels:
            pos_idx, neg_idx = balanced_positive_negative_sampler(l, batch_size, pos_fraction)
            pos_mask = torch.zeros_like(l, dtype=torch.bool)
            neg_mask = torch.zeros_like(l, dtype=torch.bool)
            pos_mask[pos_idx] = True
            neg_mask[neg_idx] = True
            pos_masks.append(pos_mask)
            neg_masks.append(neg_mask)
        return pos_masks, neg_masks

    def forward(self, images, features, targets=None):
        # 1. Generate Anchors
        # features is list
        anchors = self.anchor_generator(images, features) # list of tensors
        
        # 2. RPN Head
        # flatten features if single level
        objectness, pred_bbox_deltas = self.head(features)
        
        # Reshape: (B, A*1, D, H, W) -> (B, A, D, H, W) -> (B, N, 1/6)
        num_anchors = objectness[0].shape[1] 
        # objectness is (B, A, D, H, W)
        
        objectness_flat = []
        pred_bbox_deltas_flat = []
        
        for o, b in zip(objectness, pred_bbox_deltas):
            B, C, D, H, W = o.shape
            o = o.permute(0, 2, 3, 4, 1).reshape(B, -1) # (B, N_anchors_level)
            b = b.permute(0, 2, 3, 4, 1).reshape(B, -1, 6)
            objectness_flat.append(o)
            pred_bbox_deltas_flat.append(b)
            
        objectness_flat = torch.cat(objectness_flat, dim=1)
        pred_bbox_deltas_flat = torch.cat(pred_bbox_deltas_flat, dim=1)
        
        proposals = self._decode_proposals(anchors, params=(objectness_flat, pred_bbox_deltas_flat))
        
        losses = {}
        if self.training:
             # Match targets
             if targets is None:
                 raise ValueError("Targets missing in training")
             labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
             regression_targets = self.box_coder.encode(torch.cat(anchors, dim=0), torch.cat(matched_gt_boxes, dim=0))
             
             loss_obj, loss_box = self._compute_loss(
                 objectness_flat, pred_bbox_deltas_flat, labels, regression_targets
             )
             losses = {
                 "loss_rpn_cls": loss_obj,
                 "loss_rpn_loc": loss_box
             }
             
        return proposals, losses

    def _decode_proposals(self, anchors, params):
        objectness, pred_bbox_deltas = params
        # Apply NMS
        proposals = []
        B = objectness.shape[0]
        
        # anchors is list of (N_anchors_img_i)
        # But here anchors is list of anchors per image, same shape
        
        for i in range(B):
            scores = objectness[i].sigmoid()
            deltas = pred_bbox_deltas[i]
            anchors_i = anchors[i]
            
            # Top K pre nms
            if self.pre_nms_top_n < scores.shape[0]:
                _, topk_idx = scores.topk(self.pre_nms_top_n)
                scores = scores[topk_idx]
                deltas = deltas[topk_idx]
                anchors_i = anchors_i[topk_idx]
                
            boxes = self.box_coder.decode(deltas, anchors_i)
            
            # FIX: Clip proposals to image boundaries
            # Assume image_shape available from earlier context or passed
            # For now, clip to reasonable bounds based on typical 128^3 input
            boxes = boxes.clamp(min=0)
            
            # FIX: Filter boxes smaller than min_size
            widths = boxes[:, 3] - boxes[:, 0]
            heights = boxes[:, 4] - boxes[:, 1]
            depths = boxes[:, 5] - boxes[:, 2]
            keep_size = (widths >= self.min_size) & (heights >= self.min_size) & (depths >= self.min_size)
            boxes = boxes[keep_size]
            scores = scores[keep_size]
            
            # Filter NaNs/Infs explicitly
            keep_finite = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores)
            boxes = boxes[keep_finite]
            scores = scores[keep_finite]
            
            if boxes.numel() == 0:
                proposals.append(torch.zeros((0, 6), device=boxes.device))
                continue
            
            keep = nms_3d(boxes, scores, self.nms_thresh)
            if self.post_nms_top_n < keep.shape[0]:
                keep = keep[:self.post_nms_top_n]
                
            proposals.append(boxes[keep])
            
        return proposals

# -----------------------------------------------------------------------------
# 4. RoI Heads (Full - 3D)
# -----------------------------------------------------------------------------

class RoIHeads3D(nn.Module):
    def __init__(self, in_channels, num_classes, 
                 fg_iou_thresh=0.5, bg_iou_thresh=0.5,
                 batch_size_per_image=512, positive_fraction=0.25,
                 bbox_reg_weights=None,
                 score_thresh=0.05, nms_thresh=0.5, detections_per_img=100):
        super().__init__()
        
        self.box_roi_pool = Tool3dRoIPool(output_size=(4, 4, 4)) # Simplistic Pool
        self.box_head = nn.Sequential(
            nn.Linear(in_channels * 4 * 4 * 4, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True)
        )
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 6)
        
        self.proposal_matcher = Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=False)
        self.box_coder = BoxCoder3D(weights=(10., 10., 10., 5., 5., 5.))
        
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

    def assign_targets(self, proposals, targets):
        matched_idxs = []
        labels = []
        
        for proposals_in_image, targets_in_image in zip(proposals, targets):
            gt_boxes = targets_in_image['boxes']
            
            if gt_boxes.numel() == 0:
                 device = proposals_in_image.device
                 clamped_matched_idxs_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)
                 labels_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device) # Bg
            else:
                match_quality_matrix = box_iou_3d(gt_boxes, proposals_in_image)
                # Matcher expects (Proposals, GT) but iou is (GT, Proposals)
                # Transpose it
                match_quality_matrix = match_quality_matrix.transpose(0, 1)
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)
                
                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)
                
                labels_in_image = matched_idxs_in_image >= 0
                labels_in_image = labels_in_image.to(dtype=torch.int64)
                
                # BG
                bg_inds = matched_idxs_in_image == -1
                labels_in_image[bg_inds] = 0
                
                # Ignore
                ignore_inds = matched_idxs_in_image == -2
                labels_in_image[ignore_inds] = -1
            
            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
            
        return matched_idxs, labels

    def subsample(self, labels):
        # Balanced sampler again
        pos_fraction = self.positive_fraction
        batch_size = self.batch_size_per_image
        
        sampled_pos_inds = []
        sampled_neg_inds = []
        
        for l in labels:
            pos_idx, neg_idx = balanced_positive_negative_sampler(l, batch_size, pos_fraction)
            sampled_pos_inds.append(pos_idx)
            sampled_neg_inds.append(neg_idx)
            
        return sampled_pos_inds, sampled_neg_inds

    def forward(self, features, proposals, image_shapes, targets=None):
        """
        features: list of (B, C, D, H, W)
        proposals: list of (N_prop, 6)
        """
        if self.training:
            matched_idxs, labels = self.assign_targets(proposals, targets)
            sampled_pos_inds, sampled_neg_inds = self.subsample(labels)
            
            proposals_n = []
            labels_n = []
            matched_gt_boxes_n = []
            
            for img_idx, (pos_inds, neg_inds) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
                img_sampled_inds = torch.cat([pos_inds, neg_inds], dim=0)
                proposals_n.append(proposals[img_idx][img_sampled_inds])
                labels_n.append(labels[img_idx][img_sampled_inds])
                
                gt_boxes = targets[img_idx]['boxes']
                gt_idxs_per_img = matched_idxs[img_idx][img_sampled_inds]
                
                if gt_boxes.numel() == 0:
                     matched_gt_boxes_n.append(torch.zeros((len(gt_idxs_per_img), 6), device=gt_boxes.device))
                else:
                     matched_gt_boxes_n.append(gt_boxes[gt_idxs_per_img])
                
            proposals = proposals_n
        
        # RoI Pool
        # features[0] only for simplicity
        box_features = self.box_roi_pool(features[0], proposals, image_shapes)
        box_features = box_features.flatten(start_dim=1)
        
        box_features = self.box_head(box_features)
        class_logits = self.cls_score(box_features)
        box_regression = self.bbox_pred(box_features)
        
        result = []
        losses = {}
        
        if self.training:
            labels = torch.cat(labels_n, dim=0)
            regression_targets = []
            for i in range(len(proposals)):
                regression_targets.append(self.box_coder.encode(proposals[i], matched_gt_boxes_n[i]))
            regression_targets = torch.cat(regression_targets, dim=0)
            
            loss_cls = F.cross_entropy(class_logits, labels, ignore_index=-1, reduction='none')
            
            # OHEM: Keep top-k hardest negatives
            neg_inds = torch.where(labels == 0)[0]
            pos_inds = torch.where(labels > 0)[0]
            
            if len(neg_inds) > 0:
                neg_losses = loss_cls[neg_inds]
                # Keep top 3x positives worth of hard negatives
                num_hard_neg = min(len(neg_inds), max(128, len(pos_inds) * 3))
                _, hard_neg_idx = neg_losses.topk(num_hard_neg)
                hard_neg_inds = neg_inds[hard_neg_idx]
                
                # Combine pos + hard neg
                ohem_inds = torch.cat([pos_inds, hard_neg_inds])
                loss_cls = loss_cls[ohem_inds].mean()
            else:
                loss_cls = loss_cls.mean()
            
            # Box loss only on positives
            pos_inds = torch.where(labels > 0)[0]
            loss_box_reg = F.smooth_l1_loss(
                box_regression.reshape(-1, class_logits.shape[1], 6)[pos_inds, labels[pos_inds]],
                regression_targets[pos_inds],
                beta=1.0,
                reduction='sum'
            ) / (labels.numel() + 1e-5)
            
            losses = {
                "loss_classifier": loss_cls,
                "loss_box_reg": loss_box_reg
            }
        else:
            # Inference Post Process - FIX: Process per image to avoid cross-image NMS
            scores = F.softmax(class_logits, dim=-1)  # (N_total, C)
            deltas = box_regression.reshape(-1, class_logits.shape[1], 6)[:, 1]  # Class 1
            
            all_proposals = torch.cat(proposals, dim=0)
            boxes = self.box_coder.decode(deltas, all_proposals)
            scores_cls1 = scores[:, 1]
            
            # Track proposal counts per image
            proposal_counts = [len(p) for p in proposals]
            
            result = []
            start_idx = 0
            for count in proposal_counts:
                end_idx = start_idx + count
                
                img_boxes = boxes[start_idx:end_idx]
                img_scores = scores_cls1[start_idx:end_idx]
                
                if img_boxes.numel() == 0:
                    result.append({
                        "boxes": torch.zeros((0, 6), device=boxes.device),
                        "scores": torch.zeros((0,), device=boxes.device),
                        "labels": torch.zeros((0,), dtype=torch.int64, device=boxes.device)
                    })
                else:
                    # Per-image NMS
                    keep = nms_3d(img_boxes, img_scores, self.nms_thresh)
                    keep = keep[:self.detections_per_img]
                    
                    result.append({
                        "boxes": img_boxes[keep],
                        "scores": img_scores[keep],
                        "labels": torch.ones(len(keep), dtype=torch.int64, device=boxes.device)  # FIX: Proper dtype
                    })
                
                start_idx = end_idx
            
        return result, losses

class Tool3dRoIPool(nn.Module):
    """
    Simulated 3D ROI Pool using Grid Sample or Adaptive Max Pool on Crops
    """
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size # (d, h, w)

    def forward(self, feature, proposals, image_shapes):
        # proposals: list of (N, 6)
        # feature: (B, C, D, H, W)
        
        crops = []
        for i in range(len(proposals)):
            # Per image
            img_feat = feature[i] # (C, D, H, W)
            boxes = proposals[i]
            
            # Feature map size
            D_feat, H_feat, W_feat = img_feat.shape[-3:]

            for box in boxes:
                # box: x1, y1, z1, x2, y2, z2 (Pixels)
                # Need to map to feature map coordinates?
                # The backbone has stride 4 (Standard ResNet3DBackbone here has stride 8 or 16?)
                # backbone: output Stride 16?
                # In forward(): self.backbone(images)
                
                # Note: Proposals are in original image scale. 
                # Features are subsampled.
                # FIX: Reference backbone stride constant instead of hardcoding
                stride = float(ResNet3DBackbone.STRIDE)
                
                z1 = int(box[2].item() / stride)
                y1 = int(box[1].item() / stride)
                x1 = int(box[0].item() / stride)
                z2 = int(box[5].item() / stride)
                y2 = int(box[4].item() / stride)
                x2 = int(box[3].item() / stride)
                
                # Clamp
                z1 = max(0, min(z1, D_feat - 1))
                y1 = max(0, min(y1, H_feat - 1))
                x1 = max(0, min(x1, W_feat - 1))
                
                z2 = max(z1 + 1, min(z2, D_feat)) # Ensure at least 1 pixel
                y2 = max(y1 + 1, min(y2, H_feat))
                x2 = max(x1 + 1, min(x2, W_feat))
                
                # Slice
                roi = img_feat[:, z1:z2, y1:y2, x1:x2]
                
                # Adaptive Pool
                roi = F.adaptive_max_pool3d(roi, self.output_size)
                crops.append(roi)
                
        if len(crops) == 0:
            # Return dummy batch
            return torch.zeros((0, img_feat.shape[0], *self.output_size), device=img_feat.device)
            
        return torch.stack(crops, dim=0)

# -----------------------------------------------------------------------------
# 5. Full Model Wrapper
# -----------------------------------------------------------------------------

class FasterRCNN3D(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = ResNet3DBackbone(base_filters=32)
        
        # Anchor Generator
        # Nodule size: Min=3, Max=30, Mean=5.2, 90th=8.1
        # Adding 2mm anchor for tiny nodules
        self.anchor_generator = AnchorGenerator3D(sizes=((2, 4, 8, 16, 32),))
        
        self.rpn = RegionProposalNetwork3D(
            self.anchor_generator,
            head=RPNHead3D(self.backbone.out_channels, num_anchors=5),  # 5 anchors now
            fg_iou_thresh=0.5, bg_iou_thresh=0.2,  # Lowered from 0.7/0.3
            batch_size_per_image=256, positive_fraction=0.7,  # Increased from 0.5
            pre_nms_top_n=2000, post_nms_top_n=1000, nms_thresh=0.5
        )
        
        self.roi_heads = RoIHeads3D(
            self.backbone.out_channels, num_classes,
            detections_per_img=100
        )

    def forward(self, images, targets=None):
        """
        images: (B, 1, D, H, W)
        targets: List of Dict {'boxes': (N, 6)}
        """
        # Backbone
        features = self.backbone(images) # (B, 128, D', H', W')
        feature_list = [features] # Single scale for now
        
        # RPN
        proposals, rpn_losses = self.rpn(images, feature_list, targets)
        
        # RoI Heads
        detections, detector_losses = self.roi_heads(feature_list, proposals, images.shape[-3:], targets)
        
        losses = {}
        losses.update(rpn_losses)
        losses.update(detector_losses)
        
        if self.training:
            return losses
        else:
            return detections

def get_model(num_classes=2):
    return FasterRCNN3D(num_classes=num_classes)
