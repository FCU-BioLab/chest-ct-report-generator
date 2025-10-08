#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv7 Training Utilities

Includes:
- Loss computation (YOLOv7 loss)
- EMA (Exponential Moving Average)
- Learning rate schedulers
- Evaluation metrics
- Model evaluation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging
import numpy as np

LOGGER = logging.getLogger(__name__)


# ==================== Loss Functions ====================
class ComputeLoss:
    """
    YOLOv7 Loss Computation
    
    Combines:
    - Box regression loss (CIoU)
    - Objectness loss (BCE)
    - Classification loss (BCE)
    """
    
    def __init__(self, model, autobalance=False):
        """
        Args:
            model: YOLOv7 model
            autobalance: Auto-balance loss weights
        """
        device = next(model.parameters()).device
        h = model.yaml
        
        # Get Detect() module
        det = model.model[-1]
        self.na = det.na  # number of anchors
        self.nc = det.nc  # number of classes
        self.nl = det.nl  # number of layers
        self.anchors = det.anchors
        self.device = device
        
        # Loss weights
        self.balance = [4.0, 1.0, 0.4]
        self.box = h.get('box', 0.05)
        self.obj = h.get('obj', 1.0)
        self.cls = h.get('cls', 0.5)
        
        # Class label smoothing
        self.cp, self.cn = 1.0, 0.0
        
        # Focal loss
        self.BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
        self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
        
        self.autobalance = autobalance
        self.ssi = 0  # stride 16 index
        
        # Anchors
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))
    
    def __call__(self, p, targets):
        """
        Args:
            p: Predictions (list of tensors for each detection layer)
            targets: Ground truth targets (batch_idx, class, x, y, w, h)
        
        Returns:
            Total loss and loss components
        """
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)
        
        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
            
            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
                
                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss
                
                # Objectness
                tobj[b, a, gj, gi] = (1.0 - 0.0) + 0.0 * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio
                
                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE
            
            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
        
        # Scale losses
        lbox *= self.box
        lobj *= self.obj
        lcls *= self.cls
        bs = tobj.shape[0]  # batch size
        
        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()
    
    def build_targets(self, p, targets):
        """Build targets for compute_loss()"""
        na, nt = self.na, targets.shape[0]
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
        
        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                           [1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float() * g
        
        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
            
            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]
                j = torch.max(r, 1. / r).max(2)[0] < 4.0  # compare
                t = t[j]  # filter
                
                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0
            
            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices
            
            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
        
        return tcls, tbox, indices, anch


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculate IoU between boxes
    
    Args:
        box1: Boxes (4, n)
        box2: Boxes (4, n)
        x1y1x2y2: Box format
        GIoU, DIoU, CIoU: IoU variants
    
    Returns:
        IoU values
    """
    box2 = box2.T
    
    # Get coordinates
    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
    
    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    
    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps
    
    iou = inter / union
    
    if CIoU or DIoU or GIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        if CIoU or DIoU:
            c2 = cw ** 2 + ch ** 2 + eps
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                   (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
            if DIoU:
                return iou - rho2 / c2
            elif CIoU:
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)
        else:  # GIoU
            c_area = cw * ch + eps
            return iou - (c_area - union) / c_area
    else:
        return iou


# ==================== EMA ====================
class ModelEMA:
    """
    Exponential Moving Average for model parameters
    Keeps a moving average of everything in model.state_dict()
    """
    
    def __init__(self, model, decay=0.9999, updates=0):
        """
        Args:
            model: Model to apply EMA
            decay: EMA decay rate
            updates: Number of updates
        """
        self.ema = model.module if hasattr(model, 'module') else model
        self.ema = type(self.ema)(self.ema.yaml, ch=3, nc=self.ema.nc)
        self.ema.eval()
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        
        for p in self.ema.parameters():
            p.requires_grad_(False)
    
    def update(self, model):
        """Update EMA parameters"""
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)
            
            msd = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()


# ==================== Learning Rate Schedulers ====================
def one_cycle(y1=0.0, y2=1.0, steps=100):
    """One cycle learning rate schedule"""
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


class WarmupCosineSchedule:
    """
    Warmup + Cosine Annealing Learning Rate Scheduler
    """
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, warmup_lr_start=1e-5, min_lr=1e-5):
        """
        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of warmup epochs
            total_epochs: Total training epochs
            warmup_lr_start: Starting LR for warmup
            min_lr: Minimum LR
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_lr_start = warmup_lr_start
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self, epoch):
        """Update learning rate"""
        if epoch < self.warmup_epochs:
            # Warmup phase
            lr_scale = (epoch + 1) / self.warmup_epochs
            lrs = [self.warmup_lr_start + (base_lr - self.warmup_lr_start) * lr_scale 
                   for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr_scale = 0.5 * (1 + math.cos(math.pi * progress))
            lrs = [self.min_lr + (base_lr - self.min_lr) * lr_scale 
                   for base_lr in self.base_lrs]
        
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr
        
        return lrs[0]


# ==================== Evaluation Metrics ====================
def compute_ap(recall, precision):
    """
    Compute Average Precision
    
    Args:
        recall: Recall values
        precision: Precision values
    
    Returns:
        AP value
    """
    # Append sentinel values
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    
    # Compute precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    # Calculate area under PR curve
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return ap


def compute_metrics(tp, conf, pred_cls, target_cls, eps=1e-16):
    """
    Compute precision, recall, F1, and AP metrics
    
    Args:
        tp: True positives
        conf: Confidence scores
        pred_cls: Predicted classes
        target_cls: Target classes
        eps: Small value to avoid division by zero
    
    Returns:
        Dictionary of metrics
    """
    # Sort by confidence
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    
    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]
    
    # Create precision-recall curve
    px, py = np.linspace(0, 1, 1000), []
    
    # Initialize metrics
    ap, p, r = np.zeros((nc,)), np.zeros((nc,)), np.zeros((nc,))
    
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions
        
        if n_p == 0 or n_l == 0:
            continue
        
        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)
        
        # Recall
        recall = tpc / (n_l + eps)
        r[ci] = recall[-1]
        
        # Precision
        precision = tpc / (tpc + fpc)
        p[ci] = precision[-1]
        
        # AP from recall-precision curve
        py.append(np.interp(px, recall[::-1], precision[::-1]))
        ap[ci] = compute_ap(recall, precision)
    
    # Compute F1 score
    f1 = 2 * p * r / (p + r + eps)
    
    return {
        'precision': p.mean(),
        'recall': r.mean(),
        'f1': f1.mean(),
        'ap': ap.mean(),
        'ap_class': ap,
    }


# ==================== Utility Functions ====================
def strip_optimizer(f='best.pt', s=''):
    """
    Strip optimizer from checkpoint to reduce file size
    
    Args:
        f: Checkpoint file path
        s: Save path (empty = overwrite)
    """
    x = torch.load(f, map_location=torch.device('cpu'))
    if 'ema' in x:
        x['model'] = x['ema']
    
    for k in 'optimizer', 'training_results', 'wandb_id', 'ema', 'updates':
        x[k] = None
    
    x['epoch'] = -1
    x['model'].half()
    
    for p in x['model'].parameters():
        p.requires_grad = False
    
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6
    LOGGER.info(f"Optimizer stripped from {f},{(' saved as %s,' % s) if s else ''} {mb:.1f}MB")


def select_device(device='', batch_size=None):
    """
    Select PyTorch device
    
    Args:
        device: Device string ('', 'cpu', 'cuda', '0', '0,1', etc.)
        batch_size: Batch size for CUDA device selection
    
    Returns:
        torch.device
    """
    s = f'YOLOv7 🚀 torch {torch.__version__} '
    device = str(device).strip().lower().replace('cuda:', '')
    cpu = device == 'cpu'
    
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'
    
    cuda = not cpu and torch.cuda.is_available()
    
    if cuda:
        devices = device.split(',') if device else '0'
        n = len(devices)
        if n > 1 and batch_size:
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * (len(s) + 1)
        
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2:.0f}MB)\n"
    else:
        s += 'CPU\n'
    
    LOGGER.info(s)
    return torch.device('cuda:0' if cuda else 'cpu')


if __name__ == "__main__":
    print("YOLOv7 Training Utilities")
    print("Includes loss computation, EMA, schedulers, and metrics")
