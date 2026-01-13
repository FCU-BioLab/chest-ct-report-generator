#!/usr/bin/env python3
"""
3D U-Net Video Trainer
======================

Trainer logic for 3D U-Net.
"""

import logging
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from .dataset import VolumetricDataset, collate_video_batch
from .model import get_model
from .config import Config

logger = logging.getLogger(__name__)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice

class UNet3DTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.ckpt_dir.mkdir(exist_ok=True)
        
        # Model
        self.model = get_model(config).to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Losses
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        
        # Datasets
        self.train_loader = self._create_loader("train", shuffle=True)
        self.val_loader = self._create_loader("val", shuffle=False)
        
        self.best_val_score = 0.0
        
    def _create_loader(self, split: str, shuffle: bool):
        dataset = VolumetricDataset(
            npz_dir=self.config.data.npz_dir,
            split=split,
            image_size=self.config.model.image_size,
            max_depth=self.config.data.max_depth,
            augmentation=(split == 'train')
        )
        return DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            collate_fn=collate_video_batch,
            pin_memory=True
        )

    def train(self):
        logger.info(f"🚀 Starting training for {self.config.training.epochs} epochs")
        
        for epoch in range(1, self.config.training.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_score = self.validate(epoch)
            
            logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Dice={val_score:.4f}")
            
            # Save best
            if val_score > self.best_val_score:
                self.best_val_score = val_score
                self.save_checkpoint(epoch, val_score, is_best=True)
            
            # Save periodic
            if epoch % 5 == 0:
                self.save_checkpoint(epoch, val_score)
                
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Train Ep {epoch}")
        
        for batch in pbar:
            images = batch['image'].to(self.device).float()
            masks = batch['mask'].to(self.device).float()
            
            self.optimizer.zero_grad()
            
            logits = self.model(images)
            if isinstance(logits, list):
                # Deep supervision
                loss = 0
                weights = [1.0, 0.5, 0.25, 0.125]  # Example weights
                for i, logit in enumerate(logits[:len(weights)]):
                    # Resize mask to logit size if needed
                    if logit.shape != masks.shape:
                         target = F.interpolate(masks, size=logit.shape[2:])
                    else:
                        target = masks
                    
                    l_bce = self.bce(logit, target)
                    l_dice = self.dice(logit, target)
                    loss += weights[i] * (l_bce + l_dice)
            else:
                l_bce = self.bce(logits, masks)
                l_dice = self.dice(logits, masks)
                loss = (self.config.training.bce_weight * l_bce) + \
                       (self.config.training.dice_weight * l_dice)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        return total_loss / len(self.train_loader)

    def validate(self, epoch: int) -> float:
        self.model.eval()
        total_dice = 0
        count = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Val Ep {epoch}"):
                images = batch['image'].to(self.device).float()
                masks = batch['mask'].to(self.device).float()
                
                logits = self.model(images)
                if isinstance(logits, list):
                    logits = logits[0]
                
                # Dice score calculation
                pred = (torch.sigmoid(logits) > 0.5).float()
                inter = (pred * masks).sum()
                union = pred.sum() + masks.sum()
                dice = (2. * inter) / (union + 1e-6)
                
                total_dice += dice.item()
                count += 1
        
        return total_dice / count if count > 0 else 0

    def save_checkpoint(self, epoch: int, score: float, is_best: bool = False):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'score': score,
            'config': self.config.save(str(self.ckpt_dir / "config.json")) # save config is weird here, returns None
        }
        
        filename = f"checkpoint_ep{epoch}.pt"
        torch.save(state, self.ckpt_dir / filename)
        
        if is_best:
            torch.save(state, self.ckpt_dir / "best_model.pt")
            logger.info(f"🆕 New best model saved (Dice={score:.4f})")

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint"""
        logger.info(f"📥 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"✅ Checkpoint loaded (Epoch {checkpoint.get('epoch', '?')}, Score {checkpoint.get('score', 0.0):.4f})")

    def evaluate(self, split: str = 'test'):
        """Evaluate on specific split"""
        logger.info(f"📊 Evaluating on {split} set...")
        loader = self._create_loader(split, shuffle=False)
        self.model.eval()
        
        total_dice = 0.0
        count = 0
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Eval {split}"):
                images = batch['image'].to(self.device).float()
                masks = batch['mask'].to(self.device).float()
                
                logits = self.model(images)
                if isinstance(logits, list):
                    logits = logits[0]
                
                # Dice
                pred = (torch.sigmoid(logits) > 0.5).float()
                inter = (pred * masks).sum()
                union = pred.sum() + masks.sum()
                dice = (2. * inter) / (union + 1e-6)
                
                total_dice += dice.item()
                count += 1
        
        avg_dice = total_dice / count if count > 0 else 0.0
        logger.info(f"🏆 {split.upper()} Dice Score: {avg_dice:.4f}")
        return avg_dice
