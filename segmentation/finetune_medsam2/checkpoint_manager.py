"""
Checkpoint 管理模組

用於保存和載入模型 checkpoints
"""

import torch
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime


class CheckpointManager:
    """
    Checkpoint 管理器
    
    負責:
    - 保存模型 checkpoints
    - 載入 checkpoints
    - 管理 best model
    """
    
    def __init__(
        self,
        checkpoint_dir: Path,
        model_name: str = "medsam2_finetuned",
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化 Checkpoint 管理器
        
        Args:
            checkpoint_dir: checkpoint 保存目錄
            model_name: 模型名稱前綴
            logger: 日誌記錄器
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.logger = logger or logging.getLogger(__name__)
        
        self.best_metric = 0.0
        self.best_epoch = -1
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        epoch: int,
        metrics: Dict,
        is_best: bool = False,
        additional_info: Optional[Dict] = None
    ) -> Path:
        """
        保存 checkpoint
        
        Args:
            model: 模型
            optimizer: 優化器
            scheduler: 學習率調度器
            epoch: 當前 epoch
            metrics: 當前指標
            is_best: 是否為最佳模型
            additional_info: 額外資訊
            
        Returns:
            保存路徑
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': timestamp,
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if additional_info:
            checkpoint['additional_info'] = additional_info
        
        # 保存當前 epoch checkpoint
        checkpoint_path = self.checkpoint_dir / f"{self.model_name}_epoch_{epoch:03d}.pth"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"✅ Checkpoint 已保存: {checkpoint_path}")
        
        # 如果是最佳模型，額外保存一份
        if is_best:
            best_path = self.checkpoint_dir / f"{self.model_name}_best.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"🏆 最佳模型已更新: {best_path}")
            self.best_metric = metrics.get('dice', 0.0)
            self.best_epoch = epoch
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        checkpoint_path: Path,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: torch.device = torch.device('cuda')
    ) -> Dict:
        """
        載入 checkpoint
        
        Args:
            checkpoint_path: checkpoint 路徑
            model: 模型
            optimizer: 優化器（可選）
            scheduler: 學習率調度器（可選）
            device: 設備
            
        Returns:
            checkpoint 資訊
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint 不存在: {checkpoint_path}")
        
        self.logger.info(f"📂 載入 checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 載入模型權重
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 載入優化器狀態
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 載入調度器狀態
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"✅ Checkpoint 載入完成 (epoch {checkpoint.get('epoch', 'N/A')})")
        
        return checkpoint
    
    def load_best_checkpoint(
        self,
        model: torch.nn.Module,
        device: torch.device = torch.device('cuda')
    ) -> Dict:
        """
        載入最佳 checkpoint
        
        Args:
            model: 模型
            device: 設備
            
        Returns:
            checkpoint 資訊
        """
        best_path = self.checkpoint_dir / f"{self.model_name}_best.pth"
        return self.load_checkpoint(best_path, model, device=device)
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """
        獲取最新的 checkpoint 路徑
        
        Returns:
            最新 checkpoint 路徑，如果沒有則返回 None
        """
        checkpoints = list(self.checkpoint_dir.glob(f"{self.model_name}_epoch_*.pth"))
        if not checkpoints:
            return None
        
        # 按 epoch 數排序
        checkpoints.sort(key=lambda p: int(p.stem.split('_')[-1]))
        return checkpoints[-1]
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 3):
        """
        清理舊的 checkpoints，只保留最新的 n 個
        
        Args:
            keep_last_n: 保留的 checkpoint 數量
        """
        checkpoints = list(self.checkpoint_dir.glob(f"{self.model_name}_epoch_*.pth"))
        checkpoints.sort(key=lambda p: int(p.stem.split('_')[-1]))
        
        # 刪除舊的 checkpoints
        for ckpt in checkpoints[:-keep_last_n]:
            ckpt.unlink()
            self.logger.info(f"🗑️ 已刪除舊 checkpoint: {ckpt}")
    
    def get_checkpoint_info(self) -> Dict:
        """
        獲取 checkpoint 資訊摘要
        
        Returns:
            checkpoint 資訊字典
        """
        return {
            'checkpoint_dir': str(self.checkpoint_dir),
            'model_name': self.model_name,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'latest_checkpoint': str(self.get_latest_checkpoint()) if self.get_latest_checkpoint() else None,
        }
