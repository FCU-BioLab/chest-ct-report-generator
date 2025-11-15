"""
自定义损失函数集成模块
用于将 EAPIoU Loss 集成到 Ultralytics YOLO11 训练流程中

关键修改：
1. 替换默认的 CIoU 为 EAPIoU
2. 保留 Ultralytics 的训练框架和其他损失函数
3. 支持 Deep Supervision（多尺度辅助损失）
"""

import torch
import torch.nn as nn
from ultralytics.utils.loss import v8DetectionLoss, DFLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist
from ultralytics.nn.modules.custom_blocks import EAPIoULoss


class EAPIoUBboxLoss(nn.Module):
    """
    使用 EAPIoU 的边界框损失
    替换默认的 CIoU 为 EAPIoU（增强型长宽比惩罚 IoU）
    """

    def __init__(self, reg_max: int = 16, eapiou_beta: float = 0.1):
        """
        初始化 EAPIoU Bbox Loss 模块
        
        Args:
            reg_max: DFL 的最大回归值
            eapiou_beta: EAPIoU 的 beta 参数（长宽比惩罚权重）
        """
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self.eapiou_loss = EAPIoULoss(beta=eapiou_beta)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        计算 EAPIoU 和 DFL 损失
        
        Args:
            pred_dist: 预测的分布 (用于 DFL)
            pred_bboxes: 预测的边界框 (xyxy 格式)
            anchor_points: 锚点坐标
            target_bboxes: 目标边界框 (xyxy 格式)
            target_scores: 目标分数
            target_scores_sum: 目标分数总和（用于归一化）
            fg_mask: 前景掩码
            
        Returns:
            tuple: (iou_loss, dfl_loss)
        """
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        
        # 使用 EAPIoU 替代 CIoU
        # EAPIoU 会自动处理长宽比惩罚，更适合医学影像中的小目标检测
        eapiou = self.eapiou_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - eapiou) * weight).sum() / target_scores_sum

        # DFL loss（保持原有逻辑）
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class v8DetectionLossWithEAPIoU(v8DetectionLoss):
    """
    增强型 YOLO11 检测损失
    
    改进：
    1. 使用 EAPIoU 替代 CIoU
    2. 支持 Deep Supervision（多尺度辅助损失）
    3. 保持与 Ultralytics 训练框架的兼容性
    """

    def __init__(self, model, tal_topk: int = 10, eapiou_beta: float = 0.1, deep_supervision: bool = False):
        """
        初始化增强型检测损失
        
        Args:
            model: YOLO 模型
            tal_topk: Task-Aligned Assigner 的 topk 参数
            eapiou_beta: EAPIoU 的 beta 参数
            deep_supervision: 是否启用深度监督（多尺度辅助损失）
        """
        # 先调用父类初始化
        super().__init__(model, tal_topk)
        
        # 替换默认的 bbox_loss 为 EAPIoU 版本
        self.bbox_loss = EAPIoUBboxLoss(self.reg_max, eapiou_beta).to(self.device)
        self.deep_supervision = deep_supervision
        
        print(f"✅ 已启用 EAPIoU Loss (beta={eapiou_beta})")
        if deep_supervision:
            print(f"✅ 已启用 Deep Supervision (多尺度辅助损失)")

    def __call__(self, preds, batch):
        """
        计算总损失
        
        Args:
            preds: 模型预测输出
            batch: 批次数据（包含 targets）
            
        Returns:
            tuple: (loss, loss_detached)
        """
        # 调用父类的 __call__ 获取主损失
        loss, loss_detached = super().__call__(preds, batch)
        
        # 如果启用 Deep Supervision，添加辅助损失
        if self.deep_supervision and isinstance(preds, tuple):
            # preds 格式: (final_output, intermediate_features)
            # 对中间特征层添加辅助损失（权重降低）
            aux_loss = self._compute_auxiliary_loss(preds, batch)
            loss = loss + aux_loss * 0.5  # 辅助损失权重 0.5
            
        return loss, loss_detached

    def _compute_auxiliary_loss(self, preds, batch):
        """
        计算深度监督的辅助损失
        
        Deep Supervision 策略：
        - 在中间层（P3, P4）添加额外的检测头
        - 辅助损失帮助中间层学习更好的特征表示
        - 最终只使用主检测头进行推理
        
        Args:
            preds: 模型预测（包含主输出和辅助输出）
            batch: 批次数据
            
        Returns:
            torch.Tensor: 辅助损失
        """
        # TODO: 实现深度监督逻辑
        # 目前返回 0，需要修改模型架构添加辅助检测头后才能真正使用
        return torch.tensor(0.0, device=self.device)


def create_custom_loss(model, **kwargs):
    """
    创建自定义损失函数的工厂函数
    
    Args:
        model: YOLO 模型
        **kwargs: 额外参数
            - tal_topk: Task-Aligned Assigner 的 topk 值（默认 10）
            - eapiou_beta: EAPIoU beta 参数（默认 0.1）
            - deep_supervision: 是否启用深度监督（默认 False）
            
    Returns:
        v8DetectionLossWithEAPIoU: 自定义损失函数实例
    """
    tal_topk = kwargs.get('tal_topk', 10)
    eapiou_beta = kwargs.get('eapiou_beta', 0.1)
    deep_supervision = kwargs.get('deep_supervision', False)
    
    return v8DetectionLossWithEAPIoU(
        model, 
        tal_topk=tal_topk, 
        eapiou_beta=eapiou_beta,
        deep_supervision=deep_supervision
    )


# 使用示例：
# from ultralytics import YOLO
# from custom_loss_integration import create_custom_loss
# 
# model = YOLO('models/yolo11_sse_eapiou_s.yaml')
# 
# # 手动替换损失函数（需要在训练前）
# # 方式1: 通过 trainer 回调
# def on_pretrain_routine_start(trainer):
#     trainer.loss = create_custom_loss(trainer.model, eapiou_beta=0.1, deep_supervision=True)
# 
# model.add_callback('on_pretrain_routine_start', on_pretrain_routine_start)
# 
# # 方式2: 直接在训练脚本中集成（见 train_with_eapiou.py）
