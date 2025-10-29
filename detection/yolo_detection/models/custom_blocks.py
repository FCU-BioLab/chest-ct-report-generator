# ultralytics/nn/modules/custom_blocks.py
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "AAF_CT",
    "SATModule",
    "RRBBlock",
]

# ============== 初始化工具函數 ==============
def kaiming_init(module):
    """
    應用 Kaiming Normal 初始化以穩定訓練
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

# ============== AAF_CT (Lazy) ==============
class AAF_CT(nn.Module):
    """
    自動從輸入推論 c1，僅需可選的 c2（不給就保持通道不變）。
    YAML 用法：
      - [-1, 1, AAF_CT, []]        # c2=None，通道不變
      - [-1, 1, AAF_CT, [128]]     # c2=128，若輸入不是 128 會自動投影到 128
    """
    def __init__(self, c2: int | None = None, groups: int = 4):
        super().__init__()
        self.target_c = c2        # 目標輸出通道（可為 None）
        self.groups = groups
        self._built = False

    def _build(self, c1: int):
        c2 = self.target_c or c1
        self.proj = nn.Identity() if c1 == c2 else nn.Conv2d(c1, c2, 1, bias=False)
        hidden_dim = max(c2 // 4, 16)  # ✅ 提升最小值到 16，避免 SE 過度壓縮
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, c2, 1),
            nn.Sigmoid()
        )
        self.spatial = nn.Conv2d(c2, c2, 1, groups=min(self.groups, c2))  # ✅ 避免 groups > channels
        
        # ✅ Kaiming 初始化
        self.apply(kaiming_init)
        self._built = True

    def forward(self, x):
        if not self._built:
            self._build(x.shape[1])
        x = self.proj(x)
        x_se = x * self.se(x)
        return x_se + self.spatial(x_se)


# ============== SATModule (Lazy) ==============
class DeformableAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.offset = nn.Sequential(
            nn.Conv2d(dim, dim, 5, padding=2, groups=dim),
            nn.SiLU(),
            nn.Conv2d(dim, 2 * num_heads, 1)
        )
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # ✅ 輕量 Layer Norm（僅標準化，不過度削弱）
        q = F.layer_norm(q, q.shape[-2:])
        k = F.layer_norm(k, k.shape[-2:])
        
        offset = self.offset(x).view(B, self.num_heads, 2, H * W)
        # ✅ 激進降低 offset（1.5，強化穩定性）
        offset = torch.tanh(offset) * 1.5
        k = k + offset[:, :, 0, :].unsqueeze(2) * 0.2  # ✅ 進一步降低到 0.2
        v = v + offset[:, :, 1, :].unsqueeze(2) * 0.2
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # ✅ 更激進的梯度裁剪（±5 而非 ±10）
        attn = torch.clamp(attn, min=-5, max=5)
        out = (attn.softmax(dim=-1) @ v).reshape(B, C, H, W)
        return self.proj(out)

class SATModule(nn.Module):
    """
    Lazy 版。YAML：
      - [-1, 1, SATModule, []]        # 通道不變
      - [-1, 1, SATModule, [1024]]    # 需要時才變更通道
    """
    def __init__(self, c2: int | None = None, heads: int = 4, use_checkpoint: bool = False):
        super().__init__()
        self.target_c = c2
        self.heads = heads
        self.use_checkpoint = use_checkpoint  # ✅ Gradient Checkpointing 開關
        self._built = False

    def _build(self, c1: int):
        c2 = self.target_c or c1
        self.proj = nn.Identity() if c1 == c2 else nn.Conv2d(c1, c2, 1, bias=False)
        self.attn = DeformableAttention(c2, num_heads=self.heads)
        self.mlp = nn.Sequential(
            nn.Conv2d(c2, c2 * 2, 1),
            nn.SiLU(),
            nn.Conv2d(c2 * 2, c2, 1)
        )
        hidden_dim = max(c2 // 4, 16)  # ✅ 提升最小值到 16
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, c2, 1),
            nn.Sigmoid()
        )
        self._built = True

    def _forward_impl(self, x_input):
        """實際的前向傳播邏輯（用於 Checkpointing）"""
        # ✅ 注意力分支（標準 Transformer 權重）
        attn_out = self.attn(x_input) + x_input  # ✅ 標準殘差
        
        # ✅ Channel gate（標準 SE 模塊）
        gate = self.channel_gate(attn_out)
        
        # ✅ MLP 分支（標準 FFN）
        mlp_out = self.mlp(attn_out)
        
        return attn_out * gate + mlp_out  # ✅ 標準結構

    def forward(self, x):
        if not self._built:
            self._build(x.shape[1])
        x_input = self.proj(x)
        
        # ✅ 使用 Gradient Checkpointing（僅在訓練時啟用）
        if self.use_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(self._forward_impl, x_input, use_reentrant=False)
        else:
            return self._forward_impl(x_input)


# ============== RRBBlock (Lazy) ==============
class RRBBlock(nn.Module):
    """
    Lazy 版重參殘差。YAML：
      - [-1, 2, RRBBlock, []]        # 通道不變
      - [-1, 2, RRBBlock, [512]]     # 需要時才變更通道
    """
    def __init__(self, c2: int | None = None, deploy: bool = False):
        super().__init__()
        self.target_c = c2
        self.deploy = deploy
        self._built = False

    def _build(self, c1: int):
        c2 = self.target_c or c1
        self.proj = nn.Identity() if c1 == c2 else nn.Conv2d(c1, c2, 1, bias=False)

        if self.deploy:
            self.reparam_conv = nn.Conv2d(c2, c2, 3, padding=1, bias=True)
        else:
            # ✅ 使用 PyTorch 默認 BN 參數（更穩定）
            self.branch_3x3 = nn.Sequential(
                nn.Conv2d(c2, c2, 3, padding=1, bias=False),
                nn.BatchNorm2d(c2, eps=1e-5, momentum=0.1)  # ✅ PyTorch 默認值
            )
            self.branch_1x1 = nn.Sequential(
                nn.Conv2d(c2, c2, 1, bias=False),
                nn.BatchNorm2d(c2, eps=1e-5, momentum=0.1)
            )
            self.branch_identity = nn.BatchNorm2d(c2, eps=1e-5, momentum=0.1)
        self._built = True

    def forward(self, x):
        if not self._built:
            self._build(x.shape[1])
        x_in = self.proj(x)
        if self.deploy:
            return F.silu(self.reparam_conv(x_in)) + x_in
        # ✅ 標準 RepVGG 結構：重參數化分支 + 殘差連接
        out = self.branch_3x3(x_in) + self.branch_1x1(x_in) + self.branch_identity(x_in)
        return F.silu(out) + x_in  # ✅ 關鍵修復：添加殘差連接

    # 其餘 fuse 函式同你現有版本（不重貼）
