from __future__ import annotations

"""Building blocks for TxUNet: EfficientAttention, GatedDepthwiseFFN, TransformerBlock."""

import torch
from torch import nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W] -> [B,H,W,C] -> LN -> [B,C,H,W]
        b, c, h, w = x.shape
        y = x.permute(0, 2, 3, 1)
        y = self.ln(y)
        return y.permute(0, 3, 1, 2)


class EfficientAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by heads"
        self.dim = dim
        self.num_heads = num_heads
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.q_dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.k_dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.q_dw(self.q(x))
        k = self.k_dw(self.k(x))
        v = self.v(x)
        # [B, heads, C_head, H*W]
        c_head = c // self.num_heads
        q = q.view(b, self.num_heads, c_head, h * w)
        k = k.view(b, self.num_heads, c_head, h * w)
        v = v.view(b, self.num_heads, c_head, h * w)
        # attn: [B, heads, HW, HW]
        attn = torch.einsum("bhci,bhdi->bhcd", q, k) / (c_head ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum("bhcd,bhdi->bhci", attn, v)
        out = out.contiguous().view(b, c, h, w)
        return self.proj(out)


class GatedDepthwiseFFN(nn.Module):
    def __init__(self, dim: int, expand: float = 2.66) -> None:
        super().__init__()
        hidden = int(round(dim * expand))
        # ensure even for splitting
        if hidden % 2 != 0:
            hidden += 1
        self.expand = nn.Conv2d(dim, hidden, kernel_size=1)
        self.dw = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.project = nn.Conv2d(hidden // 2, dim, kernel_size=1)
        self.act = nn.SiLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.expand(x)
        z = self.dw(z)
        u, v = torch.chunk(z, 2, dim=1)
        z = u * self.act(v)
        return self.project(z)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, expand: float) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = EfficientAttention(dim, heads)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = GatedDepthwiseFFN(dim, expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x



