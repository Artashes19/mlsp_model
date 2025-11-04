from __future__ import annotations

"""A pragmatic Restormer-Port for radio maps.

This implementation approximates MDTA by computing channel-attention weights per head
from spatially-averaged Q/K, then applying them across all spatial positions. It keeps
GDFN and pixel-(un)shuffle as per the design intent, while avoiding O(HW*C^2) costs.
"""

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import numbers

# Canonical Restormer LayerNorm (token-wise over channels)
def to_3d(x: torch.Tensor) -> torch.Tensor:
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim: int, LayerNorm_type: str = 'WithBias') -> None:
        super().__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class FeedForward(nn.Module):
    def __init__(self, dim: int, ffn_expansion_factor: float, bias: bool) -> None:
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, bias: bool) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class RestormerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, expand: float, bias: bool = False) -> None:
        super().__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = Attention(dim, heads, bias)
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.ffn = FeedForward(dim, expand, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class PixelUnshuffle2x(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(2)
        self.proj = nn.Conv2d(in_ch * 4, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.unshuffle(x))


class PixelShuffle2x(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch * 4, 1)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.proj(x))


def make_blocks(dim: int, n: int, heads: int, expand: float) -> nn.Sequential:
    return nn.Sequential(*[RestormerBlock(dim, heads, expand) for _ in range(n)])


class RestormerPort(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 1,
        base_ch: int = 48,
        heads: list[int] | tuple[int, int, int, int] = (1, 2, 4, 8),
        refinement_blocks: int = 4,
        expand: float = 2.66,
        depth_per_level: tuple[int, int, int, int] = (2, 2, 2, 2),
    ) -> None:
        super().__init__()
        c = base_ch
        h0, h1, h2, h3 = heads
        d0, d1, d2, d3 = depth_per_level

        # Stem
        self.stem = nn.Conv2d(in_ch, c, 3, padding=1)

        # Encoder levels
        self.l0 = make_blocks(c, d0, h0, expand)
        self.down1 = PixelUnshuffle2x(c, 2 * c)
        self.l1 = make_blocks(2 * c, d1, h1, expand)
        self.down2 = PixelUnshuffle2x(2 * c, 4 * c)
        self.l2 = make_blocks(4 * c, d2, h2, expand)
        self.down3 = PixelUnshuffle2x(4 * c, 8 * c)
        self.l3 = make_blocks(8 * c, d3, h3, expand)  # bottleneck

        # Decoder levels
        self.up3 = PixelShuffle2x(8 * c, 4 * c)
        self.dec2 = make_blocks(4 * c, d2, h2, expand)
        self.up2 = PixelShuffle2x(4 * c, 2 * c)
        self.dec1 = make_blocks(2 * c, d1, h1, expand)
        self.up1 = PixelShuffle2x(2 * c, c)
        self.dec0 = make_blocks(c, d0, h0, expand)

        # Refinement at full res
        self.refine = make_blocks(c, refinement_blocks, h0, expand) if refinement_blocks > 0 else nn.Identity()

        # Head
        self.head = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(c, out_ch, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(x)
        x0 = self.l0(x0)

        x1 = self.down1(x0)
        x1 = self.l1(x1)

        x2 = self.down2(x1)
        x2 = self.l2(x2)

        x3 = self.down3(x2)
        x3 = self.l3(x3)

        y2 = self.up3(x3) + x2
        y2 = self.dec2(y2)

        y1 = self.up2(y2) + x1
        y1 = self.dec1(y1)

        y0 = self.up1(y1) + x0
        y0 = self.dec0(y0)

        y0 = self.refine(y0)
        out = self.head(y0)
        return out


