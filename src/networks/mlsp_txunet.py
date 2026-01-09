from __future__ import annotations, annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import nn

"""
Transformer building blocks for Radio U-Net.

Components:
- DConvBlock: 1×1 Conv + 3×3 Depthwise Conv
- LayerNorm2d: Channel-wise LayerNorm for 2D feature maps
- EfficientGlobalAttention: Global spatial attention with DConv on Q, K, V
- GatedDepthwiseFFN: Gated FFN with two separate DConvBlock paths + internal residual
- TransformerBlock: Complete transformer block (LN → Attention + residual → FFN)
"""


# ---------- DConv Block ----------

class DConvBlock(nn.Module):
    """
    1×1 pointwise convolution followed by 3×3 depthwise convolution.

    This is the basic building block used in attention (for Q, K, V) and FFN.
    """
    
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)
        self.dw = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, groups=out_ch, bias=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dw(self.conv1x1(x))


# ---------- Normalization ----------

class LayerNorm2d(nn.Module):
    """
    Channel-wise LayerNorm over [C] at each spatial location.

    Stability:
    - Upcasts to float32 inside LN to avoid tiny-eps issues in AMP/bfloat16.
    - eps=1e-5 (safer than 1e-6 for half precision).

    Input: [B, C, H, W]
    Output: [B, C, H, W]
    """
    
    def __init__(self, num_channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype_in = x.dtype
        # Permute to [B, H, W, C] for LayerNorm
        y = x.permute(0, 2, 3, 1).to(torch.float32)
        y = F.layer_norm(y, (y.shape[-1],), self.weight.to(y.dtype), self.bias.to(y.dtype), self.eps)
        y = y.to(dtype_in).permute(0, 3, 1, 2)
        return y


# ---------- Attention ----------

class EfficientGlobalAttention(nn.Module):
    """
    Global spatial attention with DConvBlock on Q, K, V.

    Architecture:
        X → DConvBlock → Q
        X → DConvBlock → K
        X → DConvBlock → V
        Q, K, V → Scaled Dot-Product Attention → 1×1 Proj → Output

    Features:
    - Uses Flash Attention / Memory-Efficient kernels on CUDA via SDPA
    - Falls back to streaming softmax on CPU for memory efficiency

    Input: [B, C, H, W]
    Output: [B, C, H, W]
    """
    
    def __init__(self, dim: int, num_heads: int, kv_stride: int = 1) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.h = num_heads
        self.d = dim // num_heads
        self.kv_stride = int(kv_stride)
        
        # DConvBlocks for Q, K, V (all three always have depthwise)
        self.q_block = DConvBlock(dim, dim)
        self.k_block = DConvBlock(dim, dim)
        self.v_block = DConvBlock(dim, dim)
        
        # Output projection
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
    
    @torch.no_grad()
    def _choose_chunks(self, T: int, d: int, bytes_per_el: int = 4) -> tuple[int, int]:
        """
        Heuristic chunk sizes for streaming softmax to keep memory in check.
        Aims for ~64MB per inner matmul slice.
        """
        target_bytes = 64 * 1024 * 1024
        k_chunk = min(T, max(512, (target_bytes // (d * bytes_per_el)) // 64 * 64))
        q_chunk = k_chunk
        return int(q_chunk), int(k_chunk)
    
    def _attn_streaming(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Memory-efficient global attention via streaming softmax (online log-sum-exp).

        Supports differing sequence lengths for Q and K/V.
        Input: q [Bh, Tq, d], k/v [Bh, Tk, d]
        Output: [Bh, Tq, d]
        """
        Bh, Tq, d = q.shape
        Tk = k.shape[1]
        device = q.device
        dtype = q.dtype
        
        q_chunk, k_chunk = self._choose_chunks(Tk, d, bytes_per_el=2 if dtype in (torch.float16, torch.bfloat16) else 4)
        
        out = torch.empty_like(q)
        scale = 1.0 / math.sqrt(d)
        
        # Process queries in blocks
        for qs in range(0, Tq, q_chunk):
            qe = min(qs + q_chunk, Tq)
            qi = q[:, qs:qe, :]  # [Bh, Qc, d]
            
            # Row-wise streaming stats
            m_i = torch.full((Bh, qe - qs, 1), -float("inf"), device=device, dtype=dtype)
            s_i = torch.zeros((Bh, qe - qs, 1), device=device, dtype=dtype)
            o_i = torch.zeros((Bh, qe - qs, d), device=device, dtype=dtype)
            
            # Sweep over keys in blocks
            for ks in range(0, Tk, k_chunk):
                ke = min(ks + k_chunk, Tk)
                kj = k[:, ks:ke, :]  # [Bh, Kc, d]
                vj = v[:, ks:ke, :]  # [Bh, Kc, d]
                
                # Logits: [Bh, Qc, Kc]
                logits = torch.einsum("bid,bjd->bij", qi, kj) * scale
                
                # Streaming softmax: update (m_i, s_i, o_i)
                block_max = logits.max(dim=-1, keepdim=True).values  # [Bh, Qc, 1]
                new_m = torch.maximum(m_i, block_max)  # [Bh, Qc, 1]
                
                # Renormalize previous stats
                s_i = s_i * torch.exp(m_i - new_m)
                o_i = o_i * torch.exp(m_i - new_m)
                
                # Current block contributions
                p = torch.exp(logits - new_m)  # [Bh, Qc, Kc]
                s_i = s_i + p.sum(dim=-1, keepdim=True)  # [Bh, Qc, 1]
                o_i = o_i + torch.einsum("bij,bjd->bid", p, vj)  # [Bh, Qc, d]
                
                m_i = new_m
            
            out[:, qs:qe, :] = o_i / s_i  # [Bh, Qc, d]
        
        return out
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: x [B, C, H, W]
        Output: [B, C, H, W]
        """
        B, C, H, W = x.shape
        h, d = self.h, self.d
        Tq = H * W
        kv_stride = self.kv_stride
        k_input = x if kv_stride <= 1 else F.avg_pool2d(x, kernel_size=kv_stride, stride=kv_stride)
        
        # Compute Q, K, V via DConvBlocks
        q = self.q_block(x)  # [B, C, H, W]
        k = self.k_block(k_input)  # [B, C, Hk, Wk]
        v = self.v_block(k_input)  # [B, C, Hk, Wk]
        Hk, Wk = k.shape[2], k.shape[3]
        Tk = Hk * Wk
        
        # Use PyTorch SDPA on CUDA (auto-selects Flash/MemEff based on dtype)
        # Flash Attention requires FP16/BF16 - use torch.amp.autocast in training
        if x.is_cuda:
            # Reshape to [B, h, T, d]
            def to_bhtd_q(t: torch.Tensor) -> torch.Tensor:
                return t.view(B, h, d, H, W).permute(0, 1, 3, 4, 2).contiguous().view(B, h, Tq, d)
            
            def to_bhtd_kv(t: torch.Tensor) -> torch.Tensor:
                return t.view(B, h, d, Hk, Wk).permute(0, 1, 3, 4, 2).contiguous().view(B, h, Tk, d)
            
            q_bhtd = to_bhtd_q(q)
            k_bhtd = to_bhtd_kv(k)
            v_bhtd = to_bhtd_kv(v)
            
            # SDPA auto-selects best backend (Flash for FP16/BF16, else fallback)
            out_bhtd = F.scaled_dot_product_attention(q_bhtd, k_bhtd, v_bhtd, dropout_p=0.0, is_causal=False)
            
            # Back to [B, C, H, W]
            out = out_bhtd.view(B, h, H, W, d).permute(0, 1, 4, 2, 3).contiguous().view(B, C, H, W)
            return self.proj(out)
        
        # Fallback: streaming softmax attention on CPU
        def to_bhd_q(t: torch.Tensor) -> torch.Tensor:
            t = t.view(B, h, d, H, W).permute(0, 1, 3, 4, 2).contiguous()
            return t.view(B * h, Tq, d)
        
        def to_bhd_kv(t: torch.Tensor) -> torch.Tensor:
            t = t.view(B, h, d, Hk, Wk).permute(0, 1, 3, 4, 2).contiguous()
            return t.view(B * h, Tk, d)
        
        q_bhd = to_bhd_q(q)
        k_bhd = to_bhd_kv(k)
        v_bhd = to_bhd_kv(v)
        out_bhd = self._attn_streaming(q_bhd, k_bhd, v_bhd)  # [B*h, Tq, d]
        out = out_bhd.view(B, h, H, W, d).permute(0, 1, 4, 2, 3).contiguous().view(B, C, H, W)
        return self.proj(out)


# ---------- Feed-Forward Network ----------

class GatedDepthwiseFFN(nn.Module):
    """
    Gated FFN with two SEPARATE DConvBlock paths and internal residual.

    Architecture:
        Input Y → DConvBlock → u
        Input Y → DConvBlock → GELU → v
        Gate: g = u × v (Hadamard product)
        Output: 1×1 Conv(g) + Y (internal residual)

    Input: [B, C, H, W]
    Output: [B, C, H, W]
    """
    
    def __init__(self, dim: int, expand: float = 2.66) -> None:
        super().__init__()
        hidden = int(round(dim * expand))
        
        # Two SEPARATE DConvBlock paths
        self.branch1 = DConvBlock(dim, hidden)  # u path (no activation)
        self.branch2 = DConvBlock(dim, hidden)  # v path (goes through GELU)
        
        self.act = nn.GELU()
        self.proj = nn.Conv2d(hidden, dim, kernel_size=1, bias=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.branch1(x)  # [B, Hid, H, W]
        # Clamp u to prevent explosion before multiplicative gate
        u = torch.clamp(u, -256.0, 256.0)
        v = self.act(self.branch2(x))  # [B, Hid, H, W]
        g = u * v  # Gated: [B, Hid, H, W]
        return self.proj(g) + x  # Internal residual: [B, C, H, W]


# ---------- Windowed Attention Wrapper ----------

class WindowAttention(nn.Module):
    """
    Wrap an attention module to operate over windows.

    Splits (B, C, H, W) into windows of size `window` with stride `stride`
    (non-overlapping if stride==window), applies the wrapped attention per
    window, then stitches back and crops to the original spatial size.
    """
    
    def __init__(self, attn: nn.Module, window: int = 8, stride: int | None = None) -> None:
        super().__init__()
        self.attn = attn
        self.window = int(window)
        self.stride = int(stride) if stride is not None else int(window)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ws, st = self.window, self.stride
        
        # Pad on bottom/right so unfold covers the full tensor
        pad_h = (st - H % st) % st
        pad_w = (st - W % st) % st
        x_pad = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, Hp, Wp = x_pad.shape
        
        # Unfold to windows: [B, C, nH, nW, ws, ws]
        x_win = x_pad.unfold(2, ws, st).unfold(3, ws, st).contiguous()
        nH, nW = x_win.shape[2], x_win.shape[3]
        x_win = x_win.view(B * nH * nW, C, ws, ws)
        
        # Attention per window
        y_win = self.attn(x_win)
        
        # Fold back and crop to original size
        y = y_win.view(B, nH, nW, C, ws, ws).permute(0, 3, 1, 4, 2, 5).contiguous()
        y = y.view(B, C, nH * ws, nW * ws)
        return y[:, :, :H, :W]


# ---------- Transformer Block ----------

class TransformerBlock(nn.Module):
    """
    Complete Transformer block: Pre-LN → Attention (+ residual) → Pre-LN → FFN (has internal residual).

    Architecture:
        x → LN1 → Attention → + x (residual) → LN2 → FFN (with internal residual) → output

    Note:
    - Pre-LN is applied before both attention AND FFN for stability
    - FFN has its own internal residual connection

    Input: [B, C, H, W]
    Output: [B, C, H, W]
    """
    
    def __init__(self, dim: int, heads: int, expand: float = 2.66, ln_eps: float = 1e-5, kv_stride: int = 1) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(dim, eps=ln_eps)  # Before attention
        self.norm2 = LayerNorm2d(dim, eps=ln_eps)  # Before FFN (for stability)
        self.attn = EfficientGlobalAttention(dim, heads, kv_stride=kv_stride)
        self.ffn = GatedDepthwiseFFN(dim, expand)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention with external residual
        x = x + self.attn(self.norm1(x))
        # FFN with Pre-LN for stability (FFN has internal residual)
        x = self.ffn(self.norm2(x))
        return x


class WindowedTransformerBlock(nn.Module):
    """
    Transformer block variant that applies windowed attention.
    Uses Pre-LN before both attention and FFN for stability.
    """
    
    def __init__(
        self,
        dim: int,
        heads: int,
        expand: float = 2.66,
        ln_eps: float = 1e-5,
        window: int = 8,
        stride: int | None = None,
        kv_stride: int = 1,
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(dim, eps=ln_eps)  # Before attention
        self.norm2 = LayerNorm2d(dim, eps=ln_eps)  # Before FFN (for stability)
        self.attn = WindowAttention(
            EfficientGlobalAttention(dim, heads, kv_stride=kv_stride), window=window, stride=stride
        )
        self.ffn = GatedDepthwiseFFN(dim, expand)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = self.ffn(self.norm2(x))
        return x


"""
Radio U-Net with Transformer Blocks (TxUNetModel).

A Transformer-based U-Net architecture for radio map prediction.

Architecture:
    Input [B, 4, H, W] → Stem → Encoder (4 levels) → Decoder (4 levels) → Head → Output [B, 1, H, W]

Key features:
- 4-channel input: Reflection, Transmission, Distance, Pathloss samples
- Transformer blocks with efficient global attention at each level
- Skip connections: level 0 uses concat only, levels 1-2 use 1×1 conv after concat
- F₀ residual: stem output is added back before final output
- Decoder level 0: N₁ blocks + 1 extra single block
"""


def make_blocks(
    dim: int,
    num: int,
    heads: int,
    expand: float,
    ln_eps: float,
    block_cls: type[nn.Module] = TransformerBlock,
    block_kwargs: dict | None = None,
) -> nn.Sequential:
    """
    Create a sequence of TransformerBlocks.

    Args:
        dim: Channel dimension
        num: Number of blocks
        heads: Number of attention heads
        expand: FFN expansion ratio
        ln_eps: LayerNorm epsilon
        block_cls: Block class to instantiate (default: TransformerBlock)
        block_kwargs: Optional kwargs passed to each block

    Returns:
        nn.Sequential of TransformerBlocks
    """
    block_kwargs = block_kwargs or {}
    return nn.Sequential(
        *[
            block_cls(dim, heads, expand, ln_eps=ln_eps, **block_kwargs)
            for _ in range(num)
        ]
    )


class Downsample(nn.Module):
    """
    2× spatial downsampling with channel expansion.

    Uses stride-2 3×3 convolution.

    Input: [B, in_ch, H, W]
    Output: [B, out_ch, H/2, W/2]
    """
    
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """
    2× spatial upsampling with channel reduction.

    Uses nearest neighbor upsampling followed by 1×1 convolution.

    Input: [B, in_ch, H, W]
    Output: [B, out_ch, 2H, 2W]
    """
    
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.reduce = nn.Conv2d(in_ch, out_ch, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reduce(self.up(x))


class TxUNetModel(nn.Module):
    """
    Transformer-based U-Net for radio map prediction.

    Architecture overview:
        - Stem: 3×3 conv (in_ch → C), produces F₀
        - Encoder: 4 levels with Transformer blocks and downsampling
            - Level 0: C channels, N₁ blocks
            - Level 1: 2C channels, N₂ blocks
            - Level 2: 4C channels, N₃ blocks
            - Bottleneck: 8C channels, N₄ blocks
        - Decoder: 4 levels with upsampling and skip connections
            - Level 2: concat + 1×1 fuse, N₃ blocks
            - Level 1: concat + 1×1 fuse, N₂ blocks
            - Level 0: concat only (no fuse), N₁ blocks + 1 extra block
        - Head: 3×3 conv (2C → C) → Add F₀ → 3×3 conv (C → out_ch)

    Args:
        in_ch: Input channels (default 4: R, T, D, P)
        out_ch: Output channels (default 1: predicted pathloss)
        base_ch: Base channel count C (default 48)
        depths: Transformer blocks per level [N₁, N₂, N₃, N₄] (default (4, 6, 6, 8))
        heads: Attention heads per level (default (4, 4, 8, 8))
        expand: FFN expansion ratio (default 2.66)
        use_checkpoint: Enable gradient checkpointing (default True)
        ln_eps: LayerNorm epsilon (default 1e-5)
    """
    
    def __init__(
        self,
        in_ch: int = 4,
        out_ch: int = 1,
        base_ch: int = 48,
        depths: Sequence[int] = (4, 6, 6, 8),
        heads: Sequence[int] = (4, 4, 8, 8),
        expand: float = 2.66,
        use_checkpoint: bool = True,
        ln_eps: float = 1e-5,
        window0: int | None = None,
        window0_stride: int | None = None,
        sra0_enabled: bool = False,
        sra0_stride: int = 2,
    ) -> None:
        super().__init__()
        c = base_ch
        self.use_checkpoint = use_checkpoint
        self.ln_eps = float(ln_eps)
        
        # ============ STEM ============
        # 3×3 conv: in_ch → C
        # Produces F₀ which is used for residual in head
        self.stem = nn.Conv2d(in_ch, c, kernel_size=3, padding=1)
        
        # ============ ENCODER ============
        # Level 0: [B, C, H, W]
        enc0_cls: type[nn.Module]
        enc0_kwargs: dict | None = None
        if sra0_enabled:
            enc0_cls = TransformerBlock
            enc0_kwargs = {"kv_stride": sra0_stride}
        elif window0:
            enc0_cls = WindowedTransformerBlock
            enc0_kwargs = {"window": window0, "stride": window0_stride}
        else:
            enc0_cls = TransformerBlock
        self.enc0 = make_blocks(
            c, depths[0], heads[0], expand, self.ln_eps, block_cls=enc0_cls, block_kwargs=enc0_kwargs
        )
        
        # Downsample: C → 2C, spatial /2
        self.down1 = Downsample(c, 2 * c)
        
        # Level 1: [B, 2C, H/2, W/2]
        self.enc1 = make_blocks(2 * c, depths[1], heads[1], expand, self.ln_eps)
        
        # Downsample: 2C → 4C, spatial /2
        self.down2 = Downsample(2 * c, 4 * c)
        
        # Level 2: [B, 4C, H/4, W/4]
        self.enc2 = make_blocks(4 * c, depths[2], heads[2], expand, self.ln_eps)
        
        # Downsample: 4C → 8C, spatial /2
        self.down3 = Downsample(4 * c, 8 * c)
        
        # Bottleneck (Level 3): [B, 8C, H/8, W/8]
        self.enc3 = make_blocks(8 * c, depths[3], heads[3], expand, self.ln_eps)
        
        # ============ SKIP CONNECTIONS ============
        # Level 0: NO skip conv (concat only)
        # Level 1: 1×1 conv on skip features BEFORE concat
        self.skip1 = nn.Conv2d(2 * c, 2 * c, kernel_size=1)
        # Level 2: 1×1 conv on skip features BEFORE concat
        self.skip2 = nn.Conv2d(4 * c, 4 * c, kernel_size=1)
        
        # ============ DECODER ============
        # Level 2 decoder
        # Upsample: 8C → 4C
        self.up3 = Upsample(8 * c, 4 * c)
        # Fuse after concat: 8C → 4C
        self.fuse2 = nn.Conv2d(8 * c, 4 * c, kernel_size=1)
        self.dec2 = make_blocks(4 * c, depths[2], heads[2], expand, self.ln_eps)
        
        # Level 1 decoder
        # Upsample: 4C → 2C
        self.up2 = Upsample(4 * c, 2 * c)
        # Fuse after concat: 4C → 2C
        self.fuse1 = nn.Conv2d(4 * c, 2 * c, kernel_size=1)
        self.dec1 = make_blocks(2 * c, depths[1], heads[1], expand, self.ln_eps)
        
        # Level 0 decoder (special handling)
        # Upsample: 2C → C
        self.up1 = Upsample(2 * c, c)
        # NO fuse0 - just concat (C + C = 2C)
        # N₁ transformer blocks at 2C
        if sra0_enabled:
            dec0_cls = TransformerBlock
            dec0_kwargs = {"kv_stride": sra0_stride}
        elif window0:
            dec0_cls = WindowedTransformerBlock
            dec0_kwargs = {"window": window0, "stride": window0_stride}
        else:
            dec0_cls = TransformerBlock
            dec0_kwargs = None
        self.dec0 = make_blocks(
            2 * c, depths[0], heads[0], expand, self.ln_eps, block_cls=dec0_cls, block_kwargs=dec0_kwargs
        )
        # +1 extra single transformer block
        self.dec0_extra = dec0_cls(2 * c, heads[0], expand, ln_eps=self.ln_eps, **(dec0_kwargs or {}))
        
        # ============ HEAD ============
        # 3×3 conv: 2C → C
        self.head_conv1 = nn.Conv2d(2 * c, c, kernel_size=3, padding=1)
        # After adding F₀ residual: 3×3 conv: C → out_ch
        self.head_conv2 = nn.Conv2d(c, out_ch, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, in_ch, H, W]

        Returns:
            Output tensor [B, out_ch, H, W]
        """
        # ============ STEM ============
        # Save F₀ for residual connection in head
        f0 = self.stem(x)  # [B, C, H, W]
        
        # ============ ENCODER ============
        # Level 0
        x0 = self._run_blocks(self.enc0, f0)  # [B, C, H, W]
        # Note: x0 is used directly for skip (no 1×1 conv)
        
        # Level 1
        x1 = self.down1(x0)  # [B, 2C, H/2, W/2]
        x1 = self._run_blocks(self.enc1, x1)
        s1 = self.skip1(x1)  # 1×1 on skip
        
        # Level 2
        x2 = self.down2(x1)  # [B, 4C, H/4, W/4]
        x2 = self._run_blocks(self.enc2, x2)
        s2 = self.skip2(x2)  # 1×1 on skip
        
        # Bottleneck (Level 3)
        x3 = self.down3(x2)  # [B, 8C, H/8, W/8]
        x3 = self._run_blocks(self.enc3, x3)
        
        # ============ DECODER ============
        # Level 2 decoder
        y2 = self.up3(x3)  # [B, 4C, H/4, W/4]
        y2 = torch.cat([y2, s2], dim=1)  # [B, 8C, H/4, W/4]
        y2 = self.fuse2(y2)  # [B, 4C, H/4, W/4]
        y2 = self._run_blocks(self.dec2, y2)
        
        # Level 1 decoder
        y1 = self.up2(y2)  # [B, 2C, H/2, W/2]
        y1 = torch.cat([y1, s1], dim=1)  # [B, 4C, H/2, W/2]
        y1 = self.fuse1(y1)  # [B, 2C, H/2, W/2]
        y1 = self._run_blocks(self.dec1, y1)
        
        # Level 0 decoder (special handling)
        y0 = self.up1(y1)  # [B, C, H, W]
        y0 = torch.cat([y0, x0], dim=1)  # [B, 2C, H, W] - concat only, no skip conv
        y0 = self._run_blocks(self.dec0, y0)  # N₁ transformer blocks
        y0 = self._run_single_block(self.dec0_extra, y0)  # +1 extra block
        
        # ============ HEAD ============
        y0 = self.head_conv1(y0)  # [B, C, H, W]
        y0 = y0 + f0  # Add F₀ residual
        out = self.head_conv2(y0)  # [B, out_ch, H, W]
        
        return out
    
    def _run_blocks(self, seq: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
        """
        Run a sequence of blocks with optional gradient checkpointing.
        """
        if not self.use_checkpoint or not self.training:
            return seq(x)
        for m in seq:
            x = checkpoint.checkpoint(m, x, use_reentrant=False)
        return x
    
    def _run_single_block(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        Run a single block with optional gradient checkpointing.
        """
        if not self.use_checkpoint or not self.training:
            return block(x)
        return checkpoint.checkpoint(block, x, use_reentrant=False)
