"""
Transformer building blocks for Radio U-Net.

Components:
- DConvBlock: 1×1 Conv + 3×3 Depthwise Conv
- LayerNorm2d: Channel-wise LayerNorm for 2D feature maps
- EfficientGlobalAttention: Global spatial attention with DConv on Q, K, V
- GatedDepthwiseFFN: Gated FFN with two separate DConvBlock paths + internal residual
- TransformerBlock: Complete transformer block (LN → Attention + residual → FFN)
"""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


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
        u = self.branch1(x)            # [B, Hid, H, W]
        v = self.act(self.branch2(x))  # [B, Hid, H, W]
        g = u * v                       # Gated: [B, Hid, H, W]
        return self.proj(g) + x         # Internal residual: [B, C, H, W]


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
    Complete Transformer block: LayerNorm → Attention (+ residual) → FFN (has internal residual).
    
    Architecture:
        x → LN → Attention → + x (residual) → FFN (with internal residual) → output
    
    Note: 
    - Pre-LN is applied only before attention (not before FFN)
    - FFN has its own internal residual connection
    
    Input: [B, C, H, W]
    Output: [B, C, H, W]
    """
    def __init__(self, dim: int, heads: int, expand: float = 2.66, ln_eps: float = 1e-5, kv_stride: int = 1) -> None:
        super().__init__()
        self.norm = LayerNorm2d(dim, eps=ln_eps)
        self.attn = EfficientGlobalAttention(dim, heads, kv_stride=kv_stride)
        self.ffn = GatedDepthwiseFFN(dim, expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention with external residual
        x = x + self.attn(self.norm(x))
        # FFN has internal residual
        x = self.ffn(x)
        return x


class WindowedTransformerBlock(nn.Module):
    """
    Transformer block variant that applies windowed attention.
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
        self.norm = LayerNorm2d(dim, eps=ln_eps)
        self.attn = WindowAttention(EfficientGlobalAttention(dim, heads, kv_stride=kv_stride), window=window, stride=stride)
        self.ffn = GatedDepthwiseFFN(dim, expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm(x))
        x = self.ffn(x)
        return x
