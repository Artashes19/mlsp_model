from __future__ import annotations, annotations

from typing import Sequence

import math
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


def _apply_rotary_2d(
    x: torch.Tensor,
    cos_row: torch.Tensor,
    sin_row: torch.Tensor,
    cos_col: torch.Tensor,
    sin_col: torch.Tensor,
    d_row: int,
) -> torch.Tensor:
    """
    Apply 2D rotary embedding to the last dimension of x.

    Works with shapes like [B, h, T, d] or [B*h, T, d].
    """
    x_row = x[..., :d_row]
    x_col = x[..., d_row:]

    half_r = d_row // 2
    r1, r2 = x_row[..., :half_r], x_row[..., half_r:]
    x_row = torch.cat(
        [
            r1 * cos_row - r2 * sin_row,
            r1 * sin_row + r2 * cos_row,
        ],
        dim=-1,
    )

    half_c = x_col.shape[-1] // 2
    c1, c2 = x_col[..., :half_c], x_col[..., half_c:]
    x_col = torch.cat(
        [
            c1 * cos_col - c2 * sin_col,
            c1 * sin_col + c2 * cos_col,
        ],
        dim=-1,
    )

    return torch.cat([x_row, x_col], dim=-1)


class RotaryEmbedding2D(nn.Module):
    """
    2D Rotary Position Embedding for spatial attention.

    Stores inverse frequency vectors as non-persistent buffers so state_dict
    stays compatible with checkpoints from models without RoPE.
    """

    def __init__(self, d_head: int, base: float = 10000.0) -> None:
        super().__init__()
        self.d_head = int(d_head)
        self.d_row = self.d_head // 2
        self.d_col = self.d_head - self.d_row

        inv_freq_row = 1.0 / (base ** (torch.arange(0, self.d_row, 2).float() / max(self.d_row, 1)))
        inv_freq_col = 1.0 / (base ** (torch.arange(0, self.d_col, 2).float() / max(self.d_col, 1)))
        self.register_buffer("inv_freq_row", inv_freq_row, persistent=False)
        self.register_buffer("inv_freq_col", inv_freq_col, persistent=False)

    def forward(self, x: torch.Tensor, H: int, W: int, stride: int = 1) -> torch.Tensor:
        """
        Apply 2D RoPE rotation to x where T == H * W in row-major order.
        """
        T = x.shape[-2]
        if T != H * W:
            raise ValueError(f"Expected T == H*W ({H*W}), got T={T}")

        device = x.device
        rows = torch.arange(H, device=device, dtype=torch.float32) * float(stride)
        cols = torch.arange(W, device=device, dtype=torch.float32) * float(stride)

        inv_freq_row = self.inv_freq_row.to(device=device)
        inv_freq_col = self.inv_freq_col.to(device=device)

        angles_row = rows.unsqueeze(-1) * inv_freq_row
        angles_col = cols.unsqueeze(-1) * inv_freq_col

        cos_row = angles_row.cos().unsqueeze(1).expand(H, W, -1).reshape(H * W, -1)
        sin_row = angles_row.sin().unsqueeze(1).expand(H, W, -1).reshape(H * W, -1)
        cos_col = angles_col.cos().unsqueeze(0).expand(H, W, -1).reshape(H * W, -1)
        sin_col = angles_col.sin().unsqueeze(0).expand(H, W, -1).reshape(H * W, -1)

        return _apply_rotary_2d(
            x,
            cos_row.to(dtype=x.dtype),
            sin_row.to(dtype=x.dtype),
            cos_col.to(dtype=x.dtype),
            sin_col.to(dtype=x.dtype),
            self.d_row,
        )


# ---------- NSA Building Blocks ----------

class PatchCompressor(nn.Module):
    """
    Compress p*p tokens per patch into a single summary token.

    Uses learnable intra-block positional encoding + MLP + mean pool.

    Input:  [Bh, n_patches, p*p, d]
    Output: [Bh, n_patches, d]
    """

    def __init__(self, d_head: int, patch_size: int) -> None:
        super().__init__()
        pp = int(patch_size) * int(patch_size)
        self.pos_emb = nn.Parameter(torch.randn(pp, d_head) * 0.02)
        self.mlp = nn.Sequential(
            nn.Linear(d_head, d_head),
            nn.GELU(),
            nn.Linear(d_head, d_head),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_emb
        x = self.mlp(x)
        return x.mean(dim=2)


class NSA2DAttention(nn.Module):
    """
    2D Native Sparse Attention with compression, selection, and window branches.

    Input:  [B, C, H, W]
    Output: [B, C, H, W]
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        patch_size: int = 4,
        top_n: int = 16,
        window_size: int = 8,
        rope_enabled: bool = False,
        rope_base: float = 10000.0,
        gqa_group_size: int = 1,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        if num_heads % gqa_group_size != 0:
            raise ValueError("num_heads must be divisible by gqa_group_size")

        self.dim = int(dim)
        self.h_q = int(num_heads)
        self.gqa_group_size = int(gqa_group_size)
        self.h_kv = self.h_q // self.gqa_group_size
        self.d = self.dim // self.h_q
        # Keep self.h as alias for h_q for backward compat
        self.h = self.h_q
        self.patch_size = int(patch_size)
        self.top_n = int(top_n)
        self.window_size = int(window_size)
        if self.patch_size <= 0:
            raise ValueError("patch_size must be > 0")
        if self.window_size <= 0:
            raise ValueError("window_size must be > 0")
        if self.top_n <= 0:
            raise ValueError("top_n must be > 0")

        dim_kv = self.h_kv * self.d  # C_kv = C / G

        self.q_block = DConvBlock(dim, dim)
        self.k_block = DConvBlock(dim, dim_kv)
        self.v_block = DConvBlock(dim, dim_kv)

        self.compress_k = PatchCompressor(self.d, self.patch_size)
        self.compress_v = PatchCompressor(self.d, self.patch_size)

        gate_hidden = max(self.dim // 4, 1)
        self.gate = nn.Sequential(
            nn.Linear(self.dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 3),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.gate[-2].bias)

        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.rope = RotaryEmbedding2D(self.d, base=rope_base) if rope_enabled else None

    @staticmethod
    def _to_bhtd(t: torch.Tensor, B: int, h: int, d: int, H: int, W: int) -> torch.Tensor:
        return t.view(B, h, d, H, W).permute(0, 1, 3, 4, 2).contiguous().view(B, h, H * W, d)

    @staticmethod
    def _to_bcHW(t: torch.Tensor, B: int, h: int, d: int, H: int, W: int) -> torch.Tensor:
        C = h * d
        return t.view(B, h, H, W, d).permute(0, 1, 4, 2, 3).contiguous().view(B, C, H, W)

    def _partition_patches(self, t: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Partition [Bh, d, H, W] into [Bh, n_patches, p*p, d].
        """
        Bh, d, _, _ = t.shape
        p = self.patch_size
        if H % p != 0 or W % p != 0:
            raise ValueError(f"H and W must be divisible by patch_size={p}, got {(H, W)}")
        nH, nW = H // p, W // p
        t = t.view(Bh, d, nH, p, nW, p)
        t = t.permute(0, 2, 4, 3, 5, 1).contiguous()
        return t.view(Bh, nH * nW, p * p, d)

    @staticmethod
    def _bhtd_to_patches(x: torch.Tensor, B: int, h: int, H: int, W: int, p: int) -> torch.Tensor:
        """Partition [B, h, H*W, d] -> [B, h, nH*nW, p*p, d] with correct spatial layout."""
        d = x.shape[-1]
        nH, nW = H // p, W // p
        return (
            x.view(B, h, H, W, d)
            .view(B, h, nH, p, nW, p, d)
            .permute(0, 1, 2, 4, 3, 5, 6)
            .contiguous()
            .view(B, h, nH * nW, p * p, d)
        )

    def _rope_at_centers(self, x: torch.Tensor, nH: int, nW: int) -> torch.Tensor:
        """
        Apply 2D RoPE using patch-center positions.

        x: [B, h, n_patches, d]
        """
        if self.rope is None:
            return x
        B, h, T, d = x.shape
        device = x.device
        p = self.patch_size

        center_rows = torch.arange(nH, device=device, dtype=torch.float32) * float(p) + float(p // 2)
        center_cols = torch.arange(nW, device=device, dtype=torch.float32) * float(p) + float(p // 2)

        inv_freq_row = self.rope.inv_freq_row.to(device=device)
        inv_freq_col = self.rope.inv_freq_col.to(device=device)

        angles_row = center_rows.unsqueeze(-1) * inv_freq_row
        angles_col = center_cols.unsqueeze(-1) * inv_freq_col

        cos_row = angles_row.cos().unsqueeze(1).expand(nH, nW, -1).reshape(T, -1)
        sin_row = angles_row.sin().unsqueeze(1).expand(nH, nW, -1).reshape(T, -1)
        cos_col = angles_col.cos().unsqueeze(0).expand(nH, nW, -1).reshape(T, -1)
        sin_col = angles_col.sin().unsqueeze(0).expand(nH, nW, -1).reshape(T, -1)

        return _apply_rotary_2d(
            x,
            cos_row.to(dtype=x.dtype),
            sin_row.to(dtype=x.dtype),
            cos_col.to(dtype=x.dtype),
            sin_col.to(dtype=x.dtype),
            self.rope.d_row,
        )

    def _rope_at_positions(self, x: torch.Tensor, row_pos: torch.Tensor, col_pos: torch.Tensor) -> torch.Tensor:
        """
        Apply 2D RoPE to x at arbitrary row/col positions.

        x can be [B, h, T, d] or [h, T, d]; row_pos/col_pos are [T].
        """
        if self.rope is None:
            return x
        device = x.device
        row_pos = row_pos.to(device=device, dtype=torch.float32)
        col_pos = col_pos.to(device=device, dtype=torch.float32)

        inv_freq_row = self.rope.inv_freq_row.to(device=device)
        inv_freq_col = self.rope.inv_freq_col.to(device=device)

        angles_row = row_pos.unsqueeze(-1) * inv_freq_row
        angles_col = col_pos.unsqueeze(-1) * inv_freq_col

        cos_row = angles_row.cos().to(dtype=x.dtype)
        sin_row = angles_row.sin().to(dtype=x.dtype)
        cos_col = angles_col.cos().to(dtype=x.dtype)
        sin_col = angles_col.sin().to(dtype=x.dtype)

        return _apply_rotary_2d(x, cos_row, sin_row, cos_col, sin_col, self.rope.d_row)

    def _full_grid_positions(self, H: int, W: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        rows = torch.arange(H, device=device, dtype=torch.float32)
        cols = torch.arange(W, device=device, dtype=torch.float32)
        row_grid = rows[:, None].expand(H, W).reshape(H * W)
        col_grid = cols[None, :].expand(H, W).reshape(H * W)
        return row_grid, col_grid

    def _compression_branch(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        H: int,
        W: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            o_cmp: [B, h_q, T, d]
            k_cmp: [B, h_kv, n_patches, d] (compressed keys, for importance in selection)
        """
        B = q.shape[0]
        h_kv, d = self.h_kv, self.d
        p = self.patch_size
        pp = p * p
        nH, nW = H // p, W // p
        n_patches = nH * nW

        k_patches = self._bhtd_to_patches(k, B, h_kv, H, W, p).view(B * h_kv, n_patches, pp, d)
        v_patches = self._bhtd_to_patches(v, B, h_kv, H, W, p).view(B * h_kv, n_patches, pp, d)

        k_cmp = self.compress_k(k_patches).view(B, h_kv, n_patches, d)
        v_cmp = self.compress_v(v_patches).view(B, h_kv, n_patches, d)

        k_cmp = self._rope_at_centers(k_cmp, nH, nW)

        # SDPA with GQA: q has h_q heads, k_cmp/v_cmp have h_kv heads
        o_cmp = F.scaled_dot_product_attention(q, k_cmp, v_cmp, enable_gqa=True)

        return o_cmp, k_cmp

    def _selection_branch(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cmp: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        B = q.shape[0]
        h_q, h_kv, G = self.h_q, self.h_kv, self.gqa_group_size
        d = self.d
        p = self.patch_size
        pp = p * p
        nH, nW = H // p, W // p
        n_patches = nH * nW
        n = min(self.top_n, n_patches)
        T = q.shape[2]

        # --- Adaptive importance scoring (per KV head) ---
        with torch.no_grad():
            scale = 1.0 / math.sqrt(d)
            importance = torch.zeros(B, h_kv, n_patches, device=q.device, dtype=torch.float32)

            # Adaptive chunk: largest that fits in ~30% of free GPU memory
            # Per-chunk alloc: [B, h_kv, G, chunk, n_patches] float32
            elem_per_token = B * h_kv * G * n_patches * 4  # bytes
            if q.is_cuda:
                free_mem = torch.cuda.mem_get_info(q.device)[0]
                budget = int(free_mem * 0.3)
                chunk_size = max(64, min(T, budget // max(elem_per_token, 1)))
                chunk_size = (chunk_size // 64) * 64 or 64
            else:
                chunk_size = min(T, 1024)

            # Reshape q for grouped scoring: [B, h_kv, G, T, d]
            q_grouped = q.view(B, h_kv, G, T, d)

            try:
                for start in range(0, T, chunk_size):
                    end = min(start + chunk_size, T)
                    q_chunk = q_grouped[:, :, :, start:end, :]  # [B, h_kv, G, chunk, d]
                    # logits: [B, h_kv, G, chunk, n_patches]
                    logits = torch.einsum("bhgtd,bhnd->bhgtn", q_chunk * scale, k_cmp)
                    # softmax over n_patches, sum out chunk (dim=3) and G (dim=2)
                    importance += F.softmax(logits, dim=-1).sum(dim=3).sum(dim=2)
            except RuntimeError as e:
                if "out of memory" not in str(e):
                    raise
                # OOM fallback: halve chunk and retry
                importance.zero_()
                chunk_size = max(64, chunk_size // 2)
                for start in range(0, T, chunk_size):
                    end = min(start + chunk_size, T)
                    q_chunk = q_grouped[:, :, :, start:end, :]
                    logits = torch.einsum("bhgtd,bhnd->bhgtn", q_chunk * scale, k_cmp)
                    importance += F.softmax(logits, dim=-1).sum(dim=3).sum(dim=2)

        _, top_idx = importance.topk(n, dim=-1)  # [B, h_kv, n]

        # GPU path: Triton MHA kernel for gqa_group_size==1, gather+SDPA otherwise
        if q.is_cuda and G == 1:
            from src.ops.selection_attention_2d import SelectionAttn2D, make_patch_starts
            patch_starts = make_patch_starts(H, W, p, q.device)
            return SelectionAttn2D.apply(
                q, k, v, top_idx.squeeze(1).to(torch.int32),
                patch_starts, pp, H, W, p, 1.0 / math.sqrt(d),
            )

        # Phase A path for GQA (G > 1) on GPU, and CPU fallback for all cases:
        # gather + SDPA with enable_gqa=True
        k_patches = self._bhtd_to_patches(k, B, h_kv, H, W, p)  # [B, h_kv, n_patches, pp, d]
        v_patches = self._bhtd_to_patches(v, B, h_kv, H, W, p)

        gather_idx = top_idx[:, :, :, None, None].expand(B, h_kv, n, pp, d)
        k_slc = k_patches.gather(2, gather_idx).reshape(B, h_kv, n * pp, d)
        v_slc = v_patches.gather(2, gather_idx).reshape(B, h_kv, n * pp, d)

        return F.scaled_dot_product_attention(q, k_slc, v_slc, enable_gqa=True)

    @staticmethod
    def _same_window_padding(window_size: int) -> tuple[int, int, int, int]:
        """
        Asymmetric padding for stride-1 unfold so output spatial dims stay HxW.

        Works for both odd and even window sizes.
        Returns (pad_left, pad_right, pad_top, pad_bottom).
        """
        total = window_size - 1
        pad_before = total // 2
        pad_after = total - pad_before
        return pad_before, pad_after, pad_before, pad_after

    def _window_branch(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        from src.ops.swin_window_attention import swin_window_attention
        return swin_window_attention(q, k, v, H, W, self.window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H0, W0 = x.shape
        h_q, h_kv, d = self.h_q, self.h_kv, self.d
        p = self.patch_size

        pad_h = (p - (H0 % p)) % p
        pad_w = (p - (W0 % p)) % p
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, H, W = x.shape

        q = self.q_block(x)     # [B, C, H, W]       where C = h_q * d
        k = self.k_block(x)     # [B, C_kv, H, W]    where C_kv = h_kv * d
        v = self.v_block(x)     # [B, C_kv, H, W]

        q_bhtd = self._to_bhtd(q, B, h_q, d, H, W)    # [B, h_q, T, d]
        k_bhtd = self._to_bhtd(k, B, h_kv, d, H, W)   # [B, h_kv, T, d]
        v_bhtd = self._to_bhtd(v, B, h_kv, d, H, W)   # [B, h_kv, T, d]

        if self.rope is not None:
            q_rope = self.rope(q_bhtd, H, W, stride=1)
            k_rope = self.rope(k_bhtd, H, W, stride=1)
        else:
            q_rope = q_bhtd
            k_rope = k_bhtd

        o_cmp, k_cmp = self._compression_branch(q_rope, k_bhtd, v_bhtd, H, W)
        o_slc = self._selection_branch(q_rope, k_rope, v_bhtd, k_cmp, H, W)
        o_win = self._window_branch(q_rope, k_rope, v_bhtd, H, W)

        gate_input = F.adaptive_avg_pool2d(x, 1).flatten(1)
        gates = self.gate(gate_input)
        g_cmp = gates[:, 0].view(B, 1, 1, 1)
        g_slc = gates[:, 1].view(B, 1, 1, 1)
        g_win = gates[:, 2].view(B, 1, 1, 1)

        out = (
            g_cmp * self._to_bcHW(o_cmp, B, h_q, d, H, W)
            + g_slc * self._to_bcHW(o_slc, B, h_q, d, H, W)
            + g_win * self._to_bcHW(o_win, B, h_q, d, H, W)
        )
        out = self.proj(out)

        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H0, :W0]
        return out


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
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        kv_stride: int = 1,
        rope_enabled: bool = False,
        rope_base: float = 10000.0,
    ) -> None:
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
        self.rope = RotaryEmbedding2D(self.d, base=rope_base) if rope_enabled else None
    
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
        # Flash Attention requires FP16/BF16 - autocast applied at algorithm level
        if x.is_cuda:
            # Reshape to [B, h, T, d]
            def to_bhtd_q(t: torch.Tensor) -> torch.Tensor:
                return t.view(B, h, d, H, W).permute(0, 1, 3, 4, 2).contiguous().view(B, h, Tq, d)
            
            def to_bhtd_kv(t: torch.Tensor) -> torch.Tensor:
                return t.view(B, h, d, Hk, Wk).permute(0, 1, 3, 4, 2).contiguous().view(B, h, Tk, d)
            
            q_bhtd = to_bhtd_q(q)
            k_bhtd = to_bhtd_kv(k)
            v_bhtd = to_bhtd_kv(v)

            if self.rope is not None:
                q_bhtd = self.rope(q_bhtd, H, W, stride=1)
                k_bhtd = self.rope(k_bhtd, Hk, Wk, stride=kv_stride if kv_stride > 1 else 1)
            
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

        if self.rope is not None:
            q_bhd = self.rope(q_bhd, H, W, stride=1)
            k_bhd = self.rope(k_bhd, Hk, Wk, stride=kv_stride if kv_stride > 1 else 1)

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
        v = self.act(self.branch2(x))  # [B, Hid, H, W]
        
        # Clamp u to prevent explosion in multiplicative gate
        u = torch.clamp(u, -256.0, 256.0)
        
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
    Complete Transformer block with Pre-LN architecture.

    Architecture:
        x → LN1 → Attention → + x (residual) → LN2 → FFN → + (internal residual) → output

    Input: [B, C, H, W]
    Output: [B, C, H, W]
    """
    
    def __init__(
        self,
        dim: int,
        heads: int,
        expand: float = 2.66,
        ln_eps: float = 1e-5,
        kv_stride: int = 1,
        rope_enabled: bool = False,
        rope_base: float = 10000.0,
        attn_module: nn.Module | None = None,
    ) -> None:
        super().__init__()
        # Module registration order must match korean-model for weight reproducibility
        self.norm1 = LayerNorm2d(dim, eps=ln_eps)
        if attn_module is not None:
            self.attn = attn_module
        else:
            self.attn = EfficientGlobalAttention(
                dim,
                heads,
                kv_stride=kv_stride,
                rope_enabled=rope_enabled,
                rope_base=rope_base,
            )
        self.norm2 = LayerNorm2d(dim, eps=ln_eps)
        self.ffn = GatedDepthwiseFFN(dim, expand)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = self.ffn(self.norm2(x))
        return x


class WindowedTransformerBlock(nn.Module):
    """
    Transformer block variant that applies windowed attention with Pre-LN architecture.
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
        rope_enabled: bool = False,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        # Module registration order must match korean-model for weight reproducibility
        self.norm1 = LayerNorm2d(dim, eps=ln_eps)
        self.attn = WindowAttention(
            EfficientGlobalAttention(
                dim,
                heads,
                kv_stride=kv_stride,
                rope_enabled=rope_enabled,
                rope_base=rope_base,
            ),
            window=window,
            stride=stride,
        )
        self.norm2 = LayerNorm2d(dim, eps=ln_eps)
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
        rope_enabled: bool = False,
        rope_base: float = 10000.0,
        nsa_enabled: bool = False,
        nsa_patch_sizes: Sequence[int] = (8, 8, 4, 4),
        nsa_top_n: Sequence[int] = (32, 16, 16, 16),
        nsa_window_sizes: Sequence[int] = (16, 16, 8, 8),
        nsa_gqa_group_size: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        c = base_ch
        self.use_checkpoint = use_checkpoint
        self.ln_eps = float(ln_eps)
        rope_kwargs = {"rope_enabled": rope_enabled, "rope_base": rope_base}
        self.nsa_enabled = bool(nsa_enabled)

        if len(depths) != 4 or len(heads) != 4:
            raise ValueError("depths and heads must each have 4 entries (levels L0-L3)")

        nsa_patch_sizes = tuple(int(v) for v in nsa_patch_sizes)
        nsa_top_n = tuple(int(v) for v in nsa_top_n)
        nsa_window_sizes = tuple(int(v) for v in nsa_window_sizes)
        if len(nsa_patch_sizes) != 4 or len(nsa_top_n) != 4 or len(nsa_window_sizes) != 4:
            raise ValueError("nsa_patch_sizes, nsa_top_n, and nsa_window_sizes must each have 4 entries")

        def _make_nsa_block_seq(dim: int, num: int, heads_i: int, level_idx: int) -> nn.Sequential:
            blocks = []
            for _ in range(num):
                attn = NSA2DAttention(
                    dim=dim,
                    num_heads=heads_i,
                    patch_size=nsa_patch_sizes[level_idx],
                    top_n=nsa_top_n[level_idx],
                    window_size=nsa_window_sizes[level_idx],
                    gqa_group_size=nsa_gqa_group_size,
                    **rope_kwargs,
                )
                blocks.append(
                    TransformerBlock(
                        dim,
                        heads_i,
                        expand,
                        ln_eps=self.ln_eps,
                        attn_module=attn,
                    )
                )
            return nn.Sequential(*blocks)

        def _make_nsa_single_block(dim: int, heads_i: int, level_idx: int) -> TransformerBlock:
            attn = NSA2DAttention(
                dim=dim,
                num_heads=heads_i,
                patch_size=nsa_patch_sizes[level_idx],
                top_n=nsa_top_n[level_idx],
                window_size=nsa_window_sizes[level_idx],
                gqa_group_size=nsa_gqa_group_size,
                **rope_kwargs,
            )
            return TransformerBlock(
                dim,
                heads_i,
                expand,
                ln_eps=self.ln_eps,
                attn_module=attn,
            )
        
        # ============ STEM ============
        # 3×3 conv: in_ch → C
        # Produces F₀ which is used for residual in head
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 2 * c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * c, c, kernel_size=3, padding=1)
        )
        
        # ============ ENCODER ============
        # Level 0: [B, C, H, W]
        if self.nsa_enabled:
            self.enc0 = _make_nsa_block_seq(c, depths[0], heads[0], 0)
        else:
            enc0_cls: type[nn.Module]
            enc0_kwargs: dict = {**rope_kwargs}
            if sra0_enabled:
                enc0_cls = TransformerBlock
                enc0_kwargs["kv_stride"] = sra0_stride
            elif window0:
                enc0_cls = WindowedTransformerBlock
                enc0_kwargs["window"] = window0
                enc0_kwargs["stride"] = window0_stride
            else:
                enc0_cls = TransformerBlock
            self.enc0 = make_blocks(
                c, depths[0], heads[0], expand, self.ln_eps, block_cls=enc0_cls, block_kwargs=enc0_kwargs
            )
        
        # Downsample: C → 2C, spatial /2
        self.down1 = Downsample(c, 2 * c)
        
        # Level 1: [B, 2C, H/2, W/2]
        if self.nsa_enabled:
            self.enc1 = _make_nsa_block_seq(2 * c, depths[1], heads[1], 1)
        else:
            self.enc1 = make_blocks(2 * c, depths[1], heads[1], expand, self.ln_eps, block_kwargs=rope_kwargs)
        
        # Downsample: 2C → 4C, spatial /2
        self.down2 = Downsample(2 * c, 4 * c)
        
        # Level 2: [B, 4C, H/4, W/4]
        if self.nsa_enabled:
            self.enc2 = _make_nsa_block_seq(4 * c, depths[2], heads[2], 2)
        else:
            self.enc2 = make_blocks(4 * c, depths[2], heads[2], expand, self.ln_eps, block_kwargs=rope_kwargs)
        
        # Downsample: 4C → 8C, spatial /2
        self.down3 = Downsample(4 * c, 8 * c)
        
        # Bottleneck (Level 3): [B, 8C, H/8, W/8]
        if self.nsa_enabled:
            self.enc3 = _make_nsa_block_seq(8 * c, depths[3], heads[3], 3)
        else:
            self.enc3 = make_blocks(8 * c, depths[3], heads[3], expand, self.ln_eps, block_kwargs=rope_kwargs)
        
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
        if self.nsa_enabled:
            self.dec2 = _make_nsa_block_seq(4 * c, depths[2], heads[2], 2)
        else:
            self.dec2 = make_blocks(4 * c, depths[2], heads[2], expand, self.ln_eps, block_kwargs=rope_kwargs)
        
        # Level 1 decoder
        # Upsample: 4C → 2C
        self.up2 = Upsample(4 * c, 2 * c)
        # Fuse after concat: 4C → 2C
        self.fuse1 = nn.Conv2d(4 * c, 2 * c, kernel_size=1)
        if self.nsa_enabled:
            self.dec1 = _make_nsa_block_seq(2 * c, depths[1], heads[1], 1)
        else:
            self.dec1 = make_blocks(2 * c, depths[1], heads[1], expand, self.ln_eps, block_kwargs=rope_kwargs)
        
        # Level 0 decoder (special handling)
        # Upsample: 2C → C
        self.up1 = Upsample(2 * c, c)
        # NO fuse0 - just concat (C + C = 2C)
        # N₁ transformer blocks at 2C
        if self.nsa_enabled:
            self.dec0 = _make_nsa_block_seq(2 * c, depths[0], heads[0], 0)
            self.dec0_extra = _make_nsa_single_block(2 * c, heads[0], 0)
        else:
            dec0_kwargs: dict = {**rope_kwargs}
            if sra0_enabled:
                dec0_cls = TransformerBlock
                dec0_kwargs["kv_stride"] = sra0_stride
            elif window0:
                dec0_cls = WindowedTransformerBlock
                dec0_kwargs["window"] = window0
                dec0_kwargs["stride"] = window0_stride
            else:
                dec0_cls = TransformerBlock
            self.dec0 = make_blocks(
                2 * c, depths[0], heads[0], expand, self.ln_eps, block_cls=dec0_cls, block_kwargs=dec0_kwargs
            )
            # +1 extra single transformer block
            self.dec0_extra = dec0_cls(2 * c, heads[0], expand, ln_eps=self.ln_eps, **dec0_kwargs)
        
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
