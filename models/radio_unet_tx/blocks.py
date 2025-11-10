from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F


# ---------- Norm ----------

class LayerNorm2d(nn.Module):
    """
    Channel-wise LayerNorm over [C] at each spatial location.

    Stability:
    - Upcasts to float32 inside LN to avoid tiny-eps issues in AMP/bfloat16.
    - eps=1e-5 (safer than 1e-6 for half precision).
    """
    def __init__(self, num_channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> BHWC for LayerNorm
        dtype_in = x.dtype
        y = x.permute(0, 2, 3, 1).to(torch.float32)
        y = F.layer_norm(y, (y.shape[-1],), self.weight.to(y.dtype), self.bias.to(y.dtype), self.eps)
        y = y.to(dtype_in).permute(0, 3, 1, 2)
        return y


# ---------- Attention ----------

class EfficientGlobalAttention(nn.Module):
    """
    Global spatial attention with Q/K depthwise 3x3 locality and streaming softmax.

    Shapes:
      x: [B, C, H, W]
      heads = h, d = C // h
      q,k,v: [B*h, T, d] with T=H*W
      out: [B, C, H, W]

    Complexity:
      O(T^2 * d) compute, O((Tq*dk + Tq*d) + (Tk*d)) memory via streaming (no T×T allocation).
    """
    def __init__(self, dim: int, num_heads: int, apply_dw_on_v: bool = False) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by heads"
        self.dim = dim
        self.h = num_heads
        self.d = dim // num_heads

        # 1x1 projections (channel mixing)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

        # depthwise locality on Q,K (optionally V)
        self.q_dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True)
        self.k_dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True)
        self.v_dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True) if apply_dw_on_v else None

        # output projection
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    @torch.no_grad()
    def _choose_chunks(self, T: int, d: int, bytes_per_el: int = 4) -> tuple[int, int]:
        """
        Heuristic chunk sizes for streaming softmax to keep memory in check.
        """
        # Aim ~64MB per inner matmul slice by default (tweak as needed).
        target_bytes = 64 * 1024 * 1024
        # q_chunk * k_chunk * d * bytes ≈ target
        k_chunk = min(T, max(512, (target_bytes // (d * bytes_per_el)) // 64 * 64))
        q_chunk = k_chunk
        return int(q_chunk), int(k_chunk)

    def _attn_streaming(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Memory-efficient global attention via streaming softmax (online log-sum-exp).

        q,k,v: [Bh, T, d]  ->  out: [Bh, T, d]
        """
        Bh, T, d = q.shape
        device = q.device
        dtype = q.dtype

        q_chunk, k_chunk = self._choose_chunks(T, d, bytes_per_el=2 if dtype in (torch.float16, torch.bfloat16) else 4)

        out = torch.empty_like(q)
        scale = 1.0 / math.sqrt(d)

        # process queries in blocks
        for qs in range(0, T, q_chunk):
            qe = min(qs + q_chunk, T)
            qi = q[:, qs:qe, :]                             # [Bh, Qc, d]

            # row-wise streaming stats
            m_i = torch.full((Bh, qe - qs, 1), -float("inf"), device=device, dtype=dtype)
            s_i = torch.zeros((Bh, qe - qs, 1), device=device, dtype=dtype)
            o_i = torch.zeros((Bh, qe - qs, d), device=device, dtype=dtype)

            # sweep over keys in blocks
            for ks in range(0, T, k_chunk):
                ke = min(ks + k_chunk, T)
                kj = k[:, ks:ke, :]                         # [Bh, Kc, d]
                vj = v[:, ks:ke, :]                         # [Bh, Kc, d]

                # logits: [Bh, Qc, Kc]
                logits = torch.einsum("bid,bjd->bij", qi, kj) * scale

                # streaming softmax: update (m_i, s_i, o_i)
                block_max = logits.max(dim=-1, keepdim=True).values           # [Bh, Qc, 1]
                new_m = torch.maximum(m_i, block_max)                         # [Bh, Qc, 1]

                # renormalize previous stats
                s_i = s_i * torch.exp(m_i - new_m)
                o_i = o_i * torch.exp(m_i - new_m)

                # current block contribs
                p = torch.exp(logits - new_m)                                 # [Bh, Qc, Kc]
                s_i = s_i + p.sum(dim=-1, keepdim=True)                       # [Bh, Qc, 1]
                o_i = o_i + torch.einsum("bij,bjd->bid", p, vj)               # [Bh, Qc, d]

                m_i = new_m

            out[:, qs:qe, :] = o_i / s_i                                      # [Bh, Qc, d]

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W] -> out: [B, C, H, W]
        """
        B, C, H, W = x.shape
        h, d = self.h, self.d
        T = H * W

        # 1x1 + depthwise locality
        q = self.q_dw(self.q(x))      # [B, C, H, W]
        k = self.k_dw(self.k(x))      # [B, C, H, W]
        v = self.v_dw(self.v(x)) if self.v_dw is not None else self.v(x)

        # Prefer PyTorch fused SDPA (Flash/MemEff) when available (CUDA); keep 4D [B, h, T, d]
        use_sdpa = x.is_cuda
        if use_sdpa:
            try:
                # reshape to [B, h, T, d]
                def to_bhtd(t: torch.Tensor) -> torch.Tensor:
                    return t.view(B, h, d, H, W).permute(0, 1, 3, 4, 2).contiguous().view(B, h, T, d)

                q_bhtd = to_bhtd(q)
                k_bhtd = to_bhtd(k)
                v_bhtd = to_bhtd(v)

                # Enable Flash / MemEff kernels where possible
                try:
                    from torch.nn.attention import sdpa_kernel  # type: ignore
                    sdpa_ctx = sdpa_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
                except Exception:
                    sdpa_ctx = torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)

                with sdpa_ctx:
                    out_bhtd = F.scaled_dot_product_attention(q_bhtd, k_bhtd, v_bhtd, dropout_p=0.0, is_causal=False)
                # back to [B, C, H, W]
                out = out_bhtd.view(B, h, H, W, d).permute(0, 1, 4, 2, 3).contiguous().view(B, C, H, W)
                return self.proj(out)
            except Exception:
                # fall back to streaming path below
                pass

        # Fallback: streaming softmax attention on CPU/FP32 or if SDPA is unavailable
        def to_bhd(t: torch.Tensor) -> torch.Tensor:
            t = t.view(B, h, d, H, W).permute(0, 1, 3, 4, 2).contiguous()
            return t.view(B * h, T, d)

        q_bhd = to_bhd(q)
        k_bhd = to_bhd(k)
        v_bhd = to_bhd(v)
        out_bhd = self._attn_streaming(q_bhd, k_bhd, v_bhd)  # [B*h, T, d]
        out = out_bhd.view(B, h, H, W, d).permute(0, 1, 4, 2, 3).contiguous().view(B, C, H, W)
        return self.proj(out)


# ---------- FFN ----------

class GatedDepthwiseFFN(nn.Module):
    """
    1x1 expand (to 2H) -> depthwise 3x3 -> split (H,H) -> gate -> 1x1 project (H->C).
    Uses GELU as requested.
    """
    def __init__(self, dim: int, expand: float = 2.66) -> None:
        super().__init__()
        hidden = int(round(dim * expand))
        self.expand = nn.Conv2d(dim, 2 * hidden, kernel_size=1, bias=True)
        self.dw     = nn.Conv2d(2 * hidden, 2 * hidden, kernel_size=3, padding=1, groups=2 * hidden, bias=True)
        self.proj   = nn.Conv2d(hidden, dim, kernel_size=1, bias=True)
        self.act    = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.expand(x)                      # [B, 2H, H, W]
        z = self.dw(z)                          # [B, 2H, H, W]
        u, v = torch.chunk(z, 2, dim=1)         # [B, H, H, W] each
        g = u * self.act(v)                     # gated
        return self.proj(g)                     # [B, C, H, W]


# ---------- Block ----------

class TransformerBlock(nn.Module):
    """
    One LayerNorm only (pre-attention), as requested.
    No norm before FFN.
    """
    def __init__(self, dim: int, heads: int, expand: float = 2.66,
                 ln_eps: float = 1e-5, apply_dw_on_v: bool = False,
                 residual_scale: float = 1.0) -> None:
        super().__init__()
        self.norm = LayerNorm2d(dim, eps=ln_eps)
        self.attn = EfficientGlobalAttention(dim, heads, apply_dw_on_v=apply_dw_on_v)
        self.ffn  = GatedDepthwiseFFN(dim, expand)
        self.residual_scale = float(residual_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.residual_scale
        x = x + scale * self.attn(self.norm(x))
        x = x + scale * self.ffn(x)    # no LN here per your spec
        return x
