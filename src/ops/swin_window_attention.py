"""Swin-style 2D window attention using non-overlapping tiles + SDPA."""
import torch
import torch.nn.functional as F


def swin_window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    H: int,
    W: int,
    window_size: int,
) -> torch.Tensor:
    """
    Partition spatial map into non-overlapping w*w tiles, run SDPA within tiles.

    Args:
        q: [B, h_q, H*W, d]
        k, v: [B, h_kv, H*W, d]  (h_kv <= h_q; h_kv == h_q for MHA)
        H, W: spatial dimensions
        window_size: tile size (w). H and W are padded to multiples of w internally.

    Returns:
        [B, h_q, H*W, d]
    """
    B, h_q, T, d = q.shape
    h_kv = k.shape[1]
    w = window_size

    # Pad if not divisible
    pad_h = (w - H % w) % w
    pad_w = (w - W % w) % w
    Hp, Wp = H + pad_h, W + pad_w

    if pad_h > 0 or pad_w > 0:
        def _pad(x: torch.Tensor, h: int) -> torch.Tensor:
            x = x.view(B, h, H, W, d)
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            return x.reshape(B, h, Hp * Wp, d)
        q = _pad(q, h_q)
        k, v = _pad(k, h_kv), _pad(v, h_kv)

    nH, nW = Hp // w, Wp // w

    def _partition(x: torch.Tensor, h: int) -> torch.Tensor:
        # [B, h, nH, w, nW, w, d] -> [B*nH*nW, h, w*w, d]
        x = x.view(B, h, nH, w, nW, w, d)
        x = x.permute(0, 2, 4, 1, 3, 5, 6).contiguous()
        return x.view(B * nH * nW, h, w * w, d)

    q_tiles = _partition(q, h_q)
    k_tiles = _partition(k, h_kv)
    v_tiles = _partition(v, h_kv)

    # SDPA within each tile — Flash-backed for fp16/bf16 on CUDA
    # enable_gqa=True handles h_q != h_kv; when h_q == h_kv it's a no-op
    o_tiles = F.scaled_dot_product_attention(q_tiles, k_tiles, v_tiles, enable_gqa=True)

    # Un-partition: [B*nH*nW, h_q, w*w, d] -> [B, h_q, Hp*Wp, d]
    o = o_tiles.view(B, nH, nW, h_q, w, w, d)
    o = o.permute(0, 3, 1, 4, 2, 5, 6).contiguous()
    o = o.view(B, h_q, Hp * Wp, d)

    # Crop padding
    if pad_h > 0 or pad_w > 0:
        o = o.view(B, h_q, Hp, Wp, d)[:, :, :H, :W, :].contiguous().reshape(B, h_q, H * W, d)

    return o
