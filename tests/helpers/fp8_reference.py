from __future__ import annotations

import torch

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max


def naive_fwht(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] <= 0 or x.shape[-1] & (x.shape[-1] - 1):
        raise ValueError(f"Expected power-of-two last dim, got shape={tuple(x.shape)}")

    out = x.to(dtype=torch.float32)
    width = 1
    while width < out.shape[-1]:
        chunks = out.reshape(*out.shape[:-1], -1, width * 2)
        left = chunks[..., :, :width]
        right = chunks[..., :, width:]
        out = torch.cat([left + right, left - right], dim=-1).reshape_as(out)
        width *= 2
    return out.to(dtype=x.dtype)


def naive_weighted_relu_index(q: torch.Tensor, k: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    logits = torch.einsum("bjtd,bjsd->bjts", q.to(dtype=torch.float32), k.to(dtype=torch.float32))
    scores = torch.relu(logits)
    return torch.einsum("bjts,btj->bts", scores, w.to(dtype=torch.float32)).to(dtype=q.dtype)


def reference_fp8_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = x.to(dtype=torch.float32).abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / FP8_MAX
    quantized = (x.to(dtype=torch.float32) / scale).clamp(min=-FP8_MAX, max=FP8_MAX).to(FP8_DTYPE)
    return quantized, scale
