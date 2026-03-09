from __future__ import annotations

import torch


def _validate_power_of_two_last_dim(x: torch.Tensor) -> None:
    if x.shape[-1] <= 0 or x.shape[-1] & (x.shape[-1] - 1):
        raise ValueError(f"Expected power-of-two last dim, got shape={tuple(x.shape)}")


def fwht_last_dim(x: torch.Tensor) -> torch.Tensor:
    _validate_power_of_two_last_dim(x)
    out = x.to(dtype=torch.float32)
    width = 1
    while width < out.shape[-1]:
        out = out.reshape(*out.shape[:-1], -1, width * 2)
        left = out[..., :, :width]
        right = out[..., :, width:]
        out = torch.cat([left + right, left - right], dim=-1)
        out = out.reshape(*x.shape[:-1], x.shape[-1])
        width *= 2
    return out.to(dtype=x.dtype)


def weighted_relu_index_score(q: torch.Tensor, k: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError(f"Expected q/k as [B, heads, T, D] and [B, heads, S, D], got {q.shape}, {k.shape}")
    if w.ndim != 3:
        raise ValueError(f"Expected w as [B, T, heads], got {w.shape}")
    if q.shape[0] != k.shape[0] or q.shape[0] != w.shape[0]:
        raise ValueError("Batch dims must match")
    if q.shape[1] != k.shape[1] or q.shape[1] != w.shape[-1]:
        raise ValueError("Head dims must match")
    if q.shape[2] != w.shape[1]:
        raise ValueError("Query-token dim must match between q and w")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k head dims must match")

    logits = torch.einsum("bjtd,bjsd->bjts", q.to(dtype=torch.float32), k.to(dtype=torch.float32))
    scores = torch.relu(logits)
    return torch.einsum("bjts,btj->bts", scores, w.to(dtype=torch.float32)).to(dtype=q.dtype)
