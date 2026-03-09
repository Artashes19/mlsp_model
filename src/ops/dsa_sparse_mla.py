from __future__ import annotations

import torch


def gather_sparse_mla_tokens(tokens: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 4:
        raise ValueError(f"Expected tokens as [B, heads, T, D], got shape={tuple(tokens.shape)}")
    if idx.ndim != 3:
        raise ValueError(f"Expected idx as [B, Q, K], got shape={tuple(idx.shape)}")
    if tokens.shape[0] != idx.shape[0]:
        raise ValueError("Batch dims must match")
    if idx.numel() and (idx.min() < 0 or idx.max() >= tokens.shape[2]):
        raise ValueError("Sparse indices are out of range for the token axis")

    batch, heads, _, dim = tokens.shape
    _, query_tokens, topk = idx.shape
    expanded_tokens = tokens[:, :, None, :, :].expand(batch, heads, query_tokens, tokens.shape[2], dim)
    gather_idx = idx[:, None, :, :, None].expand(batch, heads, query_tokens, topk, dim)
    return torch.gather(expanded_tokens, dim=3, index=gather_idx)
