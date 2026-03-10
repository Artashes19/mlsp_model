from __future__ import annotations

import torch

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max


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


def act_quant_reference_safe(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = x.to(dtype=torch.float32).abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / FP8_MAX
    quantized = (x.to(dtype=torch.float32) / scale).clamp(min=-FP8_MAX, max=FP8_MAX).to(FP8_DTYPE)
    return quantized, scale


def stable_topk(scores: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k > scores.shape[-1]:
        raise ValueError(f"Expected 0 < k <= {scores.shape[-1]}, got k={k}")
    return torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :k]


def _stable_score_index_topk(scores: torch.Tensor, idx: torch.Tensor, k: int) -> torch.Tensor:
    if scores.shape != idx.shape:
        raise ValueError(f"Expected scores/idx to have the same shape, got {scores.shape}, {idx.shape}")
    if k <= 0 or k > scores.shape[-1]:
        raise ValueError(f"Expected 0 < k <= {scores.shape[-1]}, got k={k}")

    # Stable sort by index first, then stable sort by score descending.
    # This makes lower absolute token index win when scores tie.
    idx_order = torch.argsort(idx, dim=-1, descending=False, stable=True)
    scores_by_idx = torch.gather(scores, dim=-1, index=idx_order)
    score_order = torch.argsort(scores_by_idx, dim=-1, descending=True, stable=True)[..., :k]
    return torch.gather(idx_order, dim=-1, index=score_order)


def streaming_weighted_relu_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    *,
    topk: int,
    block_s: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if block_s <= 0:
        raise ValueError(f"Expected block_s > 0, got block_s={block_s}")
    if topk <= 0:
        raise ValueError(f"Expected topk > 0, got topk={topk}")
    if q.ndim != 4 or k.ndim != 4 or w.ndim != 3:
        raise ValueError(f"Expected q/k/w ranks (4,4,3), got {q.ndim}, {k.ndim}, {w.ndim}")
    if q.shape[0] != k.shape[0] or q.shape[0] != w.shape[0]:
        raise ValueError("Batch dims must match")

    batch, _, query_tokens, _ = q.shape
    source_tokens = k.shape[2]
    keep = min(topk, source_tokens)

    best_scores = torch.full((batch, query_tokens, keep), float("-inf"), dtype=q.dtype, device=q.device)
    best_idx = torch.full((batch, query_tokens, keep), source_tokens, dtype=torch.int64, device=q.device)

    for start in range(0, source_tokens, block_s):
        stop = min(start + block_s, source_tokens)
        block_scores = weighted_relu_index_score(q, k[:, :, start:stop, :], w)
        block_idx = torch.arange(start, stop, device=q.device, dtype=torch.int64).view(1, 1, stop - start)
        block_idx = block_idx.expand(batch, query_tokens, stop - start)

        candidate_scores = torch.cat([best_scores, block_scores], dim=-1)
        candidate_idx = torch.cat([best_idx, block_idx], dim=-1)
        top_order = _stable_score_index_topk(candidate_scores, candidate_idx, keep)
        best_scores = torch.gather(candidate_scores, dim=-1, index=top_order)
        best_idx = torch.gather(candidate_idx, dim=-1, index=top_order)

    return best_scores, best_idx
