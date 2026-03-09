"""Minimal 2D partial RoPE helpers for DSA TDD scaffolding."""

from __future__ import annotations

import torch


def _validate_partial_rope_input(x: torch.Tensor, H: int, W: int, rope_dim: int) -> None:
    if x.ndim != 4:
        raise ValueError(f"Expected [B, heads, T, dim] input, got shape={tuple(x.shape)}")
    if x.shape[-2] != H * W:
        raise ValueError(f"Expected T == H * W ({H * W}), got T={x.shape[-2]}")
    if rope_dim < 0 or rope_dim > x.shape[-1]:
        raise ValueError(f"Expected 0 <= rope_dim <= {x.shape[-1]}, got rope_dim={rope_dim}")


def apply_partial_rope_2d_non_interleaved(
    x: torch.Tensor,
    *,
    H: int,
    W: int,
    rope_dim: int,
) -> torch.Tensor:
    """Placeholder until the real non-interleaved 2D RoPE math lands."""
    _validate_partial_rope_input(x, H=H, W=W, rope_dim=rope_dim)
    return x


def apply_partial_rope_2d_interleaved(
    x: torch.Tensor,
    *,
    H: int,
    W: int,
    rope_dim: int,
) -> torch.Tensor:
    """Placeholder until the real interleaved 2D RoPE math lands."""
    _validate_partial_rope_input(x, H=H, W=W, rope_dim=rope_dim)
    return x


def maybe_apply_rope_to_v(v: torch.Tensor) -> torch.Tensor:
    """Value projections stay untouched for the current DSA design."""
    return v
