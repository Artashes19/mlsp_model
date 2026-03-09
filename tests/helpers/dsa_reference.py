from __future__ import annotations

from collections.abc import Callable
import math

import torch

ROPE_BASE = 10000.0


def _validate_reference_input(x: torch.Tensor, H: int, W: int, rope_dim: int) -> None:
    if x.ndim != 4:
        raise ValueError(f"Expected [B, heads, T, dim] input, got shape={tuple(x.shape)}")
    if x.shape[-2] != H * W:
        raise ValueError(f"Expected T == H * W ({H * W}), got T={x.shape[-2]}")
    if rope_dim < 0 or rope_dim > x.shape[-1]:
        raise ValueError(f"Expected 0 <= rope_dim <= {x.shape[-1]}, got rope_dim={rope_dim}")
    if rope_dim and rope_dim % 4 != 0:
        raise ValueError(f"Expected rope_dim divisible by 4 for 2D partial RoPE, got rope_dim={rope_dim}")


def _positions_from_hw(H: int, W: int) -> list[tuple[int, int]]:
    return [(row, col) for row in range(H) for col in range(W)]


def _inverse_frequencies(dim: int, base: float) -> list[float]:
    if dim == 0:
        return []
    return [1.0 / (base ** (idx / dim)) for idx in range(0, dim, 2)]


def _rotate_non_interleaved(x: torch.Tensor, position: int, inv_freq: list[float]) -> torch.Tensor:
    if x.numel() == 0:
        return x.clone()

    half = x.shape[-1] // 2
    rotated = torch.empty_like(x)
    left = x[:half].to(dtype=torch.float32)
    right = x[half:].to(dtype=torch.float32)

    for pair_idx, freq in enumerate(inv_freq):
        angle = position * freq
        cos = math.cos(angle)
        sin = math.sin(angle)
        a = left[pair_idx]
        b = right[pair_idx]
        rotated[pair_idx] = (a * cos - b * sin).to(dtype=x.dtype)
        rotated[half + pair_idx] = (a * sin + b * cos).to(dtype=x.dtype)

    return rotated


def _rotate_interleaved(x: torch.Tensor, position: int, inv_freq: list[float]) -> torch.Tensor:
    if x.numel() == 0:
        return x.clone()

    rotated = torch.empty_like(x)
    pairs = x.to(dtype=torch.float32).reshape(-1, 2)

    for pair_idx, freq in enumerate(inv_freq):
        angle = position * freq
        cos = math.cos(angle)
        sin = math.sin(angle)
        a = pairs[pair_idx, 0]
        b = pairs[pair_idx, 1]
        rotated[2 * pair_idx] = (a * cos - b * sin).to(dtype=x.dtype)
        rotated[2 * pair_idx + 1] = (a * sin + b * cos).to(dtype=x.dtype)

    return rotated


def _naive_partial_rope_2d(
    x: torch.Tensor,
    *,
    H: int,
    W: int,
    rope_dim: int,
    base: float,
    rotate_half: Callable[[torch.Tensor, int, list[float]], torch.Tensor],
) -> torch.Tensor:
    _validate_reference_input(x, H=H, W=W, rope_dim=rope_dim)
    if rope_dim == 0:
        return x

    row_dim = rope_dim // 2
    col_dim = rope_dim - row_dim
    row_inv_freq = _inverse_frequencies(row_dim, base=base)
    col_inv_freq = _inverse_frequencies(col_dim, base=base)
    out = x.clone()

    for token_idx, (row, col) in enumerate(_positions_from_hw(H, W)):
        row_slice = x[:, :, token_idx, :row_dim]
        col_slice = x[:, :, token_idx, row_dim:rope_dim]
        out[:, :, token_idx, :row_dim] = torch.stack(
            [
                torch.stack([rotate_half(row_slice[b, h], row, row_inv_freq) for h in range(x.shape[1])], dim=0)
                for b in range(x.shape[0])
            ],
            dim=0,
        )
        out[:, :, token_idx, row_dim:rope_dim] = torch.stack(
            [
                torch.stack([rotate_half(col_slice[b, h], col, col_inv_freq) for h in range(x.shape[1])], dim=0)
                for b in range(x.shape[0])
            ],
            dim=0,
        )

    return out


def naive_partial_rope_2d_non_interleaved(
    x: torch.Tensor,
    *,
    H: int,
    W: int,
    rope_dim: int,
    base: float = ROPE_BASE,
) -> torch.Tensor:
    return _naive_partial_rope_2d(
        x,
        H=H,
        W=W,
        rope_dim=rope_dim,
        base=base,
        rotate_half=_rotate_non_interleaved,
    )

def naive_partial_rope_2d_interleaved(
    x: torch.Tensor,
    *,
    H: int,
    W: int,
    rope_dim: int,
    base: float = ROPE_BASE,
) -> torch.Tensor:
    return _naive_partial_rope_2d(
        x,
        H=H,
        W=W,
        rope_dim=rope_dim,
        base=base,
        rotate_half=_rotate_interleaved,
    )
