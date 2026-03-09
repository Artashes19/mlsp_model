"""2D partial RoPE helpers for DSA paths."""

from __future__ import annotations

from collections.abc import Callable

import torch

ROPE_BASE = 10000.0

def positions_from_hw(H: int, W: int) -> list[tuple[int, int]]:
    """Return row-major (row, col) coordinates for an HxW grid."""
    if H <= 0 or W <= 0:
        raise ValueError(f"Expected positive spatial shape, got H={H}, W={W}")
    return [(row, col) for row in range(H) for col in range(W)]

def _validate_partial_rope_input(x: torch.Tensor, H: int, W: int, rope_dim: int) -> None:
    if x.ndim != 4:
        raise ValueError(f"Expected [B, heads, T, dim] input, got shape={tuple(x.shape)}")
    if H <= 0 or W <= 0:
        raise ValueError(f"Expected positive spatial shape, got H={H}, W={W}")
    if x.shape[-2] != H * W:
        raise ValueError(f"Expected T == H * W ({H * W}), got T={x.shape[-2]}")
    if rope_dim < 0 or rope_dim > x.shape[-1]:
        raise ValueError(f"Expected 0 <= rope_dim <= {x.shape[-1]}, got rope_dim={rope_dim}")
    if rope_dim and rope_dim % 4 != 0:
        raise ValueError(f"Expected rope_dim divisible by 4 for 2D partial RoPE, got rope_dim={rope_dim}")

def _row_col_positions(H: int, W: int, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    rows = torch.arange(H, device=device, dtype=torch.float32)
    cols = torch.arange(W, device=device, dtype=torch.float32)
    row_grid = rows[:, None].expand(H, W).reshape(H * W)
    col_grid = cols[None, :].expand(H, W).reshape(H * W)
    return row_grid, col_grid

def _inverse_frequencies(dim: int, *, device: torch.device) -> torch.Tensor:
    if dim == 0:
        return torch.empty(0, device=device, dtype=torch.float32)
    exponents = torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim
    return 1.0 / (ROPE_BASE ** exponents)

def _angles_from_positions(positions: torch.Tensor, dim: int) -> torch.Tensor:
    if dim == 0:
        return positions.new_empty((positions.shape[0], 0))
    return positions[:, None] * _inverse_frequencies(dim, device=positions.device)[None, :]

def _rotate_non_interleaved(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] == 0:
        return x

    half = x.shape[-1] // 2
    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]
    left = x[..., :half]
    right = x[..., half:]
    return torch.cat([left * cos - right * sin, left * sin + right * cos], dim=-1)

def _rotate_interleaved(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] == 0:
        return x

    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]
    even = x[..., 0::2]
    odd = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = even * sin + odd * cos
    return out

def _apply_partial_rope_2d(
    x: torch.Tensor,
    *,
    H: int,
    W: int,
    rope_dim: int,
    rotate_half: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    _validate_partial_rope_input(x, H=H, W=W, rope_dim=rope_dim)
    if rope_dim == 0:
        return x

    row_dim = rope_dim // 2
    col_dim = rope_dim - row_dim
    row_pos, col_pos = _row_col_positions(H, W, device=x.device)
    rope_slice = x[..., :rope_dim].to(dtype=torch.float32)
    row_slice = rope_slice[..., :row_dim]
    col_slice = rope_slice[..., row_dim:]

    row_angles = _angles_from_positions(row_pos, row_dim)
    col_angles = _angles_from_positions(col_pos, col_dim)

    rotated = torch.cat(
        [
            rotate_half(row_slice, row_angles),
            rotate_half(col_slice, col_angles),
        ],
        dim=-1,
    ).to(dtype=x.dtype)
    return torch.cat([rotated, x[..., rope_dim:]], dim=-1)

def apply_partial_rope_2d_non_interleaved(
    x: torch.Tensor,
    *,
    H: int,
    W: int,
    rope_dim: int,
) -> torch.Tensor:
    """Apply indexer-style 2D partial RoPE using non-interleaved pairs."""
    return _apply_partial_rope_2d(
        x,
        H=H,
        W=W,
        rope_dim=rope_dim,
        rotate_half=_rotate_non_interleaved,
    )

def apply_partial_rope_2d_interleaved(
    x: torch.Tensor,
    *,
    H: int,
    W: int,
    rope_dim: int,
) -> torch.Tensor:
    """Apply MLA-style 2D partial RoPE using interleaved pairs."""
    return _apply_partial_rope_2d(
        x,
        H=H,
        W=W,
        rope_dim=rope_dim,
        rotate_half=_rotate_interleaved,
    )

def maybe_apply_rope_to_v(v: torch.Tensor) -> torch.Tensor:
    """Value projections stay untouched for the current DSA design."""
    return v
