from __future__ import annotations

from collections.abc import Callable
import copy
import math

import torch

from src.ops.dsa_rope import apply_partial_rope_2d_interleaved
from src.networks.dsa_2d import DSA2DMLAAttention

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


def dense_mla_reference(mod, x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected [B, C, H, W] input, got shape={tuple(x.shape)}")

    batch, channels, height, width = x.shape
    if channels != mod.dim:
        raise ValueError(f"Expected channel dim {mod.dim}, got C={channels}")

    seq_len = height * width
    tokens = x.flatten(start_dim=2).transpose(1, 2)

    q_latent = mod.q_norm(mod.wq_a(tokens))
    q = mod.wq_b(q_latent).view(batch, seq_len, mod.n_heads, mod.qk_head_dim).permute(0, 2, 1, 3)
    q_nope = q[..., :mod.qk_nope_head_dim]
    q_pe = apply_partial_rope_2d_interleaved(
        q[..., mod.qk_nope_head_dim:],
        H=height,
        W=width,
        rope_dim=mod.qk_rope_head_dim,
    )
    q = torch.cat([q_nope, q_pe], dim=-1)

    kv_latent_and_pe = mod.wkv_a(tokens)
    kv_latent = mod.kv_norm(kv_latent_and_pe[..., :mod.kv_lora_rank])
    k_pe_shared = kv_latent_and_pe[..., mod.kv_lora_rank:].view(batch, 1, seq_len, mod.qk_rope_head_dim)
    k_pe_shared = apply_partial_rope_2d_interleaved(
        k_pe_shared,
        H=height,
        W=width,
        rope_dim=mod.qk_rope_head_dim,
    )

    kv = mod.wkv_b(kv_latent).view(
        batch,
        seq_len,
        mod.n_kv_heads,
        mod.qk_nope_head_dim + mod.v_head_dim,
    ).permute(0, 2, 1, 3)
    k_nope = kv[..., :mod.qk_nope_head_dim]
    v = kv[..., mod.qk_nope_head_dim:]
    k = torch.cat([k_nope, k_pe_shared.expand(-1, mod.n_kv_heads, -1, -1)], dim=-1)
    k = k.repeat_interleave(mod.gqa_group_size, dim=1)
    v = v.repeat_interleave(mod.gqa_group_size, dim=1)

    attn_scores = torch.matmul(q * mod.softmax_scale, k.transpose(-1, -2))
    attn = torch.softmax(attn_scores, dim=-1)
    out = torch.matmul(attn, v)
    out = out.permute(0, 2, 1, 3).reshape(batch, seq_len, mod.attn_out_dim)
    out = mod.proj(out)
    return out.transpose(1, 2).reshape(batch, mod.dim, height, width)


def compare_dense_mla_backward(mod, x: torch.Tensor) -> None:
    if not x.requires_grad:
        raise ValueError("Expected x.requires_grad=True for backward comparison")

    mod_ref = copy.deepcopy(mod).float()
    mod_out = copy.deepcopy(mod).float()
    x_ref = x.detach().clone().requires_grad_(True)
    x_out = x.detach().clone().requires_grad_(True)
    grad_out = torch.randn_like(x_ref)

    ref = dense_mla_reference(mod_ref, x_ref)
    out = mod_out.forward_dense_reference(x_out)
    ref.backward(grad_out)
    out.backward(grad_out)

    torch.testing.assert_close(out, ref)
    torch.testing.assert_close(x_out.grad, x_ref.grad)

    ref_params = dict(mod_ref.named_parameters())
    out_params = dict(mod_out.named_parameters())
    assert ref_params.keys() == out_params.keys()
    for name in ref_params:
        torch.testing.assert_close(out_params[name].grad, ref_params[name].grad)


def gather_tokens_reference(tokens: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 4:
        raise ValueError(f"Expected tokens as [B, heads, T, D], got shape={tuple(tokens.shape)}")
    if idx.ndim != 3:
        raise ValueError(f"Expected idx as [B, Q, K], got shape={tuple(idx.shape)}")

    batch, heads, _, dim = tokens.shape
    _, query_tokens, topk = idx.shape
    gathered = torch.empty(batch, heads, query_tokens, topk, dim, dtype=tokens.dtype, device=tokens.device)
    for b in range(batch):
        for h in range(heads):
            for q in range(query_tokens):
                for k_idx in range(topk):
                    gathered[b, h, q, k_idx] = tokens[b, h, idx[b, q, k_idx]]
    return gathered


def sparse_mla_reference_from_indices(mod, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    q, k, v, height, width = mod._dense_mla_qkv(x)
    k = k.repeat_interleave(mod.gqa_group_size, dim=1)
    v = v.repeat_interleave(mod.gqa_group_size, dim=1)
    k_selected = gather_tokens_reference(k, idx)
    v_selected = gather_tokens_reference(v, idx)

    attn_scores = torch.einsum("bhtd,bhtkd->bhtk", q * mod.softmax_scale, k_selected)
    attn = torch.softmax(attn_scores, dim=-1)
    out = torch.einsum("bhtk,bhtkd->bhtd", attn, v_selected)
    out = out.permute(0, 2, 1, 3).reshape(x.shape[0], height * width, mod.attn_out_dim)
    out = mod.proj(out)
    return out.transpose(1, 2).reshape(x.shape[0], mod.dim, height, width)


def compare_sparse_and_dense_backward(cfg, *, H: int, W: int) -> None:
    torch.manual_seed(0)
    mod_dense = DSA2DMLAAttention(cfg).float()
    mod_sparse = copy.deepcopy(mod_dense).float()
    x_dense = torch.randn(1, cfg.dim, H, W, dtype=torch.float32, requires_grad=True)
    x_sparse = x_dense.detach().clone().requires_grad_(True)
    grad_out = torch.randn_like(x_dense)

    dense = mod_dense.forward_dense_reference(x_dense)
    sparse = mod_sparse.forward_sparse_with_forced_topk(x_sparse, topk_equals_t=True)
    dense.backward(grad_out)
    sparse.backward(grad_out)

    torch.testing.assert_close(sparse, dense)
    torch.testing.assert_close(x_sparse.grad, x_dense.grad)

    dense_params = dict(mod_dense.named_parameters())
    sparse_params = dict(mod_sparse.named_parameters())
    assert dense_params.keys() == sparse_params.keys()
    for name in dense_params:
        torch.testing.assert_close(sparse_params[name].grad, dense_params[name].grad)


def run_sparse_index_regression_case(cfg, idx: torch.Tensor) -> None:
    mod = DSA2DMLAAttention(cfg).float()
    seq_len = int(idx.max().item()) + 1
    side = int(math.isqrt(seq_len))
    if side * side != seq_len:
        raise ValueError(f"Expected square token grid, got Q={seq_len}")
    if idx.shape[1] == 1:
        idx = idx.expand(idx.shape[0], seq_len, idx.shape[2]).clone()

    x = torch.randn(1, cfg.dim, side, side, dtype=torch.float32, requires_grad=True)
    ref = sparse_mla_reference_from_indices(mod, x, idx)
    out = mod.forward_sparse_from_indices(x, idx)
    torch.testing.assert_close(out, ref)
