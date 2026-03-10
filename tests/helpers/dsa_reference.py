from __future__ import annotations

from collections.abc import Callable
import copy
import math

import torch

from src.ops.dsa_rope import apply_partial_rope_2d_interleaved
from src.networks.dsa_2d import DSA2DMLAAttention, DSA2DMLAConfig
from tests.helpers.fp8_reference import naive_fwht, naive_weighted_relu_index, reference_fp8_quant

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


def _indexer_preprocessed_reference_inputs(
    mod: DSA2DMLAAttention,
    x: torch.Tensor,
    *,
    detach_inputs: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens, height, width = mod._flatten_tokens(x)
    if detach_inputs:
        tokens = tokens.detach()
    batch, seq_len, _ = tokens.shape
    q_latent = mod.q_norm(mod.wq_a(tokens))
    if detach_inputs:
        q_latent = q_latent.detach()
    q = mod.index_wq_b(q_latent).view(batch, seq_len, mod.index_n_heads, mod.index_head_dim).permute(0, 2, 1, 3)
    k = mod.index_wk(tokens)
    k = mod.index_k_norm(k).view(batch, 1, seq_len, mod.index_head_dim)
    w = mod.index_weights_proj(tokens).to(dtype=torch.float32) * (mod.index_n_heads ** -0.5) * mod.index_softmax_scale
    q = naive_partial_rope_2d_non_interleaved(q, H=height, W=width, rope_dim=mod.index_rope_head_dim)
    k = naive_partial_rope_2d_non_interleaved(k, H=height, W=width, rope_dim=mod.index_rope_head_dim)
    q = naive_fwht(q)
    k = naive_fwht(k).expand(-1, mod.index_n_heads, -1, -1)
    q_q, q_scale = reference_fp8_quant(q)
    k_q, k_scale = reference_fp8_quant(k)
    q_ref = q_q.to(dtype=torch.float32) * q_scale
    k_ref = k_q.to(dtype=torch.float32) * k_scale
    return q_ref, k_ref, w


def indexer_logits_reference(
    mod: DSA2DMLAAttention,
    x: torch.Tensor,
    *,
    detach_inputs: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_ref, k_ref, w = _indexer_preprocessed_reference_inputs(mod, x, detach_inputs=detach_inputs)
    logits = naive_weighted_relu_index(
        q_ref,
        k_ref,
        w,
    )
    topk = min(mod.index_topk, logits.shape[-1])
    idx = torch.argsort(logits, dim=-1, descending=True, stable=True)[..., :topk]
    return logits, idx


def _stable_reference_topk_order(scores: torch.Tensor, idx: torch.Tensor, k: int) -> torch.Tensor:
    if scores.shape != idx.shape:
        raise ValueError(f"Expected scores/idx to have matching shapes, got {scores.shape}, {idx.shape}")
    if k <= 0 or k > scores.shape[-1]:
        raise ValueError(f"Expected 0 < k <= {scores.shape[-1]}, got k={k}")

    idx_order = torch.argsort(idx, dim=-1, descending=False, stable=True)
    scores_by_idx = torch.gather(scores, dim=-1, index=idx_order)
    score_order = torch.argsort(scores_by_idx, dim=-1, descending=True, stable=True)[..., :k]
    return torch.gather(idx_order, dim=-1, index=score_order)


def streaming_indexer_reference(
    mod: DSA2DMLAAttention,
    x: torch.Tensor,
    *,
    block_s: int,
    detach_inputs: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if block_s <= 0:
        raise ValueError(f"Expected block_s > 0, got block_s={block_s}")
    q_ref, k_ref, w = _indexer_preprocessed_reference_inputs(mod, x, detach_inputs=detach_inputs)
    batch, _, query_tokens, _ = q_ref.shape
    source_tokens = k_ref.shape[2]
    keep = min(mod.index_topk, source_tokens)

    best_scores = torch.full((batch, query_tokens, keep), float("-inf"), dtype=q_ref.dtype, device=q_ref.device)
    best_idx = torch.full((batch, query_tokens, keep), source_tokens, dtype=torch.int64, device=q_ref.device)

    for start in range(0, source_tokens, block_s):
        stop = min(start + block_s, source_tokens)
        block_scores = naive_weighted_relu_index(q_ref, k_ref[:, :, start:stop, :], w)
        block_idx = torch.arange(start, stop, device=q_ref.device, dtype=torch.int64).view(1, 1, stop - start)
        block_idx = block_idx.expand(batch, query_tokens, stop - start)

        candidate_scores = torch.cat([best_scores, block_scores], dim=-1)
        candidate_idx = torch.cat([best_idx, block_idx], dim=-1)
        top_order = _stable_reference_topk_order(candidate_scores, candidate_idx, keep)
        best_scores = torch.gather(candidate_scores, dim=-1, index=top_order)
        best_idx = torch.gather(candidate_idx, dim=-1, index=top_order)

    return best_scores, best_idx


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


def _make_training_cfg():
    return DSA2DMLAConfig(
        dim=32,
        n_heads=4,
        n_kv_heads=2,
        q_lora_rank=16,
        kv_lora_rank=12,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=8,
    )


def _split_indexer_and_main_params(mod):
    indexer_params = []
    main_params = []
    for name, param in mod.named_parameters():
        if name.startswith("index_"):
            indexer_params.append(param)
        else:
            main_params.append(param)
    return indexer_params, main_params


def run_tiny_indexer_warmup_steps(num_steps: int) -> list[float]:
    torch.manual_seed(0)
    cfg = _make_training_cfg()
    mod = DSA2DMLAAttention(cfg).float()
    x = torch.randn(1, cfg.dim, 4, 4, dtype=torch.float32)
    teacher = mod.build_dense_teacher_distribution(x)
    indexer_params, main_params = _split_indexer_and_main_params(mod)
    for param in main_params:
        param.requires_grad_(False)

    opt = torch.optim.Adam(indexer_params, lr=1e-2)
    history = []
    for _ in range(num_steps):
        opt.zero_grad(set_to_none=True)
        logits, _ = mod.build_indexer_logits(x, detach_inputs=True)
        loss = mod.indexer_alignment_kl_loss(logits, teacher)
        loss.backward()
        opt.step()
        history.append(loss.item())
    return history


def run_warmup_and_collect_grad_flags() -> dict[str, bool]:
    torch.manual_seed(0)
    cfg = _make_training_cfg()
    mod = DSA2DMLAAttention(cfg).float()
    x = torch.randn(1, cfg.dim, 4, 4, dtype=torch.float32, requires_grad=True)
    teacher = mod.build_dense_teacher_distribution(x)
    indexer_params, main_params = _split_indexer_and_main_params(mod)
    for param in main_params:
        param.requires_grad_(False)

    mod.zero_grad(set_to_none=True)
    logits, _ = mod.build_indexer_logits(x, detach_inputs=True)
    loss = mod.indexer_alignment_kl_loss(logits, teacher)
    loss.backward()

    indexer_has_grad = any(param.grad is not None and param.grad.abs().sum().item() > 0 for param in indexer_params)
    main_has_grad = any(param.grad is not None and param.grad.abs().sum().item() > 0 for param in main_params)
    return {"indexer": indexer_has_grad, "main_model": main_has_grad}
