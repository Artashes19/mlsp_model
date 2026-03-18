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


def unpack_mla_runtime_qkv(
    runtime: dict[str, torch.Tensor | int | str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = runtime["q"]
    kv = runtime["kv"]
    d_qk = int(runtime["d_qk"])
    d_v = int(runtime["d_v"])
    kv_layout = runtime.get("kv_layout", "v_then_k")

    if not isinstance(q, torch.Tensor) or not isinstance(kv, torch.Tensor):
        raise TypeError("runtime['q'] and runtime['kv'] must be tensors")
    if kv_layout != "v_then_k":
        raise ValueError(f"Unsupported kv_layout={kv_layout!r}")
    if q.ndim != 4 or kv.ndim != 4:
        raise ValueError(f"Expected packed q/kv as rank-4 tensors, got {q.ndim}, {kv.ndim}")
    if q.shape[:3] != (kv.shape[0], q.shape[1], kv.shape[2]):
        raise ValueError("Packed runtime batch/token dimensions must align between q and kv")
    if q.shape[-1] != d_qk:
        raise ValueError(f"Expected q last dim {d_qk}, got {q.shape[-1]}")
    if kv.shape[-1] != d_v + d_qk:
        raise ValueError(f"Expected kv last dim {d_v + d_qk}, got {kv.shape[-1]}")

    v = kv[..., :d_v]
    k = kv[..., d_v:]
    return q, k, v


def streaming_sparse_mla_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    idx: torch.Tensor,
    *,
    gqa_group_size: int,
    softmax_scale: float,
    query_block_size: int = 128,
    selected_block_size: int = 64,
) -> torch.Tensor:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(f"Expected q/k/v as rank-4 tensors, got {q.ndim}, {k.ndim}, {v.ndim}")
    if idx.ndim != 3:
        raise ValueError(f"Expected idx as [B, Q, K], got shape={tuple(idx.shape)}")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0] or q.shape[0] != idx.shape[0]:
        raise ValueError("Batch dims must match")
    if k.shape[:3] != v.shape[:3]:
        raise ValueError("k and v must match on [B, h_kv, T]")
    if q.shape[2] != idx.shape[1]:
        raise ValueError("Query-token dim of q must match idx")
    if q.shape[1] % gqa_group_size != 0:
        raise ValueError("q head count must be divisible by gqa_group_size")
    if idx.numel() and (idx.min() < 0 or idx.max() >= k.shape[2]):
        raise ValueError("Sparse indices are out of range for the token axis")
    if query_block_size <= 0:
        raise ValueError(f"Expected query_block_size > 0, got {query_block_size}")
    if selected_block_size <= 0:
        raise ValueError(f"Expected selected_block_size > 0, got {selected_block_size}")

    batch, h_q, query_tokens, _ = q.shape
    h_kv = k.shape[1]
    if h_q != h_kv * gqa_group_size:
        raise ValueError("q head count must equal h_kv * gqa_group_size")

    batch_out = []
    for b in range(batch):
        kv_group_out = []
        for kv_head in range(h_kv):
            head_start = kv_head * gqa_group_size
            head_stop = head_start + gqa_group_size
            q_group = q[b, head_start:head_stop]
            k_tokens = k[b, kv_head]
            v_tokens = v[b, kv_head]
            query_blocks = []

            for q_start in range(0, query_tokens, query_block_size):
                q_stop = min(q_start + query_block_size, query_tokens)
                q_block = q_group[:, q_start:q_stop].to(dtype=torch.float32)
                idx_block = idx[b, q_start:q_stop]
                block_q = q_block.shape[1]
                block_max = torch.full(
                    (gqa_group_size, block_q),
                    float("-inf"),
                    dtype=torch.float32,
                    device=q.device,
                )
                block_lse = torch.zeros((gqa_group_size, block_q), dtype=torch.float32, device=q.device)
                block_acc = torch.zeros(
                    (gqa_group_size, block_q, v.shape[-1]),
                    dtype=torch.float32,
                    device=q.device,
                )

                for k_start in range(0, idx_block.shape[1], selected_block_size):
                    k_stop = min(k_start + selected_block_size, idx_block.shape[1])
                    idx_slice = idx_block[:, k_start:k_stop]
                    k_sel = k_tokens[idx_slice].to(dtype=torch.float32)
                    v_sel = v_tokens[idx_slice].to(dtype=torch.float32)
                    logits = torch.einsum("gqd,qkd->gqk", q_block, k_sel) * softmax_scale
                    candidate_max = logits.max(dim=-1).values
                    new_max = torch.maximum(block_max, candidate_max)
                    rescale_old = torch.exp(block_max - new_max)
                    exp_logits = torch.exp(logits - new_max.unsqueeze(-1))
                    block_acc = block_acc * rescale_old.unsqueeze(-1) + torch.einsum(
                        "gqk,qkd->gqd",
                        exp_logits,
                        v_sel,
                    )
                    block_lse = block_lse * rescale_old + exp_logits.sum(dim=-1)
                    block_max = new_max

                query_blocks.append((block_acc / block_lse.clamp_min(1e-12).unsqueeze(-1)).to(dtype=v.dtype))

            kv_group_out.append(torch.cat(query_blocks, dim=1))
        batch_out.append(torch.cat(kv_group_out, dim=0))

    return torch.stack(batch_out, dim=0)


def streaming_sparse_mla_reference_from_runtime(
    runtime: dict[str, torch.Tensor | int | str],
    idx: torch.Tensor,
    *,
    gqa_group_size: int,
    softmax_scale: float,
    query_block_size: int = 128,
    selected_block_size: int = 64,
) -> torch.Tensor:
    q, k, v = unpack_mla_runtime_qkv(runtime)
    return streaming_sparse_mla_reference(
        q,
        k,
        v,
        idx,
        gqa_group_size=gqa_group_size,
        softmax_scale=softmax_scale,
        query_block_size=query_block_size,
        selected_block_size=selected_block_size,
    )
