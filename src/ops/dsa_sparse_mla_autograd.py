from __future__ import annotations

import torch

from src.ops.dsa_flashmla import flashmla_import_or_none, flashmla_is_supported
from src.ops.dsa_sparse_mla import packed_sparse_mla_reference, validate_packed_mla_runtime


def _packed_sparse_mla_reference_with_stats(
    q_runtime: torch.Tensor,
    kv_runtime: torch.Tensor,
    idx: torch.Tensor,
    *,
    d_v: int,
    gqa_group_size: int,
    softmax_scale: float,
    query_block_size: int = 128,
    selected_block_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if q_runtime.ndim != 4 or kv_runtime.ndim != 4:
        raise ValueError(
            f"Expected packed q/kv as rank-4 tensors, got {q_runtime.ndim}, {kv_runtime.ndim}"
        )
    if idx.ndim != 3:
        raise ValueError(f"Expected idx as [B, Q, K], got shape={tuple(idx.shape)}")
    if q_runtime.shape[0] != kv_runtime.shape[0] or q_runtime.shape[0] != idx.shape[0]:
        raise ValueError("Batch dims must match")
    if q_runtime.shape[-1] != kv_runtime.shape[-1]:
        raise ValueError("Packed q and kv last dims must match")
    if q_runtime.shape[2] != idx.shape[1]:
        raise ValueError("Query-token dim of q must match idx")
    if q_runtime.shape[1] % gqa_group_size != 0:
        raise ValueError("q head count must be divisible by gqa_group_size")
    if d_v <= 0 or d_v > kv_runtime.shape[-1]:
        raise ValueError(f"Expected 0 < d_v <= {kv_runtime.shape[-1]}, got d_v={d_v}")
    if idx.numel() and (idx.min() < 0 or idx.max() >= kv_runtime.shape[2]):
        raise ValueError("Sparse indices are out of range for the token axis")
    if query_block_size <= 0:
        raise ValueError(f"Expected query_block_size > 0, got {query_block_size}")
    if selected_block_size <= 0:
        raise ValueError(f"Expected selected_block_size > 0, got {selected_block_size}")

    batch, h_q, query_tokens, _ = q_runtime.shape
    h_kv = kv_runtime.shape[1]
    if h_q != h_kv * gqa_group_size:
        raise ValueError("q head count must equal h_kv * gqa_group_size")

    out_batches = []
    max_batches = []
    lse_batches = []
    for b in range(batch):
        kv_group_out = []
        kv_group_max = []
        kv_group_lse = []
        for kv_head in range(h_kv):
            head_start = kv_head * gqa_group_size
            head_stop = head_start + gqa_group_size
            q_group = q_runtime[b, head_start:head_stop]
            kv_tokens = kv_runtime[b, kv_head]
            query_blocks = []
            query_max_blocks = []
            query_lse_blocks = []

            for q_start in range(0, query_tokens, query_block_size):
                q_stop = min(q_start + query_block_size, query_tokens)
                q_block = q_group[:, q_start:q_stop].to(dtype=torch.float32)
                idx_block = idx[b, q_start:q_stop]
                block_q = q_block.shape[1]
                block_max = torch.full(
                    (gqa_group_size, block_q),
                    float("-inf"),
                    dtype=torch.float32,
                    device=q_runtime.device,
                )
                block_lse = torch.zeros((gqa_group_size, block_q), dtype=torch.float32, device=q_runtime.device)
                block_acc = torch.zeros(
                    (gqa_group_size, block_q, d_v),
                    dtype=torch.float32,
                    device=q_runtime.device,
                )

                for k_start in range(0, idx_block.shape[1], selected_block_size):
                    k_stop = min(k_start + selected_block_size, idx_block.shape[1])
                    idx_slice = idx_block[:, k_start:k_stop]
                    kv_sel = kv_tokens[idx_slice].to(dtype=torch.float32)
                    logits = torch.einsum("gqd,qkd->gqk", q_block, kv_sel) * softmax_scale
                    candidate_max = logits.max(dim=-1).values
                    new_max = torch.maximum(block_max, candidate_max)
                    rescale_old = torch.exp(block_max - new_max)
                    exp_logits = torch.exp(logits - new_max.unsqueeze(-1))
                    block_acc = block_acc * rescale_old.unsqueeze(-1) + torch.einsum(
                        "gqk,qkd->gqd",
                        exp_logits,
                        kv_sel[..., :d_v],
                    )
                    block_lse = block_lse * rescale_old + exp_logits.sum(dim=-1)
                    block_max = new_max

                query_blocks.append((block_acc / block_lse.clamp_min(1e-12).unsqueeze(-1)).to(dtype=kv_runtime.dtype))
                query_max_blocks.append(block_max.transpose(0, 1).contiguous())
                query_lse_blocks.append(block_lse.transpose(0, 1).contiguous())

            kv_group_out.append(torch.cat(query_blocks, dim=1))
            kv_group_max.append(torch.cat(query_max_blocks, dim=0))
            kv_group_lse.append(torch.cat(query_lse_blocks, dim=0))

        out_batches.append(torch.cat(kv_group_out, dim=0))
        max_batches.append(torch.cat(kv_group_max, dim=1))
        lse_batches.append(torch.cat(kv_group_lse, dim=1))

    return (
        torch.stack(out_batches, dim=0),
        torch.stack(max_batches, dim=0),
        torch.stack(lse_batches, dim=0),
    )


def _flashmla_sparse_mla_forward_with_stats(
    runtime: dict[str, torch.Tensor | int | str],
    idx: torch.Tensor,
    *,
    gqa_group_size: int,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q, kv, d_v = validate_packed_mla_runtime(runtime)
    flash_kernel = flashmla_import_or_none()
    if flash_kernel is None:
        raise RuntimeError("FlashMLA is not installed")

    if idx.ndim != 3:
        raise ValueError(f"Expected idx as [B, Q, K], got shape={tuple(idx.shape)}")
    if q.shape[0] != idx.shape[0] or q.shape[2] != idx.shape[1]:
        raise ValueError("Packed runtime q and idx must agree on batch/query dimensions")
    if kv.shape[1] != 1:
        raise ValueError("Real FlashMLA sparse prefill is only wired for MQA (h_kv == 1)")

    out_batches = []
    max_batches = []
    lse_batches = []
    for batch_idx in range(q.shape[0]):
        q_batch = q[batch_idx].permute(1, 0, 2).contiguous()
        kv_batch = kv[batch_idx].permute(1, 0, 2).contiguous()
        indices_batch = idx[batch_idx].to(dtype=torch.int32)
        indices_batch = indices_batch[:, None, :].expand(-1, kv.shape[1], -1).contiguous()

        out, max_logits, lse = flash_kernel(
            q_batch,
            kv_batch,
            indices_batch,
            softmax_scale,
            d_v,
            attn_sink=None,
            topk_length=None,
        )
        out_batches.append(out.permute(1, 0, 2).contiguous())
        max_batches.append(max_logits.contiguous())
        lse_batches.append(lse.contiguous())

    return (
        torch.stack(out_batches, dim=0),
        torch.stack(max_batches, dim=0),
        torch.stack(lse_batches, dim=0),
    )


class PackedSparseMLAAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q_runtime: torch.Tensor,
        kv_runtime: torch.Tensor,
        idx: torch.Tensor,
        gqa_group_size: int,
        softmax_scale: float,
        d_v: int,
    ) -> torch.Tensor:
        runtime = {
            "q": q_runtime,
            "kv": kv_runtime,
            "d_qk": q_runtime.shape[-1],
            "d_v": int(d_v),
            "kv_layout": "latent_then_rope",
        }
        q_runtime, kv_runtime, d_v = validate_packed_mla_runtime(runtime)
        ctx.gqa_group_size = int(gqa_group_size)
        ctx.softmax_scale = float(softmax_scale)
        ctx.d_v = int(d_v)
        ctx.query_block_size = 128
        ctx.selected_block_size = 64
        ctx.device = q_runtime.device
        ctx.n_kv_heads = kv_runtime.shape[1]

        if flashmla_is_supported(device=q_runtime.device, n_kv_heads=kv_runtime.shape[1]):
            out, max_logits, lse = _flashmla_sparse_mla_forward_with_stats(
                runtime,
                idx,
                gqa_group_size=ctx.gqa_group_size,
                softmax_scale=ctx.softmax_scale,
            )
        else:
            out, max_logits, lse = _packed_sparse_mla_reference_with_stats(
                q_runtime,
                kv_runtime,
                idx,
                d_v=ctx.d_v,
                gqa_group_size=ctx.gqa_group_size,
                softmax_scale=ctx.softmax_scale,
                query_block_size=ctx.query_block_size,
                selected_block_size=ctx.selected_block_size,
            )

        ctx.save_for_backward(q_runtime, kv_runtime, idx, max_logits, lse)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        q_runtime, kv_runtime, idx, _, _ = ctx.saved_tensors

        q = q_runtime.detach().clone().requires_grad_(True)
        kv = kv_runtime.detach().clone().requires_grad_(True)

        with torch.enable_grad():
            out = packed_sparse_mla_reference(
                q,
                kv,
                idx,
                d_v=ctx.d_v,
                gqa_group_size=ctx.gqa_group_size,
                softmax_scale=ctx.softmax_scale,
                query_block_size=ctx.query_block_size,
                selected_block_size=ctx.selected_block_size,
            )

        dq, dkv = torch.autograd.grad(out, (q, kv), grad_out, retain_graph=False, create_graph=False)
        return dq, dkv, None, None, None, None


def packed_sparse_mla_autograd_forward(
    runtime: dict[str, torch.Tensor | int | str],
    idx: torch.Tensor,
    *,
    gqa_group_size: int,
    softmax_scale: float,
) -> torch.Tensor:
    runtime = dict(runtime)
    runtime.setdefault("d_qk", runtime["q"].shape[-1])
    runtime.setdefault("kv_layout", "latent_then_rope")
    return PackedSparseMLAAutograd.apply(
        runtime["q"],
        runtime["kv"],
        idx,
        gqa_group_size,
        softmax_scale,
        runtime["d_v"],
    )
