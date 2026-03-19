from __future__ import annotations

import torch

from src.ops.dsa_flashmla import flashmla_is_supported, flashmla_sparse_mla_forward
from src.ops.dsa_sparse_mla import packed_sparse_mla_reference


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
            "d_v": int(d_v),
            "kv_layout": "latent_then_rope",
        }
        ctx.save_for_backward(q_runtime, kv_runtime, idx)
        ctx.gqa_group_size = int(gqa_group_size)
        ctx.softmax_scale = float(softmax_scale)
        ctx.d_v = int(d_v)
        ctx.query_block_size = 128
        ctx.selected_block_size = 64
        ctx.device = q_runtime.device
        ctx.n_kv_heads = kv_runtime.shape[1]

        if flashmla_is_supported(device=q_runtime.device, n_kv_heads=kv_runtime.shape[1]):
            return flashmla_sparse_mla_forward(
                runtime,
                idx,
                gqa_group_size=ctx.gqa_group_size,
                softmax_scale=ctx.softmax_scale,
            )

        return packed_sparse_mla_reference(
            q_runtime,
            kv_runtime,
            idx,
            d_v=ctx.d_v,
            gqa_group_size=ctx.gqa_group_size,
            softmax_scale=ctx.softmax_scale,
            query_block_size=ctx.query_block_size,
            selected_block_size=ctx.selected_block_size,
        )

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        q_runtime, kv_runtime, idx = ctx.saved_tensors

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
    q = runtime["q"]
    kv = runtime["kv"]
    d_v = int(runtime["d_v"])
    return PackedSparseMLAAutograd.apply(q, kv, idx, gqa_group_size, softmax_scale, d_v)
