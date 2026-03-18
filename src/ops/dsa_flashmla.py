from __future__ import annotations

from collections.abc import Callable

import torch

from src.ops.dsa_sparse_mla import (
    streaming_sparse_mla_reference_from_runtime,
    validate_packed_mla_runtime,
)


def flashmla_import_or_none() -> Callable | None:
    try:
        from flash_mla import flash_mla_sparse_fwd  # type: ignore
    except Exception:
        return None
    return flash_mla_sparse_fwd


def flashmla_is_supported(
    *,
    device: torch.device,
    n_kv_heads: int,
    sm: tuple[int, int] | None = None,
) -> bool:
    if device.type != "cuda":
        return False
    if n_kv_heads != 1:
        return False
    if sm is None:
        if not torch.cuda.is_available():
            return False
        sm = torch.cuda.get_device_capability(device)
    return sm >= (9, 0)


def flashmla_sparse_mla_forward(
    runtime: dict[str, torch.Tensor | int | str],
    idx: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    gqa_group_size = kwargs["gqa_group_size"]
    softmax_scale = kwargs["softmax_scale"]
    force_reference_kernel = kwargs.get("force_reference_kernel", False)

    if force_reference_kernel:
        return streaming_sparse_mla_reference_from_runtime(
            runtime,
            idx,
            gqa_group_size=gqa_group_size,
            softmax_scale=softmax_scale,
        )

    flash_kernel = flashmla_import_or_none()
    if flash_kernel is None:
        raise RuntimeError("FlashMLA is not installed")
    q, kv, d_v = validate_packed_mla_runtime(runtime)
    attn_sink = kwargs.get("attn_sink")
    topk_length = kwargs.get("topk_length")

    if idx.ndim != 3:
        raise ValueError(f"Expected idx as [B, Q, K], got shape={tuple(idx.shape)}")
    if q.shape[0] != idx.shape[0] or q.shape[2] != idx.shape[1]:
        raise ValueError("Packed runtime q and idx must agree on batch/query dimensions")
    if kv.shape[1] != 1:
        raise ValueError("Real FlashMLA sparse prefill is only wired for MQA (h_kv == 1)")

    out_batches = []
    for batch_idx in range(q.shape[0]):
        q_batch = q[batch_idx].permute(1, 0, 2).contiguous()
        kv_batch = kv[batch_idx].permute(1, 0, 2).contiguous()
        indices_batch = idx[batch_idx].to(dtype=torch.int32)
        indices_batch = indices_batch[:, None, :].expand(-1, kv.shape[1], -1).contiguous()
        topk_length_batch = None
        if topk_length is not None:
            topk_length_batch = topk_length[batch_idx].to(dtype=torch.int32)

        out, _, _ = flash_kernel(
            q_batch,
            kv_batch,
            indices_batch,
            softmax_scale,
            d_v,
            attn_sink=attn_sink,
            topk_length=topk_length_batch,
        )
        out_batches.append(out.permute(1, 0, 2).contiguous())

    return torch.stack(out_batches, dim=0)
