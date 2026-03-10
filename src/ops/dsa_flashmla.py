from __future__ import annotations

from collections.abc import Callable

import torch

from src.ops.dsa_sparse_mla import streaming_sparse_mla_reference


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


def flashmla_sparse_mla_forward(*args, **kwargs) -> torch.Tensor:
    q, k, v, idx = args
    gqa_group_size = kwargs["gqa_group_size"]
    softmax_scale = kwargs["softmax_scale"]
    force_reference_kernel = kwargs.get("force_reference_kernel", False)

    if force_reference_kernel:
        return streaming_sparse_mla_reference(
            q,
            k,
            v,
            idx,
            gqa_group_size=gqa_group_size,
            softmax_scale=softmax_scale,
        )

    flash_kernel = flashmla_import_or_none()
    if flash_kernel is None:
        raise RuntimeError("FlashMLA is not installed")
    raise NotImplementedError("Real FlashMLA sparse MLA forward call is not wired yet")
