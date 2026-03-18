from __future__ import annotations

import importlib
from collections.abc import Callable

import torch

from src.ops.dsa_indexer import act_quant_reference_safe


def deepgemm_import_or_none() -> Callable | None:
    for module_name in ("deep_gemm", "deepgemm"):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        for attr in ("fp8_mqa_logits", "mqa_logits", "fp8_index"):
            fn = getattr(module, attr, None)
            if callable(fn):
                return fn
    return None


def deepgemm_is_supported(
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


def deepgemm_weighted_relu_logits(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
) -> torch.Tensor:
    kernel = deepgemm_import_or_none()
    if kernel is None:
        raise RuntimeError("DeepGEMM is not installed")
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError(f"Expected q/k as [B, heads, T, D] and [B, heads, S, D], got {q.shape}, {k.shape}")
    if w.ndim != 3:
        raise ValueError(f"Expected w as [B, T, heads], got {w.shape}")
    if q.shape[0] != k.shape[0] or q.shape[0] != w.shape[0]:
        raise ValueError("Batch dims must match")
    if q.shape[1] != k.shape[1] or q.shape[1] != w.shape[-1]:
        raise ValueError("Head dims must match")
    if q.shape[2] != w.shape[1]:
        raise ValueError("Query-token dim must match between q and w")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k head dims must match")

    batch, heads, query_tokens, dim = q.shape
    source_tokens = k.shape[2]

    q_q, q_scale = act_quant_reference_safe(q)
    k_q, k_scale = act_quant_reference_safe(k[:, :1, :, :].contiguous())

    q_flat = q_q.permute(0, 2, 1, 3).reshape(batch * query_tokens, heads, dim).contiguous()
    kv_q = k_q[:, 0, :, :].reshape(batch * source_tokens, dim).contiguous()
    kv_scale = k_scale[:, 0, :, 0].reshape(batch * source_tokens).to(dtype=torch.float32).contiguous()

    q_scale_weights = q_scale.squeeze(-1).permute(0, 2, 1).to(dtype=torch.float32)
    weights = (w.to(dtype=torch.float32) * q_scale_weights).reshape(batch * query_tokens, heads).contiguous()

    batch_offsets = torch.arange(batch, device=q.device, dtype=torch.int32) * source_tokens
    starts = batch_offsets[:, None].expand(batch, query_tokens).reshape(-1).contiguous()
    ends = (starts + source_tokens).contiguous()

    logits_flat = kernel(
        q_flat,
        (kv_q, kv_scale),
        weights,
        starts,
        ends,
        True,
    )

    if logits_flat.ndim != 2 or logits_flat.shape[0] != batch * query_tokens:
        raise ValueError(f"Unexpected DeepGEMM logits shape {tuple(logits_flat.shape)}")

    if logits_flat.shape[1] == source_tokens:
        local_logits = logits_flat
    elif logits_flat.shape[1] == batch * source_tokens:
        local_offsets = torch.arange(source_tokens, device=q.device, dtype=torch.int64).view(1, source_tokens)
        gather_idx = starts.to(dtype=torch.int64).unsqueeze(-1) + local_offsets
        local_logits = torch.gather(logits_flat, dim=1, index=gather_idx)
    else:
        raise ValueError(
            "Unexpected DeepGEMM logits width "
            f"{logits_flat.shape[1]} for batch={batch}, source_tokens={source_tokens}"
        )

    return local_logits.reshape(batch, query_tokens, source_tokens).to(dtype=torch.float32)
