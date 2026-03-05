#!/usr/bin/env python
"""Focused A100 benchmark for per-query NSA forward and dQ-heavy paths."""

from __future__ import annotations

import json
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.ops.selection_attention_2d_per_query import SelectionAttn2DPerQuery, make_patch_starts


@dataclass(frozen=True)
class Case:
    name: str
    batch_size: int
    height: int
    width: int
    heads_q: int
    heads_kv: int
    grouped_heads: int
    head_dim: int
    patch_size: int
    top_n: int
    dtype: str


CASES = (
    Case(
        name="gqa_hq4_hkv1_g4_h256_w256_d64_p8_k16_bf16",
        batch_size=1,
        height=256,
        width=256,
        heads_q=4,
        heads_kv=1,
        grouped_heads=4,
        head_dim=64,
        patch_size=8,
        top_n=16,
        dtype="bf16",
    ),
    Case(
        name="mha_hq4_hkv4_g1_h256_w256_d64_p8_k16_bf16",
        batch_size=1,
        height=256,
        width=256,
        heads_q=4,
        heads_kv=4,
        grouped_heads=1,
        head_dim=64,
        patch_size=8,
        top_n=16,
        dtype="bf16",
    ),
)

WARMUP = 5
ITERS = 10


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported dtype name: {name}")


def _timed_cuda(fn, warmup: int, iters: int) -> dict[str, object]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    vals: list[float] = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        vals.append(float(start.elapsed_time(end)))

    return {
        "mean_ms": float(statistics.mean(vals)),
        "median_ms": float(statistics.median(vals)),
        "std_ms": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
        "all_ms": vals,
    }


def _make_case_tensors(case: Case, device: torch.device) -> dict[str, torch.Tensor | float | int]:
    dtype = _dtype_from_name(case.dtype)
    B = case.batch_size
    H = case.height
    W = case.width
    T = H * W
    D = case.head_dim
    P = case.patch_size
    pp = P * P
    n_patches = (H // P) * (W // P)
    scale = D ** -0.5

    patch_starts = make_patch_starts(H, W, P, device)
    block_idx = torch.randint(
        0,
        n_patches,
        (B, case.heads_kv, T, case.top_n),
        device=device,
        dtype=torch.int32,
    )

    tensors: dict[str, torch.Tensor | float | int] = {
        "patch_starts": patch_starts,
        "block_idx": block_idx,
        "pp": pp,
        "scale": scale,
    }

    tensors["q_f"] = torch.randn(B, case.heads_q, T, D, device=device, dtype=dtype)
    tensors["k_f"] = torch.randn(B, case.heads_kv, T, D, device=device, dtype=dtype)
    tensors["v_f"] = torch.randn(B, case.heads_kv, T, D, device=device, dtype=dtype)

    tensors["q_q"] = torch.randn(B, case.heads_q, T, D, device=device, dtype=dtype, requires_grad=True)
    tensors["k_q"] = torch.randn(B, case.heads_kv, T, D, device=device, dtype=dtype)
    tensors["v_q"] = torch.randn(B, case.heads_kv, T, D, device=device, dtype=dtype)

    tensors["q_kv"] = torch.randn(B, case.heads_q, T, D, device=device, dtype=dtype, requires_grad=True)
    tensors["k_kv"] = torch.randn(B, case.heads_kv, T, D, device=device, dtype=dtype, requires_grad=True)
    tensors["v_kv"] = torch.randn(B, case.heads_kv, T, D, device=device, dtype=dtype, requires_grad=True)
    return tensors


def _run_case(case: Case, device: torch.device) -> dict[str, object]:
    tensors = _make_case_tensors(case, device)
    H = case.height
    W = case.width
    P = case.patch_size
    G = case.grouped_heads
    pp = int(tensors["pp"])
    scale = float(tensors["scale"])
    block_idx = tensors["block_idx"]
    patch_starts = tensors["patch_starts"]

    q_f = tensors["q_f"]
    k_f = tensors["k_f"]
    v_f = tensors["v_f"]

    q_q = tensors["q_q"]
    k_q = tensors["k_q"]
    v_q = tensors["v_q"]

    q_kv = tensors["q_kv"]
    k_kv = tensors["k_kv"]
    v_kv = tensors["v_kv"]

    def forward_only() -> None:
        _ = SelectionAttn2DPerQuery.apply(q_f, k_f, v_f, block_idx, patch_starts, pp, H, W, P, scale, G)

    def backward_q_only() -> None:
        if q_q.grad is not None:
            q_q.grad = None
        out = SelectionAttn2DPerQuery.apply(q_q, k_q, v_q, block_idx, patch_starts, pp, H, W, P, scale, G)
        out.sum().backward()

    def backward_qkv() -> None:
        if q_kv.grad is not None:
            q_kv.grad = None
        if k_kv.grad is not None:
            k_kv.grad = None
        if v_kv.grad is not None:
            v_kv.grad = None
        out = SelectionAttn2DPerQuery.apply(q_kv, k_kv, v_kv, block_idx, patch_starts, pp, H, W, P, scale, G)
        out.sum().backward()

    return {
        "forward": _timed_cuda(forward_only, warmup=WARMUP, iters=ITERS),
        "backward_q_only": _timed_cuda(backward_q_only, warmup=WARMUP, iters=ITERS),
        "backward_qkv": _timed_cuda(backward_qkv, warmup=WARMUP, iters=ITERS),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    device = torch.device("cuda")
    torch.manual_seed(0)
    _ = torch.randn(1024, device=device) @ torch.randn(1024, device=device)
    torch.cuda.synchronize()

    results: dict[str, object] = {
        "meta": {
            "ts": datetime.now(timezone.utc).isoformat(),
            "gpu": torch.cuda.get_device_name(device),
            "cuda_visible_devices": str(torch.cuda.current_device()),
            "warmup": WARMUP,
            "iters": ITERS,
            "cases": [asdict(case) for case in CASES],
        },
        "results": {},
    }

    for case in CASES:
        print(f"[bench] {case.name}")
        results["results"][case.name] = _run_case(case, device)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = REPO_ROOT / "artifacts" / "nsa_diagnostics" / f"forward_dq_v3_a100_baseline_{stamp}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
