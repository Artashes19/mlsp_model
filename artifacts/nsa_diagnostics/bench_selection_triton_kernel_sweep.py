#!/usr/bin/env python
"""
Benchmark current Triton selection forward and dQ kernels over meta-parameter
choices without changing kernel algorithms.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.networks.txunet import NSA2DAttention
from src.ops.selection_attention_2d_per_query import (
    _next_power_of_2,
    _sel_perq_bwd_dq_kernel,
    _sel_perq_fwd_kernel,
    _select_num_warps_per_query,
    _select_num_warps_per_query_dkv,
    make_patch_starts,
    selection_attn_2d_per_query_forward,
    selection_per_query_bwd_dq,
)


@dataclass(frozen=True)
class ModelConfig:
    dim: int
    heads: int
    gqa_group_size: int

    @property
    def label(self) -> str:
        return f"C{self.dim}_h{self.heads}_g{self.gqa_group_size}"


@dataclass(frozen=True)
class CaseConfig:
    size: int
    top_n: int

    @property
    def label(self) -> str:
        return f"{self.size}x{self.size}_top{self.top_n}"


def cuda_sync() -> None:
    torch.cuda.synchronize()


def cleanup_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def time_cuda_ms(fn: Callable[[], object]) -> tuple[object, float]:
    cuda_sync()
    t0 = time.perf_counter()
    out = fn()
    cuda_sync()
    return out, (time.perf_counter() - t0) * 1000.0


def mean_ms(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def build_attn(
    config: ModelConfig,
    case: CaseConfig,
    dtype: torch.dtype,
    device: torch.device,
    patch_size: int,
    window_size: int,
) -> NSA2DAttention:
    return NSA2DAttention(
        dim=config.dim,
        num_heads=config.heads,
        patch_size=patch_size,
        top_n=case.top_n,
        window_size=window_size,
        rope_enabled=True,
        gqa_group_size=config.gqa_group_size,
        selection_forward_mode="unpacked",
        selection_dq_mode="auto",
    ).to(device=device, dtype=dtype)


def prepare_inputs(attn: NSA2DAttention, x: torch.Tensor) -> dict[str, torch.Tensor | int | float]:
    B, _, H, W = x.shape
    h_q, h_kv, d = attn.h_q, attn.h_kv, attn.d
    p = attn.patch_size
    scale = 1.0 / math.sqrt(d)

    q = attn.q_block(x)
    k = attn.k_block(x)
    v = attn.v_block(x)

    q_bhtd = attn._to_bhtd(q, B, h_q, d, H, W).contiguous()
    k_bhtd = attn._to_bhtd(k, B, h_kv, d, H, W).contiguous()
    v_bhtd = attn._to_bhtd(v, B, h_kv, d, H, W).contiguous()

    q_rope = attn.rope(q_bhtd, H, W, stride=1).contiguous() if attn.rope is not None else q_bhtd
    k_rope = attn.rope(k_bhtd, H, W, stride=1).contiguous() if attn.rope is not None else k_bhtd

    _, k_cmp = attn._compression_branch(q_rope, k_bhtd, v_bhtd, H, W)
    block_idx = attn._compute_selection_block_idx(q_rope, k_cmp).contiguous()
    patch_starts = make_patch_starts(H, W, p, x.device).contiguous()

    o_ref, lse_ref = selection_attn_2d_per_query_forward(
        q_rope.contiguous(),
        k_rope.contiguous(),
        v_bhtd.contiguous(),
        block_idx,
        patch_starts,
        p * p,
        H,
        W,
        p,
        scale,
        attn.gqa_group_size,
    )
    do = torch.randn_like(o_ref)
    delta = (o_ref.float() * do.float()).sum(dim=-1)
    dq_ref = selection_per_query_bwd_dq(
        q_rope.contiguous(),
        k_rope.contiguous(),
        v_bhtd.contiguous(),
        do.contiguous(),
        lse_ref,
        delta,
        block_idx,
        patch_starts,
        p * p,
        H,
        W,
        p,
        scale,
        attn.gqa_group_size,
    )

    return {
        "q": q_rope,
        "k": k_rope,
        "v": v_bhtd,
        "block_idx": block_idx,
        "patch_starts": patch_starts,
        "o_ref": o_ref,
        "lse_ref": lse_ref,
        "dq_ref": dq_ref,
        "do": do,
        "delta": delta,
        "H": H,
        "W": W,
        "P": p,
        "pp": p * p,
        "scale": scale,
        "G": attn.gqa_group_size,
    }


def launch_forward_candidate(
    prepared: dict[str, torch.Tensor | int | float],
    num_warps: int,
    num_stages: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = prepared["q"]
    k = prepared["k"]
    v = prepared["v"]
    block_idx = prepared["block_idx"]
    patch_starts = prepared["patch_starts"]
    H = int(prepared["H"])
    W = int(prepared["W"])
    P = int(prepared["P"])
    pp = int(prepared["pp"])
    scale = float(prepared["scale"])
    G = int(prepared["G"])
    B, h_q, T, d = q.shape
    h_kv = k.shape[1]
    top_n = block_idx.shape[-1]

    o = torch.empty_like(q)
    lse = torch.empty(B, h_q, T, dtype=torch.float32, device=q.device)

    BLOCK_D = max(16, _next_power_of_2(d))
    BLOCK_G = max(16, _next_power_of_2(G))
    BLOCK_KV = max(16, _next_power_of_2(pp))
    LOG2E = 1.4426950408889634
    grid = (T, B * h_kv)

    launch_kwargs = {"num_warps": num_warps}
    if num_stages is not None:
        launch_kwargs["num_stages"] = num_stages

    _sel_perq_fwd_kernel[grid](
        q,
        k,
        v,
        o,
        lse,
        block_idx,
        patch_starts,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        block_idx.stride(0),
        block_idx.stride(1),
        block_idx.stride(2),
        block_idx.stride(3),
        scale,
        T=T,
        D=d,
        W_spatial=W,
        P=P,
        PP=pp,
        TOP_N=top_n,
        H_KV=h_kv,
        G=G,
        BLOCK_G=BLOCK_G,
        BLOCK_D=BLOCK_D,
        BLOCK_KV=BLOCK_KV,
        LOG2E=LOG2E,
        **launch_kwargs,
    )
    return o, lse


def launch_dq_candidate(
    prepared: dict[str, torch.Tensor | int | float],
    block_q: int,
    num_warps: int,
    num_stages: int | None,
) -> torch.Tensor:
    q = prepared["q"]
    k = prepared["k"]
    v = prepared["v"]
    do = prepared["do"]
    lse = prepared["lse_ref"]
    delta = prepared["delta"]
    block_idx = prepared["block_idx"]
    patch_starts = prepared["patch_starts"]
    H = int(prepared["H"])
    W = int(prepared["W"])
    P = int(prepared["P"])
    pp = int(prepared["pp"])
    scale = float(prepared["scale"])
    G = int(prepared["G"])
    B, _, T, d = q.shape
    h_kv = k.shape[1]
    top_n = block_idx.shape[-1]

    dq = torch.zeros_like(q)
    BLOCK_D = max(16, _next_power_of_2(d))
    BLOCK_G = max(16, _next_power_of_2(G))
    BLOCK_KV = max(16, _next_power_of_2(pp))
    LOG2E = 1.4426950408889634
    grid = (B * h_kv, torch.div(torch.tensor(T + block_q - 1), block_q, rounding_mode="floor").item())

    launch_kwargs = {"num_warps": num_warps}
    if num_stages is not None:
        launch_kwargs["num_stages"] = num_stages

    _sel_perq_bwd_dq_kernel[grid](
        q,
        k,
        v,
        do,
        lse,
        delta,
        dq,
        block_idx,
        patch_starts,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        dq.stride(0),
        dq.stride(1),
        dq.stride(2),
        dq.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        block_idx.stride(0),
        block_idx.stride(1),
        block_idx.stride(2),
        block_idx.stride(3),
        scale,
        T=T,
        D=d,
        W_spatial=W,
        P=P,
        PP=pp,
        TOP_N=top_n,
        H_KV=h_kv,
        G=G,
        BLOCK_G=BLOCK_G,
        BLOCK_D=BLOCK_D,
        BLOCK_KV=BLOCK_KV,
        BLOCK_Q=block_q,
        LOG2E=LOG2E,
        **launch_kwargs,
    )
    return dq


def benchmark_candidate(fn: Callable[[], object], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    cuda_sync()
    times: list[float] = []
    for _ in range(iters):
        _, ms = time_cuda_ms(fn)
        times.append(ms)
    return mean_ms(times)


def dedupe_tuples(values: list[tuple[int | None, ...]]) -> list[tuple[int | None, ...]]:
    out: list[tuple[int | None, ...]] = []
    for item in values:
        if item not in out:
            out.append(item)
    return out


def current_block_q(T: int) -> int:
    if T >= 4096:
        return 32
    if T >= 512:
        return 16
    return 4


def run_case(
    model_config: ModelConfig,
    case: CaseConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    cleanup_cuda()
    attn = build_attn(
        config=model_config,
        case=case,
        dtype=torch.bfloat16,
        device=device,
        patch_size=args.patch,
        window_size=args.window,
    ).eval()
    x = torch.randn(args.batch_size, model_config.dim, case.size, case.size, device=device, dtype=torch.bfloat16)
    prepared = prepare_inputs(attn, x)

    q = prepared["q"]
    G = int(prepared["G"])
    pp = int(prepared["pp"])
    d = q.shape[-1]
    T = q.shape[2]

    BLOCK_D = max(16, _next_power_of_2(d))
    BLOCK_G = max(16, _next_power_of_2(G))
    BLOCK_KV = max(16, _next_power_of_2(pp))
    current_fwd_warps = _select_num_warps_per_query(BLOCK_G, BLOCK_KV, BLOCK_D)
    current_dq_block_q = current_block_q(T)
    current_dq_warps = _select_num_warps_per_query_dkv(BLOCK_G, BLOCK_KV, BLOCK_D, current_dq_block_q)

    fwd_candidates = dedupe_tuples(
        [(current_fwd_warps, None)]
        + [(warps, stages) for warps in (2, 4, 8) for stages in (None, 2, 4)]
    )
    dq_candidates = dedupe_tuples(
        [(current_dq_block_q, current_dq_warps, None)]
        + [(block_q, warps, stages) for block_q in (16, 32, 64) for warps in (2, 4, 8) for stages in (None, 2, 4)]
    )

    fwd_results = []
    for num_warps, num_stages in fwd_candidates:
        ms = benchmark_candidate(
            lambda num_warps=num_warps, num_stages=num_stages: launch_forward_candidate(prepared, num_warps, num_stages),
            warmup=args.warmup,
            iters=args.iters,
        )
        fwd_results.append({"num_warps": num_warps, "num_stages": num_stages, "ms": ms})
    fwd_results.sort(key=lambda row: row["ms"])

    dq_results = []
    for block_q, num_warps, num_stages in dq_candidates:
        ms = benchmark_candidate(
            lambda block_q=block_q, num_warps=num_warps, num_stages=num_stages: launch_dq_candidate(prepared, block_q, num_warps, num_stages),
            warmup=args.warmup,
            iters=args.iters,
        )
        dq_results.append({"block_q": block_q, "num_warps": num_warps, "num_stages": num_stages, "ms": ms})
    dq_results.sort(key=lambda row: row["ms"])

    best_fwd = fwd_results[0]
    best_dq = dq_results[0]
    o_best, _ = launch_forward_candidate(prepared, int(best_fwd["num_warps"]), best_fwd["num_stages"])
    dq_best = launch_dq_candidate(prepared, int(best_dq["block_q"]), int(best_dq["num_warps"]), best_dq["num_stages"])

    return {
        "model": {
            "dim": model_config.dim,
            "heads": model_config.heads,
            "gqa_group_size": model_config.gqa_group_size,
        },
        "case": {
            "size": case.size,
            "top_n": case.top_n,
            "selected_tokens_per_query": case.top_n * args.patch * args.patch,
        },
        "current_meta": {
            "forward": {"num_warps": current_fwd_warps, "num_stages": None},
            "dq": {"block_q": current_dq_block_q, "num_warps": current_dq_warps, "num_stages": None},
        },
        "validation": {
            "forward_best_max_abs_diff": max_abs_diff(prepared["o_ref"], o_best),
            "dq_best_max_abs_diff": max_abs_diff(prepared["dq_ref"], dq_best),
        },
        "forward_top3": fwd_results[:3],
        "dq_top3": dq_results[:3],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--patch", type=int, default=8)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--top-ns", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--configs", nargs="*", default=["384:6:3", "512:8:4"])
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "nsa_diagnostics")
    return parser.parse_args()


def parse_model_configs(raw_configs: list[str]) -> list[ModelConfig]:
    configs: list[ModelConfig] = []
    for raw in raw_configs:
        dim_s, heads_s, gqa_s = raw.split(":")
        configs.append(ModelConfig(dim=int(dim_s), heads=int(heads_s), gqa_group_size=int(gqa_s)))
    return configs


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    args = parse_args()
    device = torch.device("cuda:0")
    model_configs = parse_model_configs(args.configs)
    cases = [CaseConfig(size=size, top_n=top_n) for size in args.sizes for top_n in args.top_ns]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, object] = {"timestamp": timestamp, "device": torch.cuda.get_device_name(device), "results": []}
    lines = [f"Device: {results['device']}"]

    for model_config in model_configs:
        for case in cases:
            label = f"{model_config.label}_{case.label}"
            print(f"Running {label}...")
            case_result = run_case(model_config, case, args, device)
            results["results"].append(case_result)
            lines.append(f"\n=== {label} ===")
            lines.append(f"current_forward_meta: {case_result['current_meta']['forward']}")
            lines.append(f"current_dq_meta: {case_result['current_meta']['dq']}")
            lines.append(f"validation: {case_result['validation']}")
            lines.append(f"forward_top3: {case_result['forward_top3']}")
            lines.append(f"dq_top3: {case_result['dq_top3']}")

    json_path = args.out_dir / f"selection_triton_kernel_sweep_a100_{timestamp}.json"
    txt_path = args.out_dir / f"selection_triton_kernel_sweep_a100_{timestamp}.txt"
    json_path.write_text(json.dumps(results, indent=2))
    txt_path.write_text("\n".join(lines) + "\n")
    print(f"Saved JSON: {json_path}")
    print(f"Saved text: {txt_path}")


if __name__ == "__main__":
    main()
