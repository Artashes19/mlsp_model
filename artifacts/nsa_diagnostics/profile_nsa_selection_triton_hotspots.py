#!/usr/bin/env python
"""
Profile current NSA selection-path hotspots on A100 without changing behavior.

Measures, per case:
- selection scoring / block-index computation
- selection forward attention
- selection backward dQ
- selection backward dK/dV
- full selection forward+backward through current autograd

Also validates that the split low-level wrappers match the current selection op.
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
from torch.profiler import ProfilerActivity, profile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.networks.txunet import NSA2DAttention
from src.ops.selection_attention_2d_per_query import (
    SelectionAttn2DPerQuery,
    make_patch_starts,
    selection_attn_2d_per_query_forward,
    selection_per_query_bwd_dkv,
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


def mean_ms(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def time_cuda_ms(fn: Callable[[], object]) -> tuple[object, float]:
    cuda_sync()
    t0 = time.perf_counter()
    out = fn()
    cuda_sync()
    return out, (time.perf_counter() - t0) * 1000.0


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


def prepare_selection_inputs(
    attn: NSA2DAttention,
    x: torch.Tensor,
) -> dict[str, torch.Tensor | int | float]:
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
    patch_starts = make_patch_starts(H, W, p, x.device)

    return {
        "q": q_rope,
        "k": k_rope,
        "v": v_bhtd,
        "k_cmp": k_cmp.contiguous(),
        "patch_starts": patch_starts,
        "H": H,
        "W": W,
        "P": p,
        "pp": p * p,
        "scale": scale,
        "G": attn.gqa_group_size,
    }


def validate_split_wrappers(
    prepared: dict[str, torch.Tensor | int | float],
    block_idx: torch.Tensor,
) -> dict[str, float]:
    q = prepared["q"]
    k = prepared["k"]
    v = prepared["v"]
    patch_starts = prepared["patch_starts"]
    pp = int(prepared["pp"])
    H = int(prepared["H"])
    W = int(prepared["W"])
    P = int(prepared["P"])
    scale = float(prepared["scale"])
    G = int(prepared["G"])

    do = torch.randn_like(q)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    o_ref = SelectionAttn2DPerQuery.apply(
        q_ref,
        k_ref,
        v_ref,
        block_idx,
        patch_starts,
        pp,
        H,
        W,
        P,
        scale,
        G,
    )
    loss = (o_ref * do).sum()
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(loss, (q_ref, k_ref, v_ref), retain_graph=False, create_graph=False)

    o, lse = selection_attn_2d_per_query_forward(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        block_idx.contiguous(),
        patch_starts.contiguous(),
        pp,
        H,
        W,
        P,
        scale,
        G,
    )
    delta = (o.float() * do.float()).sum(dim=-1)
    dq = selection_per_query_bwd_dq(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        do.contiguous(),
        lse,
        delta,
        block_idx.contiguous(),
        patch_starts.contiguous(),
        pp,
        H,
        W,
        P,
        scale,
        G,
    )
    dk, dv = selection_per_query_bwd_dkv(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        do.contiguous(),
        lse,
        delta,
        block_idx.contiguous(),
        patch_starts.contiguous(),
        pp,
        H,
        W,
        P,
        scale,
        G,
    )
    return {
        "forward_max_abs_diff": max_abs_diff(o_ref.detach(), o.detach()),
        "dq_max_abs_diff": max_abs_diff(dq_ref.detach(), dq.detach()),
        "dk_max_abs_diff": max_abs_diff(dk_ref.detach(), dk.detach()),
        "dv_max_abs_diff": max_abs_diff(dv_ref.detach(), dv.detach()),
    }


def benchmark_selection_components(
    prepared: dict[str, torch.Tensor | int | float],
    warmup: int,
    iters: int,
) -> dict[str, float]:
    q = prepared["q"]
    k = prepared["k"]
    v = prepared["v"]
    k_cmp = prepared["k_cmp"]
    patch_starts = prepared["patch_starts"]
    H = int(prepared["H"])
    W = int(prepared["W"])
    P = int(prepared["P"])
    pp = int(prepared["pp"])
    scale = float(prepared["scale"])
    G = int(prepared["G"])

    block_idx = prepared["block_idx"]
    o = prepared["o"]
    lse = prepared["lse"]
    do = prepared["do"]
    delta = prepared["delta"]

    def bench(fn: Callable[[], object]) -> float:
        for _ in range(warmup):
            fn()
        cuda_sync()
        times: list[float] = []
        for _ in range(iters):
            _, ms = time_cuda_ms(fn)
            times.append(ms)
        return mean_ms(times)

    scoring_ms = bench(lambda: prepared["attn"]._compute_selection_block_idx(q, k_cmp))
    fwd_ms = bench(
        lambda: selection_attn_2d_per_query_forward(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            block_idx.contiguous(),
            patch_starts.contiguous(),
            pp,
            H,
            W,
            P,
            scale,
            G,
        )
    )
    dq_ms = bench(
        lambda: selection_per_query_bwd_dq(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            do.contiguous(),
            lse,
            delta,
            block_idx.contiguous(),
            patch_starts.contiguous(),
            pp,
            H,
            W,
            P,
            scale,
            G,
        )
    )
    dkv_ms = bench(
        lambda: selection_per_query_bwd_dkv(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            do.contiguous(),
            lse,
            delta,
            block_idx.contiguous(),
            patch_starts.contiguous(),
            pp,
            H,
            W,
            P,
            scale,
            G,
        )
    )

    def selection_autograd_step() -> torch.Tensor:
        q_step = q.detach().clone().requires_grad_(True)
        k_step = k.detach().clone().requires_grad_(True)
        v_step = v.detach().clone().requires_grad_(True)
        out = SelectionAttn2DPerQuery.apply(
            q_step.contiguous(),
            k_step.contiguous(),
            v_step.contiguous(),
            block_idx.contiguous(),
            patch_starts.contiguous(),
            pp,
            H,
            W,
            P,
            scale,
            G,
        )
        out.sum().backward()
        return out

    full_autograd_ms = bench(selection_autograd_step)

    return {
        "selection_block_idx_ms": scoring_ms,
        "selection_forward_ms": fwd_ms,
        "selection_bwd_dq_ms": dq_ms,
        "selection_bwd_dkv_ms": dkv_ms,
        "selection_manual_sum_ms": fwd_ms + dq_ms + dkv_ms,
        "selection_autograd_fwd_bwd_ms": full_autograd_ms,
        "selection_total_path_ms": scoring_ms + full_autograd_ms,
    }


def profile_selection_case(
    prepared: dict[str, torch.Tensor | int | float],
    args: argparse.Namespace,
    trace_dir: Path,
    label: str,
) -> dict[str, str]:
    q = prepared["q"]
    k = prepared["k"]
    v = prepared["v"]
    k_cmp = prepared["k_cmp"]
    patch_starts = prepared["patch_starts"]
    H = int(prepared["H"])
    W = int(prepared["W"])
    P = int(prepared["P"])
    pp = int(prepared["pp"])
    scale = float(prepared["scale"])
    G = int(prepared["G"])
    attn = prepared["attn"]

    block_idx = prepared["block_idx"]

    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"{label}_trace.json"
    artifacts: dict[str, str] = {"trace_json": str(trace_path)}

    execution_trace_observer = None
    execution_trace_path: Path | None = None
    if args.with_execution_trace:
        try:
            from torch.profiler import ExecutionTraceObserver

            execution_trace_path = trace_dir / f"{label}_execution_trace.json"
            execution_trace_observer = ExecutionTraceObserver().register_callback(str(execution_trace_path))
            artifacts["execution_trace_json"] = str(execution_trace_path)
        except Exception as exc:  # pragma: no cover - depends on torch build
            artifacts["execution_trace_error"] = repr(exc)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=args.with_memory_timeline,
        with_stack=True,
        execution_trace_observer=execution_trace_observer,
    ) as prof:
        with torch.profiler.record_function("selection_scoring"):
            prof.toggle_collection_dynamic(True, [ProfilerActivity.CPU, ProfilerActivity.CUDA])
            block_idx = attn._compute_selection_block_idx(q, k_cmp)
            prof.toggle_collection_dynamic(False, [ProfilerActivity.CPU, ProfilerActivity.CUDA])

        with torch.profiler.record_function("selection_forward"):
            prof.toggle_collection_dynamic(True, [ProfilerActivity.CPU, ProfilerActivity.CUDA])
            o, lse = selection_attn_2d_per_query_forward(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                block_idx.contiguous(),
                patch_starts.contiguous(),
                pp,
                H,
                W,
                P,
                scale,
                G,
            )
            prof.toggle_collection_dynamic(False, [ProfilerActivity.CPU, ProfilerActivity.CUDA])

        do = torch.randn_like(o)
        delta = (o.float() * do.float()).sum(dim=-1)

        with torch.profiler.record_function("selection_bwd_dq"):
            prof.toggle_collection_dynamic(True, [ProfilerActivity.CPU, ProfilerActivity.CUDA])
            selection_per_query_bwd_dq(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                do.contiguous(),
                lse,
                delta,
                block_idx.contiguous(),
                patch_starts.contiguous(),
                pp,
                H,
                W,
                P,
                scale,
                G,
            )
            prof.toggle_collection_dynamic(False, [ProfilerActivity.CPU, ProfilerActivity.CUDA])

        with torch.profiler.record_function("selection_bwd_dkv"):
            prof.toggle_collection_dynamic(True, [ProfilerActivity.CPU, ProfilerActivity.CUDA])
            selection_per_query_bwd_dkv(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                do.contiguous(),
                lse,
                delta,
                block_idx.contiguous(),
                patch_starts.contiguous(),
                pp,
                H,
                W,
                P,
                scale,
                G,
            )
            prof.toggle_collection_dynamic(False, [ProfilerActivity.CPU, ProfilerActivity.CUDA])

    prof.export_chrome_trace(str(trace_path))

    if args.with_memory_timeline:
        try:
            memory_path = trace_dir / f"{label}_memory_timeline.html"
            prof.export_memory_timeline(str(memory_path))
            artifacts["memory_timeline"] = str(memory_path)
        except Exception as exc:  # pragma: no cover - depends on profiler support
            artifacts["memory_timeline_error"] = repr(exc)

    if execution_trace_observer is not None:
        execution_trace_observer.unregister_callback()

    return artifacts


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
    prepared = prepare_selection_inputs(attn, x)
    prepared["attn"] = attn

    block_idx = attn._compute_selection_block_idx(prepared["q"], prepared["k_cmp"]).contiguous()
    o, lse = selection_attn_2d_per_query_forward(
        prepared["q"].contiguous(),
        prepared["k"].contiguous(),
        prepared["v"].contiguous(),
        block_idx,
        prepared["patch_starts"].contiguous(),
        int(prepared["pp"]),
        int(prepared["H"]),
        int(prepared["W"]),
        int(prepared["P"]),
        float(prepared["scale"]),
        int(prepared["G"]),
    )
    do = torch.randn_like(o)
    delta = (o.float() * do.float()).sum(dim=-1)

    prepared["block_idx"] = block_idx
    prepared["o"] = o
    prepared["lse"] = lse
    prepared["do"] = do
    prepared["delta"] = delta

    validation = validate_split_wrappers(prepared, block_idx)
    timings = benchmark_selection_components(prepared, warmup=args.warmup, iters=args.iters)
    profiler_artifacts = None
    if args.with_execution_trace or args.with_memory_timeline:
        profiler_artifacts = profile_selection_case(
            prepared,
            args,
            args.trace_dir,
            f"{model_config.label}_{case.label}",
        )

    case_result = {
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
        "validation": validation,
        "timings_ms": timings,
    }
    if profiler_artifacts is not None:
        case_result["profiler_artifacts"] = profiler_artifacts
    return case_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--patch", type=int, default=8)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--top-ns", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--configs",
        nargs="*",
        default=["64:4:4", "384:6:3", "512:8:4"],
        help="dim:heads:gqa_group_size",
    )
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "nsa_diagnostics")
    parser.add_argument("--trace-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "nsa_diagnostics")
    parser.add_argument("--with-execution-trace", action="store_true")
    parser.add_argument("--with-memory-timeline", action="store_true")
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

    results: dict[str, object] = {
        "timestamp": timestamp,
        "device": torch.cuda.get_device_name(device),
        "results": [],
    }
    lines = [f"Device: {results['device']}"]

    for model_config in model_configs:
        for case in cases:
            label = f"{model_config.label}_{case.label}"
            print(f"Running {label}...")
            case_result = run_case(model_config, case, args, device)
            results["results"].append(case_result)

            lines.append(f"\n=== {label} ===")
            for key, value in case_result["validation"].items():
                lines.append(f"{key}: {value:.6e}")
            for key, value in case_result["timings_ms"].items():
                lines.append(f"{key}: {value:.3f}")
            if "profiler_artifacts" in case_result:
                lines.append(f"profiler_artifacts: {case_result['profiler_artifacts']}")

    json_path = args.out_dir / f"nsa_selection_triton_hotspots_a100_{timestamp}.json"
    txt_path = args.out_dir / f"nsa_selection_triton_hotspots_a100_{timestamp}.txt"
    json_path.write_text(json.dumps(results, indent=2))
    txt_path.write_text("\n".join(lines) + "\n")
    print(f"Saved JSON: {json_path}")
    print(f"Saved text: {txt_path}")


if __name__ == "__main__":
    main()
