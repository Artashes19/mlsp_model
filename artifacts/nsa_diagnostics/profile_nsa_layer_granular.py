#!/usr/bin/env python
"""
Granular NSA block profiler for current A100 development.

Measures:
- Full TransformerBlock with NSA forward and forward+backward wall time
- Forward decomposition of the NSA attention layer:
  q/k/v shell, compression, selection scoring, selection attention,
  window branch, gate, branch mix, proj
- Forward decomposition of the FFN:
  branch1, branch2, GELU, clamp, gate multiply, proj, residual add
- torch.profiler tables for the full block forward+backward path

The decomposition uses the real module methods and submodules. Manual
reconstruction is used only to time segments and is validated against the
actual module outputs before any measurements are reported.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.networks.txunet import GatedDepthwiseFFN, NSA2DAttention, TransformerBlock


@dataclass(frozen=True)
class CaseConfig:
    size: int
    top_n: int

    @property
    def label(self) -> str:
        return f"{self.size}x{self.size}_top{self.top_n}"


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def time_cuda_ms(fn: Callable[[], torch.Tensor | tuple | None]) -> tuple[torch.Tensor | tuple | None, float]:
    cuda_sync()
    t0 = time.perf_counter()
    out = fn()
    cuda_sync()
    return out, (time.perf_counter() - t0) * 1000.0


def mean_ms(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def build_block(
    dim: int,
    heads: int,
    patch: int,
    top_n: int,
    window: int,
    gqa_group_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> TransformerBlock:
    attn = NSA2DAttention(
        dim=dim,
        num_heads=heads,
        patch_size=patch,
        top_n=top_n,
        window_size=window,
        rope_enabled=True,
        gqa_group_size=gqa_group_size,
        selection_forward_mode="unpacked",
        selection_dq_mode="auto",
    )
    block = TransformerBlock(
        dim=dim,
        heads=heads,
        expand=2.66,
        ln_eps=1e-5,
        attn_module=attn,
        ffn_internal_residual=True,
    )
    return block.to(device=device, dtype=dtype)


def attn_forward_manual(attn: NSA2DAttention, x: torch.Tensor) -> torch.Tensor:
    B, C, H0, W0 = x.shape
    h_q, h_kv, d = attn.h_q, attn.h_kv, attn.d
    p = attn.patch_size

    pad_h = (p - (H0 % p)) % p
    pad_w = (p - (W0 % p)) % p
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    _, _, H, W = x.shape

    q = attn.q_block(x)
    k = attn.k_block(x)
    v = attn.v_block(x)

    q_bhtd = attn._to_bhtd(q, B, h_q, d, H, W)
    k_bhtd = attn._to_bhtd(k, B, h_kv, d, H, W)
    v_bhtd = attn._to_bhtd(v, B, h_kv, d, H, W)

    if attn.rope is not None:
        q_rope = attn.rope(q_bhtd, H, W, stride=1)
        k_rope = attn.rope(k_bhtd, H, W, stride=1)
    else:
        q_rope = q_bhtd
        k_rope = k_bhtd

    o_cmp, k_cmp = attn._compression_branch(q_rope, k_bhtd, v_bhtd, H, W)
    block_idx = attn._compute_selection_block_idx(q_rope, k_cmp)
    o_slc = attn._selection_from_block_idx(q_rope, k_rope, v_bhtd, block_idx, H, W)
    o_win = attn._window_branch(q_rope, k_rope, v_bhtd, H, W)

    gate_input = F.adaptive_avg_pool2d(x, 1).flatten(1)
    gates = attn.gate(gate_input)
    g_cmp = gates[:, 0].view(B, 1, 1, 1)
    g_slc = gates[:, 1].view(B, 1, 1, 1)
    g_win = gates[:, 2].view(B, 1, 1, 1)

    out = (
        g_cmp * attn._to_bcHW(o_cmp, B, h_q, d, H, W)
        + g_slc * attn._to_bcHW(o_slc, B, h_q, d, H, W)
        + g_win * attn._to_bcHW(o_win, B, h_q, d, H, W)
    )
    out = attn.proj(out)

    if pad_h > 0 or pad_w > 0:
        out = out[:, :, :H0, :W0]
    return out


def ffn_forward_manual(ffn: GatedDepthwiseFFN, x: torch.Tensor) -> torch.Tensor:
    u = ffn.branch1(x)
    v = ffn.act(ffn.branch2(x))
    u = torch.clamp(u, -256.0, 256.0)
    g = u * v
    out = ffn.proj(g)
    if ffn.internal_residual:
        out = out + x
    return out


def validate_manual_paths(block: TransformerBlock, x: torch.Tensor) -> dict[str, float]:
    block.eval()
    attn = block.attn
    ffn = block.ffn
    with torch.no_grad():
        norm1 = block.norm1(x)
        attn_ref = attn(norm1)
        attn_manual = attn_forward_manual(attn, norm1)
        attn_diff = max_abs_diff(attn_ref, attn_manual)

        norm2 = block.norm2(x + attn_ref)
        ffn_ref = ffn(norm2)
        ffn_manual = ffn_forward_manual(ffn, norm2)
        ffn_diff = max_abs_diff(ffn_ref, ffn_manual)
    return {"attn_manual_max_abs_diff": attn_diff, "ffn_manual_max_abs_diff": ffn_diff}


def profile_attn_forward_segments(attn: NSA2DAttention, x: torch.Tensor, repeats: int) -> dict[str, float]:
    attn.eval()
    stats: dict[str, list[float]] = defaultdict(list)
    with torch.no_grad():
        for _ in range(repeats):
            B, C, H0, W0 = x.shape
            h_q, h_kv, d = attn.h_q, attn.h_kv, attn.d
            p = attn.patch_size

            x_pad, ms = time_cuda_ms(
                lambda: (
                    F.pad(x, (0, (p - (W0 % p)) % p, 0, (p - (H0 % p)) % p))
                    if ((p - (H0 % p)) % p > 0 or (p - (W0 % p)) % p > 0)
                    else x
                )
            )
            stats["attn.pad_input"].append(ms)
            _, _, H, W = x_pad.shape

            q, ms = time_cuda_ms(lambda: attn.q_block(x_pad))
            stats["attn.q_block"].append(ms)
            k, ms = time_cuda_ms(lambda: attn.k_block(x_pad))
            stats["attn.k_block"].append(ms)
            v, ms = time_cuda_ms(lambda: attn.v_block(x_pad))
            stats["attn.v_block"].append(ms)

            q_bhtd, ms = time_cuda_ms(lambda: attn._to_bhtd(q, B, h_q, d, H, W))
            stats["attn.q_reshape"].append(ms)
            k_bhtd, ms = time_cuda_ms(lambda: attn._to_bhtd(k, B, h_kv, d, H, W))
            stats["attn.k_reshape"].append(ms)
            v_bhtd, ms = time_cuda_ms(lambda: attn._to_bhtd(v, B, h_kv, d, H, W))
            stats["attn.v_reshape"].append(ms)

            if attn.rope is not None:
                q_rope, ms = time_cuda_ms(lambda: attn.rope(q_bhtd, H, W, stride=1))
                stats["attn.q_rope"].append(ms)
                k_rope, ms = time_cuda_ms(lambda: attn.rope(k_bhtd, H, W, stride=1))
                stats["attn.k_rope"].append(ms)
            else:
                q_rope = q_bhtd
                k_rope = k_bhtd

            (o_cmp, k_cmp), ms = time_cuda_ms(lambda: attn._compression_branch(q_rope, k_bhtd, v_bhtd, H, W))
            stats["attn.compression"].append(ms)

            block_idx, ms = time_cuda_ms(lambda: attn._compute_selection_block_idx(q_rope, k_cmp))
            stats["attn.selection_block_idx"].append(ms)
            o_slc, ms = time_cuda_ms(lambda: attn._selection_from_block_idx(q_rope, k_rope, v_bhtd, block_idx, H, W))
            stats["attn.selection_attn"].append(ms)

            o_win, ms = time_cuda_ms(lambda: attn._window_branch(q_rope, k_rope, v_bhtd, H, W))
            stats["attn.window"].append(ms)

            gate_input, ms = time_cuda_ms(lambda: F.adaptive_avg_pool2d(x_pad, 1).flatten(1))
            stats["attn.gate_pool"].append(ms)
            gates, ms = time_cuda_ms(lambda: attn.gate(gate_input))
            stats["attn.gate_mlp"].append(ms)

            g_cmp = gates[:, 0].view(B, 1, 1, 1)
            g_slc = gates[:, 1].view(B, 1, 1, 1)
            g_win = gates[:, 2].view(B, 1, 1, 1)
            out, ms = time_cuda_ms(
                lambda: (
                    g_cmp * attn._to_bcHW(o_cmp, B, h_q, d, H, W)
                    + g_slc * attn._to_bcHW(o_slc, B, h_q, d, H, W)
                    + g_win * attn._to_bcHW(o_win, B, h_q, d, H, W)
                )
            )
            stats["attn.branch_mix"].append(ms)
            out, ms = time_cuda_ms(lambda: attn.proj(out))
            stats["attn.proj"].append(ms)
            if H0 != H or W0 != W:
                _, ms = time_cuda_ms(lambda: out[:, :, :H0, :W0])
                stats["attn.crop"].append(ms)

    result = {name: mean_ms(values) for name, values in stats.items()}
    result["attn.selection_total"] = result.get("attn.selection_block_idx", 0.0) + result.get("attn.selection_attn", 0.0)
    result["attn.shell_total"] = sum(
        result.get(name, 0.0)
        for name in [
            "attn.pad_input",
            "attn.q_block",
            "attn.k_block",
            "attn.v_block",
            "attn.q_reshape",
            "attn.k_reshape",
            "attn.v_reshape",
            "attn.q_rope",
            "attn.k_rope",
            "attn.gate_pool",
            "attn.gate_mlp",
            "attn.branch_mix",
            "attn.proj",
            "attn.crop",
        ]
    )
    result["attn.core_total"] = (
        result.get("attn.compression", 0.0)
        + result.get("attn.selection_total", 0.0)
        + result.get("attn.window", 0.0)
    )
    return result


def profile_ffn_forward_segments(ffn: GatedDepthwiseFFN, x: torch.Tensor, repeats: int) -> dict[str, float]:
    ffn.eval()
    stats: dict[str, list[float]] = defaultdict(list)
    with torch.no_grad():
        for _ in range(repeats):
            u, ms = time_cuda_ms(lambda: ffn.branch1(x))
            stats["ffn.branch1"].append(ms)
            v_raw, ms = time_cuda_ms(lambda: ffn.branch2(x))
            stats["ffn.branch2"].append(ms)
            v, ms = time_cuda_ms(lambda: ffn.act(v_raw))
            stats["ffn.gelu"].append(ms)
            u, ms = time_cuda_ms(lambda: torch.clamp(u, -256.0, 256.0))
            stats["ffn.clamp"].append(ms)
            g, ms = time_cuda_ms(lambda: u * v)
            stats["ffn.gate_mul"].append(ms)
            out, ms = time_cuda_ms(lambda: ffn.proj(g))
            stats["ffn.proj"].append(ms)
            if ffn.internal_residual:
                _, ms = time_cuda_ms(lambda: out + x)
                stats["ffn.residual_add"].append(ms)
    result = {name: mean_ms(values) for name, values in stats.items()}
    result["ffn.total_manual"] = sum(result.values())
    return result


def profile_block_forward_segments(block: TransformerBlock, x: torch.Tensor, repeats: int) -> dict[str, float]:
    block.eval()
    stats: dict[str, list[float]] = defaultdict(list)
    with torch.no_grad():
        for _ in range(repeats):
            norm1, ms = time_cuda_ms(lambda: block.norm1(x))
            stats["block.norm1"].append(ms)
            attn_out, ms = time_cuda_ms(lambda: block.attn(norm1))
            stats["block.attn_total"].append(ms)
            x1, ms = time_cuda_ms(lambda: x + attn_out)
            stats["block.attn_residual_add"].append(ms)
            norm2, ms = time_cuda_ms(lambda: block.norm2(x1))
            stats["block.norm2"].append(ms)
            _, ms = time_cuda_ms(lambda: block.ffn(norm2))
            stats["block.ffn_total"].append(ms)
    result = {name: mean_ms(values) for name, values in stats.items()}
    result["block.forward_total_manual"] = sum(result.values())
    return result


def benchmark_forward(block: TransformerBlock, x: torch.Tensor, warmup: int, iters: int) -> float:
    block.train()
    with torch.no_grad():
        for _ in range(warmup):
            _ = block(x)
    cuda_sync()
    times: list[float] = []
    with torch.no_grad():
        for _ in range(iters):
            _, ms = time_cuda_ms(lambda: block(x))
            times.append(ms)
    return mean_ms(times)


def benchmark_forward_backward(block: TransformerBlock, x: torch.Tensor, warmup: int, iters: int) -> dict[str, float]:
    block.train()
    for _ in range(warmup):
        block.zero_grad(set_to_none=True)
        x_w = x.detach().clone().requires_grad_(True)
        out = block(x_w)
        out.sum().backward()
    cuda_sync()

    times: list[float] = []
    cleanup_cuda()
    for _ in range(iters):
        block.zero_grad(set_to_none=True)
        x_i = x.detach().clone().requires_grad_(True)
        cuda_sync()
        t0 = time.perf_counter()
        out = block(x_i)
        out.sum().backward()
        cuda_sync()
        times.append((time.perf_counter() - t0) * 1000.0)

    cleanup_cuda()
    block.zero_grad(set_to_none=True)
    x_m = x.detach().clone().requires_grad_(True)
    out = block(x_m)
    out.sum().backward()
    cuda_sync()
    peak_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
    return {"fwd_bwd_ms": mean_ms(times), "peak_mem_mb": peak_mb}


def run_with_record(name: str, fn: Callable[[], torch.Tensor]) -> torch.Tensor:
    with record_function(name):
        return fn()


class ProfiledBlockWrapper(nn.Module):
    def __init__(self, block: TransformerBlock) -> None:
        super().__init__()
        self.block = block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.block.attn
        with record_function("block.total"):
            x_norm1 = run_with_record("block.norm1", lambda: self.block.norm1(x))
            attn_out = self._attn_forward(attn, x_norm1)
            x1 = run_with_record("block.attn_residual_add", lambda: x + attn_out)
            x_norm2 = run_with_record("block.norm2", lambda: self.block.norm2(x1))
            ffn_out = self._ffn_forward(self.block.ffn, x_norm2)
            return ffn_out

    def _attn_forward(self, attn: NSA2DAttention, x: torch.Tensor) -> torch.Tensor:
        B, C, H0, W0 = x.shape
        h_q, h_kv, d = attn.h_q, attn.h_kv, attn.d
        p = attn.patch_size
        with record_function("block.attn_total"):
            x_pad = run_with_record(
                "attn.pad_input",
                lambda: (
                    F.pad(x, (0, (p - (W0 % p)) % p, 0, (p - (H0 % p)) % p))
                    if ((p - (H0 % p)) % p > 0 or (p - (W0 % p)) % p > 0)
                    else x
                ),
            )
            _, _, H, W = x_pad.shape
            q = run_with_record("attn.q_block", lambda: attn.q_block(x_pad))
            k = run_with_record("attn.k_block", lambda: attn.k_block(x_pad))
            v = run_with_record("attn.v_block", lambda: attn.v_block(x_pad))
            q_bhtd = run_with_record("attn.q_reshape", lambda: attn._to_bhtd(q, B, h_q, d, H, W))
            k_bhtd = run_with_record("attn.k_reshape", lambda: attn._to_bhtd(k, B, h_kv, d, H, W))
            v_bhtd = run_with_record("attn.v_reshape", lambda: attn._to_bhtd(v, B, h_kv, d, H, W))
            if attn.rope is not None:
                q_rope = run_with_record("attn.q_rope", lambda: attn.rope(q_bhtd, H, W, stride=1))
                k_rope = run_with_record("attn.k_rope", lambda: attn.rope(k_bhtd, H, W, stride=1))
            else:
                q_rope = q_bhtd
                k_rope = k_bhtd
            o_cmp, k_cmp = run_with_record("attn.compression", lambda: attn._compression_branch(q_rope, k_bhtd, v_bhtd, H, W))
            block_idx = run_with_record("attn.selection_block_idx", lambda: attn._compute_selection_block_idx(q_rope, k_cmp))
            o_slc = run_with_record("attn.selection_attn", lambda: attn._selection_from_block_idx(q_rope, k_rope, v_bhtd, block_idx, H, W))
            o_win = run_with_record("attn.window", lambda: attn._window_branch(q_rope, k_rope, v_bhtd, H, W))
            gate_input = run_with_record("attn.gate_pool", lambda: F.adaptive_avg_pool2d(x_pad, 1).flatten(1))
            gates = run_with_record("attn.gate_mlp", lambda: attn.gate(gate_input))
            g_cmp = gates[:, 0].view(B, 1, 1, 1)
            g_slc = gates[:, 1].view(B, 1, 1, 1)
            g_win = gates[:, 2].view(B, 1, 1, 1)
            out = run_with_record(
                "attn.branch_mix",
                lambda: (
                    g_cmp * attn._to_bcHW(o_cmp, B, h_q, d, H, W)
                    + g_slc * attn._to_bcHW(o_slc, B, h_q, d, H, W)
                    + g_win * attn._to_bcHW(o_win, B, h_q, d, H, W)
                ),
            )
            out = run_with_record("attn.proj", lambda: attn.proj(out))
            if H0 != H or W0 != W:
                out = run_with_record("attn.crop", lambda: out[:, :, :H0, :W0])
            return out

    def _ffn_forward(self, ffn: GatedDepthwiseFFN, x: torch.Tensor) -> torch.Tensor:
        with record_function("block.ffn_total"):
            u = run_with_record("ffn.branch1", lambda: ffn.branch1(x))
            v = run_with_record("ffn.branch2", lambda: ffn.branch2(x))
            v = run_with_record("ffn.gelu", lambda: ffn.act(v))
            u = run_with_record("ffn.clamp", lambda: torch.clamp(u, -256.0, 256.0))
            g = run_with_record("ffn.gate_mul", lambda: u * v)
            out = run_with_record("ffn.proj", lambda: ffn.proj(g))
            if ffn.internal_residual:
                out = run_with_record("ffn.residual_add", lambda: out + x)
            return out


def profile_with_torch_profiler(wrapper: nn.Module, x: torch.Tensor, warmup: int, active: int) -> dict[str, object]:
    wrapper.train()
    for _ in range(warmup):
        wrapper.zero_grad(set_to_none=True)
        x_w = x.detach().clone().requires_grad_(True)
        wrapper(x_w).sum().backward()
    cuda_sync()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        with_flops=False,
    ) as prof:
        for _ in range(active):
            wrapper.zero_grad(set_to_none=True)
            x_p = x.detach().clone().requires_grad_(True)
            wrapper(x_p).sum().backward()
            cuda_sync()

    key_averages = prof.key_averages()
    segment_rows = []
    for evt in key_averages:
        if evt.key.startswith(("block.", "attn.", "ffn.")):
            cuda_total = getattr(evt, "cuda_time_total", None)
            if cuda_total is None:
                cuda_total = getattr(evt, "device_time_total", 0.0)
            segment_rows.append(
                {
                    "name": evt.key,
                    "calls": int(evt.count),
                    "cpu_total_us": float(evt.cpu_time_total),
                    "cuda_total_us": float(cuda_total),
                    "self_cuda_us": float(getattr(evt, "self_cuda_time_total", 0.0)),
                }
            )
    segment_rows.sort(key=lambda row: row["cuda_total_us"], reverse=True)

    sort_key = "cuda_time_total"
    table = key_averages.table(sort_by=sort_key, row_limit=40)
    return {"segment_rows": segment_rows, "top_cuda_table": table}


def run_case(
    case: CaseConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    block = build_block(
        dim=args.dim,
        heads=args.heads,
        patch=args.patch,
        top_n=case.top_n,
        window=args.window,
        gqa_group_size=args.gqa_group_size,
        dtype=torch.bfloat16,
        device=device,
    )
    x = torch.randn(args.batch_size, args.dim, case.size, case.size, device=device, dtype=torch.bfloat16)

    cleanup_cuda()
    validation = validate_manual_paths(block, x)
    attn_segments = profile_attn_forward_segments(block.attn, block.norm1(x.detach()), repeats=args.segment_repeats)
    norm1_x = block.norm1(x.detach())
    with torch.no_grad():
        attn_out = block.attn(norm1_x)
        norm2_x = block.norm2(x.detach() + attn_out)
    ffn_segments = profile_ffn_forward_segments(block.ffn, norm2_x.detach(), repeats=args.segment_repeats)
    block_segments = profile_block_forward_segments(block, x.detach(), repeats=args.segment_repeats)
    forward_ms = benchmark_forward(block, x, warmup=args.warmup, iters=args.iters)
    fwd_bwd = benchmark_forward_backward(block, x, warmup=args.warmup, iters=args.iters)

    wrapper = ProfiledBlockWrapper(block)
    with torch.no_grad():
        ref = block(x.detach())
        profiled = wrapper(x.detach())
        wrapper_diff = max_abs_diff(ref, profiled)
    profiler_data = profile_with_torch_profiler(wrapper, x, warmup=args.profiler_warmup, active=args.profiler_active)

    return {
        "case": case.label,
        "config": {
            "batch_size": args.batch_size,
            "dim": args.dim,
            "heads": args.heads,
            "gqa_group_size": args.gqa_group_size,
            "patch": args.patch,
            "top_n": case.top_n,
            "window": args.window,
            "selected_tokens_per_query": case.top_n * args.patch * args.patch,
            "dtype": "bfloat16",
        },
        "validation": {**validation, "wrapper_max_abs_diff": wrapper_diff},
        "block_forward_ms": forward_ms,
        "block_fwd_bwd_ms": fwd_bwd["fwd_bwd_ms"],
        "block_peak_mem_mb": fwd_bwd["peak_mem_mb"],
        "block_forward_segments_ms": block_segments,
        "attn_forward_segments_ms": attn_segments,
        "ffn_forward_segments_ms": ffn_segments,
        "profiler": profiler_data,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--gqa-group-size", type=int, default=4)
    parser.add_argument("--patch", type=int, default=8)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--top-ns", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--segment-repeats", type=int, default=5)
    parser.add_argument("--profiler-warmup", type=int, default=2)
    parser.add_argument("--profiler-active", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "nsa_diagnostics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    device = torch.device("cuda:0")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "timestamp": timestamp,
        "device": torch.cuda.get_device_name(device),
        "cases": [],
    }

    for size in args.sizes:
        for top_n in args.top_ns:
            case = CaseConfig(size=size, top_n=top_n)
            print(f"Running case {case.label}...")
            case_result = run_case(case, args, device)
            results["cases"].append(case_result)

    json_path = args.out_dir / f"nsa_layer_granular_profile_a100_{timestamp}.json"
    json_path.write_text(json.dumps(results, indent=2))

    txt_path = args.out_dir / f"nsa_layer_granular_profile_a100_{timestamp}.txt"
    with txt_path.open("w") as f:
        f.write(f"Device: {results['device']}\n")
        for case in results["cases"]:
            f.write(f"\n=== {case['case']} ===\n")
            f.write(f"block_forward_ms: {case['block_forward_ms']:.3f}\n")
            f.write(f"block_fwd_bwd_ms: {case['block_fwd_bwd_ms']:.3f}\n")
            f.write(f"block_peak_mem_mb: {case['block_peak_mem_mb']:.3f}\n")
            f.write("validation:\n")
            for key, value in case["validation"].items():
                f.write(f"  {key}: {value:.6e}\n")
            f.write("block_forward_segments_ms:\n")
            for key, value in sorted(case["block_forward_segments_ms"].items()):
                f.write(f"  {key}: {value:.3f}\n")
            f.write("attn_forward_segments_ms:\n")
            for key, value in sorted(case["attn_forward_segments_ms"].items()):
                f.write(f"  {key}: {value:.3f}\n")
            f.write("ffn_forward_segments_ms:\n")
            for key, value in sorted(case["ffn_forward_segments_ms"].items()):
                f.write(f"  {key}: {value:.3f}\n")
            f.write("profiler_top_cuda:\n")
            f.write(case["profiler"]["top_cuda_table"])
            f.write("\n")

    print(f"Saved JSON: {json_path}")
    print(f"Saved text: {txt_path}")


if __name__ == "__main__":
    main()
