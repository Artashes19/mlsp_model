#!/usr/bin/env python
"""
Profile one NSA selection-path case with Triton Proton.

This is not a replacement for the timing harness. It is a scoped trace that can
be viewed with `triton.profiler.viewer` / `proton-viewer` to inspect hierarchy
and Triton-kernel attribution for the same selection workload.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import triton.profiler as proton
from triton.profiler import scope

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from artifacts.nsa_diagnostics.profile_nsa_selection_triton_hotspots import (  # noqa: E402
    CaseConfig,
    ModelConfig,
    build_attn,
    cuda_sync,
    prepare_selection_inputs,
    validate_split_wrappers,
)
from src.ops.selection_attention_2d_per_query import (  # noqa: E402
    selection_attn_2d_per_query_forward,
    selection_per_query_bwd_dkv,
    selection_per_query_bwd_dq,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--gqa-group-size", type=int, default=3)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "selection_proton_profile.hatchet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(0)

    model_config = ModelConfig(
        dim=args.dim,
        heads=args.heads,
        gqa_group_size=args.gqa_group_size,
    )
    case = CaseConfig(size=args.size, top_n=args.top_n)
    attn = build_attn(
        model_config,
        case,
        dtype=dtype,
        device=device,
        patch_size=args.patch_size,
        window_size=args.window_size,
    )
    x = torch.randn(1, model_config.dim, args.size, args.size, device=device, dtype=dtype)
    prepared = prepare_selection_inputs(attn, x)
    block_idx = attn._compute_selection_block_idx(prepared["q"], prepared["k_cmp"]).contiguous()
    prepared["block_idx"] = block_idx
    split_diff = validate_split_wrappers(prepared, block_idx)

    o, lse = selection_attn_2d_per_query_forward(
        prepared["q"].contiguous(),
        prepared["k"].contiguous(),
        prepared["v"].contiguous(),
        block_idx.contiguous(),
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

    for _ in range(args.warmup):
        _ = attn._compute_selection_block_idx(prepared["q"], prepared["k_cmp"])
        _ = selection_attn_2d_per_query_forward(
            prepared["q"].contiguous(),
            prepared["k"].contiguous(),
            prepared["v"].contiguous(),
            block_idx.contiguous(),
            prepared["patch_starts"].contiguous(),
            int(prepared["pp"]),
            int(prepared["H"]),
            int(prepared["W"]),
            int(prepared["P"]),
            float(prepared["scale"]),
            int(prepared["G"]),
        )
        _ = selection_per_query_bwd_dq(
            prepared["q"].contiguous(),
            prepared["k"].contiguous(),
            prepared["v"].contiguous(),
            do.contiguous(),
            lse,
            delta,
            block_idx.contiguous(),
            prepared["patch_starts"].contiguous(),
            int(prepared["pp"]),
            int(prepared["H"]),
            int(prepared["W"]),
            int(prepared["P"]),
            float(prepared["scale"]),
            int(prepared["G"]),
        )
        _ = selection_per_query_bwd_dkv(
            prepared["q"].contiguous(),
            prepared["k"].contiguous(),
            prepared["v"].contiguous(),
            do.contiguous(),
            lse,
            delta,
            block_idx.contiguous(),
            prepared["patch_starts"].contiguous(),
            int(prepared["pp"]),
            int(prepared["H"]),
            int(prepared["W"]),
            int(prepared["P"]),
            float(prepared["scale"]),
            int(prepared["G"]),
        )
    cuda_sync()

    proton.start(str(args.out), hook="triton")
    proton.deactivate(0)
    proton.activate(0)
    for _ in range(args.iters):
        with scope("selection_iteration", {"size": args.size, "top_n": args.top_n}):
            with scope("selection_scoring"):
                _ = attn._compute_selection_block_idx(prepared["q"], prepared["k_cmp"])
                cuda_sync()
            with scope("selection_forward"):
                _ = selection_attn_2d_per_query_forward(
                    prepared["q"].contiguous(),
                    prepared["k"].contiguous(),
                    prepared["v"].contiguous(),
                    block_idx.contiguous(),
                    prepared["patch_starts"].contiguous(),
                    int(prepared["pp"]),
                    int(prepared["H"]),
                    int(prepared["W"]),
                    int(prepared["P"]),
                    float(prepared["scale"]),
                    int(prepared["G"]),
                )
                cuda_sync()
            with scope("selection_dq"):
                _ = selection_per_query_bwd_dq(
                    prepared["q"].contiguous(),
                    prepared["k"].contiguous(),
                    prepared["v"].contiguous(),
                    do.contiguous(),
                    lse,
                    delta,
                    block_idx.contiguous(),
                    prepared["patch_starts"].contiguous(),
                    int(prepared["pp"]),
                    int(prepared["H"]),
                    int(prepared["W"]),
                    int(prepared["P"]),
                    float(prepared["scale"]),
                    int(prepared["G"]),
                )
                cuda_sync()
            with scope("selection_dkv"):
                _ = selection_per_query_bwd_dkv(
                    prepared["q"].contiguous(),
                    prepared["k"].contiguous(),
                    prepared["v"].contiguous(),
                    do.contiguous(),
                    lse,
                    delta,
                    block_idx.contiguous(),
                    prepared["patch_starts"].contiguous(),
                    int(prepared["pp"]),
                    int(prepared["H"]),
                    int(prepared["W"]),
                    int(prepared["P"]),
                    float(prepared["scale"]),
                    int(prepared["G"]),
                )
                cuda_sync()
    proton.finalize()

    print(f"Saved Proton profile: {args.out}")
    print(f"Split wrapper parity diff: {split_diff}")


if __name__ == "__main__":
    main()
