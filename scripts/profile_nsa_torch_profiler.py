#!/usr/bin/env python
"""
NSA2DAttention profiling with torch.profiler (CUDA trace) and compile graph break analysis.

Outputs:
  1. Chrome trace JSON for visual inspection in chrome://tracing
  2. Table of top 20 CUDA kernels by total time
  3. torch.compile graph break report

Usage:
    CUDA_VISIBLE_DEVICES=0 /auto/home/artashes/miniconda3/envs/dev/bin/python scripts/profile_nsa_torch_profiler.py
"""
import sys, os, gc, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

from src.networks.txunet import NSA2DAttention, TransformerBlock


def make_nsa_layer(dim, heads, patch, top_n, window, gqa=1, rope=True, device="cuda", dtype=torch.bfloat16):
    """Create an NSA2DAttention layer with given params."""
    layer = NSA2DAttention(
        dim=dim, num_heads=heads,
        patch_size=patch, top_n=top_n, window_size=window,
        gqa_group_size=gqa, rope_enabled=rope, rope_base=10000.0,
    ).to(device=device, dtype=dtype)
    return layer


def profile_layer(layer, x, label, n_warmup=5, n_active=3, output_dir="profile_traces"):
    """Profile a layer's forward+backward with torch.profiler."""
    os.makedirs(output_dir, exist_ok=True)

    # Warmup
    for _ in range(n_warmup):
        x_w = x.clone().detach().requires_grad_(True)
        out = layer(x_w)
        out.sum().backward()
    torch.cuda.synchronize()

    # Profile
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        for _ in range(n_active):
            x_p = x.clone().detach().requires_grad_(True)
            out = layer(x_p)
            out.sum().backward()
            torch.cuda.synchronize()

    # Export chrome trace
    trace_path = os.path.join(output_dir, f"{label}_trace.json")
    prof.export_chrome_trace(trace_path)
    print(f"\nChrome trace saved: {trace_path}")

    # Print top CUDA kernels
    print(f"\n{'='*80}")
    print(f"TOP 30 CUDA KERNELS — {label}")
    print(f"{'='*80}")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    # Print top CPU ops
    print(f"\n{'='*80}")
    print(f"TOP 20 CPU OPS — {label}")
    print(f"{'='*80}")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    return prof


def check_compile_graph_breaks(layer, x, label):
    """Attempt torch.compile and report graph breaks."""
    print(f"\n{'='*80}")
    print(f"COMPILE GRAPH BREAK ANALYSIS — {label}")
    print(f"{'='*80}")

    import logging
    torch._logging.set_logs(graph_breaks=True)

    try:
        compiled = torch.compile(layer, fullgraph=False, backend="inductor", mode="max-autotune")
        x_c = x.clone().detach().requires_grad_(True)
        out = compiled(x_c)
        out.sum().backward()
        torch.cuda.synchronize()
        print("  torch.compile succeeded (may have graph breaks logged above)")
    except Exception as e:
        print(f"  torch.compile FAILED: {e}")
    finally:
        torch._logging.set_logs(graph_breaks=False)


def time_eager_vs_compile(layer, x, label, n_warmup=10, n_repeat=20):
    """Compare eager vs compiled wall-clock time."""
    print(f"\n{'='*80}")
    print(f"EAGER vs COMPILE TIMING — {label}")
    print(f"{'='*80}")

    # Eager timing
    for _ in range(n_warmup):
        x_w = x.clone().detach().requires_grad_(True)
        layer(x_w).sum().backward()
    torch.cuda.synchronize()

    times_eager = []
    for _ in range(n_repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        x_e = x.clone().detach().requires_grad_(True)
        layer(x_e).sum().backward()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_eager.append((t1 - t0) * 1000)

    mean_eager = sum(times_eager) / len(times_eager)
    print(f"  Eager:    {mean_eager:.2f} ms (fwd+bwd)")

    # Compile timing
    try:
        compiled = torch.compile(layer, fullgraph=False, backend="inductor", mode="max-autotune")
        for _ in range(n_warmup + 3):  # extra warmup for compile
            x_w = x.clone().detach().requires_grad_(True)
            compiled(x_w).sum().backward()
        torch.cuda.synchronize()

        times_compiled = []
        for _ in range(n_repeat):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            x_c = x.clone().detach().requires_grad_(True)
            compiled(x_c).sum().backward()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_compiled.append((t1 - t0) * 1000)

        mean_compiled = sum(times_compiled) / len(times_compiled)
        speedup = mean_eager / mean_compiled
        print(f"  Compiled: {mean_compiled:.2f} ms (fwd+bwd)  [{speedup:.2f}x]")
    except Exception as e:
        print(f"  Compiled: FAILED — {e}")


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available"); sys.exit(1)

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Total GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB")

    # Warmup CUDA
    for _ in range(3):
        a = torch.randn(256, 256, device=device); _ = a @ a.T
    del a; gc.collect(); torch.cuda.empty_cache()

    # --- Config: L0 dimensions (the expensive level) ---
    # B=4 to fit on single A6000, same dims as real training
    B, C, H, W = 4, 48, 256, 256
    heads, patch, top_n, window = 4, 8, 32, 16

    print(f"\nConfig: B={B}, C={C}, H={H}, W={W}, heads={heads}, patch={patch}, top_n={top_n}, window={window}")
    print(f"Tokens per sample: {H*W} = {H*W:,}")

    layer = make_nsa_layer(C, heads, patch, top_n, window, gqa=1, rope=True, device=device, dtype=dtype)
    layer.train()
    x = torch.randn(B, C, H, W, device=device, dtype=dtype)

    # 1) torch.profiler trace
    profile_layer(layer, x, "nsa_L0_bf16")

    # 2) Compile graph break analysis
    layer2 = make_nsa_layer(C, heads, patch, top_n, window, gqa=1, rope=True, device=device, dtype=dtype)
    layer2.train()
    check_compile_graph_breaks(layer2, x, "nsa_L0_bf16")

    # 3) Eager vs compile timing
    layer3 = make_nsa_layer(C, heads, patch, top_n, window, gqa=1, rope=True, device=device, dtype=dtype)
    layer3.train()
    time_eager_vs_compile(layer3, x, "nsa_L0_bf16")

    # --- Also profile a full TransformerBlock with NSA for comparison ---
    print("\n\n" + "#" * 80)
    print("# FULL TransformerBlock WITH NSA (includes FFN)")
    print("#" * 80)

    nsa_attn = make_nsa_layer(C, heads, patch, top_n, window, gqa=1, rope=True, device=device, dtype=dtype)
    block = TransformerBlock(dim=C, heads=heads, expand=2.66, ln_eps=1e-5, attn_module=nsa_attn).to(device=device, dtype=dtype)
    block.train()
    profile_layer(block, x, "transformer_block_nsa_L0_bf16")
    time_eager_vs_compile(block, x, "transformer_block_nsa_L0_bf16")

    # --- Compare: TransformerBlock with standard attention (SRA4 baseline) ---
    print("\n\n" + "#" * 80)
    print("# FULL TransformerBlock WITH SRA4 (baseline)")
    print("#" * 80)

    block_sra = TransformerBlock(dim=C, heads=heads, expand=2.66, ln_eps=1e-5, kv_stride=4, rope_enabled=True, rope_base=10000.0).to(device=device, dtype=dtype)
    block_sra.train()
    profile_layer(block_sra, x, "transformer_block_sra4_L0_bf16")
    time_eager_vs_compile(block_sra, x, "transformer_block_sra4_L0_bf16")

    print("\n\nDone. Check profile_traces/ for Chrome trace JSONs.")


if __name__ == "__main__":
    main()
