#!/usr/bin/env python
"""Benchmark packing overhead against current unpacked NSA attention paths."""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.networks.txunet import NSA2DAttention
from src.ops.selection_attention_2d_per_query import _build_packed_patch_metadata


@dataclass(frozen=True)
class Case:
    name: str
    batch_size: int
    channels: int
    num_heads: int
    gqa_group_size: int
    height: int
    width: int
    patch_size: int
    top_n: int
    window_size: int
    dtype: str


CASES = (
    Case("h64_w64_c64_h4_g4_p8_k8_w16_bf16", 1, 64, 4, 4, 64, 64, 8, 8, 16, "bf16"),
    Case("h64_w64_c64_h4_g4_p8_k16_w16_bf16", 1, 64, 4, 4, 64, 64, 8, 16, 16, "bf16"),
    Case("h128_w128_c64_h4_g4_p8_k8_w16_bf16", 1, 64, 4, 4, 128, 128, 8, 8, 16, "bf16"),
    Case("h128_w128_c64_h4_g4_p8_k16_w16_bf16", 1, 64, 4, 4, 128, 128, 8, 16, 16, "bf16"),
    Case("h256_w256_c64_h4_g4_p8_k8_w16_bf16", 1, 64, 4, 4, 256, 256, 8, 8, 16, "bf16"),
    Case("h256_w256_c64_h4_g4_p8_k16_w16_bf16", 1, 64, 4, 4, 256, 256, 8, 16, 16, "bf16"),
)

WARMUP = 5
ITERS = 10


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


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


def _peak_alloc_mb(fn) -> float:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return float(torch.cuda.max_memory_allocated() / 1024 / 1024)


def _git_sha(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _make_case_tensors(case: Case, device: torch.device) -> tuple[dict[str, torch.Tensor], int]:
    dtype = _dtype_from_name(case.dtype)
    B = case.batch_size
    C = case.channels
    H = case.height
    W = case.width
    T = H * W
    h_q = case.num_heads
    h_kv = h_q // case.gqa_group_size
    d = C // h_q

    tensors: dict[str, torch.Tensor] = {
        "q": torch.randn(B, h_q, T, d, device=device, dtype=dtype),
        "k": torch.randn(B, h_kv, T, d, device=device, dtype=dtype),
        "v": torch.randn(B, h_kv, T, d, device=device, dtype=dtype),
    }
    return tensors, d


def _gather_packed_patch_tables(
    attn: NSA2DAttention,
    k: torch.Tensor,
    v: torch.Tensor,
    unique_patch_ids: torch.Tensor,
    cu_unique_counts: torch.Tensor,
    H: int,
    W: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, h_kv, _, d = k.shape
    p = attn.patch_size
    pp = p * p
    k_patches = attn._bhtd_to_patches(k, B, h_kv, H, W, p).view(B * h_kv, -1, pp, d)
    v_patches = attn._bhtd_to_patches(v, B, h_kv, H, W, p).view(B * h_kv, -1, pp, d)

    packed_k_chunks: list[torch.Tensor] = []
    packed_v_chunks: list[torch.Tensor] = []
    for bh in range(B * h_kv):
        start = int(cu_unique_counts[bh].item())
        end = int(cu_unique_counts[bh + 1].item())
        patch_ids = unique_patch_ids[start:end].to(torch.long)
        packed_k_chunks.append(k_patches[bh].index_select(0, patch_ids))
        packed_v_chunks.append(v_patches[bh].index_select(0, patch_ids))

    return torch.cat(packed_k_chunks, dim=0), torch.cat(packed_v_chunks, dim=0)


def _run_case(case: Case, device: torch.device) -> dict[str, object]:
    dtype = _dtype_from_name(case.dtype)
    attn = NSA2DAttention(
        dim=case.channels,
        num_heads=case.num_heads,
        patch_size=case.patch_size,
        top_n=case.top_n,
        window_size=case.window_size,
        rope_enabled=False,
        gqa_group_size=case.gqa_group_size,
    ).to(device=device, dtype=dtype).eval()

    tensors, d = _make_case_tensors(case, device)
    q = tensors["q"]
    k = tensors["k"]
    v = tensors["v"]
    H = case.height
    W = case.width
    B = case.batch_size
    h_kv = case.num_heads // case.gqa_group_size
    T = H * W

    with torch.no_grad():
        _, k_cmp = attn._compression_branch(q, k, v, H, W)
        block_idx = attn._compute_selection_block_idx(q, k_cmp)
        unique_patch_ids, cu_unique_counts, _ = _build_packed_patch_metadata(block_idx)

    total_slots = B * h_kv * T * case.top_n
    total_unique = int(unique_patch_ids.numel())
    per_head_unique = (cu_unique_counts[1:] - cu_unique_counts[:-1]).to(torch.int64)

    def packing_metadata_build():
        _ = _build_packed_patch_metadata(block_idx)

    def packed_kv_gather():
        _ = _gather_packed_patch_tables(attn, k, v, unique_patch_ids, cu_unique_counts, H, W)

    def unpacked_selection_forward():
        _ = attn._selection_from_block_idx(q, k, v, block_idx, H, W)

    def attention_total_baseline():
        o_cmp, k_cmp_local = attn._compression_branch(q, k, v, H, W)
        o_slc = attn._selection_branch(q, k, v, k_cmp_local, H, W)
        o_win = attn._window_branch(q, k, v, H, W)
        _ = o_cmp + o_slc + o_win

    results = {
        "packing_metadata_build": _timed_cuda(packing_metadata_build, warmup=WARMUP, iters=ITERS),
        "packed_kv_gather": _timed_cuda(packed_kv_gather, warmup=WARMUP, iters=ITERS),
        "unpacked_selection_forward": _timed_cuda(unpacked_selection_forward, warmup=WARMUP, iters=ITERS),
        "attention_total_baseline": _timed_cuda(attention_total_baseline, warmup=WARMUP, iters=ITERS),
    }

    for key, fn in {
        "packing_metadata_build": packing_metadata_build,
        "packed_kv_gather": packed_kv_gather,
        "unpacked_selection_forward": unpacked_selection_forward,
        "attention_total_baseline": attention_total_baseline,
    }.items():
        results[key]["peak_alloc_mb"] = _peak_alloc_mb(fn)

    return {
        "case": asdict(case),
        "head_dim": d,
        "selected_tokens_per_query": int(case.top_n * case.patch_size * case.patch_size),
        "window_tokens_per_query": int(case.window_size * case.window_size),
        "dedup": {
            "total_selected_slots": int(total_slots),
            "total_unique_patches": total_unique,
            "selected_slot_over_unique_ratio": float(total_slots / max(total_unique, 1)),
            "per_head_unique_min": int(per_head_unique.min().item()),
            "per_head_unique_max": int(per_head_unique.max().item()),
            "per_head_unique_mean": float(per_head_unique.float().mean().item()),
        },
        "results": results,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device("cuda")
    cases = [_run_case(case, device=device) for case in CASES]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = REPO_ROOT / "artifacts" / "nsa_diagnostics" / f"selection_packing_vs_unpacked_a100_{stamp}.json"
    payload = {
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "host": subprocess.check_output(["hostname"], text=True).strip(),
            "gpu_name": torch.cuda.get_device_name(device),
            "cuda_visible_devices": str(torch.cuda.current_device()),
            "torch": torch.__version__,
            "python": sys.version,
            "git_sha": _git_sha(REPO_ROOT),
            "warmup": WARMUP,
            "iters": ITERS,
            "notes": "Packing-overhead benchmark on fixed q/k/v. Measures metadata build, packed KV gather, current unpacked selection forward, and current attention-only total baseline.",
        },
        "cases": cases,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(out_path)


if __name__ == "__main__":
    main()
