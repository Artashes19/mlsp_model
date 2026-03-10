from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.networks.dsa_2d import DSA2DMLAAttention, DSA2DMLAConfig
from src.networks.txunet import EfficientGlobalAttention, NSA2DAttention


def _make_cfg(*, dim: int, n_heads: int, n_kv_heads: int, index_topk: int) -> DSA2DMLAConfig:
    head_dim = dim // n_heads
    if head_dim <= 0:
        raise ValueError(f"Expected positive head_dim, got dim={dim}, n_heads={n_heads}")
    qk_rope_head_dim = max(8, min(64, max(4, head_dim // 2)))
    qk_rope_head_dim = max(4, (qk_rope_head_dim // 4) * 4)
    qk_nope_head_dim = max(8, head_dim - qk_rope_head_dim)

    index_head_dim = 16 if head_dim < 32 else 64
    q_lora_rank = max(dim // 2, head_dim)
    kv_lora_rank = max(dim // 3, head_dim)
    index_n_heads = max(1, min(n_heads, n_kv_heads))
    v_head_dim = head_dim
    return DSA2DMLAConfig(
        dim=dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
    )


def _make_nsa(dim: int, n_heads: int, n_kv_heads: int, index_topk: int, spatial: int) -> NSA2DAttention:
    patch_size = 4
    top_n = max(1, math.ceil(index_topk / float(patch_size * patch_size)))
    window_size = min(4, spatial)
    gqa_group_size = n_heads // n_kv_heads
    return NSA2DAttention(
        dim=dim,
        num_heads=n_heads,
        patch_size=patch_size,
        top_n=top_n,
        window_size=window_size,
        gqa_group_size=gqa_group_size,
        rope_enabled=False,
    )


def _make_flash_mha(dim: int, n_heads: int) -> EfficientGlobalAttention:
    return EfficientGlobalAttention(
        dim=dim,
        num_heads=n_heads,
        kv_stride=1,
        rope_enabled=False,
    )


def _prepare_case_modules(
    *,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    index_topk: int,
    spatial: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Any]:
    cfg = _make_cfg(dim=dim, n_heads=n_heads, n_kv_heads=n_kv_heads, index_topk=index_topk)
    dsa = DSA2DMLAAttention(cfg).to(device=device, dtype=dtype)
    nsa = _make_nsa(dim, n_heads, n_kv_heads, index_topk, spatial).to(device=device, dtype=dtype)
    flash = _make_flash_mha(dim, n_heads).to(device=device, dtype=dtype)
    return {"cfg": cfg, "dsa": dsa, "nsa": nsa, "flash": flash}


def _maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_ms(fn, *, device: torch.device, warmup: int = 1, iters: int = 3) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        _maybe_sync(device)
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        _maybe_sync(device)
    return (time.perf_counter() - start) * 1000.0 / iters


def _time_ms_or_error(fn, *, device: torch.device, warmup: int = 1, iters: int = 3) -> dict[str, Any]:
    try:
        return {"ms": _time_ms(fn, device=device, warmup=warmup, iters=iters), "status": "ok"}
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return {"ms": None, "status": "oom", "error": str(exc)}
        raise


def _benchmark_case(
    *,
    name: str,
    batch: int,
    dim: int,
    spatial: int,
    n_heads: int,
    n_kv_heads: int,
    index_topk: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    torch.manual_seed(0)
    modules = _prepare_case_modules(
        dim=dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        index_topk=index_topk,
        spatial=spatial,
        dtype=dtype,
        device=device,
    )
    cfg: DSA2DMLAConfig = modules["cfg"]
    dsa: DSA2DMLAAttention = modules["dsa"]
    nsa: NSA2DAttention = modules["nsa"]
    flash: EfficientGlobalAttention = modules["flash"]
    dsa.eval()
    nsa.eval()
    flash.eval()
    x = torch.randn(batch, dim, spatial, spatial, dtype=dtype, device=device)

    dense_mla = _time_ms_or_error(lambda: dsa.forward_dense_reference(x), device=device, warmup=warmup, iters=iters)
    dsa_sparse = _time_ms_or_error(lambda: dsa(x), device=device, warmup=warmup, iters=iters)
    nsa_result = _time_ms_or_error(lambda: nsa(x), device=device, warmup=warmup, iters=iters)
    flash_mha = _time_ms_or_error(lambda: flash(x), device=device, warmup=warmup, iters=iters)

    return {
        "name": name,
        "shape": [batch, dim, spatial, spatial],
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "num_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "gqa_group_size": n_heads // n_kv_heads,
        "topk": index_topk,
        "selected_tokens": index_topk,
        "nsa_patch_size": nsa.patch_size,
        "nsa_top_n": nsa.top_n,
        "dense_mla_ms": dense_mla["ms"],
        "dense_mla_status": dense_mla["status"],
        "dsa_sparse_ms": dsa_sparse["ms"],
        "dsa_sparse_status": dsa_sparse["status"],
        "nsa_ms": nsa_result["ms"],
        "nsa_status": nsa_result["status"],
        "flash_mha_ms": flash_mha["ms"],
        "flash_mha_status": flash_mha["status"],
        "qk_head_dim": cfg.qk_nope_head_dim + cfg.qk_rope_head_dim,
        "v_head_dim": cfg.v_head_dim,
        "index_head_dim": cfg.index_head_dim,
    }


def run_benchmark_smoke(output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    case = _benchmark_case(
        name="smoke_4x4_topk_equals_t",
        batch=1,
        dim=32,
        spatial=4,
        n_heads=4,
        n_kv_heads=2,
        index_topk=16,
        dtype=torch.float32,
        device=torch.device("cpu"),
        warmup=0,
        iters=1,
    )
    result = {"cases": [case]}
    artifact_path = output_dir / "dsa_benchmark_smoke.json"
    artifact_path.write_text(json.dumps(result, indent=2))
    result["artifact"] = str(artifact_path)
    return result


def run_benchmark_suite(
    *,
    output_dir: str | Path,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int = 5,
    iters: int = 10,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        _benchmark_case(
            name="128x128_c384_h6_g3_topk256",
            batch=1,
            dim=384,
            spatial=128,
            n_heads=6,
            n_kv_heads=2,
            index_topk=256,
            dtype=dtype,
            device=device,
            warmup=warmup,
            iters=iters,
        ),
        _benchmark_case(
            name="256x256_c384_h6_g3_topk256",
            batch=1,
            dim=384,
            spatial=256,
            n_heads=6,
            n_kv_heads=2,
            index_topk=256,
            dtype=dtype,
            device=device,
            warmup=warmup,
            iters=iters,
        ),
        _benchmark_case(
            name="128x128_c512_h8_g4_topk256",
            batch=1,
            dim=512,
            spatial=128,
            n_heads=8,
            n_kv_heads=2,
            index_topk=256,
            dtype=dtype,
            device=device,
            warmup=warmup,
            iters=iters,
        ),
        _benchmark_case(
            name="256x256_c512_h8_g4_topk256",
            batch=1,
            dim=512,
            spatial=256,
            n_heads=8,
            n_kv_heads=2,
            index_topk=256,
            dtype=dtype,
            device=device,
            warmup=warmup,
            iters=iters,
        ),
    ]
    result = {"cases": cases}
    artifact_path = output_dir / f"dsa_benchmark_{device.type}_{str(dtype).replace('torch.', '')}.json"
    artifact_path.write_text(json.dumps(result, indent=2))
    result["artifact"] = str(artifact_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/dsa_diagnostics"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    if args.smoke:
        result = run_benchmark_smoke(args.output_dir)
    else:
        result = run_benchmark_suite(
            output_dir=args.output_dir,
            device=device,
            dtype=dtype,
            warmup=args.warmup,
            iters=args.iters,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
