from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ops.dsa_sparse_mla import packed_sparse_mla_reference
from src.ops.dsa_sparse_mla_autograd import packed_sparse_mla_autograd_forward


def _load_dsa_symbols() -> tuple[type, type]:
    module_path = REPO_ROOT / "src" / "networks" / "dsa_2d.py"
    spec = importlib.util.spec_from_file_location("dsa_2d_sparse_training_bench_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load DSA module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.DSA2DMLAAttention, module.DSA2DMLAConfig


DSA2DMLAAttention, DSA2DMLAConfig = _load_dsa_symbols()


SUPPORTED_NATIVE_H100_MQA_SHAPES: tuple[tuple[int, int], ...] = (
    (64, 512),
    (64, 576),
    (128, 512),
    (128, 576),
)


def _maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_ms(fn, *, device: torch.device, warmup: int = 1, iters: int = 3) -> float:
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
        return {"status": "ok", "forward_backward_ms": _time_ms(fn, device=device, warmup=warmup, iters=iters), "error": None}
    except torch.cuda.OutOfMemoryError as exc:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return {"status": "oom", "forward_backward_ms": None, "error": str(exc)}
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return {"status": "oom", "forward_backward_ms": None, "error": str(exc)}
        raise


def _resolve_dtype(device: torch.device, dtype: torch.dtype | None) -> torch.dtype:
    if dtype is not None:
        return dtype
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def _make_runtime(
    *,
    batch: int,
    h_q: int,
    d_qk: int,
    d_v: int,
    query_tokens: int,
    source_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, torch.Tensor | int | str], torch.Tensor, torch.Tensor]:
    q = torch.randn(batch, h_q, query_tokens, d_qk, device=device, dtype=dtype, requires_grad=True)
    kv = torch.randn(batch, 1, source_tokens, d_qk, device=device, dtype=dtype, requires_grad=True)
    idx = torch.arange(source_tokens, device=device, dtype=torch.int64)
    idx = idx[: max(1, min(source_tokens, query_tokens * 4))]
    idx = idx.view(1, 1, -1).expand(batch, query_tokens, -1).contiguous()
    grad_out = torch.randn(batch, h_q, query_tokens, d_v, device=device, dtype=dtype)
    runtime = {
        "q": q,
        "kv": kv,
        "d_qk": d_qk,
        "d_v": d_v,
        "kv_layout": "latent_then_rope",
    }
    return runtime, idx, grad_out


def _reference_sparse_operator_step(
    runtime: dict[str, torch.Tensor | int | str],
    idx: torch.Tensor,
    grad_out: torch.Tensor,
    *,
    gqa_group_size: int,
    softmax_scale: float,
) -> None:
    q = runtime["q"]
    kv = runtime["kv"]
    out = packed_sparse_mla_reference(
        q,
        kv,
        idx,
        d_v=int(runtime["d_v"]),
        gqa_group_size=gqa_group_size,
        softmax_scale=softmax_scale,
    )
    torch.autograd.grad(out, (q, kv), grad_out, retain_graph=False, create_graph=False)


def _fast_sparse_operator_step(
    runtime: dict[str, torch.Tensor | int | str],
    idx: torch.Tensor,
    grad_out: torch.Tensor,
    *,
    gqa_group_size: int,
    softmax_scale: float,
) -> None:
    q = runtime["q"]
    kv = runtime["kv"]
    out = packed_sparse_mla_autograd_forward(
        runtime,
        idx,
        gqa_group_size=gqa_group_size,
        softmax_scale=softmax_scale,
    )
    torch.autograd.grad(out, (q, kv), grad_out, retain_graph=False, create_graph=False)


def _benchmark_case(
    *,
    name: str,
    batch: int,
    h_q: int,
    d_qk: int,
    d_v: int,
    query_tokens: int,
    source_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    if h_q not in {64, 128}:
        raise ValueError(f"Unsupported h_q={h_q}; expected a native H100 MQA shape")
    if d_qk not in {512, 576}:
        raise ValueError(f"Unsupported d_qk={d_qk}; expected a native H100 MQA shape")
    if d_v != 512:
        raise ValueError(f"Unsupported d_v={d_v}; expected the native H100 MQA value width")

    runtime, idx, grad_out = _make_runtime(
        batch=batch,
        h_q=h_q,
        d_qk=d_qk,
        d_v=d_v,
        query_tokens=query_tokens,
        source_tokens=source_tokens,
        device=device,
        dtype=dtype,
    )
    softmax_scale = d_qk ** -0.5
    case: dict[str, Any] = {
        "name": name,
        "batch": batch,
        "h_q": h_q,
        "h_kv": 1,
        "gqa_group_size": h_q,
        "query_tokens": query_tokens,
        "source_tokens": source_tokens,
        "d_qk": d_qk,
        "d_v": d_v,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "reference_sparse_operator": _time_ms_or_error(
            lambda: _reference_sparse_operator_step(
                runtime,
                idx,
                grad_out,
                gqa_group_size=h_q,
                softmax_scale=softmax_scale,
            ),
            device=device,
            warmup=warmup,
            iters=iters,
        ),
        "fast_sparse_operator": _time_ms_or_error(
            lambda: _fast_sparse_operator_step(
                runtime,
                idx,
                grad_out,
                gqa_group_size=h_q,
                softmax_scale=softmax_scale,
            ),
            device=device,
            warmup=warmup,
            iters=iters,
        ),
    }
    return case


def run_benchmark_smoke(
    output_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    warmup: int = 0,
    iters: int = 1,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    dtype = _resolve_dtype(device, dtype)

    case = _benchmark_case(
        name="native_h64_dqk512_training_smoke",
        batch=1,
        h_q=64,
        d_qk=512,
        d_v=512,
        query_tokens=4,
        source_tokens=16,
        device=device,
        dtype=dtype,
        warmup=warmup,
        iters=iters,
    )
    result = {"device": str(device), "dtype": str(dtype).replace("torch.", ""), "cases": [case]}
    artifact_path = output_dir / "dsa_sparse_training_smoke.json"
    artifact_path.write_text(json.dumps(result, indent=2))
    result["artifact"] = str(artifact_path)
    return result


def run_benchmark_suite(
    *,
    output_dir: str | Path,
    device: str | torch.device,
    dtype: torch.dtype | None = None,
    warmup: int = 5,
    iters: int = 10,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    dtype = _resolve_dtype(device, dtype)

    cases = []
    for h_q, d_qk in SUPPORTED_NATIVE_H100_MQA_SHAPES:
        case_name = f"native_hq{h_q}_dqk{d_qk}_training_step"
        cases.append(
            _benchmark_case(
                name=case_name,
                batch=1,
                h_q=h_q,
                d_qk=d_qk,
                d_v=512,
                query_tokens=16,
                source_tokens=64,
                device=device,
                dtype=dtype,
                warmup=warmup,
                iters=iters,
            )
        )

    result = {"device": str(device), "dtype": str(dtype).replace("torch.", ""), "cases": cases}
    artifact_path = output_dir / f"dsa_sparse_training_step_{device.type}_{str(dtype).replace('torch.', '')}.json"
    artifact_path.write_text(json.dumps(result, indent=2))
    result["artifact"] = str(artifact_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/dsa_diagnostics"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default=None, choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype) if args.dtype is not None else None
    if args.smoke:
        result = run_benchmark_smoke(
            args.output_dir,
            device=args.device,
            dtype=dtype,
            warmup=args.warmup,
            iters=args.iters,
        )
    else:
        result = run_benchmark_suite(
            output_dir=args.output_dir,
            device=args.device,
            dtype=dtype,
            warmup=args.warmup,
            iters=args.iters,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
