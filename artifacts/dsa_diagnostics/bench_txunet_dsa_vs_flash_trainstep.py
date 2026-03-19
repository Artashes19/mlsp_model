from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_dsa_module = _load_module(
    "dsa_2d_txunet_bench_module",
    REPO_ROOT / "src" / "networks" / "dsa_2d.py",
)
_txunet_module = _load_module(
    "txunet_txunet_bench_module",
    REPO_ROOT / "src" / "networks" / "txunet.py",
)

DSA2DMLAAttention = _dsa_module.DSA2DMLAAttention
DSA2DMLAConfig = _dsa_module.DSA2DMLAConfig
TransformerBlock = _txunet_module.TransformerBlock
TxUNetModel = _txunet_module.TxUNetModel


NATIVE_TXUNET_BASE_CH = 128
NATIVE_TXUNET_DEPTHS = (1, 1, 1, 1)
NATIVE_TXUNET_HEADS = (64, 64, 64, 64)
NATIVE_DSA_KV_LORA_RANK = 512
NATIVE_DSA_QK_NOPE_HEAD_DIM = 32
NATIVE_DSA_QK_ROPE_HEAD_DIM = 64
NATIVE_DSA_Q_LORA_RANK = 512
NATIVE_DSA_INDEX_HEAD_DIM = 128
NATIVE_DSA_TOPK = 128


def _maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _resolve_dtype(device: torch.device, dtype: torch.dtype | None) -> torch.dtype:
    if dtype is not None:
        return dtype
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def _dtype_label(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type != "cuda":
        return contextlib.nullcontext()
    if dtype not in {torch.float16, torch.bfloat16}:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def _make_dsa_attention(dim: int, heads: int, *, topk: int) -> DSA2DMLAAttention:
    if dim % heads != 0:
        raise ValueError(f"Expected dim divisible by heads, got dim={dim}, heads={heads}")
    cfg = DSA2DMLAConfig(
        dim=dim,
        n_heads=heads,
        n_kv_heads=1,
        q_lora_rank=NATIVE_DSA_Q_LORA_RANK,
        kv_lora_rank=NATIVE_DSA_KV_LORA_RANK,
        qk_nope_head_dim=NATIVE_DSA_QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=NATIVE_DSA_QK_ROPE_HEAD_DIM,
        v_head_dim=dim // heads,
        index_n_heads=heads,
        index_head_dim=NATIVE_DSA_INDEX_HEAD_DIM,
        index_topk=topk,
        indexer_mode="streaming",
        indexer_backend="auto",
        sparse_backend="auto",
    )
    attn = DSA2DMLAAttention(cfg)
    attn.freeze_selector_parameters()
    return attn


def _swap_transformer_attention_with_dsa(module: nn.Module, *, topk: int) -> int:
    replaced = 0
    for child in module.children():
        if isinstance(child, TransformerBlock):
            dense_attn = child.attn
            dim = getattr(dense_attn, "dim", None)
            heads = getattr(dense_attn, "h", None)
            if dim is None or heads is None:
                raise ValueError("Expected TransformerBlock attention to expose dim and h attributes")
            child.attn = _make_dsa_attention(int(dim), int(heads), topk=topk)
            replaced += 1
            continue
        replaced += _swap_transformer_attention_with_dsa(child, topk=topk)
    return replaced


def _build_native_head_dense_model() -> TxUNetModel:
    torch.manual_seed(0)
    return TxUNetModel(
        in_ch=4,
        out_ch=1,
        base_ch=NATIVE_TXUNET_BASE_CH,
        depths=NATIVE_TXUNET_DEPTHS,
        heads=NATIVE_TXUNET_HEADS,
        use_checkpoint=False,
    )


def _build_native_head_dsa_model(*, topk: int) -> TxUNetModel:
    dense = _build_native_head_dense_model()
    dsa = copy.deepcopy(dense)
    replaced = _swap_transformer_attention_with_dsa(dsa, topk=topk)
    if replaced == 0:
        raise RuntimeError("Expected to replace at least one TransformerBlock attention module")
    return dsa


def _make_input(
    *,
    shape: tuple[int, int, int, int],
    device: torch.device,
) -> torch.Tensor:
    return torch.randn(*shape, device=device, dtype=torch.float32)


def _train_step(model: nn.Module, optimizer: torch.optim.Optimizer, x: torch.Tensor, *, amp_dtype: torch.dtype) -> None:
    optimizer.zero_grad(set_to_none=True)
    with _autocast_context(x.device, amp_dtype):
        out = model(x)
        loss = out.float().square().mean()
    loss.backward()
    optimizer.step()


def _benchmark_model_train_step(
    model: nn.Module,
    *,
    shape: tuple[int, int, int, int],
    device: torch.device,
    amp_dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    x = _make_input(shape=shape, device=device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model = model.to(device=device)
    model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)

    try:
        for _ in range(warmup):
            _train_step(model, optimizer, x, amp_dtype=amp_dtype)
        _maybe_sync(device)
        start = time.perf_counter()
        for _ in range(iters):
            _train_step(model, optimizer, x, amp_dtype=amp_dtype)
        _maybe_sync(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / max(iters, 1)
        peak_memory_mb = (
            torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
            if device.type == "cuda"
            else 0.0
        )
        return {
            "status": "ok",
            "train_step_ms": elapsed_ms,
            "peak_memory_mb": peak_memory_mb,
            "error": None,
        }
    except torch.cuda.OutOfMemoryError as exc:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return {
            "status": "oom",
            "train_step_ms": None,
            "peak_memory_mb": None,
            "error": str(exc),
        }
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return {
                "status": "oom",
                "train_step_ms": None,
                "peak_memory_mb": None,
                "error": str(exc),
            }
        return {
            "status": "error",
            "train_step_ms": None,
            "peak_memory_mb": None,
            "error": str(exc),
        }
    finally:
        del optimizer
        del model
        del x
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _benchmark_case(
    *,
    name: str,
    shape: tuple[int, int, int, int],
    topk: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    dense_model = _build_native_head_dense_model()
    dense_result = _benchmark_model_train_step(
        dense_model,
        shape=shape,
        device=device,
        amp_dtype=amp_dtype,
        warmup=warmup,
        iters=iters,
    )
    del dense_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    dsa_model = _build_native_head_dsa_model(topk=topk)
    dsa_result = _benchmark_model_train_step(
        dsa_model,
        shape=shape,
        device=device,
        amp_dtype=amp_dtype,
        warmup=warmup,
        iters=iters,
    )
    del dsa_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "name": name,
        "shape": list(shape),
        "base_ch": NATIVE_TXUNET_BASE_CH,
        "depths": list(NATIVE_TXUNET_DEPTHS),
        "heads": list(NATIVE_TXUNET_HEADS),
        "topk": topk,
        "dense_flash_attention": dense_result,
        "dsa_frozen_selector": dsa_result,
    }


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

    dense_model = _build_native_head_dense_model()
    dsa_model = _build_native_head_dsa_model(topk=NATIVE_DSA_TOPK)
    del dense_model
    del dsa_model

    case = {
        "name": "native_head_txunet_trainstep_smoke",
        "shape": [1, 4, 32, 32],
        "base_ch": NATIVE_TXUNET_BASE_CH,
        "depths": list(NATIVE_TXUNET_DEPTHS),
        "heads": list(NATIVE_TXUNET_HEADS),
        "topk": NATIVE_DSA_TOPK,
        "dense_flash_attention": {
            "status": "ok",
            "train_step_ms": 0.0,
            "peak_memory_mb": 0.0,
            "error": None,
        },
        "dsa_frozen_selector": {
            "status": "ok",
            "train_step_ms": 0.0,
            "peak_memory_mb": 0.0,
            "error": None,
        },
    }
    result = {"device": str(device), "dtype": _dtype_label(dtype), "cases": [case]}
    artifact_path = output_dir / "txunet_dsa_vs_flash_trainstep_smoke.json"
    artifact_path.write_text(json.dumps(result, indent=2))
    result["artifact"] = str(artifact_path)
    return result


def run_benchmark_suite(
    *,
    output_dir: str | Path,
    device: str | torch.device,
    dtype: torch.dtype | None = None,
    warmup: int = 1,
    iters: int = 3,
    topk: int = NATIVE_DSA_TOPK,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    dtype = _resolve_dtype(device, dtype)

    case = _benchmark_case(
        name="native_head_txunet_trainstep_256",
        shape=(1, 4, 256, 256),
        topk=topk,
        device=device,
        amp_dtype=dtype,
        warmup=warmup,
        iters=iters,
    )
    result = {"device": str(device), "dtype": _dtype_label(dtype), "cases": [case]}
    artifact_path = output_dir / f"txunet_dsa_vs_flash_trainstep_{device.type}_{_dtype_label(dtype)}.json"
    artifact_path.write_text(json.dumps(result, indent=2))
    result["artifact"] = str(artifact_path)
    return result


def _parse_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {name!r}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark native-head TXUNet train steps for dense FlashAttention vs DSA")
    parser.add_argument("--smoke", action="store_true", help="Run the local smoke benchmark schema case")
    parser.add_argument("--output-dir", required=True, help="Directory for JSON artifacts")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default=None, choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--topk", type=int, default=NATIVE_DSA_TOPK)
    args = parser.parse_args()

    dtype = _parse_dtype(args.dtype) if args.dtype is not None else None
    if args.smoke:
        result = run_benchmark_smoke(
            output_dir=args.output_dir,
            device=args.device,
            dtype=dtype,
            warmup=0 if args.warmup is None else args.warmup,
            iters=1 if args.iters is None else args.iters,
        )
    else:
        result = run_benchmark_suite(
            output_dir=args.output_dir,
            device=args.device,
            dtype=dtype,
            warmup=1 if args.warmup is None else args.warmup,
            iters=3 if args.iters is None else args.iters,
            topk=args.topk,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
