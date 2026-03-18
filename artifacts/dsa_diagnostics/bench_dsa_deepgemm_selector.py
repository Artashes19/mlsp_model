from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ops.dsa_deepgemm import deepgemm_weighted_relu_logits
from src.ops.dsa_indexer import stable_topk


def _load_dsa_symbols() -> tuple[type, type]:
    module_path = REPO_ROOT / "src" / "networks" / "dsa_2d.py"
    spec = importlib.util.spec_from_file_location("dsa_2d_bench_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load DSA module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.DSA2DMLAAttention, module.DSA2DMLAConfig


DSA2DMLAAttention, DSA2DMLAConfig = _load_dsa_symbols()


NATIVE_CFG = dict(
    dim=1152,
    n_heads=64,
    n_kv_heads=1,
    q_lora_rank=1536,
    kv_lora_rank=512,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
    index_n_heads=1,
    index_head_dim=128,
)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def bench(fn, *, warmup: int, iters: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    synchronize(device)
    return (time.perf_counter() - start) * 1000.0 / iters


def build_models(*, topk: int, device: torch.device) -> tuple[DSA2DMLAAttention, DSA2DMLAAttention]:
    base_cfg = DSA2DMLAConfig(
        **NATIVE_CFG,
        index_topk=topk,
        indexer_mode="streaming",
        indexer_backend="reference",
        sparse_backend="auto",
    )
    deepgemm_cfg = DSA2DMLAConfig(
        **NATIVE_CFG,
        index_topk=topk,
        indexer_mode="streaming",
        indexer_backend="deepgemm",
        sparse_backend="auto",
    )
    base = DSA2DMLAAttention(base_cfg).float()
    state = base.state_dict()
    ref = DSA2DMLAAttention(base_cfg).float()
    dg = DSA2DMLAAttention(deepgemm_cfg).float()
    ref.load_state_dict(state)
    dg.load_state_dict(state)
    ref = ref.to(device=device, dtype=torch.bfloat16).eval()
    dg = dg.to(device=device, dtype=torch.bfloat16).eval()
    return ref, dg


def logits_only_case(
    ref: DSA2DMLAAttention,
    dg: DSA2DMLAAttention,
    x: torch.Tensor,
    *,
    warmup: int,
    iters: int,
) -> dict[str, object]:
    device = x.device
    with torch.inference_mode():
        q, k, w = ref._prepare_indexer_qkw(x)
        logits_ref = ref.indexer(q, k, w)[0]
        logits_dg = deepgemm_weighted_relu_logits(q, k, w)
        idx_ref = stable_topk(logits_ref, k=min(ref.index_topk, logits_ref.shape[-1]))
        idx_dg = stable_topk(logits_dg, k=min(dg.index_topk, logits_dg.shape[-1]))
        result = {
            "status": "ok",
            "max_abs_diff": float((logits_ref - logits_dg).abs().amax().item()),
            "mean_abs_diff": float((logits_ref - logits_dg).abs().mean().item()),
            "topk_match": bool(torch.equal(idx_ref, idx_dg)),
        }
        result["reference_ms"] = bench(lambda: ref.indexer(q, k, w)[0], warmup=warmup, iters=iters, device=device)
        result["deepgemm_ms"] = bench(lambda: deepgemm_weighted_relu_logits(q, k, w), warmup=warmup, iters=iters, device=device)
        result["speedup"] = result["reference_ms"] / result["deepgemm_ms"]
        return result


def selector_case(
    ref: DSA2DMLAAttention,
    dg: DSA2DMLAAttention,
    x: torch.Tensor,
    *,
    warmup: int,
    iters: int,
) -> dict[str, object]:
    device = x.device
    with torch.inference_mode():
        scores_ref, idx_ref = ref.build_indexer_selection(x)
        scores_dg, idx_dg = dg.build_indexer_selection(x)
        result = {
            "status": "ok",
            "idx_match": bool(torch.equal(idx_ref, idx_dg)),
            "scores_max_abs_diff": float((scores_ref - scores_dg).abs().amax().item()),
            "scores_mean_abs_diff": float((scores_ref - scores_dg).abs().mean().item()),
        }
        result["reference_ms"] = bench(lambda: ref.build_indexer_selection(x), warmup=warmup, iters=iters, device=device)
        result["deepgemm_ms"] = bench(lambda: dg.build_indexer_selection(x), warmup=warmup, iters=iters, device=device)
        result["speedup"] = result["reference_ms"] / result["deepgemm_ms"]
        return result


def full_forward_case(
    ref: DSA2DMLAAttention,
    dg: DSA2DMLAAttention,
    x: torch.Tensor,
    *,
    warmup: int,
    iters: int,
) -> dict[str, object]:
    device = x.device
    with torch.inference_mode():
        out_ref = ref(x)
        out_dg = dg(x)
        result = {
            "status": "ok",
            "max_abs_diff": float((out_ref - out_dg).abs().amax().item()),
            "mean_abs_diff": float((out_ref - out_dg).abs().mean().item()),
        }
        result["reference_ms"] = bench(lambda: ref(x), warmup=warmup, iters=iters, device=device)
        result["deepgemm_ms"] = bench(lambda: dg(x), warmup=warmup, iters=iters, device=device)
        result["speedup"] = result["reference_ms"] / result["deepgemm_ms"]
        return result


def run_case(hw: int, topk: int, *, warmup: int, iters: int, device: torch.device) -> dict[str, object]:
    torch.cuda.empty_cache()
    torch.manual_seed(0)
    ref, dg = build_models(topk=topk, device=device)
    x = torch.randn(1, NATIVE_CFG["dim"], hw, hw, device=device, dtype=torch.bfloat16)
    case_result: dict[str, object] = {"hw": hw, "topk": topk}
    for stage_name, fn in (
        ("logits_only", logits_only_case),
        ("selector", selector_case),
        ("full_forward", full_forward_case),
    ):
        try:
            case_result[stage_name] = fn(ref, dg, x, warmup=warmup, iters=iters)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            case_result[stage_name] = {"status": "oom"}
    return case_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    results = {
        "device_name": torch.cuda.get_device_name(0),
        "cases": [],
    }
    for hw in (64, 128, 256):
        for topk in (128, 256):
            results["cases"].append(run_case(hw, topk, warmup=args.warmup, iters=args.iters, device=device))

    args.out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
