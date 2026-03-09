from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.networks.dsa_2d import DSA2DMLAAttention, DSA2DMLAConfig


def _make_cfg(index_topk: int) -> DSA2DMLAConfig:
    return DSA2DMLAConfig(
        dim=32,
        n_heads=4,
        n_kv_heads=2,
        q_lora_rank=16,
        kv_lora_rank=12,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=index_topk,
    )


def _time_ms(fn) -> float:
    start = time.perf_counter()
    fn()
    return (time.perf_counter() - start) * 1000.0


def run_benchmark_smoke(output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    cfg = _make_cfg(index_topk=16)
    mod = DSA2DMLAAttention(cfg).float()
    x = torch.randn(1, cfg.dim, 4, 4, dtype=torch.float32)

    dense_ms = _time_ms(lambda: mod.forward_dense_reference(x))
    sparse_ms = _time_ms(lambda: mod.forward_sparse_with_forced_topk(x, topk_equals_t=True))
    integrated_ms = _time_ms(lambda: mod(x))

    result = {
        "cases": [
            {
                "name": "smoke_4x4_topk_equals_t",
                "shape": [1, cfg.dim, 4, 4],
                "dense_ms": dense_ms,
                "sparse_ms": sparse_ms,
                "integrated_ms": integrated_ms,
            }
        ]
    }
    artifact_path = output_dir / "dsa_benchmark_smoke.json"
    artifact_path.write_text(json.dumps(result, indent=2))
    result["artifact"] = str(artifact_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/dsa_diagnostics"))
    args = parser.parse_args()
    result = run_benchmark_smoke(args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
