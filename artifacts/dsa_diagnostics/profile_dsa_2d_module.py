from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

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


def run_profile_smoke(output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    cfg = _make_cfg(index_topk=16)
    mod = DSA2DMLAAttention(cfg).float()
    x = torch.randn(1, cfg.dim, 4, 4, dtype=torch.float32)

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as prof:
        mod(x)

    summary = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10)
    artifact_path = output_dir / "dsa_profile_smoke.txt"
    artifact_path.write_text(summary)
    return {"artifact": str(artifact_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/dsa_diagnostics"))
    args = parser.parse_args()
    result = run_profile_smoke(args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
