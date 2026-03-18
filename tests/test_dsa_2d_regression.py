
import pytest
import torch
import runpy
from pathlib import Path

from src.networks.dsa_2d import DSA2DMLAConfig
from tests.helpers import dsa_reference


def test_dsa_test_scaffold_exists():
    from tests.helpers import dsa_reference

    assert hasattr(dsa_reference, "__file__")


def test_sparse_mla_handles_repeated_and_unsorted_indices():
    cfg = DSA2DMLAConfig(
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
        index_topk=4,
    )
    idx = torch.tensor([[[3, 1, 3, 0]]], dtype=torch.int64)

    dsa_reference.run_sparse_index_regression_case(cfg, idx)


def test_dsa_benchmark_harness_smoke(tmp_path):
    script = Path(__file__).resolve().parents[1] / "artifacts" / "dsa_diagnostics" / "bench_dsa_2d_vs_dense_mla.py"
    namespace = runpy.run_path(str(script))
    result = namespace["run_benchmark_smoke"](output_dir=tmp_path, indexer_mode="streaming")

    assert "cases" in result
    assert len(result["cases"]) == 1
    case = result["cases"][0]
    assert case["name"] == "smoke_4x4_topk_equals_t"
    assert case["shape"] == [1, 32, 4, 4]
    assert "dsa_sparse_ms" in case
    assert "dense_mla_ms" in case
    assert "nsa_ms" in case
    assert "flash_mha_ms" in case
    assert "topk" in case
    assert "num_heads" in case
    assert case["indexer_mode"] == "streaming"


def test_dsa_benchmark_harness_marks_oom_result():
    script = Path(__file__).resolve().parents[1] / "artifacts" / "dsa_diagnostics" / "bench_dsa_2d_vs_dense_mla.py"
    namespace = runpy.run_path(str(script))

    result = namespace["_time_ms_or_error"](
        lambda: (_ for _ in ()).throw(RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")),
        device=torch.device("cpu"),
        warmup=0,
        iters=1,
    )

    assert result["status"] == "oom"
    assert result["ms"] is None


def test_dsa_benchmark_timing_runs_with_grad_disabled():
    script = Path(__file__).resolve().parents[1] / "artifacts" / "dsa_diagnostics" / "bench_dsa_2d_vs_dense_mla.py"
    namespace = runpy.run_path(str(script))
    seen = []

    def _fn():
        seen.append(torch.is_grad_enabled())

    namespace["_time_ms"](_fn, device=torch.device("cpu"), warmup=1, iters=2)

    assert seen
    assert all(flag is False for flag in seen)


def test_dsa_indexer_mode_defaults_to_dense():
    from src.networks.dsa_2d import DSA2DMLAAttention, DSA2DMLAConfig

    cfg = DSA2DMLAConfig(
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
        index_topk=3,
    )

    mod = DSA2DMLAAttention(cfg).float()

    assert mod.indexer_mode == "dense"


def test_dsa_sparse_backend_defaults_to_auto():
    from src.networks.dsa_2d import DSA2DMLAAttention, DSA2DMLAConfig

    cfg = DSA2DMLAConfig(
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
        index_topk=3,
    )

    mod = DSA2DMLAAttention(cfg).float()

    assert mod.sparse_backend == "auto"


def test_dsa_sparse_backend_rejects_unknown_value():
    with pytest.raises(ValueError, match="Unsupported sparse_backend"):
        DSA2DMLAConfig(
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
            index_topk=3,
            sparse_backend="unknown",
        )


def test_flashmla_import_or_none_is_safe():
    from src.ops.dsa_flashmla import flashmla_import_or_none

    result = flashmla_import_or_none()

    assert result is None or callable(result)


def test_dsa_benchmark_harness_records_sparse_backend(tmp_path):
    script = Path(__file__).resolve().parents[1] / "artifacts" / "dsa_diagnostics" / "bench_dsa_2d_vs_dense_mla.py"
    namespace = runpy.run_path(str(script))
    result = namespace["run_benchmark_smoke"](
        output_dir=tmp_path,
        indexer_mode="streaming",
        sparse_backend="flashmla",
    )

    case = result["cases"][0]
    assert case["sparse_backend"] == "flashmla"
