import pytest
import torch
import runpy
import gc
import weakref
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


def test_dsa_indexer_backend_defaults_to_auto():
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

    assert mod.indexer_backend == "auto"


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


def test_dsa_indexer_backend_rejects_unknown_value():
    with pytest.raises(ValueError, match="Unsupported indexer_backend"):
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
            indexer_backend="unknown",
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


def test_sparse_training_bench_schema(tmp_path):
    script = Path(__file__).resolve().parents[1] / "artifacts" / "dsa_diagnostics" / "bench_dsa_sparse_training_step.py"
    namespace = runpy.run_path(str(script))

    assert "DSA2DMLAAttention" in namespace
    assert "DSA2DMLAConfig" in namespace
    assert callable(namespace["run_benchmark_smoke"])

    result = namespace["run_benchmark_smoke"](output_dir=tmp_path)

    assert "cases" in result
    assert len(result["cases"]) == 1

    case = result["cases"][0]
    assert case["name"] == "native_h64_dqk512_training_smoke"
    assert case["h_q"] == 64
    assert case["h_kv"] == 1
    assert case["d_qk"] == 512
    assert case["d_v"] == 512
    assert case["query_tokens"] == 4
    assert case["source_tokens"] == 16

    reference = case["reference_sparse_operator"]
    fast = case["fast_sparse_operator"]
    assert set(reference) == {"status", "forward_backward_ms", "error"}
    assert set(fast) == {"status", "forward_backward_ms", "error"}
    assert reference["status"] == "ok"
    assert fast["status"] == "ok"
    assert reference["forward_backward_ms"] is not None
    assert fast["forward_backward_ms"] is not None


def test_txunet_dsa_vs_flash_trainstep_bench_schema(tmp_path):
    script = (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "dsa_diagnostics"
        / "bench_txunet_dsa_vs_flash_trainstep.py"
    )
    namespace = runpy.run_path(str(script))

    assert callable(namespace["run_benchmark_smoke"])

    result = namespace["run_benchmark_smoke"](output_dir=tmp_path)

    assert "cases" in result
    assert len(result["cases"]) == 1
    assert "artifact" in result

    case = result["cases"][0]
    assert case["name"] == "native_head_txunet_trainstep_smoke"
    assert case["shape"] == [1, 4, 32, 32]
    assert case["base_ch"] == 128
    assert case["depths"] == [1, 1, 1, 1]
    assert case["heads"] == [64, 64, 64, 64]
    assert case["topk"] == 128

    dense = case["dense_flash_attention"]
    dsa = case["dsa_frozen_selector"]
    assert set(dense) == {"status", "train_step_ms", "peak_memory_mb", "error"}
    assert set(dsa) == {"status", "train_step_ms", "peak_memory_mb", "error"}
    assert dense["status"] == "ok"
    assert dsa["status"] == "ok"
    assert dense["train_step_ms"] is not None
    assert dsa["train_step_ms"] is not None


def test_txunet_trainstep_benchmark_releases_dense_model_before_dsa_benchmark():
    script = (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "dsa_diagnostics"
        / "bench_txunet_dsa_vs_flash_trainstep.py"
    )
    namespace = runpy.run_path(str(script))
    benchmark_case = namespace["_benchmark_case"]

    dense_ref: dict[str, weakref.ReferenceType | None] = {"ref": None}

    class _DummyModel(torch.nn.Module):
        def forward(self, x):
            return x

    def _build_dense_model():
        model = _DummyModel()
        dense_ref["ref"] = weakref.ref(model)
        return model

    def _build_dsa_model(*, topk: int):
        gc.collect()
        assert dense_ref["ref"] is not None
        assert dense_ref["ref"]() is None
        return _DummyModel()

    def _bench_model(model, **kwargs):
        return {
            "status": "ok",
            "train_step_ms": 1.0,
            "peak_memory_mb": 0.0,
            "error": None,
        }

    benchmark_case.__globals__["_build_native_head_dense_model"] = _build_dense_model
    benchmark_case.__globals__["_build_native_head_dsa_model"] = _build_dsa_model
    benchmark_case.__globals__["_benchmark_model_train_step"] = _bench_model

    case = benchmark_case(
        name="native_head_txunet_trainstep_smoke",
        shape=(1, 4, 32, 32),
        topk=128,
        device=torch.device("cpu"),
        amp_dtype=torch.float32,
        warmup=0,
        iters=1,
    )

    assert case["dense_flash_attention"]["status"] == "ok"
    assert case["dsa_frozen_selector"]["status"] == "ok"


def test_frozen_selector_helper_marks_selector_parameters_frozen():
    from src.networks.dsa_2d import DSA2DMLAAttention

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
    mod = DSA2DMLAAttention(cfg).float()

    mod.freeze_selector_parameters()

    selector_params = list(mod.selector_parameters())
    assert selector_params

    selector_param_ids = {id(param) for param in selector_params}
    for param in mod.parameters():
        if id(param) in selector_param_ids:
            assert param.requires_grad is False
        else:
            assert param.requires_grad is True


def test_frozen_selector_mode_keeps_forward_on_frozen_path(monkeypatch):
    from src.networks.dsa_2d import DSA2DMLAAttention

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
    mod = DSA2DMLAAttention(cfg).float()
    x = torch.randn(1, cfg.dim, 4, 4, dtype=torch.float32)

    mod.freeze_selector_parameters()

    called = []
    original = mod.forward_sparse_with_frozen_selector

    def wrapped_forward_sparse_with_frozen_selector(x_):
        called.append(True)
        return original(x_)

    monkeypatch.setattr(mod, "forward_sparse_with_frozen_selector", wrapped_forward_sparse_with_frozen_selector)

    mod(x)

    assert called
