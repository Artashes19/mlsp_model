
import pytest
import torch


def test_dsa_test_scaffold_exists():
    from tests.helpers import dsa_reference

    assert hasattr(dsa_reference, "__file__")


def test_fp8_test_scaffold_exists():
    from tests.helpers import fp8_reference

    assert hasattr(fp8_reference, "__file__")


def test_fwht_matches_naive_reference():
    from src.ops import dsa_indexer
    from tests.helpers import fp8_reference

    x = torch.randn(2, 8)
    ref = fp8_reference.naive_fwht(x)
    out = dsa_indexer.fwht_last_dim(x)

    torch.testing.assert_close(out, ref)


def test_weighted_relu_index_score_matches_naive_reference():
    from src.ops import dsa_indexer
    from tests.helpers import fp8_reference

    q = torch.randn(1, 2, 4, 128)
    k = torch.randn(1, 2, 4, 128)
    w = torch.randn(1, 4, 2)

    ref = fp8_reference.naive_weighted_relu_index(q, k, w)
    out = dsa_indexer.weighted_relu_index_score(q, k, w)

    torch.testing.assert_close(out, ref)


def test_fp8_quant_dequant_matches_reference_scales():
    from src.ops import dsa_indexer
    from tests.helpers import fp8_reference

    x = torch.randn(4, 128)
    ref_q, ref_s = fp8_reference.reference_fp8_quant(x)
    q, s = dsa_indexer.act_quant_reference_safe(x)

    torch.testing.assert_close(s, ref_s)
    assert q.shape == ref_q.shape


def test_indexer_topk_handles_ties_and_all_negative_cases():
    from src.ops import dsa_indexer

    scores = torch.tensor([[[-1.0, -1.0, -2.0, -3.0]]])
    idx = dsa_indexer.stable_topk(scores, k=2)

    assert idx.shape[-1] == 2
    assert idx.tolist() == [[[0, 1]]]


def test_streaming_weighted_relu_topk_matches_dense_small_case():
    from src.ops import dsa_indexer

    torch.manual_seed(0)
    q = torch.randn(1, 2, 3, 8)
    k = torch.randn(1, 2, 5, 8)
    w = torch.randn(1, 3, 2)

    dense_scores = dsa_indexer.weighted_relu_index_score(q, k, w)
    dense_idx = dsa_indexer.stable_topk(dense_scores, k=2)
    dense_top_scores = torch.gather(dense_scores, dim=-1, index=dense_idx)

    stream_scores, stream_idx = dsa_indexer.streaming_weighted_relu_topk(q, k, w, topk=2, block_s=2)

    torch.testing.assert_close(stream_scores, dense_top_scores)
    torch.testing.assert_close(stream_idx, dense_idx)


def test_streaming_weighted_relu_topk_breaks_ties_by_lower_index():
    from src.ops import dsa_indexer

    q = torch.ones(1, 1, 1, 4)
    k = torch.tensor([[[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]]])
    w = torch.ones(1, 1, 1)

    stream_scores, stream_idx = dsa_indexer.streaming_weighted_relu_topk(q, k, w, topk=2, block_s=1)

    assert stream_scores.tolist() == [[[4.0, 4.0]]]
    assert stream_idx.tolist() == [[[0, 1]]]


def test_streaming_weighted_relu_topk_is_block_size_invariant():
    from src.ops import dsa_indexer

    torch.manual_seed(1)
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 6, 8)
    w = torch.randn(1, 4, 2)

    scores_block_1, idx_block_1 = dsa_indexer.streaming_weighted_relu_topk(q, k, w, topk=3, block_s=1)
    scores_block_4, idx_block_4 = dsa_indexer.streaming_weighted_relu_topk(q, k, w, topk=3, block_s=4)

    torch.testing.assert_close(scores_block_1, scores_block_4)
    torch.testing.assert_close(idx_block_1, idx_block_4)


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


def test_dsa_indexer_backend_rejects_unknown_value():
    from src.networks.dsa_2d import DSA2DMLAConfig

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


def test_auto_indexer_backend_uses_deepgemm_when_supported_and_grad_disabled(monkeypatch):
    import src.networks.dsa_2d as dsa_2d_module
    from src.networks.dsa_2d import DSA2DMLAAttention, DSA2DMLAConfig

    cfg = DSA2DMLAConfig(
        dim=32,
        n_heads=4,
        n_kv_heads=1,
        q_lora_rank=16,
        kv_lora_rank=12,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=3,
        indexer_backend="auto",
    )
    mod = DSA2DMLAAttention(cfg).float()
    x = torch.randn(1, cfg.dim, 2, 2, dtype=torch.float32)
    sentinel_logits = torch.full((1, 4, 4), 3.0, dtype=torch.float32)
    sentinel_idx = torch.tensor([[[0, 2, 1]]], dtype=torch.int64).expand(1, 4, 3).clone()

    monkeypatch.setattr(dsa_2d_module, "deepgemm_is_supported", lambda **kwargs: True)
    monkeypatch.setattr(
        dsa_2d_module,
        "deepgemm_weighted_relu_logits",
        lambda *args, **kwargs: sentinel_logits,
        raising=False,
    )
    monkeypatch.setattr(
        mod.indexer,
        "forward",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("reference indexer path should not run")),
    )
    monkeypatch.setattr(
        dsa_2d_module,
        "stable_topk",
        lambda scores, k: sentinel_idx,
    )

    with torch.inference_mode():
        scores, idx = mod.build_indexer_selection(x)

    torch.testing.assert_close(scores, torch.gather(sentinel_logits, dim=-1, index=sentinel_idx))
    torch.testing.assert_close(idx, sentinel_idx)


def test_auto_indexer_backend_falls_back_to_reference_when_grad_enabled(monkeypatch):
    import src.networks.dsa_2d as dsa_2d_module
    from src.networks.dsa_2d import DSA2DMLAAttention, DSA2DMLAConfig

    cfg = DSA2DMLAConfig(
        dim=32,
        n_heads=4,
        n_kv_heads=1,
        q_lora_rank=16,
        kv_lora_rank=12,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=3,
        indexer_backend="auto",
    )
    mod = DSA2DMLAAttention(cfg).float()
    x = torch.randn(1, cfg.dim, 2, 2, dtype=torch.float32)
    sentinel_logits = torch.full((1, 4, 4), 5.0, dtype=torch.float32)
    sentinel_idx = torch.tensor([[[3, 1, 0]]], dtype=torch.int64).expand(1, 4, 3).clone()

    monkeypatch.setattr(dsa_2d_module, "deepgemm_is_supported", lambda **kwargs: True)
    monkeypatch.setattr(
        dsa_2d_module,
        "deepgemm_weighted_relu_logits",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("deepgemm path should not run with grad enabled")),
        raising=False,
    )
    monkeypatch.setattr(mod.indexer, "forward", lambda *args, **kwargs: (sentinel_logits, sentinel_idx))

    scores, idx = mod.build_indexer_selection(x)

    torch.testing.assert_close(scores, torch.gather(sentinel_logits, dim=-1, index=sentinel_idx))
    torch.testing.assert_close(idx, sentinel_idx)


def test_deepgemm_import_or_none_is_safe():
    from src.ops.dsa_deepgemm import deepgemm_import_or_none

    result = deepgemm_import_or_none()

    assert result is None or callable(result)


def test_deepgemm_support_check_rejects_cpu():
    from src.ops.dsa_deepgemm import deepgemm_is_supported

    assert not deepgemm_is_supported(device=torch.device("cpu"), n_kv_heads=1)


def test_deepgemm_support_check_rejects_non_mqa():
    from src.ops.dsa_deepgemm import deepgemm_is_supported

    assert not deepgemm_is_supported(
        device=torch.device("cuda"),
        n_kv_heads=2,
        sm=(9, 0),
    )


def test_deepgemm_support_check_rejects_pre_sm90():
    from src.ops.dsa_deepgemm import deepgemm_is_supported

    assert not deepgemm_is_supported(
        device=torch.device("cuda"),
        n_kv_heads=1,
        sm=(8, 0),
    )


def test_deepgemm_support_check_rejects_unsupported_index_head_count():
    from src.ops.dsa_deepgemm import deepgemm_is_supported

    assert not deepgemm_is_supported(
        device=torch.device("cuda"),
        n_kv_heads=1,
        index_n_heads=1,
        sm=(9, 0),
    )


def test_deepgemm_support_check_rejects_unsupported_index_head_dim():
    from src.ops.dsa_deepgemm import deepgemm_is_supported

    assert not deepgemm_is_supported(
        device=torch.device("cuda"),
        n_kv_heads=1,
        index_n_heads=32,
        index_head_dim=16,
        sm=(9, 0),
    )


def _fake_deepgemm_mqa_kernel(q, kv, weights, cu_seq_len_k_start, cu_seq_len_k_end, clean_logits):
    kv_q, kv_scale = kv
    kv_deq = kv_q.to(dtype=torch.float32) * kv_scale.to(dtype=torch.float32).unsqueeze(-1)
    out = torch.full(
        (q.shape[0], kv_deq.shape[0]),
        float("-inf") if clean_logits else 0.0,
        device=q.device,
        dtype=torch.float32,
    )
    for i in range(q.shape[0]):
        start = int(cu_seq_len_k_start[i].item())
        end = int(cu_seq_len_k_end[i].item())
        dots = torch.einsum("hd,sd->hs", q[i].to(dtype=torch.float32), kv_deq[start:end])
        logits = torch.relu(dots) * weights[i].to(dtype=torch.float32).unsqueeze(-1)
        out[i, start:end] = logits.sum(dim=0)
    return out


def test_deepgemm_logits_wrapper_matches_quantized_reference_with_fake_kernel(monkeypatch):
    from src.ops import dsa_deepgemm, dsa_indexer

    torch.manual_seed(0)
    q = torch.randn(2, 2, 3, 16, dtype=torch.float32)
    k_base = torch.randn(2, 1, 4, 16, dtype=torch.float32)
    k = k_base.expand(-1, 2, -1, -1).contiguous()
    w = torch.randn(2, 3, 2, dtype=torch.float32)

    monkeypatch.setattr(dsa_deepgemm, "deepgemm_import_or_none", lambda: _fake_deepgemm_mqa_kernel)

    q_q, q_scale = dsa_indexer.act_quant_reference_safe(q)
    k_q, k_scale = dsa_indexer.act_quant_reference_safe(k)
    ref = dsa_indexer.weighted_relu_index_score(
        q_q.to(dtype=torch.float32) * q_scale,
        k_q.to(dtype=torch.float32) * k_scale,
        w,
    )
    out = dsa_deepgemm.deepgemm_weighted_relu_logits(q, k, w)

    torch.testing.assert_close(out, ref)


def test_deepgemm_selection_matches_reference_with_fake_kernel(monkeypatch):
    import src.networks.dsa_2d as dsa_2d_module
    from src.networks.dsa_2d import DSA2DMLAAttention, DSA2DMLAConfig
    from src.ops import dsa_deepgemm

    cfg = DSA2DMLAConfig(
        dim=32,
        n_heads=4,
        n_kv_heads=1,
        q_lora_rank=16,
        kv_lora_rank=12,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=3,
        indexer_backend="deepgemm",
    )
    mod = DSA2DMLAAttention(cfg).float()
    x = torch.randn(1, cfg.dim, 2, 2, dtype=torch.float32)

    monkeypatch.setattr(dsa_2d_module, "deepgemm_is_supported", lambda **kwargs: True)
    monkeypatch.setattr(dsa_deepgemm, "deepgemm_import_or_none", lambda: _fake_deepgemm_mqa_kernel)
    monkeypatch.setattr(
        dsa_2d_module,
        "deepgemm_weighted_relu_logits",
        dsa_deepgemm.deepgemm_weighted_relu_logits,
    )

    logits_ref, idx_ref = mod.build_indexer_logits(x)
    with torch.inference_mode():
        scores, idx = mod.build_indexer_selection(x)

    torch.testing.assert_close(idx, idx_ref)
    torch.testing.assert_close(scores, torch.gather(logits_ref, dim=-1, index=idx_ref))


def test_dsa2d_indexer_returns_logits_and_indices():
    from src.networks.dsa_2d import DSA2DIndexer, DSA2DMLAConfig

    cfg = DSA2DMLAConfig(dim=32, n_heads=4, n_kv_heads=2, index_n_heads=2, index_head_dim=16, index_topk=3)
    mod = DSA2DIndexer(cfg)
    q = torch.randn(1, cfg.index_n_heads, 4, cfg.index_head_dim)
    k = torch.randn(1, cfg.index_n_heads, 4, cfg.index_head_dim)
    w = torch.randn(1, 4, cfg.index_n_heads)

    logits, idx = mod(q, k, w)

    assert logits.shape == (1, 4, 4)
    assert idx.shape == (1, 4, cfg.index_topk)


def test_build_indexer_logits_matches_reference_preprocessing_path():
    from src.networks.dsa_2d import DSA2DMLAAttention, DSA2DMLAConfig
    from tests.helpers import dsa_reference

    torch.manual_seed(0)
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
    x = torch.randn(1, cfg.dim, 2, 2, dtype=torch.float32)

    ref_logits, ref_idx = dsa_reference.indexer_logits_reference(mod, x)
    logits, idx = mod.build_indexer_logits(x)

    torch.testing.assert_close(logits, ref_logits)
    torch.testing.assert_close(idx, ref_idx)


def test_build_indexer_logits_uses_dense_path_even_in_streaming_mode(monkeypatch):
    import src.networks.dsa_2d as dsa_2d_module
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
        indexer_mode="streaming",
    )
    mod = DSA2DMLAAttention(cfg).float()
    x = torch.randn(1, cfg.dim, 2, 2, dtype=torch.float32)
    sentinel_logits = torch.full((1, 4, 4), 7.0, dtype=torch.float32)
    sentinel_idx = torch.tensor([[[0, 1, 2]]], dtype=torch.int64).expand(1, 4, 3).clone()

    def _dense_forward(q, k, w):
        return sentinel_logits, sentinel_idx

    def _streaming_fail(*args, **kwargs):
        raise AssertionError("streaming helper should not run in build_indexer_logits")

    monkeypatch.setattr(mod.indexer, "forward", _dense_forward)
    monkeypatch.setattr(dsa_2d_module, "streaming_weighted_relu_topk", _streaming_fail)

    logits, idx = mod.build_indexer_logits(x)

    torch.testing.assert_close(logits, sentinel_logits)
    torch.testing.assert_close(idx, sentinel_idx)


def test_runtime_selection_dense_uses_dense_path(monkeypatch):
    import src.networks.dsa_2d as dsa_2d_module
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
        indexer_mode="dense",
    )
    mod = DSA2DMLAAttention(cfg).float()
    x = torch.randn(1, cfg.dim, 2, 2, dtype=torch.float32)
    sentinel_logits = torch.full((1, 4, 4), 5.0, dtype=torch.float32)
    sentinel_idx = torch.tensor([[[3, 1, 0]]], dtype=torch.int64).expand(1, 4, 3).clone()

    def _dense_forward(*args, **kwargs):
        return sentinel_logits, sentinel_idx

    def _streaming_fail(*args, **kwargs):
        raise AssertionError("streaming helper should not run in dense runtime mode")

    monkeypatch.setattr(mod.indexer, "forward", _dense_forward)
    monkeypatch.setattr(dsa_2d_module, "streaming_weighted_relu_topk", _streaming_fail)

    scores, idx = mod.build_indexer_selection(x)
    expected_scores = torch.gather(sentinel_logits, dim=-1, index=sentinel_idx)

    torch.testing.assert_close(scores, expected_scores)
    torch.testing.assert_close(idx, sentinel_idx)


def test_runtime_selection_streaming_uses_streaming_helper(monkeypatch):
    import src.networks.dsa_2d as dsa_2d_module
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
        indexer_mode="streaming",
    )
    mod = DSA2DMLAAttention(cfg).float()
    x = torch.randn(1, cfg.dim, 2, 2, dtype=torch.float32)
    sentinel_scores = torch.full((1, 4, 3), 5.0, dtype=torch.float32)
    sentinel_idx = torch.tensor([[[3, 1, 0]]], dtype=torch.int64).expand(1, 4, 3).clone()

    def _dense_fail(*args, **kwargs):
        raise AssertionError("dense indexer should not run in streaming runtime mode")

    def _streaming(*args, **kwargs):
        return sentinel_scores, sentinel_idx

    monkeypatch.setattr(mod.indexer, "forward", _dense_fail)
    monkeypatch.setattr(dsa_2d_module, "streaming_weighted_relu_topk", _streaming)

    scores, idx = mod.build_indexer_selection(x)

    torch.testing.assert_close(scores, sentinel_scores)
    torch.testing.assert_close(idx, sentinel_idx)


def test_runtime_selection_dense_and_streaming_match_indices():
    from src.networks.dsa_2d import DSA2DMLAAttention, DSA2DMLAConfig

    torch.manual_seed(3)
    dense_cfg = DSA2DMLAConfig(
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
        indexer_mode="dense",
    )
    streaming_cfg = DSA2DMLAConfig(
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
        indexer_mode="streaming",
    )
    dense_mod = DSA2DMLAAttention(dense_cfg).float()
    streaming_mod = DSA2DMLAAttention(streaming_cfg).float()
    streaming_mod.load_state_dict(dense_mod.state_dict())
    x = torch.randn(1, dense_cfg.dim, 2, 2, dtype=torch.float32)

    dense_scores, dense_idx = dense_mod.build_indexer_selection(x)
    streaming_scores, streaming_idx = streaming_mod.build_indexer_selection(x)

    torch.testing.assert_close(streaming_scores, dense_scores)
    torch.testing.assert_close(streaming_idx, dense_idx)


def test_forward_uses_runtime_selection_path(monkeypatch):
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
        indexer_mode="streaming",
    )
    mod = DSA2DMLAAttention(cfg).float()
    x = torch.randn(1, cfg.dim, 2, 2, dtype=torch.float32)
    sentinel_idx = torch.tensor([[[3, 1, 0]]], dtype=torch.int64).expand(1, 4, 3).clone()
    sentinel_out = torch.randn_like(x)

    def _logits_fail(*args, **kwargs):
        raise AssertionError("forward should not call build_indexer_logits")

    def _selection(*args, **kwargs):
        return torch.zeros(1, 4, 3, dtype=torch.float32), sentinel_idx

    def _forward_sparse(_x, idx):
        torch.testing.assert_close(idx, sentinel_idx)
        return sentinel_out

    monkeypatch.setattr(mod, "build_indexer_logits", _logits_fail)
    monkeypatch.setattr(mod, "build_indexer_selection", _selection)
    monkeypatch.setattr(mod, "forward_sparse_from_indices", _forward_sparse)

    out = mod(x)

    torch.testing.assert_close(out, sentinel_out)


def test_streaming_reference_matches_dense_reference_path():
    from src.networks.dsa_2d import DSA2DMLAAttention, DSA2DMLAConfig
    from tests.helpers import dsa_reference

    torch.manual_seed(0)
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
    x = torch.randn(1, cfg.dim, 2, 2, dtype=torch.float32)

    dense_logits, dense_idx = dsa_reference.indexer_logits_reference(mod, x)
    dense_top_scores = torch.gather(dense_logits, dim=-1, index=dense_idx)
    stream_scores, stream_idx = dsa_reference.streaming_indexer_reference(mod, x, block_s=2)

    torch.testing.assert_close(stream_scores, dense_top_scores)
    torch.testing.assert_close(stream_idx, dense_idx)


def test_streaming_reference_is_block_size_invariant():
    from src.networks.dsa_2d import DSA2DMLAAttention, DSA2DMLAConfig
    from tests.helpers import dsa_reference

    torch.manual_seed(1)
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
    x = torch.randn(1, cfg.dim, 2, 2, dtype=torch.float32)

    scores_block_1, idx_block_1 = dsa_reference.streaming_indexer_reference(mod, x, block_s=1)
    scores_block_3, idx_block_3 = dsa_reference.streaming_indexer_reference(mod, x, block_s=3)

    torch.testing.assert_close(scores_block_1, scores_block_3)
    torch.testing.assert_close(idx_block_1, idx_block_3)


def test_streaming_reference_does_not_call_production_helper(monkeypatch):
    from src.networks.dsa_2d import DSA2DMLAAttention, DSA2DMLAConfig
    from src.ops import dsa_indexer
    from tests.helpers import dsa_reference

    torch.manual_seed(2)
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
    x = torch.randn(1, cfg.dim, 2, 2, dtype=torch.float32)

    def _boom(*args, **kwargs):
        raise AssertionError("production streaming helper should not be called by the reference helper")

    monkeypatch.setattr(dsa_indexer, "streaming_weighted_relu_topk", _boom)

    scores, idx = dsa_reference.streaming_indexer_reference(mod, x, block_s=2)

    assert scores.shape == (1, 4, cfg.index_topk)
    assert idx.shape == (1, 4, cfg.index_topk)


def test_build_indexer_logits_supports_bfloat16_module_dtype():
    from src.networks.dsa_2d import DSA2DMLAAttention, DSA2DMLAConfig

    torch.manual_seed(0)
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
    mod = DSA2DMLAAttention(cfg).to(dtype=torch.bfloat16)
    x = torch.randn(1, cfg.dim, 2, 2, dtype=torch.bfloat16)

    logits, idx = mod.build_indexer_logits(x)

    assert logits.dtype == torch.float32
    assert idx.dtype == torch.int64
