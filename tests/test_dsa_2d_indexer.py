
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
