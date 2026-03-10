
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
