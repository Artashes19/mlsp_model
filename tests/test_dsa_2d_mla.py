import pytest
import torch

from src.networks.dsa_2d import DSA2DMLAAttention, DSA2DMLAConfig
from tests.helpers import dsa_reference


def make_small_cfg(index_topk: int = 8) -> DSA2DMLAConfig:
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


def test_mla_config_rejects_non_gqa_shape():
    with pytest.raises(ValueError):
        DSA2DMLAConfig(dim=512, n_heads=6, n_kv_heads=4)


def test_mla_config_rejects_invalid_mla_rope_dim():
    with pytest.raises(ValueError):
        DSA2DMLAConfig(dim=512, n_heads=8, n_kv_heads=4, qk_rope_head_dim=66)


def test_mla_config_rejects_invalid_index_rope_dim():
    with pytest.raises(ValueError):
        DSA2DMLAConfig(dim=512, n_heads=8, n_kv_heads=4, index_head_dim=130)


def test_mla_config_rejects_non_power_of_two_index_head_dim():
    with pytest.raises(ValueError):
        DSA2DMLAConfig(dim=512, n_heads=8, n_kv_heads=4, index_head_dim=40)


def test_mla_projection_splits_have_expected_sizes():
    cfg = DSA2DMLAConfig(
        dim=1536,
        n_heads=16,
        n_kv_heads=4,
        q_lora_rank=512,
        kv_lora_rank=256,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        index_n_heads=4,
        index_head_dim=128,
        index_topk=64,
    )
    mod = DSA2DMLAAttention(cfg)
    assert mod.qk_head_dim == 192
    assert mod.kv_a_out_dim == 320
    assert mod.kv_proj_out_dim == 1024
    assert mod.attn_out_dim == 2048
    assert mod.index_rope_head_dim == 64
    assert mod.index_nope_head_dim == 64


def test_indexer_projection_shapes_follow_deepseek_style_contract():
    cfg = make_small_cfg()
    mod = DSA2DMLAAttention(cfg)

    assert mod.index_wq_b.weight.shape == (cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank)
    assert mod.index_wk.weight.shape == (cfg.index_head_dim, cfg.dim)
    assert mod.index_weights_proj.weight.shape == (cfg.index_n_heads, cfg.dim)
    assert mod.index_k_norm.normalized_shape == (cfg.index_head_dim,)


def test_dense_mla_forward_matches_reference_small_case():
    torch.manual_seed(0)
    cfg = make_small_cfg()
    mod = DSA2DMLAAttention(cfg).float()
    x = torch.randn(2, cfg.dim, 4, 4, dtype=torch.float32)

    ref = dsa_reference.dense_mla_reference(mod, x)
    out = mod.forward_dense_reference(x)

    torch.testing.assert_close(out, ref)


def test_dense_mla_backward_matches_reference_small_case():
    torch.manual_seed(0)
    cfg = make_small_cfg()
    mod = DSA2DMLAAttention(cfg).float()
    x = torch.randn(1, cfg.dim, 4, 4, dtype=torch.float32, requires_grad=True)

    dsa_reference.compare_dense_mla_backward(mod, x)


def test_dsa_2d_mla_attention_round_trips_image_shape():
    torch.manual_seed(0)
    cfg = make_small_cfg()
    mod = DSA2DMLAAttention(cfg).float()
    x = torch.randn(2, cfg.dim, 8, 8, dtype=torch.float32)

    out = mod(x)

    assert out.shape == x.shape


def test_dsa_sparse_and_dense_paths_share_projection_contract():
    torch.manual_seed(0)
    cfg = make_small_cfg(index_topk=16)
    mod = DSA2DMLAAttention(cfg).float()
    x = torch.randn(1, cfg.dim, 4, 4, dtype=torch.float32)

    integrated = mod(x)
    dense = mod.forward_dense_reference(x)
    sparse = mod.forward_sparse_with_forced_topk(x, topk_equals_t=True)

    assert integrated.shape == x.shape
    assert dense.shape == x.shape
    assert sparse.shape == x.shape
    assert mod.proj.weight.shape == (cfg.dim, mod.attn_out_dim)
