import pytest
import torch

from src.networks.dsa_2d import DSA2DMLAAttention, DSA2DMLAConfig
from tests.helpers import dsa_reference


def make_small_cfg() -> DSA2DMLAConfig:
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
        index_topk=8,
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
