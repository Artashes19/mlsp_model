
import torch

from src.networks.dsa_2d import DSA2DMLAAttention, DSA2DMLAConfig
from tests.helpers import dsa_reference


def test_dsa_test_scaffold_exists():
    from tests.helpers import dsa_reference

    assert hasattr(dsa_reference, "__file__")


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


def test_sparse_mla_gather_matches_reference_order():
    from src.ops import dsa_sparse_mla

    idx = torch.tensor([[[0, 3, 1]]], dtype=torch.int64)
    k = torch.randn(1, 2, 4, 16)

    ref = dsa_reference.gather_tokens_reference(k, idx)
    out = dsa_sparse_mla.gather_sparse_mla_tokens(k, idx)

    torch.testing.assert_close(out, ref)


def test_sparse_mla_matches_dense_when_topk_equals_t():
    torch.manual_seed(0)
    cfg = make_small_cfg(index_topk=16)
    mod = DSA2DMLAAttention(cfg).float()
    x = torch.randn(1, cfg.dim, 4, 4, dtype=torch.float32, requires_grad=True)

    dense = mod.forward_dense_reference(x)
    sparse = mod.forward_sparse_with_forced_topk(x, topk_equals_t=True)

    torch.testing.assert_close(sparse, dense)
