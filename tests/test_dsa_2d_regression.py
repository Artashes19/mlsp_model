
import torch

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
