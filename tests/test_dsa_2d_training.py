
import torch

from src.networks.dsa_2d import DSA2DMLAAttention, DSA2DMLAConfig


def test_dsa_test_scaffold_exists():
    from tests.helpers import dsa_reference

    assert hasattr(dsa_reference, "__file__")


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


def test_dense_teacher_distribution_is_normalized():
    torch.manual_seed(0)
    cfg = make_small_cfg()
    mod = DSA2DMLAAttention(cfg).float()
    probs = mod.build_dense_teacher_distribution(torch.randn(1, cfg.dim, 4, 4, dtype=torch.float32))

    torch.testing.assert_close(probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1)))


def test_indexer_warmup_detaches_main_model_inputs():
    cfg = make_small_cfg()
    mod = DSA2DMLAAttention(cfg).float()

    mod.assert_warmup_detach_contract()


def test_indexer_alignment_kl_loss_is_scalar_and_finite():
    cfg = make_small_cfg()
    mod = DSA2DMLAAttention(cfg).float()
    logits = torch.randn(1, 4, 4, dtype=torch.float32)
    teacher = torch.softmax(torch.randn(1, 4, 4, dtype=torch.float32), dim=-1)

    loss = mod.indexer_alignment_kl_loss(logits, teacher)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
