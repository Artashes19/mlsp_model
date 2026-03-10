
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


def test_sparse_runtime_does_not_call_gather_sparse_mla_tokens(monkeypatch):
    mod = DSA2DMLAAttention(make_small_cfg(index_topk=3)).float()
    x = torch.randn(1, mod.dim, 2, 2)
    idx = torch.tensor([[[0, 1, 2]]], dtype=torch.int64).expand(1, 4, 3).clone()

    def _fail(*args, **kwargs):
        raise AssertionError("old gather helper must not be used")

    monkeypatch.setattr("src.ops.dsa_sparse_mla.gather_sparse_mla_tokens", _fail)
    mod.forward_sparse_from_indices(x, idx)


def test_sparse_runtime_does_not_repeat_interleave_kv(monkeypatch):
    mod = DSA2DMLAAttention(make_small_cfg(index_topk=3)).float()
    x = torch.randn(1, mod.dim, 2, 2)
    idx = torch.tensor([[[0, 1, 2]]], dtype=torch.int64).expand(1, 4, 3).clone()

    calls = []
    orig = torch.Tensor.repeat_interleave

    def _wrapped(self, *args, **kwargs):
        calls.append(self.shape)
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "repeat_interleave", _wrapped)
    mod.forward_sparse_from_indices(x, idx)
    assert not calls


def _make_small_sparse_case(*, gqa_group_size: int = 2):
    batch = 1
    h_kv = 2
    topk = 3
    query_tokens = 4
    source_tokens = 5
    dim = 8
    h_q = h_kv * gqa_group_size
    q = torch.randn(batch, h_q, query_tokens, dim, dtype=torch.float32)
    k = torch.randn(batch, h_kv, source_tokens, dim, dtype=torch.float32)
    v = torch.randn(batch, h_kv, source_tokens, dim, dtype=torch.float32)
    idx = torch.tensor(
        [[[0, 3, 1], [4, 1, 0], [2, 2, 1], [3, 0, 4]]],
        dtype=torch.int64,
    )
    return q, k, v, idx, gqa_group_size


def test_streaming_sparse_mla_matches_gather_reference_small_case():
    from src.ops.dsa_sparse_mla import streaming_sparse_mla_reference

    q, k, v, idx, g = _make_small_sparse_case()
    ref = dsa_reference.gather_sparse_mla_reference(
        q,
        k,
        v,
        idx,
        gqa_group_size=g,
        softmax_scale=q.shape[-1] ** -0.5,
    )
    out = streaming_sparse_mla_reference(
        q,
        k,
        v,
        idx,
        gqa_group_size=g,
        softmax_scale=q.shape[-1] ** -0.5,
    )
    torch.testing.assert_close(out, ref)


def test_streaming_sparse_mla_matches_reference_with_explicit_block_sizes():
    from src.ops.dsa_sparse_mla import streaming_sparse_mla_reference

    q, k, v, idx, g = _make_small_sparse_case()
    ref = dsa_reference.gather_sparse_mla_reference(
        q,
        k,
        v,
        idx,
        gqa_group_size=g,
        softmax_scale=q.shape[-1] ** -0.5,
    )
    out = streaming_sparse_mla_reference(
        q,
        k,
        v,
        idx,
        gqa_group_size=g,
        softmax_scale=q.shape[-1] ** -0.5,
        query_block_size=2,
        selected_block_size=2,
    )
    torch.testing.assert_close(out, ref)


def test_streaming_sparse_mla_respects_gqa_head_mapping():
    from src.ops.dsa_sparse_mla import streaming_sparse_mla_reference

    q, k, v, idx, g = _make_small_sparse_case(gqa_group_size=3)
    out = streaming_sparse_mla_reference(
        q,
        k,
        v,
        idx,
        gqa_group_size=g,
        softmax_scale=q.shape[-1] ** -0.5,
    )
    ref = dsa_reference.slow_per_head_kv_mapping_reference(
        q,
        k,
        v,
        idx,
        gqa_group_size=g,
        softmax_scale=q.shape[-1] ** -0.5,
    )
    torch.testing.assert_close(out, ref)


def test_forward_sparse_from_indices_uses_streaming_sparse_helper(monkeypatch):
    mod = DSA2DMLAAttention(make_small_cfg(index_topk=3)).float()
    x = torch.randn(1, mod.dim, 2, 2)
    idx = torch.tensor([[[0, 1, 2]]], dtype=torch.int64).expand(1, 4, 3).clone()
    sentinel = torch.randn(1, mod.n_heads, 4, mod.v_head_dim)

    def _streaming(*args, **kwargs):
        return sentinel

    monkeypatch.setattr("src.networks.dsa_2d.streaming_sparse_mla_reference", _streaming)
    out = mod.forward_sparse_from_indices(x, idx)
    assert out.shape == x.shape


def test_forward_sparse_from_indices_matches_old_small_reference():
    mod = DSA2DMLAAttention(make_small_cfg(index_topk=3)).float()
    x = torch.randn(1, mod.dim, 2, 2)
    idx = torch.tensor([[[0, 1, 2]]], dtype=torch.int64).expand(1, 4, 3).clone()

    out = mod.forward_sparse_from_indices(x, idx)
    ref = dsa_reference.sparse_mla_reference_from_indices(mod, x, idx)
    torch.testing.assert_close(out, ref)


def test_sparse_mla_matches_dense_when_topk_equals_t():
    torch.manual_seed(0)
    cfg = make_small_cfg(index_topk=16)
    mod = DSA2DMLAAttention(cfg).float()
    x = torch.randn(1, cfg.dim, 4, 4, dtype=torch.float32, requires_grad=True)

    dense = mod.forward_dense_reference(x)
    sparse = mod.forward_sparse_with_forced_topk(x, topk_equals_t=True)

    torch.testing.assert_close(sparse, dense)


def test_sparse_mla_backward_matches_dense_when_topk_equals_t():
    cfg = make_small_cfg(index_topk=16)
    dsa_reference.compare_sparse_and_dense_backward(cfg, H=4, W=4)
