
import torch

from src.networks.dsa_2d import DSA2DMLAAttention, DSA2DMLAConfig
from tests.helpers import dsa_reference


def test_dsa_test_scaffold_exists():
    from tests.helpers import dsa_reference

    assert hasattr(dsa_reference, "__file__")


def make_small_cfg(
    index_topk: int = 8,
    *,
    sparse_backend: str = "reference",
    n_kv_heads: int = 2,
) -> DSA2DMLAConfig:
    return DSA2DMLAConfig(
        dim=32,
        n_heads=4,
        n_kv_heads=n_kv_heads,
        q_lora_rank=16,
        kv_lora_rank=12,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=index_topk,
        sparse_backend=sparse_backend,
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


def test_mla_runtime_builder_returns_packed_q_and_kv():
    mod = DSA2DMLAAttention(make_small_cfg(index_topk=2, n_kv_heads=1)).float()
    x = torch.randn(1, mod.dim, 2, 2)

    runtime = mod._dense_mla_runtime(x)

    assert set(runtime.keys()) >= {"q", "kv", "height", "width", "d_qk", "d_v"}
    assert runtime["height"] == 2
    assert runtime["width"] == 2
    assert runtime["d_qk"] == mod.kv_lora_rank + mod.qk_rope_head_dim
    assert runtime["d_v"] == mod.kv_lora_rank
    assert runtime["q"].shape == (1, mod.n_heads, 4, runtime["d_qk"])
    assert runtime["kv"].shape[:3] == (1, mod.n_kv_heads, 4)
    assert runtime["kv"].shape[-1] == runtime["d_qk"]


def test_mla_runtime_builder_uses_absorbed_query_contract():
    mod = DSA2DMLAAttention(make_small_cfg(index_topk=2, n_kv_heads=1)).float()
    x = torch.randn(1, mod.dim, 2, 2)

    q_old, _, _, _, _ = mod._dense_mla_qkv(x)
    runtime = mod._dense_mla_runtime(x)

    assert runtime["q"].shape[-1] != q_old.shape[-1]
    assert runtime["q"].shape[-1] == mod.kv_lora_rank + mod.qk_rope_head_dim


def test_flashmla_backend_falls_back_to_reference_on_cpu(monkeypatch):
    mod = DSA2DMLAAttention(
        make_small_cfg(index_topk=2, sparse_backend="flashmla", n_kv_heads=1)
    ).float()
    x = torch.randn(1, mod.dim, 2, 2)
    idx = torch.tensor([[[0, 1]]], dtype=torch.int64).expand(1, 4, 2).clone()

    called = {"reference": False}

    def _reference(*args, **kwargs):
        called["reference"] = True
        return torch.randn(1, mod.n_heads, 4, mod.kv_lora_rank)

    monkeypatch.setattr("src.networks.dsa_2d.streaming_sparse_mla_reference_from_runtime", _reference)
    mod.forward_sparse_from_indices(x, idx)

    assert called["reference"]


def test_flashmla_backend_rejects_non_mqa_kernel_path(monkeypatch):
    mod = DSA2DMLAAttention(
        make_small_cfg(index_topk=2, sparse_backend="flashmla", n_kv_heads=2)
    ).float()
    x = torch.randn(1, mod.dim, 2, 2)
    idx = torch.tensor([[[0, 1]]], dtype=torch.int64).expand(1, 4, 2).clone()

    called = {"flash": False, "reference": False}

    def _flash(*args, **kwargs):
        called["flash"] = True
        return torch.randn(1, mod.n_heads, 4, mod.kv_lora_rank)

    def _reference(*args, **kwargs):
        called["reference"] = True
        return torch.randn(1, mod.n_heads, 4, mod.kv_lora_rank)

    monkeypatch.setattr(
        "src.networks.dsa_2d.flashmla_sparse_mla_forward",
        _flash,
        raising=False,
    )
    monkeypatch.setattr("src.networks.dsa_2d.streaming_sparse_mla_reference_from_runtime", _reference)
    mod.forward_sparse_from_indices(x, idx)

    assert not called["flash"]
    assert called["reference"]


def test_auto_backend_uses_flashmla_when_supported_and_grad_disabled(monkeypatch):
    mod = DSA2DMLAAttention(
        make_small_cfg(index_topk=2, sparse_backend="auto", n_kv_heads=1)
    ).float()
    x = torch.randn(1, mod.dim, 2, 2)
    idx = torch.tensor([[[0, 1]]], dtype=torch.int64).expand(1, 4, 2).clone()

    called = {"flash": False, "reference": False}

    def _flash(*args, **kwargs):
        called["flash"] = True
        return torch.randn(1, mod.n_heads, 4, mod.kv_lora_rank)

    def _reference(*args, **kwargs):
        called["reference"] = True
        return torch.randn(1, mod.n_heads, 4, mod.kv_lora_rank)

    monkeypatch.setattr("src.networks.dsa_2d.flashmla_is_supported", lambda **kwargs: True)
    monkeypatch.setattr("src.networks.dsa_2d.flashmla_sparse_mla_forward", _flash)
    monkeypatch.setattr("src.networks.dsa_2d.streaming_sparse_mla_reference_from_runtime", _reference)

    with torch.inference_mode():
        mod.forward_sparse_from_indices(x, idx)

    assert called["flash"]
    assert not called["reference"]


def test_auto_backend_falls_back_to_reference_when_grad_enabled(monkeypatch):
    mod = DSA2DMLAAttention(
        make_small_cfg(index_topk=2, sparse_backend="auto", n_kv_heads=1)
    ).float()
    x = torch.randn(1, mod.dim, 2, 2)
    idx = torch.tensor([[[0, 1]]], dtype=torch.int64).expand(1, 4, 2).clone()

    called = {"flash": False, "reference": False}

    def _flash(*args, **kwargs):
        called["flash"] = True
        return torch.randn(1, mod.n_heads, 4, mod.kv_lora_rank)

    def _reference(*args, **kwargs):
        called["reference"] = True
        return torch.randn(1, mod.n_heads, 4, mod.kv_lora_rank)

    monkeypatch.setattr("src.networks.dsa_2d.flashmla_is_supported", lambda **kwargs: True)
    monkeypatch.setattr("src.networks.dsa_2d.flashmla_sparse_mla_forward", _flash)
    monkeypatch.setattr("src.networks.dsa_2d.streaming_sparse_mla_reference_from_runtime", _reference)

    mod.forward_sparse_from_indices(x, idx)

    assert not called["flash"]
    assert called["reference"]


def test_flashmla_support_check_rejects_cpu():
    from src.ops.dsa_flashmla import flashmla_is_supported

    assert not flashmla_is_supported(device=torch.device("cpu"), n_kv_heads=1)


def test_flashmla_support_check_rejects_non_mqa():
    from src.ops.dsa_flashmla import flashmla_is_supported

    assert not flashmla_is_supported(
        device=torch.device("cuda"),
        n_kv_heads=2,
        sm=(9, 0),
    )


def _make_small_flashmla_mqa_case():
    batch = 1
    h_kv = 1
    gqa_group_size = 4
    h_q = h_kv * gqa_group_size
    query_tokens = 4
    source_tokens = 5
    d_qk = 8
    d_v = 4
    q = torch.randn(batch, h_q, query_tokens, d_qk, dtype=torch.float32)
    kv = torch.randn(batch, h_kv, source_tokens, d_qk, dtype=torch.float32)
    runtime = {
        "q": q,
        "kv": kv,
        "height": 2,
        "width": 2,
        "d_qk": d_qk,
        "d_v": d_v,
        "kv_layout": "latent_then_rope",
    }
    idx = torch.tensor(
        [[[0, 3, 1], [4, 1, 0], [2, 2, 1], [3, 0, 4]]],
        dtype=torch.int64,
    )
    return runtime, idx, gqa_group_size


def test_flashmla_adapter_preserves_output_shape_small_case():
    from src.ops.dsa_flashmla import flashmla_sparse_mla_forward

    runtime, idx, g = _make_small_flashmla_mqa_case()
    out = flashmla_sparse_mla_forward(
        runtime,
        idx,
        gqa_group_size=g,
        softmax_scale=runtime["d_qk"] ** -0.5,
        force_reference_kernel=True,
    )

    assert out.shape == (1, runtime["q"].shape[1], runtime["q"].shape[2], runtime["d_v"])


def test_flashmla_adapter_matches_reference_small_case():
    from src.ops.dsa_flashmla import flashmla_sparse_mla_forward
    from src.ops.dsa_sparse_mla import streaming_sparse_mla_reference_from_runtime

    runtime, idx, g = _make_small_flashmla_mqa_case()
    ref = streaming_sparse_mla_reference_from_runtime(
        runtime,
        idx,
        gqa_group_size=g,
        softmax_scale=runtime["d_qk"] ** -0.5,
    )
    out = flashmla_sparse_mla_forward(
        runtime,
        idx,
        gqa_group_size=g,
        softmax_scale=runtime["d_qk"] ** -0.5,
        force_reference_kernel=True,
    )

    torch.testing.assert_close(out, ref)


def test_flashmla_adapter_packs_runtime_for_kernel_call(monkeypatch):
    from src.ops.dsa_flashmla import flashmla_sparse_mla_forward

    runtime, idx, g = _make_small_flashmla_mqa_case()
    called = {}

    def _kernel(q, kv, indices, sm_scale, d_v, attn_sink=None, topk_length=None):
        called["q_shape"] = tuple(q.shape)
        called["kv_shape"] = tuple(kv.shape)
        called["indices_shape"] = tuple(indices.shape)
        called["indices_dtype"] = indices.dtype
        called["sm_scale"] = sm_scale
        called["d_v"] = d_v
        out = torch.zeros(q.shape[0], q.shape[1], d_v, dtype=q.dtype)
        max_logits = torch.zeros(q.shape[0], q.shape[1], dtype=torch.float32)
        lse = torch.zeros(q.shape[0], q.shape[1], dtype=torch.float32)
        return out, max_logits, lse

    monkeypatch.setattr("src.ops.dsa_flashmla.flashmla_import_or_none", lambda: _kernel)

    out = flashmla_sparse_mla_forward(
        runtime,
        idx,
        gqa_group_size=g,
        softmax_scale=runtime["d_qk"] ** -0.5,
        force_reference_kernel=False,
    )

    assert called["q_shape"] == (runtime["q"].shape[2], runtime["q"].shape[1], runtime["d_qk"])
    assert called["kv_shape"] == (runtime["kv"].shape[2], runtime["kv"].shape[1], runtime["kv"].shape[3])
    assert called["indices_shape"] == (idx.shape[1], runtime["kv"].shape[1], idx.shape[2])
    assert called["indices_dtype"] == torch.int32
    assert called["sm_scale"] == runtime["d_qk"] ** -0.5
    assert called["d_v"] == runtime["d_v"]
    assert out.shape == (1, runtime["q"].shape[1], runtime["q"].shape[2], runtime["d_v"])


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


def test_packed_sparse_mla_reference_matches_old_small_reference():
    from src.ops.dsa_sparse_mla import streaming_sparse_mla_reference_from_runtime

    mod = DSA2DMLAAttention(make_small_cfg(index_topk=3, n_kv_heads=2)).float()
    x = torch.randn(1, mod.dim, 2, 2)
    idx = torch.tensor([[[0, 1, 2]]], dtype=torch.int64).expand(1, 4, 3).clone()

    runtime = mod._dense_mla_runtime(x)
    q, k, v, _, _ = mod._dense_mla_qkv(x)
    ref = dsa_reference.gather_sparse_mla_reference(
        q,
        k,
        v,
        idx,
        gqa_group_size=mod.gqa_group_size,
        softmax_scale=mod.softmax_scale,
    )
    latent = streaming_sparse_mla_reference_from_runtime(
        runtime,
        idx,
        gqa_group_size=mod.gqa_group_size,
        softmax_scale=mod.softmax_scale,
        query_block_size=2,
        selected_block_size=2,
    )
    out = mod._apply_uv_output_projection(latent)

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


def test_forward_sparse_from_indices_uses_packed_runtime_helper(monkeypatch):
    mod = DSA2DMLAAttention(make_small_cfg(index_topk=3)).float()
    x = torch.randn(1, mod.dim, 2, 2)
    idx = torch.tensor([[[0, 1, 2]]], dtype=torch.int64).expand(1, 4, 3).clone()
    sentinel = torch.randn(1, mod.n_heads, 4, mod.kv_lora_rank)

    def _packed_runtime(*args, **kwargs):
        return sentinel

    def _fail(*args, **kwargs):
        raise AssertionError("old unpacked sparse helper must not be used")

    monkeypatch.setattr("src.networks.dsa_2d.streaming_sparse_mla_reference", _fail)
    monkeypatch.setattr(
        "src.networks.dsa_2d.streaming_sparse_mla_reference_from_runtime",
        _packed_runtime,
        raising=False,
    )
    out = mod.forward_sparse_from_indices(x, idx)
    assert out.shape == x.shape


def test_forward_sparse_from_indices_passes_packed_runtime_to_flashmla(monkeypatch):
    mod = DSA2DMLAAttention(
        make_small_cfg(index_topk=3, sparse_backend="flashmla", n_kv_heads=1)
    ).float()
    x = torch.randn(1, mod.dim, 2, 2)
    idx = torch.tensor([[[0, 1, 2]]], dtype=torch.int64).expand(1, 4, 3).clone()
    sentinel = torch.randn(1, mod.n_heads, 4, mod.kv_lora_rank)
    seen = {}

    def _flash(runtime, idx_arg, **kwargs):
        seen["runtime_keys"] = set(runtime.keys())
        seen["idx_shape"] = tuple(idx_arg.shape)
        return sentinel

    monkeypatch.setattr("src.networks.dsa_2d.flashmla_is_supported", lambda **kwargs: True)
    monkeypatch.setattr("src.networks.dsa_2d.flashmla_sparse_mla_forward", _flash)

    with torch.inference_mode():
        out = mod.forward_sparse_from_indices(x, idx)

    assert out.shape == x.shape
    assert seen["runtime_keys"] >= {"q", "kv", "height", "width", "d_qk", "d_v"}
    assert seen["idx_shape"] == tuple(idx.shape)


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
