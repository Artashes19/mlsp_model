import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

def _naive_gqa_selection(q, k, v, top_idx, patch_starts, pp, P, W_spatial, G):
    """Reference: gather + SDPA for GQA selection attention."""
    B, h_q, T, d = q.shape
    h_kv = h_q // G
    top_n = top_idx.shape[2]
    H_spatial = T // W_spatial
    nH, nW = H_spatial // P, W_spatial // P
    n_patches = nH * nW

    k_2d = k.view(B, h_kv, H_spatial, W_spatial, d)
    k_patches = k_2d.view(B, h_kv, nH, P, nW, P, d).permute(0, 1, 2, 4, 3, 5, 6).contiguous()
    k_patches = k_patches.view(B, h_kv, n_patches, pp, d)
    v_2d = v.view(B, h_kv, H_spatial, W_spatial, d)
    v_patches = v_2d.view(B, h_kv, nH, P, nW, P, d).permute(0, 1, 2, 4, 3, 5, 6).contiguous()
    v_patches = v_patches.view(B, h_kv, n_patches, pp, d)

    idx = top_idx.long()[:, :, :, None, None].expand(B, h_kv, top_n, pp, d)
    k_sel = k_patches.gather(2, idx).reshape(B, h_kv, top_n * pp, d)
    v_sel = v_patches.gather(2, idx).reshape(B, h_kv, top_n * pp, d)

    return F.scaled_dot_product_attention(q, k_sel, v_sel, enable_gqa=True)


class TestGQASelectionForward:
    @pytest.mark.parametrize("h_q,h_kv,d", [(8, 2, 24), (4, 1, 12), (8, 2, 48)])
    def test_fwd_matches_naive(self, h_q, h_kv, d):
        from src.ops.selection_attention_2d_gqa import SelectionAttn2DGQA, make_patch_starts
        B, H, W, P = 1, 16, 16, 4
        T, pp, top_n, G = H * W, P * P, 4, h_q // h_kv
        n_patches = (H // P) * (W // P)

        torch.manual_seed(42)
        q = torch.randn(B, h_q, T, d, device="cuda")
        k = torch.randn(B, h_kv, T, d, device="cuda")
        v = torch.randn(B, h_kv, T, d, device="cuda")
        top_idx = torch.randint(0, n_patches, (B, h_kv, top_n), device="cuda", dtype=torch.int32)
        patch_starts = make_patch_starts(H, W, P, q.device)
        scale = 1.0 / (d ** 0.5)

        o_triton = SelectionAttn2DGQA.apply(q, k, v, top_idx, patch_starts, pp, H, W, P, scale, G)
        o_naive = _naive_gqa_selection(q, k, v, top_idx, patch_starts, pp, P, W, G)
        assert torch.allclose(o_triton, o_naive, atol=1e-2, rtol=1e-2)

    def test_fwd_bf16(self):
        from src.ops.selection_attention_2d_gqa import SelectionAttn2DGQA, make_patch_starts
        B, h_q, h_kv, d, H, W, P = 1, 8, 2, 24, 16, 16, 4
        T, pp, top_n, G = H * W, P * P, 4, 4
        n_patches = (H // P) * (W // P)

        q = torch.randn(B, h_q, T, d, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, h_kv, T, d, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, h_kv, T, d, device="cuda", dtype=torch.bfloat16)
        top_idx = torch.randint(0, n_patches, (B, h_kv, top_n), device="cuda", dtype=torch.int32)
        patch_starts = make_patch_starts(H, W, P, q.device)
        scale = 1.0 / (d ** 0.5)

        o = SelectionAttn2DGQA.apply(q, k, v, top_idx, patch_starts, pp, H, W, P, scale, G)
        assert o.dtype == torch.bfloat16
        assert not torch.isnan(o).any()


class TestGQASelectionBackward:
    @pytest.mark.parametrize("h_q,h_kv,d", [(8, 2, 24), (4, 1, 12)])
    def test_backward_dq_matches_naive(self, h_q, h_kv, d):
        from src.ops.selection_attention_2d_gqa import SelectionAttn2DGQA, make_patch_starts
        B, H, W, P = 1, 16, 16, 4
        T, pp, top_n, G = H * W, P * P, 4, h_q // h_kv
        n_patches = (H // P) * (W // P)

        torch.manual_seed(42)
        q = torch.randn(B, h_q, T, d, device="cuda", requires_grad=True)
        k = torch.randn(B, h_kv, T, d, device="cuda")
        v = torch.randn(B, h_kv, T, d, device="cuda")
        top_idx = torch.randint(0, n_patches, (B, h_kv, top_n), device="cuda", dtype=torch.int32)
        patch_starts = make_patch_starts(H, W, P, q.device)
        scale = 1.0 / (d ** 0.5)

        o = SelectionAttn2DGQA.apply(q, k, v, top_idx, patch_starts, pp, H, W, P, scale, G)
        o.sum().backward()
        dq_triton = q.grad.clone()

        q2 = q.detach().clone().requires_grad_(True)
        o_naive = _naive_gqa_selection(q2, k, v, top_idx, patch_starts, pp, P, W, G)
        o_naive.sum().backward()
        dq_naive = q2.grad

        assert torch.allclose(dq_triton, dq_naive, atol=5e-2, rtol=5e-2)

    @pytest.mark.parametrize("h_q,h_kv,d", [(8, 2, 24), (4, 1, 12)])
    def test_backward_dk_dv_matches_naive(self, h_q, h_kv, d):
        from src.ops.selection_attention_2d_gqa import SelectionAttn2DGQA, make_patch_starts
        B, H, W, P = 1, 16, 16, 4
        T, pp, top_n, G = H * W, P * P, 4, h_q // h_kv
        n_patches = (H // P) * (W // P)

        torch.manual_seed(42)
        q = torch.randn(B, h_q, T, d, device="cuda")
        k = torch.randn(B, h_kv, T, d, device="cuda", requires_grad=True)
        v = torch.randn(B, h_kv, T, d, device="cuda", requires_grad=True)
        top_idx = torch.randint(0, n_patches, (B, h_kv, top_n), device="cuda", dtype=torch.int32)
        patch_starts = make_patch_starts(H, W, P, q.device)
        scale = 1.0 / (d ** 0.5)

        o = SelectionAttn2DGQA.apply(q, k, v, top_idx, patch_starts, pp, H, W, P, scale, G)
        o.sum().backward()
        dk_triton = k.grad.clone()
        dv_triton = v.grad.clone()

        k2 = k.detach().clone().requires_grad_(True)
        v2 = v.detach().clone().requires_grad_(True)
        o_naive = _naive_gqa_selection(q, k2, v2, top_idx, patch_starts, pp, P, W, G)
        o_naive.sum().backward()

        assert torch.allclose(dk_triton, k2.grad, atol=5e-2, rtol=5e-2)
        assert torch.allclose(dv_triton, v2.grad, atol=5e-2, rtol=5e-2)


def _naive_gqa_selection_per_query(q, k, v, block_idx, patch_starts, P, W_spatial, G, scale=None):
    """Reference per-query GQA: block_idx is [B, h_kv, T, top_n]."""
    B, h_q, T, d = q.shape
    h_kv = h_q // G
    top_n = block_idx.shape[-1]
    out = torch.empty_like(q)

    for b in range(B):
        for kv in range(h_kv):
            for t in range(T):
                flat_idx: list[int] = []
                for i in range(top_n):
                    patch = int(block_idx[b, kv, t, i].item())
                    base = int(patch_starts[patch].item())
                    for dr in range(P):
                        for dc in range(P):
                            flat_idx.append(base + dr * W_spatial + dc)
                idx_t = torch.tensor(flat_idx, device=q.device, dtype=torch.long)
                k_t = k[b, kv, idx_t, :].unsqueeze(0)
                v_t = v[b, kv, idx_t, :].unsqueeze(0)
                for g in range(G):
                    qh = kv * G + g
                    q_t = q[b, qh, t : t + 1, :].unsqueeze(0)
                    o_t = F.scaled_dot_product_attention(q_t, k_t, v_t, scale=scale)
                    out[b, qh, t, :] = o_t[0, 0, :]
    return out


class TestGQAPerQuerySelectionForward:
    @pytest.mark.per_query_parity
    def test_packed_per_query_forward_matches_unpacked_gqa(self):
        """Packed per-query forward should match unpacked forward in GQA mode."""
        from src.networks.txunet import NSA2DAttention

        B, h_q, h_kv, d = 1, 4, 2, 8
        H, W, P = 4, 4, 2
        T = H * W
        top_n = 2
        C = h_q * d
        G = h_q // h_kv
        n_patches = (H // P) * (W // P)

        torch.manual_seed(9191)
        attn_unpacked = NSA2DAttention(
            dim=C,
            num_heads=h_q,
            patch_size=P,
            top_n=top_n,
            window_size=2,
            rope_enabled=False,
            gqa_group_size=G,
            selection_forward_mode="unpacked",
        ).to(device="cuda", dtype=torch.float32)
        attn_packed = NSA2DAttention(
            dim=C,
            num_heads=h_q,
            patch_size=P,
            top_n=top_n,
            window_size=2,
            rope_enabled=False,
            gqa_group_size=G,
            selection_forward_mode="packed",
        ).to(device="cuda", dtype=torch.float32)

        q = torch.randn(B, h_q, T, d, device="cuda", dtype=torch.float32)
        k = torch.randn(B, h_kv, T, d, device="cuda", dtype=torch.float32)
        v = torch.randn(B, h_kv, T, d, device="cuda", dtype=torch.float32)
        block_idx = torch.randint(0, n_patches, (B, h_kv, T, top_n), device="cuda", dtype=torch.int32)

        with torch.no_grad():
            o_unpacked = attn_unpacked._selection_from_block_idx(q, k, v, block_idx, H, W)
            o_packed = attn_packed._selection_from_block_idx(q, k, v, block_idx, H, W)

        torch.testing.assert_close(o_packed, o_unpacked, atol=1e-2, rtol=1e-2)

    @pytest.mark.per_query_parity
    def test_per_query_forward_matches_naive_gqa(self):
        """Per-query GQA forward should match naive reference."""
        from src.ops.selection_attention_2d_per_query import SelectionAttn2DPerQuery, make_patch_starts

        B, h_q, h_kv, d = 1, 4, 2, 8
        H, W, P = 4, 4, 2
        T = H * W
        top_n = 2
        pp = P * P
        G = h_q // h_kv
        n_patches = (H // P) * (W // P)
        scale = 1.0 / d**0.5

        torch.manual_seed(17)
        q = torch.randn(B, h_q, T, d, device="cuda", dtype=torch.float32)
        k = torch.randn(B, h_kv, T, d, device="cuda", dtype=torch.float32)
        v = torch.randn(B, h_kv, T, d, device="cuda", dtype=torch.float32)
        block_idx = torch.randint(0, n_patches, (B, h_kv, T, top_n), device="cuda", dtype=torch.int32)
        patch_starts = make_patch_starts(H, W, P, q.device)

        o_triton = SelectionAttn2DPerQuery.apply(q, k, v, block_idx, patch_starts, pp, H, W, P, scale, G)
        o_naive = _naive_gqa_selection_per_query(q, k, v, block_idx, patch_starts, P, W, G, scale=scale)
        torch.testing.assert_close(o_triton, o_naive, atol=1e-2, rtol=1e-2)

    @pytest.mark.per_query_parity
    @pytest.mark.parametrize("h_q,h_kv", [(4, 2), (4, 1)])
    def test_per_query_forward_matches_naive_gqa_grouped_head_regimes(self, h_q, h_kv):
        """Per-query forward stays correct for explicit grouped-head regimes."""
        from src.ops.selection_attention_2d_per_query import SelectionAttn2DPerQuery, make_patch_starts

        B, d = 1, 8
        H, W, P = 4, 4, 2
        T = H * W
        top_n = 2
        pp = P * P
        G = h_q // h_kv
        n_patches = (H // P) * (W // P)
        scale = 1.0 / d**0.5

        torch.manual_seed(3030 + G)
        q = torch.randn(B, h_q, T, d, device="cuda", dtype=torch.float32)
        k = torch.randn(B, h_kv, T, d, device="cuda", dtype=torch.float32)
        v = torch.randn(B, h_kv, T, d, device="cuda", dtype=torch.float32)
        block_idx = torch.randint(0, n_patches, (B, h_kv, T, top_n), device="cuda", dtype=torch.int32)
        patch_starts = make_patch_starts(H, W, P, q.device)

        o_triton = SelectionAttn2DPerQuery.apply(q, k, v, block_idx, patch_starts, pp, H, W, P, scale, G)
        o_naive = _naive_gqa_selection_per_query(q, k, v, block_idx, patch_starts, P, W, G, scale=scale)
        torch.testing.assert_close(o_triton, o_naive, atol=1e-2, rtol=1e-2)


class TestGQAPerQuerySelectionBackward:
    @pytest.mark.per_query_parity
    def test_packed_per_query_backward_dq_matches_unpacked_gqa(self):
        """Packed per-query dQ should match unpacked dQ in GQA mode."""
        from src.networks.txunet import NSA2DAttention

        B, h_q, h_kv, d = 1, 4, 2, 8
        H, W, P = 4, 4, 2
        T = H * W
        top_n = 2
        C = h_q * d
        G = h_q // h_kv
        n_patches = (H // P) * (W // P)

        torch.manual_seed(7070)
        attn_unpacked = NSA2DAttention(
            dim=C,
            num_heads=h_q,
            patch_size=P,
            top_n=top_n,
            window_size=2,
            rope_enabled=False,
            gqa_group_size=G,
            selection_forward_mode="unpacked",
            selection_dq_mode="unpacked",
        ).to(device="cuda", dtype=torch.float32)
        attn_packed = NSA2DAttention(
            dim=C,
            num_heads=h_q,
            patch_size=P,
            top_n=top_n,
            window_size=2,
            rope_enabled=False,
            gqa_group_size=G,
            selection_forward_mode="unpacked",
            selection_dq_mode="packed",
        ).to(device="cuda", dtype=torch.float32)

        q = torch.randn(B, h_q, T, d, device="cuda", dtype=torch.float32, requires_grad=True)
        k = torch.randn(B, h_kv, T, d, device="cuda", dtype=torch.float32)
        v = torch.randn(B, h_kv, T, d, device="cuda", dtype=torch.float32)
        block_idx = torch.randint(0, n_patches, (B, h_kv, T, top_n), device="cuda", dtype=torch.int32)

        o_unpacked = attn_unpacked._selection_from_block_idx(q, k, v, block_idx, H, W)
        o_unpacked.sum().backward()
        dq_unpacked = q.grad.detach().clone()

        q2 = q.detach().clone().requires_grad_(True)
        o_packed = attn_packed._selection_from_block_idx(q2, k, v, block_idx, H, W)
        o_packed.sum().backward()
        dq_packed = q2.grad.detach().clone()

        torch.testing.assert_close(dq_packed, dq_unpacked, atol=5e-2, rtol=5e-2)

    @pytest.mark.per_query_parity
    def test_packed_per_query_backward_dq_matches_naive_gqa(self):
        """Packed per-query dQ should match naive autograd dQ in GQA mode."""
        from src.networks.txunet import NSA2DAttention
        from src.ops.selection_attention_2d_per_query import make_patch_starts

        B, h_q, h_kv, d = 1, 4, 2, 8
        H, W, P = 4, 4, 2
        T = H * W
        top_n = 2
        C = h_q * d
        G = h_q // h_kv
        n_patches = (H // P) * (W // P)
        scale = 1.0 / d**0.5

        torch.manual_seed(7171)
        attn_packed = NSA2DAttention(
            dim=C,
            num_heads=h_q,
            patch_size=P,
            top_n=top_n,
            window_size=2,
            rope_enabled=False,
            gqa_group_size=G,
            selection_forward_mode="unpacked",
            selection_dq_mode="packed",
        ).to(device="cuda", dtype=torch.float32)

        q = torch.randn(B, h_q, T, d, device="cuda", dtype=torch.float32, requires_grad=True)
        k = torch.randn(B, h_kv, T, d, device="cuda", dtype=torch.float32)
        v = torch.randn(B, h_kv, T, d, device="cuda", dtype=torch.float32)
        block_idx = torch.randint(0, n_patches, (B, h_kv, T, top_n), device="cuda", dtype=torch.int32)

        o_packed = attn_packed._selection_from_block_idx(q, k, v, block_idx, H, W)
        o_packed.sum().backward()
        dq_packed = q.grad.detach().clone()

        q2 = q.detach().clone().requires_grad_(True)
        patch_starts = make_patch_starts(H, W, P, q2.device)
        o_naive = _naive_gqa_selection_per_query(q2, k, v, block_idx, patch_starts, P, W, G, scale=scale)
        o_naive.sum().backward()
        torch.testing.assert_close(dq_packed, q2.grad, atol=5e-2, rtol=5e-2)

    @pytest.mark.per_query_parity
    def test_per_query_backward_dq_matches_naive_gqa(self):
        """Per-query GQA dQ should match naive autograd dQ."""
        from src.ops.selection_attention_2d_per_query import SelectionAttn2DPerQuery, make_patch_starts

        B, h_q, h_kv, d = 1, 4, 2, 8
        H, W, P = 4, 4, 2
        T = H * W
        top_n = 2
        pp = P * P
        G = h_q // h_kv
        n_patches = (H // P) * (W // P)
        scale = 1.0 / d**0.5

        torch.manual_seed(29)
        q = torch.randn(B, h_q, T, d, device="cuda", dtype=torch.float32, requires_grad=True)
        k = torch.randn(B, h_kv, T, d, device="cuda", dtype=torch.float32)
        v = torch.randn(B, h_kv, T, d, device="cuda", dtype=torch.float32)
        block_idx = torch.randint(0, n_patches, (B, h_kv, T, top_n), device="cuda", dtype=torch.int32)
        patch_starts = make_patch_starts(H, W, P, q.device)

        o_triton = SelectionAttn2DPerQuery.apply(q, k, v, block_idx, patch_starts, pp, H, W, P, scale, G)
        o_triton.sum().backward()
        dq_triton = q.grad.detach().clone()

        q2 = q.detach().clone().requires_grad_(True)
        o_naive = _naive_gqa_selection_per_query(q2, k, v, block_idx, patch_starts, P, W, G, scale=scale)
        o_naive.sum().backward()
        torch.testing.assert_close(dq_triton, q2.grad, atol=5e-2, rtol=5e-2)

    @pytest.mark.per_query_parity
    @pytest.mark.parametrize("h_q,h_kv", [(4, 2), (4, 1)])
    def test_per_query_backward_dq_matches_naive_gqa_grouped_head_regimes(self, h_q, h_kv):
        """Per-query dQ stays correct for explicit grouped-head regimes."""
        from src.ops.selection_attention_2d_per_query import SelectionAttn2DPerQuery, make_patch_starts

        B, d = 1, 8
        H, W, P = 4, 4, 2
        T = H * W
        top_n = 2
        pp = P * P
        G = h_q // h_kv
        n_patches = (H // P) * (W // P)
        scale = 1.0 / d**0.5

        torch.manual_seed(4040 + G)
        q = torch.randn(B, h_q, T, d, device="cuda", dtype=torch.float32, requires_grad=True)
        k = torch.randn(B, h_kv, T, d, device="cuda", dtype=torch.float32)
        v = torch.randn(B, h_kv, T, d, device="cuda", dtype=torch.float32)
        block_idx = torch.randint(0, n_patches, (B, h_kv, T, top_n), device="cuda", dtype=torch.int32)
        patch_starts = make_patch_starts(H, W, P, q.device)

        o_triton = SelectionAttn2DPerQuery.apply(q, k, v, block_idx, patch_starts, pp, H, W, P, scale, G)
        o_triton.sum().backward()
        dq_triton = q.grad.detach().clone()

        q2 = q.detach().clone().requires_grad_(True)
        o_naive = _naive_gqa_selection_per_query(q2, k, v, block_idx, patch_starts, P, W, G, scale=scale)
        o_naive.sum().backward()
        torch.testing.assert_close(dq_triton, q2.grad, atol=5e-2, rtol=5e-2)

    @pytest.mark.per_query_parity
    def test_per_query_backward_dk_dv_matches_naive_gqa(self):
        """Per-query GQA dK/dV should match naive autograd dK/dV."""
        from src.ops.selection_attention_2d_per_query import SelectionAttn2DPerQuery, make_patch_starts

        B, h_q, h_kv, d = 1, 4, 2, 8
        H, W, P = 4, 4, 2
        T = H * W
        top_n = 2
        pp = P * P
        G = h_q // h_kv
        n_patches = (H // P) * (W // P)
        scale = 1.0 / d**0.5

        torch.manual_seed(1337)
        q = torch.randn(B, h_q, T, d, device="cuda", dtype=torch.float32)
        k = torch.randn(B, h_kv, T, d, device="cuda", dtype=torch.float32, requires_grad=True)
        v = torch.randn(B, h_kv, T, d, device="cuda", dtype=torch.float32, requires_grad=True)
        block_idx = torch.randint(0, n_patches, (B, h_kv, T, top_n), device="cuda", dtype=torch.int32)
        patch_starts = make_patch_starts(H, W, P, q.device)

        o_triton = SelectionAttn2DPerQuery.apply(q, k, v, block_idx, patch_starts, pp, H, W, P, scale, G)
        o_triton.sum().backward()
        dk_triton = k.grad.detach().clone()
        dv_triton = v.grad.detach().clone()

        k2 = k.detach().clone().requires_grad_(True)
        v2 = v.detach().clone().requires_grad_(True)
        o_naive = _naive_gqa_selection_per_query(q, k2, v2, block_idx, patch_starts, P, W, G, scale=scale)
        o_naive.sum().backward()
        torch.testing.assert_close(dk_triton, k2.grad, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(dv_triton, v2.grad, atol=5e-2, rtol=5e-2)
