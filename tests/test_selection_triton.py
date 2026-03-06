"""Tests for Triton-accelerated 2D selection attention (forward + backward)."""
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Skip entire module if no CUDA
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for Triton kernels"
)
DEVICE = torch.device("cuda:0")


# ─── helpers ────────────────────────────────────────────────────────────────


def _make_patch_starts(H: int, W: int, patch_size: int, device=DEVICE):
    """Compute flat offsets for each patch's top-left token (row-major)."""
    p = patch_size
    nH, nW = H // p, W // p
    ph = torch.arange(nH, device=device)
    pw = torch.arange(nW, device=device)
    starts = (ph[:, None] * p * W + pw[None, :] * p).reshape(-1)
    return starts.to(torch.int32)


def _naive_selection_attention(q, k, v, top_idx, patch_starts, pp, P, W, scale=None):
    """
    Reference: gather KV tokens from selected patches, run SDPA.

    q, k, v:       [B, h, T, d]
    top_idx:        [B, top_n]  int
    patch_starts:   [n_patches] int
    pp:             tokens per patch (P*P)
    P:              patch edge length
    W:              spatial width (for row-major stride)
    """
    B, h, T, d = q.shape
    top_n = top_idx.shape[1]

    # Build flat KV indices per batch element
    kv_indices = []
    for b in range(B):
        idx = []
        for i in range(top_n):
            base = patch_starts[top_idx[b, i]].item()
            for dr in range(P):
                for dc in range(P):
                    idx.append(base + dr * W + dc)
        kv_indices.append(idx)

    kv_idx = torch.tensor(kv_indices, device=q.device, dtype=torch.long)  # [B, top_n*pp]
    kv_idx_exp = kv_idx[:, None, :, None].expand(B, h, top_n * pp, d)
    k_sel = k.gather(2, kv_idx_exp)
    v_sel = v.gather(2, kv_idx_exp)
    return torch.nn.functional.scaled_dot_product_attention(q, k_sel, v_sel, scale=scale)


# ─── Forward Tests ──────────────────────────────────────────────────────────


class TestSelectionForward:
    """Tests for the Triton selection attention forward kernel."""

    def test_fwd_shape(self):
        """Output and LSE have correct shapes."""
        from src.ops.selection_attention_2d import selection_attn_2d_forward, make_patch_starts

        B, h, d = 2, 4, 16
        H, W, P = 16, 16, 4
        T = H * W
        top_n = 4
        pp = P * P

        q = torch.randn(B, h, T, d, device=DEVICE)
        k = torch.randn(B, h, T, d, device=DEVICE)
        v = torch.randn(B, h, T, d, device=DEVICE)
        top_idx = torch.randint(0, (H // P) * (W // P), (B, top_n), device=DEVICE, dtype=torch.int32)
        patch_starts = make_patch_starts(H, W, P, DEVICE)

        o, lse = selection_attn_2d_forward(q, k, v, top_idx, patch_starts, pp, H, W, P, 1.0 / d**0.5)
        assert o.shape == (B, h, T, d)
        assert lse.shape == (B, h, T)

    def test_fwd_no_nan(self):
        """No NaN or Inf in output."""
        from src.ops.selection_attention_2d import selection_attn_2d_forward, make_patch_starts

        B, h, d = 2, 4, 16
        H, W, P = 16, 16, 4
        T = H * W
        top_n = 4
        pp = P * P

        q = torch.randn(B, h, T, d, device=DEVICE)
        k = torch.randn(B, h, T, d, device=DEVICE)
        v = torch.randn(B, h, T, d, device=DEVICE)
        top_idx = torch.randint(0, (H // P) * (W // P), (B, top_n), device=DEVICE, dtype=torch.int32)
        patch_starts = make_patch_starts(H, W, P, DEVICE)

        o, lse = selection_attn_2d_forward(q, k, v, top_idx, patch_starts, pp, H, W, P, 1.0 / d**0.5)
        assert not torch.isnan(o).any(), "NaN in output"
        assert not torch.isinf(o).any(), "Inf in output"
        assert not torch.isnan(lse).any(), "NaN in LSE"
        assert not torch.isinf(lse).any(), "Inf in LSE"

    def test_fwd_matches_naive_small(self):
        """Triton forward matches naive gather+SDPA (small: B=1, h=1, 8x8, 2 patches)."""
        from src.ops.selection_attention_2d import selection_attn_2d_forward, make_patch_starts

        B, h, d = 1, 1, 16
        H, W, P = 8, 8, 4
        T = H * W
        top_n = 2
        pp = P * P

        torch.manual_seed(42)
        q = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32)
        k = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32)
        v = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32)
        top_idx = torch.tensor([[0, 3]], device=DEVICE, dtype=torch.int32)
        patch_starts = make_patch_starts(H, W, P, DEVICE)

        o_triton, lse = selection_attn_2d_forward(q, k, v, top_idx, patch_starts, pp, H, W, P, 1.0 / d**0.5)
        o_naive = _naive_selection_attention(q, k, v, top_idx, patch_starts, pp, P, W)

        torch.testing.assert_close(o_triton, o_naive, atol=1e-2, rtol=1e-2)

    def test_fwd_matches_naive_multihead(self):
        """Triton forward matches naive with multiple heads."""
        from src.ops.selection_attention_2d import selection_attn_2d_forward, make_patch_starts

        B, h, d = 1, 4, 16
        H, W, P = 8, 8, 4
        T = H * W
        top_n = 2
        pp = P * P

        torch.manual_seed(123)
        q = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32)
        k = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32)
        v = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32)
        top_idx = torch.tensor([[1, 2]], device=DEVICE, dtype=torch.int32)
        patch_starts = make_patch_starts(H, W, P, DEVICE)

        o_triton, _ = selection_attn_2d_forward(q, k, v, top_idx, patch_starts, pp, H, W, P, 1.0 / d**0.5)
        o_naive = _naive_selection_attention(q, k, v, top_idx, patch_starts, pp, P, W)

        torch.testing.assert_close(o_triton, o_naive, atol=1e-2, rtol=1e-2)

    def test_fwd_matches_naive_multibatch(self):
        """Triton forward matches naive with batch > 1."""
        from src.ops.selection_attention_2d import selection_attn_2d_forward, make_patch_starts

        B, h, d = 2, 2, 16
        H, W, P = 8, 8, 4
        T = H * W
        top_n = 3
        pp = P * P

        torch.manual_seed(999)
        q = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32)
        k = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32)
        v = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32)
        top_idx = torch.tensor([[0, 1, 3], [2, 3, 0]], device=DEVICE, dtype=torch.int32)
        patch_starts = make_patch_starts(H, W, P, DEVICE)

        o_triton, _ = selection_attn_2d_forward(q, k, v, top_idx, patch_starts, pp, H, W, P, 1.0 / d**0.5)
        o_naive = _naive_selection_attention(q, k, v, top_idx, patch_starts, pp, P, W)

        torch.testing.assert_close(o_triton, o_naive, atol=1e-2, rtol=1e-2)

    def test_fwd_matches_naive_16x16(self):
        """Triton forward matches naive on larger spatial grid (16x16)."""
        from src.ops.selection_attention_2d import selection_attn_2d_forward, make_patch_starts

        B, h, d = 1, 2, 32
        H, W, P = 16, 16, 4
        T = H * W
        top_n = 4
        pp = P * P

        torch.manual_seed(7)
        q = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32)
        k = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32)
        v = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32)
        top_idx = torch.tensor([[0, 5, 10, 15]], device=DEVICE, dtype=torch.int32)
        patch_starts = make_patch_starts(H, W, P, DEVICE)

        o_triton, _ = selection_attn_2d_forward(q, k, v, top_idx, patch_starts, pp, H, W, P, 1.0 / d**0.5)
        o_naive = _naive_selection_attention(q, k, v, top_idx, patch_starts, pp, P, W)

        torch.testing.assert_close(o_triton, o_naive, atol=1e-2, rtol=1e-2)

    def test_fwd_fp16(self):
        """Forward works in half precision without NaN."""
        from src.ops.selection_attention_2d import selection_attn_2d_forward, make_patch_starts

        B, h, d = 1, 2, 16
        H, W, P = 8, 8, 4
        T = H * W
        top_n = 2
        pp = P * P

        q = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float16)
        k = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float16)
        v = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float16)
        top_idx = torch.randint(0, 4, (B, top_n), device=DEVICE, dtype=torch.int32)
        patch_starts = make_patch_starts(H, W, P, DEVICE)

        o, lse = selection_attn_2d_forward(q, k, v, top_idx, patch_starts, pp, H, W, P, 1.0 / d**0.5)
        assert o.dtype == torch.float16
        assert not torch.isnan(o).any()

    def test_fwd_bf16(self):
        """Forward works in bfloat16."""
        from src.ops.selection_attention_2d import selection_attn_2d_forward, make_patch_starts

        B, h, d = 1, 2, 16
        H, W, P = 8, 8, 4
        T = H * W
        top_n = 2
        pp = P * P

        q = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.bfloat16)
        k = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.bfloat16)
        v = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.bfloat16)
        top_idx = torch.randint(0, 4, (B, top_n), device=DEVICE, dtype=torch.int32)
        patch_starts = make_patch_starts(H, W, P, DEVICE)

        o, lse = selection_attn_2d_forward(q, k, v, top_idx, patch_starts, pp, H, W, P, 1.0 / d**0.5)
        assert o.dtype == torch.bfloat16
        assert not torch.isnan(o).any()

    def test_fwd_deterministic(self):
        """Same input gives same output."""
        from src.ops.selection_attention_2d import selection_attn_2d_forward, make_patch_starts

        B, h, d = 1, 2, 16
        H, W, P = 8, 8, 4
        T = H * W
        pp = P * P

        q = torch.randn(B, h, T, d, device=DEVICE)
        k = torch.randn(B, h, T, d, device=DEVICE)
        v = torch.randn(B, h, T, d, device=DEVICE)
        top_idx = torch.tensor([[0, 2]], device=DEVICE, dtype=torch.int32)
        patch_starts = make_patch_starts(H, W, P, DEVICE)

        o1, lse1 = selection_attn_2d_forward(q, k, v, top_idx, patch_starts, pp, H, W, P, 1.0 / d**0.5)
        o2, lse2 = selection_attn_2d_forward(q, k, v, top_idx, patch_starts, pp, H, W, P, 1.0 / d**0.5)
        torch.testing.assert_close(o1, o2)
        torch.testing.assert_close(lse1, lse2)

    def test_fwd_all_patches_selected(self):
        """When all patches are selected, output should match full attention."""
        from src.ops.selection_attention_2d import selection_attn_2d_forward, make_patch_starts

        B, h, d = 1, 1, 16
        H, W, P = 8, 8, 4
        T = H * W
        n_patches = (H // P) * (W // P)  # 4
        pp = P * P

        torch.manual_seed(42)
        q = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32)
        k = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32)
        v = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32)
        # Select all 4 patches
        top_idx = torch.arange(n_patches, device=DEVICE, dtype=torch.int32).unsqueeze(0)
        patch_starts = make_patch_starts(H, W, P, DEVICE)

        o_triton, _ = selection_attn_2d_forward(
            q, k, v, top_idx, patch_starts, pp, H, W, P, 1.0 / d**0.5
        )
        # When all patches selected, should equal full SDPA
        o_full = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        torch.testing.assert_close(o_triton, o_full, atol=1e-2, rtol=1e-2)

    def test_fwd_lse_values_sane(self):
        """LSE values should be finite and in a reasonable range."""
        from src.ops.selection_attention_2d import selection_attn_2d_forward, make_patch_starts

        B, h, d = 1, 2, 16
        H, W, P = 8, 8, 4
        T = H * W
        pp = P * P

        q = torch.randn(B, h, T, d, device=DEVICE)
        k = torch.randn(B, h, T, d, device=DEVICE)
        v = torch.randn(B, h, T, d, device=DEVICE)
        top_idx = torch.tensor([[0, 3]], device=DEVICE, dtype=torch.int32)
        patch_starts = make_patch_starts(H, W, P, DEVICE)

        _, lse = selection_attn_2d_forward(q, k, v, top_idx, patch_starts, pp, H, W, P, 1.0 / d**0.5)
        assert lse.dtype == torch.float32
        assert not torch.isnan(lse).any()
        assert not torch.isinf(lse).any()
        # LSE should be at least log(top_n * pp) ≈ log(32) ≈ 3.47 for uniform weights
        # but can be larger; just check it's not all zeros
        assert lse.abs().sum() > 0

    def test_fwd_nonsquare_spatial(self):
        """Works with non-square spatial (H != W)."""
        from src.ops.selection_attention_2d import selection_attn_2d_forward, make_patch_starts

        B, h, d = 1, 2, 16
        H, W, P = 8, 16, 4  # non-square
        T = H * W
        top_n = 3
        pp = P * P

        q = torch.randn(B, h, T, d, device=DEVICE)
        k = torch.randn(B, h, T, d, device=DEVICE)
        v = torch.randn(B, h, T, d, device=DEVICE)
        n_patches = (H // P) * (W // P)
        top_idx = torch.randint(0, n_patches, (B, top_n), device=DEVICE, dtype=torch.int32)
        patch_starts = make_patch_starts(H, W, P, DEVICE)

        o, lse = selection_attn_2d_forward(q, k, v, top_idx, patch_starts, pp, H, W, P, 1.0 / d**0.5)
        assert o.shape == (B, h, T, d)
        assert not torch.isnan(o).any()

    def test_fwd_nonsquare_matches_naive(self):
        """Triton matches naive on non-square spatial grid."""
        from src.ops.selection_attention_2d import selection_attn_2d_forward, make_patch_starts

        B, h, d = 1, 1, 16
        H, W, P = 8, 16, 4
        T = H * W
        top_n = 2
        pp = P * P

        torch.manual_seed(55)
        q = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32)
        k = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32)
        v = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32)
        n_patches = (H // P) * (W // P)
        top_idx = torch.tensor([[0, n_patches - 1]], device=DEVICE, dtype=torch.int32)
        patch_starts = make_patch_starts(H, W, P, DEVICE)

        o_triton, _ = selection_attn_2d_forward(q, k, v, top_idx, patch_starts, pp, H, W, P, 1.0 / d**0.5)
        o_naive = _naive_selection_attention(q, k, v, top_idx, patch_starts, pp, P, W)

        torch.testing.assert_close(o_triton, o_naive, atol=1e-2, rtol=1e-2)

    def test_fwd_respects_explicit_scale(self):
        """Forward should match reference when using non-default attention scale."""
        from src.ops.selection_attention_2d import selection_attn_2d_forward, make_patch_starts

        B, h, d = 1, 1, 16
        H, W, P = 8, 8, 4
        T = H * W
        top_n = 2
        pp = P * P
        scale = 0.125  # non-default (default would be 0.25 for d=16)

        torch.manual_seed(314)
        q = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32)
        k = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32)
        v = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32)
        top_idx = torch.tensor([[0, 3]], device=DEVICE, dtype=torch.int32)
        patch_starts = make_patch_starts(H, W, P, DEVICE)

        o_triton, _ = selection_attn_2d_forward(q, k, v, top_idx, patch_starts, pp, H, W, P, scale)
        o_naive = _naive_selection_attention(q, k, v, top_idx, patch_starts, pp, P, W, scale=scale)

        torch.testing.assert_close(o_triton, o_naive, atol=1e-2, rtol=1e-2)


# ─── Backward Tests ─────────────────────────────────────────────────────────


class TestSelectionBackward:
    """Tests for the Triton selection attention backward kernels."""

    def test_top_idx_duplicate_detection_helper(self):
        """Helper should flag duplicate patch indices per batch element."""
        from src.ops.selection_attention_2d import _top_idx_has_duplicates

        unique = torch.tensor([[0, 2, 3], [4, 5, 6]], device=DEVICE, dtype=torch.int32)
        dup = torch.tensor([[0, 2, 2], [4, 5, 6]], device=DEVICE, dtype=torch.int32)

        assert _top_idx_has_duplicates(unique) is False
        assert _top_idx_has_duplicates(dup) is True

    def test_backward_shapes(self):
        """dQ, dK, dV have same shapes as Q, K, V."""
        from src.ops.selection_attention_2d import SelectionAttn2D, make_patch_starts

        B, h, d = 2, 4, 16
        H, W, P = 16, 16, 4
        T = H * W
        top_n = 4
        pp = P * P

        q = torch.randn(B, h, T, d, device=DEVICE, requires_grad=True)
        k = torch.randn(B, h, T, d, device=DEVICE, requires_grad=True)
        v = torch.randn(B, h, T, d, device=DEVICE, requires_grad=True)
        top_idx = torch.randint(0, 16, (B, top_n), device=DEVICE, dtype=torch.int32)
        patch_starts = make_patch_starts(H, W, P, DEVICE)

        o = SelectionAttn2D.apply(q, k, v, top_idx, patch_starts, pp, H, W, P, 1.0 / d**0.5)
        o.sum().backward()

        assert q.grad.shape == q.shape
        assert k.grad.shape == k.shape
        assert v.grad.shape == v.shape

    def test_backward_no_nan(self):
        """Gradients are free of NaN and Inf."""
        from src.ops.selection_attention_2d import SelectionAttn2D, make_patch_starts

        B, h, d = 1, 2, 16
        H, W, P = 8, 8, 4
        T = H * W
        pp = P * P

        q = torch.randn(B, h, T, d, device=DEVICE, requires_grad=True)
        k = torch.randn(B, h, T, d, device=DEVICE, requires_grad=True)
        v = torch.randn(B, h, T, d, device=DEVICE, requires_grad=True)
        top_idx = torch.tensor([[0, 3]], device=DEVICE, dtype=torch.int32)
        patch_starts = make_patch_starts(H, W, P, DEVICE)

        o = SelectionAttn2D.apply(q, k, v, top_idx, patch_starts, pp, H, W, P, 1.0 / d**0.5)
        o.sum().backward()

        assert not torch.isnan(q.grad).any(), "NaN in dQ"
        assert not torch.isnan(k.grad).any(), "NaN in dK"
        assert not torch.isnan(v.grad).any(), "NaN in dV"

    def test_backward_grads_nonzero(self):
        """All three gradients are nonzero."""
        from src.ops.selection_attention_2d import SelectionAttn2D, make_patch_starts

        B, h, d = 1, 2, 16
        H, W, P = 8, 8, 4
        T = H * W
        pp = P * P

        q = torch.randn(B, h, T, d, device=DEVICE, requires_grad=True)
        k = torch.randn(B, h, T, d, device=DEVICE, requires_grad=True)
        v = torch.randn(B, h, T, d, device=DEVICE, requires_grad=True)
        top_idx = torch.tensor([[0, 3]], device=DEVICE, dtype=torch.int32)
        patch_starts = make_patch_starts(H, W, P, DEVICE)

        o = SelectionAttn2D.apply(q, k, v, top_idx, patch_starts, pp, H, W, P, 1.0 / d**0.5)
        o.sum().backward()

        assert q.grad.abs().sum() > 0, "dQ is all zeros"
        assert k.grad.abs().sum() > 0, "dK is all zeros"
        assert v.grad.abs().sum() > 0, "dV is all zeros"

    def test_backward_matches_naive_all(self):
        """Comprehensive backward check: dQ, dK, dV all match naive implementation."""
        from src.ops.selection_attention_2d import SelectionAttn2D, make_patch_starts

        B, h, d = 1, 2, 16
        H, W, P = 8, 8, 4
        T = H * W
        top_n = 2
        pp = P * P

        q = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32, requires_grad=True)
        k = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32, requires_grad=True)
        v = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32, requires_grad=True)
        top_idx = torch.tensor([[0, 2]], device=DEVICE, dtype=torch.int32)
        patch_starts = make_patch_starts(H, W, P, DEVICE)
        scale = 1.0 / d**0.5

        # Triton path
        o_tri = SelectionAttn2D.apply(q, k, v, top_idx, patch_starts, pp, H, W, P, scale)
        loss_tri = o_tri.sum()
        loss_tri.backward()
        dq_tri, dk_tri, dv_tri = q.grad.clone(), k.grad.clone(), v.grad.clone()
        q.grad, k.grad, v.grad = None, None, None

        # Naive path
        o_naive = _naive_selection_attention(q, k, v, top_idx, patch_starts, pp, P, W)
        loss_naive = o_naive.sum()
        loss_naive.backward()

        torch.testing.assert_close(dq_tri, q.grad, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(dk_tri, k.grad, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(dv_tri, v.grad, atol=1e-2, rtol=1e-2)

    def test_backward_matches_naive_dv(self):
        """dV from Triton backward matches dV from naive (autograd on gather+SDPA)."""
        from src.ops.selection_attention_2d import SelectionAttn2D, make_patch_starts

        B, h, d = 1, 1, 16
        H, W, P = 8, 8, 4
        T = H * W
        top_n = 2
        pp = P * P

        torch.manual_seed(42)
        q = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32, requires_grad=True)
        k = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32, requires_grad=True)
        v = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32, requires_grad=True)
        top_idx = torch.tensor([[0, 3]], device=DEVICE, dtype=torch.int32)
        patch_starts = make_patch_starts(H, W, P, DEVICE)

        # Triton path
        o_triton = SelectionAttn2D.apply(q, k, v, top_idx, patch_starts, pp, H, W, P, 1.0 / d**0.5)
        loss_triton = o_triton.sum()
        loss_triton.backward()
        dv_triton = v.grad.clone()

        # Naive path
        q2 = q.detach().clone().requires_grad_(True)
        k2 = k.detach().clone().requires_grad_(True)
        v2 = v.detach().clone().requires_grad_(True)
        o_naive = _naive_selection_attention(q2, k2, v2, top_idx, patch_starts, pp, P, W)
        o_naive.sum().backward()

        torch.testing.assert_close(dv_triton, v2.grad, atol=5e-2, rtol=5e-2)

    def test_backward_matches_naive_dq(self):
        """dQ from Triton backward matches dQ from naive."""
        from src.ops.selection_attention_2d import SelectionAttn2D, make_patch_starts

        B, h, d = 1, 1, 16
        H, W, P = 8, 8, 4
        T = H * W
        top_n = 2
        pp = P * P

        torch.manual_seed(42)
        q = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32, requires_grad=True)
        k = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32, requires_grad=True)
        v = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32, requires_grad=True)
        top_idx = torch.tensor([[0, 3]], device=DEVICE, dtype=torch.int32)
        patch_starts = make_patch_starts(H, W, P, DEVICE)

        o_triton = SelectionAttn2D.apply(q, k, v, top_idx, patch_starts, pp, H, W, P, 1.0 / d**0.5)
        o_triton.sum().backward()
        dq_triton = q.grad.clone()

        q2 = q.detach().clone().requires_grad_(True)
        k2 = k.detach().clone().requires_grad_(True)
        v2 = v.detach().clone().requires_grad_(True)
        o_naive = _naive_selection_attention(q2, k2, v2, top_idx, patch_starts, pp, P, W)
        o_naive.sum().backward()

        torch.testing.assert_close(dq_triton, q2.grad, atol=5e-2, rtol=5e-2)

    def test_backward_respects_explicit_scale(self):
        """Backward dQ/dK/dV should match reference for non-default scale."""
        from src.ops.selection_attention_2d import SelectionAttn2D, make_patch_starts

        B, h, d = 1, 1, 16
        H, W, P = 8, 8, 4
        T = H * W
        top_n = 2
        pp = P * P
        scale = 0.125  # non-default (default would be 0.25 for d=16)

        torch.manual_seed(2718)
        q = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32, requires_grad=True)
        k = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32, requires_grad=True)
        v = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32, requires_grad=True)
        top_idx = torch.tensor([[0, 2]], device=DEVICE, dtype=torch.int32)
        patch_starts = make_patch_starts(H, W, P, DEVICE)
        grad_out = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32)

        # Triton path
        o_triton = SelectionAttn2D.apply(q, k, v, top_idx, patch_starts, pp, H, W, P, scale)
        torch.autograd.backward(o_triton, grad_out)
        dq_tri, dk_tri, dv_tri = q.grad.clone(), k.grad.clone(), v.grad.clone()

        # Naive path
        q2 = q.detach().clone().requires_grad_(True)
        k2 = k.detach().clone().requires_grad_(True)
        v2 = v.detach().clone().requires_grad_(True)
        o_naive = _naive_selection_attention(q2, k2, v2, top_idx, patch_starts, pp, P, W, scale=scale)
        torch.autograd.backward(o_naive, grad_out)

        torch.testing.assert_close(dq_tri, q2.grad, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(dk_tri, k2.grad, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(dv_tri, v2.grad, atol=5e-2, rtol=5e-2)


# ─── make_patch_starts Tests ────────────────────────────────────────────────


class TestMakePatchStarts:
    """Tests for the make_patch_starts helper."""

    def test_patch_starts_values(self):
        """Verify patch start offsets for 8x8, patch_size=4."""
        from src.ops.selection_attention_2d import make_patch_starts

        starts = make_patch_starts(8, 8, 4, DEVICE)
        # 2x2 grid of 4x4 patches:
        # Patch(0,0): top-left=0     -> 0*8+0 = 0
        # Patch(0,1): top-left=4     -> 0*8+4 = 4
        # Patch(1,0): top-left=32    -> 4*8+0 = 32
        # Patch(1,1): top-left=36    -> 4*8+4 = 36
        expected = torch.tensor([0, 4, 32, 36], dtype=torch.int32, device=DEVICE)
        torch.testing.assert_close(starts, expected)

    def test_patch_starts_count(self):
        """Number of patches = (H//P) * (W//P)."""
        from src.ops.selection_attention_2d import make_patch_starts

        starts = make_patch_starts(16, 16, 4, DEVICE)
        assert starts.shape[0] == 16  # 4 * 4

    def test_patch_starts_nonsquare(self):
        """Works for non-square spatial."""
        from src.ops.selection_attention_2d import make_patch_starts

        starts = make_patch_starts(8, 16, 4, DEVICE)
        assert starts.shape[0] == 8  # 2 * 4


def _naive_selection_attention_per_query(q, k, v, block_idx, patch_starts, P, W, scale=None):
    """Reference per-query selection: each query token has its own top-k patches."""
    B, h, T, d = q.shape
    top_n = block_idx.shape[-1]
    out = torch.empty_like(q)

    for b in range(B):
        for hh in range(h):
            for t in range(T):
                flat_idx: list[int] = []
                for i in range(top_n):
                    patch = int(block_idx[b, hh, t, i].item())
                    base = int(patch_starts[patch].item())
                    for dr in range(P):
                        for dc in range(P):
                            flat_idx.append(base + dr * W + dc)
                idx_t = torch.tensor(flat_idx, device=q.device, dtype=torch.long)
                q_t = q[b, hh, t : t + 1, :].unsqueeze(0)
                k_t = k[b, hh, idx_t, :].unsqueeze(0)
                v_t = v[b, hh, idx_t, :].unsqueeze(0)
                o_t = torch.nn.functional.scaled_dot_product_attention(q_t, k_t, v_t, scale=scale)
                out[b, hh, t, :] = o_t[0, 0, :]
    return out


class TestPerQuerySelectionForward:
    @pytest.mark.per_query_parity
    def test_packed_per_query_forward_matches_unpacked_mha(self):
        """Packed per-query forward should match unpacked forward in MHA mode."""
        from src.networks.txunet import NSA2DAttention

        B, h, d = 1, 4, 8
        H, W, P = 4, 4, 2
        T = H * W
        top_n = 2
        C = h * d
        n_patches = (H // P) * (W // P)

        torch.manual_seed(9090)
        attn_unpacked = NSA2DAttention(
            dim=C,
            num_heads=h,
            patch_size=P,
            top_n=top_n,
            window_size=2,
            rope_enabled=False,
            gqa_group_size=1,
            selection_forward_mode="unpacked",
        ).to(device=DEVICE, dtype=torch.float32)
        attn_packed = NSA2DAttention(
            dim=C,
            num_heads=h,
            patch_size=P,
            top_n=top_n,
            window_size=2,
            rope_enabled=False,
            gqa_group_size=1,
            selection_forward_mode="packed",
        ).to(device=DEVICE, dtype=torch.float32)

        q = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32)
        k = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32)
        v = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32)
        block_idx = torch.randint(0, n_patches, (B, h, T, top_n), device=DEVICE, dtype=torch.int32)

        with torch.no_grad():
            o_unpacked = attn_unpacked._selection_from_block_idx(q, k, v, block_idx, H, W)
            o_packed = attn_packed._selection_from_block_idx(q, k, v, block_idx, H, W)

        torch.testing.assert_close(o_packed, o_unpacked, atol=1e-2, rtol=1e-2)

    @pytest.mark.per_query_parity
    def test_per_query_forward_shape_contract_mha(self):
        """Per-query MHA signature uses block_idx [B, h_q, T, top_k]."""
        from src.ops.selection_attention_2d_per_query import SelectionAttn2DPerQuery, make_patch_starts

        B, h, d = 1, 2, 8
        H, W, P = 4, 4, 2
        T = H * W
        top_n = 2
        pp = P * P
        n_patches = (H // P) * (W // P)

        q = torch.randn(B, h, T, d, device=DEVICE)
        k = torch.randn(B, h, T, d, device=DEVICE)
        v = torch.randn(B, h, T, d, device=DEVICE)
        block_idx = torch.randint(0, n_patches, (B, h, T, top_n), device=DEVICE, dtype=torch.int32)
        patch_starts = make_patch_starts(H, W, P, DEVICE)

        o = SelectionAttn2DPerQuery.apply(q, k, v, block_idx, patch_starts, pp, H, W, P, 1.0 / d**0.5, 1)
        assert o.shape == (B, h, T, d)

    @pytest.mark.per_query_parity
    def test_per_query_forward_matches_naive_mha(self):
        """Per-query Triton forward should match naive per-query gather+attention."""
        from src.ops.selection_attention_2d_per_query import SelectionAttn2DPerQuery, make_patch_starts

        B, h, d = 1, 2, 8
        H, W, P = 4, 4, 2
        T = H * W
        top_n = 2
        pp = P * P
        n_patches = (H // P) * (W // P)
        scale = 1.0 / d**0.5

        torch.manual_seed(123)
        q = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32)
        k = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32)
        v = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32)
        block_idx = torch.randint(0, n_patches, (B, h, T, top_n), device=DEVICE, dtype=torch.int32)
        patch_starts = make_patch_starts(H, W, P, DEVICE)

        o_triton = SelectionAttn2DPerQuery.apply(q, k, v, block_idx, patch_starts, pp, H, W, P, scale, 1)
        o_naive = _naive_selection_attention_per_query(q, k, v, block_idx, patch_starts, P, W, scale=scale)
        torch.testing.assert_close(o_triton, o_naive, atol=1e-2, rtol=1e-2)

    @pytest.mark.per_query_parity
    def test_per_query_forward_matches_naive_mha_g1_multihead_regime(self):
        """Per-query forward stays correct in the explicit G=1 multihead regime."""
        from src.ops.selection_attention_2d_per_query import SelectionAttn2DPerQuery, make_patch_starts

        B, h, d = 1, 4, 8
        H, W, P = 4, 4, 2
        T = H * W
        top_n = 2
        pp = P * P
        n_patches = (H // P) * (W // P)
        scale = 1.0 / d**0.5

        torch.manual_seed(2026)
        q = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32)
        k = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32)
        v = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32)
        block_idx = torch.randint(0, n_patches, (B, h, T, top_n), device=DEVICE, dtype=torch.int32)
        patch_starts = make_patch_starts(H, W, P, DEVICE)

        o_triton = SelectionAttn2DPerQuery.apply(q, k, v, block_idx, patch_starts, pp, H, W, P, scale, 1)
        o_naive = _naive_selection_attention_per_query(q, k, v, block_idx, patch_starts, P, W, scale=scale)
        torch.testing.assert_close(o_triton, o_naive, atol=1e-2, rtol=1e-2)


class TestPerQuerySelectionBackward:
    @pytest.mark.per_query_parity
    def test_packed_per_query_backward_dq_avoids_unpacked_dispatch_mha(self, monkeypatch):
        """Packed dQ mode must not fall back to the unpacked dQ helper in MHA mode."""
        import src.ops.selection_attention_2d_per_query as perq_ops
        from src.networks.txunet import NSA2DAttention

        def _raise_if_called(*args, **kwargs):
            raise AssertionError("packed dQ fell back to unpacked dispatch")

        monkeypatch.setattr(perq_ops, "selection_per_query_bwd_dq", _raise_if_called)

        B, h, d = 1, 2, 8
        H, W, P = 4, 4, 2
        T = H * W
        top_n = 2
        C = h * d
        n_patches = (H // P) * (W // P)

        torch.manual_seed(6050)
        attn_packed = NSA2DAttention(
            dim=C,
            num_heads=h,
            patch_size=P,
            top_n=top_n,
            window_size=2,
            rope_enabled=False,
            gqa_group_size=1,
            selection_forward_mode="unpacked",
            selection_dq_mode="packed",
        ).to(device=DEVICE, dtype=torch.float32)

        q = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32, requires_grad=True)
        k = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32)
        v = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32)
        block_idx = torch.randint(0, n_patches, (B, h, T, top_n), device=DEVICE, dtype=torch.int32)

        o = attn_packed._selection_from_block_idx(q, k, v, block_idx, H, W)
        o.sum().backward()

    @pytest.mark.per_query_parity
    def test_packed_per_query_backward_dq_matches_unpacked_mha(self):
        """Packed per-query dQ should match unpacked dQ in MHA mode."""
        from src.networks.txunet import NSA2DAttention

        B, h, d = 1, 4, 8
        H, W, P = 4, 4, 2
        T = H * W
        top_n = 2
        C = h * d
        n_patches = (H // P) * (W // P)

        torch.manual_seed(6060)
        attn_unpacked = NSA2DAttention(
            dim=C,
            num_heads=h,
            patch_size=P,
            top_n=top_n,
            window_size=2,
            rope_enabled=False,
            gqa_group_size=1,
            selection_forward_mode="unpacked",
            selection_dq_mode="unpacked",
        ).to(device=DEVICE, dtype=torch.float32)
        attn_packed = NSA2DAttention(
            dim=C,
            num_heads=h,
            patch_size=P,
            top_n=top_n,
            window_size=2,
            rope_enabled=False,
            gqa_group_size=1,
            selection_forward_mode="unpacked",
            selection_dq_mode="packed",
        ).to(device=DEVICE, dtype=torch.float32)

        q = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32, requires_grad=True)
        k = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32)
        v = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32)
        block_idx = torch.randint(0, n_patches, (B, h, T, top_n), device=DEVICE, dtype=torch.int32)

        o_unpacked = attn_unpacked._selection_from_block_idx(q, k, v, block_idx, H, W)
        o_unpacked.sum().backward()
        dq_unpacked = q.grad.detach().clone()

        q2 = q.detach().clone().requires_grad_(True)
        o_packed = attn_packed._selection_from_block_idx(q2, k, v, block_idx, H, W)
        o_packed.sum().backward()
        dq_packed = q2.grad.detach().clone()

        torch.testing.assert_close(dq_packed, dq_unpacked, atol=5e-2, rtol=5e-2)

    @pytest.mark.per_query_parity
    def test_packed_per_query_backward_dq_matches_naive_mha(self):
        """Packed per-query dQ should match naive autograd dQ in MHA mode."""
        from src.networks.txunet import NSA2DAttention

        B, h, d = 1, 2, 8
        H, W, P = 4, 4, 2
        T = H * W
        top_n = 2
        C = h * d
        n_patches = (H // P) * (W // P)
        scale = 1.0 / d**0.5

        torch.manual_seed(6161)
        attn_packed = NSA2DAttention(
            dim=C,
            num_heads=h,
            patch_size=P,
            top_n=top_n,
            window_size=2,
            rope_enabled=False,
            gqa_group_size=1,
            selection_forward_mode="unpacked",
            selection_dq_mode="packed",
        ).to(device=DEVICE, dtype=torch.float32)

        q = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32, requires_grad=True)
        k = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32)
        v = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32)
        block_idx = torch.randint(0, n_patches, (B, h, T, top_n), device=DEVICE, dtype=torch.int32)

        o_packed = attn_packed._selection_from_block_idx(q, k, v, block_idx, H, W)
        o_packed.sum().backward()
        dq_packed = q.grad.detach().clone()

        q2 = q.detach().clone().requires_grad_(True)
        patch_starts = _make_patch_starts(H, W, P, DEVICE)
        o_naive = _naive_selection_attention_per_query(q2, k, v, block_idx, patch_starts, P, W, scale=scale)
        o_naive.sum().backward()
        torch.testing.assert_close(dq_packed, q2.grad, atol=5e-2, rtol=5e-2)

    @pytest.mark.per_query_parity
    def test_per_query_backward_dq_matches_naive_mha(self):
        """Per-query Triton dQ should match naive autograd dQ."""
        from src.ops.selection_attention_2d_per_query import SelectionAttn2DPerQuery, make_patch_starts

        B, h, d = 1, 2, 8
        H, W, P = 4, 4, 2
        T = H * W
        top_n = 2
        pp = P * P
        n_patches = (H // P) * (W // P)
        scale = 1.0 / d**0.5

        torch.manual_seed(777)
        q = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32, requires_grad=True)
        k = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32)
        v = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32)
        block_idx = torch.randint(0, n_patches, (B, h, T, top_n), device=DEVICE, dtype=torch.int32)
        patch_starts = make_patch_starts(H, W, P, DEVICE)

        o_triton = SelectionAttn2DPerQuery.apply(q, k, v, block_idx, patch_starts, pp, H, W, P, scale, 1)
        o_triton.sum().backward()
        dq_triton = q.grad.detach().clone()

        q2 = q.detach().clone().requires_grad_(True)
        o_naive = _naive_selection_attention_per_query(q2, k, v, block_idx, patch_starts, P, W, scale=scale)
        o_naive.sum().backward()
        torch.testing.assert_close(dq_triton, q2.grad, atol=5e-2, rtol=5e-2)

    @pytest.mark.per_query_parity
    def test_per_query_backward_dq_matches_naive_mha_g1_multihead_regime(self):
        """Per-query dQ stays correct in the explicit G=1 multihead regime."""
        from src.ops.selection_attention_2d_per_query import SelectionAttn2DPerQuery, make_patch_starts

        B, h, d = 1, 4, 8
        H, W, P = 4, 4, 2
        T = H * W
        top_n = 2
        pp = P * P
        n_patches = (H // P) * (W // P)
        scale = 1.0 / d**0.5

        torch.manual_seed(2027)
        q = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32, requires_grad=True)
        k = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32)
        v = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32)
        block_idx = torch.randint(0, n_patches, (B, h, T, top_n), device=DEVICE, dtype=torch.int32)
        patch_starts = make_patch_starts(H, W, P, DEVICE)

        o_triton = SelectionAttn2DPerQuery.apply(q, k, v, block_idx, patch_starts, pp, H, W, P, scale, 1)
        o_triton.sum().backward()
        dq_triton = q.grad.detach().clone()

        q2 = q.detach().clone().requires_grad_(True)
        o_naive = _naive_selection_attention_per_query(q2, k, v, block_idx, patch_starts, P, W, scale=scale)
        o_naive.sum().backward()
        torch.testing.assert_close(dq_triton, q2.grad, atol=5e-2, rtol=5e-2)

    @pytest.mark.per_query_parity
    def test_per_query_backward_dk_dv_matches_naive_mha(self):
        """Per-query Triton dK/dV should match naive autograd dK/dV."""
        from src.ops.selection_attention_2d_per_query import SelectionAttn2DPerQuery, make_patch_starts

        B, h, d = 1, 2, 8
        H, W, P = 4, 4, 2
        T = H * W
        top_n = 2
        pp = P * P
        n_patches = (H // P) * (W // P)
        scale = 1.0 / d**0.5

        torch.manual_seed(991)
        q = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32)
        k = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32, requires_grad=True)
        v = torch.randn(B, h, T, d, device=DEVICE, dtype=torch.float32, requires_grad=True)
        block_idx = torch.randint(0, n_patches, (B, h, T, top_n), device=DEVICE, dtype=torch.int32)
        patch_starts = make_patch_starts(H, W, P, DEVICE)

        o_triton = SelectionAttn2DPerQuery.apply(q, k, v, block_idx, patch_starts, pp, H, W, P, scale, 1)
        o_triton.sum().backward()
        dk_triton = k.grad.detach().clone()
        dv_triton = v.grad.detach().clone()

        k2 = k.detach().clone().requires_grad_(True)
        v2 = v.detach().clone().requires_grad_(True)
        o_naive = _naive_selection_attention_per_query(q, k2, v2, block_idx, patch_starts, P, W, scale=scale)
        o_naive.sum().backward()
        torch.testing.assert_close(dk_triton, k2.grad, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(dv_triton, v2.grad, atol=5e-2, rtol=5e-2)
