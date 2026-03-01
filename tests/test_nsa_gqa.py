"""Tests for GQA support in NSA2DAttention (Tasks 1-4)."""
import pytest
import torch
from src.networks.txunet import NSA2DAttention


# ============================================================
# Task 1: Constructor tests
# ============================================================

class TestNSA2DAttentionGQA:
    def test_gqa_constructor_accepts_group_size(self):
        """GQA with group_size=4: 8 Q heads, 2 KV heads."""
        attn = NSA2DAttention(dim=192, num_heads=8, gqa_group_size=4)
        assert attn.h_q == 8
        assert attn.h_kv == 2
        assert attn.gqa_group_size == 4

    def test_gqa_group_size_1_is_mha(self):
        """group_size=1 means MHA (backward compat)."""
        attn = NSA2DAttention(dim=192, num_heads=8, gqa_group_size=1)
        assert attn.h_q == 8
        assert attn.h_kv == 8

    def test_gqa_kv_projection_channels(self):
        """KV projections should output C/G channels, not C."""
        attn = NSA2DAttention(dim=192, num_heads=8, gqa_group_size=4)
        # k_block and v_block output dim // gqa_group_size = 48
        x = torch.randn(1, 192, 16, 16)
        k_out = attn.k_block(x)
        v_out = attn.v_block(x)
        assert k_out.shape == (1, 48, 16, 16)
        assert v_out.shape == (1, 48, 16, 16)
        # q_block still outputs full dim
        q_out = attn.q_block(x)
        assert q_out.shape == (1, 192, 16, 16)

    def test_gqa_invalid_group_size(self):
        """Group size must divide num_heads evenly."""
        with pytest.raises(ValueError):
            NSA2DAttention(dim=192, num_heads=8, gqa_group_size=3)

    def test_gqa_compressor_uses_kv_heads(self):
        """PatchCompressor should use d_head from KV heads, not Q heads."""
        attn = NSA2DAttention(dim=192, num_heads=8, gqa_group_size=4)
        # d = dim // num_heads = 24 (same for Q and KV since d is per-head)
        # compress_k operates on [Bh_kv, n_patches, pp, d]
        assert attn.compress_k.pos_emb.shape[1] == 24


# ============================================================
# Task 2: Forward tests
# ============================================================

class TestNSA2DForwardGQA:
    def test_forward_shape_gqa(self):
        """Forward should produce [B, C, H, W] with GQA enabled."""
        attn = NSA2DAttention(dim=192, num_heads=8, gqa_group_size=4,
                              patch_size=4, top_n=4, window_size=4)
        x = torch.randn(2, 192, 16, 16)
        out = attn(x)
        assert out.shape == (2, 192, 16, 16)

    def test_forward_shape_gqa_small(self):
        """Forward with smallest production-like config: dim=48, heads=4, G=4."""
        attn = NSA2DAttention(dim=48, num_heads=4, gqa_group_size=4,
                              patch_size=4, top_n=4, window_size=4)
        x = torch.randn(1, 48, 16, 16)
        out = attn(x)
        assert out.shape == (1, 48, 16, 16)

    def test_forward_backward_gqa(self):
        """Full forward+backward should work with GQA."""
        attn = NSA2DAttention(dim=192, num_heads=8, gqa_group_size=4,
                              patch_size=4, top_n=4, window_size=4)
        x = torch.randn(1, 192, 16, 16, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_forward_gqa1_matches_mha(self):
        """gqa_group_size=1 should give identical results to old MHA path."""
        torch.manual_seed(42)
        attn_mha = NSA2DAttention(dim=96, num_heads=4, gqa_group_size=1,
                                  patch_size=4, top_n=4, window_size=4)
        x = torch.randn(1, 96, 16, 16)
        out_mha = attn_mha(x)

        torch.manual_seed(42)
        attn_gqa1 = NSA2DAttention(dim=96, num_heads=4, gqa_group_size=1,
                                   patch_size=4, top_n=4, window_size=4)
        out_gqa1 = attn_gqa1(x)
        assert torch.allclose(out_mha, out_gqa1, atol=1e-5)


# ============================================================
# Task 3: Compression branch tests
# ============================================================

class TestCompressionBranchGQA:
    def test_compression_output_has_hq_heads(self):
        """Compression output o_cmp should have h_q heads (not h_kv)."""
        attn = NSA2DAttention(dim=192, num_heads=8, gqa_group_size=4,
                              patch_size=4, top_n=4, window_size=4)
        B, H, W, d = 1, 16, 16, 24
        h_q, h_kv = 8, 2
        q = torch.randn(B, h_q, H * W, d)
        k = torch.randn(B, h_kv, H * W, d)
        v = torch.randn(B, h_kv, H * W, d)
        o_cmp, k_cmp = attn._compression_branch(q, k, v, H, W)
        assert o_cmp.shape == (B, h_q, H * W, d)
        assert k_cmp.shape == (B, h_kv, 16, d)  # 16 patches = (16//4)*(16//4)


# ============================================================
# Task 4: Selection branch tests
# ============================================================

class TestSelectionBranchGQA:
    def test_selection_output_has_hq_heads(self):
        """Selection output should have h_q heads."""
        attn = NSA2DAttention(dim=192, num_heads=8, gqa_group_size=4,
                              patch_size=4, top_n=4, window_size=4)
        B, H, W, d = 1, 16, 16, 24
        h_q, h_kv = 8, 2
        q = torch.randn(B, h_q, H * W, d)
        k = torch.randn(B, h_kv, H * W, d)
        v = torch.randn(B, h_kv, H * W, d)
        k_cmp = torch.randn(B, h_kv, 16, d)  # 16 = n_patches
        o_slc = attn._selection_branch(q, k, v, k_cmp, H, W)
        assert o_slc.shape == (B, h_q, H * W, d)

    def test_selection_per_kv_head_topk(self):
        """Different KV heads should potentially select different patches."""
        attn = NSA2DAttention(dim=192, num_heads=8, gqa_group_size=4,
                              patch_size=4, top_n=2, window_size=4)
        B, H, W, d = 1, 16, 16, 24
        h_q, h_kv = 8, 2
        q = torch.randn(B, h_q, H * W, d)
        k = torch.randn(B, h_kv, H * W, d)
        v = torch.randn(B, h_kv, H * W, d)
        # Make k_cmp very different per kv head to force different selections
        k_cmp = torch.zeros(B, h_kv, 16, d)
        k_cmp[0, 0, 0, :] = 10.0   # KV head 0 should prefer patch 0
        k_cmp[0, 1, 15, :] = 10.0  # KV head 1 should prefer patch 15
        o_slc = attn._selection_branch(q, k, v, k_cmp, H, W)
        assert o_slc.shape == (B, h_q, H * W, d)

    def test_selection_backward_gqa(self):
        """Selection branch backward should produce gradients for GQA inputs."""
        attn = NSA2DAttention(dim=192, num_heads=8, gqa_group_size=4,
                              patch_size=4, top_n=4, window_size=4)
        B, H, W, d = 1, 16, 16, 24
        h_q, h_kv = 8, 2
        q = torch.randn(B, h_q, H * W, d, requires_grad=True)
        k = torch.randn(B, h_kv, H * W, d, requires_grad=True)
        v = torch.randn(B, h_kv, H * W, d, requires_grad=True)
        k_cmp = torch.randn(B, h_kv, 16, d)
        o_slc = attn._selection_branch(q, k, v, k_cmp, H, W)
        o_slc.sum().backward()
        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None
