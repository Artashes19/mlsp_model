"""Tests for 2D Rotary Position Embedding."""
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.networks.txunet import (
    EfficientGlobalAttention,
    RotaryEmbedding2D,
    TxUNetModel,
    _apply_rotary_2d,
)


def test_apply_rotary_2d_output_shape():
    """Output shape must match input shape."""
    B, h, T, d = 2, 4, 16, 12
    d_row = d // 2
    d_col = d - d_row
    x = torch.randn(B, h, T, d)
    cos_row = torch.randn(T, d_row // 2)
    sin_row = torch.randn(T, d_row // 2)
    cos_col = torch.randn(T, d_col // 2)
    sin_col = torch.randn(T, d_col // 2)
    out = _apply_rotary_2d(x, cos_row, sin_row, cos_col, sin_col, d_row)
    assert out.shape == x.shape


def test_apply_rotary_2d_preserves_norm():
    """RoPE is a rotation — it preserves vector norm per token."""
    B, h, T, d = 2, 4, 16, 12
    d_row = d // 2
    d_col = d - d_row
    x = torch.randn(B, h, T, d)
    # Use valid cos/sin (from actual angles, not random)
    angles_r = torch.randn(T, d_row // 2)
    angles_c = torch.randn(T, d_col // 2)
    out = _apply_rotary_2d(
        x,
        angles_r.cos(), angles_r.sin(),
        angles_c.cos(), angles_c.sin(),
        d_row,
    )
    x_norm = x.norm(dim=-1)
    out_norm = out.norm(dim=-1)
    torch.testing.assert_close(x_norm, out_norm, atol=1e-5, rtol=1e-5)


def test_rope2d_relative_position_property():
    """
    Core RoPE property: dot(RoPE(q, pos_q), RoPE(k, pos_k)) depends only on
    (pos_q - pos_k), not on absolute positions.
    """
    rope = RotaryEmbedding2D(d_head=12, base=10000.0)
    H, W = 8, 8

    q_raw = torch.randn(1, 1, 1, 12)  # single query token
    k_raw = torch.randn(1, 1, 1, 12)  # single key token

    # Place q at (2,3), k at (5,6): relative = (-3, -3)
    q1 = rope(_expand_to_grid(q_raw, 2, 3, H, W), H, W, stride=1)
    k1 = rope(_expand_to_grid(k_raw, 5, 6, H, W), H, W, stride=1)
    dot1 = (q1[0, 0, 2 * W + 3] * k1[0, 0, 5 * W + 6]).sum()

    # Place q at (1,2), k at (4,5): relative still (-3, -3)
    q2 = rope(_expand_to_grid(q_raw, 1, 2, H, W), H, W, stride=1)
    k2 = rope(_expand_to_grid(k_raw, 4, 5, H, W), H, W, stride=1)
    dot2 = (q2[0, 0, 1 * W + 2] * k2[0, 0, 4 * W + 5]).sum()

    torch.testing.assert_close(dot1, dot2, atol=1e-5, rtol=1e-5)


def _expand_to_grid(x_single, row, col, H, W):
    """Place a single token vector at position (row, col) in an HxW grid of zeros."""
    B, h, _, d = x_single.shape
    grid = torch.zeros(B, h, H * W, d)
    grid[:, :, row * W + col, :] = x_single[:, :, 0, :]
    return grid


def test_rope2d_forward_shape():
    """RotaryEmbedding2D.forward preserves shape."""
    rope = RotaryEmbedding2D(d_head=24, base=10000.0)
    x = torch.randn(2, 4, 64, 24)  # B=2, h=4, T=8*8=64, d=24
    out = rope(x, H=8, W=8, stride=1)
    assert out.shape == x.shape


def test_rope2d_sra_stride():
    """With SRA stride, K at reduced resolution gets scaled positions."""
    rope = RotaryEmbedding2D(d_head=12, base=10000.0)

    # Full-res Q
    q = torch.randn(1, 1, 64, 12)  # H=8, W=8
    q_rot = rope(q, H=8, W=8, stride=1)

    # Reduced-res K with stride=4: Hk=2, Wk=2, positions are 0,4
    k = torch.randn(1, 1, 4, 12)   # Hk=2, Wk=2
    k_rot = rope(k, H=2, W=2, stride=4)

    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    # k_rot should differ from k (rotation was applied)
    assert not torch.equal(k, k_rot)


def test_rope2d_stride1_vs_identity():
    """At position (0,0), all angles are 0, so cos=1, sin=0 -> rotation is identity for that token."""
    rope = RotaryEmbedding2D(d_head=12, base=10000.0)
    x = torch.randn(1, 1, 4, 12)  # H=2, W=2
    out = rope(x, H=2, W=2, stride=1)
    # Token at (0,0) = index 0 should be unchanged
    torch.testing.assert_close(out[0, 0, 0], x[0, 0, 0], atol=1e-6, rtol=1e-6)


def test_rope2d_no_learnable_params():
    """RotaryEmbedding2D must have zero learnable parameters."""
    rope = RotaryEmbedding2D(d_head=48, base=10000.0)
    assert sum(p.numel() for p in rope.parameters()) == 0


def test_rope2d_not_in_state_dict():
    """Buffers must not appear in state_dict (persistent=False)."""
    rope = RotaryEmbedding2D(d_head=24)
    assert len(rope.state_dict()) == 0


def test_rope2d_dtype_preservation():
    """Output dtype must match input dtype (including fp16/bf16)."""
    rope = RotaryEmbedding2D(d_head=12)
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        x = torch.randn(1, 1, 4, 12, dtype=dtype)
        out = rope(x, H=2, W=2, stride=1)
        assert out.dtype == dtype, f"expected {dtype}, got {out.dtype}"


def test_attn_rope_disabled_no_rope_module():
    """When rope_enabled=False, self.rope should be None."""
    attn = EfficientGlobalAttention(dim=48, num_heads=4, rope_enabled=False)
    assert attn.rope is None


def test_attn_rope_enabled_has_rope_module():
    """When rope_enabled=True, self.rope should be a RotaryEmbedding2D."""
    attn = EfficientGlobalAttention(dim=48, num_heads=4, rope_enabled=True)
    assert isinstance(attn.rope, RotaryEmbedding2D)


def test_attn_rope_output_shape():
    """Forward pass shape must be preserved with rope_enabled=True."""
    attn = EfficientGlobalAttention(dim=48, num_heads=4, rope_enabled=True)
    x = torch.randn(1, 48, 16, 16)
    out = attn(x)
    assert out.shape == x.shape


def test_attn_rope_sra_output_shape():
    """Forward pass shape with SRA + RoPE."""
    attn = EfficientGlobalAttention(dim=48, num_heads=4, kv_stride=4, rope_enabled=True)
    x = torch.randn(1, 48, 16, 16)
    out = attn(x)
    assert out.shape == x.shape


def test_attn_rope_changes_output():
    """RoPE must actually change the attention output (not a no-op)."""
    torch.manual_seed(42)
    attn_no_rope = EfficientGlobalAttention(dim=48, num_heads=4, rope_enabled=False)
    attn_rope = EfficientGlobalAttention(dim=48, num_heads=4, rope_enabled=True)
    attn_rope.load_state_dict(attn_no_rope.state_dict())
    x = torch.randn(1, 48, 8, 8)
    out_no = attn_no_rope(x)
    out_yes = attn_rope(x)
    assert not torch.allclose(out_no, out_yes, atol=1e-6), "RoPE should change attention output"


def test_attn_rope_no_new_params():
    """RoPE must not add learnable parameters to the attention module."""
    attn_no = EfficientGlobalAttention(dim=48, num_heads=4, rope_enabled=False)
    attn_yes = EfficientGlobalAttention(dim=48, num_heads=4, rope_enabled=True)
    params_no = sum(p.numel() for p in attn_no.parameters())
    params_yes = sum(p.numel() for p in attn_yes.parameters())
    assert params_no == params_yes


def test_attn_rope_state_dict_unchanged():
    """State dict keys must be identical with or without RoPE."""
    attn_no = EfficientGlobalAttention(dim=48, num_heads=4, rope_enabled=False)
    attn_yes = EfficientGlobalAttention(dim=48, num_heads=4, rope_enabled=True)
    assert set(attn_no.state_dict().keys()) == set(attn_yes.state_dict().keys())


def _make_tiny_model(**kwargs):
    """Create a minimal TxUNet for testing."""
    defaults = dict(
        in_ch=4,
        out_ch=1,
        base_ch=4,
        depths=(1, 1, 1, 1),
        heads=(1, 1, 1, 1),
        expand=1.0,
        use_checkpoint=False,
        sra0_enabled=False,
    )
    defaults.update(kwargs)
    return TxUNetModel(**defaults)


def test_txunet_rope_forward_shape():
    """TxUNet with rope_enabled=True must produce correct output shape."""
    model = _make_tiny_model(rope_enabled=True)
    x = torch.randn(1, 4, 16, 16)
    out = model(x)
    assert out.shape == (1, 1, 16, 16)


def test_txunet_rope_sra_forward_shape():
    """TxUNet with rope + SRA must produce correct output shape."""
    model = _make_tiny_model(rope_enabled=True, sra0_enabled=True, sra0_stride=4)
    x = torch.randn(1, 4, 16, 16)
    out = model(x)
    assert out.shape == (1, 1, 16, 16)


def test_txunet_rope_no_new_params():
    """RoPE must not change TxUNet parameter count."""
    model_no = _make_tiny_model(rope_enabled=False)
    model_yes = _make_tiny_model(rope_enabled=True)
    params_no = sum(p.numel() for p in model_no.parameters())
    params_yes = sum(p.numel() for p in model_yes.parameters())
    assert params_no == params_yes


def test_txunet_rope_checkpoint_compat():
    """Old state_dict (no rope) must load into model with rope_enabled=True."""
    model_old = _make_tiny_model(rope_enabled=False)
    state = model_old.state_dict()
    model_new = _make_tiny_model(rope_enabled=True)
    model_new.load_state_dict(state, strict=True)


def test_txunet_rope_all_levels_have_rope():
    """Verify RoPE is present in attention at ALL encoder/decoder levels."""
    model = _make_tiny_model(rope_enabled=True)
    for name, module in model.named_modules():
        if isinstance(module, EfficientGlobalAttention):
            assert module.rope is not None, f"{name} missing RoPE"


def test_txunet_no_rope_no_modules():
    """Verify NO RoPE modules when rope_enabled=False."""
    model = _make_tiny_model(rope_enabled=False)
    for name, module in model.named_modules():
        if isinstance(module, EfficientGlobalAttention):
            assert module.rope is None, f"{name} has unexpected RoPE"


def test_txunet_rope_no_nan():
    """Forward pass with RoPE must not produce NaN."""
    model = _make_tiny_model(rope_enabled=True, sra0_enabled=True, sra0_stride=2)
    x = torch.randn(2, 4, 16, 16)
    out = model(x)
    assert not torch.isnan(out).any(), "NaN in output"
    assert not torch.isinf(out).any(), "Inf in output"


def test_txunet_rope_backward():
    """Gradients must flow through RoPE without error."""
    model = _make_tiny_model(rope_enabled=True)
    x = torch.randn(1, 4, 16, 16)
    out = model(x)
    loss = out.sum()
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad, "No gradients computed"


if __name__ == "__main__":
    test_apply_rotary_2d_output_shape()
    test_apply_rotary_2d_preserves_norm()
    test_rope2d_relative_position_property()
    test_rope2d_forward_shape()
    test_rope2d_sra_stride()
    test_rope2d_stride1_vs_identity()
    test_rope2d_no_learnable_params()
    test_rope2d_not_in_state_dict()
    test_rope2d_dtype_preservation()
    print("All RoPE unit tests passed.")
