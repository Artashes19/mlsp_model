"""Test FFN residual path variants."""
import pytest
import torch
from src.networks.txunet import GatedDepthwiseFFN, TransformerBlock


def test_ffn_internal_residual_default():
    """Default (internal_residual=True): FFN(x) = proj(g) + x."""
    ffn = GatedDepthwiseFFN(dim=48, expand=2.66, internal_residual=True)
    x = torch.randn(1, 48, 8, 8)
    out = ffn(x)
    assert out.shape == x.shape
    # With internal residual, output should be close to input when weights are small
    # (not a strict test, just shape check)


def test_ffn_no_internal_residual():
    """Standard pre-LN: FFN(x) = proj(g), no internal residual."""
    ffn = GatedDepthwiseFFN(dim=48, expand=2.66, internal_residual=False)
    x = torch.randn(1, 48, 8, 8)
    out = ffn(x)
    assert out.shape == x.shape


def test_transformer_block_standard_residual():
    """Standard pre-LN: x = x + ffn(norm2(x)) when ffn_internal_residual=False."""
    block = TransformerBlock(dim=48, heads=4, expand=2.66, ffn_internal_residual=False)
    x = torch.randn(1, 48, 16, 16)
    out = block(x)
    assert out.shape == x.shape


def test_transformer_block_internal_residual():
    """Legacy: x = ffn(norm2(x)) when ffn_internal_residual=True."""
    block = TransformerBlock(dim=48, heads=4, expand=2.66, ffn_internal_residual=True)
    x = torch.randn(1, 48, 16, 16)
    out = block(x)
    assert out.shape == x.shape


def test_backward_both_variants():
    """Both variants should produce valid gradients."""
    for internal_res in [True, False]:
        block = TransformerBlock(dim=48, heads=4, expand=2.66, ffn_internal_residual=internal_res)
        x = torch.randn(1, 48, 16, 16, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
