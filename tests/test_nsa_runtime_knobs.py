"""Tests for safe NSA runtime knobs."""

import pytest
import torch

from src.networks.txunet import NSA2DAttention, TxUNetModel


def test_nsa_attention_runtime_knobs_init():
    attn = NSA2DAttention(
        dim=48,
        num_heads=4,
        patch_size=8,
        top_n=16,
        window_size=8,
        importance_chunk_size=2048,
        importance_use_mem_get_info=False,
    )
    assert attn.importance_chunk_size == 2048
    assert attn.importance_use_mem_get_info is False


def test_nsa_attention_invalid_chunk_size():
    with pytest.raises(ValueError, match="importance_chunk_size must be > 0"):
        NSA2DAttention(
            dim=48,
            num_heads=4,
            patch_size=8,
            top_n=16,
            window_size=8,
            importance_chunk_size=0,
        )


def test_txunet_threads_nsa_runtime_knobs():
    model = TxUNetModel(
        in_ch=11,
        out_ch=1,
        base_ch=48,
        depths=(1, 1, 1, 1),
        heads=(4, 4, 8, 8),
        use_checkpoint=False,
        nsa_enabled=True,
        nsa_levels=[0, 1],
        nsa_importance_chunk_size=4096,
        nsa_importance_use_mem_get_info=False,
        rope_enabled=True,
    )
    nsa_modules = [m for m in model.modules() if isinstance(m, NSA2DAttention)]
    assert nsa_modules, "Expected at least one NSA2DAttention module."
    for mod in nsa_modules:
        assert mod.importance_chunk_size == 4096
        assert mod.importance_use_mem_get_info is False


def test_nsa_attention_forward_backward_with_fixed_chunk():
    attn = NSA2DAttention(
        dim=48,
        num_heads=4,
        patch_size=8,
        top_n=16,
        window_size=8,
        importance_chunk_size=256,
        importance_use_mem_get_info=False,
        rope_enabled=True,
    )
    x = torch.randn(1, 48, 32, 32, requires_grad=True)
    y = attn(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
