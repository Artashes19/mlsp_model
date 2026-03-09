import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ops.dsa_rope import apply_partial_rope_2d_non_interleaved, maybe_apply_rope_to_v


def test_partial_rope_leaves_nope_slice_unchanged():
    q = torch.randn(1, 2, 4, 128)
    q_out = apply_partial_rope_2d_non_interleaved(q, H=2, W=2, rope_dim=64)
    assert torch.equal(q[..., :64], q_out[..., :64])


def test_partial_rope_does_not_touch_v():
    v = torch.randn(1, 2, 4, 128)
    v_out = maybe_apply_rope_to_v(v)
    assert torch.equal(v, v_out)
