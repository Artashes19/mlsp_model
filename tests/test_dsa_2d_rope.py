import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ops.dsa_rope import (
    apply_partial_rope_2d_interleaved,
    apply_partial_rope_2d_non_interleaved,
    maybe_apply_rope_to_v,
    positions_from_hw,
)
from tests.helpers.dsa_reference import (
    naive_partial_rope_2d_interleaved,
    naive_partial_rope_2d_non_interleaved,
)


def test_partial_rope_leaves_nope_slice_unchanged():
    q = torch.randn(1, 2, 4, 128)
    q_out = apply_partial_rope_2d_non_interleaved(q, H=2, W=2, rope_dim=64)
    torch.testing.assert_close(q_out[..., 64:], q[..., 64:])

def test_interleaved_rope_matches_naive_reference():
    q = torch.randn(2, 3, 16, 128)
    ref = naive_partial_rope_2d_interleaved(q, H=4, W=4, rope_dim=64)
    out = apply_partial_rope_2d_interleaved(q, H=4, W=4, rope_dim=64)
    torch.testing.assert_close(out, ref)

def test_non_interleaved_rope_matches_naive_reference():
    q = torch.randn(2, 3, 16, 128)
    ref = naive_partial_rope_2d_non_interleaved(q, H=4, W=4, rope_dim=64)
    out = apply_partial_rope_2d_non_interleaved(q, H=4, W=4, rope_dim=64)
    torch.testing.assert_close(out, ref)

def test_row_major_position_mapping_matches_explicit_coordinates():
    coords = positions_from_hw(H=3, W=5)
    assert coords[7] == (1, 2)

def test_partial_rope_does_not_touch_v():
    v = torch.randn(1, 2, 4, 128)
    v_out = maybe_apply_rope_to_v(v)
    assert torch.equal(v, v_out)
