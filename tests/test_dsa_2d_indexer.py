
import torch


def test_dsa_test_scaffold_exists():
    from tests.helpers import dsa_reference

    assert hasattr(dsa_reference, "__file__")


def test_fp8_test_scaffold_exists():
    from tests.helpers import fp8_reference

    assert hasattr(fp8_reference, "__file__")


def test_fwht_matches_naive_reference():
    from src.ops import dsa_indexer
    from tests.helpers import fp8_reference

    x = torch.randn(2, 8)
    ref = fp8_reference.naive_fwht(x)
    out = dsa_indexer.fwht_last_dim(x)

    torch.testing.assert_close(out, ref)


def test_weighted_relu_index_score_matches_naive_reference():
    from src.ops import dsa_indexer
    from tests.helpers import fp8_reference

    q = torch.randn(1, 2, 4, 128)
    k = torch.randn(1, 2, 4, 128)
    w = torch.randn(1, 4, 2)

    ref = fp8_reference.naive_weighted_relu_index(q, k, w)
    out = dsa_indexer.weighted_relu_index_score(q, k, w)

    torch.testing.assert_close(out, ref)
