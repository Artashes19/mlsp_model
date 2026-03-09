
def test_dsa_test_scaffold_exists():
    from tests.helpers import dsa_reference

    assert hasattr(dsa_reference, "__file__")


def test_fp8_test_scaffold_exists():
    from tests.helpers import fp8_reference

    assert hasattr(fp8_reference, "__file__")
