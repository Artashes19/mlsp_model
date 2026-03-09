
def test_dsa_test_scaffold_exists():
    from tests.helpers import dsa_reference

    assert hasattr(dsa_reference, "__file__")
