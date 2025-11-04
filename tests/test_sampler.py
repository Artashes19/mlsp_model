import numpy as np

from data.sampling import stratified_k_points


def test_stratified_k_points_basic():
    h, w = 32, 32
    free = np.ones((h, w), dtype=np.uint8)
    k = 25
    idxs = stratified_k_points(free, k)
    assert idxs.shape == (k, 2)
    # uniqueness and inside bounds
    assert len({(int(r), int(c)) for r, c in idxs}) == k
    assert np.all(idxs[:, 0] >= 0) and np.all(idxs[:, 0] < h)
    assert np.all(idxs[:, 1] >= 0) and np.all(idxs[:, 1] < w)




