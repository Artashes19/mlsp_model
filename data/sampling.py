from __future__ import annotations

"""Stratified k-point sampler and P-channel injection (deferred for 3-channel setup)."""

from math import ceil, sqrt
from typing import Tuple

import numpy as np


def stratified_k_points(free_mask: np.ndarray, k: int) -> np.ndarray:
    """
    Returns K unique (row, col) indices within free space.
    s = ceil(sqrt(k)); one pick per cell; fallback to global sampling if a cell is empty.
    """
    if k <= 0:
        return np.zeros((0, 2), dtype=np.int64)
    h, w = free_mask.shape[:2]
    s = int(ceil(sqrt(k)))
    rows = np.linspace(0, h, s + 1, dtype=int)
    cols = np.linspace(0, w, s + 1, dtype=int)
    picks: list[Tuple[int, int]] = []
    rng = np.random.default_rng()
    # Global pool of free pixels for fallback
    free_idxs = np.argwhere(free_mask > 0)
    used = set()
    for i in range(s):
        for j in range(s):
            r0, r1 = rows[i], rows[i + 1]
            c0, c1 = cols[j], cols[j + 1]
            cell = free_mask[r0:r1, c0:c1]
            if cell.size == 0:
                continue
            candidates = np.argwhere(cell > 0)
            if candidates.size > 0:
                rr, cc = candidates[rng.integers(0, len(candidates))]
                rr += r0
                cc += c0
                if (rr, cc) not in used:
                    picks.append((rr, cc))
                    used.add((rr, cc))
    # Fallback to reach exactly k
    if len(picks) < k and free_idxs.size > 0:
        rng.shuffle(free_idxs)
        for rr, cc in free_idxs:
            if (rr, cc) in used:
                continue
            picks.append((int(rr), int(cc)))
            used.add((int(rr), int(cc)))
            if len(picks) >= k:
                break
    return np.array(picks[:k], dtype=np.int64)


def inject_samples(P: np.ndarray, y: np.ndarray, idxs: np.ndarray) -> np.ndarray:
    """Zero P and write y at sampled indices. P.shape == y.shape == (H, W)."""
    out = np.zeros_like(P)
    for rr, cc in idxs:
        out[rr, cc] = y[rr, cc]
    return out




