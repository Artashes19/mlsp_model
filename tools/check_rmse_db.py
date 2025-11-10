from __future__ import annotations

"""Smoke test for RMSE in dB conversion.

We construct normalized targets/preds that differ by a constant delta in dB and
verify that rmse(y_hat * y_db_max, y * y_db_max) recovers that delta.
"""

import torch

from losses.l1_rmse import rmse


def run_smoke(y_db_max: float = 160.0) -> None:
    torch.manual_seed(0)

    # Use a 1x1x8x8 tensor and a constant base so RMSE equals |delta|
    base_db = 50.0
    shape = (1, 1, 8, 8)
    y_db = torch.full(shape, base_db, dtype=torch.float32)

    deltas = [0.0, 1.0, 3.5, 10.0]
    print(f"y_db_max={y_db_max}")
    for d in deltas:
        yhat_db = y_db + d
        y = y_db / y_db_max
        yhat = yhat_db / y_db_max
        # Unmasked
        val = float(rmse(yhat * y_db_max, y * y_db_max).item())
        print(f"delta={d:.2f} dB -> rmse_db={val:.6f} (expected {abs(d):.6f})")
        # Masked: mask half the pixels
        m = torch.ones_like(y)
        m[..., :, :4] = 0.0
        val_m = float(rmse(yhat * y_db_max, y * y_db_max, m).item())
        print(f"  masked half -> rmse_db={val_m:.6f} (expected {abs(d):.6f})")

    # Random base values away from boundaries to avoid clipping concerns
    y_db_rand = 20.0 + 100.0 * torch.rand(shape)
    d = 2.75
    yhat_db_rand = y_db_rand + d
    y = y_db_rand / y_db_max
    yhat = yhat_db_rand / y_db_max
    val_rand = float(rmse(yhat * y_db_max, y * y_db_max).item())
    print(f"random base, delta=2.75 dB -> rmse_db={val_rand:.6f} (expected 2.750000)")


if __name__ == "__main__":
    run_smoke()



