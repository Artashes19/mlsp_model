from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch

# Allow running as a script from anywhere
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import (
    IndoorRadioMapDataset,
    gather_task2_samples,
    find_task2_paths,
    list_pngs,
)


def assert_no_nan_inf(name: str, arr: np.ndarray) -> None:
    if not np.isfinite(arr).all():
        bad = np.logical_not(np.isfinite(arr))
        raise AssertionError(f"{name} has NaN/Inf at {bad.sum()} positions")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check a batch of samples for shapes, ranges, pairing, and channel heuristics.")
    parser.add_argument("--root", type=str, required=True, help="Dataset root, e.g., /auto/home/artashes/data")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Split to check")
    parser.add_argument("--num", type=int, default=64, help="Number of samples to scan")
    parser.add_argument("--resize", type=int, nargs=2, default=[256, 256], help="Resize (H W)")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()

    # 1:1 pairing check (train/val only)
    if args.split in ("train", "val"):
        paths = find_task2_paths(root)
        inputs = list_pngs(paths["train_inputs"]) if paths["train_inputs"] else []
        outputs = list_pngs(paths["train_outputs"]) if paths["train_outputs"] else []
        in_set = {p.stem for p in inputs}
        out_set = {p.stem for p in outputs}
        missing_in = sorted(list(out_set - in_set))
        missing_out = sorted(list(in_set - out_set))
        if missing_in:
            raise AssertionError(f"Outputs without inputs: {len(missing_in)} e.g., {missing_in[:5]}")
        if missing_out:
            raise AssertionError(f"Inputs without outputs: {len(missing_out)} e.g., {missing_out[:5]}")
        print(f"Pairing OK: {len(in_set)} input/output pairs")

    # Dataset and sampling
    resize_hw = (int(args.resize[0]), int(args.resize[1]))
    all_pairs = gather_task2_samples(root, split=args.split)
    if len(all_pairs) == 0:
        raise RuntimeError("No samples found for the requested split")
    stride = max(1, len(all_pairs) // max(1, args.num))
    pairs = all_pairs[::stride][: args.num]

    ds = IndoorRadioMapDataset(root=root, split=args.split, resize_hw=resize_hw, file_pairs=pairs)

    # Stats accumulators
    x_min = np.full(3, np.inf, dtype=np.float64)
    x_max = np.full(3, -np.inf, dtype=np.float64)
    x_mean = np.zeros(3, dtype=np.float64)
    x_var = np.zeros(3, dtype=np.float64)
    y_min = np.inf
    y_max = -np.inf
    y_mean = 0.0
    y_var = 0.0
    n_pixels = 0
    n_samples = 0
    ch2_widest_count = 0

    H, W = resize_hw

    for i in range(len(ds)):
        x, y, m, meta = ds[i]
        # Shapes
        assert tuple(x.shape) == (3, H, W), f"X shape mismatch at {i}: {tuple(x.shape)}"
        assert tuple(y.shape) == (1, H, W), f"y shape mismatch at {i}: {tuple(y.shape)}"
        assert tuple(m.shape) == (1, H, W), f"mask shape mismatch at {i}: {tuple(m.shape)}"

        # Ranges and NaN/Inf checks
        xn = x.detach().cpu().numpy()
        yn = y.detach().cpu().numpy()
        assert_no_nan_inf("X", xn)
        assert_no_nan_inf("y", yn)
        if not ((xn >= 0.0).all() and (xn <= 1.0).all()):
            raise AssertionError(f"X outside [0,1] at sample {i}")
        if not ((yn >= 0.0).all() and (yn <= 1.0).all()):
            raise AssertionError(f"y outside [0,1] at sample {i}")

        # Channel heuristic: ch2 (D) has widest dynamic range
        ranges = [float(xn[c].max() - xn[c].min()) for c in range(3)]
        if ranges[2] >= max(ranges[0], ranges[1]):
            ch2_widest_count += 1

        # Accumulate stats
        for c in range(3):
            xc = xn[c]
            x_min[c] = min(x_min[c], float(xc.min()))
            x_max[c] = max(x_max[c], float(xc.max()))
            x_mean[c] += float(xc.mean())
            x_var[c] += float(((xc - xc.mean()) ** 2).mean())
        y_min = min(y_min, float(yn.min()))
        y_max = max(y_max, float(yn.max()))
        y_mean += float(yn.mean())
        y_var += float(((yn - yn.mean()) ** 2).mean())
        n_samples += 1

    # Finalize stats
    x_mean /= max(1, n_samples)
    x_std = np.sqrt(x_var / max(1, n_samples))
    y_mean /= max(1, n_samples)
    y_std = float(np.sqrt(y_var / max(1, n_samples)))
    ch2_ratio = ch2_widest_count / max(1, n_samples)

    print("\nDataset checks passed:")
    print(f"- Samples checked: {n_samples}")
    print(f"- Resize: {H}x{W}")
    print(f"- X channel mins: {x_min}")
    print(f"- X channel maxs: {x_max}")
    print(f"- X channel means: {x_mean}")
    print(f"- X channel stds:  {x_std}")
    print(f"- y min/max/mean/std: {y_min:.4f} / {y_max:.4f} / {y_mean:.4f} / {y_std:.4f}")
    print(f"- Heuristic D widest range ratio (should be high): {ch2_ratio:.3f}")


if __name__ == "__main__":
    main()


