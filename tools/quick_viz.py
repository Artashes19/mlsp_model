from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
import torch

from data.dataset import IndoorRadioMapDataset, gather_task2_samples


def to_uint8(img_01: np.ndarray) -> np.ndarray:
    img = np.clip(img_01, 0.0, 1.0)
    return (img * 255.0).round().astype(np.uint8)


def auto_contrast01(x: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    # Per-image contrast stretch for visualization only
    lo = float(np.percentile(x, p_low))
    hi = float(np.percentile(x, p_high))
    if hi <= lo + 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def make_grid_2x2(r: np.ndarray, t: np.ndarray, d: np.ndarray, y: np.ndarray) -> Image.Image:
    # Each input is [H,W] in [0,1]
    # Apply auto-contrast to highlight low-dynamic-range channels (R,T)
    r_v = auto_contrast01(r)
    t_v = auto_contrast01(t)
    d_v = auto_contrast01(d)
    y_v = auto_contrast01(y)
    r8, t8, d8, y8 = map(to_uint8, [r_v, t_v, d_v, y_v])
    H, W = r8.shape
    canvas = Image.new("L", (2 * W, 2 * H), color=0)
    canvas.paste(Image.fromarray(r8, mode="L"), (0, 0))
    canvas.paste(Image.fromarray(t8, mode="L"), (W, 0))
    canvas.paste(Image.fromarray(d8, mode="L"), (0, H))
    canvas.paste(Image.fromarray(y8, mode="L"), (W, H))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Save quick 2x2 grids (R,T,D,y) for random samples")
    parser.add_argument("--root", type=str, required=True, help="Dataset root, e.g., /auto/home/artashes/data")
    parser.add_argument("--out", type=str, default="viz_quick", help="Output directory")
    parser.add_argument("--num", type=int, default=8, help="Number of samples")
    parser.add_argument("--resize", type=int, nargs=2, default=[256, 256], help="Resize (H W)")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    resize_hw: Tuple[int, int] = (int(args.resize[0]), int(args.resize[1]))
    pairs = gather_task2_samples(root, split="train")
    if len(pairs) == 0:
        raise RuntimeError("No train samples found")

    # sample without replacement up to len(pairs)
    idxs = list(range(len(pairs)))
    random.shuffle(idxs)
    idxs = idxs[: args.num]

    ds = IndoorRadioMapDataset(root=root, split="train", resize_hw=resize_hw)

    for i, idx in enumerate(idxs):
        x, y, m, meta = ds[idx]
        # x: [3,H,W] -> HWC
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()[0]
        r, t, d = x_np[0], x_np[1], x_np[2]
        grid = make_grid_2x2(r, t, d, y_np)
        scene_id = meta.get("scene_id", f"sample_{idx}")
        grid.save(out_dir / f"{scene_id}.png")


if __name__ == "__main__":
    main()
