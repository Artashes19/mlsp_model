from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

from data.dataset import IndoorRadioMapDataset, gather_task2_samples
from models.radio_unet_tx.unet import TxUNet
from models.restormer_port.wrapper import RestormerRadio


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Forward-only smoke test for TxUNet/Restormer on a mini-batch")
    parser.add_argument("--root", type=str, default="/auto/home/artashes/data", help="Dataset root")
    parser.add_argument("--model", type=str, choices=["txunet", "restormer"], required=True)
    parser.add_argument("--num", type=int, default=8, help="Number of samples")
    parser.add_argument("--resize", type=int, nargs=2, default=[256, 256], help="Resize (H W)")
    parser.add_argument("--out", type=str, default="smoke_preds", help="Directory to save a few predictions")
    parser.add_argument("--train_mode", action="store_true", help="Use train() mode (BN uses batch stats) for debug")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.root).expanduser().resolve()
    resize_hw: Tuple[int, int] = (int(args.resize[0]), int(args.resize[1]))

    # Dataset
    pairs = gather_task2_samples(root, split="train")
    if len(pairs) == 0:
        raise RuntimeError("No train samples found")
    ds = IndoorRadioMapDataset(root=root, split="train", resize_hw=resize_hw)
    dl = DataLoader(ds, batch_size=min(args.num, 4), shuffle=False, num_workers=2)

    # Model
    if args.model == "txunet":
        model = TxUNet(in_ch=3, out_ch=1, base_ch=32, depths=(1, 1, 1, 1), heads=(1, 1, 1, 1))
    else:
        model = RestormerRadio(in_ch=3, out_ch=1, base_ch=32, heads=(1, 1, 1, 1), refinement_blocks=1)
    model = model.to(device)
    model = model.train() if args.train_mode else model.eval()

    # Run one batch
    batch = next(iter(dl))
    x, y, m, meta = batch
    x = x.to(device)
    y_hat_raw = model(x).detach().cpu()
    y_hat = y_hat_raw.clamp(0, 1)
    print(f"Input shape:   {tuple(x.shape)}")
    print(f"Output shape:  {tuple(y_hat.shape)}")
    print(f"Pred raw min/max: [{float(y_hat_raw.min()):.6f}, {float(y_hat_raw.max()):.6f}] | clamped: [{float(y_hat.min()):.6f}, {float(y_hat.max()):.6f}]")

    # Save a few preds
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_n = min(y_hat.shape[0], 4)
    for i in range(save_n):
        pred = y_hat[i, 0].numpy()
        pred8 = (pred * 255.0).round().astype(np.uint8)
        scene_id = meta["scene_id"][i] if isinstance(meta["scene_id"], list) else meta.get("scene_id", f"sample_{i}")
        Image.fromarray(pred8, mode="L").save(out_dir / f"{scene_id}_pred.png")
        # Also save auto-contrast version for visibility
        p = y_hat_raw[i, 0].numpy()
        lo, hi = float(np.percentile(p, 1.0)), float(np.percentile(p, 99.0))
        if hi <= lo + 1e-8:
            p_v = np.zeros_like(p, dtype=np.float32)
        else:
            p_v = (p - lo) / (hi - lo)
        p8 = (np.clip(p_v, 0, 1) * 255.0).round().astype(np.uint8)
        Image.fromarray(p8, mode="L").save(out_dir / f"{scene_id}_pred_autocontrast.png")
    print(f"Saved {save_n} predictions to {str(out_dir)}")


if __name__ == "__main__":
    main()


