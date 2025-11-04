"""Unified evaluation/visualization entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

from data.dataset import IndoorRadioMapDataset, gather_task2_samples
from losses.l1_rmse import rmse
from train import build_model, load_yaml, split_pairs_by_hash, split_pairs_by_building


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate radio map models and optionally save visualizations")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--data", type=str, required=True, help="Path to data YAML config")
    parser.add_argument("--save_viz", action="store_true", help="Save qualitative visualizations")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model_cfg = ckpt["model_cfg"]
    data_cfg = load_yaml(args.data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_cfg).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    resize_hw = tuple(data_cfg.get("resize", [256, 256]))  # type: ignore[assignment]
    val_ratio = float(data_cfg.get("val_ratio", 0.1))
    seed = int(data_cfg.get("seed", 42))
    split_mode = str(data_cfg.get("split_mode", "hash")).lower()
    metric_scale = float(data_cfg.get("y_db_max", ckpt.get("y_db_max", 160.0)))

    all_pairs = gather_task2_samples(Path(data_cfg["data_root"]), split="train")
    if split_mode == "building":
        train_buildings = [int(b) for b in data_cfg.get("train_buildings", [])]
        val_buildings = [int(b) for b in data_cfg.get("val_buildings", [])]
        if not val_buildings:
            raise ValueError("split_mode=building requires val_buildings in data config")
        _, val_pairs = split_pairs_by_building(all_pairs, train_buildings, val_buildings)
    else:
        _, val_pairs = split_pairs_by_hash(all_pairs, val_ratio=val_ratio, seed=seed)
    ds_val = IndoorRadioMapDataset(root=data_cfg["data_root"], split="val", resize_hw=resize_hw, file_pairs=val_pairs)
    loader_val = DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=2)

    total_rmse = 0.0
    for i, (x, y, m, meta) in enumerate(loader_val):
        x = x.to(device)
        y = y.to(device)
        m = m.to(device)
        y_hat = model(x)
        y_hat = torch.clamp(y_hat, 0.0, 1.0)
        total_rmse += float(rmse(y_hat * metric_scale, y * metric_scale, m).item())
        if args.save_viz and i < 16:
            # save predicted map as PNG (8-bit)
            pred = y_hat.detach().cpu().clamp(0, 1).squeeze(0).squeeze(0).numpy()
            pred8 = (pred * 255.0).round().astype(np.uint8)
            filename = meta.get("scene_id", [f"sample_{i}"])[0]
            out_dir = Path("viz")
            out_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pred8, mode="L").save(out_dir / f"{filename}_pred.png")

    mean_rmse = total_rmse / max(1, len(loader_val))
    print(f"Val RMSE_dB: {mean_rmse:.4f} dB")


if __name__ == "__main__":
    main()


