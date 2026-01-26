#!/usr/bin/env python3
"""
Minimal feature inspection:
1) Load 10 samples from a manifest
2) Run through full dataset pipeline (including featurization)
3) Save stacked input tensor as a NumPy matrix
4) Compute stats and make plots
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))

from src.datamodules.indoor import IndoorDatamodule
from src.datamodules.datasets.indoor import PathlossDataset
from src.utils.indoor.config_overrides import get_config
from src.utils.indoor.featurizer import FREQ_VALUES, get_num_channels


def parse_args() -> argparse.Namespace:
    config = get_config()
    parser = argparse.ArgumentParser(description="Inspect MLSP feature tensors.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="",
        help="Path to ICASSP manifest CSV (e.g., icassp_val_21_22_23_24_25.csv).",
    )
    parser.add_argument(
        "--synthetic-manifest",
        type=str,
        default="",
        help="Path to synthetic manifest CSV (e.g., synthetic_train.csv).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/feature_inspect",
        help="Output directory for numpy tensor, stats, and plots.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to process.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for deterministic sampling.",
    )
    parser.add_argument(
        "--freq-mhz",
        type=float,
        default=868.0,
        help="Frequency in MHz to filter manifest rows.",
    )
    parser.add_argument(
        "--freq-idx",
        type=int,
        default=1,
        help="Frequency index (1-based) to filter manifest rows.",
    )
    parser.add_argument(
        "--all-freqs",
        action="store_true",
        help="Include all frequencies from the manifest.",
    )
    parser.add_argument(
        "--channels",
        type=str,
        default=config.channels,
        help="Channel string, e.g., rtdgfmps (11ch with Fourier freq) or rtd (3ch).",
    )
    parser.add_argument(
        "--sparse-range",
        type=float,
        nargs=2,
        default=(0.005, 0.01),
        metavar=("LOW", "HIGH"),
        help="Uniform range for sparse sampling ratio.",
    )
    parser.add_argument(
        "--modality-dropout-prob",
        type=float,
        default=0.0,
        help="Probability of dropping one modality.",
    )
    parser.add_argument(
        "--sparse-dropout-given-dropout",
        type=float,
        default=0.5,
        help="If dropout happens, probability that sparse is dropped.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs_list = []
    if args.synthetic_manifest:
        inputs_list = IndoorDatamodule.get_inputs_list(
            freqs_mhz=[],
            freqs=[],
            manifest_path=args.synthetic_manifest,
        )
        inputs_list = sorted(inputs_list, key=lambda x: x["file_name"])[: args.num_samples]
    else:
        if not args.manifest:
            raise ValueError("--manifest is required when --synthetic-manifest is not set.")
        if args.all_freqs:
            freq_values = set()
            with open(args.manifest, "r", newline="") as fp:
                reader = csv.DictReader(fp)
                for row in reader:
                    freq_val = row.get("freq_MHz") or row.get("frequency_MHz")
                    if freq_val not in (None, ""):
                        freq_values.add(float(freq_val))
            if not freq_values:
                raise ValueError("No freq_MHz values found in manifest.")
            freqs_mhz = sorted(freq_values)
            freqs = []
        else:
            freqs_mhz = [args.freq_mhz]
            freqs = [args.freq_idx]
        inputs_list = IndoorDatamodule.get_inputs_list(
            freqs_mhz=freqs_mhz,
            freqs=freqs,
            manifest_path=args.manifest,
        )
        inputs_list = sorted(inputs_list, key=lambda x: x["file_name"])[: args.num_samples]

    dataset = PathlossDataset(
        inputs_list=inputs_list,
        training=False,
        mlsp_task1=False,
        mlsp_task_idx=1,
        task_idx=1,
        pl_clip=None,
        use_transmittance_loss=False,
        inference=True,
        reps_per_epoch=1,
        augment_val=False,
        augmentations=None,
        sparse_range=list(args.sparse_range),
        modality_dropout_prob=args.modality_dropout_prob,
        sparse_dropout_given_dropout=args.sparse_dropout_given_dropout,
        channels=args.channels,
    )

    tensors = []
    file_names = []
    for i in range(min(len(dataset), args.num_samples)):
        input_tensor, _, _, meta = dataset[i]
        tensors.append(input_tensor)
        file_names.append(meta["file_name"])

    stacked = torch.stack(tensors, dim=0)  # (N, C, H, W)
    stacked_np = stacked.cpu().numpy()
    np.save(out_dir / "features.npy", stacked_np)

    with open(out_dir / "samples.csv", "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["index", "file_name"])
        for i, name in enumerate(file_names):
            writer.writerow([i, name])

    flat = stacked_np.reshape(stacked_np.shape[0], stacked_np.shape[1], -1)
    mean = flat.mean(axis=(0, 2))
    std = flat.std(axis=(0, 2))
    vmin = flat.min(axis=(0, 2))
    vmax = flat.max(axis=(0, 2))

    channel_labels = []
    for ch in args.channels:
        if ch == "f":
            channel_labels.extend([f"f{idx+1}" for idx in range(len(FREQ_VALUES))])
        else:
            channel_labels.append(ch)
    if len(channel_labels) != get_num_channels(args.channels):
        raise ValueError("Channel labels do not match tensor channels.")

    with open(out_dir / "stats.csv", "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["channel_idx", "channel_label", "mean", "std", "min", "max"])
        for idx, label in enumerate(channel_labels):
            writer.writerow([idx, label, float(mean[idx]), float(std[idx]), float(vmin[idx]), float(vmax[idx])])

    with open(out_dir / "stats_nonzero.csv", "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["channel_idx", "channel_label", "nonzero_mean", "nonzero_std", "nonzero_min", "nonzero_max", "nonzero_count"])
        for idx, label in enumerate(channel_labels):
            channel_flat = flat[:, idx, :].ravel()
            nz_mask = channel_flat != 0
            nz_count = int(nz_mask.sum())
            if nz_count == 0:
                writer.writerow([idx, label, float("nan"), float("nan"), float("nan"), float("nan"), nz_count])
            else:
                nz_vals = channel_flat[nz_mask]
                writer.writerow([
                    idx,
                    label,
                    float(nz_vals.mean()),
                    float(nz_vals.std()),
                    float(nz_vals.min()),
                    float(nz_vals.max()),
                    nz_count,
                ])

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean_maps = stacked_np.mean(axis=0)
    for idx, label in enumerate(channel_labels):
        plt.figure(figsize=(5, 4))
        plt.imshow(mean_maps[idx], cmap="viridis")
        plt.colorbar()
        plt.title(f"Mean map: channel {idx} ({label})")
        plt.tight_layout()
        plt.savefig(out_dir / f"mean_map_ch{idx}_{label}.png", dpi=150)
        plt.close()

        hist_vals = flat[:, idx, :].ravel()
        if np.isfinite(hist_vals).any():
            finite_vals = hist_vals[np.isfinite(hist_vals)]
            vmin = float(np.min(finite_vals))
            vmax = float(np.max(finite_vals))
            if np.isfinite(vmin) and np.isfinite(vmax) and (vmax - vmin) >= 1e-6:
                plt.figure(figsize=(5, 4))
                plt.hist(finite_vals, bins=100)
                plt.title(f"Histogram: channel {idx} ({label})")
                plt.tight_layout()
                plt.savefig(out_dir / f"hist_ch{idx}_{label}.png", dpi=150)
                plt.close()


if __name__ == "__main__":
    main()
