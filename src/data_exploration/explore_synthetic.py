#!/usr/bin/env python3
import argparse
import json
import os
import random
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def find_synthetic_samples(data_dir: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if not f.endswith('.npz'):
                continue
            npz_path = os.path.join(root, f)
            stem = os.path.splitext(f)[0]
            json_path = os.path.join(root, stem + '.json')
            if os.path.exists(json_path):
                pairs.append((npz_path, json_path))
    return pairs


def ensure_out_dir(path: str) -> str:
    path = os.path.expanduser(path)
    os.makedirs(path, exist_ok=True)
    return path


def _safe_load_arrays(npz_path: str):
    with np.load(npz_path) as data:
        reflectance = data['reflectance'].astype(np.float32, copy=False)
        transmittance = data['transmittance'].astype(np.float32, copy=False)
        mask = data['mask'].astype(np.uint8, copy=False) if 'mask' in data else None
    return reflectance, transmittance, mask


def aggregate_hist(hist_counts: np.ndarray | None, bin_edges: np.ndarray | None, values: np.ndarray, bins: int | np.ndarray):
    counts, edges = np.histogram(values, bins=bins)
    if hist_counts is None:
        return counts.astype(np.int64), edges
    hist_counts += counts
    return hist_counts, edges


def main():
    parser = argparse.ArgumentParser("Explore synthetic NPZ+JSON samples and save visualizations")
    parser.add_argument('--data_dir', type=str, default='/nfs/dgx/raid/iot/mlsp_wair_d_data/synthetic_train', help='Root of synthetic dataset')
    parser.add_argument('--num', type=int, default=1000, help='Number of random samples to analyze')
    parser.add_argument('--out_dir', type=str, default='~/synthetic_samples_exploration/', help='Where to save PNGs')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = ensure_out_dir(args.out_dir)
    pairs = find_synthetic_samples(args.data_dir)
    if not pairs:
        print(f"No synthetic samples found under: {args.data_dir}")
        return

    sample_ct = min(args.num, len(pairs))
    chosen = random.sample(pairs, sample_ct)

    ref_bins = np.linspace(0, 15, 16 + 1)
    trans_bins = np.linspace(0, 15, 16 + 1)
    ref_hist = None; ref_edges = None
    trans_hist = None; trans_edges = None

    nonzero_density_ref = []
    nonzero_density_trans = []
    size_counter: Counter[tuple[int, int]] = Counter()
    freq_counter: Counter[int] = Counter()
    heat_h = heat_w = 100
    antenna_heat = np.zeros((heat_h, heat_w), dtype=np.int64)

    for npz_path, json_path in chosen:
        try:
            with open(json_path, 'r') as f:
                meta = json.load(f)
        except Exception:
            continue

        try:
            reflectance, transmittance, _ = _safe_load_arrays(npz_path)
        except Exception:
            continue

        H, W = reflectance.shape
        size_counter[(H, W)] += 1

        ref_hist, ref_edges = aggregate_hist(ref_hist, ref_edges, reflectance.ravel(), ref_bins)
        trans_hist, trans_edges = aggregate_hist(trans_hist, trans_edges, transmittance.ravel(), trans_bins)

        total = reflectance.size
        nonzero_density_ref.append(float(np.count_nonzero(reflectance)) / float(total))
        nonzero_density_trans.append(float(np.count_nonzero(transmittance)) / float(total))

        try:
            freq_mhz = int(meta.get('frequency_MHz', 0))
            if freq_mhz > 0:
                freq_counter[freq_mhz] += 1
        except Exception:
            pass

        try:
            ant = meta.get('antenna', {})
            x_px = float(ant.get('x_px', 0))
            y_px = float(ant.get('y_px', 0))
            gx = min(heat_w - 1, max(0, int(round((x_px / max(1, W - 1)) * (heat_w - 1)))))
            gy = min(heat_h - 1, max(0, int(round((y_px / max(1, H - 1)) * (heat_h - 1)))))
            antenna_heat[gy, gx] += 1
        except Exception:
            pass

    saved_paths: list[str] = []

    try:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        if ref_hist is not None:
            centers = 0.5 * (ref_edges[:-1] + ref_edges[1:])
            ax[0].bar(centers, ref_hist, width=(ref_edges[1] - ref_edges[0]))
        ax[0].set_title('Reflectance distribution (aggregated)')
        ax[0].set_xlabel('Value')
        ax[0].set_ylabel('Count')
        if trans_hist is not None:
            centers = 0.5 * (trans_edges[:-1] + trans_edges[1:])
            ax[1].bar(centers, trans_hist, width=(trans_edges[1] - trans_edges[0]))
        ax[1].set_title('Transmittance distribution (aggregated)')
        ax[1].set_xlabel('Value')
        ax[1].set_ylabel('Count')
        fig.tight_layout()
        p = os.path.join(out_dir, 'hist_reflectance_transmittance.png')
        fig.savefig(p, dpi=150)
        plt.close(fig)
        saved_paths.append(p)
    except Exception:
        pass

    try:
        sizes, counts = zip(*size_counter.most_common(20)) if size_counter else ([], [])
        labels = [f"{h}x{w}" for h, w in sizes]
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(range(len(counts)), counts)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_xlabel('HxW')
        ax.set_ylabel('Count (top 20)')
        ax.set_title('Size distribution (top 20)')
        fig.tight_layout()
        p = os.path.join(out_dir, 'size_distribution_top20.png')
        fig.savefig(p, dpi=150)
        plt.close(fig)
        saved_paths.append(p)
    except Exception:
        pass

    try:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].hist(nonzero_density_ref, bins=30, range=(0, 1))
        ax[0].set_title('Reflectance non-zero density')
        ax[0].set_xlabel('Fraction non-zero')
        ax[0].set_ylabel('Count')
        ax[1].hist(nonzero_density_trans, bins=30, range=(0, 1))
        ax[1].set_title('Transmittance non-zero density')
        ax[1].set_xlabel('Fraction non-zero')
        ax[1].set_ylabel('Count')
        fig.tight_layout()
        p = os.path.join(out_dir, 'nonzero_density_histograms.png')
        fig.savefig(p, dpi=150)
        plt.close(fig)
        saved_paths.append(p)
    except Exception:
        pass

    try:
        items = sorted(freq_counter.items())
        if items:
            freqs, counts = zip(*items)
        else:
            freqs, counts = [], []
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar([str(f) for f in freqs], counts)
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('Count')
        ax.set_title('Frequency distribution')
        fig.tight_layout()
        p = os.path.join(out_dir, 'frequency_distribution.png')
        fig.savefig(p, dpi=150)
        plt.close(fig)
        saved_paths.append(p)
    except Exception:
        pass

    try:
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(antenna_heat, cmap='hot', origin='lower', interpolation='nearest')
        ax.set_title('Antenna locations heatmap (normalized grid)')
        ax.set_xlabel('X (normalized bins)')
        ax.set_ylabel('Y (normalized bins)')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        p = os.path.join(out_dir, 'antenna_locations_heatmap.png')
        fig.savefig(p, dpi=150)
        plt.close(fig)
        saved_paths.append(p)
    except Exception:
        pass

    for p in saved_paths:
        print(p)


if __name__ == '__main__':
    main()


