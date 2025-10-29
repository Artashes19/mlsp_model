#!/usr/bin/env python3
import argparse
import json
import os
from typing import Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _resolve_sample_paths(sample_path: str) -> Tuple[str, str]:
    """
    Resolve (npz_path, json_path) from either an .npz path or a directory
    containing <name>.npz and <name>.json.
    """
    sample_path = os.path.expanduser(sample_path)
    if os.path.isfile(sample_path) and sample_path.endswith('.npz'):
        npz_path = sample_path
        stem = os.path.splitext(os.path.basename(npz_path))[0]
        json_path = os.path.join(os.path.dirname(npz_path), stem + '.json')
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON sidecar not found for {npz_path}")
        return npz_path, json_path

    if os.path.isdir(sample_path):
        # try <dir>/<name>.npz + <dir>/<name>.json where name==dirname
        dirname = os.path.basename(os.path.normpath(sample_path))
        npz_path = os.path.join(sample_path, dirname + '.npz')
        json_path = os.path.join(sample_path, dirname + '.json')
        if os.path.exists(npz_path) and os.path.exists(json_path):
            return npz_path, json_path
        # otherwise, scan for exactly one pair
        candidates = [f for f in os.listdir(sample_path) if f.endswith('.npz')]
        if not candidates:
            raise FileNotFoundError(f"No .npz found in {sample_path}")
        # prefer a .npz that has a matching .json
        for f in candidates:
            stem = os.path.splitext(f)[0]
            npz_c = os.path.join(sample_path, f)
            json_c = os.path.join(sample_path, stem + '.json')
            if os.path.exists(json_c):
                return npz_c, json_c
        # fallback to first npz even without json (will error if missing)
        stem = os.path.splitext(candidates[0])[0]
        return os.path.join(sample_path, candidates[0]), os.path.join(sample_path, stem + '.json')

    raise FileNotFoundError(f"Path not found or not a valid sample: {sample_path}")


def _load_sample(npz_path: str, json_path: str):
    with np.load(npz_path) as data:
        reflectance = data['reflectance'].astype(np.float32, copy=False)
        transmittance = data['transmittance'].astype(np.float32, copy=False)
        pathloss = data['pathloss'].astype(np.float32, copy=False) if 'pathloss' in data else None
        mask = data['mask'].astype(np.float32, copy=False) if 'mask' in data else None
    with open(json_path, 'r') as f:
        meta = json.load(f)
    return reflectance, transmittance, pathloss, mask, meta


def _print_info(npz_path: str, json_path: str, refl: np.ndarray, trans: np.ndarray, pl: np.ndarray | None, mask: np.ndarray | None, meta: dict):
    H, W = refl.shape
    freq = meta.get('frequency_MHz', None)
    pix = meta.get('pixel_size_m', None)
    ant = meta.get('antenna', {}) if isinstance(meta.get('antenna', {}), dict) else {}
    ids = meta.get('ids', {}) if isinstance(meta.get('ids', {}), dict) else {}

    print(f"Sample: {npz_path}")
    print(f"Meta: json={json_path}")
    print(f"Shape: HxW={H}x{W}")
    print(f"Frequency_MHz: {freq}")
    print(f"Pixel_size_m: {pix}")
    print(f"Antenna: x_px={ant.get('x_px')} y_px={ant.get('y_px')}")
    print(f"IDs: {ids}")

    def stats(name: str, arr: np.ndarray | None):
        if arr is None:
            print(f"{name}: <missing>")
            return
        nz = int(np.count_nonzero(arr)); total = int(arr.size)
        print(f"{name}: min={float(arr.min()):.3f} max={float(arr.max()):.3f} mean={float(arr.mean()):.3f} nz={nz}/{total} ({nz/total:.3f})")

    stats('Reflectance', refl)
    stats('Transmittance', trans)
    stats('Pathloss', pl)
    stats('Mask', mask)


def _save_figure(out_path: str, refl: np.ndarray, trans: np.ndarray, pl: np.ndarray | None, ant_x: float | None, ant_y: float | None, title: str):
    # Create subplots
    cols = 3
    fig, axes = plt.subplots(1, cols, figsize=(18, 6))
    # Reflectance
    im0 = axes[0].imshow(refl, cmap='viridis', origin='upper')
    axes[0].set_title('Reflectance')
    if ant_x is not None and ant_y is not None:
        axes[0].plot(ant_x, ant_y, marker='*', color='red', markersize=8)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    # Transmittance
    im1 = axes[1].imshow(trans, cmap='viridis', origin='upper')
    axes[1].set_title('Transmittance')
    if ant_x is not None and ant_y is not None:
        axes[1].plot(ant_x, ant_y, marker='*', color='red', markersize=8)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    # Pathloss (if present)
    if pl is not None:
        im2 = axes[2].imshow(pl, cmap='viridis', origin='upper')
        axes[2].set_title('Pathloss (dB)')
        if ant_x is not None and ant_y is not None:
            axes[2].plot(ant_x, ant_y, marker='*', color='red', markersize=8)
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    else:
        axes[2].axis('off')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser("Visualize one synthetic sample (reflectance, transmittance, pathloss)")
    parser.add_argument('--sample', type=str, required=True, help='Path to .npz file or sample directory')
    parser.add_argument('--out', type=str, default=None, help='Output image path (default: <sample_dir>/<name>_viz.png)')
    args = parser.parse_args()

    npz_path, json_path = _resolve_sample_paths(args.sample)
    refl, trans, pl, mask, meta = _load_sample(npz_path, json_path)

    ant = meta.get('antenna', {}) if isinstance(meta.get('antenna', {}), dict) else {}
    ant_x = float(ant.get('x_px')) if 'x_px' in ant else None
    ant_y = float(ant.get('y_px')) if 'y_px' in ant else None
    freq = meta.get('frequency_MHz', None)
    sample_name = os.path.splitext(os.path.basename(npz_path))[0]

    _print_info(npz_path, json_path, refl, trans, pl, mask, meta)

    if args.out is None:
        out_dir = os.path.dirname(npz_path)
        out_path = os.path.join(out_dir, f"{sample_name}_viz.png")
    else:
        out_path = os.path.expanduser(args.out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    title = f"{sample_name}  |  freq={freq} MHz"
    _save_figure(out_path, refl, trans, pl, ant_x, ant_y, title)
    print(out_path)


if __name__ == '__main__':
    main()


