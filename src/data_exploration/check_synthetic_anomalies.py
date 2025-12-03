#!/usr/bin/env python3
import argparse
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Tuple, Dict

import numpy as np


def find_synthetic_samples(data_dir: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
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


def _safe_load_arrays(npz_path: str):
    with np.load(npz_path) as data:
        reflectance = data['reflectance'].astype(np.float32, copy=False)
        transmittance = data['transmittance'].astype(np.float32, copy=False)
        pathloss = data['pathloss'].astype(np.float32, copy=False) if 'pathloss' in data else None
        mask = data['mask'].astype(np.float32, copy=False) if 'mask' in data else None
    return reflectance, transmittance, pathloss, mask


def check_sample(npz_path: str, json_path: str, cfg: Dict) -> List[str]:
    issues: List[str] = []
    try:
        with open(json_path, 'r') as f:
            meta = json.load(f)
    except Exception as ex:
        return [f"failed to read json: {ex}"]

    try:
        reflectance, transmittance, pathloss, mask = _safe_load_arrays(npz_path)
    except Exception as ex:
        return [f"failed to read arrays: {ex}"]

    # Shape checks
    H, W = reflectance.shape
    if transmittance.shape != (H, W):
        issues.append(f"transmittance shape {transmittance.shape} != reflectance shape {(H, W)}")
    if pathloss is not None and pathloss.shape != (H, W):
        issues.append(f"pathloss shape {pathloss.shape} != reflectance shape {(H, W)}")
    if mask is not None and mask.shape != (H, W):
        issues.append(f"mask shape {mask.shape} != reflectance shape {(H, W)}")

    # Finite checks
    if not np.isfinite(reflectance).all():
        issues.append("reflectance contains NaN/Inf")
    if not np.isfinite(transmittance).all():
        issues.append("transmittance contains NaN/Inf")
    if pathloss is not None and not np.isfinite(pathloss).all():
        issues.append("pathloss contains NaN/Inf")

    # Non-zero counts (reuse across checks)
    nz_ref_count = int(np.count_nonzero(reflectance))
    nz_trans_count = int(np.count_nonzero(transmittance))

    # Empty / constant channels
    if nz_ref_count == 0:
        issues.append("reflectance is all zeros")
    if nz_trans_count == 0:
        issues.append("transmittance is all zeros")
    if float(reflectance.max() - reflectance.min()) == 0.0:
        issues.append(f"reflectance constant value={float(reflectance.min()):.3f}")
    if float(transmittance.max() - transmittance.min()) == 0.0:
        issues.append(f"transmittance constant value={float(transmittance.min()):.3f}")
    if pathloss is not None and float(pathloss.max() - pathloss.min()) == 0.0:
        issues.append(f"pathloss constant value={float(pathloss.min()):.3f}")

    # Ratio checks and non-zero range checks (zeros are common; min is often 0)
    total = float(H * W)
    nz_ref = nz_ref_count / total
    nz_trans = nz_trans_count / total
    z_ref = 1.0 - nz_ref
    z_trans = 1.0 - nz_trans
    if nz_ref < cfg['density_low'] or nz_ref > cfg['density_high']:
        issues.append(f"reflectance density extreme (nonzero={nz_ref:.4f}, zeros={z_ref:.4f})")
    if nz_trans < cfg['density_low'] or nz_trans > cfg['density_high']:
        issues.append(f"transmittance density extreme (nonzero={nz_trans:.4f}, zeros={z_trans:.4f})")

    # Non-zero value ranges only
    if nz_ref_count > 0:
        ref_gt0 = reflectance > 0
        r_nz_min = float(np.min(reflectance, where=ref_gt0, initial=np.inf))
        r_nz_max = float(np.max(reflectance, where=ref_gt0, initial=-np.inf))
        if r_nz_min < cfg['refl_nz_min_ok'] or r_nz_max > cfg['refl_nz_max_ok']:
            issues.append(f"reflectance non-zero out-of-range min={r_nz_min:.3f} max={r_nz_max:.3f}")
    if nz_trans_count > 0:
        trans_gt0 = transmittance > 0
        t_nz_min = float(np.min(transmittance, where=trans_gt0, initial=np.inf))
        t_nz_max = float(np.max(transmittance, where=trans_gt0, initial=-np.inf))
        if t_nz_min < cfg['trans_nz_min_ok'] or t_nz_max > cfg['trans_nz_max_ok']:
            issues.append(f"transmittance non-zero out-of-range min={t_nz_min:.3f} max={t_nz_max:.3f}")
    if pathloss is not None:
        p_min, p_max = float(np.nanmin(pathloss)), float(np.nanmax(pathloss))
        if p_min < cfg['pl_min_ok'] or p_max > cfg['pl_max_ok']:
            issues.append(f"pathloss out-of-range min={p_min:.3f} max={p_max:.3f}")

    # Density extremes
    # (density checks moved above with zero ratios)

    # Metadata sanity checks
    freq = meta.get('frequency_MHz', None)
    if freq is None or float(freq) <= 0:
        issues.append("missing/invalid frequency_MHz in json")

    pix = meta.get('pixel_size_m', None)
    if pix is None or not (cfg['pix_min_ok'] <= float(pix) <= cfg['pix_max_ok']):
        issues.append(f"pixel_size_m abnormal={pix}")

    ant = meta.get('antenna', {}) if isinstance(meta.get('antenna', {}), dict) else {}
    try:
        x_px = float(ant.get('x_px', np.nan))
        y_px = float(ant.get('y_px', np.nan))
        if not np.isfinite(x_px) or not np.isfinite(y_px):
            issues.append("antenna coords NaN/Inf")
        else:
            if x_px < 0 or x_px > (W - 1) or y_px < 0 or y_px > (H - 1):
                issues.append(f"antenna out-of-bounds x={x_px:.1f} y={y_px:.1f} for {H}x{W}")
    except Exception:
        issues.append("antenna coords parse error")

    # IDs structure
    ids = meta.get('ids', None)
    if not isinstance(ids, dict):
        issues.append("ids missing or wrong type (expected dict)")
    else:
        for key in ('building', 'antenna', 'frequency_index', 'sample_index'):
            if key not in ids:
                issues.append(f"ids missing '{key}'")
        try:
            fidx = int(ids.get('frequency_index', -1))
            # Synthetic is 0-based (0..2), ICASSP may be 1-based (1..3). Accept both ranges.
            if fidx < 0:
                issues.append(f"frequency_index invalid={fidx}")
        except Exception:
            issues.append("frequency_index parse error")

    # Mask sanity (if present): binary-ish and non-empty
    if mask is not None:
        if not np.isfinite(mask).all():
            issues.append("mask contains NaN/Inf")
        if mask.shape == (H, W):
            mask_nz = int(np.count_nonzero(mask))
            if mask_nz == 0:
                issues.append("mask all zeros")
            # Fast path for typical binary masks; fall back to unique() only if needed
            m_min = float(mask.min())
            m_max = float(mask.max())
            if not (m_min == m_max or (m_min in (0.0, 1.0) and m_max in (0.0, 1.0))):
                u = np.unique(mask)
                if len(u) > 4:  # very non-binary
                    issues.append(f"mask has many unique values (len={len(u)})")

    return issues


def main():
    parser = argparse.ArgumentParser("Check synthetic NPZ+JSON samples for anomalies")
    parser.add_argument('--data_dir', type=str, default='/nfs/dgx/raid/iot/mlsp_wair_d_data/synthetic_train', help='Root of synthetic dataset')
    parser.add_argument('--num', type=int, default=1000, help='Number of random samples to test (<= total)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--workers', type=int, default=0, help='Number of parallel workers (0=auto)')
    # Non-zero range thresholds (reflectance/transmittance values are >=0 with zeros common)
    parser.add_argument('--refl_nz_min_ok', type=float, default=0.0)
    parser.add_argument('--refl_nz_max_ok', type=float, default=30.0)
    parser.add_argument('--trans_nz_min_ok', type=float, default=0.0)
    parser.add_argument('--trans_nz_max_ok', type=float, default=30.0)
    parser.add_argument('--pl_min_ok', type=float, default=5.0)
    parser.add_argument('--pl_max_ok', type=float, default=2000.0)
    parser.add_argument('--density_low', type=float, default=0.002)
    parser.add_argument('--density_high', type=float, default=0.2)
    parser.add_argument('--pix_min_ok', type=float, default=0.24)
    parser.add_argument('--pix_max_ok', type=float, default=0.26)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    pairs = find_synthetic_samples(args.data_dir)
    if not pairs:
        print(f"No synthetic samples found under: {args.data_dir}")
        return

    sample_ct = min(args.num, len(pairs))
    chosen = random.sample(pairs, sample_ct)

    cfg = dict(
        refl_nz_min_ok=args.refl_nz_min_ok,
        refl_nz_max_ok=args.refl_nz_max_ok,
        trans_nz_min_ok=args.trans_nz_min_ok,
        trans_nz_max_ok=args.trans_nz_max_ok,
        pl_min_ok=args.pl_min_ok,
        pl_max_ok=args.pl_max_ok,
        density_low=args.density_low,
        density_high=args.density_high,
        pix_min_ok=args.pix_min_ok,
        pix_max_ok=args.pix_max_ok,
    )

    total_flagged = 0

    # Determine worker count
    workers = args.workers if args.workers and args.workers > 0 else (os.cpu_count() or 1)
    workers = max(1, min(workers, 32))  # sensible cap to avoid oversubscription

    if workers == 1:
        for npz_path, json_path in tqdm(chosen):
            issues = check_sample(npz_path, json_path, cfg)
            if issues:
                total_flagged += 1
                print(f"[ANOMALY] {npz_path}")
                for msg in issues:
                    print(f"  - {msg}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(check_sample, npz_path, json_path, cfg): (npz_path, json_path)
                for (npz_path, json_path) in chosen
            }
            with tqdm(total=sample_ct) as pbar:
                for fut in as_completed(futures):
                    npz_path, _ = futures[fut]
                    try:
                        issues = fut.result()
                    except Exception as ex:
                        issues = [f"worker error: {ex}"]
                    if issues:
                        total_flagged += 1
                        print(f"[ANOMALY] {npz_path}")
                        for msg in issues:
                            print(f"  - {msg}")
                    pbar.update(1)

    print(f"Checked {sample_ct} samples; flagged {total_flagged} with at least one anomaly.")


if __name__ == '__main__':
    main()


