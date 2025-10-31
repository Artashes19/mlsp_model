import argparse
import csv
import json
import os
from typing import List, Sequence
from tqdm import tqdm


def parse_freqs_mhz(freqs_str: str) -> List[float]:
    parts = [p.strip() for p in freqs_str.split(',') if p.strip()]
    return [float(p) for p in parts]


def nearest_freq_index(freq_mhz: float, freqs_mhz: List[float]) -> int:
    diffs = [abs(freq_mhz - f) for f in freqs_mhz]
    nearest = min(range(len(diffs)), key=lambda i: diffs[i])
    return 1 + nearest  # 1-based


def generate_manifest(root: str, out: str, freqs_mhz: Sequence[float]) -> int:
    rows = []
    for dirpath, dirnames, filenames in tqdm(os.walk(root)):
        npz_files = [f for f in filenames if f.endswith('.npz')]
        for npz in npz_files:
            sample_name = os.path.splitext(npz)[0]
            npz_path = os.path.join(dirpath, npz)
            json_path = os.path.join(dirpath, sample_name + '.json')
            if not os.path.exists(json_path):
                continue
            try:
                with open(json_path, 'r') as fp:
                    meta = json.load(fp)
            except Exception:
                continue
            ids = meta.get('ids', {}) if isinstance(meta, dict) else {}
            try:
                b = int(ids.get('building', 0))
            except Exception:
                b = 0
            try:
                ant = int(ids.get('antenna', 0))
            except Exception:
                ant = 0
            # frequency index: prefer explicit, else map from frequency_MHz
            freq_idx = ids.get('frequency_index')
            if freq_idx is not None:
                try:
                    freq_idx = int(freq_idx)
                    if freq_idx in (0, 1, 2):
                        freq_idx = freq_idx + 1
                except Exception:
                    freq_idx = None
            if freq_idx is None:
                try:
                    fmhz = float(meta.get('frequency_MHz'))
                    freq_idx = nearest_freq_index(fmhz, list(freqs_mhz))
                except Exception:
                    freq_idx = 1
            try:
                sp = int(ids.get('sample_index', 0))
            except Exception:
                sp = 0
            row = {
                'file_name': sample_name,
                'npz_file': npz_path,
                'json_file': json_path,
                'building': b,
                'antenna': ant,
                'freq_idx': freq_idx,
                'sample_index': sp,
            }
            # optional: store frequency_MHz if present
            if isinstance(meta, dict) and 'frequency_MHz' in meta:
                try:
                    row['frequency_MHz'] = float(meta['frequency_MHz'])
                except Exception:
                    pass
            rows.append(row)

    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'w', newline='') as csvfile:
        fieldnames = ['file_name', 'npz_file', 'json_file', 'building', 'antenna', 'freq_idx', 'sample_index', 'frequency_MHz']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate samples.csv manifest for synthetic dataset")
    parser.add_argument('--root', required=True, help='Root directory of the synthetic dataset')
    parser.add_argument('--out', required=True, help='Output CSV path (e.g., /path/to/samples.csv)')
    parser.add_argument('--freqs-mhz', default='868,1800,3500', help='Comma-separated list of known freqs (MHz)')
    args = parser.parse_args()

    freqs_mhz = parse_freqs_mhz(args.freqs_mhz)
    n = generate_manifest(args.root, args.out, freqs_mhz)
    print(f"Wrote {n} rows to {args.out}")


if __name__ == '__main__':
    main()
