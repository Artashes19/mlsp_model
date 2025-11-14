import argparse
import csv
import hashlib
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

    # Write/update a companion meta file with a dataset signature for cache invalidation
    meta_path = out + '.meta.json'
    try:
        sig = compute_dataset_signature(root)
        meta = {
            'root': os.path.abspath(root),
            'num_pairs': sig['num_pairs'],
            'sha256': sig['sha256'],
        }
        with open(meta_path, 'w') as fp:
            json.dump(meta, fp, indent=2, sort_keys=True)
    except Exception:
        # Do not fail manifest creation if meta write fails
        pass

    return len(rows)


def compute_dataset_signature(root: str) -> dict:
    """Compute a fast signature of the dataset contents based on relative paths,
    file sizes, and mtimes of paired npz+json files. Avoids hashing file bytes.
    """
    entries: List[tuple[str, int, int, int, int]] = []
    root_abs = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root_abs):
        # Collect candidate basenames from npz files only; require json to exist
        for f in filenames:
            if not f.endswith('.npz'):
                continue
            base = os.path.splitext(f)[0]
            npz_path = os.path.join(dirpath, f)
            json_path = os.path.join(dirpath, base + '.json')
            if not os.path.exists(json_path):
                continue
            try:
                npz_stat = os.stat(npz_path)
                json_stat = os.stat(json_path)
            except FileNotFoundError:
                continue
            rel = os.path.relpath(os.path.join(dirpath, base), root_abs)
            entries.append(
                (
                    rel.replace('\\', '/'),
                    int(npz_stat.st_size),
                    int(npz_stat.st_mtime),
                    int(json_stat.st_size),
                    int(json_stat.st_mtime),
                )
            )
    # Deterministic order
    entries.sort(key=lambda x: x[0])
    h = hashlib.sha256()
    total = 0
    for rel, ns, nm, js, jm in entries:
        line = f"{rel}|{ns}|{nm}|{js}|{jm}\n".encode('utf-8')
        h.update(line)
        total += 1
    return {'root': root_abs, 'num_pairs': total, 'sha256': h.hexdigest()}


def ensure_manifest(root: str, out: str, freqs_mhz: Sequence[float]) -> int:
    """Ensure a fresh manifest exists for the dataset.
    - If the manifest or its meta is missing, rebuild it.
    - If the signature differs from the current dataset state, rebuild it.
    - Otherwise, leave it as is.
    Returns the number of rows (if rebuilt) or -1 if left unchanged.
    """
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    meta_path = out + '.meta.json'
    try:
        current_sig = compute_dataset_signature(root)
    except Exception:
        # If signature computation fails, force rebuild
        return generate_manifest(root, out, freqs_mhz)

    need_rebuild = not os.path.exists(out) or not os.path.exists(meta_path)
    if not need_rebuild:
        try:
            with open(meta_path, 'r') as fp:
                meta = json.load(fp)
            if (
                str(meta.get('root')) != str(current_sig.get('root')) or
                str(meta.get('sha256')) != str(current_sig.get('sha256')) or
                int(meta.get('num_pairs', -1)) != int(current_sig.get('num_pairs', -2))
            ):
                need_rebuild = True
        except Exception:
            need_rebuild = True

    if need_rebuild:
        return generate_manifest(root, out, freqs_mhz)
    return -1


# ===== ICASSP MANIFEST (general, signature-checked) =====

def _icassp_expected_paths(root: str, task: str, b: int, ant: int, f: int, sp: int):
    input_dir = os.path.join(root, f"Inputs/{task}")
    output_dir = os.path.join(root, f"Outputs/{task}")
    positions_dir = os.path.join(root, "Positions/")
    radiation_patterns_dir = os.path.join(root, "Radiation_Patterns/")
    input_file = os.path.join(input_dir, f"B{b}_Ant{ant}_f{f}_S{sp}.png")
    output_file = os.path.join(output_dir, f"B{b}_Ant{ant}_f{f}_S{sp}.png")
    position_file = os.path.join(positions_dir, f"Positions_B{b}_Ant{ant}_f{f}.csv")
    radiation_pattern_file = os.path.join(radiation_patterns_dir, f"Ant{ant}_Pattern.csv")
    return input_file, output_file, position_file, radiation_pattern_file


def compute_icassp_signature(root: str) -> dict:
    """Compute a simple signature of the ICASSP dataset layout."""
    root_abs = os.path.abspath(os.path.expanduser(root))
    h = hashlib.sha256()
    total = 0
    for dirpath, dirnames, filenames in os.walk(root_abs):
        dirnames.sort()
        filenames.sort()
        h.update(dirpath.encode("utf-8"))
        for f in filenames:
            p = os.path.join(dirpath, f)
            try:
                st = os.stat(p)
            except Exception:
                continue
            h.update(f.encode("utf-8"))
            h.update(str(int(st.st_mtime)).encode("utf-8"))
            h.update(str(int(st.st_size)).encode("utf-8"))
            total += 1
    return {"root": root_abs, "num_files": total, "sha256": h.hexdigest()}


def generate_icassp_manifest(root: str, out_csv: str, freqs_mhz: Sequence[float], task: str = "Task_2_ICASSP") -> int:
    """Scan ICASSP tree and write a complete manifest CSV. Returns number of rows written."""
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow([
            "file_name",
            "building",
            "antenna",
            "frequency_index",
            "sample_index",
            "freq_MHz",
            "input_file",
            "output_file",
            "position_file",
            "radiation_pattern_file",
            "sampling_position",
        ])
        n = 0
        for b in range(1, 26):
            for ant in range(1, 3):
                for f in range(1, 1 + len(freqs_mhz)):
                    for sp in range(80):
                        input_file, output_file, position_file, radiation = _icassp_expected_paths(root, task, b, ant, f, sp)
                        if os.path.exists(input_file):
                            file_name = os.path.basename(input_file)
                            writer.writerow([
                                file_name,
                                b,
                                ant,
                                f,
                                sp,
                                float(freqs_mhz[f - 1]),
                                input_file,
                                output_file,
                                position_file,
                                radiation,
                                sp,
                            ])
                            n += 1
    # meta next to CSV
    meta_path = out_csv + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(compute_icassp_signature(root), mf, indent=2)
    return n


def ensure_icassp_manifest(root: str, out_csv: str, freqs_mhz: Sequence[float], task: str = "Task_2_ICASSP") -> int:
    """Ensure an up-to-date manifest exists for ICASSP root. Rebuild if signature changed."""
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    meta_path = out_csv + ".meta.json"
    try:
        cur_sig = compute_icassp_signature(root)
    except Exception:
        # Force rebuild on failure
        return generate_icassp_manifest(root, out_csv, freqs_mhz, task=task)
    need = not os.path.exists(out_csv) or not os.path.exists(meta_path)
    if not need:
        try:
            with open(meta_path, "r", encoding="utf-8") as mf:
                prev = json.load(mf)
            if (
                str(prev.get("root")) != str(cur_sig.get("root")) or
                str(prev.get("sha256")) != str(cur_sig.get("sha256")) or
                int(prev.get("num_files", -1)) != int(cur_sig.get("num_files", -2))
            ):
                need = True
        except Exception:
            need = True
    if need:
        return generate_icassp_manifest(root, out_csv, freqs_mhz, task=task)
    return -1

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
