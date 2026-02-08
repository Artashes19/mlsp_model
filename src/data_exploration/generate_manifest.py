import argparse
import csv
import json
import os
import random
from typing import List, Sequence

from tqdm import tqdm


def parse_freqs_mhz(freqs_str: str) -> List[float]:
    parts = [p.strip() for p in freqs_str.split(',') if p.strip()]
    return [float(p) for p in parts]


def generate_manifest(root: str, out: str, limit: int) -> int:
    """Generate synthetic manifest. Reads freq_mhz from each sample's JSON frequency_MHz field."""
    rows = []
    for dirpath, dirnames, filenames in tqdm(os.walk(root)):
        npz_files = [f for f in filenames if f.endswith('.npz')]
        for npz in npz_files:
            sample_name = os.path.splitext(npz)[0]
            npz_path = os.path.join(dirpath, npz)
            json_path = os.path.join(dirpath, sample_name + '.json')
            if not os.path.exists(json_path):
                continue
            with open(json_path, 'r') as fp:
                meta = json.load(fp)
            ids = meta.get('ids', {}) if isinstance(meta, dict) else {}
            b = int(ids.get('building', 0))
            ant = int(ids.get('antenna', 0))
            sp = int(ids.get('sample_index', 0))
            freq_mhz = float(meta['frequency_MHz'])
            row = {
                'file_name': sample_name,
                'npz_file': npz_path,
                'json_file': json_path,
                'building': b,
                'antenna': ant,
                'sample_index': sp,
                'freq_mhz': freq_mhz,
            }
            rows.append(row)
        if limit and len(rows) >= limit:
            break
    
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'w', newline='') as csvfile:
        fieldnames = ['file_name', 'npz_file', 'json_file', 'building', 'antenna', 'sample_index', 'freq_mhz']
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
    """Compute a simple signature of the ICASSP dataset layout."""
    import re, hashlib, os
    
    # lists all the names in the root directory
    # checks if name is a valid sample name, s folloed by 6 digits
    # if so, updates the hashlib signiture in the for loop, if not, skip
    # return the number of samples and the sha256 hash - return {'root': root_abs, 'num_samples': count, 'sha256': h.hexdigest()}
    root_abs = os.path.abspath(os.path.expanduser(root))
    names = os.listdir(root_abs)
    count = 0
    h = hashlib.sha256()
    for n in names:
        if re.match(r'^s\d{6}$', n):
            h.update(n.encode('utf-8'))
            count += 1
    return {'root': root_abs, 'num_pairs': count, 'sha256': h.hexdigest()}


def ensure_manifest(root: str, out: str, limit: int) -> int:
    """Ensure a fresh manifest exists for the synthetic dataset.
    - If the manifest or its meta is missing, rebuild it.
    - If the signature differs from the current dataset state, rebuild it.
    - Otherwise, leave it as is.
    Returns the number of rows (if rebuilt) or -1 if left unchanged.
    """
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    meta_path = out + '.meta.json'
    current_sig = compute_dataset_signature(root)
    
    need_rebuild = not os.path.exists(out) or not os.path.exists(meta_path)
    if not need_rebuild:
        with open(meta_path, 'r') as fp:
            meta = json.load(fp)
        if (
            str(meta.get('root')) != str(current_sig.get('root')) or
            str(meta.get('sha256')) != str(current_sig.get('sha256')) or
            int(meta.get('num_pairs', -1)) != int(current_sig.get('num_pairs', -2))
        ):
            need_rebuild = True
    
    if need_rebuild:
        return generate_manifest(root, out, limit=limit)
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
    results = compute_dataset_signature(root)
    results['num_files'] = results.pop('num_pairs')
    return results


def generate_icassp_manifest(root: str, out_csv: str, freqs_mhz: Sequence[float], task: str = "Task_2_ICASSP") -> int:
    """Scan ICASSP tree and write a complete manifest CSV. 
    freqs_mhz maps frequency index (1-based from filename f1/f2/f3) to MHz values.
    Returns number of rows written.
    """
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow([
            "file_name", "building", "antenna", "sample_index", "freq_mhz",
            "input_file", "output_file", "position_file", "radiation_pattern_file",
        ])
        n = 0
        for b in range(1, 26):
            for ant in range(1, 6):
                for f in range(1, 1 + len(freqs_mhz)):
                    for sp in range(80):
                        input_file, output_file, position_file, radiation = _icassp_expected_paths(
                            root, task, b, ant, f, sp
                        )
                        if os.path.exists(input_file):
                            file_name = os.path.basename(input_file)
                            writer.writerow([
                                file_name, b, ant, sp, float(freqs_mhz[f - 1]),
                                input_file, output_file, position_file, radiation,
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


# ===== TEST MANIFEST (evaluation data, no outputs) =====

def _icassp_test_expected_paths(
    root: str,
    eval_data_name: str,
    task: str,
    b: int,
    ant: int,
    f: int,
    sp: int,
    sparse_dir: str,
):
    """Generate expected paths for test/evaluation data (no outputs)."""
    input_dir = os.path.join(root, eval_data_name, f"Inputs/{task}")
    positions_dir = os.path.join(root, eval_data_name, "Positions/")
    radiation_patterns_dir = os.path.join(root, eval_data_name, "Radiation_Patterns/")
    
    input_file = os.path.join(input_dir, f"B{b}_Ant{ant}_f{f}_S{sp}.png")
    position_file = os.path.join(positions_dir, f"Positions_B{b}_Ant{ant}_f{f}.csv")
    radiation_pattern_file = os.path.join(radiation_patterns_dir, f"Ant{ant}_Pattern.csv")
    
    sparse_file = ""
    if sparse_dir:
        sparse_file = os.path.join(sparse_dir, f"B{b}_Ant{ant}_f{f}_S{sp}.png")
    
    return input_file, position_file, radiation_pattern_file, sparse_file


def generate_icassp_test_manifest(
    root: str,
    out_csv: str,
    freqs_mhz: Sequence[float],
    eval_data_name: str,
    task: str,
    sparse_dir: str,
) -> int:
    """
    Generate test manifest for evaluation data (no ground truth outputs).
    freqs_mhz maps frequency index (1-based from filename f1/f2) to MHz values.
    
    Args:
        root: Root directory of evaluation data (e.g., /nfs/dgx/raid/iot/data/icassp2025eval)
        out_csv: Output CSV path
        freqs_mhz: List of frequencies in MHz (e.g., [868, 2400] for test data)
        eval_data_name: Name of evaluation data folder (e.g., "Evaluation_Data_T2")
        task: Task subfolder name (e.g., "Task_2_ICASSP")
        sparse_dir: Optional path to sparse measurements directory
    
    Returns:
        Number of rows written.
    """
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    
    fieldnames = [
        "file_name", "building", "antenna", "sample_index", "freq_mhz",
        "input_file", "output_file", "position_file", "radiation_pattern_file",
    ]
    if sparse_dir:
        fieldnames.append("sparse_file")
    
    with open(out_csv, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(fieldnames)
        n = 0
        for b in range(1, 26):
            for ant in range(1, 6):
                for f in range(1, 1 + len(freqs_mhz)):
                    for sp in range(80):
                        input_file, position_file, radiation_file, sparse_file = _icassp_test_expected_paths(
                            root=root,
                            eval_data_name=eval_data_name,
                            task=task,
                            b=b,
                            ant=ant,
                            f=f,
                            sp=sp,
                            sparse_dir=sparse_dir,
                        )
                        if os.path.exists(input_file):
                            file_name = os.path.basename(input_file)
                            row = [
                                file_name, b, ant, sp, float(freqs_mhz[f - 1]),
                                input_file, "",  # output_file empty for test
                                position_file, radiation_file,
                            ]
                            if sparse_dir:
                                row.append(sparse_file)
                            writer.writerow(row)
                            n += 1
    return n


# ===== FULL EVAL MANIFEST (different directory structure) =====

def _fulleval_expected_paths(
    root: str,
    task: str,
    b: int,
    ant: int,
    f: int,
    sp: int,
    sparse_dir: str,
):
    """Generate expected paths for full eval data (different directory structure).
    
    Full eval structure:
    - Inputs: root/Inputs/Task_X/
    - Positions: root/Test_Data_Positions/
    - Radiation: root/Test_Radiation_Patterns/
    - Sparse: root/rate0.XX/sampledGT/
    """
    input_file = os.path.join(root, "Inputs", task, f"B{b}_Ant{ant}_f{f}_S{sp}.png")
    position_file = os.path.join(root, "Test_Data_Positions", f"Positions_B{b}_Ant{ant}_f{f}.csv")
    radiation_pattern_file = os.path.join(root, "Test_Radiation_Patterns", f"Ant{ant}_Pattern.csv")
    
    sparse_file = ""
    if sparse_dir:
        sparse_file = os.path.join(sparse_dir, f"B{b}_Ant{ant}_f{f}_S{sp}.png")
    
    return input_file, position_file, radiation_pattern_file, sparse_file


def generate_fulleval_test_manifest(
    root: str,
    out_csv: str,
    freqs_mhz: Sequence[float],
    task: str,
    sparse_dir: str,
) -> int:
    """
    Generate test manifest for full evaluation data (different directory structure).
    
    Args:
        root: Root directory of full eval data (e.g., /nfs/dgx/raid/iot/data/icassp2025fulleval)
        out_csv: Output CSV path
        freqs_mhz: List of frequencies in MHz (e.g., [868, 2400])
        task: Task subfolder name (e.g., "Task_1", "Task_2", "Task_3")
        sparse_dir: Optional path to sparse measurements directory
    
    Returns:
        Number of rows written.
    """
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    
    fieldnames = [
        "file_name",
        "building",
        "antenna",
        "sample_index",
        "freq_mhz",
        "input_file",
        "output_file",
        "position_file",
        "radiation_pattern_file",
    ]
    if sparse_dir:
        fieldnames.append("sparse_file")
    
    with open(out_csv, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(fieldnames)
        n = 0
        # Full eval has fewer buildings (6), antennas (up to 3), samples (up to 80)
        # We iterate over reasonable ranges and check existence
        for b in range(1, 26):
            for ant in range(1, 6):
                for f in range(1, 1 + len(freqs_mhz)):
                    for sp in range(80):
                        input_file, position_file, radiation_file, sparse_file = _fulleval_expected_paths(
                            root=root,
                            task=task,
                            b=b,
                            ant=ant,
                            f=f,
                            sp=sp,
                            sparse_dir=sparse_dir,
                        )
                        if os.path.exists(input_file):
                            file_name = os.path.basename(input_file)
                            row = [
                                file_name,
                                b,
                                ant,
                                sp,
                                float(freqs_mhz[f - 1]),
                                input_file,
                                "",  # output_file empty for test
                                position_file,
                                radiation_file,
                            ]
                            if sparse_dir:
                                row.append(sparse_file)
                            writer.writerow(row)
                            n += 1
    return n


# ===== FILTER HELPERS (for per-run manifests) =====

def filter_icassp_manifest(
    src_csv: str, out_csv: str, allow_buildings: Sequence[int], limit_per_building: int | None) -> int:
    """Write a filtered ICASSP manifest containing only rows for allow_buildings,
    with an optional per-building limit. Returns number of rows written."""
    if not src_csv or not os.path.exists(src_csv):
        return -1
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    counts: dict[int, int] = {}
    kept = 0
    with open(src_csv, "r", newline="") as fin, open(out_csv, "w", newline="") as fout:
        rdr = csv.DictReader(fin)
        fieldnames = rdr.fieldnames or [
            "file_name", "building", "antenna", "sample_index", "freq_mhz",
            "input_file", "output_file", "position_file", "radiation_pattern_file",
        ]
        w = csv.DictWriter(fout, fieldnames=fieldnames)
        w.writeheader()
        allow = set(int(b) for b in allow_buildings)
        for row in rdr:
            try:
                b = int(row.get("building"))
            except Exception:
                continue
            if b not in allow:
                continue
            if limit_per_building and limit_per_building > 0:
                c = counts.get(b, 0)
                if c >= limit_per_building:
                    continue
                counts[b] = c + 1
            kept += 1
            w.writerow(row)
    return kept


def filter_synthetic_manifest(src_csv: str, out_csv: str, limit_total: int | None) -> int:
    """Write a filtered synthetic manifest containing up to limit_total samples in order.
    Returns number of rows written."""
    if not src_csv or not os.path.exists(src_csv):
        return -1
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    kept = 0
    with open(src_csv, "r", newline="") as fin, open(out_csv, "w", newline="") as fout:
        rdr = csv.DictReader(fin)
        fieldnames = rdr.fieldnames or ["file_name", "npz_file", "json_file", "building", "antenna", "freq_idx",
                                        "sample_index", "frequency_MHz"]
        w = csv.DictWriter(fout, fieldnames=fieldnames)
        w.writeheader()
        for row in rdr:
            if limit_total and limit_total > 0 and kept >= limit_total:
                break
            w.writerow(row)
            kept += 1
    return kept


def split_synthetic_manifest(
    src_csv: str,
    train_csv: str,
    val_csv: str,
    val_size: int,
) -> tuple[int, int]:
    """
    Split synthetic manifest into disjoint train/val sets.
    
    1. Read all rows from src_csv
    2. Shuffle (assumes seed_everything already called)
    3. First val_size rows -> val_csv
    4. Remaining rows -> train_csv
    
    Returns (n_train, n_val).
    """
    if not src_csv or not os.path.exists(src_csv):
        return -1, -1
    
    # Read all rows
    with open(src_csv, "r", newline="") as fin:
        rdr = csv.DictReader(fin)
        fieldnames = rdr.fieldnames or [
            "file_name",
            "npz_file",
            "json_file",
            "building",
            "antenna",
            "freq_idx",
            "sample_index",
            "frequency_MHz",
        ]
        rows = list(rdr)
    
    # Shuffle (seed_everything already sets the global random state)
    random.shuffle(rows)
    
    # Split: first val_size -> validation, rest -> training
    val_rows = rows[:val_size]
    train_rows = rows[val_size:]
    
    # Write validation manifest
    os.makedirs(os.path.dirname(val_csv) or ".", exist_ok=True)
    with open(val_csv, "w", newline="") as fout:
        w = csv.DictWriter(fout, fieldnames=fieldnames)
        w.writeheader()
        for row in val_rows:
            w.writerow(row)
    
    # Write training manifest
    os.makedirs(os.path.dirname(train_csv) or ".", exist_ok=True)
    with open(train_csv, "w", newline="") as fout:
        w = csv.DictWriter(fout, fieldnames=fieldnames)
        w.writeheader()
        for row in train_rows:
            w.writerow(row)
    
    return len(train_rows), len(val_rows)


def main():
    parser = argparse.ArgumentParser(description="Generate samples.csv manifest for synthetic dataset")
    parser.add_argument('--root', required=True, help='Root directory of the synthetic dataset')
    parser.add_argument('--out', required=True, help='Output CSV path (e.g., /path/to/samples.csv)')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of rows (0 for no limit)')
    args = parser.parse_args()
    
    n = generate_manifest(args.root, args.out, limit=args.limit)
    print(f"Wrote {n} rows to {args.out}")


if __name__ == '__main__':
    main()
