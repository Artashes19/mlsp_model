import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image


def parse_png_bit_depth_and_color_type(png_path: Path) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse the PNG IHDR chunk to get (bit_depth, color_type).
    Returns (None, None) if parsing fails or file is not a PNG.
    PNG spec: IHDR data (13 bytes): width(4), height(4), bit depth(1), color type(1),
              compression(1), filter(1), interlace(1).
    """
    try:
        with png_path.open("rb") as f:
            sig = f.read(8)
            if sig != b"\x89PNG\r\n\x1a\n":
                return None, None
            _length = int.from_bytes(f.read(4), "big")
            chunk_type = f.read(4)
            if chunk_type != b"IHDR":
                return None, None
            ihdr = f.read(13)
            if len(ihdr) != 13:
                return None, None
            # width = int.from_bytes(ihdr[0:4], "big")
            # height = int.from_bytes(ihdr[4:8], "big")
            bit_depth = ihdr[8]
            color_type = ihdr[9]
            return int(bit_depth), int(color_type)
    except Exception:
        return None, None


def image_to_array(img: Image.Image) -> np.ndarray:
    arr = np.array(img)
    # Ensure H, W, C format
    if arr.ndim == 2:
        arr = arr[:, :, None]
    return arr


def compute_channel_stats(arr: np.ndarray, scale_denominator: float) -> List[Dict[str, float]]:
    stats: List[Dict[str, float]] = []
    h, w, c = arr.shape
    for ch in range(c):
        v = arr[:, :, ch].astype(np.float64)
        v_min = float(np.min(v))
        v_max = float(np.max(v))
        v_mean = float(np.mean(v))
        v_std = float(np.std(v))
        v_min_n = v_min / scale_denominator
        v_max_n = v_max / scale_denominator
        v_mean_n = v_mean / scale_denominator
        v_std_n = v_std / scale_denominator
        stats.append(
            {
                "min": v_min,
                "max": v_max,
                "mean": v_mean,
                "std": v_std,
                "min_norm": v_min_n,
                "max_norm": v_max_n,
                "mean_norm": v_mean_n,
                "std_norm": v_std_n,
            }
        )
    return stats


def find_task2_paths(root: Path) -> Dict[str, Optional[Path]]:
    """
    Return dict with train_inputs, train_outputs, test_inputs paths for Task_2.
    Handles both 'Task_2_ICASSP' (train) and 'Task_2' (test) naming observed in the tree.
    """
    train_inputs = root / "train/Inputs/Task_2_ICASSP"
    train_outputs = root / "train/Outputs/Task_2_ICASSP"
    test_inputs = root / "test/Inputs/Task_2"
    return {
        "train_inputs": train_inputs if train_inputs.is_dir() else None,
        "train_outputs": train_outputs if train_outputs.is_dir() else None,
        "test_inputs": test_inputs if test_inputs.is_dir() else None,
    }


def list_pngs(folder: Optional[Path]) -> List[Path]:
    if folder is None:
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() == ".png" and p.is_file()])


def basename_no_ext(path: Path) -> str:
    return path.stem


def pair_train_files(inputs: List[Path], outputs: List[Path]) -> List[Tuple[Path, Path]]:
    out_map: Dict[str, Path] = {basename_no_ext(p): p for p in outputs}
    pairs: List[Tuple[Path, Path]] = []
    missing = 0
    for ip in inputs:
        key = basename_no_ext(ip)
        op = out_map.get(key)
        if op is not None:
            pairs.append((ip, op))
        else:
            missing += 1
    return pairs


def sample_every(paths: List[Path], stride: int, limit: Optional[int]) -> List[Path]:
    sampled = paths[::max(1, stride)]
    if limit is not None:
        return sampled[:limit]
    return sampled


def scan_folder_for_stats(paths: List[Path], max_files: int) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    if not paths:
        return results
    stride = max(1, len(paths) // max(1, max_files))
    for p in sample_every(paths, stride=stride, limit=max_files):
        bit_depth, color_type = parse_png_bit_depth_and_color_type(p)
        try:
            with Image.open(p) as img:
                mode = img.mode
                if mode == "P":
                    # Palette: convert to RGB for analysis
                    img = img.convert("RGB")
                    mode = img.mode
                arr = image_to_array(img)
        except Exception as e:
            results.append(
                {
                    "file": str(p),
                    "error": f"{type(e).__name__}: {e}",
                    "bit_depth": bit_depth,
                    "color_type": color_type,
                }
            )
            continue
        # Determine scale based on bit depth
        if bit_depth in (8, 10, 12, 16):
            denom = float((1 << bit_depth) - 1)
        else:
            # Fallback assume 8-bit
            denom = 255.0
        ch_stats = compute_channel_stats(arr, scale_denominator=denom)
        results.append(
            {
                "file": str(p),
                "mode": mode,
                "shape": list(arr.shape),
                "bit_depth": bit_depth,
                "color_type": color_type,
                "channels": len(ch_stats),
                "stats": ch_stats,
            }
        )
    return results


def write_csv(results: List[Dict[str, object]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "file",
            "mode",
            "shape",
            "bit_depth",
            "color_type",
            "channels",
            "ch0_min",
            "ch0_max",
            "ch0_mean",
            "ch0_std",
            "ch0_min_norm",
            "ch0_max_norm",
            "ch0_mean_norm",
            "ch0_std_norm",
            "ch1_min",
            "ch1_max",
            "ch1_mean",
            "ch1_std",
            "ch1_min_norm",
            "ch1_max_norm",
            "ch1_mean_norm",
            "ch1_std_norm",
            "ch2_min",
            "ch2_max",
            "ch2_mean",
            "ch2_std",
            "ch2_min_norm",
            "ch2_max_norm",
            "ch2_mean_norm",
            "ch2_std_norm",
        ]
        writer.writerow(header)
        for r in results:
            if "stats" not in r:
                writer.writerow([r.get("file"), None, None, r.get("bit_depth"), r.get("color_type"), None])
                continue
            stats = r["stats"]
            # Pad to 3 channels for consistent CSV width
            padded = list(stats) + [
                {"min": None, "max": None, "mean": None, "std": None, "min_norm": None, "max_norm": None, "mean_norm": None, "std_norm": None}
            ] * (3 - len(stats))
            row = [
                r.get("file"),
                r.get("mode"),
                r.get("shape"),
                r.get("bit_depth"),
                r.get("color_type"),
                r.get("channels"),
            ]
            for ch in padded[:3]:
                row.extend([ch.get("min"), ch.get("max"), ch.get("mean"), ch.get("std"), ch.get("min_norm"), ch.get("max_norm"), ch.get("mean_norm"), ch.get("std_norm")])
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan Task_2 PNGs for bit depth, color type, and stats.")
    parser.add_argument("--root", type=str, required=True, help="Dataset root, e.g., /auto/home/artashes/data")
    parser.add_argument("--task", type=str, default="Task_2", help="Task name (default: Task_2)")
    parser.add_argument("--max_files", type=int, default=200, help="Max files to sample per split")
    parser.add_argument("--csv", type=str, default="tools/scan_report.csv", help="CSV path for results")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    paths = find_task2_paths(root)

    train_inputs = list_pngs(paths["train_inputs"]) if paths["train_inputs"] else []
    train_outputs = list_pngs(paths["train_outputs"]) if paths["train_outputs"] else []
    test_inputs = list_pngs(paths["test_inputs"]) if paths["test_inputs"] else []

    print(f"Root: {root}")
    print(f"Train Inputs: {paths['train_inputs']} ({len(train_inputs)} files)")
    print(f"Train Outputs: {paths['train_outputs']} ({len(train_outputs)} files)")
    print(f"Test Inputs: {paths['test_inputs']} ({len(test_inputs)} files)")

    results: List[Dict[str, object]] = []

    # Scan train input PNGs
    if train_inputs:
        print("\nScanning train inputs (sampled)...")
        res_in = scan_folder_for_stats(train_inputs, max_files=args.max_files)
        results.extend(res_in)

    # Scan train output PNGs
    if train_outputs:
        print("\nScanning train outputs (sampled)...")
        res_out = scan_folder_for_stats(train_outputs, max_files=args.max_files)
        results.extend(res_out)

    # Scan test input PNGs
    if test_inputs:
        print("\nScanning test inputs (sampled)...")
        res_test = scan_folder_for_stats(test_inputs, max_files=args.max_files)
        results.extend(res_test)

    # Simple heuristics to print channel semantics
    if train_inputs:
        sample = results[0]
        if "stats" in sample and sample.get("channels") in (3, 4):
            print("\nAssuming input channels are [R, T, D] (from RGB order).")
            print("R/T likely dB attenuation; values shown are normalized if 8- or 16-bit.")

    csv_path = Path(args.csv)
    write_csv(results, csv_path)
    print(f"\nWrote CSV report: {csv_path}")


if __name__ == "__main__":
    main()




