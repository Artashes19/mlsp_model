#!/usr/bin/env python3
"""
Cross-branch comparison: runs data loading from both branches and compares outputs.

Usage:
    # From devbugfix_khoren branch:
    python tests/compare_branches.py --run-devbugfix
    
    # Then checkout korean-model and run:
    python tests/compare_branches.py --run-korean
    
    # Finally compare (from either branch):
    python tests/compare_branches.py --compare
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np

SAVE_DIR = Path("/tmp/branch_comparison")
DEVBUGFIX_FILE = SAVE_DIR / "devbugfix_outputs.pt"
KOREAN_FILE = SAVE_DIR / "korean_outputs.pt"
NUM_SAMPLES = 10


def get_icassp_input_files():
    """Find ICASSP input PNG files."""
    icassp_root = os.environ.get("ICASSP_ORIG_PATH", "")
    candidates = [
        Path(icassp_root) / "train" / "Inputs" / "Task_2_ICASSP",
        Path(icassp_root) / "Inputs" / "Task_2_ICASSP",
    ]
    for c in candidates:
        if c.exists():
            return sorted(c.glob("*.png"))[:NUM_SAMPLES]
    raise RuntimeError(f"No ICASSP data found. Set ICASSP_ORIG_PATH env var.")


def run_devbugfix():
    """Run devbugfix_khoren's featurizer with num_channels=3."""
    print("Running devbugfix_khoren featurizer (num_channels=3)...")
    
    # Import from current branch
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.utils import normalize_size, RadarSample
    from src.utils.mlsp.featurizer import featurizer
    from torchvision.io import read_image
    
    input_files = get_icassp_input_files()
    results = {}
    
    for f in input_files:
        input_img = read_image(str(f)).float()
        C, H, W = input_img.shape
        
        sample = RadarSample(
            file_name=str(f),
            task_idx=1,
            pl_clip=float("inf"),
            use_approximator_feature=False,
            use_transmittance_loss=False,
            H=H, W=W,
            x_ant=H // 2, y_ant=W // 2,
            azimuth=0.0,
            freq_MHz=868.0,
            input_img=input_img,
            output_img="",
            radiation_pattern=torch.ones(360),
            pixel_size=0.25,
            mask=torch.ones(H, W),
        )
        sample = normalize_size(sample, target_size=256)
        output = featurizer(sample, num_channels=3, modality_dropout_prob=0.0)
        results[f.name] = output.clone()
        print(f"  {f.name}: shape={output.shape}, range=[{output.min():.4f}, {output.max():.4f}]")
    
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(results, DEVBUGFIX_FILE)
    print(f"\nSaved {len(results)} samples to {DEVBUGFIX_FILE}")


def run_korean():
    """Run korean-model's data loader."""
    print("Running korean-model data loader...")
    
    # Import from current branch (korean-model)
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.dataset import IndoorRadioMapDataset, SamplePaths
    
    input_files = get_icassp_input_files()
    
    # Create dataset with just these files (no outputs needed for input comparison)
    file_pairs = [SamplePaths(input_path=f, output_path=None) for f in input_files]
    dataset = IndoorRadioMapDataset(
        root=".",  # not used when file_pairs provided
        split="train",
        resize_hw=(256, 256),
        file_pairs=file_pairs,
    )
    
    results = {}
    for i, f in enumerate(input_files):
        x, y, m, meta = dataset[i]
        results[f.name] = x.clone()
        print(f"  {f.name}: shape={x.shape}, range=[{x.min():.4f}, {x.max():.4f}]")
    
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(results, KOREAN_FILE)
    print(f"\nSaved {len(results)} samples to {KOREAN_FILE}")


def compare():
    """Compare saved outputs from both branches."""
    print("Comparing outputs from both branches...")
    
    if not DEVBUGFIX_FILE.exists():
        print(f"ERROR: {DEVBUGFIX_FILE} not found. Run --run-devbugfix first.")
        return 1
    if not KOREAN_FILE.exists():
        print(f"ERROR: {KOREAN_FILE} not found. Run --run-korean first.")
        return 1
    
    devbugfix = torch.load(DEVBUGFIX_FILE)
    korean = torch.load(KOREAN_FILE)
    
    print(f"\nDevbugfix samples: {len(devbugfix)}")
    print(f"Korean samples: {len(korean)}")
    
    common_keys = set(devbugfix.keys()) & set(korean.keys())
    print(f"Common samples: {len(common_keys)}\n")
    
    all_diffs = []
    for key in sorted(common_keys):
        d = devbugfix[key]
        k = korean[key]
        
        if d.shape != k.shape:
            print(f"  {key}: SHAPE MISMATCH {d.shape} vs {k.shape}")
            continue
        
        diff = (d - k).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        all_diffs.append(max_diff)
        
        status = "OK" if max_diff < 1e-5 else "DIFF"
        print(f"  {key}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e} [{status}]")
    
    print(f"\n{'='*60}")
    print(f"Overall max diff: {max(all_diffs):.2e}")
    print(f"Overall mean of max diffs: {np.mean(all_diffs):.2e}")
    
    if max(all_diffs) < 1e-5:
        print("\nRESULT: EQUIVALENT (within floating point tolerance)")
        return 0
    else:
        print("\nRESULT: DIFFERENT")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Cross-branch data loader comparison")
    parser.add_argument("--run-devbugfix", action="store_true", help="Run devbugfix_khoren featurizer")
    parser.add_argument("--run-korean", action="store_true", help="Run korean-model data loader")
    parser.add_argument("--compare", action="store_true", help="Compare saved outputs")
    args = parser.parse_args()
    
    if not any([args.run_devbugfix, args.run_korean, args.compare]):
        parser.print_help()
        return 1
    
    if args.run_devbugfix:
        run_devbugfix()
    if args.run_korean:
        run_korean()
    if args.compare:
        return compare()
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
