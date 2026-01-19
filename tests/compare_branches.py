#!/usr/bin/env python3
"""
Cross-branch comparison: runs REAL data loading APIs from both branches and compares outputs.

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


def run_devbugfix(resize_backend=None):
    """Run devbugfix_khoren's REAL PathlossDataset pipeline."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    # Only override if explicitly specified (otherwise use config/default)
    if resize_backend is not None:
        from src.utils.mlsp.augmentations import set_resize_backend
        set_resize_backend(resize_backend)
    
    from src.utils.mlsp.config_overrides import get_config
    config = get_config()
    
    print(f"Running devbugfix_khoren PathlossDataset (num_channels={config.num_channels}, resize={config.resize_backend})...")
    
    from src.datamodules.mlsp import MLSPDatamodule
    from src.datamodules.datasets.mlsp import PathlossDataset
    
    icassp_root = os.environ.get("ICASSP_ORIG_PATH", "")
    
    # Use the real get_inputs_list to build inputs
    inputs_list = MLSPDatamodule.get_inputs_list(
        data_dir=icassp_root,
        freqs_mhz=[868],  # Just one freq for simplicity
        freqs=[1],
        task="Task_2_ICASSP",
        manifest_path=None
    )
    
    # Take first NUM_SAMPLES
    inputs_list = sorted(inputs_list, key=lambda x: x.file_name)[:NUM_SAMPLES]
    print(f"  Found {len(inputs_list)} samples")
    
    # Create the REAL dataset
    dataset = PathlossDataset(
        inputs_list=inputs_list,
        training=False,  # Deterministic
        mlsp_task1=False,
        mlsp_task_idx=1,
        task_idx=1,
        pl_clip=None,
        use_approximator_feature=False,
        use_transmittance_loss=False,
        inference=True,
        reps_per_epoch=1,
        augment_val=False,
        augmentations=None,
        sparse_range=[0.0, 0.0],
        modality_dropout_prob=0.0,  # No dropout
        sparse_dropout_given_dropout=0.0,
        # num_channels from config (default 9, or 3 if korean_mode.yaml loaded)
    )
    
    results = {}
    for i in range(len(inputs_list)):
        input_tensor, output_tensor, mask, meta = dataset[i]
        fname = Path(meta["file_name"]).name
        results[fname] = {
            "input": input_tensor.clone(),
            "mask": mask.clone(),
        }
        print(f"  {fname}: input={input_tensor.shape}, range=[{input_tensor.min():.4f}, {input_tensor.max():.4f}]")
    
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(results, DEVBUGFIX_FILE)
    print(f"\nSaved {len(results)} samples to {DEVBUGFIX_FILE}")


def run_korean():
    """Run korean-model's REAL IndoorRadioMapDataset pipeline."""
    print("Running korean-model IndoorRadioMapDataset...")
    
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.dataset import IndoorRadioMapDataset, gather_task2_samples, SamplePaths
    
    icassp_root = os.environ.get("ICASSP_ORIG_PATH", "")
    
    # Gather samples the korean-model way
    all_samples = gather_task2_samples(Path(icassp_root), split="train")
    # Filter to match devbugfix (freq=1 means f1 in filename)
    samples = [s for s in all_samples if "_f1_" in s.input_path.name]
    samples = sorted(samples, key=lambda x: x.input_path.name)[:NUM_SAMPLES]
    print(f"  Found {len(samples)} samples")
    
    # Create the REAL dataset
    dataset = IndoorRadioMapDataset(
        root=icassp_root,
        split="train",
        resize_hw=(256, 256),
        file_pairs=samples,
    )
    
    results = {}
    for i in range(len(samples)):
        x, y, m, meta = dataset[i]
        fname = samples[i].input_path.name
        results[fname] = {
            "input": x.clone(),
            "mask": m.clone(),
        }
        print(f"  {fname}: input={x.shape}, range=[{x.min():.4f}, {x.max():.4f}]")
    
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
    
    if not common_keys:
        print("WARNING: No common samples found!")
        print(f"  Devbugfix keys: {list(devbugfix.keys())[:5]}...")
        print(f"  Korean keys: {list(korean.keys())[:5]}...")
        return 1
    
    all_input_diffs = []
    all_mask_diffs = []
    
    for key in sorted(common_keys):
        d = devbugfix[key]
        k = korean[key]
        
        d_input = d["input"] if isinstance(d, dict) else d
        k_input = k["input"] if isinstance(k, dict) else k
        
        if d_input.shape != k_input.shape:
            print(f"  {key}: INPUT SHAPE MISMATCH {d_input.shape} vs {k_input.shape}")
            continue
        
        input_diff = (d_input - k_input).abs()
        max_input_diff = input_diff.max().item()
        mean_input_diff = input_diff.mean().item()
        all_input_diffs.append(max_input_diff)
        
        # Compare masks if available
        mask_diff_str = ""
        if isinstance(d, dict) and isinstance(k, dict) and "mask" in d and "mask" in k:
            d_mask = d["mask"]
            k_mask = k["mask"]
            if d_mask.shape == k_mask.shape:
                mask_diff = (d_mask - k_mask).abs()
                max_mask_diff = mask_diff.max().item()
                all_mask_diffs.append(max_mask_diff)
                mask_diff_str = f", mask_diff={max_mask_diff:.2e}"
            else:
                mask_diff_str = f", MASK SHAPE MISMATCH {d_mask.shape} vs {k_mask.shape}"
        
        status = "OK" if max_input_diff < 1e-5 else "DIFF"
        print(f"  {key}: input_diff={max_input_diff:.2e} (mean={mean_input_diff:.2e}){mask_diff_str} [{status}]")
    
    print(f"\n{'='*60}")
    print(f"INPUT - Overall max diff: {max(all_input_diffs):.2e}")
    print(f"INPUT - Overall mean of max diffs: {np.mean(all_input_diffs):.2e}")
    if all_mask_diffs:
        print(f"MASK  - Overall max diff: {max(all_mask_diffs):.2e}")
    
    if max(all_input_diffs) < 1e-5:
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
    parser.add_argument("--resize-backend", choices=["torchvision", "pil"], default=None,
                        help="Override resize backend (default: use MLSP_OVERRIDES_CONFIG or torchvision)")
    args = parser.parse_args()
    
    if not any([args.run_devbugfix, args.run_korean, args.compare]):
        parser.print_help()
        return 1
    
    if args.run_devbugfix:
        run_devbugfix(resize_backend=args.resize_backend)
    if args.run_korean:
        run_korean()
    if args.compare:
        return compare()
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
