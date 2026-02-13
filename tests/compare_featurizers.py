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
import subprocess
from pathlib import Path

import torch
import numpy as np

SAVE_DIR = Path("/tmp/branch_comparison")
DEVBUGFIX_FILE = SAVE_DIR / "devbugfix_outputs.pt"
KOREAN_FILE = SAVE_DIR / "korean_outputs.pt"
NUM_SAMPLES = 10
REPO_ROOT = Path(__file__).parent.parent.resolve()


def _run_cmd(cmd, cwd: Path | None = None) -> str:
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def _has_korean_module(repo_root: Path) -> bool:
    return (repo_root / "data" / "dataset.py").is_file()


def _ensure_korean_worktree(repo_root: Path, worktree_path: Path, branch: str = "korean-model") -> None:
    output = _run_cmd(["git", "worktree", "list", "--porcelain"], cwd=repo_root)
    worktree_paths = [
        line.split(" ", 1)[1] for line in output.splitlines() if line.startswith("worktree ")
    ]
    if str(worktree_path) in worktree_paths:
        return
    if worktree_path.exists():
        git_marker = worktree_path / ".git"
        if not git_marker.exists():
            raise RuntimeError(f"Worktree path exists and is not a git worktree: {worktree_path}")
    else:
        worktree_path.parent.mkdir(parents=True, exist_ok=True)
    _run_cmd(["git", "worktree", "add", str(worktree_path), branch], cwd=repo_root)


def run_devbugfix():
    """Run devbugfix_khoren's REAL PathlossDataset pipeline."""
    sys.path.insert(0, str(REPO_ROOT))
    
    print("Running devbugfix_khoren PathlossDataset...")
    
    from src.datamodules.indoor import IndoorDatamodule
    from src.datamodules.datasets.indoor import PathlossDataset
    
    icassp_root = os.environ.get("ICASSP_ORIG_PATH", "")
    manifest_path = os.path.join(icassp_root, "manifests", "icassp_val_21_22_23_24_25.csv")
    
    # Use the real get_inputs_list to build inputs from manifest
    inputs_list = IndoorDatamodule.get_inputs_list(
        freqs_mhz=[868],  # Just one freq for simplicity
        freqs=[1],
        manifest_path=manifest_path,
    )
    
    # Take first NUM_SAMPLES (sorted by file_name from dict)
    inputs_list = sorted(inputs_list, key=lambda x: x["file_name"])[:NUM_SAMPLES]
    print(f"  Found {len(inputs_list)} samples")
    
    # Create the REAL dataset
    dataset = PathlossDataset(
        inputs_list=inputs_list,
        training=False,  # Deterministic
        inference=True,
        augmentations=None,
        sparse_range=[0.0, 0.0],
        modality_dropout_prob=0.0,  # No dropout
        sparse_dropout_given_dropout=0.0,
    )
    
    results = {}
    for i in range(min(len(inputs_list), NUM_SAMPLES)):
        input_tensor, output_tensor, mask, meta = dataset[i]
        fname = Path(meta["file_name"]).name if isinstance(meta["file_name"], str) else meta["file_name"]
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
    
    sys.path.insert(0, str(REPO_ROOT))
    from data.dataset import IndoorRadioMapDataset, SamplePaths
    
    icassp_root = Path(os.environ.get("ICASSP_ORIG_PATH", ""))
    
    # Gather samples manually since directory structure differs from korean-model expectation
    # Actual structure: Inputs/Task_2_ICASSP/, Outputs/Task_2_ICASSP/
    input_dir = icassp_root / "Inputs" / "Task_2_ICASSP"
    output_dir = icassp_root / "Outputs" / "Task_2_ICASSP"
    
    input_files = sorted(input_dir.glob("*.png"))
    output_map = {p.stem: p for p in output_dir.glob("*.png")}
    
    # Create SamplePaths for files that have both input and output
    all_samples = [
        SamplePaths(input_path=inp, output_path=output_map.get(inp.stem))
        for inp in input_files
        if inp.stem in output_map
    ]
    
    # Filter to match devbugfix: validation buildings (21-25) + freq=1
    val_buildings = {"B21", "B22", "B23", "B24", "B25"}
    samples = [
        s for s in all_samples 
        if "_f1_" in s.input_path.name and any(s.input_path.name.startswith(b) for b in val_buildings)
    ]
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


def run_korean_via_worktree():
    worktree_path = Path("/tmp/branch_comparison/korean_model_worktree")
    _ensure_korean_worktree(REPO_ROOT, worktree_path)
    inline = """
import os
from pathlib import Path
import torch
from data.dataset import IndoorRadioMapDataset, SamplePaths

icassp_root = Path(os.environ.get("ICASSP_ORIG_PATH", ""))
input_dir = icassp_root / "Inputs" / "Task_2_ICASSP"
output_dir = icassp_root / "Outputs" / "Task_2_ICASSP"

input_files = sorted(input_dir.glob("*.png"))
output_map = {p.stem: p for p in output_dir.glob("*.png")}

all_samples = [
    SamplePaths(input_path=inp, output_path=output_map.get(inp.stem))
    for inp in input_files
    if inp.stem in output_map
]

val_buildings = {"B21", "B22", "B23", "B24", "B25"}
samples = [
    s for s in all_samples
    if "_f1_" in s.input_path.name and any(s.input_path.name.startswith(b) for b in val_buildings)
]
samples = sorted(samples, key=lambda x: x.input_path.name)[:10]

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
    results[fname] = {"input": x.clone(), "mask": m.clone()}

Path("/tmp/branch_comparison").mkdir(parents=True, exist_ok=True)
torch.save(results, "/tmp/branch_comparison/korean_outputs.pt")
"""
    cmd = [sys.executable, "-c", inline]
    result = subprocess.run(cmd, cwd=str(worktree_path))
    if result.returncode != 0:
        raise RuntimeError("Failed to run korean-model loader in worktree.")


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
    if not all_input_diffs:
        print("INPUT - No comparable inputs (shape mismatches).")
        return 1
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
    parser.add_argument("--run-korean-local", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--compare", action="store_true", help="Compare saved outputs")
    args = parser.parse_args()
    
    if args.run_korean_local:
        run_korean()
        return 0
    
    if not any([args.run_devbugfix, args.run_korean, args.compare]):
        parser.print_help()
        return 1
    
    if args.run_devbugfix:
        run_devbugfix()
    if args.run_korean:
        if _has_korean_module(REPO_ROOT):
            run_korean()
        else:
            run_korean_via_worktree()
    if args.compare:
        return compare()
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
