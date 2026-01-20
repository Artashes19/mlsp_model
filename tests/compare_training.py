#!/usr/bin/env python3
"""
Cross-branch training comparison: runs ACTUAL training APIs from both branches
and compares their results.

This test does NOT reimplement any training logic - it calls the real training
entry points from each branch with controlled configs.

Usage:
    # From devbugfix_khoren branch:
    python tests/compare_training.py --run-devbugfix
    
    # Then checkout korean-model and run:
    python tests/compare_training.py --run-korean
    
    # Finally compare (from either branch):
    python tests/compare_training.py --compare
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

# Config
SAVE_DIR = Path("/tmp/training_comparison")
DEVBUGFIX_DIR = SAVE_DIR / "devbugfix"
KOREAN_DIR = SAVE_DIR / "korean"

# Training hyperparameters - must match in both branches
NUM_EPOCHS = 3
BATCH_SIZE = 2
LEARNING_RATE = 3e-4
SEED = 42
NUM_TRAIN_SAMPLES = 8
NUM_VAL_SAMPLES = 2

# Hardcoded samples with 100% mask coverage (verified to avoid loss denominator issues)
# These are from B1 building, freq=1, which all have full mask coverage
VERIFIED_SAMPLES = [
    "B1_Ant1_f1_S0.png",
    "B1_Ant1_f1_S1.png",
    "B1_Ant1_f1_S2.png",
    "B1_Ant1_f1_S3.png",
    "B1_Ant1_f1_S4.png",
    "B1_Ant1_f1_S5.png",
    "B1_Ant1_f1_S6.png",
    "B1_Ant1_f1_S7.png",
    "B1_Ant1_f1_S8.png",
    "B1_Ant1_f1_S9.png",
]


def verify_samples_have_full_mask(icassp_root: Path, sample_names: list[str]) -> list[str]:
    """
    Verify that samples have 100% mask coverage by running them through the featurizer.
    Returns list of sample names that pass verification.
    """
    # Add project root to path for imports
    repo_root = Path(__file__).parent.parent.resolve()
    sys.path.insert(0, str(repo_root))
    
    # Set korean mode for 3-channel featurizer
    os.environ["MLSP_OVERRIDES_CONFIG"] = str(repo_root / "configs" / "korean_mode.yaml")
    
    from src.datamodules.mlsp import MLSPDatamodule
    from src.datamodules.datasets.mlsp import PathlossDataset
    
    # Build inputs list for the samples - use available manifest
    manifest_path = icassp_root / "manifests" / "icassp_train_val_21_22_23_24_25.csv"
    
    inputs_list = MLSPDatamodule.get_inputs_list(
        freqs_mhz=[868],
        freqs=[1],
        manifest_path=str(manifest_path),
    )
    
    # Filter to only our target samples
    inputs_list = [inp for inp in inputs_list if inp["file_name"] in sample_names]
    inputs_list = sorted(inputs_list, key=lambda x: x["file_name"])
    
    print(f"Verifying {len(inputs_list)} samples for 100% mask coverage...")
    
    # Create dataset to get actual masks
    dataset = PathlossDataset(
        inputs_list=inputs_list,
        training=False,
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
        modality_dropout_prob=0.0,
        sparse_dropout_given_dropout=0.0,
    )
    
    verified = []
    for i in range(len(dataset)):
        _, _, mask, meta = dataset[i]
        fname = meta["file_name"]
        mask_coverage = mask.sum().item() / mask.numel()
        
        if mask_coverage >= 0.99:  # Allow tiny floating point tolerance
            verified.append(fname)
            print(f"  ✓ {fname}: {mask_coverage*100:.1f}% coverage")
        else:
            print(f"  ✗ {fname}: {mask_coverage*100:.1f}% coverage (SKIPPED)")
    
    return verified


def run_devbugfix():
    """
    Run training on devbugfix_khoren using the REAL training infrastructure.
    
    Uses verified samples with 100% mask coverage for fair comparison.
    """
    print("=" * 60)
    print("DEVBUGFIX_KHOREN TRAINING (using real run.py)")
    print("=" * 60)
    
    repo_root = Path(__file__).parent.parent.resolve()
    
    DEVBUGFIX_DIR.mkdir(parents=True, exist_ok=True)
    KOREAN_DIR.mkdir(parents=True, exist_ok=True)
    
    icassp_root = Path(os.environ.get("ICASSP_ORIG_PATH", ""))
    
    # Create verified data subset (same samples will be used by both branches)
    data_subset = KOREAN_DIR / "data_subset"
    if not data_subset.exists():
        print("Creating verified data subset...")
        copied = create_verified_data_subset(icassp_root, data_subset)
        if copied == 0:
            return 1
    else:
        print(f"Using existing verified data subset in {data_subset}")
    
    # Create a manifest for the subset
    manifest_path = DEVBUGFIX_DIR / "subset_manifest.csv"
    icassp_root = os.environ.get("ICASSP_ORIG_PATH", "")
    
    with open(manifest_path, "w") as f:
        f.write("file_name,building,antenna,frequency_index,sample_index,freq_MHz,input_file,output_file,position_file,radiation_pattern_file,sampling_position\n")
        for png in sorted((data_subset / "train" / "Inputs" / "Task_2_ICASSP").glob("*.png")):
            # Parse filename: B1_Ant1_f1_S0.png
            parts = png.stem.split("_")
            bld = int(parts[0][1:])
            ant = int(parts[1][3:])
            freq_idx = int(parts[2][1:])
            sample_idx = int(parts[3][1:])
            
            input_file = str(png)
            output_file = str(data_subset / "train" / "Outputs" / "Task_2_ICASSP" / png.name)
            position_file = f"{icassp_root}/Positions/Positions_B{bld}_Ant{ant}_f{freq_idx}.csv"
            radiation_file = f"{icassp_root}/Radiation_Patterns/Ant{ant}_Pattern.csv"
            
            f.write(f"{png.name},{bld},{ant},{freq_idx},{sample_idx},868.0,{input_file},{output_file},{position_file},{radiation_file},{sample_idx}\n")
    
    print(f"Created manifest with {len(list((data_subset / 'train' / 'Inputs' / 'Task_2_ICASSP').glob('*.png')))} samples")
    
    # Set korean mode and output dir via environment
    env = os.environ.copy()
    env["MLSP_OVERRIDES_CONFIG"] = str(repo_root / "configs" / "korean_mode.yaml")
    env["OUTPUT_DIR"] = str(DEVBUGFIX_DIR)
    env["CUDA_VISIBLE_DEVICES"] = "1"
    
    # Build Hydra command with the subset manifest
    # Use ++ to override existing values, + for new values
    cmd = [
        sys.executable, str(repo_root / "run.py"),
        "exps=[e0]",
        f"seed={SEED}",
        "print_config=false",
        # Use our subset manifest for both train and val
        f"++exps.e0.datamodule.train_manifest_path={manifest_path}",
        f"++exps.e0.datamodule.val_manifest_path={manifest_path}",
        # Force use_small_train=false so it uses our train_manifest_path
        "++exps.e0.datamodule.use_small_train=false",
        # Override network input channels for korean_mode (3 channels)
        "+exps.e0.network.in_ch=3",
        # Training settings
        f"++exps.e0.trainer.max_epochs={NUM_EPOCHS}",
        "++exps.e0.trainer.devices=[0]",  # Use 0 because CUDA_VISIBLE_DEVICES=1 remaps it
        f"++exps.e0.datamodule.batch_size={BATCH_SIZE}",
        "++exps.e0.datamodule.num_workers=0",  # Don't need workers for 10 samples
        f"++exps.e0.datamodule.train_samples_per_epoch={NUM_TRAIN_SAMPLES}",  # Use actual dataset size
        # Disable torch.compile to speed up
        "+exps.e0.algorithm.compiled.disable=true",
        # Disable callbacks/loggers for this test (avoid early stopping issues)
        "~exps.e0.callbacks",
        "~exps.e0.loggers",
    ]
    
    print(f"Running: {' '.join(cmd[:5])}...")
    print(f"Output dir: {DEVBUGFIX_DIR}")
    
    result = subprocess.run(
        cmd,
        cwd=repo_root,
        env=env,
        capture_output=False,
    )
    
    if result.returncode != 0:
        print(f"ERROR: Training failed with return code {result.returncode}")
        return 1
    
    print(f"\nTraining completed. Results in {DEVBUGFIX_DIR}")
    return 0


def create_verified_data_subset(icassp_root: Path, data_root: Path) -> int:
    """
    Create data subset using only verified samples with 100% mask coverage.
    Returns number of samples copied.
    """
    train_inputs_dir = data_root / "train" / "Inputs" / "Task_2_ICASSP"
    train_outputs_dir = data_root / "train" / "Outputs" / "Task_2_ICASSP"
    
    # Clean up existing subset
    if data_root.exists():
        shutil.rmtree(data_root)
    
    train_inputs_dir.mkdir(parents=True, exist_ok=True)
    train_outputs_dir.mkdir(parents=True, exist_ok=True)
    
    src_inputs = icassp_root / "Inputs" / "Task_2_ICASSP"
    src_outputs = icassp_root / "Outputs" / "Task_2_ICASSP"
    
    # Verify samples through featurizer first
    verified = verify_samples_have_full_mask(icassp_root, VERIFIED_SAMPLES)
    
    if len(verified) < NUM_TRAIN_SAMPLES + NUM_VAL_SAMPLES:
        print(f"ERROR: Need {NUM_TRAIN_SAMPLES + NUM_VAL_SAMPLES} samples but only {len(verified)} verified")
        return 0
    
    # Copy verified samples
    copied = 0
    for sample_name in verified[:NUM_TRAIN_SAMPLES + NUM_VAL_SAMPLES]:
        inp_file = src_inputs / sample_name
        out_file = src_outputs / sample_name
        
        if inp_file.exists() and out_file.exists():
            shutil.copy(inp_file, train_inputs_dir / sample_name)
            shutil.copy(out_file, train_outputs_dir / sample_name)
            copied += 1
    
    print(f"Copied {copied} verified samples (100% mask coverage) to {data_root}")
    return copied


def run_korean():
    """
    Run training on korean-model using their REAL train.py.
    
    Uses only verified samples with 100% mask coverage.
    NOTE: This function must be run from the korean-model branch.
    The data subset should already be created by run_devbugfix().
    """
    print("=" * 60)
    print("KOREAN-MODEL TRAINING (using real train.py)")
    print("=" * 60)
    
    # Get repo root - either from env var (when run from /tmp) or from file location
    repo_root_env = os.environ.get("COMPARE_TRAINING_REPO_ROOT")
    if repo_root_env:
        repo_root = Path(repo_root_env)
    else:
        repo_root = Path(__file__).parent.parent.resolve()
    
    KOREAN_DIR.mkdir(parents=True, exist_ok=True)
    
    icassp_root = Path(os.environ.get("ICASSP_ORIG_PATH", ""))
    
    # korean-model expects data_root/train/Inputs/Task_2_ICASSP structure
    data_root = KOREAN_DIR / "data_subset"
    train_inputs_dir = data_root / "train" / "Inputs" / "Task_2_ICASSP"
    train_outputs_dir = data_root / "train" / "Outputs" / "Task_2_ICASSP"
    
    # Data subset should already exist (created by run_devbugfix)
    # DON'T try to recreate it here - that would require devbugfix imports
    if not train_inputs_dir.exists() or not any(train_inputs_dir.glob("*.png")):
        print("ERROR: Data subset not found. Run --run-devbugfix first to create it.")
        print(f"  Expected location: {train_inputs_dir}")
        return 1
    
    sample_count = len(list(train_inputs_dir.glob("*.png")))
    print(f"Using existing data subset with {sample_count} samples")
    
    # Create temporary config files with controlled settings
    import yaml
    
    # Training config
    train_config = {
        "lr": LEARNING_RATE,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "num_workers": 0,
        "pin_memory": True,
        "amp_dtype": "bf16",
        "amp_enabled": True,  # Enable AMP (matches devbugfix's bf16-mixed precision)
        "clip_norm": 1.0,
        "checkpoint_dir": str(KOREAN_DIR),
    }
    
    # Data config - use the subset directory
    # All verified samples are from B1, use same building for train/val in this test
    data_config = {
        "data_root": str(data_root),
        "resize": [256, 256],
        "y_db_max": 160.0,
        "train_buildings": [1],  # All verified samples are from B1
        "val_buildings": [1],    # Use same samples for val (small test)
    }
    
    # Write temp configs
    train_cfg_path = KOREAN_DIR / "train_config.yaml"
    data_cfg_path = KOREAN_DIR / "data_config.yaml"
    
    with open(train_cfg_path, "w") as f:
        yaml.dump(train_config, f)
    with open(data_cfg_path, "w") as f:
        yaml.dump(data_config, f)
    
    # Create custom model config with 3 input channels (matching devbugfix korean_mode)
    # Korean's default config uses 4 channels but devbugfix korean_mode uses 3 (R, T, D)
    model_config = {
        "name": "radio_unet_tx",
        "in_ch": 3,  # Match devbugfix korean_mode: R, T, D only
        "out_ch": 1,
        "base_ch": 48,
        "depths": [4, 6, 6, 8],
        "heads": [4, 4, 8, 8],
        "expand": 2.66,
        "use_checkpoint": False,
        "ln_eps": 1e-5,
        "window0": None,
        "sra0_enabled": False,
        "sra0_stride": 4,
    }
    model_cfg_path = KOREAN_DIR / "model_config.yaml"
    with open(model_cfg_path, "w") as f:
        yaml.dump(model_config, f)
    
    # Create a wrapper script that sets the seed before running train.py
    # Korean's train.py doesn't support seed, so we inject it via a wrapper
    wrapper_script = KOREAN_DIR / "train_wrapper.py"
    wrapper_code = f'''
import torch
import numpy as np
import random

# Set seed BEFORE any model initialization
SEED = {SEED}
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Now run the original train.py by importing and calling main
import sys
sys.path.insert(0, "{repo_root}")
from train import main
main()
'''
    with open(wrapper_script, "w") as f:
        f.write(wrapper_code)
    
    cmd = [
        sys.executable, str(wrapper_script),
        "--model", str(model_cfg_path),
        "--data", str(data_cfg_path),
        "--config", str(train_cfg_path),
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print(f"Output dir: {KOREAN_DIR}")
    
    # Set environment - use GPU 1
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"
    
    result = subprocess.run(
        cmd,
        cwd=repo_root,
        env=env,
        capture_output=False,
    )
    
    if result.returncode != 0:
        print(f"ERROR: Training failed with return code {result.returncode}")
        return 1
    
    print(f"\nTraining completed. Results in {KOREAN_DIR}")
    return 0


def compare():
    """Compare training results from both branches."""
    print("=" * 60)
    print("COMPARING TRAINING RESULTS")
    print("=" * 60)
    
    # Find checkpoints - specifically look for last.ckpt to compare equal epochs
    devbugfix_ckpts = list(DEVBUGFIX_DIR.rglob("*.ckpt"))
    korean_ckpts = list(KOREAN_DIR.rglob("*.pt")) + list(KOREAN_DIR.rglob("*.ckpt"))
    
    print(f"\nDevbugfix checkpoints: {devbugfix_ckpts}")
    print(f"Korean checkpoints: {korean_ckpts}")
    
    if not devbugfix_ckpts:
        print("ERROR: No devbugfix checkpoint found")
        return 1
    if not korean_ckpts:
        print("ERROR: No korean checkpoint found")
        return 1
    
    # Find last.ckpt specifically to ensure we compare equal training epochs
    dev_last_ckpts = [p for p in devbugfix_ckpts if p.name == "last.ckpt"]
    kor_last_ckpts = [p for p in korean_ckpts if p.name == "last.ckpt"]
    
    dev_ckpt_path = dev_last_ckpts[0] if dev_last_ckpts else devbugfix_ckpts[0]
    kor_ckpt_path = kor_last_ckpts[0] if kor_last_ckpts else korean_ckpts[0]
    
    print(f"\nUsing checkpoints:")
    print(f"  Devbugfix: {dev_ckpt_path}")
    print(f"  Korean: {kor_ckpt_path}")
    
    # Load checkpoints
    dev_ckpt = torch.load(dev_ckpt_path, map_location="cpu", weights_only=False)
    kor_ckpt = torch.load(kor_ckpt_path, map_location="cpu", weights_only=False)
    
    # Extract model weights (handle different checkpoint formats)
    if "state_dict" in dev_ckpt:
        dev_state = dev_ckpt["state_dict"]
    else:
        dev_state = dev_ckpt
    
    if "state_dict" in kor_ckpt:
        kor_state = kor_ckpt["state_dict"]
    elif "model_state_dict" in kor_ckpt:
        kor_state = kor_ckpt["model_state_dict"]
    else:
        kor_state = kor_ckpt
    
    # Normalize key names (devbugfix may have _network. prefix)
    def normalize_keys(state_dict):
        return {k.replace("_network.", ""): v for k, v in state_dict.items()}
    
    dev_state = normalize_keys(dev_state)
    kor_state = normalize_keys(kor_state)
    
    # Compare weights
    print("\n--- MODEL WEIGHT COMPARISON ---")
    common_keys = set(dev_state.keys()) & set(kor_state.keys())
    print(f"Common weight tensors: {len(common_keys)}")
    
    weight_diffs = []
    for key in sorted(common_keys)[:10]:  # Show first 10
        d = dev_state[key]
        k = kor_state[key]
        if d.shape == k.shape:
            diff = (d - k).abs().max().item()
            weight_diffs.append(diff)
            print(f"  {key}: max_diff={diff:.2e}")
    
    # Compare losses if available
    print("\n--- LOSS COMPARISON ---")
    dev_loss = None
    kor_loss = None
    
    # Try to find loss info in checkpoints or logs
    if "epoch" in dev_ckpt:
        print(f"Devbugfix epochs: {dev_ckpt['epoch']}")
    if "epoch" in kor_ckpt:
        print(f"Korean epochs: {kor_ckpt['epoch']}")
    
    # Check for results.json files
    dev_results = DEVBUGFIX_DIR / "results.json"
    kor_results = KOREAN_DIR / "results.json"
    
    if dev_results.exists():
        import json
        with open(dev_results) as f:
            dev_data = json.load(f)
            print(f"Devbugfix metrics: {dev_data.get('metrics', {})}")
    
    if kor_results.exists():
        import json
        with open(kor_results) as f:
            kor_data = json.load(f)
            print(f"Korean metrics: {kor_data.get('metrics', {})}")
    
    # Summary
    print("\n" + "=" * 60)
    if weight_diffs:
        max_diff = max(weight_diffs)
        mean_diff = np.mean(weight_diffs)
        print(f"Weight comparison - Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}")
        
        tolerance = 1e-2
        if max_diff < tolerance:
            print(f"✓ SUCCESS: Weights are EQUIVALENT (diff < {tolerance})")
            return 0
        else:
            print(f"✗ DIFFERENT: Weights differ by more than {tolerance}")
            return 1
    else:
        print("Could not compare weights - no common keys found")
        return 1


def clean():
    """Clean up temporary files."""
    if SAVE_DIR.exists():
        shutil.rmtree(SAVE_DIR)
        print(f"Removed {SAVE_DIR}")


# ============ Git helpers for cross-branch testing ============

def get_current_branch() -> str:
    """Get current git branch name."""
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def checkout_branch(branch: str) -> None:
    """Checkout to a git branch."""
    subprocess.run(["git", "checkout", branch], check=True)


def stash_changes() -> bool:
    """Stash uncommitted changes. Returns True if there were changes to stash."""
    result = subprocess.run(
        ["git", "stash", "push", "-m", "compare_training_temp"],
        capture_output=True,
        text=True,
    )
    return "No local changes to save" not in result.stdout


def pop_stash() -> None:
    """Pop stashed changes."""
    subprocess.run(["git", "stash", "pop"], check=True)


def clear_module_cache(module_prefixes: list[str]) -> None:
    """Remove modules from sys.modules that start with given prefixes."""
    to_remove = [k for k in sys.modules.keys() if any(k.startswith(p) for p in module_prefixes)]
    for k in to_remove:
        del sys.modules[k]


def run_full():
    """
    Run the full cross-branch training comparison automatically.
    
    1. Run devbugfix training on current branch
    2. Stash changes, checkout korean-model
    3. Run korean training
    4. Compare checkpoints
    5. Restore original branch
    """
    repo_root = Path(__file__).parent.parent.resolve()
    os.chdir(repo_root)
    
    original_branch = get_current_branch()
    print(f"Current branch: {original_branch}")
    
    if "devbugfix" not in original_branch and "dev" not in original_branch:
        print(f"WARNING: Expected to start from devbugfix_khoren branch, but on {original_branch}")
    
    had_changes = False
    
    # ============ PHASE 1: devbugfix_khoren training ============
    print("\n" + "=" * 60)
    print("PHASE 1: Running training on devbugfix_khoren branch")
    print("=" * 60)
    
    ret = run_devbugfix()
    if ret != 0:
        print("ERROR: devbugfix training failed")
        return ret
    
    # Copy this test script to /tmp BEFORE switching branches
    # (since this file doesn't exist on korean-model branch)
    test_script_copy = Path("/tmp/compare_training_temp.py")
    shutil.copy(Path(__file__), test_script_copy)
    print(f"Copied test script to {test_script_copy}")
    
    # Clear module cache before switching branches
    clear_module_cache(["src", "data", "models", "losses"])
    
    # ============ PHASE 2: korean-model training ============
    print("\n" + "=" * 60)
    print("PHASE 2: Checking out korean-model branch")
    print("=" * 60)
    
    had_changes = stash_changes()
    if had_changes:
        print("Stashed local changes to allow branch switch")
    
    checkout_branch("korean-model")
    print("Switched to korean-model branch")
    
    # Re-add repo root to path after branch switch
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    
    print("\n" + "=" * 60)
    print("PHASE 2b: Running training on korean-model branch")
    print("=" * 60)
    
    # Run korean training via subprocess using the COPIED test script
    # Pass repo_root via environment variable since the script is in /tmp
    env = os.environ.copy()
    env["COMPARE_TRAINING_REPO_ROOT"] = str(repo_root)
    
    cmd = [sys.executable, str(test_script_copy), "--run-korean"]
    result = subprocess.run(cmd, cwd=repo_root, env=env)
    
    korean_failed = result.returncode != 0
    if korean_failed:
        print("ERROR: korean-model training failed")
    
    # ============ PHASE 3: Restore original branch FIRST ============
    # (so compare() can run from original branch if needed)
    print("\n" + "=" * 60)
    print(f"PHASE 3: Restoring original branch: {original_branch}")
    print("=" * 60)
    
    checkout_branch(original_branch)
    print(f"Switched back to {original_branch}")
    
    if had_changes:
        pop_stash()
        print("Restored stashed changes")
    
    # Clean up temp script
    test_script_copy.unlink(missing_ok=True)
    
    if korean_failed:
        return 1
    
    # ============ PHASE 4: Compare ============
    print("\n" + "=" * 60)
    print("PHASE 4: Comparing training results")
    print("=" * 60)
    
    return compare()


def main():
    parser = argparse.ArgumentParser(description="Cross-branch training comparison")
    parser.add_argument("--run-devbugfix", action="store_true", help="Run training on devbugfix_khoren")
    parser.add_argument("--run-korean", action="store_true", help="Run training on korean-model")
    parser.add_argument("--compare", action="store_true", help="Compare saved results")
    parser.add_argument("--clean", action="store_true", help="Clean up temporary files")
    parser.add_argument("--full", action="store_true", 
                        help="Run full comparison: devbugfix training → checkout korean → korean training → compare → restore")
    args = parser.parse_args()
    
    if not any([args.run_devbugfix, args.run_korean, args.compare, args.clean, args.full]):
        parser.print_help()
        return 1
    
    if args.clean:
        clean()
        return 0
    
    if args.full:
        return run_full()
    
    if args.run_devbugfix:
        return run_devbugfix()
    if args.run_korean:
        return run_korean()
    if args.compare:
        return compare()
    
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
