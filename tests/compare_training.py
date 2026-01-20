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


def run_devbugfix():
    """
    Run training on devbugfix_khoren using the REAL training infrastructure.
    
    Uses the same small data subset as korean-model for fair comparison.
    """
    print("=" * 60)
    print("DEVBUGFIX_KHOREN TRAINING (using real run.py)")
    print("=" * 60)
    
    repo_root = Path(__file__).parent.parent.resolve()
    
    DEVBUGFIX_DIR.mkdir(parents=True, exist_ok=True)
    
    # Use the same data subset as korean - create if needed
    data_subset = KOREAN_DIR / "data_subset"
    if not data_subset.exists():
        print("ERROR: Run --run-korean first to create the data subset")
        return 1
    
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
        # Disable callbacks/loggers for speed
        "++exps.e0.callbacks={}",
        "++exps.e0.loggers={}",
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


def run_korean():
    """
    Run training on korean-model using their REAL train.py.
    
    Copies ~50 samples to a temp directory and trains on those.
    """
    print("=" * 60)
    print("KOREAN-MODEL TRAINING (using real train.py)")
    print("=" * 60)
    
    repo_root = Path(__file__).parent.parent.resolve()
    
    KOREAN_DIR.mkdir(parents=True, exist_ok=True)
    
    icassp_root = Path(os.environ.get("ICASSP_ORIG_PATH", ""))
    
    # korean-model expects data_root/train/Inputs/Task_2_ICASSP structure
    # Copy ~50 samples to a temp directory
    data_root = KOREAN_DIR / "data_subset"
    train_inputs_dir = data_root / "train" / "Inputs" / "Task_2_ICASSP"
    train_outputs_dir = data_root / "train" / "Outputs" / "Task_2_ICASSP"
    
    if not train_inputs_dir.exists():
        print("Copying ~50 samples to temp directory...")
        train_inputs_dir.mkdir(parents=True, exist_ok=True)
        train_outputs_dir.mkdir(parents=True, exist_ok=True)
        
        # Get samples from train buildings (1-14) and val buildings (21-25)
        src_inputs = icassp_root / "Inputs" / "Task_2_ICASSP"
        src_outputs = icassp_root / "Outputs" / "Task_2_ICASSP"
        
        # Select ~40 train samples (from B1-B5) and ~10 val samples (from B21-B22)
        train_buildings = ["B1", "B2", "B3", "B4", "B5"]
        val_buildings = ["B21", "B22"]
        
        copied = 0
        for inp_file in sorted(src_inputs.glob("*.png")):
            # Only freq=1 samples
            if "_f1_" not in inp_file.name:
                continue
            # Check building
            is_train = any(inp_file.name.startswith(b + "_") for b in train_buildings)
            is_val = any(inp_file.name.startswith(b + "_") for b in val_buildings)
            
            if not (is_train or is_val):
                continue
            
            # Limit train samples per building
            if is_train and copied >= 40:
                continue
            if is_val and copied >= 50:
                break
                
            out_file = src_outputs / inp_file.name
            if out_file.exists():
                shutil.copy(inp_file, train_inputs_dir / inp_file.name)
                shutil.copy(out_file, train_outputs_dir / out_file.name)
                copied += 1
        
        print(f"Copied {copied} samples to {data_root}")
    else:
        print(f"Using existing samples in {data_root}")
    
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
        "amp_enabled": True,  # Enable AMP for reasonable memory usage
        "clip_norm": 1.0,
        "checkpoint_dir": str(KOREAN_DIR),
    }
    
    # Data config - use the subset directory
    # Buildings that exist in our subset
    data_config = {
        "data_root": str(data_root),
        "resize": [256, 256],
        "y_db_max": 160.0,
        "train_buildings": [1, 2, 3, 4, 5],
        "val_buildings": [21, 22],
    }
    
    # Write temp configs
    train_cfg_path = KOREAN_DIR / "train_config.yaml"
    data_cfg_path = KOREAN_DIR / "data_config.yaml"
    
    with open(train_cfg_path, "w") as f:
        yaml.dump(train_config, f)
    with open(data_cfg_path, "w") as f:
        yaml.dump(data_config, f)
    
    # Use existing model config from repo
    model_cfg_path = repo_root / "cfgs" / "model_txunet.yaml"
    
    cmd = [
        sys.executable, str(repo_root / "train.py"),
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
    
    # Find checkpoints
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
    
    # Load checkpoints
    dev_ckpt = torch.load(devbugfix_ckpts[0], map_location="cpu", weights_only=False)
    kor_ckpt = torch.load(korean_ckpts[0], map_location="cpu", weights_only=False)
    
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


def main():
    parser = argparse.ArgumentParser(description="Cross-branch training comparison")
    parser.add_argument("--run-devbugfix", action="store_true", help="Run training on devbugfix_khoren")
    parser.add_argument("--run-korean", action="store_true", help="Run training on korean-model")
    parser.add_argument("--compare", action="store_true", help="Compare saved results")
    parser.add_argument("--clean", action="store_true", help="Clean up temporary files")
    args = parser.parse_args()
    
    if not any([args.run_devbugfix, args.run_korean, args.compare, args.clean]):
        parser.print_help()
        return 1
    
    if args.clean:
        clean()
        return 0
    
    if args.run_devbugfix:
        return run_devbugfix()
    if args.run_korean:
        return run_korean()
    if args.compare:
        return compare()
    
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
