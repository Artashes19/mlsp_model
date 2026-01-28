import json
import logging
import os
from glob import glob

import numpy as np
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def calculate_rmse(
    pred: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    out_norm: float,
) -> float:
    """
    Calculate RMSE between prediction and targets.
    
    Predictions and targets are normalized [0, 1], multiply by out_norm for dB.
    """
    # Convert from normalized to dB scale
    pred_db = pred * out_norm
    targets_db = targets * out_norm
    
    # Masked squared error
    se = ((pred_db - targets_db) ** 2) * masks
    mse = se.sum() / masks.sum()
    rmse = float(np.sqrt(mse))
    return rmse


def find_npz_directories(root_dir: str) -> list[str]:
    """
    Find all directories containing .npz files.
    
    If root_dir contains .npz files directly, return [root_dir].
    Otherwise, recursively find all subdirectories with .npz files.
    """
    # Check if root has .npz files directly
    if glob(os.path.join(root_dir, "*.npz")):
        return [root_dir]
    
    # Recursively find directories with .npz files
    npz_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if any(f.endswith(".npz") for f in filenames):
            npz_dirs.append(dirpath)
    
    return sorted(npz_dirs)


def evaluate_directory(
    inference_dir: str,
    out_norm: float,
) -> tuple[float, int]:
    """
    Evaluate a single directory containing .npz files.
    
    Returns (average_rmse, num_samples).
    """
    # Find all .npz files
    npz_files = sorted(glob(os.path.join(inference_dir, "*.npz")))
    
    if len(npz_files) == 0:
        log.warning(f"[evaluate] No .npz files found in {inference_dir}")
        return 0.0, 0
    
    # Calculate RMSE for each sample
    results = {}
    total_rmse = 0.0
    
    for npz_path in npz_files:
        data = np.load(npz_path, allow_pickle=True)
        
        pred = data["pred"]
        targets = data["targets"]
        masks = data["masks"]
        file_name = str(data["file_name"])
        
        # Skip if targets is None
        if targets is None:
            log.warning(f"[evaluate] Skipping {file_name}: no targets")
            continue
        
        # Calculate RMSE
        rmse = calculate_rmse(
            pred=pred,
            targets=targets,
            masks=masks,
            out_norm=out_norm,
        )
        
        results[file_name] = rmse
        total_rmse += rmse
    
    if len(results) == 0:
        log.warning(f"[evaluate] No valid samples with targets found in {inference_dir}")
        return 0.0, 0
    
    # Calculate average RMSE
    avg_rmse = total_rmse / len(results)
    
    # Write results to JSON file named with average RMSE
    output_filename = f"RMSE_{avg_rmse:.6f}.json"
    output_path = os.path.join(inference_dir, output_filename)
    
    with open(output_path, "w") as f:
        json.dump(dict(sorted(results.items())), f, indent=4)
    
    return avg_rmse, len(results)


def evaluate_prep(
    config: DictConfig,
    project_root: str,
) -> None:
    """
    Evaluate inference predictions against ground truth.
    
    Supports recursive evaluation - if the provided directory doesn't contain
    .npz files directly, it will find and evaluate all subdirectories that do.
    """
    inference_dir = os.path.abspath(str(config["inference_dir"]))
    out_norm = float(config.get("out_norm", 160.0))
    
    # Validate inference directory exists
    if not os.path.isdir(inference_dir):
        raise RuntimeError(f"Inference directory not found: {inference_dir}")
    
    log.info(f"[evaluate] Inference directory: {inference_dir}")
    log.info(f"[evaluate] Output normalization: {out_norm}")
    
    # Find all directories with .npz files
    npz_dirs = find_npz_directories(inference_dir)
    
    if not npz_dirs:
        raise RuntimeError(f"No .npz files found in {inference_dir} or subdirectories")
    
    log.info(f"[evaluate] Found {len(npz_dirs)} directories to evaluate")
    
    # Evaluate each directory
    all_results = {}
    total_samples = 0
    
    for npz_dir in npz_dirs:
        rel_path = os.path.relpath(npz_dir, inference_dir)
        if rel_path == ".":
            rel_path = os.path.basename(inference_dir)
        log.info(f"[evaluate] Evaluating: {rel_path}")
        
        avg_rmse, num_samples = evaluate_directory(
            inference_dir=npz_dir,
            out_norm=out_norm,
        )
        
        if num_samples > 0:
            all_results[npz_dir] = {
                "rmse": avg_rmse,
                "samples": num_samples,
            }
            total_samples += num_samples
            log.info(f"[evaluate]   RMSE: {avg_rmse:.6f} dB ({num_samples} samples)")
    
    if total_samples == 0:
        raise RuntimeError("No valid samples with targets found in any directory")
    
    # Summary
    log.info(f"[evaluate] === Summary ===")
    for path, result in all_results.items():
        rel_path = os.path.relpath(path, inference_dir)
        if rel_path == ".":
            rel_path = os.path.basename(inference_dir)
        log.info(f"[evaluate]   {rel_path}: RMSE={result['rmse']:.6f} dB ({result['samples']} samples)")
    
    log.info(f"[evaluate] Total: {total_samples} samples across {len(all_results)} directories")
