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


def evaluate_prep(
    config: DictConfig,
    project_root: str,
) -> None:
    """
    Evaluate inference predictions against ground truth.
    """
    inference_dir = os.path.abspath(str(config["inference_dir"]))
    out_norm = float(config.get("out_norm", 160.0))
    
    # Validate inference directory exists
    if not os.path.isdir(inference_dir):
        raise RuntimeError(f"Inference directory not found: {inference_dir}")
    
    log.info(f"[evaluate] Inference directory: {inference_dir}")
    log.info(f"[evaluate] Output normalization: {out_norm}")
    
    # Find all .npz files
    npz_files = sorted(glob(os.path.join(inference_dir, "*.npz")))
    
    if len(npz_files) == 0:
        raise RuntimeError(f"No .npz files found in {inference_dir}")
    
    log.info(f"[evaluate] Found {len(npz_files)} .npz files")
    
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
        raise RuntimeError("No valid samples with targets found")
    
    # Calculate average RMSE
    avg_rmse = total_rmse / len(results)
    log.info(f"[evaluate] Average RMSE: {avg_rmse:.6f} dB")
    log.info(f"[evaluate] Evaluated {len(results)} samples")
    
    # Write results to JSON file named with average RMSE
    output_filename = f"RMSE_{avg_rmse:.6f}.json"
    output_path = os.path.join(inference_dir, output_filename)
    
    with open(output_path, "w") as f:
        json.dump(dict(sorted(results.items())), f, indent=4)
    
    log.info(f"[evaluate] Results written to: {output_path}")
