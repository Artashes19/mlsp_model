import csv
import glob
import logging
import os
import re
import time
from typing import Optional

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from src.algorithms.algorithm_base import AlgorithmBase
from src.datamodules.indoor import IndoorDatamodule
from src.utils.indoor.channel_config import NUM_CHANNELS

log = logging.getLogger(__name__)


def resolve_ckpt_paths(
    paths: list[str],
) -> list[str]:
    """
    Resolve checkpoint paths from a list of files and/or directories.
    
    For each path:
    - If it's a file: validate it exists and add to results
    - If it's a directory: recursively scan for all .ckpt files
    
    Args:
        paths: List of checkpoint file paths or directories to scan
    
    Returns:
        Deduplicated, sorted list of absolute checkpoint paths
    """
    resolved = []
    
    for path in paths:
        abs_path = os.path.abspath(str(path))
        
        if os.path.isfile(abs_path):
            resolved.append(abs_path)
        elif os.path.isdir(abs_path):
            # Recursively scan for .ckpt files
            pattern = os.path.join(abs_path, "**", "*.ckpt")
            ckpt_files = glob.glob(pattern, recursive=True)
            resolved.extend(ckpt_files)
        else:
            log.warning(f"[inference] Path does not exist: {abs_path}")
    
    # Deduplicate and sort
    resolved = sorted(set(resolved))
    
    return resolved


def _get_dataset_by_split(
    datamodule: IndoorDatamodule,
    split: str,
) -> Optional[Dataset]:
    """
    Get dataset by split name. Returns None if split is not available.
    
    Test sets are stored in a dict with descriptive names:
    - "test" for single test sets (ICASSP tasks)
    - "test_0.02", "test_0.5" for MLSP sparse rate variants
    """
    if split == "train":
        return datamodule.train_set
    elif split == "test":
        # Legacy: return first test set if only one exists, or look up "test" key
        if len(datamodule.test_sets) == 1:
            return list(datamodule.test_sets.values())[0]
        return datamodule.test_sets.get("test")
    elif split.startswith("test_"):
        # Named test sets: test_0.02, test_0.5, etc.
        return datamodule.test_sets.get(split)
    elif split == "val_no_sparse":
        return datamodule.val_set_no_sparse
    elif split == "val_no_trans_ref":
        return datamodule.val_set_no_trans_ref
    elif split == "val_all_enabled":
        return datamodule.val_set_all_enabled
    elif split == "synth_val_no_sparse":
        return datamodule.synth_val_set_no_sparse
    elif split == "synth_val_no_trans_ref":
        return datamodule.synth_val_set_no_trans_ref
    elif split == "synth_val_all_enabled":
        return datamodule.synth_val_set_all_enabled
    else:
        log.warning(f"[inference] Unknown split name: {split}")
        return None


def parse_output_dir(
    ckpt_path: str,
    predictions_dir: str,
    datamodule_name: str,
    split_name: str,
) -> str:
    """
    Parse checkpoint path to derive output directory.
    
    Extracts timestamp and checkpoint name from checkpoint path.
    """
    # Extract timestamp pattern (YYYY-MM-DD_HH-MM-SS.microseconds)
    timestamp_pattern = r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d+"
    match = re.search(timestamp_pattern, ckpt_path)
    
    if match:
        timestamp = match.group(0)
    else:
        # Fallback: use parent directory name
        timestamp = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(ckpt_path))))
    
    # Extract checkpoint name (without extension)
    ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    
    output_dir = os.path.join(
        predictions_dir,
        timestamp,
        ckpt_name,
        datamodule_name,
        split_name,
    )
    return output_dir


def generate_csv_rows(
    file_name: str,
    pred: np.ndarray,
    mask: np.ndarray,
    out_norm: float = 160.0,
) -> list[tuple[str, float]]:
    """
    Generate CSV rows for a single prediction (Kaggle submission format).
    
    Args:
        file_name: Sample file name (e.g., "B1_Ant1_f1_S0.png")
        pred: Prediction array (1, H, W) or (H, W), normalized [0, 1]
        mask: Mask array (H, W), 0 for sparse measurement pixels to exclude
        out_norm: Output normalization factor (default 160.0 for dB)
    
    Returns:
        List of (ID, PL) tuples where ID is "{base_name}_{pixel_index}"
    """
    # Parse file_name to get base (e.g., "B1_Ant1_f1_S0")
    base_name = os.path.splitext(file_name)[0]  # Remove .png
    
    # Get prediction in dB scale
    if pred.ndim == 3:
        pred = pred.squeeze(0)  # (1, H, W) -> (H, W)
    pred_db = pred * out_norm
    
    H, W = pred.shape
    rows = []
    
    for pixel_idx in range(H * W):
        row = pixel_idx // W
        col = pixel_idx % W
        
        # Skip pixels where mask is 0 (sparse measurement locations for MLSP)
        if mask[row, col] == 0:
            continue
        
        pl_value = float(pred_db[row, col])
        id_str = f"{base_name}_{pixel_idx}"
        rows.append((id_str, pl_value))
    
    return rows


def save_prediction_png(
    pred: np.ndarray,
    output_path: str,
    out_norm: float,
) -> None:
    """
    Save prediction as 8-bit grayscale PNG with raw dB values.
    
    Pixel value = dB value directly (no scaling).
    """
    if pred.ndim == 3:
        pred = pred.squeeze(0)
    
    # Convert normalized [0,1] to dB and cast directly to uint8
    pred_db = pred * out_norm
    pred_uint8 = pred_db.astype(np.uint8)
    
    img = Image.fromarray(pred_uint8, mode="L")
    img.save(output_path)


def inference_prep(
    config: DictConfig,
    project_root: str,
) -> None:
    """
    Run inference on multiple dataset splits using trained checkpoints.
    
    Steps:
    1. Parse checkpoint paths and splits from config
    2. Resolve checkpoint paths (expand directories to .ckpt files)
    3. Instantiate datamodule and algorithm (once)
    4. For each checkpoint: load weights and run inference on all splits
    """
    t0 = time.perf_counter()
    
    # Get inference parameters
    raw_ckpt_paths = list(config["ckpt_paths"])
    gpu = int(config.get("gpu", 0))
    splits = list(config["split"])  # Must be a list
    predictions_dir = os.path.expanduser(str(config["predictions_dir"]))
    
    # Resolve checkpoint paths (expand directories to .ckpt files)
    ckpt_paths = resolve_ckpt_paths(paths=raw_ckpt_paths)
    
    if not ckpt_paths:
        raise RuntimeError(f"No checkpoint files found in: {raw_ckpt_paths}")
    
    log.info(f"[inference] Resolved {len(ckpt_paths)} checkpoint(s): {ckpt_paths}")
    log.info(f"[inference] Splits to process: {splits}")
    
    # Instantiate datamodule via hydra
    log.info(f"[inference] Instantiating datamodule: {config.datamodule._target_}")
    datamodule: IndoorDatamodule = hydra.utils.instantiate(
        config.datamodule,
        multi_gpu=False,
    )
    
    # Use NUM_CHANNELS from channel_config (single source of truth)
    num_channels = NUM_CHANNELS
    if "in_ch" in config.network:
        config.network.in_ch = num_channels
    if "n_channels" in config.network:
        config.network.n_channels = num_channels
    if "in_chans" in config.network:
        config.network.in_chans = num_channels
    log.info(f"[inference] Network input channels: {num_channels}")
    
    # Instantiate algorithm via hydra
    log.info(f"[inference] Instantiating algorithm: {config.algorithm._target_}")
    algorithm: AlgorithmBase = hydra.utils.instantiate(
        config.algorithm,
        network=None,
        network_conf=OmegaConf.to_yaml(config.network),
        optimizer_conf=None,
        scheduler_conf=None,
        gpu=gpu,
    )
    
    algorithm.cuda(gpu)
    log.info(f"[inference] Using GPU: {gpu}")
    
    # Process each checkpoint
    total_samples = 0
    for ckpt_idx, ckpt_path in enumerate(ckpt_paths):
        log.info(f"[inference] Processing checkpoint {ckpt_idx + 1}/{len(ckpt_paths)}: {ckpt_path}")
        
        # Load checkpoint weights
        ckpt = torch.load(ckpt_path, weights_only=False)
        algorithm.load_state_dict(ckpt["state_dict"])
        algorithm.eval()
        
        # Process each split for this checkpoint
        for split in splits:
            dataset = _get_dataset_by_split(datamodule=datamodule, split=split)
            if dataset is None:
                log.warning(f"[inference] Skipping split {split}: dataset not available")
                continue
            
            # Derive output directory for this split
            output_dir = parse_output_dir(
                ckpt_path=ckpt_path,
                predictions_dir=predictions_dir,
                datamodule_name=str(config.datamodule.name),
                split_name=split,
            )
            os.makedirs(output_dir, exist_ok=True)
            log.info(f"[inference] Processing split: {split} ({len(dataset)} samples)")
            log.info(f"[inference] Output directory: {output_dir}")
            
            # Collect CSV rows for test splits (Kaggle submission)
            csv_rows = []
            is_test_split = split.startswith("test")
            is_full_eval = "full_" in str(config.datamodule.name)
            
            # Run inference for this split
            with torch.no_grad():
                for idx in tqdm(range(len(dataset)), desc=f"Inference ({split})"):
                    # Get sample from dataset
                    batch = dataset[idx]
                    
                    # Run prediction
                    result = algorithm.pred(batch=batch)
                    
                    # Extract file name for output
                    sample_meta = batch[3]  # (inputs, targets, masks, sample_meta)
                    file_name = sample_meta["file_name"]
                    
                    # Clean file name for use as filename (remove path separators, etc.)
                    safe_file_name = re.sub(r"[^\w\-_.]", "_", str(file_name))
                    
                    if is_full_eval and is_test_split:
                        # Save PNG only for full eval test sets
                        # Remove existing extension if present to avoid double extension
                        base_name = os.path.splitext(safe_file_name)[0]
                        png_path = os.path.join(output_dir, f"{base_name}.png")
                        save_prediction_png(
                            pred=result["pred"],
                            output_path=png_path,
                            out_norm=config.algorithm.out_norm,
                        )
                    else:
                        # Original behavior: save .npz file
                        output_path = os.path.join(output_dir, f"{safe_file_name}.npz")
                        np.savez(
                            output_path,
                            pred=result["pred"],
                            inputs=result["inputs"],
                            targets=result["targets"],
                            masks=result["masks"],
                            file_name=result["sample"]["file_name"],
                            pixel_size=result["sample"]["pixel_size"],
                        )
                        
                        # Generate CSV rows for test splits (not full eval)
                        if is_test_split:
                            rows = generate_csv_rows(
                                file_name=file_name,
                                pred=result["pred"],
                                mask=result["masks"],
                            )
                            csv_rows.extend(rows)
            
            log.info(f"[inference] Saved {len(dataset)} predictions to {output_dir}")
            
            # Write CSV for test splits (Kaggle submission format) - skip for full eval
            if csv_rows and not is_full_eval:
                csv_path = os.path.join(output_dir, "predictions.csv")
                with open(csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["ID", "PL"])
                    for id_str, pl_value in csv_rows:
                        writer.writerow([id_str, round(pl_value, 1)])
                log.info(f"[inference] Saved CSV with {len(csv_rows)} rows to {csv_path}")
            total_samples += len(dataset)
    
    elapsed = time.perf_counter() - t0
    log.info(f"[inference] Completed in {elapsed:.2f}s ({total_samples} total samples across {len(ckpt_paths)} checkpoints, {len(splits)} splits)")
