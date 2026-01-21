import logging
import os
import re
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.algorithms.algorithm_base import AlgorithmBase
from src.datamodules.mlsp import MLSPDatamodule

log = logging.getLogger(__name__)


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
        datamodule_name,
        split_name,
        timestamp,
        ckpt_name,
    )
    return output_dir


def inference_prep(
    config: DictConfig,
    project_root: str,
) -> None:
    """
    Run inference on a dataset split using a trained checkpoint.
    
    Steps:
    1. Parse checkpoint path to derive output directory
    2. Instantiate datamodule for chosen split
    3. Instantiate algorithm and load checkpoint weights
    4. Loop through dataset samples and save predictions as .npz files
    """
    t0 = time.perf_counter()
    
    # Get inference parameters
    ckpt_path = os.path.abspath(str(config["ckpt_path"]))
    gpu = int(config.get("gpu", 0))
    split = str(config.get("split", "val"))
    predictions_dir = os.path.expanduser(str(config["predictions_dir"]))
    
    # Validate checkpoint exists
    if not os.path.isfile(ckpt_path):
        raise RuntimeError(f"Checkpoint not found: {ckpt_path}")
    
    # Derive output directory
    output_dir = parse_output_dir(
        ckpt_path=ckpt_path,
        predictions_dir=predictions_dir,
        datamodule_name=str(config.datamodule.name),
        split_name=split,
    )
    os.makedirs(output_dir, exist_ok=True)
    log.info(f"[inference] Output directory: {output_dir}")
    log.info(f"[inference] Using split: {split}")
    
    # Instantiate datamodule via hydra
    log.info(f"[inference] Instantiating datamodule: {config.datamodule._target_}")
    datamodule: MLSPDatamodule = hydra.utils.instantiate(
        config.datamodule,
        multi_gpu=False,
    )
    
    # Select dataset based on split
    if split == "train":
        dataset = datamodule.train_set
    elif split == "val":
        dataset = datamodule.val_set
    elif split == "test":
        dataset = datamodule.test_set
    else:
        raise ValueError(f"Invalid split: {split}. Must be one of: train, val, test")
    
    log.info(f"[inference] Dataset size: {len(dataset)}")
    
    # Compute num_channels from datamodule.channels
    num_channels = len(config.datamodule.channels)
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
    
    # Load checkpoint weights
    log.info(f"[inference] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=False)
    algorithm.load_state_dict(ckpt["state_dict"])
    
    algorithm.eval()
    algorithm.cuda(gpu)
    log.info(f"[inference] Using GPU: {gpu}")
    
    # Run inference
    log.info(f"[inference] Starting inference on {len(dataset)} samples...")
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Inference"):
            # Get sample from dataset
            batch = dataset[idx]
            
            # Run prediction
            result = algorithm.pred(batch=batch)
            
            # Extract file name for output
            sample_meta = batch[3]  # (inputs, targets, masks, sample_meta)
            file_name = sample_meta["file_name"]
            
            # Clean file name for use as filename (remove path separators, etc.)
            safe_file_name = re.sub(r"[^\w\-_.]", "_", str(file_name))
            
            # Save to .npz file
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
    
    elapsed = time.perf_counter() - t0
    log.info(f"[inference] Completed in {elapsed:.2f}s")
    log.info(f"[inference] Saved {len(dataset)} predictions to {output_dir}")
