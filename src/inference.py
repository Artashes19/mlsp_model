import logging
import os
import re
import time
from typing import Optional

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset
from tqdm import tqdm

from src.algorithms.algorithm_base import AlgorithmBase
from src.datamodules.indoor import IndoorDatamodule

log = logging.getLogger(__name__)


def _get_dataset_by_split(
    datamodule: IndoorDatamodule,
    split: str,
) -> Optional[Dataset]:
    """
    Get dataset by split name. Returns None if split is not available.
    """
    if split == "train":
        return datamodule.train_set
    elif split == "test":
        # test_sets is a list; return first if available
        return datamodule.test_sets[0] if datamodule.test_sets else None
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
    Run inference on multiple dataset splits using a trained checkpoint.
    
    Steps:
    1. Parse checkpoint path and splits from config
    2. Instantiate datamodule and algorithm (once)
    3. Load checkpoint weights
    4. For each split: loop through dataset samples and save predictions as .npz files
    """
    t0 = time.perf_counter()
    
    # Get inference parameters
    ckpt_path = os.path.abspath(str(config["ckpt_path"]))
    gpu = int(config.get("gpu", 0))
    splits = list(config["split"])  # Must be a list
    predictions_dir = os.path.expanduser(str(config["predictions_dir"]))
    
    # Validate checkpoint exists
    if not os.path.isfile(ckpt_path):
        raise RuntimeError(f"Checkpoint not found: {ckpt_path}")
    
    log.info(f"[inference] Splits to process: {splits}")
    
    # Instantiate datamodule via hydra
    log.info(f"[inference] Instantiating datamodule: {config.datamodule._target_}")
    datamodule: IndoorDatamodule = hydra.utils.instantiate(
        config.datamodule,
        multi_gpu=False,
    )
    
    # Compute num_channels from datamodule.channels
    # Note: 'f' channel expands to 4 Fourier frequency channels
    channel_str = config.datamodule.channels
    num_channels = len(channel_str)
    if "f" in channel_str:
        num_channels += 3  # 'f' expands to 4 channels (3 extra)
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
    
    # Process each split
    total_samples = 0
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
        
        log.info(f"[inference] Saved {len(dataset)} predictions to {output_dir}")
        total_samples += len(dataset)
    
    elapsed = time.perf_counter() - t0
    log.info(f"[inference] Completed in {elapsed:.2f}s ({total_samples} total samples across {len(splits)} splits)")
