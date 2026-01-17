#!/usr/bin/env python
"""
Evaluation script for Korean checkpoint.
Calculates RMSE on validation buildings using Hydra configs.
"""
import csv
import logging
import os
import sys
import tempfile

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.algorithms.mlsp import MLSP
from src.datamodules.mlsp import MLSPDatamodule

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


def create_filtered_manifest(
    source_manifest: str,
    buildings: list[int],
    output_path: str
) -> str:
    """Filter a manifest file to only include specified buildings."""
    with open(source_manifest, "r", newline="") as src:
        reader = csv.DictReader(src)
        fieldnames = reader.fieldnames

        with open(output_path, "w", newline="") as dst:
            writer = csv.DictWriter(dst, fieldnames=fieldnames)
            writer.writeheader()

            count = 0
            for row in reader:
                b = int(row["building"])
                if b in buildings:
                    writer.writerow(row)
                    count += 1

    log.info(f"Created filtered manifest with {count} samples for buildings {buildings}")
    return output_path


def load_algorithm(
    checkpoint_path: str,
    algorithm_cfg: DictConfig,
    network_cfg: DictConfig,
    device: torch.device
) -> MLSP:
    """Load MLSP algorithm from checkpoint using config."""
    network_conf = OmegaConf.to_yaml(network_cfg)

    # Instantiate compiled params from config
    compiled = hydra.utils.instantiate(algorithm_cfg.compiled)

    log.info(f"Loading MLSP algorithm from {checkpoint_path}")
    log.info(f"Network config: {network_cfg.depths}")

    # Create algorithm instance
    algorithm = MLSP(
        out_norm=float(algorithm_cfg.out_norm),
        use_sip2net=bool(algorithm_cfg.use_sip2net),
        sip2net_params=OmegaConf.to_container(algorithm_cfg.sip2net_params, resolve=True),
        compiled=compiled,
        optimizer_conf=None,
        scheduler_conf=None,
        network=None,
        network_conf=network_conf,
        gpu=None,
        finetune=OmegaConf.to_container(algorithm_cfg.finetune, resolve=True)
    )

    # Load checkpoint weights
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    # Remap keys: handle _network. prefix
    remapped = {}
    for k, v in state_dict.items():
        if k.startswith("_network."):
            remapped[k] = v
        else:
            # Add _network. prefix for network weights
            remapped[f"_network.{k}"] = v

    missing, unexpected = algorithm.load_state_dict(remapped, strict=False)
    if missing:
        log.warning(f"Missing keys: {len(missing)}")
    if unexpected:
        log.warning(f"Unexpected keys: {len(unexpected)}")

    algorithm = algorithm.to(device)
    algorithm.eval()
    log.info("Algorithm loaded successfully")
    return algorithm


def create_datamodule(
    datamodule_cfg: DictConfig,
    manifest_path: str,
    val_buildings: list[int]
) -> MLSPDatamodule:
    """Create MLSPDatamodule for validation using config."""
    # Create a temporary filtered manifest for validation buildings
    temp_dir = tempfile.mkdtemp()
    val_manifest = os.path.join(temp_dir, "val_manifest.csv")
    create_filtered_manifest(
        source_manifest=manifest_path,
        buildings=val_buildings,
        output_path=val_manifest
    )

    # Override config values for evaluation
    cfg = OmegaConf.to_container(datamodule_cfg, resolve=True)
    cfg["train_manifest_path"] = val_manifest
    cfg["val_manifest_path"] = val_manifest
    cfg["val_buildings"] = val_buildings
    cfg["inference"] = False
    cfg["multi_gpu"] = False
    cfg["batch_size"] = 1
    cfg["num_workers"] = 4
    # Disable augmentations for evaluation
    cfg["aug_p"] = 0.0
    cfg["modality_dropout_prob"] = 0.0
    cfg["sparse_dropout_given_dropout"] = 0.0

    # Remove _target_ as we instantiate directly
    cfg.pop("_target_", None)
    cfg.pop("name", None)

    datamodule = MLSPDatamodule(**cfg)
    return datamodule


@torch.no_grad()
def evaluate(
    algorithm: MLSP,
    datamodule: MLSPDatamodule,
    device: torch.device,
    out_norm: float
) -> dict:
    """Evaluate algorithm on validation set and compute RMSE."""
    algorithm.eval()

    # Get the validation set from datamodule
    val_set = datamodule.val_set
    if isinstance(val_set, tuple):
        val_set = val_set[0]

    log.info(f"Validation set size: {len(val_set)}")

    # Use datamodule's val_dataloader
    val_dataloaders = datamodule.val_dataloader()
    if isinstance(val_dataloaders, list):
        dataloader = val_dataloaders[-1]
    else:
        dataloader = val_dataloaders

    total_se = 0.0
    total_count = 0
    building_se = {}
    building_count = {}

    for batch in tqdm(dataloader, desc="Evaluating"):
        inputs, targets, masks, meta = batch

        # Move to device
        inputs = inputs.to(device)
        targets = targets.to(device)
        masks = masks.to(device)

        # Forward pass with bfloat16 (same as MLSP._step)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            preds = algorithm._network(inputs)

        # Squeeze output
        if preds.dim() == 4:
            preds = preds.squeeze(1)

        # Clamp prediction to [0, 1] (same as MLSP._step for validation)
        preds = torch.clamp(preds, 0.0, 1.0)

        # Compute squared error for each sample in batch
        for i in range(inputs.shape[0]):
            pred = preds[i]
            target = targets[i]
            mask = masks[i]

            se = ((pred - target) ** 2 * mask).sum().item()
            count = mask.sum().item()

            total_se += se
            total_count += count

            # Extract building from filename (format: B{num}_...)
            filename = meta["file_name"][i]
            b = int(filename.split("_")[0][1:])

            if b not in building_se:
                building_se[b] = 0.0
                building_count[b] = 0
            building_se[b] += se
            building_count[b] += count

    # Compute RMSE (same as MLSP.get_metrics)
    mse = total_se / (total_count + 1e-8)
    rmse_normalized = np.sqrt(mse)
    rmse_db = rmse_normalized * out_norm

    # Per-building RMSE
    building_rmse = {}
    for b in building_se:
        b_mse = building_se[b] / (building_count[b] + 1e-8)
        b_rmse = np.sqrt(b_mse) * out_norm
        building_rmse[b] = b_rmse

    return {
        "rmse_db": rmse_db,
        "mse": mse,
        "total_samples": len(val_set),
        "building_rmse": building_rmse
    }


@hydra.main(
    config_path="../configs/exps",
    config_name="eval",
    version_base="1.2"
)
def main(cfg: DictConfig) -> None:
    # Print resolved config
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # Load algorithm using configs
    algorithm = load_algorithm(
        checkpoint_path=cfg.checkpoint,
        algorithm_cfg=cfg.algorithm,
        network_cfg=cfg.network,
        device=device
    )

    # Manifest path
    data_dir = cfg.datamodule.data_dir
    manifest_path = os.path.join(data_dir, "icassp_manifest.csv")
    if not os.path.exists(manifest_path):
        log.error(f"Manifest not found: {manifest_path}")
        sys.exit(1)

    # Create datamodule with validation buildings from config
    datamodule = create_datamodule(
        datamodule_cfg=cfg.datamodule,
        manifest_path=manifest_path,
        val_buildings=list(cfg.val_buildings)
    )

    # Evaluate
    results = evaluate(
        algorithm=algorithm,
        datamodule=datamodule,
        device=device,
        out_norm=float(cfg.algorithm.out_norm)
    )

    # Print results
    log.info("=" * 60)
    log.info("EVALUATION RESULTS")
    log.info("=" * 60)
    log.info(f"Checkpoint: {cfg.checkpoint}")
    log.info(f"Network depths: {cfg.network.depths}")
    log.info(f"Validation buildings: {list(cfg.val_buildings)}")
    log.info(f"Total samples: {results['total_samples']}")
    log.info("-" * 60)
    log.info(f"Overall RMSE: {results['rmse_db']:.4f} dB")
    log.info("-" * 60)
    log.info("Per-building RMSE:")
    for b in sorted(results["building_rmse"].keys()):
        log.info(f"  Building {b}: {results['building_rmse'][b]:.4f} dB")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
