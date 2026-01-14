#!/bin/bash
# Evaluate Korean checkpoint on validation buildings (B21-25)
# Uses Hydra configs for algorithm, network, and datamodule

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python scripts/eval_korean_checkpoint.py \
    checkpoint=/nfs/dgx/raid/iot/mlsp_wair_d_data/torch_korean_runs/2026-01-10_18-10-45_txunet/best.ckpt \
    network.depths="[4,6,6,8]" \
    val_buildings="[21,22,23,24,25]" \
    datamodule.data_dir=/nfs/dgx/raid/iot/mlsp_wair_d_data/icassp2025 \
    device=cuda
