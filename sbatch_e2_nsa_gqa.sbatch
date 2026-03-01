#!/usr/bin/env bash
#SBATCH --job-name=e2_nsa_gqa
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=160
#SBATCH --mem=0
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

WORK_DIR="${WORK_DIR:-$HOME/mlsp_model/dev-clean}"
mkdir -p "${WORK_DIR}/logs"
cd "${WORK_DIR}"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

export ICASSP_TRAIN_MANIFEST="${ICASSP_TRAIN_MANIFEST:-${ICASSP_TRAIN_MANIFEST_TASK_1:-}}"
export ICASSP_TRAIN_SMALL_MANIFEST="${ICASSP_TRAIN_SMALL_MANIFEST:-${ICASSP_TRAIN_SMALL_MANIFEST_TASK_1:-}}"
export ICASSP_VAL_MANIFEST="${ICASSP_VAL_MANIFEST:-${ICASSP_VAL_MANIFEST_TASK_1:-}}"
export WANDB_SAVE_DIR="${WANDB_SAVE_DIR:-${WORK_DIR}/runs/wandb}"
mkdir -p "${WANDB_SAVE_DIR}"

export HYDRA_FULL_ERROR=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO

nvidia-smi -pm 1 2>/dev/null || true

echo "[$(date)] Starting e2_nsa_gqa (8-GPU DDP)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

torchrun --standalone --nproc_per_node=8 run.py --config-name=train_synthetic \
  strategy=ddp \
  'trainer.devices=8' \
  '+network.nsa_enabled=True' \
  'network.sra0_enabled=False' \
  '+network.nsa_gqa_group_size=4' \
  '+network.rope_enabled=True' \
  '+network.rope_base=10000.0' \
  'loggers.wandb.name=e2_nsa_gqa4_8gpu'
