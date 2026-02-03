#!/usr/bin/env bash
#SBATCH --job-name=icassp2_8x
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=75G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-7%8
#SBATCH --output=/home/indoor/mlsp_wair_d/logs/%x_%A_%a.out
#SBATCH --error=/home/indoor/mlsp_wair_d/logs/%x_%A_%a.err

set -euo pipefail

mkdir -p /home/indoor/mlsp_wair_d/logs

# --- activate env ---
# set +u
# source ~/.bashrc
# conda init
# conda activate indooro
# set -u

cd /home/indoor/mlsp_wair_d/

PY_CODE='import os,sys,shlex,runpy; sys.argv=["run.py", *shlex.split(os.environ["HYDRA_ARGS"])]; runpy.run_path("run.py", run_name="__main__")'

CKPT_ROOT="/data/indoor/outputs/2026-01-28_02-15-05.415620/e2/checkpoints/every"

CKPTS=(
  "${CKPT_ROOT}/epoch_0009_every.ckpt"
  "${CKPT_ROOT}/epoch_0019_every.ckpt"
  "${CKPT_ROOT}/epoch_0039_every.ckpt"
  "${CKPT_ROOT}/epoch_0079_every.ckpt"
  "${CKPT_ROOT}/epoch_0159_every.ckpt"
  "${CKPT_ROOT}/epoch_0319_every.ckpt"
  "${CKPT_ROOT}/epoch_0639_every.ckpt"
  ""
)

EXP_KEYS=(
  "icassp_task2_finetune"
  "icassp_task2_finetune"
  "icassp_task2_finetune"
  "icassp_task2_finetune"
  "icassp_task2_finetune"
  "icassp_task2_finetune"
  "icassp_task2_finetune"
  "icassp_task2"
)

i="${SLURM_ARRAY_TASK_ID}"
EXP_KEY="${EXP_KEYS[$i]}"

COMMON_ARGS=(
  "exps=[${EXP_KEY}]"
  "++exps.${EXP_KEY}.datamodule.batch_size=15"
  "++exps.${EXP_KEY}.datamodule.num_workers=8"
)

EXTRA_ARGS=()
if [[ -n "${CKPTS[$i]}" ]]; then
  EXTRA_ARGS+=("++exps.${EXP_KEY}.ckpt_path=${CKPTS[$i]}")
fi

HYDRA_ARGS="${COMMON_ARGS[*]} ${EXTRA_ARGS[*]}"
export HYDRA_ARGS

echo "[$(date)] task_id=$i"
echo "exp_key=${EXP_KEY}"
echo "ckpt=${CKPTS[$i]}"
echo "HYDRA_ARGS=$HYDRA_ARGS"

python -c "$PY_CODE"
