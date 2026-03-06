#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/artifacts/nsa_diagnostics}"
PYTHON_BIN="${PYTHON_BIN:-/auto/home/artashes/miniconda3/envs/dev/bin/python}"
NCU_BIN="${NCU_BIN:-/usr/local/cuda/bin/ncu}"

CONFIG_SPEC="${CONFIG_SPEC:-384:6:3}"
SIZE="${SIZE:-256}"
TOP_N="${TOP_N:-8}"
WARMUP="${WARMUP:-1}"
ITERS="${ITERS:-1}"
KERNEL_REGEX="${KERNEL_REGEX:-regex:_sel_perq_bwd_dq_kernel}"
GPU_LABEL="${GPU_LABEL:-${CUDA_VISIBLE_DEVICES:-unset}}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
KERNEL_SLUG="$(echo "$KERNEL_REGEX" | tr -cs '[:alnum:]' '_')"
PREFIX="${OUT_DIR}/selection_ncu_gpu${GPU_LABEL}_${KERNEL_SLUG}_C${CONFIG_SPEC//:/_}_S${SIZE}_top${TOP_N}_${TIMESTAMP}"
TRACE_SCRIPT="${ROOT_DIR}/artifacts/nsa_diagnostics/profile_nsa_selection_triton_hotspots.py"

mkdir -p "$OUT_DIR"

"$NCU_BIN" \
  --target-processes all \
  --kernel-name-base function \
  --kernel-name "$KERNEL_REGEX" \
  --launch-count 1 \
  --set detailed \
  --section SchedulerStats \
  --section SpeedOfLight_RooflineChart \
  --section SourceCounters \
  --apply-rules yes \
  --import-source yes \
  --force-overwrite \
  -o "$PREFIX" \
  "$PYTHON_BIN" "$TRACE_SCRIPT" \
    --configs "$CONFIG_SPEC" \
    --sizes "$SIZE" \
    --top-ns "$TOP_N" \
    --warmup "$WARMUP" \
    --iters "$ITERS"

REPORT_PATH="${PREFIX}.ncu-rep"

"$NCU_BIN" \
  -i "$REPORT_PATH" \
  --page details \
  --print-details all > "${PREFIX}_details.txt"

"$NCU_BIN" \
  -i "$REPORT_PATH" \
  --page source \
  --print-source cuda,sass \
  --resolve-source-file "$ROOT_DIR" > "${PREFIX}_source.txt"

echo "Saved Nsight Compute report: ${REPORT_PATH}"
echo "Saved Nsight Compute details: ${PREFIX}_details.txt"
echo "Saved Nsight Compute source view: ${PREFIX}_source.txt"
