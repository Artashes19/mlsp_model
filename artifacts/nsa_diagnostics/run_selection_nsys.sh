#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/artifacts/nsa_diagnostics}"
PYTHON_BIN="${PYTHON_BIN:-/auto/home/artashes/miniconda3/envs/dev/bin/python}"
NSYS_BIN="${NSYS_BIN:-nsys}"

CONFIG_SPEC="${CONFIG_SPEC:-384:6:3}"
SIZE="${SIZE:-256}"
TOP_N="${TOP_N:-8}"
WARMUP="${WARMUP:-2}"
ITERS="${ITERS:-3}"
GPU_LABEL="${GPU_LABEL:-${CUDA_VISIBLE_DEVICES:-unset}}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
PREFIX="${OUT_DIR}/selection_nsys_gpu${GPU_LABEL}_C${CONFIG_SPEC//:/_}_S${SIZE}_top${TOP_N}_${TIMESTAMP}"
TRACE_SCRIPT="${ROOT_DIR}/artifacts/nsa_diagnostics/profile_nsa_selection_triton_hotspots.py"

mkdir -p "$OUT_DIR"

"$NSYS_BIN" profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  --sample=process-tree \
  --cpuctxsw=process-tree \
  --python-backtrace=cuda \
  --cuda-graph-trace=node \
  -o "$PREFIX" \
  "$PYTHON_BIN" "$TRACE_SCRIPT" \
    --configs "$CONFIG_SPEC" \
    --sizes "$SIZE" \
    --top-ns "$TOP_N" \
    --warmup "$WARMUP" \
    --iters "$ITERS"

REPORT_PATH="${PREFIX}.nsys-rep"

"$NSYS_BIN" stats \
  --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum \
  "$REPORT_PATH" > "${PREFIX}_stats.txt"

"$NSYS_BIN" stats \
  --format csv \
  --output "${PREFIX}_stats" \
  --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum \
  "$REPORT_PATH" >/dev/null

echo "Saved Nsight Systems report: ${REPORT_PATH}"
echo "Saved Nsight Systems stats: ${PREFIX}_stats.txt"
