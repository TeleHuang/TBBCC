#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SUITE="${TBBCC_MIXED_SUITE:-benchmarks/model_zoo/suites/mixed_alignment_30min.json}"
GPU_ROOT="${TBBCC_GPU_ROOT:-reports/mixed_alignment_gpu}"
NPU_ROOT="${TBBCC_NPU_ROOT:-reports/mixed_alignment_torch4ms_npu}"
OUT="${TBBCC_COMPARE_OUT:-reports/mixed_alignment_gpu_vs_npu}"

python scripts/tbbcc_model_suite.py compare \
  --suite "$SUITE" \
  --gpu-reference "$GPU_ROOT" \
  --npu-bridge "$NPU_ROOT" \
  --out "$OUT"

echo "Comparison summary written to $OUT/summary.json"
