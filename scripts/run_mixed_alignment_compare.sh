#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SUITE="${TBBCC_MIXED_SUITE:-benchmarks/model_zoo/suites/mixed_alignment_30min.json}"
AGENTS_CHAT_ROOT="${TBBCC_AGENTS_CHAT_ROOT:-$ROOT/../HTLsAgentsChat}"
MATERIALS_ROOT="${TBBCC_MATERIALS_ROOT:-$AGENTS_CHAT_ROOT/artifacts/paper2}"
GPU_ROOT="${TBBCC_GPU_ROOT:-$MATERIALS_ROOT/experiments/numeric-alignment/gpu-reference}"
NPU_ROOT="${TBBCC_NPU_ROOT:-$MATERIALS_ROOT/experiments/numeric-alignment/npu-bridge}"
OUT="${TBBCC_COMPARE_OUT:-$MATERIALS_ROOT/results/numeric-alignment}"

python scripts/tbbcc_model_suite.py compare \
  --suite "$SUITE" \
  --gpu-reference "$GPU_ROOT" \
  --npu-bridge "$NPU_ROOT" \
  --out "$OUT"

echo "Comparison summary written to $OUT/summary.json"
