#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SUITE="${TBBCC_MIXED_SUITE:-benchmarks/model_zoo/suites/mixed_alignment_30min.json}"
AGENTS_CHAT_ROOT="${TBBCC_AGENTS_CHAT_ROOT:-$ROOT/../HTLsAgentsChat}"
MATERIALS_ROOT="${TBBCC_MATERIALS_ROOT:-$AGENTS_CHAT_ROOT/artifacts/paper2}"
GPU_ROOT="${TBBCC_GPU_ROOT:-$MATERIALS_ROOT/experiments/numeric-alignment/gpu-reference}"
OUT="${TBBCC_NPU_OUT:-$MATERIALS_ROOT/experiments/numeric-alignment/npu-bridge}"
ADAPTER="${TBBCC_ADAPTER:-$MATERIALS_ROOT/experiments/numeric-alignment/config/adapter.generated.json}"
MODEL_CACHE="${TBBCC_MODEL_CACHE:-/home/ma-user/work/tbbcc_model_cache}"
BUDGET="${TBBCC_NPU_BUDGET_SECONDS:-1800}"
MAX_MODELS="${TBBCC_NPU_MAX_MODELS:-4}"
ISOLATE_MODELS="${TBBCC_NPU_ISOLATE_MODELS:-1}"

export TBBCC_MODEL_CACHE="$MODEL_CACHE"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1}"
export TBBCC_LLM_DEVICE_MAP="${TBBCC_LLM_DEVICE_MAP:-auto}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"

python scripts/tbbcc_model_suite.py validate --suite "$SUITE"
COLLECT_ARGS=(python scripts/tbbcc_model_suite.py collect
  --suite "$SUITE"
  --role npu-bridge
  --adapter "$ADAPTER"
  --input-root "$GPU_ROOT"
  --time-budget-seconds "$BUDGET"
  --max-models "$MAX_MODELS"
  --out "$OUT"
  --keep-going)
if [[ "$ISOLATE_MODELS" != "0" ]]; then
  COLLECT_ARGS+=(--isolate-models)
fi
"${COLLECT_ARGS[@]}"

echo "NPU bridge artifacts written to $OUT"
