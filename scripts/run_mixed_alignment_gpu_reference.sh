#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SUITE="${TBBCC_MIXED_SUITE:-benchmarks/model_zoo/suites/mixed_alignment_30min.json}"
AGENTS_CHAT_ROOT="${TBBCC_AGENTS_CHAT_ROOT:-$ROOT/../HTLsAgentsChat}"
MATERIALS_ROOT="${TBBCC_MATERIALS_ROOT:-$AGENTS_CHAT_ROOT/artifacts/paper2}"
OUT="${TBBCC_GPU_OUT:-$MATERIALS_ROOT/experiments/numeric-alignment/gpu-reference}"
MODEL_CACHE="${TBBCC_MODEL_CACHE:-/root/autodl-tmp/tbbcc_model_cache}"

export TBBCC_MODEL_CACHE="$MODEL_CACHE"

python scripts/tbbcc_model_suite.py validate --suite "$SUITE"
python scripts/tbbcc_model_suite.py plan --suite "$SUITE" --out "$OUT.plan.json"
python scripts/tbbcc_model_suite.py collect \
  --suite "$SUITE" \
  --role gpu-reference \
  --device cuda \
  --out "$OUT" \
  --keep-going

echo "GPU reference written to $OUT"
sha256sum "$OUT/manifest.json"
