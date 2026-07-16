#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SUITE="${TBBCC_NUMERIC_SUITE:-benchmarks/model_zoo/suites/mixed_alignment_30min.json}"
AGENTS_CHAT_ROOT="${TBBCC_AGENTS_CHAT_ROOT:-$ROOT/../HTLsAgentsChat}"
MATERIALS_ROOT="${TBBCC_MATERIALS_ROOT:-$AGENTS_CHAT_ROOT/artifacts/paper2}"
GPU_ROOT="${TBBCC_GPU_ROOT:-$MATERIALS_ROOT/experiments/numeric-alignment/gpu-reference}"
NPU_ROOT="${TBBCC_NPU_ROOT:-$MATERIALS_ROOT/experiments/numeric-alignment/npu-bridge}"
OUT="${TBBCC_COMPARE_OUT:-$MATERIALS_ROOT/results/numeric-alignment}"
STRICT_EXIT="${TBBCC_COMPARE_STRICT_EXIT:-0}"

if [ -z "$NPU_ROOT" ] || [ ! -d "$NPU_ROOT" ]; then
  echo "Missing NPU artifact root. Set TBBCC_NPU_ROOT=/path/to/npu-artifacts." >&2
  exit 2
fi
if [ ! -d "$GPU_ROOT" ]; then
  echo "Missing GPU reference root: $GPU_ROOT" >&2
  exit 2
fi

python scripts/tbbcc_model_suite.py validate --suite "$SUITE"

set +e
python scripts/tbbcc_model_suite.py compare \
  --suite "$SUITE" \
  --gpu-reference "$GPU_ROOT" \
  --npu-bridge "$NPU_ROOT" \
  --out "$OUT"
compare_status=$?
set -e

if [ ! -f "$OUT/summary.json" ]; then
  exit "$compare_status"
fi

python - "$OUT/summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
data = json.loads(summary_path.read_text(encoding="utf-8"))
print(json.dumps(
    {
        "summary_json": str(summary_path.resolve()),
        "summary_md": data.get("human_report") or str((summary_path.parent / "summary.md").resolve()),
        "figures": data.get("figure_outputs") or [],
        "benchmark_verdict": data.get("benchmark_verdict"),
        "totals": data.get("totals"),
        "source_data": [
            str((summary_path.parent / "source_data" / "layerwise_fne.csv").resolve()),
            str((summary_path.parent / "source_data" / "model_summary.csv").resolve()),
        ],
    },
    indent=2,
    ensure_ascii=False,
))
PY

if [ "$STRICT_EXIT" = "1" ]; then
  exit "$compare_status"
fi
exit 0
