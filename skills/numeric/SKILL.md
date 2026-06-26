---
name: numeric
description: Run TorchBridgeBench numeric-only GPU-vs-NPU model artifact comparison without generating adapters, suites, or running bridge evaluation.
argument-hint: 仅数值比对，可指定 GPU artifact、NPU artifact、suite、输出目录
allowed-tools: [Bash, Read, Glob, Grep]
---

# TorchBridgeBench Numeric-Only Comparison

Use this command when the user asks for `仅数值比对`, `numeric-only`,
`GPU-vs-NPU 数值比对`, `FNE`, `数值校验`, or wants to inspect already collected
GPU reference and NPU bridge artifacts.

## Scope

This workflow only compares existing artifacts. It must not:

- generate or rewrite bridge adapters;
- generate or rewrite benchmark suites;
- run `eval-suite`;
- collect GPU or NPU model artifacts unless the user explicitly asks for
  collection.

## Defaults

Use these defaults unless the user provides another path:

- suite: `${CLAUDE_PLUGIN_ROOT}/benchmarks/model_zoo/suites/mixed_alignment_30min.json`
- GPU root: `${CLAUDE_PLUGIN_ROOT}/reports/mixed_alignment_gpu`
- NPU root: newest `${CLAUDE_PLUGIN_ROOT}/reports/mixed_alignment_torch4ms_npu*`
  directory that is not marked failed, or `TBBCC_NPU_ROOT` when provided
- output: `${CLAUDE_PLUGIN_ROOT}/reports/mixed_alignment_gpu_vs_npu_numeric`

## Commands

Run from the plugin root:

```bash
cd ${CLAUDE_PLUGIN_ROOT}
TBBCC_GPU_ROOT=<gpu-root> \
TBBCC_NPU_ROOT=<npu-root> \
TBBCC_COMPARE_OUT=<out-dir> \
bash ${CLAUDE_PLUGIN_ROOT}/scripts/run_numeric_compare_only.sh
```

If the user does not provide paths, omit the environment variables and let the
wrapper use defaults or discover the newest NPU artifact root.

## Interpretation

The wrapper returns success when a comparison report is written, even if strict
numeric thresholds fail. Numerical pass/fail is data inside the report, not a
shell execution failure. Use `TBBCC_COMPARE_STRICT_EXIT=1` only when the user
explicitly asks for CI-style strict exit behavior.

Always read:

- `<out-dir>/summary.json`
- `<out-dir>/source_data/model_summary.csv`
- `<out-dir>/source_data/layerwise_fne.csv`

Report:

- `benchmark_verdict`
- totals: expected, compared, aligned, usable_with_drift, outlier_dominated,
  diverged, missing
- per-model `numerical_verdict`
- first divergence / first quality-drop layer
- missing model `npu_error` when present
- source-data CSV paths

Do not describe `eval-suite` local-pair metrics as formal GPU-vs-NPU numeric
accuracy. Formal numeric accuracy uses GPU reference artifacts versus NPU bridge
artifacts through this comparison path.
