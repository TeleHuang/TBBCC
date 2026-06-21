---
name: "ar-baseline"
description: "Calibrate and audit TorchBridgeBench AR baseline constants through Claude Code. Use when the user asks for AR baseline_effort, baseline constants, migration-effort calibration, or model/agent-system-specific AR data."
argument-hint: "[--suite benchmarks/v1.0.0/suites/all_noop.json] [--out reports/ar_baseline/deepseek-v4-pro-cc-v1] [--rerolls 1]"
allowed-tools: [Bash, Read, Write, Edit, Glob, Grep]
---

# AR Baseline Calibration

Use this skill to produce the `baseline_effort` constant for:

```text
AR = 1 - ME / baseline_effort
```

The constant is valid only for the benchmark task set, model, agent system, and
effort formula recorded in the output JSON. Do not reuse it across a different
model or agent system without recalibration.

## Required Identity

For the current TorchBridgeBench Claude Code baseline, record:

- `model`: `deepseek-v4-pro`
- `provider`: `claude-code`
- `agent_system`: `Claude Code`
- `protocol`: `claude-code-cli-print-json-bare`
- `effort_formula_version`: `shared-effort-v1`

The raw Claude Code `modelUsage` must be retained per sample. If `modelUsage`
shows models other than `deepseek-v4-pro[1m]`, report that honestly instead of
calling the result a pure `deepseek-v4-pro` baseline.

## Workflow

1. Verify Claude Code can run with the target model:

   ```bash
   claude -p 'Return exactly OK.' \
     --bare --tools '' \
     --output-format json \
     --no-session-persistence \
     --model 'deepseek-v4-pro[1m]'
   ```

2. Run calibration from the plugin root:

   ```bash
   python ${CLAUDE_PLUGIN_ROOT}/scripts/calibrate_ar_cc.py \
     --suite ${CLAUDE_PLUGIN_ROOT}/benchmarks/v1.0.0/suites/all_noop.json \
     --out ${CLAUDE_PLUGIN_ROOT}/reports/ar_baseline/deepseek-v4-pro-cc-v1 \
     --rerolls 1 \
     --workers 1 \
     --model 'deepseek-v4-pro[1m]'
   ```

3. For a stability check, use `--rerolls 2` or higher. A single reroll is an
   exhaustive case-set pass but does not estimate variance.

4. If interrupted, rerun the same command. The script resumes from
   `samples.jsonl` unless `--no-resume` is set.

5. Inspect:

   ```bash
   python -m json.tool ${CLAUDE_PLUGIN_ROOT}/reports/ar_baseline/deepseek-v4-pro-cc-v1/baseline.json >/dev/null
   sed -n '1,120p' ${CLAUDE_PLUGIN_ROOT}/reports/ar_baseline/deepseek-v4-pro-cc-v1/baseline.md
   ```

## Outputs

The calibration directory contains:

- `samples.jsonl`: append-only raw per-case samples, including translated code,
  Claude Code session ids, costs, usage, and observed model usage.
- `baseline.json`: compact baseline constant and audit metadata.
- `baseline.md`: human-readable summary.

Do not commit `reports/` outputs unless the user explicitly requests publishing
the generated calibration artifact. The directory is intentionally ignored by
git.

## Validity Checks

Before concluding, confirm:

- `sample_count == expected_sample_count`
- `error_count == 0`
- `baseline_effort` is not null
- `model_observed` only contains `deepseek-v4-pro[1m]` for a pure pro baseline
- `source_cases` matches the intended suite, normally all 175 benchmark cases
