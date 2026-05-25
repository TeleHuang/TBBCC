# AR Baseline Handoff

This note hands off the completed AR baseline calibration for the Claude Code
TorchBridgeBench system.

## Result

- Baseline artifact:
  `reports/ar_baseline/deepseek-v4-pro-cc-batch-v1/baseline.json`
- Raw samples:
  `reports/ar_baseline/deepseek-v4-pro-cc-batch-v1/samples.jsonl`
- Human summary:
  `reports/ar_baseline/deepseek-v4-pro-cc-batch-v1/baseline.md`
- `baseline_effort`: `2756.8809999999994`
- Covered cases: `175/175`
- Batch records: `44/44`
- Error count: `0`
- Rerolls: `1`
- Task set digest:
  `19e425d7f3f4bcc64e0f27d93a7ee4519822e82ed7e95d873e952acf18dd691e`

## Identity Metadata

Use this constant only with the matching agent/model/task identity:

- `model`: `deepseek-v4-pro`
- `provider`: `claude-code`
- `agent_system`: `Claude Code`
- `agent_system_version`: `claude-code-2.1.150-batch-deepseek-v4-pro`
- `claude_version`: `2.1.150 (Claude Code)`
- `protocol`: `claude-code-cli-print-json-batch`
- `effort_formula_version`: `shared-effort-v1`
- `effort_formula`:
  `rounds + (prompt_chars + completion_chars + edit_units) / 1000.0`

Important caveat: the target model argument was `deepseek-v4-pro[1m]`, but
Claude Code `modelUsage` observed both `deepseek-v4-pro[1m]` and
`deepseek-v4-flash[1m]` in some calls. Therefore describe the result as a
Claude Code baseline with target model `deepseek-v4-pro`, retaining
`model_observed`; do not claim it is a pure pro-only provider trace.

## AR Formula

For any evaluated migration run:

```text
AR = max(0, 1 - ME / baseline_effort)
```

For this baseline:

```text
AR = max(0, 1 - ME / 2756.8809999999994)
```

`ME` must be measured with the same `shared-effort-v1` formula:

```text
ME = rounds + (prompt_chars + completion_chars + edit_units) / 1000.0
```

The current helper implementation is in `scripts/tbbcc_metrics.py`:

- `calc_effort_total(...)`
- `calc_ar(me, baseline_effort)`
- `summarize_effort_ledger(entries, baseline_effort)`

## Effort Ledger Input

The deterministic core can read an optional effort ledger via `--effort-ledger`.
The ledger must include `baseline_effort` and an `entries` array.

Minimal example:

```json
{
  "baseline_effort": 2756.8809999999994,
  "entries": [
    {
      "case_id": "bench_v1.0.0/L1/act/relu",
      "bridge_id": "my-bridge",
      "phase": "adapt",
      "rounds": 1,
      "prompt_chars": 1200,
      "completion_chars": 800,
      "edit_units": 120
    },
    {
      "case_id": "bench_v1.0.0/L1/act/relu",
      "bridge_id": "my-bridge",
      "phase": "repair",
      "rounds": 1,
      "prompt_chars": 600,
      "completion_chars": 500,
      "edit_units": 40,
      "attempts": 1
    }
  ]
}
```

Each entry may alternatively provide a precomputed `effort` value. Otherwise the
core computes it from `rounds`, `prompt_chars`, `completion_chars`, and
`edit_units` or `diff_size`.

`phase` should be:

- `adapt` for initial migration/adaptation work,
- `repair` for repair work after confirmation/evaluation.

The core filters ledger entries by matching `case_id` and `bridge_id` when those
fields are present.

## How to Consume

Single case:

```bash
python scripts/tbbcc.py eval \
  --case <case.json> \
  --adapter <adapter.json> \
  --out <report-dir> \
  --effort-ledger <ledger.json>
```

Suite:

```bash
python scripts/tbbcc.py eval-suite \
  --suite benchmarks/v1.0.0/suites/all_noop.json \
  --out <suite-report-dir> \
  --effort-ledger <ledger.json>
```

Reports will include:

- `effort_adapt`
- `effort_repair`
- `effort_total`
- `me`
- `baseline_effort`
- `ar`
- `repair_attempts`
- effort confidence interval
- `migrate_at_k`

## Implementation Notes for the Next Session

1. Treat `baseline.json` as the authoritative calibration artifact.
2. Load `baseline_effort` and metadata from that artifact, not from a hardcoded
   constant when possible.
3. Preserve `model`, `model_observed`, `provider`, `protocol`,
   `agent_system_version`, `task_set_digest`, and `effort_formula_version` in
   downstream result metadata.
4. Do not recompute the baseline unless changing model, agent system, task set,
   protocol, or effort formula.
5. If implementing a Python Agent system separate from Claude Code, calibrate a
   separate baseline constant for that system.

