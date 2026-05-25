---
description: Generate an adapter from bridge docs, run TorchBridgeBench, and produce a bridge evaluation report
argument-hint: "--bridge-id <id> --docs <path> [--suite <suite.json>] [--out <dir>] [--ar-baseline <baseline.json>]"
allowed-tools: [Bash, Read, Write, Edit, Glob, Grep]
---

# TorchBridgeBench `/eval`

Run a full Track A bridge evaluation from inside Claude Code TUI. This command
starts from adapter authoring; the user should not need to prepare an adapter by
hand.

The user invoked:

```text
/eval $ARGUMENTS
```

## User-Facing Usage

```text
/eval --bridge-id torch4ms --docs docs/torch4ms.md --suite benchmarks/v1.0.0/suites/smoke_noop.json --out reports/torch4ms_eval
```

Required:

- `--bridge-id <id>`: short bridge identifier used in reports.
- `--docs <path>`: bridge documentation, README, install note, or adapter guide.

Optional:

- `--suite <suite.json>`: defaults to
  `${CLAUDE_PLUGIN_ROOT}/benchmarks/v1.0.0/suites/smoke_noop.json`.
- `--out <dir>`: defaults to a timestamped directory under
  `${CLAUDE_PLUGIN_ROOT}/reports/eval_<bridge-id>_<timestamp>`.
- `--ar-baseline <baseline.json>`: defaults to
  `${CLAUDE_PLUGIN_ROOT}/reports/ar_baseline/deepseek-v4-pro-cc-batch-v1/baseline.json`
  if that file exists.

## Workflow

1. Parse `$ARGUMENTS`. If required arguments are missing, ask for them in one
   concise message.
2. Resolve paths relative to the current working directory first, then relative
   to `${CLAUDE_PLUGIN_ROOT}`.
3. Read:
   - `${CLAUDE_PLUGIN_ROOT}/references/data-contracts.md`
   - `${CLAUDE_PLUGIN_ROOT}/references/agent-workflow.md`
   - the supplied bridge docs
4. Create the output directory.
5. Write `<out>/adapter.generated.json`.
6. Write `<out>/effort_ledger.json` with at least one `phase: "adapt"` entry.
7. Write `<out>/suite.generated.json`, copying the selected suite cases and
   replacing `adapters` with `<out>/adapter.generated.json`.
8. Validate the generated adapter:

   ```bash
   python ${CLAUDE_PLUGIN_ROOT}/scripts/tbbcc.py validate-inputs \
     --adapter <out>/adapter.generated.json
   ```

9. Run evaluation:

   ```bash
   python ${CLAUDE_PLUGIN_ROOT}/scripts/tbbcc.py eval-suite \
     --suite <out>/suite.generated.json \
     --out <out>/eval \
     --effort-ledger <out>/effort_ledger.json \
     --ar-baseline <baseline-if-available>
   ```

10. Read `<out>/eval/summary.json` and `<out>/eval/summary.md` before
    answering.

## AdapterSpec Requirements

Generate this shape:

```json
{
  "bridge_id": "<bridge-id>",
  "track": "intercept",
  "preamble": "<bridge import/startup code>",
  "source_preamble": "",
  "env": {},
  "atol": 1e-5,
  "rtol": 1e-5,
  "timeout_seconds": 120,
  "docs": "<docs path or short docs summary>",
  "known_gaps": []
}
```

The `preamble` must:

- import and initialize the bridge according to the docs;
- avoid editing bridge internals;
- fail with clear import/configuration errors if the bridge package is absent;
- keep environment variables in `env` where possible.

## Effort Ledger Requirements

Write `<out>/effort_ledger.json`:

```json
{
  "schema_version": "tbbcc.effort_ledger.v0.2",
  "entries": [
    {
      "bridge_id": "<bridge-id>",
      "phase": "adapt",
      "rounds": 1,
      "prompt_chars": 0,
      "completion_chars": 0,
      "edit_units": 0,
      "measurement": "local_char_proxy",
      "classification_confirmation_counted": false,
      "environment_remediation_counted": false
    }
  ],
  "migration_samples": []
}
```

If exact Claude Code telemetry is unavailable, estimate `prompt_chars` from the
docs plus task prompt, `completion_chars` from generated text, and `edit_units`
from generated file character count. State that `local_char_proxy` was used.

## Failure Handling

- If the generated adapter fails because the bridge package is missing, keep the
  report. It should be classified as dependency/environment failure and excluded
  from compatibility.
- If the failure is bridge-relevant, confirm classification before repair.
- Repair only adapter or migration-side files.
- Append repair effort as `phase: "repair"` and rerun the modified case or
  suite.
- Do not count environment remediation or independent classification
  confirmation as effort.

## Final Answer

Return only:

- generated adapter path;
- summary report path;
- compatibility rate;
- first-pass rate;
- ME / AR if available;
- dominant failure class if any;
- whether environment failures were excluded.

