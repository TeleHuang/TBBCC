---
description: Run TorchBridgeBench evaluation cases or suites
argument-hint: "--suite <suite.json> --out <dir> | --case <case.json> --adapter <adapter.json> --out <dir>"
allowed-tools: [Bash, Read, Glob, Grep]
---

# TorchBridgeBench Evaluation Command

Run the TorchBridgeBench deterministic evaluation workflow from this plugin.

The user invoked this command with:

```text
$ARGUMENTS
```

## Instructions

1. Parse `$ARGUMENTS`.
2. Resolve relative paths from the current working directory first. If a
   benchmark path is omitted, use the plugin smoke suite:
   `${CLAUDE_PLUGIN_ROOT}/benchmarks/v1.0.0/suites/smoke_noop.json`.
3. If `--suite` is present, run:

   ```bash
   python ${CLAUDE_PLUGIN_ROOT}/scripts/tbbcc.py eval-suite --suite <suite.json> --out <out-dir>
   ```

4. Otherwise run a single case:

   ```bash
   python ${CLAUDE_PLUGIN_ROOT}/scripts/tbbcc.py eval --case <case.json> --adapter <adapter.json> --out <out-dir>
   ```

5. If `--adapter` is omitted for a single case, use:
   `${CLAUDE_PLUGIN_ROOT}/examples/adapters/noop.json`.
6. If `--out` is omitted, use a timestamped directory under `reports/`.
7. Read the generated `summary.json` or `report.json` before answering.
8. Summarize final state, pass/fail counts, failure class, and report paths.

## Constraints

- Keep generated artifacts under the requested output directory.
- Do not edit bridge internals during evaluation.
- Do not claim Agent repair effort, MC sampling, or baseline effort unless
  those steps were actually run.
