# Data Contracts

All benchmark inputs and outputs should be JSON-serializable and reproducible.

## TestCase

Required:

```json
{
  "id": "bench/local/example",
  "level": "L1",
  "track": "intercept",
  "code": "RESULT = [1.0, 2.0]",
  "ground_truth": {
    "atol": 1e-5,
    "rtol": 1e-5
  }
}
```

Optional fields:

- `expected_ops`: list of expected APIs/operators.
- `training_config`: required for future Tier-3 cases.
- `difficulty`: `basic`, `intermediate`, or `advanced`.
- `failure_mode`: expected/pilot label.
- `seed`: deterministic seed.

Execution contract for `code`:

- Define `RESULT`, or
- define `run()` returning the result.

Optional Tier-2 contract:

- Define `ACTIVATIONS` as a JSON-normalizable structure.
- Define `GRADIENTS` as a JSON-normalizable structure.

If either side exposes a Tier-2 channel, the core compares it after Tier-1 and
includes the result under `tiers.T2`.

Optional Tier-3 contract:

- Define `TASK_METRICS` as a JSON-normalizable structure, such as loss curves or
  accuracy values.

If either side exposes `TASK_METRICS`, the core compares it after Tier-2 and
includes the result under `tiers.T3`.

The core normalizes tensors, numpy arrays, lists, tuples, dictionaries, numbers,
and booleans into JSON-compatible values.

## AdapterSpec

Required:

```json
{
  "bridge_id": "noop",
  "track": "intercept",
  "preamble": "",
  "atol": 1e-5,
  "rtol": 1e-5
}
```

Optional fields:

- `env`: environment variables for target execution.
- `source_preamble`: optional preamble for the baseline execution.
- `timeout_seconds`: per-run timeout.
- `docs`: documentation path or summary.
- `known_gaps`: known unsupported APIs.
- `op_api_map`: translation mapping.
- `fne_threshold`, `gc_threshold`, `tca_threshold`, `dtw_threshold`.

## Report

The core emits:

- `report.json`: machine-readable result.
- `report.md`: human-readable summary.

Reports include environment, case, adapter, tier results, metrics,
classification, counting policy, and optional agent effort fields.

## Effort Ledger and AR

The deterministic core can consume an optional effort ledger:

```json
{
  "entries": [
    {
      "case_id": "bench_v1.0.0/L1/conv/conv2d_fp32",
      "bridge_id": "noop",
      "phase": "adapt",
      "reroll_index": 1,
      "rounds": 1,
      "prompt_chars": 1200,
      "completion_chars": 800,
      "edit_units": 120
    }
  ],
  "migration_samples": [
    {
      "task_id": "bench_v1.0.0/L1/conv/conv2d_fp32",
      "reroll_index": 1,
      "exec_passed": true,
      "full_passed": true
    }
  ]
}
```

Effort uses `shared-effort-v1`:

```text
ME = rounds + (prompt_chars + completion_chars + edit_units) / 1000.0
AR = max(0, 1 - ME / effective_baseline_effort)
```

For a suite, `ME` is the total counted adapt and repair effort across the
evaluated task set. The primary `AR` always uses the calibrated
`baseline_effort` artifact directly. If the supplied AR baseline was calibrated
on a larger task set and the current suite is a strict subset, the summary also
records `scope_adjusted_ar` using a linear case-count effective baseline. That
field is a subset diagnostic and must not be presented as the full benchmark AR.

Environment remediation and classification confirmation are not effort ledger
phases. Only `adapt` and `repair` are counted.

## Suite

Suites define a small benchmark matrix:

```json
{
  "suite_id": "local-smoke",
  "cases": ["../cases/pure_python_vector.json"],
  "adapters": ["../adapters/noop.json"]
}
```

Run with:

```bash
python scripts/tbbcc.py eval-suite --suite examples/suites/local_smoke.json --out reports/local_smoke_suite
```

The core writes `summary.json`, `summary.md`, and per-run reports under
`<out>/runs/`.
