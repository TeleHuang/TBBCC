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
classification, and counting policy. Future Agent effort fields must be appended
without breaking existing keys.

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
