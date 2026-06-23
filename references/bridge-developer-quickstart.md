# Bridge Developer Quickstart

This guide is for bridge developers who want a fast, paper-ready evaluation of a
PyTorch-to-MindSpore bridge.

## What You Get

Every suite run produces:

- `summary.md`: high-readability report for humans.
- `summary.json`: machine-readable metrics for tables and plots.
- `runs/*/report.md`: per-case details.
- `runs/*/report.json`: per-case raw evidence.

The top of `summary.md` is the main artifact. It lists the metrics most bridge
papers usually need: compatibility, first-pass rate, numeric consistency,
performance, ME, AR, effort split, migrate@k, and reroll stability.

## One-Command Automatic Run

In Claude Code TUI, use `/torchbridgebench:eval` when you want the system to
start from adapter authoring. Prefer natural language over CLI-style flags:

```text
/torchbridgebench:eval 评测 torch4ms，优先从本机 ascend-torch4ms-ms272-stable 找文档或最小用例，输出到 reports/torch4ms_eval
```

The agent infers the bridge id, selects the benchmark scope, and first checks
for a reusable generated adapter/suite cache. If a valid cache exists, it reuses
it by default. If no valid cache exists, it searches local bridge repositories,
README/docs, examples, and `test_*.py` minimal cases, then writes:

- `reports/<run>/adapter.generated.json`
- `reports/<run>/effort_ledger.json`
- `reports/<run>/suite.generated.json`
- `reports/<run>/summary.md`
- `reports/<run>/summary.json`

Use this path for normal bridge evaluation. It records adapter creation as
`Effort_adapt` and then runs the generated adapter through the benchmark suite.
If documentation is missing but a minimal example exists, the agent should use
the example as adapter evidence. If neither exists, it should ask for a minimal
example, local docs, or online docs.

By default, a normal bridge evaluation should use the full 175-case benchmark
suite `benchmarks/v1.0.0/suites/all_noop.json`. The 4-case `smoke_noop.json`
suite is only for explicit quick smoke checks and must be reported as such.
Existing `reports/**/adapter.generated.json` and `reports/**/suite.generated.json`
files are reusable caches only when `cache-status` confirms that bridge id and
suite case ids match the requested evaluation. Say `fresh`, `regenerate`,
`no-cache`, or `重新生成` when you want to force new adapter/suite generation.

For bridges that ship a reference adapter example under `../-demo/examples`,
the automatic workflow should use it as high-priority evidence. For torch4ms,
verify whether the generated adapter preserves `Configuration.default_device_target`
and `TORCH4MS_DEVICE_TARGET`; a bare `default_env().__enter__()` may select CPU
and produce misleading numeric failures. The report must state the actual
backend/device. If an intended Ascend/NPU run falls back to CPU, treat the result
as a backend configuration diagnostic rather than a final compatibility claim.

## Manual Run

From the plugin root:

```bash
python scripts/tbbcc.py eval-suite \
  --suite benchmarks/v1.0.0/suites/smoke_noop.json \
  --out reports/my_bridge_smoke \
  --effort-ledger reports/core_capability_final/focused_effort_ledger.json \
  --ar-baseline reports/ar_baseline/deepseek-v4-pro-cc-batch-v1/baseline.json
```

For a real bridge, replace the adapter path inside the suite JSON with your
bridge adapter.

## Metrics To Quote

Use these fields from `summary.json` or the `Paper-Ready Values` section in
`summary.md`:

| Metric | JSON Path | Meaning |
| --- | --- | --- |
| Compatibility rate | `totals.compatibility_rate` | Pass rate after excluding environment/config failures. |
| First-pass rate | `quality.first_pass_rate` | Fraction passing without counted repair effort. |
| ME | `effort.me` | Total counted migration effort. |
| AR | `effort.ar` | Work avoided versus no-bridge baseline. |
| Effort split | `effort.effort_adapt`, `effort.effort_repair` | Adapter setup vs repair burden. |
| Numeric consistency | `quality.numeric.*` | MAE, p95, max error, cosine, failed element count. |
| Performance | `quality.performance.*` | Source time, target time, wall time, target/source ratio. |
| Tier-2 rate | `quality.tiers.t2_pass_rate_when_implemented` | Activation/gradient correctness where available. |
| Tier-3 rate | `quality.tiers.t3_pass_rate_when_implemented` | Task-metric correctness where available. |
| migrate@k | `effort.migrate_at_k` | Probability at least one of k rerolls passes. |
| Reroll stability | `effort.variance_control.cv` | Whether agent-related effort is stable enough. |

## Interpreting AR

Primary AR is:

```text
AR = max(0, 1 - ME / baseline_effort)
```

The current Claude Code DeepSeek baseline is:

```text
baseline_effort = 2756.8809999999994
```

Only reuse that baseline when the model, agent system, protocol, task set, and
effort formula match the AR handoff. For subset smoke tests, the report also
shows `scope_adjusted_ar`; treat it as a subset diagnostic, not the full
benchmark AR.

## Compatibility Counting

Environment-only failures are reported but excluded from compatibility:

- `EnvironmentFailure`
- `DependencyMissing`
- `ImportOrderError`

Bridge-relevant failures are counted:

- `OperatorNotFound`
- `TypeMismatch`
- `DeviceMismatch`
- `ShapeMismatch`
- `RuntimeCrash`
- `NumericMismatch`
- `AutogradFailure`
- `TrainingDivergence`

This keeps missing packages or local configuration problems from polluting the
bridge compatibility score.

## Agent Effort Rules

Counted:

- `adapt`: reading bridge docs and building adapter/preamble behavior.
- `repair`: fixing adapter or migration-side code after a confirmed failure.

Excluded:

- automatic classification;
- independent agent confirmation session;
- environment remediation;
- package installation/debugging.

## Recommended Workflow

1. Run `dev_noop` or your smallest suite only to validate adapter loading.
2. Run `smoke_noop` only for a quick cross-level smoke check.
3. Inspect `summary.md` first. Only open per-case reports for failures.
4. Use `summary.json` to generate paper tables.
5. For final claims, run the full benchmark suite and use the matching AR
   baseline identity.
