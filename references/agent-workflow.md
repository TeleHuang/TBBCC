# Agent Workflow

The agentic loop follows the design document while keeping deterministic work in
the local core.

## Phase 0: Adapt or Translate

- Intercept track: read bridge docs and create or refine adapter / preamble
  behavior.
- Translate track: create or refine translated target-side code.

Effort here counts as `effort_adapt`.

In Claude Code TUI, the intended Track A entrypoint is:

```text
/eval --bridge-id <id> --docs <bridge-docs.md> --suite <suite.json> --out <out-dir>
```

That command starts with adapter authoring, writes `adapter.generated.json`,
records an `adapt` ledger entry, generates a suite pointing to the adapter, and
then invokes the deterministic core.

## Phase 1: Deterministic Verification

Run the core:

```bash
python ${CLAUDE_PLUGIN_ROOT}/scripts/tbbcc.py eval --case CASE --adapter ADAPTER --out OUT
```

Current core behavior:

- Tier-1 compares baseline and target results.
- Tier-2 runs when either side emits `ACTIVATIONS` or `GRADIENTS`.
- Tier-3 runs when either side emits `TASK_METRICS`.
- On the pass path, the ordering is T1 -> T2 -> T3.

## Phase 2: Classification

Use automatic classification first, then ask a diagnostician agent to confirm.
Confirmation is separate from repair and should not count toward repair effort.

## Phase 3: Repair

Repair only migration-side code or adapter specifications. Record:

- attempt count,
- changed files,
- diff size,
- strategy,
- rerun result,
- context summary.

After every repair, rerun from Tier-1.

Current limitation:

- The deterministic core consumes an effort ledger and reports ME, AR,
  migrate@k, and reroll stability. The full repair loop is still orchestrated by
  the Claude Code workflow rather than by `scripts/tbbcc.py`. Adapter authoring
  is exposed through the `/eval` Claude Code slash command.

## Stop Conditions

Stop when:

- all enabled tiers pass,
- max attempts is reached,
- the failure is environment-only,
- repair requires bridge internals,
- the case is outside current implementation scope.

## Benchmark Usage

For routine plugin validation, prefer the lightweight generated smoke suite:

```bash
python ${CLAUDE_PLUGIN_ROOT}/scripts/tbbcc.py eval-suite \
  --suite ${CLAUDE_PLUGIN_ROOT}/benchmarks/v1.0.0/suites/dev_noop.json \
  --out reports/bench_dev
```

Use the cross-level smoke suite when you explicitly want L3/L4 coverage:

```bash
python ${CLAUDE_PLUGIN_ROOT}/scripts/tbbcc.py eval-suite \
  --suite ${CLAUDE_PLUGIN_ROOT}/benchmarks/v1.0.0/suites/smoke_noop.json \
  --out reports/bench_smoke
```

Use `all_noop.json` or the per-level suites when you need broader coverage and
accept substantially higher runtime.
