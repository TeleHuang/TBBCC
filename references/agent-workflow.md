# Agent Workflow

The agentic loop follows the design document while keeping deterministic work in
the local core.

## Phase 0: Adapt or Translate

- Intercept track: read bridge docs and create or refine adapter / preamble
  behavior.
- Translate track: create or refine translated target-side code.

Effort here counts as `effort_adapt`.

In Claude Code TUI, the intended Track A entrypoint is the plugin namespaced
slash command:

```text
/torchbridgebench:eval 评测 torch4ms，优先从本机 ascend-torch4ms-ms272-stable 找文档或最小用例，输出到 reports/torch4ms_eval
```

That command accepts natural language. It should infer the bridge id, search
local bridge repositories, docs, examples, and minimal tests, then start with
adapter authoring. It writes `adapter.generated.json`, records an `adapt` ledger
entry, generates a suite pointing to the adapter, and invokes the deterministic
core. If no docs or minimal examples can be found, it asks the user for one of
those inputs instead of failing on a missing placeholder path.

Regression guard: normal bridge evaluation must not silently collapse to the
4-case `smoke_noop` suite or reuse stale `reports/**/suite.generated.json`
artifacts. Historical reports can inform diagnosis, but every new natural
language eval should create fresh adapter/suite artifacts unless the user
explicitly asks to resume or reuse. When the user does not specify a suite and
does not request a quick smoke/dev run, default to the full benchmark
`benchmarks/v1.0.0/suites/all_noop.json` covering 175 cases (L1=67, L2=42,
L3=25, L4=41). Always state the evaluated case count in the final summary.

Adapter/backend guard: normal bridge evaluation must not silently replace a
reference adapter with a weaker ad-hoc preamble. If `../-demo/examples` contains
a bridge adapter, read it and preserve its backend configuration semantics. For
torch4ms this includes `torch4ms.config.Configuration`,
`default_device_target`, `TORCH4MS_DEVICE_TARGET`, graph-mode flags, activation
script notes, and PYTHONPATH/source-tree assumptions. Before a large run,
execute a small backend sanity check and record the actual backend/device. If
the target bridge is expected to run on Ascend/NPU but logs show
`device_target: CPU`, classify the run as an environment/backend diagnostic
until remediated; do not present it as the intended NPU compatibility result.

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

If verification fails because a bridge package is missing, the workflow must
attempt environment remediation before final classification. Examples include
checking activation scripts, local source checkouts, `PYTHONPATH`, and working
directory assumptions. This remediation is not counted as `effort_adapt` or
`effort_repair`.

If verification produces broad NumericMismatch across very simple deterministic
cases such as ReLU, reshape, or linear layers, first audit whether source and
target used the same inputs, seeds, adapter configuration, and backend. A
near-zero cosine on simple cases is configuration evidence until disproven, not
automatic proof that the bridge internals are unfixable.

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
  is exposed through the `/torchbridgebench:eval` Claude Code plugin slash
  command.

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

Use the cross-level smoke suite only when you explicitly want a quick L1/L2/L3/L4
check rather than a representative benchmark:

```bash
python ${CLAUDE_PLUGIN_ROOT}/scripts/tbbcc.py eval-suite \
  --suite ${CLAUDE_PLUGIN_ROOT}/benchmarks/v1.0.0/suites/smoke_noop.json \
  --out reports/bench_smoke
```

Use `all_noop.json` for normal bridge evaluation and final claims. Use
per-level suites when isolating a level-specific regression.
