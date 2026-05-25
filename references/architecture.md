# Architecture

TorchBridgeBench measures migration burden for PyTorch-to-MindSpore/Ascend
bridges and translators.

This plugin treats `ClaudeCodePluginDesign.md` as the source of truth. The
implementation here follows the closed decisions in that document and calls out
where the current plugin is still partial.

## Tracks

- `intercept`: PyTorch code remains mostly intact while adapter or bridge
  preamble redirects execution to another backend.
- `translate`: PyTorch code is translated to target-framework code before
  execution.

Report track-specific rankings separately. Migration effort is not identical
across tracks.

## Closed Decisions

- D1 Benchmark prebuild: benchmark cases are materialized as static JSON under
  `benchmarks/v1.0.0/`.
- D2 Layered verification with fail-fast: verification is ordered T1 -> T2 ->
  T3, and any failing tier blocks the pass path until classification and repair.
- D3 Agent split: adapt effort and repair effort are conceptually distinct.
- D4 Track split: intercept uses adapter/preamble style migration, translate
  uses translated target code.
- D5 Independent failure confirmation: classification confirmation is separate
  from repair.
- D6 Monte Carlo is for agent repair sampling, not deterministic tensor
  comparison.

## Verification Layers

The target architecture has three verification tiers:

1. Tier-1 Tensor: final outputs match within tolerance.
2. Tier-2 Activation + Gradient: first diverging forward activation or backward
   gradient is identified.
3. Tier-3 Task: training or task metrics remain within accepted drift.

The current plugin core implements Tier-1 and generic Tier-2 / Tier-3 channel
contracts:

- Tier-2: test code may expose `ACTIVATIONS` and/or `GRADIENTS`.
- Tier-3: test code may expose `TASK_METRICS`.

This means the report schema and deterministic comparisons already match the
design intent, while framework hook automation and training-loop instrumentation
remain future work.

## Agent Boundary

Claude Code agents represent the migration user. They may:

- read bridge documentation,
- write adapter preambles,
- diagnose failures,
- repair user-side migration code,
- record effort.

They should not silently patch bridge internals. Bridge fixes belong to a
different project.

The convenient Track A entrypoint in Claude Code TUI is `/eval`. It starts from
bridge documentation, writes `adapter.generated.json`, records `Effort_adapt`,
creates a suite pointing at that generated adapter, and then runs the benchmark
report path.

Current implementation status:

- `agents/adapter-author.md` models adapter generation from bridge docs.
- `agents/diagnostician.md` models the independent confirmation role.
- `agents/repairer.md` models user-side repair only.
- The plugin does not yet automate the full repair state machine with stored
  `AgentContext` inside the core CLI; that orchestration is handled by Claude
  Code slash-command workflow.

## Dual-System Plan

This plugin is one of two intended products:

- Claude Code Plugin: native skills and agents orchestrate evaluation.
- Python Agent package: future pip-installable system with equivalent behavior.

The deterministic core in this plugin is new and should inform the future
Python package. It does not depend on `../-demo`.
