# Core Capability Experiment Plan

This plan verifies only the architecture-map capabilities that matter for the
current report. It intentionally excludes broad benchmark expansion and cosmetic
reporting checks.

## Goal

Show that TorchBridgeBench can evaluate PyTorch-to-MindSpore bridge schemes from
the user workflow perspective, with a report that foregrounds only the core
claims:

- inject adapter startup code for Track A and keep Track B conceptually separate;
- run real benchmark tasks through T1/T2/T3 verification;
- separate bridge compatibility failures from environment/configuration noise;
- record the two-stage agent workflow and effort accounting;
- report total ME, AR, migrate@k, and Monte Carlo stability with readable
  evidence;
- complete practical model/training/numeric/benchmark tasks rather than only
  synthetic probes.

## AR Baseline

Use the calibrated Claude Code baseline from:

- `reports/ar_baseline/deepseek-v4-pro-cc-batch-v1/baseline.json`

Relevant constant:

- `baseline_effort = 2756.8809999999994`

Rules:

- Full benchmark AR is computed as
  `max(0, 1 - ME / 2756.8809999999994)`.
- Subset smoke-suite reports must state their baseline scope. Primary AR still
  uses the full calibrated baseline. A separate `scope_adjusted_ar` may be shown
  as a subset-only diagnostic, but it must not be presented as the full
  benchmark AR.
- The constant is bound to `deepseek-v4-pro`, `claude-code`, `Claude Code`,
  `claude-code-cli-print-json-batch`, and `shared-effort-v1`.
- Do not reuse it for a different model, agent system, protocol, or effort
  formula.

## Scope

Use three experiment groups.

| Group | Purpose | Required Evidence |
| --- | --- | --- |
| Deterministic core | Prove adapter injection, tier verification, compatibility counting, and report generation. | `summary.json`, `report.json`, Markdown report with T1/T2/T3, failure class, compatibility policy. |
| Agent workflow and AR | Prove classification/repair boundaries, ledger-driven effort, suite-level ME/AR, reroll stability, and migrate@k. | Effort ledger, `effort_adapt`, `effort_repair`, `repair_attempts`, `ME`, `AR`, `migrate@k_exec/full`, CI/CV, baseline identity. |
| Real task execution | Prove the system can complete practical model/training/numeric/benchmark tasks, not only toy probes. | Cross-level plugin reports, `-demo` full noop/benchmark reports, real bridge environment-blocking reports with failure-source separation. |

## Experiments

1. Adapter and tier verification.
   Run plugin `dev_noop`, `tier2_smoke`, `tier3_smoke`, and cross-level
   `smoke_noop`. Expected result: compatible cases pass; negative controls fail
   with deterministic classes; T2/T3 are present when the case exposes
   `ACTIVATIONS`, `GRADIENTS`, or `TASK_METRICS`.

2. Effort and agent workflow accounting.
   Use an effort ledger together with the calibrated AR baseline artifact, then
   run `eval-suite --effort-ledger --ar-baseline`. Expected result: reports show
   total `Effort_adapt`, total `Effort_repair`, suite-level `ME`, suite-level
   `AR`, `repair_attempts`, and `migrate@k`, with AR derived from the calibrated
   constant or the explicitly marked subset effective baseline. When real agent
   rerolls are required, use Claude Code non-TUI mode (`claude -p ...
   --output-format json --no-session-persistence`) and store each reroll as a
   ledger/sample record.

3. Failure-source separation.
   Run intentional dependency/import failures and real-bridge smoke attempts.
   Expected result: `DependencyMissing`, `ImportOrderError`, or
   `EnvironmentFailure` are surfaced with evidence and excluded from compatibility
   totals; semantic/numeric failures remain counted.

4. Monte Carlo stability.
   Use reroll sample files from agent repair or AR baseline calibration. Expected
   result: deterministic metrics are measured once, agent-related metrics report
   `mean`, `95% CI`, `cv`, and auto-expand status when variance remains high.

5. Real task completion.
   Run `-demo` noop full suite and benchmark suite. Expected result: smoke,
   core, models, training, numeric, and benchmark suites complete; benchmark
   reports include latency/throughput; the report proves the system works on
   real test tasks.

## Pass Criteria

- Reports show the core fields without manual post-processing: compatibility
  rate, failure class, counted/not-counted policy, T1/T2/T3 result, total ME,
  AR, migrate@k, effort CI/CV, and AR baseline identity.
- Environment/configuration problems are not counted as bridge compatibility
  failures.
- Agent repair effort is counted only from adapt/repair ledger entries;
  classification confirmation and environment remediation are not counted.
- Re-roll data either has acceptable variance or explicitly says more samples
  are required.
- The final report has one focused evidence table per capability rather than
  large raw artifact dumps.
