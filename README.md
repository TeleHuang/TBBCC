# TorchBridgeBench Claude Code Plugin

TorchBridgeBench evaluates PyTorch-to-MindSpore/Ascend bridge and translation
tools from the migration user's perspective. This Claude Code plugin packages
the agent workflow, deterministic benchmark core, prompts, and reference
materials needed to run that evaluation inside Claude Code.

This is a new implementation root. It intentionally does not import the early
`../-demo` core.

## Development Load

From this directory or the workspace root:

```bash
claude --plugin-dir /home/ma-user/work/torchbridgebenchCCplugin
```

Then invoke:

```text
/torchbridgebench:eval --case examples/cases/pure_python_vector.json --adapter examples/adapters/noop.json --out reports/smoke
```

After editing plugin files inside a running Claude Code session:

```text
/reload-plugins
```

## Local Core Smoke Test

```bash
cd /home/ma-user/work/torchbridgebenchCCplugin
python scripts/tbbcc.py eval \
  --case examples/cases/pure_python_vector.json \
  --adapter examples/adapters/noop.json \
  --out reports/smoke
```

Expected result: a passing Tier-1 report in `reports/smoke/`.

Run a small local benchmark matrix:

```bash
python scripts/tbbcc.py eval-suite \
  --suite examples/suites/local_smoke.json \
  --out reports/local_smoke_suite
```

This intentionally includes one numeric-drift adapter, so the suite command
returns nonzero while still producing `summary.json` and `summary.md`.

## Plugin Validation

```bash
cd /home/ma-user/work/torchbridgebenchCCplugin
claude plugin validate .
claude --plugin-dir . plugin details torchbridgebench
```

Additional installation and cache notes are in
`references/installation-debugging.md`.

## Benchmark Library

The static benchmark catalog generated from `ClaudeCodePluginDesign.md`
Appendix A lives under:

```text
benchmarks/v1.0.0/cases/
benchmarks/v1.0.0/suites/
benchmarks/v1.0.0/manifest.json
```

Rebuild the catalog after editing the generator:

```bash
python scripts/generate_benchmark_library.py
```

Validate every generated case JSON:

```bash
python scripts/tbbcc.py validate-inputs \
  --case benchmarks/v1.0.0/cases/L1/conv/conv2d_fp32.json
```

Run the bundled cross-level smoke suite:

```bash
python scripts/tbbcc.py eval-suite \
  --suite benchmarks/v1.0.0/suites/smoke_noop.json \
  --out reports/bench_smoke_retry \
  --timeout 180
```

Run the fast development suite when you only need to validate the evaluation
flow itself:

```bash
python scripts/tbbcc.py eval-suite \
  --suite benchmarks/v1.0.0/suites/dev_noop.json \
  --out reports/bench_dev_noop
```

Observed on the current machine:

- `examples/cases/pure_python_vector.json` with `noop`: about `0.21s` wall
  clock.
- `benchmarks/v1.0.0/suites/dev_noop.json`: about `9-10s`.
- `benchmarks/v1.0.0/suites/smoke_noop.json`: about `125s`, because it includes
  real L3/L4 benchmark cases.

Run the full generated matrix when you want exhaustive coverage instead of a
quick smoke:

```bash
python scripts/tbbcc.py eval-suite \
  --suite benchmarks/v1.0.0/suites/all_noop.json \
  --out reports/bench_all_noop \
  --timeout 180
```

## Version Control

Repository: https://github.com/TeleHuang/TBBCC

Use semantic versions for plugin releases. The current development baseline is
`0.1.0`, matching `.claude-plugin/plugin.json`.

## Structure

```text
.claude-plugin/plugin.json       Plugin manifest
skills/torchbridgebench/SKILL.md Primary Claude Code workflow
skills/inspect/SKILL.md          Project and environment inspection workflow
agents/evaluator.md              Runs deterministic evaluation
agents/diagnostician.md          Confirms failure classification
agents/repairer.md               Attempts migration-side repair
references/                      Progressive-disclosure design notes
scripts/tbbcc.py                 Deterministic benchmark core
bin/tbbcc                        Plugin PATH wrapper for the core
examples/                        Self-contained smoke inputs
benchmarks/                      Static benchmark library generated from Appendix A
```

## Runtime Model

The plugin has two layers:

1. Deterministic core: schema loading, execution, Tier-1 metrics,
   auto-classification, report generation.
2. Claude Code agent workflow: adapter planning, failure confirmation, repair,
   audit logging, and user-facing orchestration.

The deterministic core is deliberately standard-library only for the MVP smoke
path. Framework-specific tests can be added through JSON cases and adapter
preambles without changing the plugin component layout.

The current core implements:

- Tier-1 final result comparison through `RESULT` or `run()`.
- Tier-2 generic activation/gradient comparison through optional `ACTIVATIONS`
  and `GRADIENTS`.
- Tier-3 generic task metric comparison through optional `TASK_METRICS`.

Design alignment notes:

- The benchmark catalog is prebuilt as static JSON, matching design decision
  D1.
- Tier execution is still ordered T1 -> T2 -> T3 and only implemented tiers are
  checked on the pass path.
- Classification and repair remain separate agent roles.
- MC repair sampling, explicit agent-context accumulation, and baseline effort
  measurement are still not automated in this plugin root.
