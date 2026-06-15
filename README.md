# TorchBridgeBench Claude Code Plugin

TorchBridgeBench evaluates PyTorch-to-MindSpore/Ascend migration tools from the
user's point of view. This plugin gives you a deterministic benchmark core plus
Claude Code commands for running cases, suites, and environment checks.

## Quick Start

Start Claude Code with this plugin loaded:

```bash
claude --plugin-dir /path/torchbridgebenchCCplugin
```

Fill path with your actual path.

Inside Claude Code, reload plugins:

```text
/reload-plugins
```

After editing command or skill files, reload before testing the same TUI window.
Otherwise Claude Code may continue using the previously loaded prompt.

Confirm the plugin is loaded:

```text
/torchbridgebench:torchbridgebench-inspect
```

Run a single smoke case:

```bash
python scripts/tbbcc.py eval \
  --case examples/cases/pure_python_vector.json \
  --adapter examples/adapters/noop.json \
  --out reports/smoke
```

Or run the bundled cross-level smoke suite:

```bash
python scripts/tbbcc.py eval-suite \
  --suite benchmarks/v1.0.0/suites/smoke_noop.json \
  --out reports/bench_smoke
```

## What You Get

- A deterministic benchmark core in `scripts/tbbcc.py`.
- Slash commands for evaluation and inspection.
- A generated benchmark library under `benchmarks/v1.0.0/`.
- Agent roles for evaluation, diagnosis, and repair.

## Benchmark Library

The generated benchmark library contains 175 static cases:

- `L1`: 67
- `L2`: 42
- `L3`: 25
- `L4`: 41

Locations:

```text
benchmarks/v1.0.0/cases/
benchmarks/v1.0.0/suites/
benchmarks/v1.0.0/manifest.json
```

Useful suites:

- `benchmarks/v1.0.0/suites/dev_noop.json`: fast routine validation.
- `benchmarks/v1.0.0/suites/smoke_noop.json`: cross-level smoke with L3/L4.
- `benchmarks/v1.0.0/suites/all_noop.json`: full generated matrix.

Rebuild the library after editing the generator:

```bash
python scripts/generate_benchmark_library.py
```

## Validation

Validate the plugin manifest and inspect the plugin inventory:

```bash
cd /home/ma-user/work/torchbridgebenchCCplugin
claude plugin validate .
claude --plugin-dir . plugin details torchbridgebench
```

Validate benchmark JSON files:

```bash
python scripts/tbbcc.py validate-inputs \
  --case benchmarks/v1.0.0/cases/L1/conv/conv2d_fp32.json \
  --adapter examples/adapters/noop.json
```

Run the fast benchmark suite:

```bash
python scripts/tbbcc.py eval-suite \
  --suite benchmarks/v1.0.0/suites/dev_noop.json \
  --out reports/bench_dev_noop
```

## AR Baseline Calibration

The AR constant is model- and agent-system-specific. For the Claude Code
baseline with DeepSeek pro, use the bundled `ar-baseline` skill or run:

```bash
python scripts/calibrate_ar_cc.py \
  --suite benchmarks/v1.0.0/suites/all_noop.json \
  --out reports/ar_baseline/deepseek-v4-pro-cc-v1 \
  --rerolls 1 \
  --workers 1 \
  --model 'deepseek-v4-pro[1m]'
```

Outputs are written under the selected report directory:

- `samples.jsonl`: raw per-case Claude Code samples and usage metadata.
- `baseline.json`: the baseline constant and audit metadata.
- `baseline.md`: a short human-readable summary.

Rerun the same command to resume an interrupted calibration. The expected full
pass covers all 175 generated cases. Use `--rerolls 2` or higher when you need a
variance estimate rather than a single exhaustive pass.

## Output

Benchmark runs write JSON and Markdown reports into the output directory you
choose. Suite runs also write a summary file under `runs/`.

For bridge developers, start with
`references/bridge-developer-quickstart.md`. It explains how to run a suite and
which report fields are paper-ready: compatibility rate, first-pass rate,
numeric consistency, performance, ME, AR, effort split, migrate@k, and reroll
stability.

In Claude Code TUI, the easiest end-to-end entrypoint is the plugin namespaced
slash command `/torchbridgebench:eval`. Use natural language first; the agent
will search local bridge repositories, docs, examples, and minimal tests before
asking for missing input:

```text
/torchbridgebench:eval 评测 torch4ms，优先从本机 ascend-torch4ms-ms272-stable 找文档或最小用例，输出到 reports/torch4ms_eval
```

This command starts from adapter authoring, writes `adapter.generated.json` and
`effort_ledger.json`, runs the suite, then returns the generated report paths.
If documentation is unavailable, a local minimal example such as `test_*.py` is
acceptable evidence for adapter generation.

Optional compatibility-analysis figures can be generated from one or more
summary reports:

```bash
python scripts/tbbcc.py plot-reports \
  --summary reports/plugin_smoke_noop/summary.json \
  --summary reports/torch4ms_eval/eval/summary.json \
  --out reports/analysis_figures
```

This writes PDF and PNG figures such as `failure_taxonomy.pdf` and
`compatibility_overview.pdf`. Plotting is optional and requires Matplotlib.

## Versioning

Repository: https://github.com/TeleHuang/TBBCC

Current release baseline: `0.1.0`.

## Internal Layout

```text
.claude-plugin/plugin.json       Plugin manifest
commands/                        Slash commands
skills/                          Skill-based workflows
skills/ar-baseline/SKILL.md      AR baseline calibration workflow
agents/                          Specialist agent prompts
references/                      Design and runtime notes
scripts/tbbcc.py                 Deterministic benchmark core
scripts/tbbcc_report_plots.py    Optional report visualization helpers
scripts/calibrate_ar_cc.py       Claude Code AR baseline calibrator
scripts/generate_benchmark_library.py
                                 Benchmark library generator
benchmarks/                      Generated static benchmark catalog
examples/                        Minimal smoke cases and adapters
```

## Notes

- The plugin root is independent from `../-demo`.
- The benchmark catalog is prebuilt as static JSON.
- The plugin currently supports Tier-1, Tier-2 channel comparison, and Tier-3
  task-metric comparison in the deterministic core.
