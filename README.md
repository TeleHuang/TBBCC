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

Confirm the plugin is loaded:

```text
/torchbridgebench-inspect
```

Run a single smoke case:

```text
/torchbridgebench --case examples/cases/pure_python_vector.json --adapter examples/adapters/noop.json --out reports/smoke
```

Or run the bundled cross-level smoke suite:

```text
/torchbridgebench --suite benchmarks/v1.0.0/suites/smoke_noop.json --out reports/bench_smoke
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

## Output

Benchmark runs write JSON and Markdown reports into the output directory you
choose. Suite runs also write a summary file under `runs/`.

## Versioning

Repository: https://github.com/TeleHuang/TBBCC

Current release baseline: `0.1.0`.

## Internal Layout

```text
.claude-plugin/plugin.json       Plugin manifest
commands/                        Slash commands
skills/                          Skill-based workflows
agents/                          Specialist agent prompts
references/                      Design and runtime notes
scripts/tbbcc.py                 Deterministic benchmark core
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
