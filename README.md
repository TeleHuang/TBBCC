# TorchBridgeBench Claude Code Plugin

TorchBridgeBench evaluates PyTorch-to-MindSpore/Ascend migration tools from the
user's point of view. This plugin gives you a deterministic benchmark core plus
Claude Code commands for running cases, suites, and environment checks.

## Quick Start

Validate the plugin first:

```bash
cd /home/ma-user/work/torchbridgebenchCCplugin
claude plugin validate .
```

For local development or manual inspection, start Claude Code with this plugin
loaded for the current session:

```bash
claude --plugin-dir /home/ma-user/work/torchbridgebenchCCplugin
```

If you are already in an old Claude Code TUI, exit and restart with
`--plugin-dir`. `/reload-plugins` only reloads plugins already known to that
session; it does not replace `--plugin-dir` startup for an uninstalled local
plugin.

Inside Claude Code, confirm the plugin is loaded:

```text
/torchbridgebench:inspect
```

Run an end-to-end bridge evaluation with natural language:

```text
/torchbridgebench:eval 评测 torch4ms，优先从本机 ascend-torch4ms-ms272-stable 找文档或最小用例，参考 test_train_cnn.py 里的 default_env、loss_wrapper 和 Torch4msOptimizer 用法，输出到 reports/torch4ms_eval
```

For a normal bridge evaluation, the plugin defaults to reusing a valid cached
`adapter.generated.json` + `suite.generated.json` from `reports/` when bridge id
and suite scope match. Say `fresh`, `regenerate`, `no-cache`, or `重新生成` when
you want new artifacts. If no suite is specified, the default scope remains the
full 175-case benchmark unless you explicitly ask for a quick smoke/dev run.
Suite execution also resumes per-case results by default: existing valid
`runs/<case>__<bridge>/report.json` files are skipped and counted in the final
summary. Use `--no-resume` only when you intentionally want a full fresh rerun.
For performance, `eval-suite` uses persistent source/target workers by default:
the PyTorch baseline side and bridge target side each initialize once per suite,
not once per case. Use `--isolated-per-case` only for debugging process
isolation or import-order issues.

Claude Code 2.1.183 does not provide `claude plugin add`. Use `--plugin-dir`
for local manual checks, or the marketplace/install workflow for packaged
distribution.

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
- `benchmarks/v1.0.0/suites/smoke_noop.json`: 4-case cross-level smoke only.
- `benchmarks/v1.0.0/suites/all_noop.json`: full 175-case generated matrix for
  normal bridge evaluation and final claims.

Rebuild the library after editing the generator:

```bash
python scripts/generate_benchmark_library.py
```

## GPU Reference Collection

Formal numerical accuracy uses a GPU PyTorch reference versus an Ascend NPU
bridge result. The GPU reference must be collected from the same canonical
benchmark suite used by the NPU run. The canonical case id is the `id` field in
`benchmarks/v1.0.0/cases/**/*.json`; do not create a separate GPU-only case id
scheme.

On the GPU server, run:

```bash
cd /path/to/torchbridgebenchCCplugin
python scripts/tbbcc_gpu_reference.py \
  --suite benchmarks/v1.0.0/suites/all_noop.json \
  --out reports/gpu_reference_all_noop \
  --device cuda
```

For a quick local smoke test without CUDA:

```bash
python scripts/tbbcc_gpu_reference.py \
  --suite benchmarks/v1.0.0/suites/dev_noop.json \
  --out reports/gpu_reference_dev_cpu \
  --device cpu
```

The collector writes:

```text
reports/gpu_reference_all_noop/
├── manifest.json
├── summary.json
└── cases/<canonical-case-slug>/reference.json
```

Each `reference.json` stores the canonical `case_id`, `case_sha256`,
environment metadata, result summaries, activation summaries, gradient
summaries, task metrics, and `.npy` tensor artifacts. See
`references/unified-case-system.md` for the full contract.

To verify whether an existing GPU artifact directory can be aligned with a
suite:

```bash
python scripts/tbbcc.py gpu-reference-status \
  --artifact-root reports/gpu_reference_all_noop \
  --suite benchmarks/v1.0.0/suites/all_noop.json
```

If `mapping_required=true`, the artifact ids do not directly match the suite's
canonical case ids and must not be used for formal GPU-vs-NPU plots until a
reviewed mapping file exists.

On the NPU host, collect bridge target artifacts with the same suite and the
validated bridge adapter:

```bash
python scripts/tbbcc_bridge_artifacts.py \
  --suite benchmarks/v1.0.0/suites/all_noop.json \
  --adapter reports/torch4ms_eval/adapter.generated.json \
  --out reports/torch4ms_bridge_artifacts
```

Then compare the GPU reference artifact root with the NPU bridge artifact root:

```bash
python scripts/tbbcc_compare_artifacts.py \
  --gpu-reference reports/gpu_reference_all_noop \
  --npu-bridge reports/torch4ms_bridge_artifacts \
  --suite benchmarks/v1.0.0/suites/all_noop.json \
  --out reports/gpu_vs_torch4ms
```

This comparison is the formal GPU-vs-NPU numeric path. The older `eval-suite`
local-pair report remains useful for adapter and harness diagnostics, but it
deletes temporary tensor artifacts and should not be used as the final
GPU-vs-NPU numerical evidence.

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
choose. Suite runs also write `summary.json` and `summary.md` in the selected
output directory. Reports are compact by default: large tensors are stored as
shape/dtype/sample/hash/statistical summaries plus numeric comparison metrics,
not as full tensor payloads.

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
  --summary /path/to/gpu-ground-truth/L1/summary.json \
  --summary reports/plugin_smoke_noop/summary.json \
  --summary reports/torch4ms_eval/eval/summary.json \
  --out reports/analysis_figures
```

This command reads one or more evaluation `summary.json` files. It can also read
a GPU ground-truth `summary.json` containing reference PyTorch outputs,
intermediate tensors and gradients for later GPU-vs-NPU numerical comparison.
It writes an optional compatibility-analysis bundle:

- SVG/PDF/PNG figures with editable vector text
- `source_data/*.csv` for every figure
- `plot_manifest.json` with source summaries and provenance notes
- default plots: compatibility overview, failure taxonomy, tolerance sweep,
  model/component heatmap, 2D design-space bubble chart, metric scorecard, and
  GPU ground-truth coverage when a ground-truth summary is supplied

The ground-truth coverage figure reports reference case count, unique operators,
pass rate, artifact completeness, operator-family coverage and known-risk
labels. It is a reference-data readiness check, not a bridge compatibility
score.

Pass flags such as `--metric-scorecard`, `--tolerance-sweep` or
`--ground-truth-coverage` to generate only selected plots. Plotting is optional
and requires Matplotlib.

## Versioning

Repository: https://github.com/TeleHuang/TBBCC

Current release baseline: `0.1.0`.

## Internal Layout

```text
.claude-plugin/plugin.json       Plugin manifest
skills/                          Skill-based slash workflows
skills/torchbridgebench/SKILL.md Primary /torchbridgebench:eval workflow
skills/inspect/SKILL.md          /torchbridgebench:inspect workflow
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
