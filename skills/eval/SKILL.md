---
name: eval
description: Run TorchBridgeBench migration evaluation for a bridge or translator. Use when the user asks to evaluate PyTorch-to-MindSpore migration tools, run bridge benchmarks, diagnose compatibility failures, measure migration effort, or produce TorchBridgeBench reports.
argument-hint: --case <case.json> --adapter <adapter.json> [--out reports/run] [--track intercept|translate]
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

# TorchBridgeBench Evaluation

You are running the Claude Code plugin workflow for TorchBridgeBench.

## Arguments

`$ARGUMENTS`

## Required Context

Read these files before doing nontrivial work:

- `${CLAUDE_PLUGIN_ROOT}/references/architecture.md`
- `${CLAUDE_PLUGIN_ROOT}/references/data-contracts.md`
- `${CLAUDE_PLUGIN_ROOT}/references/agent-workflow.md`
- `${CLAUDE_PLUGIN_ROOT}/references/local-runtime.md`
- `${CLAUDE_PLUGIN_ROOT}/references/installation-debugging.md` when plugin
  loading, validation, or installation behavior is relevant.

## Workflow

1. Parse `$ARGUMENTS`. If `--case` or `--adapter` is missing, use the examples:
   - `examples/cases/pure_python_vector.json`
   - `examples/adapters/noop.json`
2. Run the deterministic core from the current project or plugin root. For one
   case/adapter pair:

   ```bash
   python ${CLAUDE_PLUGIN_ROOT}/scripts/tbbcc.py eval --case <case> --adapter <adapter> --out <out-dir>
   ```

   For a suite matrix:

   ```bash
   python ${CLAUDE_PLUGIN_ROOT}/scripts/tbbcc.py eval-suite --suite <suite.json> --out <out-dir>
   ```

3. Read the generated JSON and Markdown reports.
4. If the run fails, classify whether the failure is environment/setup,
   dependency, import order, operator, dtype, shape, device, autograd, numeric,
   training, or runtime crash.
5. If repair is appropriate, use the `torchbridgebench:repairer` agent or do a
   minimal migration-side patch. Do not modify bridge implementation internals
   unless the user explicitly asks.
6. Report:
   - final state,
   - report paths,
   - failure class and evidence if any,
   - next concrete action.

## Constraints

- Do not print secrets from Claude or provider settings.
- Keep run artifacts under the requested output directory.
- Treat `../-demo` as reference only, not as a runtime dependency.
- Prefer deterministic reruns over one-off manual conclusions.
