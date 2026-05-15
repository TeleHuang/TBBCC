# TorchBridgeBench Claude Code Plugin Task

This directory is the development root for the Claude Code plugin variant of
TorchBridgeBench. Keep generated plugin files, core scripts, examples, and
documentation inside this directory unless Claude Code itself installs or caches
the plugin under `~/.claude/plugins`.

## Product Goal

Build a Claude Code plugin that evaluates PyTorch-to-MindSpore/Ascend migration
tools from the user's perspective. The plugin should use Claude Code native
skills and agents for adaptation, diagnosis, confirmation, repair, and audit
while relying on a new local core for deterministic execution, schemas, metrics,
and reports.

This plugin is not a thin wrapper around `../-demo`. The `-demo` project is only
early prototype evidence. The new core in this directory should become the
reference for a future pip-installable Python Agent system.

## Scope Discipline

- Do not put plugin components inside `.claude-plugin/`; only `plugin.json`
  belongs there.
- Keep runtime artifacts under ignored output directories such as `reports/`.
- Do not read or print API keys from Claude settings or `.secrets`.
- Treat old `../torchbridgebench` as reference material, not as an import-time
  dependency.
- Prefer reproducible JSON inputs and JSON/Markdown outputs over ad hoc logs.

## Current MVP Contract

- `scripts/tbbcc.py` implements the deterministic core.
- `/torchbridgebench:eval` is the primary user workflow.
- `examples/cases/pure_python_vector.json` and `examples/adapters/noop.json`
  provide a no-framework smoke test.
- `claude plugin validate .` must pass from this directory.
- `python scripts/tbbcc.py eval ...` must produce JSON and Markdown reports.
