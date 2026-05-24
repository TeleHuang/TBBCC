---
name: eval
description: Run the primary TorchBridgeBench Claude Code workflow. Use when the user wants to execute a TorchBridgeBench case or suite, inspect a deterministic report, or drive the design-document workflow around classification and repair.
argument-hint: --case <case.json> --adapter <adapter.json> [--out reports/run] [--suite <suite.json>]
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

# TorchBridgeBench

This is the primary TorchBridgeBench workflow entrypoint.

## Required Context

Read these files before nontrivial work:

- `${CLAUDE_PLUGIN_ROOT}/references/architecture.md`
- `${CLAUDE_PLUGIN_ROOT}/references/data-contracts.md`
- `${CLAUDE_PLUGIN_ROOT}/references/agent-workflow.md`
- `${CLAUDE_PLUGIN_ROOT}/references/local-runtime.md`
- `${CLAUDE_PLUGIN_ROOT}/references/installation-debugging.md` when relevant

## Workflow

1. Parse `$ARGUMENTS`.
2. If `--suite` is present, run:

   ```bash
   python ${CLAUDE_PLUGIN_ROOT}/scripts/tbbcc.py eval-suite --suite <suite.json> --out <out-dir>
   ```

3. Otherwise run:

   ```bash
   python ${CLAUDE_PLUGIN_ROOT}/scripts/tbbcc.py eval --case <case.json> --adapter <adapter.json> --out <out-dir>
   ```

4. Read the generated report or summary before concluding.
5. Confirm failure classification before repair when the result is nontrivial.
6. Repair only migration-side files unless the user explicitly asks for bridge
   internals.

## Constraints

- Keep artifacts under the requested output directory.
- Treat `../-demo` as a sibling implementation, not a dependency.
- Do not claim MC repair sampling or baseline effort metrics unless they were
  actually computed.
