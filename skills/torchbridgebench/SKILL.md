---
name: eval
description: Run the primary TorchBridgeBench Claude Code workflow. Use when the user wants to execute a TorchBridgeBench case or suite, inspect a deterministic report, or drive the design-document workflow around classification and repair.
argument-hint: 评测 <bridge-id>，可提供文档、源码目录、最小用例、suite 或输出目录
allowed-tools: [Bash, Read, Write, Edit, Glob, Grep]
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

1. Parse `$ARGUMENTS` as natural language first. Do not require GNU-style
   flags when the user already supplied the bridge id, local path, minimal
   example, suite, or output directory in prose.
2. If the user did not provide bridge startup documentation, search likely
   local sources before asking:
   - directories in the current workspace, parent workspace, and sibling
     directories whose names contain the bridge id;
   - `README*`, `docs/`, `examples/`, `test_*.py`, and activation scripts in
     found bridge checkouts;
   - `${CLAUDE_PLUGIN_ROOT}/../-demo/examples/<bridge>_adapter.json` when it
     exists.
3. If a complete AdapterSpec already exists, validate it. Otherwise create the
   smallest adapter from the discovered docs or minimal example, and record
   adapter-authoring effort in an effort ledger. Environment remediation is not
   counted as effort.
4. If a suite path is present or inferred, run:

   ```bash
   python ${CLAUDE_PLUGIN_ROOT}/scripts/tbbcc.py eval-suite --suite <suite.json> --out <out-dir>
   ```

5. Otherwise run:

   ```bash
   python ${CLAUDE_PLUGIN_ROOT}/scripts/tbbcc.py eval --case <case.json> --adapter <adapter.json> --out <out-dir>
   ```

6. Read the generated report or summary before concluding.
7. Confirm failure classification before repair when the result is nontrivial.
8. Repair only migration-side files unless the user explicitly asks for bridge
   internals.

## Constraints

- Keep artifacts under the requested output directory.
- Treat `../-demo` as a sibling implementation, not a dependency.
- Do not claim MC repair sampling or baseline effort metrics unless they were
  actually computed.
- If bridge docs cannot be found, ask the user for a minimal example or online
  documentation instead of failing immediately.
