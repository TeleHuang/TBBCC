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
3. Treat existing files under `reports/` as historical artifacts. Read them for
   context if useful, but do not reuse `reports/**/adapter.generated.json` or
   `reports/**/suite.generated.json` as the current run configuration unless
   the user explicitly asks to resume or reuse that exact artifact.
4. If a complete user-supplied AdapterSpec exists outside historical reports,
   validate it. Otherwise create a fresh adapter in the requested output
   directory from the discovered docs or minimal example, and record
   adapter-authoring effort in an effort ledger. Environment remediation is not
   counted as effort.
5. Select benchmark scope explicitly:
   - If the user names a suite, use that suite.
   - If the user asks for a quick smoke/dev check, use `smoke_noop.json` or
     `dev_noop.json` and state the small case count clearly.
   - Otherwise default to `${CLAUDE_PLUGIN_ROOT}/benchmarks/v1.0.0/suites/all_noop.json`
     for a full benchmark run. The benchmark asset baseline is 175 cases
     (L1=67, L2=42, L3=25, L4=41); do not silently downgrade to a 4-case smoke
     suite for a normal bridge evaluation.
6. Use a fresh output directory by default. If the user gives `reports/<name>`
   and it already contains generated artifacts, create a timestamped child or
   sibling such as `reports/<name>_<YYYYmmdd_HHMMSS>` unless the user explicitly
   says to overwrite, resume, or reuse.
7. Generate a suite for the fresh adapter by copying the selected benchmark
   scope and replacing only the adapter path.
8. Run:

   ```bash
   python ${CLAUDE_PLUGIN_ROOT}/scripts/tbbcc.py eval-suite --suite <suite.json> --out <out-dir>
   ```

9. For a single explicit case, run:

   ```bash
   python ${CLAUDE_PLUGIN_ROOT}/scripts/tbbcc.py eval --case <case.json> --adapter <adapter.json> --out <out-dir>
   ```

10. Read the generated report or summary before concluding.
11. Always state the benchmark scope and sample size in the final response, for
    example `0/4 smoke cases` versus `160/175 full benchmark cases`. Never imply
    that a smoke result represents the full benchmark.
12. Confirm failure classification before repair when the result is nontrivial.
13. Repair only migration-side files unless the user explicitly asks for bridge
   internals.

## Constraints

- Keep artifacts under the requested output directory.
- Treat `../-demo` as a sibling implementation, not a dependency.
- Do not claim MC repair sampling or baseline effort metrics unless they were
  actually computed.
- If bridge docs cannot be found, ask the user for a minimal example or online
  documentation instead of failing immediately.
- Do not allow the 175-case benchmark asset to become invisible through
  convenience defaults. Full evaluation is the default; smoke/dev is opt-in.
