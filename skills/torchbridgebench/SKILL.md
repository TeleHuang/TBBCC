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
5. For bridges with a reference adapter under
   `${CLAUDE_PLUGIN_ROOT}/../-demo/examples/<bridge>_adapter.json`, use it as
   high-priority evidence. Preserve backend configuration semantics such as
   torch4ms `Configuration.default_device_target`, `TORCH4MS_DEVICE_TARGET`, and
   required activation/PYTHONPATH notes. Do not replace a reference adapter with
   a weaker bare import/default context unless you explicitly justify the
   difference.
6. Before running a large suite, run a tiny adapter/backend sanity check:
   - verify the bridge imports in the same environment that will run the suite;
   - verify the actual backend/device selected by the bridge;
   - for torch4ms, inspect the `Initialized MindSpore with configuration`
     message or equivalent runtime config.
   If the intended target is Ascend/NPU but the runtime selects CPU, first try
   environment remediation. If it remains CPU, clearly label the run as a CPU
   backend diagnostic (`CPU backend diagnostic`) and do not present it as the bridge's intended NPU
   compatibility result.
7. Select benchmark scope explicitly:
   - If the user names a suite, use that suite.
   - If the user asks for a quick smoke/dev check, use `smoke_noop.json` or
     `dev_noop.json` and state the small case count clearly.
   - Otherwise default to `${CLAUDE_PLUGIN_ROOT}/benchmarks/v1.0.0/suites/all_noop.json`
     for a full benchmark run. The benchmark asset baseline is 175 cases
     (L1=67, L2=42, L3=25, L4=41); do not silently downgrade to a 4-case smoke
     suite for a normal bridge evaluation.
8. Use a fresh output directory by default. If the user gives `reports/<name>`
   and it already contains generated artifacts, create a timestamped child or
   sibling such as `reports/<name>_<YYYYmmdd_HHMMSS>` unless the user explicitly
   says to overwrite, resume, or reuse.
9. Generate a suite for the fresh adapter by copying the selected benchmark
   scope and replacing only the adapter path.
10. Run:

   ```bash
   python ${CLAUDE_PLUGIN_ROOT}/scripts/tbbcc.py eval-suite --suite <suite.json> --out <out-dir>
   ```

11. For a single explicit case, run:

   ```bash
   python ${CLAUDE_PLUGIN_ROOT}/scripts/tbbcc.py eval --case <case.json> --adapter <adapter.json> --out <out-dir>
   ```

12. Read the generated report or summary before concluding.
13. Always state the benchmark scope, sample size, and actual backend/device in
    the final response. If simple cases such as ReLU or reshape fail with broad
    NumericMismatch and cosine near zero, audit adapter/backend configuration
    before concluding the bridge is internally wrong.
14. Always state the benchmark scope and sample size in the final response, for
    example `0/4 smoke cases` versus `160/175 full benchmark cases`. Never imply
    that a smoke result represents the full benchmark.
15. Confirm failure classification before repair when the result is nontrivial.
16. Repair only migration-side files unless the user explicitly asks for bridge
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
