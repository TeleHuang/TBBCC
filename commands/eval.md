---
description: Evaluate a PyTorch-to-MindSpore bridge from natural-language instructions
argument-hint: "自然语言描述，例如：评测 torch4ms；优先使用本机 ascend-torch4ms 的最小用例；输出到 reports/torch4ms_eval"
allowed-tools: [Bash, Read, Write, Edit, Glob, Grep]
---

# TorchBridgeBench `/torchbridgebench:eval`

This command is exposed by Claude Code as `/torchbridgebench:eval`, not as a
global `/eval` command. It should feel like an LLM-native workflow: the user may
describe the bridge and constraints in natural language, and the agent should
infer paths and defaults before asking for missing information.

The user invoked:

```text
/torchbridgebench:eval $ARGUMENTS
```

## User-Facing Usage

Preferred examples:

```text
/torchbridgebench:eval 评测 torch4ms，优先从本机 ascend-torch4ms-ms272-stable 找文档或最小用例，输出到 reports/torch4ms_eval
/torchbridgebench:eval 用 /home/ma-user/work/ascend-torch4ms-ms272-stable/test_train_cnn.py 作为最小用例评测 torch4ms
/torchbridgebench:eval 评测 mindtorch；如果找不到文档或最小用例就告诉我需要什么
```

Optional expert-style flags such as `--bridge-id`, `--docs`, `--suite`, `--out`,
and `--ar-baseline` may be accepted when the user provides them, but do not
require or recommend flag syntax for normal use.

## Natural-Language Resolution

1. Extract the bridge id from the request when possible. Use names appearing in
   the text, repository names, documentation paths, package import names, or
   example filenames.
2. Resolve explicit paths first. Treat placeholder paths such as
   `/path/to/...`, `path/to/...`, `<docs>`, or `<out>` as invalid examples, not
   as real user input.
3. Search for local evidence before asking the user:
   - current working directory;
   - `${CLAUDE_PLUGIN_ROOT}`;
   - `/home/ma-user/work`;
   - sibling repositories whose names contain the bridge id;
   - `README*`, `docs/**`, `examples/**`, `test_*.py`, `*_test.py`, and package
     `__init__.py` files related to the bridge id.
4. For `torch4ms`, specifically check these likely local sources when present:
   - `/home/ma-user/work/ascend-torch4ms-ms272-stable`;
   - `/home/ma-user/work/ascend-torch4ms`;
   - `/home/ma-user/work/activate_torch4ms_ms272_cann85.sh`;
   - `/home/ma-user/work/ascend-torch4ms-ms272-stable/test_train_cnn.py`;
   - `/home/ma-user/work/ascend-torch4ms-ms272-stable/test_import.py`;
   - `/home/ma-user/work/ascend-torch4ms-ms272-stable/test_simple_op.py`;
   - `/home/ma-user/work/ascend-torch4ms-ms272-stable/test_train.py`;
   - `/home/ma-user/work/-demo/examples/torch4ms_adapter.json`.
5. If documentation is missing but minimal examples exist, infer adapter startup
   from the examples and record those examples as the evidence source.
6. If neither documentation nor minimal examples can be found, stop and ask the
   user for one of:
   - a minimal working PyTorch-to-bridge example;
   - a local README/install guide/adapter note;
   - an online documentation link.

When asking for help, include the paths already checked and any partial
candidates found. Do not ask only for a documentation path if a minimal example
would be enough.

## Defaults

- Suite: `${CLAUDE_PLUGIN_ROOT}/benchmarks/v1.0.0/suites/smoke_noop.json`
- Output: `${CLAUDE_PLUGIN_ROOT}/reports/eval_<bridge-id>_<timestamp>`
- AR baseline:
  `${CLAUDE_PLUGIN_ROOT}/reports/ar_baseline/deepseek-v4-pro-cc-batch-v1/baseline.json`
  when that file exists.

## Workflow

1. Parse the natural-language request and any optional expert flags.
2. Resolve bridge id, documentation, examples, suite, output directory, and AR
   baseline using the rules above.
3. Read:
   - `${CLAUDE_PLUGIN_ROOT}/references/data-contracts.md`;
   - `${CLAUDE_PLUGIN_ROOT}/references/agent-workflow.md`;
   - resolved bridge docs and/or minimal examples.
4. Create the output directory.
5. Write `<out>/adapter.generated.json`.
6. Write `<out>/effort_ledger.json` with at least one `phase: "adapt"` entry.
7. Write `<out>/suite.generated.json`, copying the selected suite cases and
   replacing `adapters` with `<out>/adapter.generated.json`.
8. Validate the generated adapter:

   ```bash
   python ${CLAUDE_PLUGIN_ROOT}/scripts/tbbcc.py validate-inputs \
     --adapter <out>/adapter.generated.json
   ```

9. Run evaluation:

   ```bash
   python ${CLAUDE_PLUGIN_ROOT}/scripts/tbbcc.py eval-suite \
     --suite <out>/suite.generated.json \
     --out <out>/eval \
     --effort-ledger <out>/effort_ledger.json \
     --ar-baseline <baseline-if-available>
   ```

10. Read `<out>/eval/summary.json` and `<out>/eval/summary.md` before
    answering.

## AdapterSpec Requirements

Generate this shape:

```json
{
  "bridge_id": "<bridge-id>",
  "track": "intercept",
  "preamble": "<bridge import/startup code>",
  "source_preamble": "",
  "env": {},
  "atol": 1e-5,
  "rtol": 1e-5,
  "timeout_seconds": 120,
  "docs": "<docs/examples path or short evidence summary>",
  "known_gaps": []
}
```

The `preamble` must:

- import and initialize the bridge according to the docs or minimal examples;
- include local source checkout import paths when the bridge is available as a
  workspace repository rather than an installed wheel;
- avoid editing bridge internals;
- fail with clear import/configuration errors if the bridge package is absent;
- keep environment variables in `env` where possible.

## Effort Ledger Requirements

Write `<out>/effort_ledger.json`:

```json
{
  "schema_version": "tbbcc.effort_ledger.v0.2",
  "entries": [
    {
      "bridge_id": "<bridge-id>",
      "phase": "adapt",
      "rounds": 1,
      "prompt_chars": 0,
      "completion_chars": 0,
      "edit_units": 0,
      "measurement": "local_char_proxy",
      "classification_confirmation_counted": false,
      "environment_remediation_counted": false
    }
  ],
  "migration_samples": []
}
```

If exact Claude Code telemetry is unavailable, estimate `prompt_chars` from the
docs/examples plus task prompt, `completion_chars` from generated text, and
`edit_units` from generated file character count. State that `local_char_proxy`
was used.

## Failure Handling

- If the generated adapter fails because the bridge package is missing, keep the
  report only after environment remediation has been attempted. It should then
  be classified as dependency/environment failure and excluded from
  compatibility.
- If a dependency or local environment problem looks repairable, attempt
  environment remediation first and do not count that work as effort.
- Environment remediation includes checking activation scripts, `PYTHONPATH`,
  editable local source checkouts, and obvious missing working-directory issues.
  For a local source tree, prefer setting `PYTHONPATH` in AdapterSpec `env` or
  adding a narrowly scoped `sys.path.insert(0, "<repo>")` in the adapter
  preamble.
- For `torch4ms`, if `/home/ma-user/work/ascend-torch4ms-ms272-stable` exists
  and `import torch4ms` fails, retry with that directory on `PYTHONPATH` before
  concluding `DependencyMissing`.
- If remediation fails, report the concrete failure and what the user must
  provide or install.
- If the failure is bridge-relevant, confirm classification before repair.
- Repair only adapter or migration-side files.
- Append repair effort as `phase: "repair"` and rerun the modified case or
  suite.
- Do not count environment remediation or independent classification
  confirmation as effort.

## Final Answer

Return only:

- generated adapter path;
- summary report path;
- compatibility rate;
- first-pass rate;
- ME / AR if available;
- dominant failure class if any;
- whether environment failures were excluded;
- any missing user-provided input only if the workflow could not proceed.
