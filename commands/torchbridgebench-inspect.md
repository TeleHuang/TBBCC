---
description: Inspect TorchBridgeBench plugin readiness and benchmark environment
argument-hint: "[optional focus]"
allowed-tools: [Bash, Read, Glob, Grep]
---

# TorchBridgeBench Inspection Command

Inspect the local TorchBridgeBench Claude Code plugin and benchmark runtime.

The user invoked this command with:

```text
$ARGUMENTS
```

## Instructions

Run the relevant checks without printing secrets:

```bash
claude --version
claude plugin validate ${CLAUDE_PLUGIN_ROOT}
claude --plugin-dir ${CLAUDE_PLUGIN_ROOT} plugin details torchbridgebench
python ${CLAUDE_PLUGIN_ROOT}/scripts/tbbcc.py inspect-env
find ${CLAUDE_PLUGIN_ROOT} -maxdepth 3 -type f
```

Then inspect the benchmark manifest:

```bash
python -m json.tool ${CLAUDE_PLUGIN_ROOT}/benchmarks/v1.0.0/manifest.json
```

## Report

Summarize:

- Claude Code version and plugin validation status.
- Whether commands, skills, and agents are visible in plugin details.
- Benchmark library case counts from `manifest.json`.
- PyTorch, MindSpore, and relevant runtime availability from `inspect-env`.
- Any missing prerequisite or likely reason a slash command is unavailable.

Do not read or print API keys, tokens, or `.secrets` content.
