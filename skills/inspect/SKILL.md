---
name: inspect
description: Inspect the local TorchBridgeBench Claude Code plugin, Claude Code installation, and benchmark environment. Use when checking plugin readiness, local Claude Code startup, or bridge benchmark prerequisites.
argument-hint: [optional focus]
allowed-tools: Bash, Read, Glob, Grep
---

# TorchBridgeBench Plugin Inspection

Inspect the local setup without exposing secrets.

## Commands

Run the relevant checks:

```bash
claude --version
claude plugin validate ${CLAUDE_PLUGIN_ROOT}
python ${CLAUDE_PLUGIN_ROOT}/scripts/tbbcc.py inspect-env
find ${CLAUDE_PLUGIN_ROOT} -maxdepth 3 -type f | sort
```

If the user asks about installed marketplaces or plugins:

```bash
claude plugin list --json
claude plugin marketplace list --json
```

## Report

Summarize:

- Claude Code executable and version,
- plugin validation status,
- relevant framework/package availability,
- plugin file structure,
- any missing prerequisites or risk.

Do not read or print API keys, tokens, or `.secrets` content.
