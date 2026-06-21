---
name: inspect
description: Inspect the local TorchBridgeBench Claude Code plugin, Claude Code installation, and benchmark environment. Use when checking plugin readiness, local Claude Code startup, or bridge benchmark prerequisites.
argument-hint: [optional focus]
allowed-tools: [Bash, Read, Glob, Grep]
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

Use `inspect-env --import-versions` only when exact package versions are needed;
some ML frameworks print noisy initialization warnings on import.

If the user asks about installed marketplaces or plugins:

```bash
claude plugin list --json
claude plugin marketplace list --json
```

## Report

Summarize:

- Claude Code executable and version,
- plugin validation status,
- Python importability of relevant frameworks/packages,
- nearby local bridge source trees separately from importability,
- plugin file structure,
- any missing prerequisites or risk.

Use precise wording:

- `available: false` from `inspect-env` means the module is not importable in
  the current Python environment. It does not prove that a local source checkout
  or minimal example is absent.
- For bridge libraries such as torch4ms or mindtorch, distinguish
  "not importable yet" from "no local source discovered".
- `torch_npu` is the PyTorch Ascend backend. Its absence affects PyTorch-on-NPU
  runs, not NVIDIA GPU ground-truth collection on a separate GPU host.
- Do not claim GPU ground-truth is blocked on this NPU host; report it as a
  separate-machine workflow when appropriate.

Do not read or print API keys, tokens, or `.secrets` content.
