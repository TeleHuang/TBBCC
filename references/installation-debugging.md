# Installation and Debugging

## Development Mode

Use this while editing the plugin:

```bash
claude --plugin-dir /home/ma-user/work/torchbridgebenchCCplugin
```

Inside Claude Code:

```text
/torchbridgebench:torchbridgebench-inspect
/reload-plugins
```

Run `/reload-plugins` after editing command or skill files in an already-open
TUI window. Without reload, Claude Code may keep using the old command prompt.

Run deterministic smoke checks from a shell:

```bash
python scripts/tbbcc.py eval \
  --case examples/cases/pure_python_vector.json \
  --adapter examples/adapters/noop.json \
  --out reports/smoke
```

## Validation

```bash
cd /home/ma-user/work/torchbridgebenchCCplugin
claude plugin validate .
claude --plugin-dir . plugin details torchbridgebench
python -m py_compile scripts/tbbcc.py
python scripts/tbbcc.py validate-inputs \
  --case examples/cases/pure_python_vector.json \
  --adapter examples/adapters/noop.json
```

## Local Marketplace Installation

Claude Code marketplace layout wants a marketplace root with:

```text
my-marketplace/
  .claude-plugin/marketplace.json
  plugins/torchbridgebench/
    .claude-plugin/plugin.json
    skills/
    agents/
    scripts/
```

The current development directory is the plugin root itself, so the clean
development path is `--plugin-dir`. To test marketplace installation later,
copy this directory under a marketplace root as `plugins/torchbridgebench` and
create `.claude-plugin/marketplace.json` at the marketplace root. Do not make a
marketplace entry with `../` pointing back to this directory; installed plugins
are copied into Claude's cache and path traversal is not portable.

## Cache Behavior

Marketplace-installed plugins are copied to:

```text
~/.claude/plugins/cache
```

Plugin code should reference bundled files via `${CLAUDE_PLUGIN_ROOT}` and write
persistent mutable data to `${CLAUDE_PLUGIN_DATA}`.
