# Local Runtime Notes

## Claude Code

On the prepared machine, Claude Code is available at:

```bash
/home/ma-user/work/npm-global/bin/claude
```

The version observed during preparation was `2.1.142`.

Claude settings persist through:

```text
/home/ma-user/.claude -> /home/ma-user/work/.claude-data
```

Do not print API tokens from settings. It is enough to confirm key names and
provider endpoints when debugging.

## Plugin Development

Development load:

```bash
claude --plugin-dir /home/ma-user/work/torchbridgebenchCCplugin
```

Validate:

```bash
claude plugin validate /home/ma-user/work/torchbridgebenchCCplugin
```

Reload inside an interactive Claude Code session:

```text
/reload-plugins
```

Installed marketplace plugins are copied into Claude's plugin cache. Do not rely
on files outside this plugin directory via `../` traversal.

## Ascend Runtime

For torch4ms/MindSpore experiments, the known activation entry is:

```bash
source /home/ma-user/work/activate_torch4ms_ms272_cann85.sh
```

The pure Python smoke case does not require this environment.
