# TorchBridgeBench CC Plugin 修复需求与验收标准

本文档是本轮修复的实施基准。目标不是继续追加实验结果，而是先把
Claude Code 插件的结构、易用入口、测试基础和评测运行正确性修到可信。

## 本轮目标

1. 插件符合 Claude Code 现代 plugin/skill 结构，避免旧 `commands/`
   目录和空 skill 造成命令语义污染。
2. 从本轮开始采用测试驱动开发：新增 `tests/`，先用自动测试固定
   插件结构、frontmatter、路径规范和核心执行隔离行为。
3. 修复核心评测运行中的 source/target 进程隔离问题，避免 PyTorch
   baseline 侧 import 状态污染 bridge target 侧初始化。
4. 保留自然语言 `/torchbridgebench:eval ...` 工作流语义，不要求用户
   用固定 `--docs`、`--suite` 参数才能启动评测。
5. 不提交生成的 `reports/`、图片、临时实验产物，除非用户明确要求。

## P0 正确性要求

### P0-1 删除 legacy slash command 入口

本项目只保留现代 skill 入口。删除 `commands/`，避免出现全局 `/eval`
这类容易与 Claude Code 内置或其他插件冲突的命令。

保留的真实入口是：

- `/torchbridgebench:eval`
- `/torchbridgebench:inspect`
- `/torchbridgebench:ar-baseline`

### P0-2 删除空 skill

删除空目录 `skills/eval/`。每个 `skills/*/` 目录必须包含有效的
`SKILL.md`。

### P0-3 source/target 默认隔离执行

`scripts/tbbcc.py` 的 pair evaluation 必须默认将 source 和 target 放在
独立 Python 子进程中执行。原因：

- source 侧 PyTorch import、`sys.modules`、全局环境、线程池或 device
  初始化可能污染 target 侧 bridge adapter。
- torch4ms 等桥接器对 import 顺序敏感，同进程顺序执行会制造非真实
  兼容性失败。
- 环境变量可以通过 AdapterSpec `env` 传给 target，但 source 侧运行
  不得改变 target 侧解释器状态。

验收测试必须覆盖：source 侧故意污染 Python 进程状态时，target 侧仍
能看到干净状态。

### P0-4 Benchmark 资产不可退化

`benchmarks/v1.0.0/` 是系统的核心资产。当前恢复标准：

- 总用例数必须保持 175。
- 分层数量必须保持 L1=67、L2=42、L3=25、L4=41。
- `all_noop.json` 必须覆盖 175 个 case。
- `smoke_noop.json` 只是 4-case 快速检查，不能作为普通桥接器评测默认值。

自然语言 `/torchbridgebench:eval` 在用户未指定 suite 且未要求 quick
smoke/dev 时，必须默认选择 full benchmark。旧 `reports/**/suite.generated.json`
只能作为历史参考，不能被默认复用为新评测配置。

## P1 插件合规要求

### P1-1 frontmatter 数组格式

`skills/*/SKILL.md` 与 `agents/*.md` 中的 `allowed-tools` / `tools`
必须是 YAML list，不允许逗号字符串。

示例：

```yaml
allowed-tools: [Bash, Read, Write, Edit, Glob, Grep]
tools: [Read, Bash, Grep, Glob]
```

### P1-2 agent 执行边界

所有 agent 必须有 `maxTurns`：

- `evaluator`: 10
- `diagnostician`: 8
- `adapter-author`: 15
- `repairer`: 20

需要 eval 上下文的 agent 增加 `skills: [eval]`：

- `evaluator`
- `adapter-author`
- `repairer`

### P1-3 插件路径可移植

插件内引用自身脚本、references、examples 时必须使用
`${CLAUDE_PLUGIN_ROOT}`。不得在 skill/agent prompt 中硬编码
具体机器的工作目录绝对路径。

对于 torch4ms 等本机源码树，prompt 应要求 agent 根据用户自然语言、
本机兄弟目录、最小用例和在线文档自行搜索，而不是依赖固定绝对路径。

### P1-4 plugin metadata

`.claude-plugin/plugin.json` 至少包含：

- `name`
- `description`
- `version`
- `author`
- `homepage`
- `repository`
- `license`
- `keywords`

license 是否为 `MIT` 取决于项目真实授权；没有授权文件前继续保留
`UNLICENSED`。

## P2 TDD 验收测试

新增 `tests/`，至少包含以下测试：

1. 插件结构测试：不存在 `commands/` 和空 skill，三个必需 skill 存在。
2. frontmatter 测试：`allowed-tools` / `tools` 解析后必须是 list。
3. 路径规范测试：skills/agents 中不出现本机工作区绝对路径，skills
   中不出现无 `${CLAUDE_PLUGIN_ROOT}` 前缀的插件脚本调用。
4. pair isolation 回归测试：source 污染 `sys.modules` 或内置进程状态，
   target 子进程不受污染。
5. suite 路径解析测试：suite 位于输出目录时，case/adapter 相对路径
   仍按 suite 文件所在目录解析。
6. benchmark 防退化测试：175-case 资产、分层数量、suite 覆盖和 eval
   skill 防复用旧 4-case 报告产物策略必须被测试固定。

测试可以用 `pytest`，必要时使用 Python 标准库 fallback，但仓库内必须
提供可直接运行的测试文件。

## P3 易用性要求

`skills/torchbridgebench/SKILL.md` 应表达自然语言入口：

```text
/torchbridgebench:eval 评测 torch4ms，优先从本机 ascend-torch4ms...
```

agent 应按以下顺序获取桥接器启动信息：

1. 用户直接给出的文档、源码目录、最小用例或自然语言说明。
2. 当前工作区及其父/兄弟目录中名称包含 bridge id 的目录。
3. `test_*.py`、`README*`、`docs/`、`examples/`、activation scripts。
4. 如果仍无法判断，向用户请求最小用例或在线文档。

不得把缺少文档简单处理为失败；应先尝试本机搜索和环境修复。环境修复
不计入 ME。

## 非本轮范围

以下内容暂不在本轮修复中实现，除非用户另行批准：

- 重新生成正式实验报告或论文图表。
- 提交 `reports/`、PDF/SVG/PNG 图片或大型实验数据。
- 修改 benchmark case 语义。
- 修改桥接器源码本身。
- 完整重构绘图系统。

## 验收命令

建议按顺序执行：

```bash
python -m pytest tests
python ${CLAUDE_PLUGIN_ROOT}/scripts/tbbcc.py inspect-env
claude plugin validate .
```

如果本机没有 Claude Code 或当前环境无法执行 `claude plugin validate`，
必须在最终说明中明确标注。
