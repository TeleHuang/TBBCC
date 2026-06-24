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

### P0-0 数值准确性默认语义：GPU PyTorch reference vs NPU bridge result

论文与正式系统默认讨论的“PyTorch 数值基线”不是本机 CPU PyTorch，也不是
`torch_npu`。正式数值准确性结论必须以独立 GPU 服务器采集的 PyTorch
reference artifact 为 source，以 Ascend NPU 上 bridge 执行产生的 artifact
为 target。

因此系统必须区分两种运行口径：

- `local-pair`：同一宿主机上 source/target 双跑，用于 adapter sanity、
  process isolation、bridge 启动和快速诊断。该模式产生的 `NumericMismatch`
  只能解释为 local source result 与 local target result 不一致，不能直接写成
  GPU-vs-NPU 数值准确性结论。
- `gpu-reference`：加载 GPU PyTorch ground-truth artifact，与 NPU bridge
  artifact 比较。该模式才是 FNE/GC/TCA 的默认论文口径。

GPU artifact 必须记录 case id、seed、PyTorch/CUDA/cuDNN/GPU/driver、输出张量、
中间 activation、gradient 和可选 task metrics 的路径或摘要。若 GPU artifact
case id 与 CC benchmark case id 不一致，系统必须显式报告 mapping 缺失或使用
受版本控制的 mapping 文件；不得把无 overlap 的数据强行合并进图表或兼容率。

失败分类必须能区分：

- `NumericMismatch`：比较双方输入/case 语义已确认一致后的真实数值不一致。
- `InputMismatch` / `RNGMismatch`：source/target 输入数据不同，数值比较无效。
- `AdapterIncomplete`：adapter 未使用桥接器要求的 API，如训练 loss/optimizer
  包装缺失。
- `HarnessFailure` / `ProtocolContamination`：测试框架、worker 协议或日志解析
  导致的失败。

`InputMismatch`、`RNGMismatch`、`AdapterIncomplete`、`HarnessFailure` 和
`ProtocolContamination` 默认不计入 bridge compatibility failure。报告必须把它们
列为“需修复测评系统/adapter/harness 后重测”，不能标记为 bridge-internal
MARK_UNFIXABLE。

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
smoke/dev 时，必须默认选择 full benchmark。系统默认复用已验证缓存；缓存
必须匹配 bridge id 与 suite case ids/case count。只有当没有有效缓存，或
用户明确要求 fresh/no-cache/regenerate/重新生成时，才重新生成 adapter/suite。

原版无 agent `torchbridgebench` 的 torch4ms 回归资产也是防退化基准：

- 参考报告：
  `/home/ma-user/work/torchbridgebench/artifacts/reports/report_torch4ms_ms272_cann85_npu_20260510_clean.json`
- 该报告包含 41 个通过用例，覆盖 operator/module/autograd/model/end2end/
  repo_training_regression。
- CC 插件必须提供等价的 anti-regression 入口或导入检查，至少能证明这 41 个
  原版用例的名称、suite、通过状态和 adapter 关键语义没有在 CC 插件迁移中丢失。
- 如果 CC 插件 full benchmark 的结果低于原版 41-case 基准，报告必须先诊断
  adapter/harness/input/gpu-reference mapping 问题，不能直接宣称 torch4ms 能力退化。

### P0-5 评测运行性能不可退化

175-case full benchmark 是正式能力评测，但它不能因为框架自身设计问题变成
不可用的小时级黑盒任务。本轮恢复标准：

- `eval-suite` 默认不得为每个 case 重启完整 Python/MindSpore/Ascend runtime。
  suite 级运行必须使用 source worker 与 target worker 两个长驻解释器：
  source 与 target 仍保持进程隔离，但每一侧在同一个 suite 内只初始化一次。
  这用于避免 Ascend 驱动、MindSpore runtime、bridge adapter 反复加载造成
  分钟级额外开销。只有用户显式要求强隔离调试，或 worker 启动/协议失败时，
  才回退到 per-case isolated subprocess。
- 默认报告必须是 compact report：保存 shape、dtype、样本、hash、误差统计、
  stdout/stderr 摘要和必要 traceback，不保存完整大张量、不保存完整
  `TBBCC_PAYLOAD_JSON` stdout。
- 单个普通 case 的 `report.json` 不应因为张量 payload 膨胀到数百 MB 或 GB。
  大张量 case 必须通过摘要比较完成，报告体积保持可读、可传输、可后处理。
- `eval-suite` 必须支持结果级 resume/skip-completed。默认复用同一输出目录中
  已存在且结构有效的 `runs/<case>__<bridge>/report.json`，中断后再次运行不应
  从第一个 case 重新开始。
- 必须区分 config cache 与 result cache：
  config cache 复用 `adapter.generated.json` 和 `suite.generated.json`；
  result cache/resume 复用已完成 per-case report。
- 全量正式评测和快速诊断要在入口语义上明确区分。full benchmark 仍是未指定
  suite 时的正式默认；quick/smoke/dev 必须由用户显式要求，或由系统明确声明
  它只是诊断，不可冒充正式评测。
- 长任务必须具备可观测进度：至少在 suite summary 和 CLI 输出中体现 total、
  completed、skipped、executed、failed、当前输出目录和最终报告位置。Claude
  skill 在最终回复中必须说明本次是 fresh run、config-cache reuse、还是
  result resume。

验收测试必须覆盖：

1. 大张量用例报告不会保存完整 tensor/list payload，报告大小受控。
2. 同一 suite/out 第二次运行会跳过已完成 report，而不是重复执行。
3. `--no-resume` 或等价显式 fresh 选项可以强制重跑。
4. summary 中能区分 `executed` 与 `skipped`。
5. suite 默认 persistent worker 模式下，两个 case 不会触发四次
   source/target 解释器初始化；source preamble 与 target preamble 各只执行一次。
6. persistent worker 必须忽略 MindSpore/torch4ms 打到 stdout 的非协议日志，直到
   读到 `TBBCC_WORKER_INIT_JSON=` 或 `TBBCC_WORKER_RESULT_JSON=`；非协议日志不得导致
   `Invalid worker response`。
7. GPU ground-truth artifact 与 CC benchmark case id 无 direct overlap 时，系统必须
   报告 mapping 缺失，而不是生成伪 GPU-vs-NPU 数值结论。
8. 原版 torchbridgebench 41-case torch4ms 报告必须作为 anti-regression fixture
   被测试固定：总数 41，全部 compatibility/correctness 通过，并且包含 autograd、
   model、module、operator、end2end、repo_training_regression 层级。

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
