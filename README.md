# EvoCore (`evo-core`)

模块的能力声明、输入/输出 Schema、身份、权限、后台任务与数据引用，见 [模块契约 v1](docs/MODULE_CONTRACTS.md)。可运行的新模块示例为 `modules/example_catalog.py`，配置见 `examples/native_modules.yaml`。

模块框架、替换配置、研究员/Coder 边界及第一版本地实现，见 [可插拔模块](docs/PLUGGABLE_MODULES.md)。默认配置启用代码版本审阅后执行；科研顺序保持自主。研究员给出最终答复即结束，不设置独立 LLM 最终评审或自动返工；本地检查仅记录问题，`acceptance="not_reviewed"` 不代表科学结论已获批准。

默认 Coder 已接入真实 **OpenHands Software Agent SDK 1.62.0**，使用独立依赖环境、持久工作目录与 SDK 会话。研究员通过 `implement_coding_task` 委托实现，通过 `prepare_coding_submission` 冻结实际文件后审阅；正式实验仍由 Harness 执行。安装、恢复和验证记录见 [OpenHands 编码模块](docs/OPENHANDS_CODER.md)。

`evo-core` 是从 **EvoScientist v0.4.3** 精炼重构并升级而成的**现象驱动、证据可追溯、支持人机双 Agent 协同的极简自主科研系统**：

- **极简核心内核 (`evo_core/`)**：由配置、统一 OpenAI/DeepSeek 客户端（含 `thinking` 推理链保留与 tool-call 修复）、工作区沙箱（含 Windows Git Bash 执行与后台进程管理）、统一 `before_tool_call / after_tool_call / after_agent_finish` 执行边界、分别记录的运行/未评审交付/科学结论状态（`status / acceptance / scientific_outcome`）与 `ResearchKernel` 状态循环构成。
- **全功能插件化 (`evo_core/plugins/`)**：所有高级能力实现统一 `ResearchPlugin` 协议，可通过 YAML 或 `--disable-plugin` 一键开关：
  - `harness`（`ResearchHarnessPlugin`）：把独立实验队列、冻结源码快照、并行线程池、实验独立校验（`verify_experiment`）、失效隔离（`invalidate_experiment`）、研究笔记账本（`record_research_note`）、文献归档与用户指令应用统一接入主循环，并持续输出 `live_snapshot.json` 与外部追踪记录。
  - `literature`：内置容错的 **arXiv / OpenAlex / GitHub / Semantic Scholar / 网页 / 全文阅读 (`read_paper_fulltext`)** 工具；自动抽取 HTML/PDF 原文、标记 `fulltext_read` vs `abstract_only`、本地缓存至 `literature_db/` 并写入证据账本。
  - `sanity_gate`：写入 `.py` 后立即 AST 语法检查，并通过 `before_tool_call` **硬拦截**未过测试或含语法错误的批量/正式实验启动。
  - `budget`：支持无限制轮次 (`max_calls: null`) 或有界预算控制，主 Agent 与子 Agent 共享调用配额。
  - `compactor`：大输出自动转存文件 + 上下文窗口阈值摘要压缩（不丢失正式实验与研究记录）。
  - `subagents`：按需派发 `planner / research / code / debug / analysis / writing` 子智能体，复用统一 `before_tool_call / after_tool_call` 钩子与预算核算。
  - `memory` / `skills`：按需加载持久经验与 `SKILL.md`。
- **只读伴随问答、实时看板与运行中干预 (`evo_core/system/`)**：
  - `CompanionQAAgent`：只读读取一致快照与研究证据，回答用户进度与结果提问，不阻塞、不污染主 Agent 上下文。
  - `ResearchDashboardServer`：内置 HTTP/HTML 实时看板（`/`, `/api/progress`, `/api/events`, `/api/ask`, `/api/command`），支持查看研究进展、向问答 Agent 提问、发送 `suggestion / change_direction / pause / resume` 指令。

详见 [总体架构](docs/ARCHITECTURE.md)、[实际代码模块](docs/MODULES.md) 与 [实施路线](docs/ROADMAP.md)。

Agent 自主科研、Harness 文献/笔记/总结归档、交付包与原版 EvoScientist MCP 接入，见 [自主科研与 Harness 接入](docs/RESEARCH_WORKFLOW.md)。

正式实验的强制入口、普通命令用途、结果到决策的检查以及旧脚本接入方式，见 [实验执行与证据约束](docs/EXPERIMENT_ROUTING.md)。

## 快速运行

```powershell
# 0. 首次安装 OpenHands 独立依赖环境（在 evo-core 目录执行）
.\scripts\install_openhands.ps1

# 1. API 连通性与工具调用探测
..\EvoScientist\.venv\Scripts\python.exe -m evo_core probe --config config.default.yaml

# 2. 不限轮次运行完整自主科研（可选同时开启实时 Web 看板）
..\EvoScientist\.venv\Scripts\python.exe -m evo_core run --config config.default.yaml --topic-file ..\research-harness\datasets\research_topics.jsonl --topic-id rt-005 --workspace runs/rt-005/workspace --run-dir runs/rt-005/run_meta --dashboard-port 8765

# 3. 运行中查看进度、只读问答、干预方向或单独启动看板
..\EvoScientist\.venv\Scripts\python.exe -m evo_core status runs/rt-005/run_meta
..\EvoScientist\.venv\Scripts\python.exe -m evo_core ask runs/rt-005/run_meta --question "目前在做哪个方向，有哪些已验证结果？"
..\EvoScientist\.venv\Scripts\python.exe -m evo_core steer runs/rt-005/run_meta --kind suggestion --text "补充对高维特征漂移的消融实验"
..\EvoScientist\.venv\Scripts\python.exe -m evo_core dashboard runs/rt-005/run_meta --port 8765
```
