# 实际代码模块与调用关系

本文件对应 `evo-core/` 中已经集成落地的完整代码结构。

## 1. 实际目录

```text
evo_core/
├── engine.py                         # ResearchKernel 主循环、统一工具执行边界与正式完成/验收/科学状态
├── protocol.py                       # ResearchPlugin、ToolCallDecision、TurnDecision、RunState、HookContext
├── config.py / prompts.py / cli.py   # 配置与 CLI (run / probe / inspect / status / ask / steer / dashboard)
├── plugins/
│   ├── __init__.py                   # load_plugins 注册全部可插拔模块
│   ├── harness.py                    # ResearchHarnessPlugin：衔接内核与外置实验账本、并行池、指令与快照
│   ├── literature.py                 # arXiv / OpenAlex / GitHub / S2 / Web / read_paper_fulltext
│   ├── sanity_gate.py                # after_tool_call 语法检查 + before_tool_call 实验前测试门禁
│   ├── subagents.py                  # 主/子共用钩子与预算核算的按需子智能体
│   ├── budget.py / compactor.py / memory.py / skills.py
└── system/
    ├── __main__.py                   # 独立 CLI 演示与 Worker 入口
    ├── bootstrap.py                  # 装配 ResearchSystem（含 worker_pool 与 tracker）
    ├── domain.py / ports.py          # 数据契约（含 ExperimentSpec.check_entrypoint、Job.validity）与接口协议
    ├── runtime/                      # AgentCoordinator、ResearchContextBuilder
    ├── research/                     # controller.py、notebook.py、literature.py (防钻牛角尖触发策略)
    ├── experiments/
    │   ├── harness.py                # 校验规格、冻结源码、幂等提交与查询
    │   ├── worker.py                 # LocalPythonWorker（含 verify_job/run_check_script）与 ParallelWorkerPool
    │   └── docker_worker.py          # DockerExperimentWorker（镜像不可用时可选安全回退本地执行）
    ├── storage/                      # SqliteResearchStore (verify_job / invalidate_job / WAL)、ArtifactStore
    ├── interaction/
    │   ├── commands.py               # 用户指令入队、版本检查、应用回执
    │   ├── queries.py                # 只读快照和进度查询（含 validity 与 checks）
    │   └── qa.py                     # CompanionQAAgent（结构化或只读 LLM 问答）
    ├── observability/
    │   ├── events.py                 # 按游标增量读取事件
    │   ├── tracker.py                # ExperimentTracker（experiments.jsonl / spans.jsonl + 可选 MLflow）
    │   └── server.py                 # ResearchDashboardServer (HTTP API + Web 看板)
    └── delivery/reports.py           # ProgressReportWriter（输出含 validity 的进度文档）

tests/
├── test_core.py
├── test_plugins.py
├── test_system.py
└── test_integrated_system.py         # 统一执行边界、并行与校验、全文文献、QA 与实时看板/干预测试
```

## 2. 一次研究与人机协同的数据流

```text
ResearchKernel.run()
  ├── before_run → ResearchHarnessPlugin 创建/恢复 Study，构建 ParallelWorkerPool
  ├── before_model_call
  │     ├── UserCommandInbox.apply_pending() → suggestion / change_direction / pause / resume
  │     ├── LiteratureReviewPolicy.reasons() → 提醒文献核查（缺来源/改方向/连败/方法阶段/报告前）
  │     └── 写入 live_snapshot.json
  ├── execute_tool_with_hooks() （主 Agent 与 Subagent 共用）
  │     ├── SanityGatePlugin.before_tool_call() → 语法错误或未测修改直接阻断 run_in_background / submit_experiment
  │     ├── submit_experiment → LocalExperimentHarness.submit() 冻结源码 → ParallelWorkerPool 并发执行
  │     ├── verify_experiment → LocalPythonWorker.verify_job() 校验 outputs/checks.json → verified / invalid
  │     └── 同步产物至 workspace/results/<job_id>/ 并写入 tracking/experiments.jsonl & spans.jsonl
  └── after_agent_finish
        ├── 同步实验产物并记录本地检查问题（不请求新模型轮次）
        └── 研究员自主结束，status="completed"、acceptance="not_reviewed"

独立并发读取与干预（不阻塞主循环）：
  ResearchDashboardServer / evocore status / ask / steer
  ├── GET /api/progress & /api/events → ResearchQueries / ResearchEvents / live_snapshot.json
  ├── POST /api/ask → CompanionQAAgent（只读快照与工作区文件）
  └── POST /api/command → UserCommandInbox.send() → 下一检查点生效
```
