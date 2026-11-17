# 正式实验执行与证据约束

本次修复针对 rt-003 的实际问题：插件启用了，但模型经由普通 shell 完成校准，实验账本一直为空。

## 执行入口

Harness 插件可以整体关闭。启用时，正式测量、校准、试验、基线、消融和扫描统一使用 `submit_experiment`。

普通 `execute` 必须声明 `purpose`：

| 用途 | 规则 |
|---|---|
| `setup` | 环境检查、依赖和数据准备 |
| `test` | pytest/unittest 等自检；通过的输出与当前 Python 源码哈希关联 |
| `debug` | 最多 30 秒的临时诊断，不作为正式实验证据 |
| `analysis` | 显式引用已完成的 `experiment_ids`，用于结果分析和绘图 |
| `experiment` 或缺失 | 拒绝，提示使用实验接口 |

启用 harness 后，工具列表不再提供 `run_in_background`；插件前置检查也会拒绝这条旧路径。关闭 harness 后保持原来工具行为。

用途声明用于正常代理的路由约束，不是操作系统级隔离。正式交付另外检查账本和引用，因此误标用途的 shell 输出也不能仅靠文件存在就充当已登记实验。

## 降低脚本迁移成本

无需强制把现有 JSON 输出改成 `outputs/metrics.json`。例如包模块脚本可这样提交：

```json
{
  "purpose": "校准检测器误报水平",
  "entrypoint": "src/run_calibrate.py",
  "module": true,
  "protocol": "stationary-calibration-v1",
  "data_version": "generator-v1",
  "metrics_path": "results/calibration.json",
  "wait": false
}
```

- `module=true` 使用 `python -m src.run_calibrate`，支持包内相对导入。
- `files` 省略时收集 `src/**/*.py` 和入口/检查器；使用数据或配置文件时仍应明确把它们加入 `files`。
- `metrics_path` 必须指向 `outputs/` 或 `results/` 下的非空 JSON 对象；该输出目录会被归档。
- 源码快照不能夹带旧 `results/`，避免把旧结果当成本次执行输出。
- 并行使用同一批次的多个 `wait=false` 提交，随后调用 `poll_experiments`。

## 自主研究与记录

Harness 更新实验执行状态，不自动推进科研阶段。Agent 自主决定何时实验、检索、记录观察或总结，允许连续及并行提交实验。已有结果无需先补笔记或决策才能继续。

研究焦点是可选的自由文本标签，供用户查看，不参与工具准入或交付判断。

## 验证和交付

- shell 生成的 JSON/CSV 文件不能替代登记实验。
- 报告需引用对应 `exp_*`，未知实验 ID 被拒绝；引用的成功实验需经过独立检查。
- 无效和失败实验保留，报告按现有披露规则说明局限。
- 确实无法实验时，允许明确记录 `inconclusive` 和资源/执行局限；这不等于取得了实验成果。
- 假设不成立仍可以交付；正确性检查不应要求方法必须胜出。
- 评审返回无效 JSON、没有有效 verdict、超过评审轮数或调用失败，均不再默认通过。
- 检查器运行前清除本次工作副本里的旧 `checks.json`；外加检查源码不允许覆盖归档证据。

## 运行中控制和恢复

指令在模型调用边界与工具执行前消费。用户改方向后，旧计划生成的修改或执行动作被阻止，等待下一轮重新规划。

恢复运行会协调已有 Study 的运行状态；此前未登记的文件仍保留，但不会自动包装成已验证成果。rt-003 已按用户要求停止，修复期间未自动重启。

本次不替代后续独立 Worker 服务、租约/心跳、完整资源调度和跨机器环境复现工作。
