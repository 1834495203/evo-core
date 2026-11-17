"""Unit tests for all pluggable EvoCore modules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evo_core.config import CoreConfig
from evo_core.engine import ResearchKernel
from evo_core.llm import LLMResponse, ToolCallRequest
from evo_core.plugins import (
    BudgetPlugin,
    ContextCompactorPlugin,
    MemoryPlugin,
    SanityGatePlugin,
    SkillsPlugin,
    SubagentsPlugin,
    load_plugins,
)
from evo_core.protocol import HookContext, RunState
from evo_core.sandbox import BackgroundProcessManager, WorkspaceSandbox


class ScriptedLLM:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.api_key = "sk-test"

    def complete(self, messages, *, tools=None, **kwargs) -> LLMResponse:
        if self._responses:
            return self._responses.pop(0)
        return LLMResponse(content='{"verdict": "pass", "feedback": ""}', reasoning_content="")


def _tc(tc_id: str, name: str, args: dict[str, Any]) -> ToolCallRequest:
    return ToolCallRequest(id=tc_id, name=name, arguments_raw=json.dumps(args), arguments=args)


def test_plugins_toggle_cleanly_from_config() -> None:
    cfg = CoreConfig()
    cfg.plugins.literature.enabled = False
    cfg.plugins.subagents.enabled = False
    cfg.plugins.memory.mode = "off"
    enabled = [p.name for p in load_plugins(cfg) if p.enabled(cfg)]
    assert "literature" not in enabled
    assert "subagents" not in enabled
    assert "memory" not in enabled
    assert "budget" in enabled
    assert "rubric_judge" not in enabled
    assert not hasattr(cfg.plugins, "rubric_judge")


def test_sanity_gate_flags_syntax_error_immediately_and_blocks_finish(tmp_path: Path) -> None:
    cfg = CoreConfig(workspace=str(tmp_path / "ws"), run_dir=str(tmp_path / "meta"))
    sb = WorkspaceSandbox(cfg.workspace, run_dir=cfg.run_dir)
    ctx = HookContext(
        config=cfg,
        workspace=sb.workspace,
        run_dir=sb.run_dir,
        sandbox=sb,
        bg_manager=BackgroundProcessManager(sb),
        llm_client=None,
        emit_event=lambda *a, **k: None,
    )
    plugin = SanityGatePlugin()
    state = RunState(run_id="r1", task_prompt="test")
    plugin.before_run(state, ctx)

    bad = sb.resolve_path("src/broken.py")
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("if True:\n    x = 1\nelif False:\n    pass\n", encoding="utf-8")
    # Actually induce syntax error (else followed by elif):
    bad.write_text("if True:\n    x = 1\nelse:\n    x = 2\nelif False:\n    x = 3\n", encoding="utf-8")
    plugin.after_tool_call(state, "write_file", {"file_path": "src/broken.py"}, "Wrote", ctx)
    assert any("SyntaxError" in n for n in state.pending_notices)
    decision = plugin.after_agent_finish(state, "done", ctx)
    assert decision.action == "revise"


def test_removed_reviewer_config_is_ignored_on_legacy_load(tmp_path: Path) -> None:
    from evo_core.config import load_config
    old = tmp_path / "legacy.yaml"
    old.write_text("plugins:\n  rubric_judge:\n    enabled: true\n    max_revisions: null\n", encoding="utf-8")
    cfg = load_config(old)
    assert not hasattr(cfg.plugins, "rubric_judge")
    assert "rubric_judge" not in {p.name for p in load_plugins(cfg)}


def test_memory_cold_start_has_zero_prompt_bloat(tmp_path: Path) -> None:
    cfg = CoreConfig(workspace=str(tmp_path / "ws"), run_dir=str(tmp_path / "meta"))
    cfg.plugins.memory.mode = "lightweight"
    sb = WorkspaceSandbox(cfg.workspace, run_dir=cfg.run_dir)
    ctx = HookContext(
        config=cfg,
        workspace=sb.workspace,
        run_dir=sb.run_dir,
        sandbox=sb,
        bg_manager=BackgroundProcessManager(sb),
        llm_client=None,
        emit_event=lambda *a, **k: None,
    )
    mem = MemoryPlugin()
    state = RunState(run_id="r3", task_prompt="test")
    assert mem.system_prompt_section(state, ctx) is None
    assert mem.tools(ctx) == []


def test_compactor_offloads_large_output_and_compacts_history(tmp_path: Path) -> None:
    cfg = CoreConfig(workspace=str(tmp_path / "ws"), run_dir=str(tmp_path / "meta"))
    cfg.plugins.compactor.max_tool_result_chars = 200
    cfg.plugins.compactor.context_limit_tokens = 150
    cfg.plugins.compactor.trigger_ratio = 0.5
    cfg.plugins.compactor.keep_recent_messages = 2

    llm = ScriptedLLM([LLMResponse(content="- Summary of early work", reasoning_content="")])
    sb = WorkspaceSandbox(cfg.workspace, run_dir=cfg.run_dir)
    ctx = HookContext(
        config=cfg,
        workspace=sb.workspace,
        run_dir=sb.run_dir,
        sandbox=sb,
        bg_manager=BackgroundProcessManager(sb),
        llm_client=llm,
        emit_event=lambda *a, **k: None,
    )
    comp = ContextCompactorPlugin()
    state = RunState(
        run_id="r4",
        task_prompt="task",
        messages=[
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "a1 " * 100},
            {"role": "user", "content": "u2 " * 100},
            {"role": "assistant", "content": "a2"},
            {"role": "tool", "tool_call_id": "t1", "name": "execute", "content": "X" * 1000},
        ],
    )
    comp.after_tool_call(state, "execute", {}, "X" * 1000, ctx)
    assert "offloaded 1000 chars" in state.messages[-1]["content"]

    outgoing = comp.before_model_call(state, [{"role": "system", "content": "sys"}, *state.messages], ctx)
    assert any("<compacted_history>" in str(m.get("content", "")) for m in outgoing)


def test_subagent_and_skills_plugins_work_end_to_end(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills" / "demo-skill"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text("---\nname: demo-skill\ndescription: Demo guide\n---\nFollow steps.", encoding="utf-8")

    cfg = CoreConfig(workspace=str(tmp_path / "ws"), run_dir=str(tmp_path / "meta"))
    cfg.plugins.skills.enabled = True
    cfg.plugins.skills.dirs = [str(tmp_path / "skills")]
    cfg.plugins.budget.max_calls = 10

    llm = ScriptedLLM(
        [
            LLMResponse(
                content="",
                reasoning_content="",
                tool_calls=[
                    _tc("c1", "read_skill", {"name": "demo-skill"}),
                    _tc("c2", "delegate_task", {"role": "writing", "instruction": "Write note"}),
                ],
            ),
            # Child response inside delegate_task:
            LLMResponse(content="Child writing done.", reasoning_content=""),
            # Parent finish:
            LLMResponse(content="All complete.", reasoning_content=""),
        ]
    )
    kernel = ResearchKernel(
        cfg,
        plugins=[BudgetPlugin(), SkillsPlugin(), SubagentsPlugin()],
        llm_client=llm,
    )
    res = kernel.run("Run skill + subagent")
    assert res.status == "completed"
    assert res.model_calls == 2
