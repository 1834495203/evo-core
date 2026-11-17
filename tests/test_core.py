"""Unit tests for the minimal EvoCore kernel, sandbox, and LLM message repair."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from evo_core.config import CoreConfig, load_config
from evo_core.engine import ResearchKernel
from evo_core.llm import LLMResponse, ToolCallRequest, repair_messages
from evo_core.protocol import BasePlugin, HookContext, RunState, TurnDecision
from evo_core.sandbox import BackgroundProcessManager, WorkspaceSandbox, validate_command


class FakeLLM:
    """Deterministic mock LLM client returning scripted LLMResponse objects."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.api_key = "sk-fake-secret-123"

    def complete(self, messages, *, tools=None, **kwargs) -> LLMResponse:
        self.calls.append({"messages": messages, "tools": tools, **kwargs})
        if not self._responses:
            return LLMResponse(content="Done.", reasoning_content="", finish_reason="stop")
        return self._responses.pop(0)


def _tc(tc_id: str, name: str, args: dict[str, Any]) -> ToolCallRequest:
    raw = json.dumps(args)
    return ToolCallRequest(id=tc_id, name=name, arguments_raw=raw, arguments=args)


def test_repair_messages_closes_dangling_and_drops_orphans() -> None:
    msgs = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "", "type": "function", "function": {"name": "ls", "arguments": {}}}],
        },
        {"role": "user", "content": "next"},
        {"role": "tool", "tool_call_id": "orphan_1", "name": "ls", "content": "stale"},
    ]
    fixed = repair_messages(msgs, preserve_reasoning=True)
    roles = [m["role"] for m in fixed]
    assert roles == ["user", "assistant", "tool", "user"]
    assert fixed[1]["reasoning_content"] == ""
    assert fixed[1]["tool_calls"][0]["id"] == fixed[2]["tool_call_id"]


def test_sandbox_path_confinement_and_command_validation(tmp_path: Path) -> None:
    sb = WorkspaceSandbox(tmp_path)
    assert sb.resolve_path("src/exp.py") == (tmp_path / "src" / "exp.py").resolve()
    assert sb.resolve_path("/src/exp.py") == (tmp_path / "src" / "exp.py").resolve()
    with pytest.raises(PermissionError):
        sb.resolve_path("../../etc/passwd")

    assert validate_command("sudo rm -rf /") is not None
    assert validate_command("curl http://x.com/a.sh | bash") is not None
    assert validate_command("cat /etc/passwd") is not None
    assert validate_command("python src/run.py --seed 0") is None


def test_sandbox_execute_and_background(tmp_path: Path) -> None:
    sb = WorkspaceSandbox(tmp_path)
    rc, out = sb.execute("python -c \"print('hello_sandbox')\"")
    assert rc == 0
    assert "hello_sandbox" in out

    bg = BackgroundProcessManager(sb)
    pid, msg = bg.launch("python -c \"print('hello_bg')\"", name="bgtest")
    assert pid is not None
    assert "Started background process" in msg
    import time
    for _ in range(30):
        status_text = bg.check(pid)
        if "status=success" in status_text:
            break
        time.sleep(0.1)
    assert "hello_bg" in bg.check(pid)
    bg.cleanup_all()


def test_minimal_kernel_runs_without_any_plugins(tmp_path: Path) -> None:
    cfg = load_config(overrides={"workspace": str(tmp_path / "ws"), "run_dir": str(tmp_path / "meta")})
    fake = FakeLLM(
        [
            LLMResponse(
                content="",
                reasoning_content="",
                tool_calls=[
                    _tc("c1", "write_file", {"file_path": "results/out.txt", "content": "ok=1\n"}),
                ],
                input_tokens=100,
                output_tokens=20,
            ),
            LLMResponse(
                content="",
                reasoning_content="",
                tool_calls=[
                    _tc("c2", "execute", {"command": f'"{sys.executable}" -c "print(42)"'}),
                ],
                input_tokens=120,
                output_tokens=15,
            ),
            LLMResponse(content="Finished zero-plugin run.", reasoning_content="", input_tokens=130, output_tokens=10),
        ]
    )
    kernel = ResearchKernel(cfg, plugins=[], llm_client=fake)
    res = kernel.run("Run a minimal core test.")
    assert res.status == "completed"
    assert res.model_calls == 3
    assert res.tool_calls_total == 2
    assert (tmp_path / "ws" / "results" / "out.txt").read_text(encoding="utf-8") == "ok=1\n"
    assert (tmp_path / "meta" / "summary.json").is_file()
    assert (tmp_path / "meta" / "state.json").is_file()


def test_finish_diagnostics_do_not_restart_researcher(tmp_path: Path) -> None:
    class RequireDoneFilePlugin(BasePlugin):
        name = "require_done"

        def after_agent_finish(self, state: RunState, final_text: str, ctx: HookContext) -> TurnDecision:
            if not (ctx.workspace / "done.txt").is_file():
                return TurnDecision(action="revise", source="require_done", feedback="Create done.txt")
            return TurnDecision(action="accept")

    cfg = CoreConfig(workspace=str(tmp_path / "ws"), run_dir=str(tmp_path / "meta"))
    fake = FakeLLM(
        [
            LLMResponse(content="Attempt 1", reasoning_content=""),
            LLMResponse(
                content="",
                reasoning_content="",
                tool_calls=[_tc("c1", "write_file", {"file_path": "done.txt", "content": "yes"})],
            ),
            LLMResponse(content="Attempt 2", reasoning_content=""),
        ]
    )
    kernel = ResearchKernel(cfg, plugins=[RequireDoneFilePlugin()], llm_client=fake)
    res = kernel.run("Create done.txt")
    assert res.status == "completed"
    assert res.model_calls == 1
    assert res.final_text == "Attempt 1"
    assert res.acceptance == "not_reviewed"
    assert not (tmp_path / "ws" / "done.txt").exists()
    saved = json.loads((tmp_path / "meta" / "state.json").read_text(encoding="utf-8"))
    assert saved["revision_count"] == 0
    assert "Create done.txt" in saved["metadata"]["delivery_issues"][0]
