"""Pluggable module protocol and shared kernel runtime state."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable


@dataclass
class ToolDef:
    """Self-contained tool schema + handler registered into the kernel."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., str]
    permissions: tuple[str, ...] = ()
    owner: str = "host"
    wire_result: bool = False
    permission_resolver: Callable | None = None
    reference_kind: str = ""
    output_schema: dict | None = None

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class TurnDecision:
    """Local finish diagnostic; feedback is recorded without restarting the agent.

    The legacy action field remains for plugin compatibility, not as an approval gate.
    """

    action: str = "accept"  # "accept" | "revise"
    feedback: str = ""
    source: str = ""


@dataclass
class ToolCallDecision:
    """Decision returned by before_tool_call hooks prior to tool execution."""

    action: str = "allow"  # "allow" | "block"
    feedback: str = ""
    source: str = ""


@dataclass
class HookContext:
    """Shared dependencies exposed to core tools and plugins."""

    config: Any
    workspace: Path
    run_dir: Path
    sandbox: Any
    bg_manager: Any
    llm_client: Any
    emit_event: Callable[..., None]
    state: Any = None
    execute_tool: Callable[..., str] | None = None
    research_system: Any = None
    study_id: str | None = None
    module_registry: Any = None
    actor: str = "main"
    host: Any = None
    tool_specs: dict = field(default_factory=dict)
    observe_file: Callable | None = None


@dataclass
class RunState:
    """Mutable state of an autonomous research run, persisted across turns."""

    run_id: str
    task_prompt: str
    rubric: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    todos: list[dict[str, str]] = field(default_factory=list)
    model_calls: int = 0
    tool_calls_total: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    started_at: float = 0.0
    elapsed_seconds: float = 0.0
    revision_count: int = 0
    status: str = "init"
    acceptance: str = "pending"
    scientific_outcome: str = "unassessed"
    study_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    pending_notices: list[str] = field(default_factory=list)

    def add_notice(self, text: str) -> None:
        if text and text.strip():
            self.pending_notices.append(text.strip())


@runtime_checkable
class ResearchPlugin(Protocol):
    """Unified extension interface replacing monolithic middleware stacks."""

    name: str

    def enabled(self, cfg: Any) -> bool:
        ...

    def system_prompt_section(self, state: RunState, ctx: HookContext) -> str | None:
        """Static system-prompt section (computed once per run to preserve KV cache)."""
        ...

    def tools(self, ctx: HookContext) -> list[ToolDef]:
        ...

    def before_run(self, state: RunState, ctx: HookContext) -> None:
        ...

    def before_model_call(
        self,
        state: RunState,
        messages: list[dict[str, Any]],
        ctx: HookContext,
    ) -> list[dict[str, Any]]:
        ...

    def before_tool_call(
        self,
        state: RunState,
        tool_name: str,
        arguments: dict[str, Any],
        ctx: HookContext,
    ) -> ToolCallDecision:
        ...

    def after_tool_call(
        self,
        state: RunState,
        tool_name: str,
        arguments: dict[str, Any],
        result: str,
        ctx: HookContext,
    ) -> None:
        ...

    def after_agent_finish(
        self,
        state: RunState,
        final_text: str,
        ctx: HookContext,
    ) -> TurnDecision:
        ...

    def after_run(self, state: RunState, ctx: HookContext) -> None:
        ...


class BasePlugin:
    """Convenience no-op base class so plugins only override hooks they need."""

    name: str = "base"

    def enabled(self, cfg: Any) -> bool:
        return True

    def system_prompt_section(self, state: RunState, ctx: HookContext) -> str | None:
        return None

    def tools(self, ctx: HookContext) -> list[ToolDef]:
        return []

    def before_run(self, state: RunState, ctx: HookContext) -> None:
        return None

    def before_model_call(
        self,
        state: RunState,
        messages: list[dict[str, Any]],
        ctx: HookContext,
    ) -> list[dict[str, Any]]:
        return messages

    def before_tool_call(
        self,
        state: RunState,
        tool_name: str,
        arguments: dict[str, Any],
        ctx: HookContext,
    ) -> ToolCallDecision:
        return ToolCallDecision(action="allow")

    def after_tool_call(
        self,
        state: RunState,
        tool_name: str,
        arguments: dict[str, Any],
        result: str,
        ctx: HookContext,
    ) -> None:
        return None

    def after_agent_finish(
        self,
        state: RunState,
        final_text: str,
        ctx: HookContext,
    ) -> TurnDecision:
        return TurnDecision(action="accept")

    def after_run(self, state: RunState, ctx: HookContext) -> None:
        return None
