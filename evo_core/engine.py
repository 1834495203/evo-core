"""Minimal ResearchKernel: ReAct loop + plugin hook orchestration + persistence."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .config import CoreConfig
from .llm import ChatClient, LLMResponse, redact_secret
from .prompts import assemble_system_prompt
from .protocol import HookContext, ResearchPlugin, RunState, ToolCallDecision, ToolDef, TurnDecision
from .sandbox import BackgroundProcessManager, WorkspaceSandbox
from .tools_core import build_core_tools


@dataclass
class RunResult:
    run_id: str
    status: str  # completed | stopped | paused | failed
    final_text: str
    model_calls: int
    tool_calls_total: int
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    elapsed_seconds: float
    workspace: Path
    run_dir: Path
    error: str | None = None
    acceptance: str = "pending"
    scientific_outcome: str = "unassessed"
    study_id: str | None = None


class ResearchKernel:
    """Minimal autonomous research execution loop driven by enabled ResearchPlugins."""

    def __init__(
        self,
        config: CoreConfig,
        plugins: list[ResearchPlugin] | None = None,
        *,
        llm_client: ChatClient | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = config
        self.modules = None
        self.workspace = Path(config.workspace).resolve()
        self.run_dir = Path(config.run_dir).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "requests").mkdir(parents=True, exist_ok=True)

        if llm_client is None:
            api_key, base_url = config.resolved_api_credentials()
            llm_client = ChatClient(
                api_key=api_key,
                base_url=base_url,
                default_model=config.model,
                thinking=config.thinking,
                reasoning_effort=config.reasoning_effort,
                temperature=config.temperature,
                max_output_tokens=config.max_output_tokens,
                timeout=config.request_timeout,
                max_retries=config.max_retries,
            )
        self.llm = llm_client
        from .runtime.host import Host
        from .runtime.models import ObservedModels
        from .system.storage.artifacts import ArtifactStore
        from .modules.adapters import configure_host
        module_root = self.workspace / ".research_modules"
        self.host = Host(module_root, ArtifactStore(module_root / "host_artifacts"), emit=self._emit, workspace=self.workspace)
        configure_host(self.host, config)
        self.llm = ObservedModels(self.llm, self.host, self.run_dir)
        self.host.models = self.llm
        self.llm.trace = self._trace_model
        self.sandbox = WorkspaceSandbox(
            self.workspace,
            run_dir=self.run_dir,
            dangerous=config.dangerous_mode,
            default_timeout=config.command_timeout,
            max_output_chars=config.max_output_chars,
            science_python=config.science_python,
        )
        self.bg_manager = BackgroundProcessManager(self.sandbox)
        self.on_event = on_event

        from .plugins import load_plugins

        all_plugins = plugins if plugins is not None else load_plugins(config)
        self.plugins: list[ResearchPlugin] = [
            p for p in all_plugins if p.enabled(config)
        ]

    def _emit(self, kind: str, **details: Any) -> None:
        payload = {"time": round(time.time(), 3), "event": kind, **details}
        if "identity" not in payload and hasattr(self, "host"):
            from .runtime.host import CURRENT_CALL
            payload["identity"] = asdict(CURRENT_CALL.get() or self.host.identity)
        safe = json.loads(
            redact_secret(
                json.dumps(payload, ensure_ascii=False, default=str),
                getattr(self.llm, "api_key", None),
            )
        )
        with (self.run_dir / "events.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(safe, ensure_ascii=False) + "\n")
        from .runtime.host import EXPORTING_TRACE
        if self.modules is not None and self.modules.has_capability("observability.trace") and not EXPORTING_TRACE.get():
            token = EXPORTING_TRACE.set(True)
            try:
                with self.llm.scope("host"):
                    self.modules.service("observability.trace").log_span(kind, safe)
            except Exception as exc:
                self._export_error("tracing", exc)
            finally:
                EXPORTING_TRACE.reset(token)
        if self.on_event is not None:
            try:
                self.on_event(safe)
            except Exception as exc:
                self._export_error("observer", exc)

    def _export_error(self, source, exc):
        with (self.run_dir / "export_errors.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"source": source, "error": redact_secret(str(exc), getattr(self.llm, "api_key", None)), "time": time.time()}) + "\n")

    def _trace_model(self, identity, sequence, messages, response, duration):
        from .runtime.host import EXPORTING_TRACE
        if self.modules is not None and self.modules.has_capability("observability.trace"):
            token = EXPORTING_TRACE.set(True)
            try:
                with self.llm.scope("host"):
                    tracer = self.modules.service("observability.trace")
                    if hasattr(tracer, "log_llm_call"):
                        tracer.log_llm_call(identity.actor_id, sequence, messages, response,
                                            duration_ms=duration, parent_span_id=identity.parent_call_id)
            except Exception as exc:
                self._export_error("tracing", exc)
            finally:
                EXPORTING_TRACE.reset(token)

    def _save_state(self, state: RunState) -> None:
        state.metadata["model_calls_by_actor"] = self.llm.counts_snapshot()
        state.metadata["all_model_usage"] = self.llm.usage_snapshot()
        if self.modules is not None and self.modules.has_capability("conversation.context"):
            with self.llm.scope("host"):
                self.modules.service("conversation.context").save(state.run_id, "researcher", state.messages)
        state.elapsed_seconds = round(time.time() - state.started_at, 2)
        state_file = self.run_dir / "state.json"
        raw = json.dumps(asdict(state), ensure_ascii=False, indent=2, default=str)
        state_file.write_text(
            redact_secret(raw, getattr(self.llm, "api_key", None)),
            encoding="utf-8",
        )
        summary = {
            "run_id": state.run_id,
            "status": state.status,
            "acceptance": state.acceptance,
            "scientific_outcome": state.scientific_outcome,
            "study_id": state.study_id,
            "workspace": str(self.workspace),
            "model": self.config.model,
            "model_calls": state.model_calls,
            "tool_calls_total": state.tool_calls_total,
            "input_tokens": state.input_tokens,
            "output_tokens": state.output_tokens,
            "cached_input_tokens": state.cached_input_tokens,
            "elapsed_seconds": state.elapsed_seconds,
            "revision_count": state.revision_count,
            "delivery_issues": state.metadata.get("delivery_issues", []),
            "enabled_plugins": [p.name for p in self.plugins],
            "todos": state.todos,
            "model_calls_by_actor": self.llm.counts_snapshot(),
            "total_model_calls": sum(self.llm.counts_snapshot().values()),
            "all_model_usage": self.llm.usage_snapshot(),
        }
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _save_trace(self, call_num: int, req_msgs: list[dict[str, Any]], resp: LLMResponse) -> None:
        secret = getattr(self.llm, "api_key", None)
        req_path = self.run_dir / "requests" / f"{call_num:04d}-input.json"
        resp_path = self.run_dir / "requests" / f"{call_num:04d}-output.json"
        req_path.write_text(
            redact_secret(json.dumps(req_msgs, ensure_ascii=False, indent=2, default=str), secret),
            encoding="utf-8",
        )
        resp_payload = {
            "finish_reason": resp.finish_reason,
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "cached_tokens": resp.cached_tokens,
            "content": resp.content,
            "reasoning_chars": len(resp.reasoning_content),
            "tool_calls": [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments_raw, "parse_error": tc.parse_error}
                for tc in resp.tool_calls
            ],
        }
        resp_path.write_text(
            redact_secret(json.dumps(resp_payload, ensure_ascii=False, indent=2, default=str), secret),
            encoding="utf-8",
        )

    def _build_ctx(self) -> HookContext:
        return HookContext(
            config=self.config,
            workspace=self.workspace,
            run_dir=self.run_dir,
            sandbox=self.sandbox,
            bg_manager=self.bg_manager,
            llm_client=self.llm,
            emit_event=self._emit,
            host=self.host,
        )

    def _collect_tools(self, state: RunState, ctx: HookContext) -> dict[str, ToolDef]:
        tools_list: list[ToolDef] = list(build_core_tools(state, ctx))
        for plugin in self.plugins:
            tools_list.extend(plugin.tools(ctx))
        registry: dict[str, ToolDef] = {}
        for t in tools_list:
            from .modules.adapters import annotate_tool
            t = annotate_tool(t, ctx.module_registry)
            if t.name in registry:
                raise ValueError(f"Duplicate tool: {t.name}")
            registry[t.name] = t
        if ctx.module_registry is not None and "list_modules" not in registry:
            from .modules.tools import module_tools

            for t in module_tools(ctx):
                if t.name in registry: raise ValueError(f"Duplicate tool: {t.name}")
                registry[t.name] = annotate_tool(t, ctx.module_registry)
        ctx.tool_specs = registry
        if ctx.module_registry is not None:
            from .modules.adapters import bind_runtime_tools
            bind_runtime_tools(ctx.module_registry, registry, lambda name, args, actor: self.execute_tool_with_hooks(
                state, ctx, registry, name, args, actor=actor, return_result=True))
        return registry

    def execute_tool_with_hooks(
        self, state, ctx, tool_registry, tool_name, arguments, *, call_num=0,
        actor="main", run_after_hooks=True, fallback_tool=None, return_result=False, parent_identity=None,
    ):
        from .runtime.dispatch import invoke_tool
        from .runtime.contracts import Result
        from .runtime.host import call_scope, CURRENT_CALL
        from dataclasses import replace
        from .modules.adapters import annotate_tool
        previous = ctx.actor
        ctx.actor = actor
        try:
            tool = tool_registry.get(tool_name) or fallback_tool
            if tool is None:
                result = Result.failure("tool_missing", tool_name)
                return result if return_result else f"Unknown tool '{tool_name}'"
            annotate_tool(tool, ctx.module_registry)
            with call_scope(replace(parent_identity or CURRENT_CALL.get() or self.host.identity, actor_id=actor)):
                result, text = invoke_tool(self.host, tool, arguments, lambda: self._execute_tool_with_hooks(
                    state, ctx, tool_registry, tool_name, arguments, call_num=call_num, actor=actor,
                    run_after_hooks=run_after_hooks, fallback_tool=fallback_tool))
            if result.error and result.error.code == "permission_denied":
                self._emit("tool_blocked", call=call_num, tool=tool_name, actor=actor, source="permissions", reason=result.error.message)
            return result if return_result else text
        finally:
            ctx.actor = previous

    def _execute_tool_with_hooks(
        self,
        state: RunState,
        ctx: HookContext,
        tool_registry: dict[str, ToolDef],
        tool_name: str,
        arguments: dict[str, Any],
        *,
        call_num: int = 0,
        actor: str = "main",
        run_after_hooks: bool = True,
        fallback_tool: ToolDef | None = None,
    ) -> str:
        """Execute a tool through unified before_tool_call and after_tool_call plugin hooks."""
        tool_def = tool_registry.get(tool_name) or fallback_tool
        if tool_def is None:
            return f"Unknown tool '{tool_name}'. Available: {', '.join(tool_registry)}"

        for plugin in self.plugins:
            decision: ToolCallDecision = plugin.before_tool_call(state, tool_name, arguments, ctx)
            if decision.action == "block":
                source = decision.source or plugin.name
                self._emit(
                    "tool_blocked",
                    call=call_num,
                    tool=tool_name,
                    actor=actor,
                    source=source,
                    reason=decision.feedback[:300],
                )
                return f"[TOOL BLOCKED by {source}] {decision.feedback}"
        if self.modules is not None:
            decision = self.modules.before_tool_call(state, tool_name, arguments, ctx)
            if getattr(decision, "action", "allow") == "block":
                source = getattr(decision, "source", "") or "module"
                self._emit(
                    "tool_blocked",
                    call=call_num,
                    tool=tool_name,
                    actor=actor,
                    source=source,
                    reason=decision.feedback[:300],
                )
                return f"[TOOL BLOCKED by {source}] {decision.feedback}"

        state.tool_calls_total += 1
        raw_result = None
        try:
            from .runtime.contracts import Result
            raw_result = tool_def.handler(**arguments)
            result = json.dumps(raw_result.wire(), ensure_ascii=False) if isinstance(raw_result, Result) else str(raw_result)
        except Exception as exc:
            result = f"[TOOL ERROR] {tool_name} raised {type(exc).__name__}: {exc}"

        self._emit(
            "tool_end",
            call=call_num,
            tool=tool_name,
            actor=actor,
            result_chars=len(result),
            ok=not result.startswith(("[TOOL ERROR]", "Tool call argument error", "Error:")),
        )
        if run_after_hooks:
            for plugin in self.plugins:
                plugin.after_tool_call(state, tool_name, arguments, result, ctx)
            if self.modules is not None:
                self.modules.after_tool_call(state, tool_name, arguments, result, ctx)
        return raw_result if isinstance(raw_result, Result) else result

    def _build_initial_state(self, task_prompt: str, rubric: str | None, resume: bool) -> RunState:
        state_file = self.run_dir / "state.json"
        if resume and state_file.is_file():
            raw = json.loads(state_file.read_text(encoding="utf-8"))
            state = RunState(**raw)
            state.status = "running"
            if state.acceptance in {"unverified_budget_stop", "rejected", "not_reviewed"}:
                state.acceptance = "pending"
            state.metadata.pop("budget_stop_requested", None)
            state.metadata.pop("budget_stop_reason", None)
            state.started_at = time.time() - float(state.elapsed_seconds or 0.0)
            state.add_notice("Resuming interrupted run from saved state. Continue from existing files.")
            return state
        return RunState(
            run_id=uuid.uuid4().hex[:10],
            task_prompt=task_prompt,
            rubric=rubric,
            messages=[{"role": "user", "content": task_prompt}],
            started_at=time.time(),
            status="running",
        )

    def _append_notices(self, state: RunState, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Attach ephemeral per-turn notices to the tail message, preserving static prefix cache."""
        if not state.pending_notices:
            return messages
        notices_block = "\n".join(f"[System Notice] {n}" for n in state.pending_notices)
        state.pending_notices.clear()
        updated = [dict(m) for m in messages]
        if updated and updated[-1].get("role") in {"user", "tool"}:
            updated[-1]["content"] = f"{updated[-1].get('content', '')}\n\n{notices_block}".strip()
        else:
            updated.append({"role": "user", "content": notices_block})
        return updated

    def run(
        self,
        task_prompt: str,
        *,
        rubric: str | None = None,
        resume: bool = False,
    ) -> RunResult:
        """Run until the researcher finishes or execution is stopped; no approval loop."""
        ctx = self._build_ctx()
        state = self._build_initial_state(task_prompt, rubric, resume)
        from dataclasses import replace
        self.host.identity = replace(self.host.identity, run_id=state.run_id, session_id=state.run_id)
        self.llm.seed_counts(state.metadata.get("model_calls_by_actor", {}), state.metadata.get("all_model_usage"))
        ctx.state = state
        for plugin in self.plugins:
            plugin.before_run(state, ctx)
        if ctx.module_registry is None and self.config.modules:
            from .modules.registry import build_modules

            ctx.module_registry = build_modules(
                self.workspace / ".research_modules",
                ctx.research_system,
                self.config.modules,
                host=self.host,
            )
        self.modules = ctx.module_registry
        if resume and len(state.messages) <= 1 and self.modules is not None and self.modules.has_capability("conversation.context"):
            with self.llm.scope("host"):
                saved_msgs = self.modules.service("conversation.context").load(state.run_id, "researcher")
            if saved_msgs:
                state.messages = list(saved_msgs)

        tool_registry = self._collect_tools(state, ctx)
        ctx.execute_tool = lambda name, args, actor="subagent", fallback=None: self.execute_tool_with_hooks(
            state,
            ctx,
            tool_registry,
            name,
            args,
            call_num=state.model_calls,
            actor=actor,
            run_after_hooks=True,
            fallback_tool=fallback,
        )
        tool_schemas = [t.openai_schema() for t in tool_registry.values() if self.host.policy.permits("main", t.permissions)]

        plugin_sections: list[str] = []
        for plugin in self.plugins:
            sec = plugin.system_prompt_section(state, ctx)
            if sec:
                plugin_sections.append(sec)
        if self.modules is not None:
            plugin_sections.extend(self.modules.system_prompt_sections(state, ctx))
        system_prompt = assemble_system_prompt(plugin_sections, rubric=rubric)
        self._emit(
            "run_start",
            run_id=state.run_id,
            study_id=state.study_id,
            model=self.config.model,
            plugins=[p.name for p in self.plugins],
            tools=list(tool_registry.keys()),
            system_prompt_chars=len(system_prompt),
        )

        final_text = ""
        error_msg: str | None = None
        last_call_sig: tuple[str, str] | None = None
        repeat_error_streak = 0

        try:
            while True:
                if state.metadata.get("budget_stop_requested"):
                    state.status = "stopped"
                    if state.acceptance == "pending":
                        state.acceptance = "unverified_budget_stop"
                    error_msg = state.metadata.get("budget_stop_reason", "Budget limit reached")
                    break

                # Build message view for this call: static system prompt + state messages + plugin hooks
                outgoing = [{"role": "system", "content": system_prompt}, *state.messages]
                for plugin in self.plugins:
                    outgoing = plugin.before_model_call(state, outgoing, ctx)
                if self.modules is not None:
                    outgoing = self.modules.before_model_call(state, outgoing, ctx)
                outgoing = self._append_notices(state, outgoing)

                if state.metadata.get("user_pause_requested"):
                    state.status = "paused"
                    error_msg = "Paused by user command"
                    break

                if state.metadata.get("budget_stop_requested"):
                    state.status = "stopped"
                    if state.acceptance == "pending":
                        state.acceptance = "unverified_budget_stop"
                    error_msg = state.metadata.get("budget_stop_reason", "Budget limit reached")
                    break

                state.model_calls += 1
                call_num = state.model_calls
                self._emit("model_start", call=call_num, msg_count=len(outgoing))
                t0 = time.perf_counter()
                resp = self.llm.complete(outgoing, tools=tool_schemas)
                duration_ms = round((time.perf_counter() - t0) * 1000, 2)
                state.input_tokens += resp.input_tokens
                state.output_tokens += resp.output_tokens
                state.cached_input_tokens += resp.cached_tokens
                self._save_trace(call_num, outgoing, resp)

                tool_names = [tc.name for tc in resp.tool_calls]
                self._emit(
                    "model_end",
                    call=call_num,
                    finish_reason=resp.finish_reason,
                    tools=tool_names,
                    input_tokens=resp.input_tokens,
                    output_tokens=resp.output_tokens,
                    cached_tokens=resp.cached_tokens,
                )

                # If model hit output token limit with broken tool JSON, recover cleanly
                if resp.finish_reason == "length" and any(tc.parse_error for tc in resp.tool_calls):
                    state.messages.append({"role": "assistant", "content": resp.content or "(truncated)"})
                    state.messages.append(
                        {
                            "role": "user",
                            "content": (
                                "[Recovery] Your previous response exceeded the output token limit "
                                "and truncated a tool call. Split large writes into smaller modules or use edit_file."
                            ),
                        }
                    )
                    self._save_state(state)
                    continue

                assistant_msg = resp.raw_assistant_message or {
                    "role": "assistant",
                    "content": resp.content,
                    **({
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": tc.name, "arguments": tc.arguments_raw},
                            }
                            for tc in resp.tool_calls
                        ]
                    } if resp.tool_calls else {}),
                }
                state.messages.append(assistant_msg)

                if resp.tool_calls:
                    for tc in resp.tool_calls:
                        if tc.parse_error:
                            state.tool_calls_total += 1
                            result = f"Tool call argument error for '{tc.name}': {tc.parse_error}"
                            self._emit(
                                "tool_end",
                                call=call_num,
                                tool=tc.name,
                                actor="main",
                                result_chars=len(result),
                                ok=False,
                            )
                        else:
                            result = self.execute_tool_with_hooks(
                                state,
                                ctx,
                                tool_registry,
                                tc.name,
                                tc.arguments,
                                call_num=call_num,
                                actor="main",
                                parent_identity=getattr(resp, "call_identity", None),
                                run_after_hooks=False,
                            )

                        state.messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "name": tc.name,
                                "content": result,
                            }
                        )
                        if not result.startswith("[TOOL BLOCKED"):
                            for plugin in self.plugins:
                                plugin.after_tool_call(state, tc.name, tc.arguments, result, ctx)
                            if self.modules is not None:
                                self.modules.after_tool_call(state, tc.name, tc.arguments, result, ctx)

                        # Loop-guard against stuck identical failing calls
                        sig = (tc.name, tc.arguments_raw)
                        failed = "failed with exit code" in result or result.startswith(
                            ("Error:", "[TOOL ERROR]", "[TOOL BLOCKED")
                        )
                        if failed and sig == last_call_sig:
                            repeat_error_streak += 1
                            if repeat_error_streak >= 4:
                                state.add_notice(
                                    f"Tool '{tc.name}' failed identically {repeat_error_streak} times. "
                                    "Do not repeat the same call; diagnose the root cause or try a different approach."
                                )
                        elif failed:
                            last_call_sig = sig
                            repeat_error_streak = 1
                        else:
                            last_call_sig = None
                            repeat_error_streak = 0

                    self._save_state(state)
                    if state.metadata.get("user_pause_requested"):
                        state.status = "paused"
                        error_msg = "Paused by user command"
                        break
                    continue

                # The researcher decides to finish. Local checks only record diagnostics;
                # they cannot require another model turn or approve scientific conclusions.
                final_text = resp.content or ""
                finish_issues: list[str] = []
                for plugin in self.plugins:
                    try:
                        decision: TurnDecision = plugin.after_agent_finish(state, final_text, ctx)
                        if decision.feedback:
                            finish_issues.append(f"[{decision.source or plugin.name}] {decision.feedback}")
                    except Exception as exc:
                        finish_issues.append(f"[{plugin.name}] Local finish check failed: {type(exc).__name__}: {exc}")
                if self.modules is not None:
                    for decision in self.modules.after_agent_finish(state, final_text, ctx):
                        if getattr(decision, "feedback", ""):
                            finish_issues.append(f"[{getattr(decision, 'source', 'module') or 'module'}] {decision.feedback}")
                state.metadata["delivery_issues"] = finish_issues
                state.metadata.pop("unresolved_acceptance_issues", None)
                self._emit("delivery_check", issues=finish_issues, scientific_review=False)
                state.status = "completed"
                state.acceptance = "not_reviewed"
                self._save_state(state)
                break

        except Exception as exc:
            state.status = "failed"
            error_msg = redact_secret(f"{type(exc).__name__}: {exc}", getattr(self.llm, "api_key", None))
            self._save_state(state)
            self._emit("run_error", error=error_msg)
        finally:
            if state.status in {"stopped", "failed"}:
                self.host.cancellation.request()
            for plugin in self.plugins:
                try:
                    plugin.after_run(state, ctx)
                except Exception:
                    pass
            self.bg_manager.cleanup_all()
            self._save_state(state)
            self._emit(
                "run_end",
                status=state.status,
                acceptance=state.acceptance,
                scientific_outcome=state.scientific_outcome,
                model_calls=state.model_calls,
                tool_calls=state.tool_calls_total,
                elapsed_seconds=state.elapsed_seconds,
            )
            if self.modules is not None:
                self.modules.close()

        return RunResult(
            run_id=state.run_id,
            status=state.status,
            final_text=final_text,
            model_calls=state.model_calls,
            tool_calls_total=state.tool_calls_total,
            input_tokens=state.input_tokens,
            output_tokens=state.output_tokens,
            cached_input_tokens=state.cached_input_tokens,
            elapsed_seconds=state.elapsed_seconds,
            workspace=self.workspace,
            run_dir=self.run_dir,
            error=error_msg,
            acceptance=state.acceptance,
            scientific_outcome=state.scientific_outcome,
            study_id=state.study_id,
        )
