"""Pluggable sub-agent delegation module (isolated child loops sharing the workspace)."""

from __future__ import annotations

from itertools import count
import json
import time
from typing import Any

from ..llm import redact_secret
from ..protocol import BasePlugin, HookContext, RunState, ToolDef
from ..tools_core import build_core_tools

ROLE_PROMPTS: dict[str, str] = {
    "coder": (
        "You are the dedicated Coder agent with an isolated context. Implement clean, modular Python code "
        "and unit tests under src/ and tests/ according to the Researcher's specification. "
        "When a coding task_id is provided, freeze your finished files and run specification by calling "
        "`submit_code(task_id=..., files=[...], run={...})` and report the resulting `submission_id` and "
        "test command back to the Researcher. You cannot approve code or execute experiments yourself."
        " Include only the complete source/data dependencies and tests for this submission; unrelated workspace files are not required. "
        "For result checks, set run.check_entrypoint to a pytest test file (check_runner='pytest' or auto), "
        "or provide a standalone script writing a nonempty outputs/checks.json (check_runner='checks_json')."
        "Set submit_code.test_paths to the unit tests to run before the experiment; post-run checks that need result files are separate."
    ),
    "planner": (
        "You are the planner sub-agent. Design concrete, falsifiable experimental plans, "
        "baselines, ablations, metrics, and sanity checks. Do not fabricate results."
    ),
    "research": (
        "You are the research sub-agent. Survey papers and open-source implementations using "
        "available search tools, verify URLs/arXiv IDs, and summarize actionable baselines and datasets."
    ),
    "code": (
        "You are the code sub-agent. Write clean, modular, reproducible Python scripts and unit tests "
        "under src/ and tests/. Run quick syntax/sanity checks after writing code."
    ),
    "debug": (
        "You are the debug sub-agent. Reproduce runtime failures or suspicious numbers, isolate root causes, "
        "apply minimal verified patches, and re-run checks."
    ),
    "analysis": (
        "You are the data-analysis sub-agent. Compute metrics, statistical uncertainty/confidence intervals, "
        "and generate publication-grade matplotlib figures under figures/ from saved raw results."
    ),
    "writing": (
        "You are the writing sub-agent. Draft or refine final_report.md strictly grounded in verified files "
        "in results/, figures/, and literature_notes.md. Never fabricate numbers or citations."
    ),
}

_ROLE_CORE_TOOLS: dict[str, set[str]] = {
    "planner": {"ls", "read_file", "write_file", "think", "get_research_status"},
    "research": {
        "ls",
        "read_file",
        "write_file",
        "think",
        "search_arxiv",
        "search_openalex",
        "search_github",
        "search_semantic_scholar",
        "fetch_webpage",
        "read_paper_fulltext",
        "get_research_status",
        "record_research_note",
    },
    "code": {"ls", "read_file", "write_file", "edit_file", "execute", "think", "run_in_background", "get_research_status"},
    "debug": {"ls", "read_file", "write_file", "edit_file", "execute", "think", "get_research_status"},
    "analysis": {"ls", "read_file", "write_file", "edit_file", "execute", "think", "get_research_status", "record_research_note"},
    "writing": {"ls", "read_file", "write_file", "edit_file", "think", "get_research_status"},
}


class SubagentsPlugin(BasePlugin):
    """Provides on-demand `delegate_task` sub-agent execution when enabled."""

    name = "subagents"

    def enabled(self, cfg: Any) -> bool:
        return bool(cfg.plugins.subagents.enabled)

    def system_prompt_section(self, state: RunState, ctx: HookContext) -> str | None:
        roles = ", ".join(ctx.config.plugins.subagents.roles)
        return (
            "# Sub-Agent Delegation\n"
            f"You may optionally use `delegate_task(role, instruction)` for focused sub-tasks "
            f"(available roles: {roles}). Pass workspace file paths rather than pasting large files. "
            "For straightforward sequential steps, working directly with core tools is equally valid."
        )

    def run_role(self, role: str, instruction: str, ctx: HookContext) -> str:
        scfg = ctx.config.plugins.subagents
        max_child_calls = max(3, int(scfg.max_child_calls)) if scfg.max_child_calls is not None else None

        from .budget import BudgetPlugin
        from .literature import LiteratureSearchPlugin

        lit_plugin = LiteratureSearchPlugin()
        budget_plugin = BudgetPlugin()

        active_state = ctx.state if isinstance(ctx.state, RunState) else RunState(run_id="child", task_prompt=instruction)
        candidate_tools: list[ToolDef] = list(build_core_tools(active_state, ctx))
        if lit_plugin.enabled(ctx.config):
            candidate_tools.extend(lit_plugin.tools(ctx))
        if ctx.research_system is not None and ctx.config.plugins.harness.enabled:
            from .harness import ResearchHarnessPlugin

            candidate_tools.extend(ResearchHarnessPlugin().tools(ctx))
        elif ctx.module_registry is not None:
            from ..modules.tools import module_tools

            candidate_tools.extend(module_tools(ctx))

        allowed_names = _ROLE_CORE_TOOLS.get("code" if role == "coder" else role, {"ls", "read_file", "write_file", "execute", "think"})
        has_modules = ctx.research_system is not None or ctx.module_registry is not None
        if has_modules:
            if role in {"coder", "code"}:
                allowed_names = {
                    "ls", "read_file", "write_file", "edit_file", "think", "get_research_status",
                    "submit_code", "inspect_code_submission", "search_long_term_notes",
                    "write_long_term_note", "read_long_term_note", "list_papers",
                    "read_library_paper", "list_modules", "query_research_records",
                    "read_research_document",
                }
            else:
                allowed_names = allowed_names | {
                    "record_research_note", "get_research_status", "record_research_document",
                    "query_research_records", "read_research_document", "set_research_focus",
                    "submit_code", "search_long_term_notes", "write_long_term_note",
                    "read_long_term_note", "list_papers", "read_library_paper", "list_modules",
                }
                if role == "writing":
                    allowed_names = allowed_names | {"archive_research_report", "export_research_bundle"}
                if role in {"debug", "analysis"}:
                    allowed_names = allowed_names | {"submit_experiment", "poll_experiments", "verify_experiment"}
            allowed_names = allowed_names - {"run_in_background"}
        child_tools = {t.name: t for t in candidate_tools if t.name in allowed_names}
        from ..modules.adapters import annotate_tool
        schemas = [t.openai_schema() for t in child_tools.values()
                   if ctx.host is None or ctx.host.policy.permits(f"subagent:{role}", annotate_tool(t).permissions)]

        prompt_key = role if role in ROLE_PROMPTS else ("code" if role == "coder" else role)
        system_msg = (
            f"{ROLE_PROMPTS.get(prompt_key, 'You are a focused research sub-agent.')}\n"
            "Operate inside workspace root `.` using relative paths. "
            "Verify any code you edit and report exact file paths and observed numbers."
        )
        registry = ctx.module_registry or (ctx.research_system.modules if ctx.research_system else None)
        contexts = registry.service("conversation.context", actor="host") if registry and registry.has_capability("conversation.context") else None
        tracer = registry.service("observability.trace") if registry and registry.has_capability("observability.trace") else None
        session_key = active_state.run_id
        history = contexts.load(session_key, role) if contexts else []
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_msg}, *history,
                                        {"role": "user", "content": instruction}]

        ctx.emit_event("subagent_start", role=role, instruction=instruction[:200])
        final = ""
        secret = getattr(ctx.llm_client, "api_key", None)
        files_sample = sorted(
            ctx.sandbox.rel_path(p)
            for p in ctx.workspace.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and ".bg_processes" not in p.parts
        )[:60] if hasattr(ctx, "sandbox") and hasattr(ctx, "workspace") else []

        for step in (range(max_child_calls) if max_child_calls is not None else count()):
            if active_state.metadata.get("user_pause_requested"):
                if contexts:
                    contexts.save(session_key, role, messages[1:])
                return f"[{role} sub-agent stopped by user pause]"
            if budget_plugin.enabled(ctx.config):
                budget_plugin.before_model_call(active_state, messages, ctx)
                if active_state.metadata.get("budget_stop_requested"):
                    reason = active_state.metadata.get("budget_stop_reason", "Budget limit reached")
                    if contexts:
                        contexts.save(session_key, role, messages[1:])
                    ctx.emit_event("subagent_end", role=role, calls=step, budget_stopped=True)
                    return f"[{role} sub-agent stopped by budget: {reason}]"

            if contexts and hasattr(contexts, "edit_and_compact"):
                def _on_sub_compact(prev_tokens, summarized_msgs, kept_msgs, cresp) -> None:
                    active_state.input_tokens += getattr(cresp, "input_tokens", 0)
                    active_state.output_tokens += getattr(cresp, "output_tokens", 0)

                messages[:] = contexts.edit_and_compact(
                    session_key,
                    role,
                    messages,
                    llm_client=ctx.llm_client,
                    model=ctx.config.effective_aux_model(),
                    context_limit_tokens=ctx.config.plugins.compactor.context_limit_tokens,
                    compact_fraction=ctx.config.plugins.compactor.trigger_ratio,
                    keep_recent_messages=ctx.config.plugins.compactor.keep_recent_messages,
                    workspace_files=files_sample,
                    todos=active_state.todos,
                    on_compact=_on_sub_compact,
                )

            t0 = time.perf_counter()
            from contextlib import nullcontext
            model_scope = ctx.llm_client.scope(f"subagent:{role}") if hasattr(ctx.llm_client, "scope") else nullcontext()
            with model_scope:
                resp = ctx.llm_client.complete(messages, tools=schemas)
            duration_ms = round((time.perf_counter() - t0) * 1000, 2)
            active_state.metadata["subagent_model_calls"] = (
                int(active_state.metadata.get("subagent_model_calls") or 0) + 1
            )
            sub_seq = int(active_state.metadata["subagent_model_calls"])
            active_state.input_tokens += resp.input_tokens
            active_state.output_tokens += resp.output_tokens
            active_state.cached_input_tokens += resp.cached_tokens


            if hasattr(ctx, "run_dir") and ctx.run_dir:
                req_dir = ctx.run_dir / "requests"
                req_dir.mkdir(parents=True, exist_ok=True)
                (req_dir / f"subagent-{role}-{sub_seq:04d}-input.json").write_text(
                    redact_secret(json.dumps(messages, ensure_ascii=False, indent=2, default=str), secret),
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
                (req_dir / f"subagent-{role}-{sub_seq:04d}-output.json").write_text(
                    redact_secret(json.dumps(resp_payload, ensure_ascii=False, indent=2, default=str), secret),
                    encoding="utf-8",
                )

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
            messages.append(assistant_msg)
            if not resp.tool_calls:
                final = resp.content or "(no response)"
                ctx.emit_event("subagent_end", role=role, calls=step + 1)
                if contexts:
                    contexts.save(session_key, role, messages[1:])
                return f"[{role} sub-agent completed in {step + 1} call(s)]\n\n{final}"
            for tc in resp.tool_calls:
                if tc.parse_error or tc.name not in child_tools:
                    out = f"Error: {tc.parse_error or 'unavailable tool'}"
                elif ctx.execute_tool is not None:
                    from ..runtime.host import call_scope, CURRENT_CALL
                    with call_scope(getattr(resp, "call_identity", None) or CURRENT_CALL.get()):
                        out = ctx.execute_tool(
                            tc.name,
                            tc.arguments,
                            actor=f"subagent:{role}",
                            fallback=child_tools[tc.name],
                        )
                    active_state.metadata["subagent_tool_calls"] = (
                        int(active_state.metadata.get("subagent_tool_calls") or 0) + 1
                    )
                else:
                    active_state.tool_calls_total += 1
                    active_state.metadata["subagent_tool_calls"] = (
                        int(active_state.metadata.get("subagent_tool_calls") or 0) + 1
                    )
                    try:
                        out = str(child_tools[tc.name].handler(**tc.arguments))
                    except Exception as exc:
                        out = f"[TOOL ERROR] {tc.name}: {exc}"
                if contexts and hasattr(contexts, "offload_tool_output"):
                    out = contexts.offload_tool_output(session_key, role, tc.name, out, max_chars=12_000)
                elif len(out) > 12_000:
                    out = out[:6000] + "\n...[truncated]...\n" + out[-4000:]
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": out}
                )

        if contexts:
            contexts.save(session_key, role, messages[1:])
        ctx.emit_event("subagent_end", role=role, calls=max_child_calls, truncated=True)
        return f"[{role} sub-agent hit max_child_calls={max_child_calls}]\n\n{final}"

    def tools(self, ctx: HookContext) -> list[ToolDef]:
        scfg = ctx.config.plugins.subagents
        allowed_roles = list(scfg.roles)

        def delegate_task(role: str, instruction: str) -> str:
            if role not in allowed_roles:
                return f"Unknown role '{role}'. Allowed roles: {', '.join(allowed_roles)}"
            return self.run_role(role, instruction, ctx)

        return [
            ToolDef(
                name="delegate_task",
                description=(
                    f"Delegate a focused research sub-task to a specialized sub-agent ({', '.join(allowed_roles)}). "
                    "Pass file paths and clear success criteria."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "role": {
                            "type": "string",
                            "enum": allowed_roles,
                            "description": "Sub-agent specialization role.",
                        },
                        "instruction": {
                            "type": "string",
                            "description": "Complete task prompt with workspace paths and expected deliverables.",
                        },
                    },
                    "required": ["role", "instruction"],
                },
                handler=delegate_task,
            )
        ]
