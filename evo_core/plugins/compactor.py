"""Context compaction plugin: offloads large tool outputs and summarizes older turns."""

from __future__ import annotations

import json
import uuid
from typing import Any

from ..llm import approximate_tokens
from ..protocol import BasePlugin, HookContext, RunState

SUMMARY_PROMPT = (
    "You are compacting an ongoing autonomous research session.\n"
    "Summarize the older messages concisely in structured bullet points preserving:\n"
    "1) Chosen research question, datasets, and hypotheses\n"
    "2) Files created/modified and exact Python scripts/tests run\n"
    "3) Exact verified quantitative metrics vs invalidated/buggy runs\n"
    "4) Remaining unresolved issues and next planned steps\n"
    "Do NOT invent any numbers or file paths."
)


def _safe_split_index(messages: list[dict[str, Any]], keep_recent: int) -> int:
    """Find a split index >= 1 that preserves at least *keep_recent* tail messages without splitting tool pairs."""
    n = len(messages)
    if n <= keep_recent + 1:
        return 1
    candidate = max(1, n - keep_recent)
    while candidate > 1 and messages[candidate].get("role") == "tool":
        candidate -= 1
    while candidate > 1 and (
        messages[candidate - 1].get("role") == "assistant"
        and messages[candidate - 1].get("tool_calls")
    ):
        candidate -= 1
    return max(1, candidate)


class ContextCompactorPlugin(BasePlugin):
    """Keeps conversation context bounded without breaking prompt prefix caching."""

    name = "compactor"

    def enabled(self, cfg: Any) -> bool:
        return bool(cfg.plugins.compactor.enabled)

    def after_tool_call(
        self,
        state: RunState,
        tool_name: str,
        arguments: dict[str, Any],
        result: str,
        ctx: HookContext,
    ) -> None:
        ccfg = ctx.config.plugins.compactor
        if not state.messages:
            return
        last = state.messages[-1]
        if last.get("role") != "tool":
            return
        content = str(last.get("content") or "")
        if len(content) <= ccfg.max_tool_result_chars:
            return

        if (
            ctx.module_registry is not None
            and ctx.module_registry.has_capability("conversation.context")
            and hasattr(ctx.module_registry.service("conversation.context", actor="host"), "offload_tool_output")
        ):
            offloaded = ctx.module_registry.service("conversation.context", actor="host").offload_tool_output(
                state.run_id, "researcher", tool_name, content, max_chars=ccfg.max_tool_result_chars
            )
            last["content"] = offloaded
            return

        offload_dir = ctx.run_dir / "tool_logs"
        offload_dir.mkdir(parents=True, exist_ok=True)
        path = offload_dir / f"{tool_name}_{uuid.uuid4().hex[:8]}.txt"
        path.write_text(content, encoding="utf-8")
        half = max(500, ccfg.max_tool_result_chars // 3)
        last["content"] = (
            f"{content[:half]}\n\n"
            f"... [offloaded {len(content)} chars to {path}] ...\n\n"
            f"{content[-half:]}"
        )

    def before_model_call(
        self,
        state: RunState,
        messages: list[dict[str, Any]],
        ctx: HookContext,
    ) -> list[dict[str, Any]]:
        from contextlib import nullcontext
        scope = ctx.llm_client.scope("compactor:researcher") if hasattr(ctx.llm_client, "scope") else nullcontext()
        with scope:
            return self._before_model_call(state, messages, ctx)

    def _before_model_call(self, state, messages, ctx):
        ccfg = ctx.config.plugins.compactor
        files_on_disk = sorted(
            ctx.sandbox.rel_path(p)
            for p in ctx.workspace.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and ".bg_processes" not in p.parts
        )[:60]
        if (
            ctx.module_registry is not None
            and ctx.module_registry.has_capability("conversation.context")
            and hasattr(ctx.module_registry.service("conversation.context", actor="host"), "edit_and_compact")
        ):
            context_mod = ctx.module_registry.service("conversation.context", actor="host")
            header_tokens = max(0, approximate_tokens(messages) - approximate_tokens(state.messages))

            def _on_compact(prev_tokens, summarized_msgs, kept_msgs, resp) -> None:
                state.input_tokens += getattr(resp, "input_tokens", 0)
                state.output_tokens += getattr(resp, "output_tokens", 0)
                ctx.emit_event(
                    "context_compacted",
                    prev_tokens=prev_tokens,
                    summarized_msgs=summarized_msgs,
                    kept_msgs=kept_msgs,
                )

            context_mod.edit_and_compact(
                state.run_id,
                "researcher",
                state.messages,
                llm_client=ctx.llm_client,
                model=ctx.config.effective_aux_model(),
                context_limit_tokens=ccfg.context_limit_tokens,
                compact_fraction=ccfg.trigger_ratio,
                keep_recent_messages=ccfg.keep_recent_messages,
                header_tokens=header_tokens,
                workspace_files=files_on_disk,
                todos=state.todos,
                on_compact=_on_compact,
            )
            return [messages[0], *state.messages] if messages else list(state.messages)

        est_tokens = approximate_tokens(messages)
        threshold = int(ccfg.context_limit_tokens * ccfg.trigger_ratio)
        if est_tokens < threshold or len(state.messages) <= ccfg.keep_recent_messages + 2:
            return messages

        split_at = _safe_split_index(state.messages, ccfg.keep_recent_messages)
        if split_at <= 1:
            return messages

        head_task = state.messages[0]
        to_summarize = state.messages[1:split_at]
        tail = state.messages[split_at:]

        serialized = json.dumps(to_summarize, ensure_ascii=False, default=str)
        if len(serialized) > 90_000:
            serialized = serialized[:45_000] + "\n...[middle omitted]...\n" + serialized[-45_000:]

        summary_req = [
            {"role": "system", "content": SUMMARY_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Workspace files: {files_on_disk}\n"
                    f"Current todos: {state.todos}\n\n"
                    f"History to compact:\n{serialized}"
                ),
            },
        ]
        try:
            resp = ctx.llm_client.complete(
                summary_req,
                model=ctx.config.effective_aux_model(),
                max_tokens=2000,
                thinking_override="disabled",
            )
            summary_text = resp.content.strip() or "(empty summary)"
            state.input_tokens += resp.input_tokens
            state.output_tokens += resp.output_tokens
        except Exception as exc:
            summary_text = f"(Compaction fallback due to {type(exc).__name__}; see workspace files: {files_on_disk[:25]})"

        summary_msg = {
            "role": "user",
            "content": (
                "<compacted_history>\n"
                f"{summary_text}\n\n"
                f"Current workspace files (sample): {files_on_disk[:30]}\n"
                "</compacted_history>"
            ),
        }
        state.messages = [head_task, summary_msg, *tail]
        ctx.emit_event(
            "context_compacted",
            prev_tokens=est_tokens,
            summarized_msgs=len(to_summarize),
            kept_msgs=len(state.messages),
        )
        return [messages[0], *state.messages]
