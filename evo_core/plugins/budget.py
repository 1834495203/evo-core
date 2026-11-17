"""Budget & convergence control plugin (supports both unbounded and bounded runs)."""

from __future__ import annotations

import time
from typing import Any

from ..protocol import BasePlugin, HookContext, RunState


class BudgetPlugin(BasePlugin):
    """Tracks calls/wall-time, emits soft convergence hints when bounded, or runs unbounded."""

    name = "budget"

    def enabled(self, cfg: Any) -> bool:
        return bool(cfg.plugins.budget.enabled)

    def before_run(self, state: RunState, ctx: HookContext) -> None:
        state.metadata.setdefault("budget_warned_soft", False)
        state.metadata.setdefault("budget_warned_hard", False)

    def before_model_call(
        self,
        state: RunState,
        messages: list[dict[str, Any]],
        ctx: HookContext,
    ) -> list[dict[str, Any]]:
        bcfg = ctx.config.plugins.budget
        elapsed = time.time() - state.started_at
        effective_calls = state.model_calls + int(state.metadata.get("subagent_model_calls") or 0)

        if bcfg.max_calls is not None and effective_calls >= bcfg.max_calls:
            state.metadata["budget_stop_requested"] = True
            state.metadata["budget_stop_reason"] = (
                f"Reached max_calls={bcfg.max_calls} (main={state.model_calls}, "
                f"subagent={int(state.metadata.get('subagent_model_calls') or 0)})"
            )
            return messages

        if bcfg.max_wall_seconds is not None and elapsed >= bcfg.max_wall_seconds:
            state.metadata["budget_stop_requested"] = True
            state.metadata["budget_stop_reason"] = f"Reached max_wall_seconds={bcfg.max_wall_seconds}"
            return messages

        if bcfg.max_calls is not None and bcfg.max_calls > 0:
            ratio = effective_calls / bcfg.max_calls
            if ratio >= bcfg.hard_report_ratio and not state.metadata.get("budget_warned_hard"):
                state.metadata["budget_warned_hard"] = True
                remaining = max(0, bcfg.max_calls - effective_calls)
                state.add_notice(
                    f"[Budget Alert] {effective_calls}/{bcfg.max_calls} calls used ({remaining} left). "
                    "Freeze new experiments, generate final figures, and write/verify final_report.md immediately."
                )
            elif ratio >= bcfg.soft_converge_ratio and not state.metadata.get("budget_warned_soft"):
                state.metadata["budget_warned_soft"] = True
                remaining = max(0, bcfg.max_calls - effective_calls)
                state.add_notice(
                    f"[Budget Notice] {effective_calls}/{bcfg.max_calls} calls used ({remaining} left). "
                    "Converge on current hypotheses, complete ablations, and prepare report."
                )
        return messages
