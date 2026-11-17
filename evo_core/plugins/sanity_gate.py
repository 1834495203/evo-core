"""Sanity gate plugin: catches Python SyntaxErrors immediately and enforces pre-sweep verification."""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path
from typing import Any

from ..protocol import BasePlugin, HookContext, RunState, ToolCallDecision, TurnDecision

_IGNORED_PARTS = {".venv", "venv", ".git", "__pycache__", ".bg_processes", ".evo_run", ".research_modules", "archive_invalid"}


def python_manifest(workspace: Path) -> dict[str, str]:
    return {p.relative_to(workspace).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in workspace.rglob("*.py") if not any(part in _IGNORED_PARTS for part in p.parts)}


def check_python_tree(workspace: Path) -> list[str]:
    """Parse every non-ignored .py file in workspace and return syntax error descriptions."""
    errors: list[str] = []
    for path in sorted(workspace.rglob("*.py")):
        if any(part in _IGNORED_PARTS for part in path.parts):
            continue
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            rel = path.relative_to(workspace).as_posix()
            errors.append(f"{rel}:{exc.lineno}: {exc.msg} ({(exc.text or '').strip()})")
    return errors


class SanityGatePlugin(BasePlugin):
    """Immediate AST syntax verification and test-before-sweep tracking."""

    name = "sanity_gate"

    def enabled(self, cfg: Any) -> bool:
        return bool(cfg.plugins.sanity_gate.enabled)

    def before_run(self, state: RunState, ctx: HookContext) -> None:
        state.metadata.setdefault("modified_py_since_test", [])
        state.metadata.setdefault("tests_executed", 0)

    def before_tool_call(
        self,
        state: RunState,
        tool_name: str,
        arguments: dict[str, Any],
        ctx: HookContext,
    ) -> ToolCallDecision:
        scfg = ctx.config.plugins.sanity_gate
        if tool_name not in {"run_in_background", "submit_experiment"}:
            return ToolCallDecision(action="allow")

        submission_id = arguments.get("code_submission_id") if tool_name == "submit_experiment" else None
        registry = ctx.module_registry or (ctx.research_system.modules if ctx.research_system else None)
        if submission_id and registry and registry.has_capability("code.implementation"):
            coding = registry.service("code.implementation")
            try:
                submission = coding.inspect(submission_id)
                manifest = submission["details"]["manifest"]
                if scfg.check_python_syntax:
                    for name in manifest:
                        if name.endswith(".py"):
                            ast.parse(coding.inspect(submission_id, name)["content"], filename=name)
                receipt = submission["details"].get("validation") or {}
                if scfg.enforce_test_before_sweep and not (
                    receipt.get("passed") is True and receipt.get("source_manifest") == manifest
                    and int(receipt.get("tests_run", 0)) > 0
                ):
                    return ToolCallDecision(action="block", source=self.name,
                        feedback=f"Submission {submission_id} has no passing test receipt for its frozen snapshot. "
                                 f"Run test_code_submission(submission_id='{submission_id}'); unrelated workspace files need not be included.")
                return ToolCallDecision(action="allow")
            except Exception as exc:
                return ToolCallDecision(action="block", source=self.name, feedback=f"Cannot validate submission {submission_id}: {exc}")

        if scfg.check_python_syntax:
            errors = check_python_tree(ctx.workspace)
            if errors:
                return ToolCallDecision(
                    action="block",
                    source=self.name,
                    feedback="Cannot launch experiment while workspace has Python SyntaxErrors:\n"
                    + "\n".join(f"- {e}" for e in errors),
                )

        if scfg.enforce_test_before_sweep and state.metadata.get("modified_py_since_test"):
            dirty = state.metadata["modified_py_since_test"]
            return ToolCallDecision(
                action="block",
                source=self.name,
                feedback=(
                    f"Modified Python files {dirty} have not been verified with a test/sanity run since last edit. "
                    "Test the declared code_submission_id with test_code_submission, or use execute(purpose='test') "
                    "only in legacy execution mode."
                ),
            )
        if scfg.enforce_test_before_sweep and ctx.research_system is not None:
            if state.metadata.get("tested_python_manifest") != python_manifest(ctx.workspace):
                return ToolCallDecision(action="block", source=self.name,
                    feedback="Current Python snapshot has not passed tests. Run execute(purpose='test') with pytest/unittest; shell edits also invalidate prior tests.")
        return ToolCallDecision(action="allow")

    def after_tool_call(
        self,
        state: RunState,
        tool_name: str,
        arguments: dict[str, Any],
        result: str,
        ctx: HookContext,
    ) -> None:
        scfg = ctx.config.plugins.sanity_gate
        if result.startswith("[TOOL BLOCKED"):
            return
        if tool_name in {"write_file", "edit_file"} and not result.startswith("Error:"):
            file_path = str(arguments.get("file_path") or "")
            if file_path.endswith(".py"):
                try:
                    target = ctx.sandbox.resolve_path(file_path)
                    if scfg.check_python_syntax and target.is_file():
                        ast.parse(target.read_text(encoding="utf-8", errors="replace"), filename=str(target))
                except SyntaxError as exc:
                    state.add_notice(
                        f"[SanityGate] SyntaxError in {file_path} at line {exc.lineno}: "
                        f"{exc.msg} (`{(exc.text or '').strip()}`). Fix this immediately."
                    )
                except Exception:
                    pass
                dirty = list(state.metadata.get("modified_py_since_test") or [])
                if file_path not in dirty:
                    dirty.append(file_path)
                    state.metadata["modified_py_since_test"] = dirty

        if tool_name == "execute":
            cmd = str(arguments.get("command") or "")
            strict = ctx.research_system is not None
            passed = (arguments.get("purpose") == "test" and
                      any(token in cmd for token in ("pytest", "unittest")) and
                      bool(re.search(r"\b[1-9]\d* passed\b|Ran [1-9]\d* tests?", result)) and
                      not re.search(r"\b[1-9]\d* (failed|errors?)\b|FAILED\s*\(", result)) if strict else any(
                          tok in cmd for tok in ("pytest", "unittest", "test_", "sanity", "check_", "verify"))
            if passed and "[Command succeeded]" in result:
                state.metadata["modified_py_since_test"] = []
                state.metadata["tests_executed"] = int(state.metadata.get("tests_executed") or 0) + 1
                state.metadata["tested_python_manifest"] = python_manifest(ctx.workspace)

        if ctx.research_system is None and tool_name == "verify_experiment" and '"validity": "verified"' in result:
            state.metadata["modified_py_since_test"] = []
            state.metadata["tests_executed"] = int(state.metadata.get("tests_executed") or 0) + 1

    def after_agent_finish(
        self,
        state: RunState,
        final_text: str,
        ctx: HookContext,
    ) -> TurnDecision:
        if not ctx.config.plugins.sanity_gate.check_python_syntax:
            return TurnDecision(action="accept")
        errors = check_python_tree(ctx.workspace)
        if errors:
            return TurnDecision(
                action="revise",
                source=self.name,
                feedback="Python syntax check failed for workspace files:\n" + "\n".join(f"- {e}" for e in errors),
            )
        return TurnDecision(action="accept")
