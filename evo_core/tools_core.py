"""Minimal core toolset required for end-to-end autonomous research."""

from __future__ import annotations

from typing import Any

from .protocol import HookContext, RunState, ToolDef


def build_core_tools(state: RunState, ctx: HookContext) -> list[ToolDef]:
    """Build the minimal workspace, shell, background, planning, and reflection tools."""
    sandbox = ctx.sandbox
    bg = ctx.bg_manager

    def ls(path: str = ".") -> str:
        target = sandbox.resolve_path(path)
        if not target.exists():
            return f"Error: path does not exist: {path}"
        if target.is_file():
            return f"{sandbox.rel_path(target)} ({target.stat().st_size} bytes)"
        entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        if not entries:
            return f"(empty directory: {sandbox.rel_path(target)})"
        lines: list[str] = []
        for entry in entries[:200]:
            if entry.name in {"__pycache__", ".git"}:
                continue
            if entry.is_dir():
                lines.append(f"{entry.name}/")
            else:
                lines.append(f"{entry.name}  ({entry.stat().st_size} bytes)")
        if len(entries) > 200:
            lines.append(f"... ({len(entries) - 200} more entries)")
        return "\n".join(lines)

    def read_file(file_path: str, offset: int = 0, limit: int = 250) -> str:
        target = sandbox.resolve_path(file_path)
        if not target.is_file():
            return f"Error: file not found: {file_path}"
        raw = target.read_text(encoding="utf-8", errors="replace").splitlines()
        total = len(raw)
        start = max(0, int(offset))
        count = max(1, min(600, int(limit)))
        sliced = raw[start : start + count]
        body = "\n".join(f"{start + i + 1}: {line}" for i, line in enumerate(sliced))
        if len(body) > 28_000:
            body = body[:28_000] + "\n... (output truncated by byte cap)"
        header = f"[{sandbox.rel_path(target)}: lines {start + 1}-{start + len(sliced)} of {total}]"
        return f"{header}\n{body}"

    def write_file(file_path: str, content: str) -> str:
        target = sandbox.resolve_path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        rel = sandbox.rel_path(target)
        note_id = ctx.observe_file(rel, content) if ctx.observe_file else None
        return f"Wrote {len(content)} chars ({len(content.splitlines())} lines) to {rel}" + (f"; archived note {note_id}" if note_id else "")

    def edit_file(
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        target = sandbox.resolve_path(file_path)
        if not target.is_file():
            return f"Error: file not found: {file_path}"
        text = target.read_text(encoding="utf-8", errors="replace")
        count = text.count(old_string)
        if count == 0:
            return f"Error: old_string not found in {file_path}"
        if count > 1 and not replace_all:
            return (
                f"Error: old_string matched {count} times in {file_path}. "
                "Provide more context or pass replace_all=true."
            )
        updated = (
            text.replace(old_string, new_string)
            if replace_all
            else text.replace(old_string, new_string, 1)
        )
        target.write_text(updated, encoding="utf-8")
        note_id = ctx.observe_file(sandbox.rel_path(target), updated) if ctx.observe_file else None
        return f"Updated {sandbox.rel_path(target)} ({count if replace_all else 1} replacement(s))." + (f" Archived note {note_id}" if note_id else "")

    def execute(command: str, timeout: int | None = None, purpose: str | None = None,
                experiment_ids: list[str] | None = None) -> str:
        _rc, output = sandbox.execute(command, timeout=timeout)
        return output

    def run_in_background(command: str, name: str | None = None) -> str:
        _pid, msg = bg.launch(command, name=name)
        return msg

    def check_process(process_id: str) -> str:
        return bg.check(process_id)

    def stop_process(process_id: str) -> str:
        return bg.stop(process_id)

    def list_processes() -> str:
        rows = bg.list_all()
        if not rows:
            return "No background processes launched in this run."
        return "\n".join(
            f"- {r['process_id']} ({r['name']}): {r['status']} [{r['elapsed_s']}s] log={r['log']}"
            for r in rows
        )

    def write_todos(todos: list[dict[str, Any]]) -> str:
        cleaned: list[dict[str, str]] = []
        for idx, item in enumerate(todos, start=1):
            if not isinstance(item, dict):
                continue
            cleaned.append(
                {
                    "id": str(item.get("id") or idx),
                    "content": str(item.get("content") or ""),
                    "status": str(item.get("status") or "pending"),
                }
            )
        state.todos = cleaned
        summary = "\n".join(f"- [{t['status']}] ({t['id']}) {t['content']}" for t in cleaned)
        return f"Updated {len(cleaned)} todo(s):\n{summary}" if summary else "Cleared todo list."

    def think(reflection: str) -> str:
        return "Reflection recorded."

    tools = [
        ToolDef(
            name="ls",
            description="List files and subdirectories inside the workspace.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative directory path (default '.')."},
                },
            },
            handler=ls,
        ),
        ToolDef(
            name="read_file",
            description="Read a text file in the workspace with line numbers and optional pagination.",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Workspace-relative file path."},
                    "offset": {"type": "integer", "description": "0-based start line (default 0)."},
                    "limit": {"type": "integer", "description": "Max lines to read (default 250)."},
                },
                "required": ["file_path"],
            },
            handler=read_file,
        ),
        ToolDef(
            name="write_file",
            description="Create or overwrite a file in the workspace.",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Workspace-relative file path."},
                    "content": {"type": "string", "description": "Full text content to write."},
                },
                "required": ["file_path", "content"],
            },
            handler=write_file,
        ),
        ToolDef(
            name="edit_file",
            description="Edit an existing file in the workspace using exact string replacement.",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Workspace-relative file path."},
                    "old_string": {"type": "string", "description": "Exact existing text to replace."},
                    "new_string": {"type": "string", "description": "Replacement text."},
                    "replace_all": {"type": "boolean", "description": "Replace all occurrences (default false)."},
                },
                "required": ["file_path", "old_string", "new_string"],
            },
            handler=edit_file,
        ),
        ToolDef(
            name="execute",
            description=(
                "Run a shell command in the workspace root (cwd='.'). "
                "When harness is enabled, declare purpose=setup/test/debug/analysis. "
                "Formal measurements, calibration and sweeps MUST use submit_experiment. "
                "Analysis must cite experiment_ids from the ledger; debug is limited to 30s."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute."},
                    "timeout": {"type": "integer", "description": "Optional timeout in seconds (1-3600)."},
                    "purpose": {"type": "string", "enum": ["setup", "test", "debug", "analysis", "experiment"]},
                    "experiment_ids": {"type": "array", "items": {"type": "string"}, "description": "Completed ledger experiments used by analysis."},
                },
                "required": ["command"],
            },
            handler=execute,
        ),
        ToolDef(
            name="run_in_background",
            description="Start a long-running experiment in the background and return a process_id immediately.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run detached."},
                    "name": {"type": "string", "description": "Optional short label for this run."},
                },
                "required": ["command"],
            },
            handler=run_in_background,
        ),
        ToolDef(
            name="check_process",
            description="Check status and recent log output of a background process.",
            parameters={
                "type": "object",
                "properties": {
                    "process_id": {"type": "string", "description": "ID returned by run_in_background."},
                },
                "required": ["process_id"],
            },
            handler=check_process,
        ),
        ToolDef(
            name="stop_process",
            description="Terminate a running background process started via run_in_background.",
            parameters={
                "type": "object",
                "properties": {
                    "process_id": {"type": "string", "description": "ID returned by run_in_background."},
                },
                "required": ["process_id"],
            },
            handler=stop_process,
        ),
        ToolDef(
            name="list_processes",
            description="List all background processes launched in this run and their current status.",
            parameters={"type": "object", "properties": {}},
            handler=list_processes,
        ),
        ToolDef(
            name="write_todos",
            description="Maintain a structured list of research stages and their status (pending/in_progress/completed).",
            parameters={
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "content": {"type": "string"},
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                },
                            },
                            "required": ["content", "status"],
                        },
                    }
                },
                "required": ["todos"],
            },
            handler=write_todos,
        ),
        ToolDef(
            name="think",
            description="Record brief private reasoning before major hypotheses, experimental pivots, or sanity checks.",
            parameters={
                "type": "object",
                "properties": {
                    "reflection": {"type": "string", "description": "Concise reasoning checkpoint."},
                },
                "required": ["reflection"],
            },
            handler=think,
        ),
    ]
    if ctx.research_system is not None:
        tools = [t for t in tools if t.name != "run_in_background"]
        next(t for t in tools if t.name == "execute").parameters["required"].append("purpose")
    return tools
