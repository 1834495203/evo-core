"""Read-only Companion Q&A Agent grounded strictly in durable snapshots, records, and workspace files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..observability.events import ResearchEvents
from ..storage.artifacts import ArtifactStore
from .queries import ResearchQueries

QA_SYSTEM_PROMPT = """You are the Read-Only Companion Q&A Agent for an autonomous research run.
Your role is to answer the user's questions about current research progress, active hypotheses,
experiments, failures, literature sources, results, and next plans WITHOUT interrupting or altering
the main research agent.

Rules:
1. Rely ONLY on the provided research snapshot and the read-only tools (`get_experiment_detail`,
   `list_notes_by_kind`, `read_workspace_file`). Never invent numbers, papers, or plans.
2. Always mention the snapshot timestamp (`as_of`) and cite concrete IDs (`exp_*`, `note_*`, `cmd_*`).
3. Clearly distinguish `verified` vs `unverified` vs `invalid` experiments and completed vs running jobs.
4. Keep answers clear, direct, and concise in the user's language.
"""


class CompanionQAAgent:
    """Answers user questions from consistent snapshots without touching main execution or LLM history."""

    def __init__(
        self,
        queries: ResearchQueries,
        events: ResearchEvents,
        artifacts: ArtifactStore | None = None,
        workspace: Path | None = None,
        llm_client: Any = None,
        model: str | None = None,
    ) -> None:
        self.queries = queries
        self.events = events
        self.artifacts = artifacts
        self.workspace = Path(workspace).resolve() if workspace else None
        self.llm_client = llm_client
        self.model = model

    @staticmethod
    def _build_structured_summary(
        progress: dict[str, Any],
        snapshot: dict[str, Any],
        recent_events: list[dict[str, Any]],
    ) -> str:
        study = progress["study"]
        exps = progress["experiments"]
        by_status: dict[str, int] = {}
        by_validity: dict[str, int] = {}
        for exp in exps:
            st = str(exp.get("status", "unknown"))
            val = str(exp.get("validity", "unverified"))
            by_status[st] = by_status.get(st, 0) + 1
            by_validity[val] = by_validity.get(val, 0) + 1

        lines = [
            f"Study ID: {study['id']} (as_of={progress['as_of']}, event_cursor={progress['event_cursor']})",
            f"Direction: {study['direction']}",
            f"Status: {study['status']} | Phase: {study['phase']} | Plan v{study['plan_version']}",
            f"Acceptance: {study['acceptance']} | Scientific Outcome: {study['scientific_outcome']}",
            f"Next Planned Action: {study['next_action']}",
            f"Experiments ({len(exps)}/{study['max_experiments']}): by_status={by_status}, by_validity={by_validity}",
        ]
        if exps:
            lines.append("Recent Experiments:")
            for exp in exps[-6:]:
                metrics = (exp.get("result") or {}).get("metrics") or {}
                err = (exp.get("result") or {}).get("error") or ""
                lines.append(
                    f"  - {exp['id']} [{exp['status']} / {exp.get('validity', 'unverified')}] "
                    f"{exp['purpose']} | metrics={metrics}" + (f" | err={err}" if err else "")
                )

        notes = snapshot.get("records") or []
        if notes:
            lines.append(f"Recent Research Notes (total={len(notes)}):")
            for note in notes[-6:]:
                lines.append(
                    f"  - {note['id']} ({note['kind']}, refs={note.get('refs', [])}): {note['text'][:180]}"
                )

        cmds = progress.get("commands") or []
        if cmds:
            lines.append("User Commands:")
            for cmd in cmds[-5:]:
                lines.append(
                    f"  - {cmd['id']} [{cmd['kind']} -> {cmd['status']}]: {cmd.get('text', '')[:140]}"
                )

        if recent_events:
            lines.append(f"Latest Event: #{recent_events[-1]['sequence']} {recent_events[-1]['kind']} ({recent_events[-1]['actor']})")
        return "\n".join(lines)

    def _build_read_tools(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        jobs_by_id = {j["id"]: j for j in snapshot.get("jobs", [])}

        def get_experiment_detail(job_id: str) -> str:
            job = jobs_by_id.get(job_id)
            if not job:
                return f"Unknown experiment ID: {job_id}"
            payload = dict(job)
            res = job.get("result") or {}
            if self.artifacts and res.get("artifacts", {}).get("stderr.log"):
                try:
                    stderr = self.artifacts.read(res["artifacts"]["stderr.log"]).decode("utf-8", errors="replace")
                    payload["stderr_tail"] = stderr[-2000:]
                except Exception:
                    pass
            return json.dumps(payload, ensure_ascii=False, indent=2)[:12_000]

        def list_notes_by_kind(kind: str = "") -> str:
            notes = snapshot.get("records", [])
            if kind:
                notes = [n for n in notes if n.get("kind") == kind]
            return json.dumps(notes[-25:], ensure_ascii=False, indent=2)[:12_000]

        def read_workspace_file(file_path: str, max_chars: int = 8000) -> str:
            if self.workspace is None:
                return "Workspace path not configured for this Q&A session."
            rel = file_path.strip().lstrip("/")
            try:
                target = (self.workspace / rel).resolve()
                target.relative_to(self.workspace)
            except Exception:
                return f"Error: invalid path outside workspace: {file_path}"
            if not target.is_file():
                return f"File not found: {file_path}"
            cap = max(500, min(24_000, int(max_chars)))
            return target.read_text(encoding="utf-8", errors="replace")[:cap]

        return {
            "get_experiment_detail": (
                {
                    "type": "function",
                    "function": {
                        "name": "get_experiment_detail",
                        "description": "Inspect full spec, checks, validity, metrics, and error details of an experiment.",
                        "parameters": {
                            "type": "object",
                            "properties": {"job_id": {"type": "string"}},
                            "required": ["job_id"],
                        },
                    },
                },
                get_experiment_detail,
            ),
            "list_notes_by_kind": (
                {
                    "type": "function",
                    "function": {
                        "name": "list_notes_by_kind",
                        "description": "List recorded notes optionally filtered by kind (source, observation, hypothesis, method, decision, limitation).",
                        "parameters": {
                            "type": "object",
                            "properties": {"kind": {"type": "string"}},
                        },
                    },
                },
                list_notes_by_kind,
            ),
            "read_workspace_file": (
                {
                    "type": "function",
                    "function": {
                        "name": "read_workspace_file",
                        "description": "Read a file inside the study workspace (e.g., final_report.md, literature_notes.md).",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "file_path": {"type": "string"},
                                "max_chars": {"type": "integer"},
                            },
                            "required": ["file_path"],
                        },
                    },
                },
                read_workspace_file,
            ),
        }

    def answer(self, study_id: str, question: str, *, use_llm: bool = True) -> dict[str, Any]:
        from contextlib import nullcontext
        scope = self.llm_client.scope("qa") if self.llm_client is not None and hasattr(self.llm_client, "scope") else nullcontext()
        with scope:
            return self._answer(study_id, question, use_llm=use_llm)

    def _answer(self, study_id: str, question: str, *, use_llm: bool = True) -> dict[str, Any]:
        if not question or not question.strip():
            raise ValueError("Question text cannot be empty")

        snapshot = self.queries.snapshot(study_id)
        progress = self.queries.progress(study_id)
        cursor = int(progress.get("event_cursor") or 0)
        recent_events = self.events.read(study_id, after=max(0, cursor - 25), limit=25)["events"]
        structured_text = self._build_structured_summary(progress, snapshot, recent_events)

        if not use_llm or self.llm_client is None:
            return {
                "study_id": study_id,
                "as_of": progress["as_of"],
                "event_cursor": cursor,
                "mode": "structured",
                "answer": structured_text,
                "progress": progress,
                "recent_events": recent_events,
            }

        ro_tools = self._build_read_tools(snapshot)
        schemas = [entry[0] for entry in ro_tools.values()]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": QA_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Consistent Research Snapshot Summary:\n{structured_text}\n\n"
                    f"User Question: {question}"
                ),
            },
        ]
        answer_text = structured_text
        for _ in range(4):
            resp = self.llm_client.complete(
                messages,
                tools=schemas,
                model=self.model,
                max_tokens=1600,
                thinking_override="disabled",
            )
            messages.append(resp.raw_assistant_message)
            if not resp.tool_calls:
                answer_text = resp.content.strip() or structured_text
                break
            for tc in resp.tool_calls:
                if tc.parse_error or tc.name not in ro_tools:
                    out = f"Error: {tc.parse_error or 'unknown read-only tool'}"
                else:
                    try:
                        out = str(ro_tools[tc.name][1](**tc.arguments))
                    except Exception as exc:
                        out = f"Error: {exc}"
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": out}
                )

        return {
            "study_id": study_id,
            "as_of": progress["as_of"],
            "event_cursor": cursor,
            "mode": "llm",
            "answer": answer_text,
        }
