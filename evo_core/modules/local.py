"""Simple functional modules. Replace each factory independently without changing the kernel."""
from contextlib import contextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from uuid import uuid4

from ..llm import approximate_tokens, repair_messages
from ..system.domain import now
from ..system.storage.artifacts import ArtifactStore


def _clean_rel(path: str) -> str:
    return str(path).strip().lstrip("/").replace("\\", "/")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid4().hex[:8]}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _safe_split_index(messages: list[dict], keep_recent: int) -> int:
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


class SQLiteDocuments:
    def __init__(self, path, *, mirror_dir: Path | None = None, prompt_index: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.mirror_dir = Path(mirror_dir) if mirror_dir is not None else None
        self.prompt_index = prompt_index
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS docs(id TEXT PRIMARY KEY,kind TEXT,body TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS history(id TEXT,body TEXT)")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _write_markdown_mirror(self, record: dict) -> None:
        if self.mirror_dir is None:
            return
        details = record.get("details") or {}
        refs = details.get("refs") or []
        header = [
            "---",
            f"id: {json.dumps(record['id'])}",
            f"kind: {json.dumps(record['kind'], ensure_ascii=False)}",
            f"memory_type: {json.dumps(details.get('memory_type', 'semantic'), ensure_ascii=False)}",
            f"scope: {json.dumps(details.get('scope', 'project'), ensure_ascii=False)}",
            f"summary: {json.dumps(details.get('summary', ''), ensure_ascii=False)}",
            f"refs: {json.dumps(list(refs), ensure_ascii=False)}",
            f"updated_at: {json.dumps(record['updated_at'])}",
            "---",
            "",
            record["text"],
            "",
        ]
        _atomic_write_text(self.mirror_dir / f"{record['id']}.md", "\n".join(header))

    def put(self, kind, text, *, record_id="", **details):
        if not kind.strip() or not text.strip():
            raise ValueError("Document kind and text are required")
        identifier = record_id or f"doc_{uuid4().hex}"
        enriched = dict(details)
        if self.mirror_dir is not None or self.prompt_index:
            summary = str(enriched.get("summary") or "").strip()
            if not summary:
                first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), text.strip())
                summary = first_line[:160]
            enriched.setdefault("summary", summary)
            enriched.setdefault("memory_type", "semantic")
            enriched.setdefault("scope", "project")
            enriched["refs"] = list(enriched.get("refs") or ())
        record = {"id": identifier, "kind": kind, "text": text, "details": enriched, "updated_at": now()}
        raw = json.dumps(record, ensure_ascii=False)
        with self.connect() as db:
            db.execute("INSERT INTO history VALUES(?,?)", (identifier, raw))
            db.execute("INSERT OR REPLACE INTO docs VALUES(?,?,?)", (identifier, kind, raw))
        self._write_markdown_mirror(record)
        return record

    def get(self, record_id):
        with self.connect() as db:
            row = db.execute("SELECT body FROM docs WHERE id=?", (record_id,)).fetchone()
        if not row:
            raise ValueError("Unknown document")
        return json.loads(row[0])

    def search(self, query="", kind=""):
        with self.connect() as db:
            rows = db.execute("SELECT body FROM docs WHERE (?='' OR kind=?) ORDER BY rowid DESC", (kind, kind)).fetchall()
        return [json.loads(row[0]) for row in rows if query.casefold() in row[0].casefold()]

    def index(self, limit: int = 15) -> list[dict]:
        cap = max(1, min(100, int(limit)))
        with self.connect() as db:
            rows = db.execute("SELECT body FROM docs ORDER BY rowid DESC LIMIT ?", (cap,)).fetchall()
        items: list[dict] = []
        for row in rows:
            rec = json.loads(row[0])
            details = rec.get("details") or {}
            summary = str(details.get("summary") or "").strip()
            if not summary:
                summary = next((ln.strip() for ln in rec["text"].splitlines() if ln.strip()), rec["text"])[:140]
            items.append({
                "id": rec["id"],
                "kind": rec["kind"],
                "summary": summary,
                "refs": list(details.get("refs") or []),
                "updated_at": rec["updated_at"],
            })
        return items

    def system_prompt_section(self, state, ctx) -> str | None:
        if not self.prompt_index:
            return None
        items = self.index(limit=12)
        if not items:
            return None
        lines = ["# Long-Term Project Notes Index"]
        lines.append("Recent project notes exist (inspect full content with `search_long_term_notes` / `read_long_term_note`):")
        for item in items:
            refs_txt = f" refs={item['refs']}" if item["refs"] else ""
            lines.append(f"- {item['id']} [{item['kind']}]: {item['summary']}{refs_txt}")
        return "\n".join(lines)


class LocalLiterature:
    def __init__(self, root, provider):
        self.root = Path(root)
        self.documents = SQLiteDocuments(self.root / "library.sqlite3")
        self.searches = SQLiteDocuments(self.root / "search_history.sqlite3")
        self.artifacts = ArtifactStore(self.root / "papers")
        self.provider = provider

    def _write_card(self, record: dict) -> None:
        details = record.get("details") or {}
        tags = list(details.get("tags") or [])
        lines = [
            f"# {record['text']}",
            "",
            f"- **ID**: `{record['id']}`",
            f"- **URL**: {details.get('url', '')}",
        ]
        if details.get("authors"):
            lines.append(f"- **Authors**: {details['authors']}")
        if details.get("year"):
            lines.append(f"- **Year**: {details['year']}")
        if details.get("venue"):
            lines.append(f"- **Venue**: {details['venue']}")
        if details.get("arxiv_id"):
            lines.append(f"- **arXiv**: {details['arxiv_id']}")
        if details.get("doi"):
            lines.append(f"- **DOI**: {details['doi']}")
        if tags:
            lines.append(f"- **Tags**: {', '.join(str(t) for t in tags)}")
        lines.extend([
            "",
            "## Abstract",
            str(details.get("abstract") or "(none)"),
            "",
            "## Researcher Notes",
            str(details.get("notes") or "(none)"),
            "",
        ])
        _atomic_write_text(self.root / "cards" / f"{record['id']}.md", "\n".join(lines))

    def search(self, query, limit=5):
        search_id = "search_" + uuid4().hex
        try:
            candidates = [{**row, "search_id": search_id, "candidate_id": f"{search_id}:{index}"}
                          for index, row in enumerate(self.provider.search(query, limit))]
        except Exception as exc:
            self.searches.put("search", query, record_id=search_id, candidates=[], status="failed", error=str(exc))
            raise
        self.searches.put("search", query, record_id=search_id, candidates=candidates,
                          status="completed" if candidates else "no_results")
        return candidates  # Saved search evidence is distinct from explicit library admission.

    def search_history(self, query="", limit=20):
        return [{"id": r["id"], "query": r["text"], "count": len(r["details"]["candidates"]),
                 "status": r["details"]["status"], "updated_at": r["updated_at"]}
                for r in self.searches.search(query)[:max(1, min(100, int(limit)))]]

    def read_search(self, search_id):
        return self.searches.get(search_id)

    def candidate(self, candidate_id):
        search_id, index = candidate_id.rsplit(":", 1)
        result = self.read_search(search_id)["details"]["candidates"][int(index)]
        if result["candidate_id"] != candidate_id:
            raise ValueError("Candidate does not belong to this search")
        return result

    def add(self, title, abstract, notes, url, **metadata):
        if not url or not title:
            raise ValueError("A paper needs a title and source identifier")
        meta = dict(metadata)
        if meta.get("candidate_id"):
            existing = [p for p in self.documents.search("", "paper") if p["details"].get("candidate_id") == meta["candidate_id"]]
            if existing: return existing[0]
        if meta.get("tags") is not None:
            meta["tags"] = [str(t).strip() for t in meta["tags"] if str(t).strip()]
        record = self.documents.put("paper", title, abstract=abstract, notes=notes, url=url, **meta)
        self._write_card(record)
        return record

    def library(self, query="", tag="", limit=50):
        rows = self.documents.search(query, "paper")
        tag_cf = str(tag or "").strip().casefold()
        out = []
        for r in rows:
            details = dict(r.get("details") or {})
            row_tags = [str(t).casefold() for t in (details.get("tags") or [])]
            if tag_cf and tag_cf not in row_tags:
                continue
            has_fulltext = bool(details.get("content_hash"))
            details.pop("content_hash", None)
            out.append({**r, "details": {**details, "has_fulltext": has_fulltext}})
            if len(out) >= max(1, min(200, int(limit))):
                break
        return out

    def annotate(self, paper_id, notes, tags=None):
        old = self.documents.get(paper_id)
        updated_details = {**old["details"], "notes": notes}
        if tags is not None:
            updated_details["tags"] = [str(t).strip() for t in tags if str(t).strip()]
        rec = self.documents.put("paper", old["text"], record_id=paper_id, **updated_details)
        self._write_card(rec)
        return rec

    def read(self, paper_id, offset=0, limit=16000, focus_query=""):
        paper = self.documents.get(paper_id)
        details = paper["details"]
        if not details.get("content_hash"):
            result = self.provider.read(details["url"])
            content = result.get("content") or result["excerpt"]
            details = {**details, "content_hash": self.artifacts.put(content.encode()),
                       "reading_depth": result.get("reading_depth", "unknown"), "locator": result.get("locator", "")}
            self.documents.put("paper", paper["text"], record_id=paper_id, **details)
        content = self.artifacts.read(details["content_hash"]).decode()
        cap = max(1, min(32000, limit))
        if focus_query and str(focus_query).strip():
            from ..plugins.literature import _select_focused_excerpt
            excerpt, focus_loc = _select_focused_excerpt(content, str(focus_query), cap)
            locator = f"{details.get('locator') or ''} | {focus_loc}".strip(" |")
            return {"paper_id": paper_id, "text": excerpt, "total_chars": len(content),
                    "next_offset": None, "locator": locator}
        start = max(0, offset)
        return {"paper_id": paper_id, "text": content[start:start+cap], "total_chars": len(content),
                "next_offset": start+cap if start+cap < len(content) else None, "locator": details.get("locator")}


class LocalCoding:
    def system_prompt_section(self, state, ctx):
        if not self.require_review:
            return None
        return (
            "# Coding handoff\nUse create_coding_task and implement_coding_task for implementation. "
            "Researcher raw shell execution and code submission are not granted in strict mode. "
            "Inspect the frozen submission, use test_code_submission for unit tests, and review it. "
            "Formal execution and analysis scripts run through submit_experiment with the approved submission."
        )

    def __init__(self, root, require_review=False, runner=None):
        self.root = Path(root)
        self.docs = SQLiteDocuments(self.root / "coding.sqlite3")
        self.artifacts = ArtifactStore(self.root / "code")
        self.require_review = require_review
        self.runner = runner

    def task(self, text):
        return self.docs.put("coding_task", text)

    def implement(self, task_id: str, instruction: str = "", *, ctx=None) -> dict:
        task_record = self.docs.get(task_id)
        if task_record["kind"] != "coding_task":
            raise ValueError("Expected coding task")
        if self.runner is not None:
            return self.runner(task_id, task_record, instruction, ctx)
        if ctx is None:
            raise ValueError("Coding implementation requires an active HookContext or custom runner")
        from ..plugins.subagents import SubagentsPlugin
        prompt_parts = [f"Coding task_id: {task_id}", f"Specification:\n{task_record['text']}"]
        if instruction and instruction.strip():
            prompt_parts.append(f"Additional instructions:\n{instruction.strip()}")
        prompt_parts.append(
            "After writing/editing files, call `submit_code(task_id=..., files=[...], run={...})` "
            "and report the submission_id and proposed verification test command."
        )
        output = SubagentsPlugin().run_role("coder", "\n\n".join(prompt_parts), ctx)
        submissions = [
            s["id"] for s in self.docs.search("", "code_submission")
            if (s.get("details") or {}).get("task_id") == task_id
        ]
        return {"task_id": task_id, "output": output, "submissions": submissions}

    def submit(self, task_id, source_dir, files, run, *, actor: str = "coder", test_paths=None):
        if self.docs.get(task_id)["kind"] != "coding_task":
            raise ValueError("Expected coding task")
        clean_files: list[str] = []
        for f in files:
            cf = _clean_rel(f)
            if cf and cf not in clean_files:
                clean_files.append(cf)
        normalized_run = dict(run or {})
        for key in ("entrypoint", "check_entrypoint"):
            if normalized_run.get(key):
                cf = _clean_rel(normalized_run[key])
                normalized_run[key] = cf
                if cf and cf not in clean_files:
                    clean_files.append(cf)
        if normalized_run.get("metrics_path"):
            normalized_run["metrics_path"] = _clean_rel(normalized_run["metrics_path"])
        return self.docs.put(
            "code_submission",
            "Code submitted for researcher review",
            task_id=task_id,
            submitted_by=actor,
            files=clean_files,
            manifest=self.artifacts.freeze(Path(source_dir), tuple(clean_files)),
            run=normalized_run,
            approved=False,
            unit_test_paths=test_paths,
        )

    def review(self, submission_id, approved, notes, *, actor: str = "researcher"):
        if actor in {"coder", "subagent:coder", "subagent:code"}:
            raise ValueError("Coder cannot approve or review code submissions")
        old = self.docs.get(submission_id)
        if old["kind"] != "code_submission":
            raise ValueError("Expected code submission")
        return self.docs.put(old["kind"], old["text"], record_id=submission_id,
                             **{**old["details"], "approved": bool(approved), "review_notes": notes, "reviewed_by": actor})

    def inspect(self, submission_id, file_path=""):
        record = self.docs.get(submission_id)
        if record["kind"] != "code_submission":
            raise ValueError("Expected code submission")
        if not file_path:
            return record
        clean = _clean_rel(file_path)
        key = record["details"]["manifest"].get(clean)
        if not key:
            raise ValueError("File is not part of this submission")
        return {"submission_id":submission_id, "file_path":clean,
                "content":self.artifacts.read(key).decode("utf-8"), "hash":key}

    def authorize(self, submission_id, spec):
        if not submission_id:
            raise ValueError("Researcher approval required: pass code_submission_id from an approved submission")
        record = self.docs.get(submission_id)["details"]
        if not record.get("approved"):
            raise ValueError("Researcher approval required for this code version")
        actual = self.artifacts.freeze(Path(spec.source_dir), tuple(spec.files))
        if actual != record["manifest"]:
            raise ValueError("Code changed after submission; submit and review the new version")
        run = asdict(spec)
        run.pop("source_dir")
        run.pop("files")
        actual_run = json.loads(json.dumps(run))
        expected_run = {"check_runner": "auto", **record["run"]}
        if actual_run != expected_run:
            diffs = {
                k: {"approved": expected_run.get(k), "requested": actual_run.get(k)}
                for k in sorted(set(expected_run) | set(actual_run))
                if expected_run.get(k) != actual_run.get(k)
            }
            raise ValueError(f"Run specification differs from the approved submission: {diffs}")
        return record["manifest"]

    def test_submission(self, submission_id, *, timeout=60, python_executable=None, test_paths=None):
        """Persist a pytest receipt bound to this submission's complete frozen manifest."""
        import sys
        from ..system.experiments.validation import run_pytest
        submission = self.inspect(submission_id)
        record = submission["details"]
        selected = test_paths if test_paths is not None else record.get("unit_test_paths")
        if selected is None:
            selected = [name for name in record["manifest"] if Path(name).name.startswith("test_") or Path(name).name.endswith("_test.py")]
            result_check = record["run"].get("check_entrypoint")
            if len(selected) > 1 and result_check in selected:
                selected.remove(result_check)
        if not selected:
            raise ValueError("No unit test paths in submission; declare test_paths containing frozen pytest test files")
        if any(name not in record["manifest"] or not name.endswith(".py") for name in selected):
            raise ValueError("test_paths must name Python files included in the frozen submission")
        target = self.root / "test_runs" / uuid4().hex[:12]
        self.artifacts.materialize(record["manifest"], target)
        receipt = run_pytest(python_executable or sys.executable, target, targets=selected, timeout=timeout)
        output = receipt.pop("stdout") + receipt.pop("stderr")
        receipt.update(submission_id=submission_id, manifest=record["manifest"],
                       source_manifest=record["manifest"], scope="submission", output=output[-24000:],
                       test_paths=list(selected),
                       log_hash=self.artifacts.put(output.encode()))
        report = target / "outputs/pytest.xml"
        if report.is_file(): receipt["report_hash"] = self.artifacts.put(report.read_bytes())
        self.docs.put(submission["kind"], submission["text"], record_id=submission_id,
                      **{**record, "validation": receipt})
        return receipt


class LocalContexts:
    def __init__(self, root):
        self.root = Path(root)
        self.docs = SQLiteDocuments(self.root / "sessions.sqlite3")

    def load(self, session, role):
        try:
            return self.docs.get(f"{session}:{role}")["details"]["messages"]
        except ValueError:
            return []

    def save(self, session, role, messages):
        self.docs.put("context", role, record_id=f"{session}:{role}", messages=messages)

    def offload_tool_output(self, session: str, role: str, tool_name: str, content: str, max_chars: int = 12000) -> str:
        if len(content) <= max_chars:
            return content
        offload_dir = self.root / "tool_logs"
        offload_dir.mkdir(parents=True, exist_ok=True)
        path = offload_dir / f"{session}_{role}_{tool_name}_{uuid4().hex[:8]}.txt"
        path.write_text(content, encoding="utf-8")
        half = max(500, max_chars // 3)
        return (
            f"{content[:half]}\n\n"
            f"... [offloaded {len(content)} chars to {path}] ...\n\n"
            f"{content[-half:]}"
        )

    def _append_conversation_history(self, session: str, role: str, evicted: list[dict]) -> Path:
        lines = [f"## Summarized at {now()}", ""]
        for msg in evicted:
            mrole = msg.get("role", "unknown")
            if mrole == "assistant" and msg.get("tool_calls"):
                lines.append('<message type="ai">')
                if msg.get("content"):
                    lines.append(f"  <content>{msg['content']}</content>")
                for tc in msg.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    lines.append(f'  <tool_call id="{tc.get("id", "")}" name="{fn.get("name", "")}">{fn.get("arguments", "")}</tool_call>')
                lines.append("</message>")
            elif mrole == "tool":
                lines.append(f'<message type="tool" name="{msg.get("name", "")}">{msg.get("content", "")}</message>')
            else:
                mtype = "human" if mrole == "user" else mrole
                lines.append(f'<message type="{mtype}">{msg.get("content", "")}</message>')
        lines.append("")
        hist_path = self.root / "conversation_history" / f"{session}_{role}.md"
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        with hist_path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        return hist_path

    def edit_and_compact(
        self,
        session: str,
        role: str,
        messages: list[dict],
        *,
        llm_client=None,
        model: str | None = None,
        context_limit_tokens: int = 96000,
        edit_fraction: float = 0.50,
        compact_fraction: float = 0.72,
        keep_recent_tools: int = 5,
        keep_recent_messages: int = 10,
        header_tokens: int = 0,
        workspace_files: list[str] | None = None,
        todos: list[dict] | None = None,
        on_compact=None,
    ) -> list[dict]:
        messages[:] = repair_messages(messages, preserve_reasoning=True)
        limit = max(50, int(context_limit_tokens))

        # Stage 1: EvoScientist-style ClearToolUsesEdit beyond 50% window (keep recent 5 + protected tools).
        if approximate_tokens(messages) + int(header_tokens) >= int(limit * edit_fraction):
            tool_indices = [
                i for i, m in enumerate(messages)
                if m.get("role") == "tool" and m.get("name") not in {"think", "write_long_term_note", "record_research_note"}
            ]
            for idx in tool_indices[:-keep_recent_tools] if keep_recent_tools > 0 else tool_indices:
                text = str(messages[idx].get("content") or "")
                if len(text) > 600 and not text.startswith("[Cleared older "):
                    messages[idx] = {
                        **messages[idx],
                        "content": f"[Cleared older {messages[idx].get('name', 'tool')} output ({len(text)} chars) to preserve context window]",
                    }

        # Stage 2: EvoScientist-style offload evicted history to conversation_history/*.md + summarize.
        est_tokens = approximate_tokens(messages) + int(header_tokens)
        has_system = bool(messages and messages[0].get("role") == "system")
        prefix_len = 2 if has_system and len(messages) >= 2 else (1 if messages else 0)
        if est_tokens >= int(limit * compact_fraction) and len(messages) > prefix_len + keep_recent_messages + 1:
            body = messages[prefix_len:]
            split_at = _safe_split_index(body, keep_recent_messages)
            if split_at > 0 and split_at < len(body):
                evicted = body[:split_at]
                tail = body[split_at:]
                hist_path = self._append_conversation_history(session, role, evicted)
                serialized = json.dumps(evicted, ensure_ascii=False, default=str)
                if len(serialized) > 90_000:
                    serialized = serialized[:45_000] + "\n...[middle omitted]...\n" + serialized[-45_000:]
                files_sample = list(workspace_files or [])[:60]
                summary_text = ""
                if llm_client is not None:
                    summary_req = [
                        {
                            "role": "system",
                            "content": (
                                "You are compacting an ongoing autonomous research session.\n"
                                "Summarize the older messages concisely in structured bullet points preserving:\n"
                                "1) Chosen research question, datasets, and hypotheses\n"
                                "2) Files created/modified and exact Python scripts/tests run\n"
                                "3) Exact verified quantitative metrics vs invalidated/buggy runs\n"
                                "4) Remaining unresolved issues and next planned steps\n"
                                "Do NOT invent any numbers or file paths."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Role: {role}\nWorkspace files: {files_sample}\n"
                                f"Current todos: {todos or []}\n\nHistory to compact:\n{serialized}"
                            ),
                        },
                    ]
                    try:
                        from contextlib import nullcontext
                        scope = llm_client.scope(f"compactor:{role}") if hasattr(llm_client, "scope") else nullcontext()
                        with scope:
                            resp = llm_client.complete(summary_req, model=model, max_tokens=2000, thinking_override="disabled")
                        summary_text = (resp.content or "").strip() or "(empty summary)"
                        if callable(on_compact):
                            on_compact(est_tokens, len(evicted), len(tail) + prefix_len + 1, resp)
                    except Exception as exc:
                        summary_text = f"(Compaction fallback due to {type(exc).__name__}; see {hist_path})"
                else:
                    summary_text = f"(Evicted {len(evicted)} messages archived to {hist_path})"
                summary_msg = {
                    "role": "user",
                    "content": (
                        "<compacted_history>\n"
                        f"{summary_text}\n\n"
                        f"Full evicted history archived at: {hist_path}\n"
                        f"Current workspace files (sample): {files_sample[:30]}\n"
                        "</compacted_history>"
                    ),
                }
                messages[:] = [*messages[:prefix_len], summary_msg, *tail]
                self.save(session, role, messages)
        return messages


def literature_factory(ctx, registry, options):
    from ..system.research.literature import PluginLiteratureProvider
    return LocalLiterature(ctx.root / "literature", PluginLiteratureProvider(timeout=options.get("timeout", 25)))


def notebook_factory(ctx, registry, options):
    return SQLiteDocuments(ctx.root / "notebook.sqlite3", mirror_dir=ctx.root / "notebook_md", prompt_index=True)


def coding_factory(ctx, registry, options):
    runner = options.get("runner")
    if isinstance(runner, str):
        from .registry import import_factory
        runner = import_factory(runner)
    return LocalCoding(ctx.root / "coding", options.get("require_review", False), runner=runner)


def experiments_factory(ctx, registry, options):
    services = ctx.services
    if services is None:
        from ..system.experiments.harness import LocalExperimentHarness
        from ..system.interaction.queries import ResearchQueries
        from ..system.observability.tracker import ExperimentTracker
        from ..system.research.controller import ResearchController
        from ..system.research.notebook import ResearchNotebook
        from ..system.storage.sqlite import SqliteResearchStore

        exp_root = ctx.root / "experiments"
        store = SqliteResearchStore(exp_root / "research.sqlite3")
        artifacts = ArtifactStore(exp_root / "artifacts")
        notebook = ResearchNotebook(store, artifacts)
        services = SimpleNamespace(
            root=exp_root,
            store=store,
            artifacts=artifacts,
            notebook=notebook,
            controller=ResearchController(store, notebook),
            harness=LocalExperimentHarness(store, artifacts),
            queries=ResearchQueries(store),
            tracker=ExperimentTracker(exp_root / "tracking"),
        )
    return LocalExperiments(services)


class LocalExperiments:
    """Adapter around the existing freeze/queue and local/Docker workers."""
    def __init__(self, services):
        self.backend = services.harness
        self.services = services

    def submit(self, *args, **kwargs):
        return self.backend.submit(*args, **kwargs)

    def get(self, job_id):
        return self.backend.get(job_id)

    def environment_info(self, *, python_executable=None):
        import subprocess, sys
        probe = (
            "import json,sys,platform,os,importlib.util,importlib.metadata\n"
            "names={'numpy':'numpy','scipy':'scipy','pandas':'pandas','matplotlib':'matplotlib','pytest':'pytest','sklearn':'scikit-learn','torch':'torch'}\n"
            "packages={}\n"
            "for module,dist in names.items():\n"
            " try: version=importlib.metadata.version(dist)\n"
            " except importlib.metadata.PackageNotFoundError: version=None\n"
            " packages[module]={'available':importlib.util.find_spec(module) is not None,'version':version}\n"
            "print(json.dumps({'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'cpu_count':os.cpu_count(),'packages':packages}))\n"
        )
        process = subprocess.run([python_executable or sys.executable, "-B", "-X", "utf8", "-c", probe],
            capture_output=True, text=True, encoding="utf-8", timeout=30, check=True,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        return json.loads(process.stdout)

    def list_jobs(self, study_id: str) -> list[dict]:
        return list(self.services.store.snapshot(study_id)["jobs"])

    def verify(self, job_id: str, check_entrypoint: str, *, source_dir=None, files=(), args=(), timeout_seconds: int | None = 60, python_executable: str | None = None, runner: str | None = None):
        worker = self.worker("checker", python_executable=python_executable)
        return worker.verify_job(
            job_id,
            check_entrypoint,
            source_dir=source_dir,
            files=tuple(files),
            args=tuple(args),
            timeout_seconds=timeout_seconds,
            runner=runner,
        )

    def invalidate(self, job_id: str, reason: str, *, actor: str = "agent"):
        return self.services.store.invalidate_job(job_id, reason, actor=actor)

    def annotate(self, job_id: str, note: str, *, kind: str = "observation", registry=None) -> dict:
        job = self.get(job_id)
        study_id = getattr(job, "study_id", "") or ""
        record: dict | None = None
        if registry is not None and registry.has("notebook"):
            record = registry.get("notebook").put(kind, note, refs=[job_id], study_id=study_id)
        if getattr(self.services, "notebook", None) is not None and study_id:
            try:
                sys_rec = self.services.notebook.record(study_id, kind, note, (job_id,))
                if record is None:
                    record = {"id": sys_rec.id, "kind": sys_rec.kind, "text": sys_rec.text, "details": {"refs": [job_id]}}
            except Exception:
                pass
        return record or {"id": "", "kind": kind, "text": note, "details": {"refs": [job_id]}}

    def worker(self, worker_id, *, python_executable=None):
        from ..system.experiments.worker import LocalPythonWorker
        s = self.services
        return LocalPythonWorker(s.store, s.artifacts, s.root / "executions", worker_id,
                                 python_executable=python_executable)

    def worker_pool(self, max_workers=2, *, executor="local", python_executable=None):
        from ..system.experiments.worker import ParallelWorkerPool
        s = self.services
        kwargs = {}
        if executor == "docker":
            from ..system.experiments.docker_worker import DockerExperimentWorker
            kwargs["worker_factory"] = lambda wid: DockerExperimentWorker(s.store, s.artifacts,
                s.root / "executions", wid, python_executable=python_executable)
        elif executor != "local":
            raise ValueError("Unsupported executor")
        return ParallelWorkerPool(s.store,s.artifacts,s.root / "executions",max_workers=max_workers,
                                  python_executable=python_executable, **kwargs)


def context_factory(ctx, registry, options):
    return LocalContexts(ctx.root / "context")


def tracing_factory(ctx, registry, options):
    if ctx.services is not None:
        return ctx.services.tracker
    from ..system.observability.tracker import ExperimentTracker
    return ExperimentTracker(ctx.root / "tracking", mlflow_tracking_uri=options.get("mlflow_tracking_uri"))
