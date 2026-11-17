"""Tool adapters only. No agent loop or scientific workflow lives here."""
from dataclasses import asdict
import inspect
import json

from ..protocol import ToolDef
from ..system.domain import ExperimentSpec


def _call_supported(fn, /, **kwargs):
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(**kwargs)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(**kwargs)
    return fn(**{k: v for k, v in kwargs.items() if k in sig.parameters})


def module_tools(ctx):
    registry = ctx.module_registry or (ctx.research_system.modules if ctx.research_system else None)
    if registry is None:
        return []
    tools = []
    from .documents import install_document_capture
    install_document_capture(ctx, registry)

    def expose(name, description, handler, properties=None, required=()):
        def call(**kwargs):
            return json.dumps(handler(**kwargs), ensure_ascii=False, default=str)
        tools.append(ToolDef(name, description, {"type": "object", "properties": properties or {},
                                                "required": list(required)}, call))

    string = {"type": "string"}
    strings = {"type": "array", "items": string}
    refs_schema = {"type": "array", "items": {"anyOf": [string, {"type": "object", "properties": {
        "kind": string, "owner": string, "id": string, "version": {"type": ["string", "null"]}},
        "required": ["kind", "owner", "id"], "additionalProperties": False}]}}
    expose("list_modules", "Inspect enabled modules and their replaceable implementations.", registry.describe)

    if registry.has_legacy("literature"):
        lib = registry.get("literature")
        def find_papers(query, limit=5):
            candidates = lib.search(query, limit)
            if ctx.research_system is not None and ctx.study_id:
                content_hash = ctx.research_system.artifacts.put(json.dumps(candidates, ensure_ascii=False).encode())
                ctx.research_system.notebook.record(ctx.study_id, "search", query,
                    details={"candidate_count": len(candidates), "search_id": candidates[0].get("search_id", "") if candidates else "",
                             "content_hash": content_hash, "content_type": "application/json", "admitted": False})
            return candidates
        expose("find_papers", "Search candidates and save search history; admission to the paper library remains your choice.", find_papers,
               {"query": string, "limit": {"type": "integer"}}, ["query"])
        if hasattr(lib, "search_history"):
            expose("list_paper_searches", "Retrieve saved searches without admitting their candidates.", lib.search_history,
                   {"query": string, "limit": {"type": "integer"}})
            expose("read_paper_search", "Read the candidates and original metadata from a saved search.", lib.read_search,
                   {"search_id": string}, ["search_id"])

        def add_paper(title, abstract, notes, url, authors="", year="", venue="", arxiv_id="", doi="", tags=None, candidate_id=""):
            meta = {
                k: v for k, v in {
                    "authors": authors, "year": year, "venue": venue,
                    "arxiv_id": arxiv_id, "doi": doi, "tags": tags, "candidate_id": candidate_id,
                }.items() if v
            }
            rec = _call_supported(lib.add, title=title, abstract=abstract, notes=notes, url=url, **meta)
            if ctx.research_system is not None and ctx.study_id:
                try:
                    excerpt = f"{abstract}\n\nNotes: {notes}".strip() or title
                    if str(url).startswith(("http://", "https://")):
                        ctx.research_system.notebook.source(
                            ctx.study_id, title, url, excerpt[:600], "admitted_card", reading_depth="abstract_only",
                            module_ref={"owner": registry._manifests["literature"].id, "kind": "paper", "id": rec["id"]}
                        )
                    else:
                        study = ctx.research_system.store.get(ctx.study_id)
                        ctx.research_system.notebook.record(
                            ctx.study_id, "source", title,
                            details={"url": url, "excerpt": excerpt[:600], "locator": "admitted_card",
                                     "reading_depth": "abstract_only", "plan_version": study.plan_version},
                        )
                except Exception:
                    pass
            return rec

        if hasattr(lib, "candidate"):
            def admit_paper(candidate_id, notes, tags=None):
                candidate = lib.candidate(candidate_id)
                return add_paper(candidate["title"], candidate.get("abstract") or "", notes, candidate["url"],
                    authors=str(candidate.get("authors") or ""), venue=str(candidate.get("venue") or ""),
                    year=str(candidate.get("year") or str(candidate.get("published") or "")[:4]),
                    arxiv_id=str(candidate.get("arxiv_id") or ""), doi=candidate.get("doi") or "", tags=tags, candidate_id=candidate_id)
            expose("admit_paper", "Explicitly admit one saved search candidate with your own notes; preserves its original metadata.",
                   admit_paper, {"candidate_id": string, "notes": string, "tags": strings}, ["candidate_id", "notes"])

        expose("add_paper", "Explicitly admit a paper to the project library with abstract, metadata and your own notes.", add_paper,
               {"title": string, "abstract": string, "notes": string, "url": string,
                "authors": string, "year": string, "venue": string, "arxiv_id": string,
                "doi": string, "tags": strings}, ["title", "abstract", "notes", "url"])

        def list_papers(query="", tag="", limit=50):
            return _call_supported(lib.library, query=query, tag=tag, limit=limit)

        expose("list_papers", "Search admitted paper cards (title, abstract, notes), without full-text payloads.", list_papers,
               {"query": string, "tag": string, "limit": {"type": "integer"}})

        def read_library_paper(paper_id, offset=0, limit=16000, focus_query=""):
            out = _call_supported(lib.read, paper_id=paper_id, offset=offset, limit=limit, focus_query=focus_query)
            if ctx.research_system is not None and ctx.study_id and int(offset or 0) == 0:
                try:
                    card = lib.documents.get(paper_id) if hasattr(lib, "documents") else {}
                    details = card.get("details") or {}
                    url = str(details.get("url") or "")
                    title = str(card.get("text") or paper_id)
                    text = str(out.get("text") or "")
                    if url.startswith(("http://", "https://")) and text:
                        ctx.research_system.notebook.source(
                            ctx.study_id, title, url, text[:600], str(out.get("locator") or "fulltext"),
                            reading_depth=str(details.get("reading_depth") or "fulltext_read"), content=text,
                            module_ref={"owner": registry._manifests["literature"].id, "kind": "paper", "id": paper_id},
                        )
                except Exception:
                    pass
            return out

        expose("read_library_paper", "Fetch/cache full text only on explicit reading; paginate or focus by query.", read_library_paper,
               {"paper_id": string, "offset": {"type": "integer"}, "limit": {"type": "integer"}, "focus_query": string}, ["paper_id"])

        def annotate_paper(paper_id, notes, tags=None):
            kwargs = {"paper_id": paper_id, "notes": notes}
            if tags is not None:
                kwargs["tags"] = tags
            return _call_supported(lib.annotate, **kwargs)

        expose("annotate_paper", "Update your paper notes/tags without fetching the full text.", annotate_paper,
               {"paper_id": string, "notes": string, "tags": strings}, ["paper_id", "notes"])

    if registry.has_legacy("notebook"):
        notes = registry.get("notebook")

        def write_note(kind, text, record_id="", refs=None, summary="", memory_type="semantic", scope="project", details=None):
            ref_list = list(refs or [])
            put_kwargs = {**dict(details or {}), "kind": kind, "text": text, "record_id": record_id, "refs": ref_list}
            if ctx.study_id:
                put_kwargs["study_id"] = ctx.study_id
            if summary:
                put_kwargs["summary"] = summary
            if memory_type:
                put_kwargs["memory_type"] = memory_type
            if scope:
                put_kwargs["scope"] = scope
            rec = _call_supported(notes.put, **put_kwargs)
            if ctx.research_system is not None and ctx.study_id and kind in {
                "question", "reproduction", "observation", "hypothesis", "method",
                "decision", "limitation", "summary", "review", "protocol", "document",
            }:
                try:
                    extra = {**dict(details or {}), "module_record_id": rec["id"], "module_record_owner": registry._manifests["notebook"].id}
                    if kind == "decision":
                        extra["choice"] = "continue"
                        extra["next_action"] = summary or text[:120]
                    from .adapters import ledger_refs
                    ctx.research_system.notebook.record(ctx.study_id, kind, text, ledger_refs(ctx.research_system, ctx.study_id, ref_list), extra)
                except Exception as exc:
                    rec["projection_warning"] = str(exc)
                    if getattr(ctx, "emit_event", None):
                        ctx.emit_event("projection_warning", source="notes", record_id=rec["id"], error=str(exc))
            return rec

        expose("write_long_term_note", "Persist arbitrary tasks, goals, ideas, questions or experiment notes across runs in this project; record_id creates a new revision.", write_note,
               {"kind": string, "text": string, "record_id": string, "refs": refs_schema,
                "summary": string, "memory_type": string, "scope": string, "details": {"type": "object"}}, ["kind", "text"])
        expose("search_long_term_notes", "Retrieve project memory across runs, optionally filtered by kind.", notes.search,
               {"query": string, "kind": string})
        expose("read_long_term_note", "Read the full content and revision details of a long-term note by ID.", notes.get,
               {"record_id": string}, ["record_id"])

    if registry.has_legacy("coding"):
        coding = registry.get("coding")
        expose("create_coding_task", "Give the coder a self-contained implementation request.", coding.task, {"text": string}, ["text"])

        def implement_coding_task(task_id, instruction=""):
            if not hasattr(coding, "implement"):
                raise ValueError("Configured coding module does not support implement()")
            return coding.implement(task_id, instruction, ctx=ctx)

        expose("implement_coding_task", "Run the isolated Coder agent to implement a coding_task and return its code submission.", implement_coding_task,
               {"task_id": string, "instruction": string}, ["task_id"])

        def submit_code(task_id, files, run, test_paths=None):
            # Normalize defaults with the same contract used by the experiment service.
            clean_files = [str(f).strip().lstrip("/").replace("\\", "/") for f in files if str(f).strip()]
            raw_run = dict(run or {})
            for key in ("entrypoint", "check_entrypoint", "metrics_path"):
                if raw_run.get(key):
                    raw_run[key] = str(raw_run[key]).strip().lstrip("/").replace("\\", "/")
            if "timeout_seconds" not in raw_run:
                raw_run["timeout_seconds"] = ctx.config.plugins.harness.experiment_timeout_seconds
            spec = ExperimentSpec(source_dir=str(ctx.workspace), files=tuple(clean_files), **raw_run)
            canonical = asdict(spec)
            canonical.pop("source_dir")
            canonical.pop("files")
            return _call_supported(
                coding.submit,
                task_id=task_id,
                source_dir=str(ctx.workspace),
                files=clean_files,
                run=canonical,
                actor=getattr(ctx, "actor", "main"),
                test_paths=test_paths,
            )

        expose("submit_code", "Freeze implementation and exact run specification for researcher review; does not execute code.", submit_code,
               {"task_id": string, "files": strings, "run": {"type": "object"}, "test_paths": strings}, ["task_id", "files", "run"])

        def review_code_submission(submission_id, approved, notes):
            return _call_supported(
                coding.review,
                submission_id=submission_id,
                approved=approved,
                notes=notes,
                actor=getattr(ctx, "actor", "main"),
            )

        expose("review_code_submission", "Researcher approves/rejects a specific frozen code version and run specification.", review_code_submission,
               {"submission_id": string, "approved": {"type": "boolean"}, "notes": string}, ["submission_id", "approved", "notes"])
        expose("inspect_code_submission", "Read submitted version metadata or a frozen source file for review.", coding.inspect,
               {"submission_id": string, "file_path": string}, ["submission_id"])
        if hasattr(coding, "test_submission"):
            def test_code_submission(submission_id, timeout=60, test_paths=None):
                from ..plugins.harness import _resolve_science_python
                result = _call_supported(coding.test_submission, submission_id=submission_id, timeout=timeout,
                                         python_executable=_resolve_science_python(ctx), test_paths=test_paths)
                if result.get("passed") and ctx.state is not None:
                    from ..plugins.sanity_gate import python_manifest
                    current = python_manifest(ctx.workspace)
                    tested = {k: v for k, v in result["manifest"].items() if k.endswith(".py")}
                    if current == tested:
                        ctx.state.metadata["tested_python_manifest"] = current
                        ctx.state.metadata["modified_py_since_test"] = []
                    ctx.state.metadata["tests_executed"] = int(ctx.state.metadata.get("tests_executed", 0)) + 1
                    ctx.state.metadata.setdefault("validated_submissions", {})[submission_id] = result["manifest"]
                return result
            expose("test_code_submission", "Run pytest on a frozen Coder submission; no raw shell and no formal measurements.", test_code_submission,
                   {"submission_id": string, "timeout": {"type": "integer", "minimum": 1, "maximum": 300}, "test_paths": strings}, ["submission_id"])

    if registry.has_legacy("experiments"):
        exps = registry.get("experiments")
        if hasattr(exps, "environment_info"):
            def get_execution_environment():
                from ..plugins.harness import _resolve_science_python
                return exps.environment_info(python_executable=_resolve_science_python(ctx))
            expose("get_execution_environment", "Inspect the configured local experiment Python interpreter and installed scientific packages; Docker images may differ. No installation or network access.", get_execution_environment)
        if hasattr(exps, "annotate"):
            def annotate_experiment(job_id, note, kind="observation"):
                return exps.annotate(job_id, note, kind=kind, registry=registry)

            expose("annotate_experiment", "Attach an observation or evaluation note directly to an experiment record in the durable notebook.", annotate_experiment,
                   {"job_id": string, "note": string, "kind": string}, ["job_id", "note"])

    tools.extend(registry.native_tools())
    from ..runtime.contracts import TaskRef
    for tool_name, operation in (("get_module_task", "get"), ("get_module_result", "result"), ("cancel_module_task", "cancel")):
        def task_call(owner, task_id, operation=operation):
            return registry.task(TaskRef(owner, task_id), operation)
        tools.append(ToolDef(tool_name, "Query or cancel a task owned by a module; unsupported operations return an explicit error.",
                             {"type": "object", "properties": {"owner": string, "task_id": string},
                              "required": ["owner", "task_id"], "additionalProperties": False}, task_call,
                             permissions=("tasks.cancel" if operation == "cancel" else "tasks.read",), wire_result=True))
    from ..runtime.contracts import ObjectRef
    def resolve_ref(owner, kind, object_id):
        return registry.resolve(ObjectRef(kind, owner, object_id))
    tools.append(ToolDef("resolve_module_ref", "Resolve an owner-qualified object reference without knowing its backend.",
                         {"type": "object", "properties": {"owner": string, "kind": string, "object_id": string},
                          "required": ["owner", "kind", "object_id"], "additionalProperties": False}, resolve_ref,
                         permissions=("objects.read",), wire_result=True))
    def query_trace(after=0, limit=50, actor="", run_id=""):
        return registry.invoke("observability.trace", "query", {"after": after, "limit": limit, "actor": actor, "run_id": run_id})
    tools.append(ToolDef("query_runtime_trace", "Read model/tool trace events with parent IDs and input/output references.",
        {"type": "object", "properties": {"after": {"type": "integer", "minimum": 0}, "limit": {"type": "integer", "minimum": 1, "maximum": 500},
                                         "actor": string, "run_id": string}, "additionalProperties": False},
        query_trace, permissions=("trace.read",), wire_result=True))

    for item in registry.describe():
        if registry.is_native(item["name"]):
            continue
        module = registry.get(item["name"])
        contribute = getattr(module, "tools", None)
        if contribute:
            tools.extend(contribute(ctx))
    names = [t.name for t in tools]
    if len(names) != len(set(names)):
        raise ValueError("Modules contributed duplicate tool names")
    return tools
