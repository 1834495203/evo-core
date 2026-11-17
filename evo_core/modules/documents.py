"""Application bridge: file-based research writing is durable project memory."""
from contextvars import ContextVar
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

_CAPTURING = ContextVar("evo_capturing_document", default=False)
_IGNORED = {".research_modules", ".git", "results", "outputs", "archive_invalid", ".pytest_tmp", ".test_tmp"}


def install_document_capture(ctx, registry):
    if not registry.has_capability("notes.store") or not getattr(ctx, "workspace", None): return
    root = Path(ctx.workspace).resolve()

    def capture(path, text):
        if _CAPTURING.get(): return None
        target = (root / path).resolve()
        if not target.is_relative_to(root): return None
        relative = target.relative_to(root)
        if target.suffix.lower() not in {".md", ".txt", ".rst"} or _IGNORED.intersection(relative.parts): return None
        token = _CAPTURING.set(True)
        try:
            name = relative.as_posix()
            record_id = "file_" + sha256(name.encode()).hexdigest()[:24]
            digest = sha256(text.encode()).hexdigest()
            previous = registry.invoke("notes.store", "get", {"record_id": record_id})
            if previous.status == "completed" and previous.data.get("details", {}).get("source_hash") == digest:
                return record_id
            artifact = ctx.host.artifacts.put(text.encode()) if getattr(ctx, "host", None) else None
            result = registry.invoke("notes.store", "put", {"kind": "document", "text": text or "(Document is now empty.)",
                "record_id": record_id, "details": {"source_path": name, "source_hash": digest,
                "origin": "workspace_document", "draft": True,
                "artifact": asdict(artifact) if artifact else None}})
            if result.error: raise RuntimeError(result.error.message)
            if getattr(ctx, "research_system", None) is not None and ctx.study_id and not registry.has_legacy("notebook"):
                provider = registry._capabilities["notes.store"][0]
                ctx.research_system.notebook.record(ctx.study_id, "document", text or "(Document is now empty.)",
                    details={"source_path": name, "source_hash": digest, "draft": True,
                             "module_record_id": result.data["id"], "module_record_owner": registry._manifests[provider].id})
            if getattr(ctx, "emit_event", None):
                ctx.emit_event("document_archived", path=name, record_id=result.data["id"], source_hash=digest)
            return result.data["id"]
        except Exception as exc:
            if getattr(ctx, "emit_event", None): ctx.emit_event("document_archive_error", path=str(path), error=str(exc))
            if getattr(ctx, "state", None): ctx.state.add_notice(f"File saved but memory archive failed: {path}: {exc}")
            return None
        finally:
            _CAPTURING.reset(token)

    ctx.observe_file = capture
    if getattr(ctx, "host", None): ctx.host.observe_file = capture
