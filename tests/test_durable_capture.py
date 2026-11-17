"""Research recording is automatic persistence, not a prescribed scientific workflow."""
from dataclasses import asdict
from pathlib import Path

from evo_core.config import CoreConfig
from evo_core.harness_server import HarnessSession
from evo_core.runtime import Manifest, Capability, Module, Export, Result, ObjectRef, native_factory


class Papers:
    reads = 0
    def search(self, query, limit):
        return [{"title": "Candidate", "abstract": "An abstract", "url": "https://example.test/paper", "published": "2025-01-01"}]
    def read(self, url):
        self.reads += 1
        return {"content": "Full text", "excerpt": "Full text", "reading_depth": "fulltext_read", "locator": "section 1"}


def test_search_history_survives_restart_without_automatic_admission(tmp_path):
    cfg = CoreConfig(workspace=str(tmp_path / "ws"), run_dir=str(tmp_path / "meta"))
    session = HarnessSession(cfg, "Search freely")
    provider = Papers(); session.ctx.module_registry.get("literature").provider = provider
    result = session.invoke("find_papers", {"query": "time series", "limit": 1})
    candidate = result.data[0]
    assert session.invoke("list_papers", {}).data == []
    assert provider.reads == 0
    assert session.invoke("list_paper_searches", {}).data[0]["count"] == 1
    session.close()
    resumed = HarnessSession(cfg, "Search freely")
    resumed.ctx.module_registry.get("literature").provider = provider
    try:
        saved = resumed.invoke("read_paper_search", {"search_id": candidate["search_id"]}).data
        assert saved["details"]["candidates"][0]["abstract"] == "An abstract"
        assert resumed.invoke("list_papers", {}).data == []
        admitted = resumed.invoke("admit_paper", {"candidate_id": candidate["candidate_id"], "notes": "Use as baseline"})
        assert admitted.status == "completed"
        assert admitted.data["details"]["abstract"] == "An abstract"
        assert provider.reads == 0
        same = resumed.invoke("admit_paper", {"candidate_id": candidate["candidate_id"], "notes": "Use as baseline"})
        assert same.data["id"] == admitted.data["id"]
        assert len(resumed.invoke("list_papers", {}).data) == 1
        assert resumed.invoke("read_library_paper", {"paper_id": admitted.data["id"]}).data["text"] == "Full text"
        assert provider.reads == 1
        assert any(r["kind"] == "search" for r in resumed.ctx.research_system.queries.snapshot(resumed.ctx.study_id)["records"])
    finally:
        resumed.close()


def test_file_writing_and_editing_create_durable_drafts_not_scientific_sources(tmp_path):
    cfg = CoreConfig(workspace=str(tmp_path / "ws"), run_dir=str(tmp_path / "meta"))
    session = HarnessSession(cfg, "Write notes naturally")
    try:
        first = session.invoke("write_file", {"file_path": "research_request.md", "content": "# Goal\nHypothesis A"})
        assert "archived note" in first.data
        notes = session.ctx.module_registry.get("notebook")
        original = notes.search("Hypothesis A", "document")[0]
        assert original["details"]["draft"] is True
        session.invoke("edit_file", {"file_path": "research_request.md", "old_string": "Hypothesis A", "new_string": "Hypothesis B"})
        updated = notes.get(original["id"])
        assert updated["text"] == "# Goal\nHypothesis B"
        assert updated["details"]["source_path"] == "research_request.md"
        assert session.kernel.host.artifacts.read(ObjectRef(**updated["details"]["artifact"])).decode() == updated["text"]
        session.invoke("write_file", {"file_path": "results/generated.md", "content": "Raw result"})
        assert len(notes.search("", "document")) == 1
        ledger = session.ctx.research_system.queries.snapshot(session.ctx.study_id)["records"]
        assert [r["kind"] for r in ledger] == ["document", "document"]
        assert session.invoke("list_papers", {}).data == []
    finally:
        session.close()
    later_cfg = CoreConfig(workspace=cfg.workspace, run_dir=str(tmp_path / "later"))
    later = HarnessSession(later_cfg, "Continue another run")
    try:
        assert later.ctx.module_registry.get("notebook").search("Hypothesis B")
    finally:
        later.close()


REMOTE = Manifest("remote.notes", provides=(Capability("notes.store"),))
@native_factory(REMOTE)
def remote_notes(ctx):
    def put(call, args):
        rec = {"id": args["record_id"], "kind": args["kind"], "text": args["text"], "details": dict(args.get("details") or {})}
        ctx.storage.put(rec["id"] + ".json", rec)
        return Result(data=rec)
    def get(call, args):
        try: return Result(data=ctx.storage.get(args["record_id"] + ".json"))
        except FileNotFoundError: return Result.failure("not_found", "No note")
    return Module(REMOTE, (Export("notes.store", "put", put, {"type": "object"}, {"type": "object"}, ("notes.write",)),
                           Export("notes.store", "get", get, {"type": "object"}, {"type": "object"}, ("notes.read",))))


def test_native_workspace_capture_works_with_a_replaced_notes_provider(tmp_path):
    cfg = CoreConfig(workspace=str(tmp_path / "ws"), run_dir=str(tmp_path / "meta"), modules={
        "notebook": {"enabled": False}, "remote": {"contract": "native", "factory": "test_durable_capture:remote_notes"}})
    session = HarnessSession(cfg, "Native capture")
    try:
        session.kernel.host.workspace.write("notes/idea.md", "An idea")
        records = session.ctx.research_system.queries.snapshot(session.ctx.study_id)["records"]
        assert records[0]["kind"] == "document"
        assert records[0]["details"]["module_record_owner"] == "remote.notes"
        assert not any(hasattr(session.ctx.module_registry._contexts["remote"], k) for k in ("state", "research_system"))
    finally:
        session.close()
