"""Runtime-neutral tools: the harness owns documents, summaries and evidence queries."""
import json
import re

from ...protocol import ToolDef
from .evidence_contract import delivery_issues
from ..delivery.bundle import export_bundle


def archive_tools(ctx):
    def record_document(kind, text, refs=None, details=None):
        if kind in {"report", "source", "stage"}:
            raise ValueError("Use archive_research_report, read_paper_fulltext/record_research_note or set_research_focus for this kind")
        record = ctx.research_system.notebook.record(ctx.study_id, kind, text, tuple(refs or ()), details)
        if ctx.research_system.modules is not None and ctx.research_system.modules.has("notebook"):
            try:
                ctx.research_system.modules.get("notebook").put(
                    kind, text, record_id=record.id, refs=list(refs or ()), study_id=ctx.study_id, ledger_record_id=record.id
                )
            except Exception:
                pass
        return json.dumps({"record_id": record.id, "details": record.details}, ensure_ascii=False)

    def query_records(kind="", record_ids=None, offset=0, limit=20):
        snapshot = ctx.research_system.queries.snapshot(ctx.study_id)
        records = [r for r in snapshot["records"] if (not kind or r["kind"] == kind)
                   and (not record_ids or r["id"] in record_ids)]
        start, count = max(0, offset), max(1, min(100, limit))
        return json.dumps({"total": len(records), "records": records[start:start+count],
                           "event_cursor": snapshot["event_cursor"]}, ensure_ascii=False)

    def read_document(record_id, offset=0, max_chars=16000):
        snapshot = ctx.research_system.queries.snapshot(ctx.study_id)
        record = next(r for r in snapshot["records"] if r["id"] == record_id)
        details = record["details"]
        key = details.get("content_hash") or details.get("text_hash")
        content = ctx.research_system.artifacts.read(key).decode("utf-8") if key else record["text"]
        start, cap = max(0, offset), max(1, min(32000, max_chars))
        return json.dumps({"record_id": record_id, "content": content[start:start+cap],
                           "total_chars": len(content), "next_offset": start+cap if start+cap < len(content) else None,
                           "content_hash": key}, ensure_ascii=False)

    def archive_report(content, refs):
        system = ctx.research_system
        snapshot = system.queries.snapshot(ctx.study_id)
        issues = delivery_issues(snapshot, content)
        if issues:
            raise ValueError("\n".join(issues))
        images = []
        for raw in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", content):
            name = raw.strip().split()[0]
            if not name.startswith(("https://", "http://", "data:")):
                images.append(name)
        assets = system.artifacts.freeze(ctx.workspace, tuple(dict.fromkeys(images))) if images else {}
        record = system.notebook.record(ctx.study_id, "report", content, tuple(refs),
                                        {"assets": assets, "based_on_event_cursor": snapshot["event_cursor"]})
        (ctx.workspace / "final_report.md").write_text(content, encoding="utf-8")
        system.reports.export(ctx.study_id, system.root / "reports" / "research_history.md")
        return json.dumps({"record_id": record.id, "content_hash": record.details["text_hash"],
                           "report": str(ctx.workspace / "final_report.md")})

    def export_delivery():
        path = export_bundle(ctx.research_system, ctx.study_id,
                             ctx.research_system.root / "reports" / "research_bundle.zip")
        return json.dumps({"bundle": str(path)})

    def tool(name, description, handler, properties=None, required=None):
        return ToolDef(name, description, {"type": "object", "properties": properties or {},
                                          "required": required or []}, handler)
    string = {"type": "string"}
    refs = {"type": "array", "items": string}
    return [
        tool("record_research_document", "Persist summary, literature review, protocol, question or other research note in the harness. Reviews/summaries require evidence refs.",
             record_document, {"kind": string, "text": string, "refs": refs, "details": {"type": "object"}}, ["kind", "text"]),
        tool("query_research_records", "Query complete durable research history by kind/IDs with pagination; survives context compaction.",
             query_records, {"kind": string, "record_ids": refs, "offset": {"type": "integer"}, "limit": {"type": "integer"}}),
        tool("read_research_document", "Read archived paper or note bytes with hash verification and pagination.",
             read_document, {"record_id": string, "offset": {"type": "integer"}, "max_chars": {"type": "integer"}}, ["record_id"]),
        tool("archive_research_report", "Validate evidence links and archive the final report in the harness; materialize final_report.md.",
             archive_report, {"content": string, "refs": refs}, ["content", "refs"]),
        tool("export_research_bundle", "Export archived report, papers, notes, code, metrics CSV, raw results, plots and Quarto source as a portable ZIP.", export_delivery),
    ]
