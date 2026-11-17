"""The observer receives this read interface, not the controller or harness."""

from __future__ import annotations

from ..ports import SnapshotReader
from ..research.evidence_contract import pending_interpretations


class ResearchQueries:
    def __init__(self, reader: SnapshotReader):
        self._reader = reader

    def snapshot(self, study_id: str) -> dict:
        return self._reader.snapshot(study_id)

    def records(self, study_id: str, *, kind: str = "", offset: int = 0, limit: int = 20) -> dict:
        snapshot = self.snapshot(study_id)
        records = [r for r in snapshot["records"] if not kind or r["kind"] == kind]
        start, cap = max(0, offset), max(1, min(100, limit))
        return {"total": len(records), "records": records[start:start+cap], "event_cursor": snapshot["event_cursor"]}

    def progress(self, study_id: str) -> dict:
        snapshot = self.snapshot(study_id)
        return {
            "as_of": snapshot["as_of"], "event_cursor": snapshot["event_cursor"],
            "study": snapshot["study"],
            "evidence_obligations": {
                "uninterpreted_experiments": pending_interpretations(snapshot),
                "unverified_experiments": [j["id"] for j in snapshot["jobs"]
                                           if j["status"] == "succeeded" and j.get("validity", "unverified") == "unverified"],
            },
            "experiments": [
                {
                    "id": j["id"],
                    "purpose": j["spec"]["purpose"],
                    "status": j["status"],
                    "validity": j.get("validity", "unverified"),
                    "invalidated_reason": j.get("invalidated_reason", ""),
                    "checks": j.get("checks", []),
                    "plan_version": j["plan_version"],
                    "result": j["result"],
                }
                for j in snapshot["jobs"]
            ],
            "recent_notes": snapshot["records"][-10:], "commands": snapshot["commands"],
        }
