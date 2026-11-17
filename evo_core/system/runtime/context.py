"""Build a bounded working view while the complete records remain in storage."""

from __future__ import annotations

from ..research.literature import LiteratureReviewPolicy


class ResearchContextBuilder:
    def __init__(self, literature_policy: LiteratureReviewPolicy):
        self.literature_policy = literature_policy

    def build(self, snapshot: dict) -> dict:
        notes = []
        for record in snapshot["records"][-12:]:
            notes.append({"id": record["id"], "kind": record["kind"], "text": record["text"][:1800],
                          "refs": record["refs"]})
        jobs = []
        for job in snapshot["jobs"][-12:]:
            result = job["result"] or {}
            metrics = {key: value[:300] if isinstance(value, str) else value
                       for key, value in list(result.get("metrics", {}).items())[:12]
                       if isinstance(value, (str, int, float, bool)) or value is None}
            jobs.append({"id": job["id"], "purpose": job["spec"]["purpose"][:600], "status": job["status"],
                         "protocol": job["spec"]["protocol"], "metrics_preview": metrics,
                         "artifact_paths": list(result.get("artifacts", {})), "error": result.get("error", "")[:900]})
        return {"study": snapshot["study"], "as_of": snapshot["as_of"],
                "records": notes, "experiments": jobs,
                "user_messages": snapshot["commands"][-12:],
                "history_note": "This is a working view. Query stored records and artifacts for complete evidence."}
