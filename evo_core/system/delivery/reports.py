"""Export an honest progress notebook. Does not declare scientific acceptance."""

from __future__ import annotations

from pathlib import Path

from ..domain import encode
from ..interaction.queries import ResearchQueries


class ProgressReportWriter:
    def __init__(self, queries: ResearchQueries):
        self.queries = queries

    def export(self, study_id: str, path: Path) -> Path:
        snapshot = self.queries.snapshot(study_id)
        study = snapshot["study"]
        lines = ["# Research progress export", "",
                 "This is a recorded-progress export, not an accepted scientific paper.", "",
                 f"Snapshot: {snapshot['as_of']}; event cursor: {snapshot['event_cursor']}", "",
                 f"Task: {study['task']}", f"Direction: {study['direction']}",
                 f"Phase: {study['phase']}; execution: {study['status']}; acceptance: {study['acceptance']}",
                 f"Next action: {study['next_action']}", "", "## Recorded research", ""]
        for record in snapshot["records"]:
            lines.extend([f"### {record['kind']} / {record['id']}", "", record["text"], "",
                          f"Evidence: {', '.join(record['refs']) or 'none'}", "",
                          "```json", encode(record["details"]), "```", ""])
        lines.extend(["## Experiments", ""])
        for job in snapshot["jobs"]:
            lines.extend([
                f"### {job['id']} / {job['status']} (validity={job.get('validity', 'unverified')})",
                "",
                job["spec"]["purpose"],
                "",
                "```json",
                encode({
                    "plan_version": job["plan_version"],
                    "validity": job.get("validity", "unverified"),
                    "invalidated_reason": job.get("invalidated_reason", ""),
                    "checks": job.get("checks", []),
                    "spec": job["spec"],
                    "manifest": job["manifest"],
                    "result": job["result"],
                }),
                "```",
                "",
            ])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")
        return path
