"""Evidence-linked notes; an observation is distinct from an execution result."""

from __future__ import annotations

from typing import Any

from ..domain import ResearchRecord, identity
from ..storage.sqlite import SqliteResearchStore


class ResearchNotebook:
    def __init__(self, store: SqliteResearchStore, artifacts=None):
        self.store, self.artifacts = store, artifacts

    def record(self, study_id: str, kind: str, text: str, refs: tuple[str, ...] = (),
               details: dict[str, Any] | None = None, *, expected_revision: int | None = None) -> ResearchRecord:
        if kind not in {"source", "search", "review", "summary", "protocol", "stage", "report", "question", "reproduction", "observation", "hypothesis", "method", "decision", "limitation", "document"}:
            raise ValueError("Unknown research record kind")
        if not text.strip():
            raise ValueError("A research record needs an explicit statement")
        details = dict(details or {})
        details["plan_version"] = self.store.get(study_id).plan_version
        if self.artifacts is not None:
            details["text_hash"] = self.artifacts.put(text.encode("utf-8"))
        if kind in {"review", "summary"} and not refs:
            raise ValueError("Reviews and summaries must cite their underlying records or experiments")
        if kind == "review":
            source_ids = {n["id"] for n in self.store.snapshot(study_id)["records"] if n["kind"] == "source"}
            if not source_ids.intersection(refs):
                raise ValueError("A literature review must cite at least one read source")
        if kind == "decision" and (details.get("choice") not in {"continue", "pivot", "park", "report"}
                                   or not str(details.get("next_action", "")).strip()):
            raise ValueError("A decision needs a supported choice and next_action")
        if kind == "observation" and not refs:
            raise ValueError("Observations must cite evidence")
        if kind in {"observation", "decision"} and refs:
            snapshot = self.store.snapshot(study_id)
            known_jobs = {job["id"]: job for job in snapshot["jobs"]}
            known_sources = {
                n["id"] for n in snapshot["records"]
                if n["kind"] in {"source", "observation", "hypothesis", "method", "reproduction"}
            }
            for ref in refs:
                if ref in known_jobs:
                    if kind == "observation" and known_jobs[ref]["status"] not in {"succeeded", "failed"}:
                        raise ValueError("A queued or running experiment is not an observed result")
                    if known_jobs[ref].get("validity") == "invalid" and not details.get("allow_invalid_ref"):
                        raise ValueError(f"Experiment {ref} is marked invalid and cannot be cited as valid evidence")
                elif kind == "observation" and ref not in known_sources:
                    raise ValueError("An observation needs a result, read source or prior observation")
        record = ResearchRecord(identity("note"), study_id, kind, text, refs, details)
        self.store.append(record, expected_revision=expected_revision)
        return record

    def source(
        self,
        study_id: str,
        title: str,
        url: str,
        excerpt: str,
        locator: str,
        *,
        reading_depth: str = "abstract_only",
        saved_path: str = "",
        content: str = "",
        module_ref: dict | None = None,
    ) -> ResearchRecord:
        if not url.startswith(("https://", "http://")) or not excerpt.strip() or not locator.strip():
            raise ValueError("A source needs its URL, actually read excerpt and locator")
        return self.record(
            study_id,
            "source",
            title,
            details={
                "url": url,
                "excerpt": excerpt,
                "locator": locator,
                "reading_depth": reading_depth,
                "saved_path": saved_path,
                "content_hash": self.artifacts.put((content or excerpt).encode("utf-8")) if self.artifacts else "",
                "plan_version": self.store.get(study_id).plan_version,
                "module_ref": module_ref,
            },
        )
