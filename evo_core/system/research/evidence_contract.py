"""Deterministic evidence obligations; never require a positive scientific result."""

from __future__ import annotations

from collections.abc import Iterable
import re


def resolve_cited_experiment_ids(report: str, job_ids: Iterable[str]) -> tuple[set[str], set[str]]:
    known = set(job_ids)
    tokens = set(re.findall(r"\bexp_[a-f0-9]+\b", report))
    resolved: set[str] = set()
    unknown: set[str] = set()
    for token in tokens:
        if token in known:
            resolved.add(token)
        elif len(token) >= 12:
            matches = [jid for jid in known if jid.startswith(token)]
            if len(matches) == 1:
                resolved.add(matches[0])
            else:
                unknown.add(token)
        else:
            unknown.add(token)
    return resolved, unknown


def pending_interpretations(snapshot: dict) -> list[str]:
    interpreted = {ref for note in snapshot["records"]
                   if note["kind"] in {"observation", "limitation"} for ref in note["refs"]}
    return [j["id"] for j in snapshot["jobs"]
            if j["status"] in {"succeeded", "failed"} and j["id"] not in interpreted]


def delivery_issues(snapshot: dict, report: str) -> list[str]:
    jobs = {j["id"]: j for j in snapshot["jobs"]}
    issues = []
    if not jobs:
        honest_no_execution = (snapshot["study"]["scientific_outcome"] == "inconclusive"
                               and any(n["kind"] == "limitation" for n in snapshot["records"]))
        if not honest_no_execution:
            issues.append("No registered experiment evidence. Shell-generated result files do not satisfy the harness contract. "
                          "Submit experiments, or explicitly record an inconclusive outcome and the execution limitation.")
        return issues
    cited, unknown = resolve_cited_experiment_ids(report, jobs.keys())
    if unknown:
        issues.append(f"Report cites unknown experiment IDs: {sorted(unknown)}.")
    if report and not cited and any(j["status"] == "succeeded" for j in jobs.values()):
        issues.append("Report must cite the ledger experiment IDs underlying its experimental results.")
    for key in sorted(cited & jobs.keys()):
        job = jobs[key]
        if job["status"] == "succeeded" and job.get("validity", "unverified") == "unverified":
            issues.append(f"Cited experiment {key} has not been independently checked. Use verify_experiment.")
    return issues
