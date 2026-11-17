"""One proposal per tick; commands and stale snapshots are checked before mutation.

The old ResearchKernel is not implicitly wrapped or started. A production model
adapter must implement ResearchAgent.next_step and account for calls separately.
"""

from __future__ import annotations

from ..domain import ExperimentSpec
from ..interaction.commands import UserCommandInbox
from ..ports import ExperimentHarness, ResearchAgent
from ..research.controller import ResearchController
from ..research.notebook import ResearchNotebook
from ..storage.sqlite import ConflictError, SqliteResearchStore
from .context import ResearchContextBuilder


class AgentCoordinator:
    def __init__(self, store: SqliteResearchStore, controller: ResearchController, notebook: ResearchNotebook,
                 harness: ExperimentHarness, inbox: UserCommandInbox, contexts: ResearchContextBuilder):
        self.store, self.controller, self.notebook = store, controller, notebook
        self.harness, self.inbox, self.contexts = harness, inbox, contexts

    def tick(self, study_id: str, agent: ResearchAgent) -> dict:
        self.inbox.apply_pending(study_id)
        snapshot = self.store.snapshot(study_id)
        if snapshot["study"]["status"] != "running":
            return {"status": "paused"}
        proposed = agent.next_step(self.contexts.build(snapshot))
        self.inbox.apply_pending(study_id)
        current = self.store.get(study_id)
        if current.revision != proposed.based_on_revision or current.status != "running":
            return {"status": "refresh_required", "reason": "State changed while the agent was planning"}
        data = proposed.payload
        try:
            if proposed.kind == "experiment":
                spec = ExperimentSpec(**data["spec"])
                job = self.harness.submit(study_id, current.plan_version, spec, data["request_key"],
                                          expected_revision=current.revision)
                return {"status": "submitted", "job_id": job.id}
            if proposed.kind == "decision":
                record = self.controller.decide(study_id, data["choice"], data["rationale"],
                                                tuple(data.get("refs", [])), data["next_action"],
                                                expected_revision=current.revision)
            elif proposed.kind in {"question", "observation", "method", "limitation", "reproduction"}:
                record = self.notebook.record(study_id, proposed.kind, data["text"],
                                              tuple(data.get("refs", [])), data.get("details"),
                                              expected_revision=current.revision)
            else:
                raise ValueError(f"Unsupported agent step: {proposed.kind}")
            return {"status": "recorded", "record_id": record.id}
        except ConflictError as exc:
            return {"status": "refresh_required", "reason": str(exc)}
