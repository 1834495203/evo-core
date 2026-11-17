"""Own study scope and record decisions; never run experiments in the model loop."""

from __future__ import annotations

from dataclasses import replace

from ..domain import ResearchRecord, Study, identity
from ..storage.sqlite import SqliteResearchStore
from .notebook import ResearchNotebook


class ResearchController:
    def __init__(self, store: SqliteResearchStore, notebook: ResearchNotebook):
        self.store, self.notebook = store, notebook

    def create(self, task: str, *, max_experiments: int | None = 12) -> Study:
        if not task.strip() or (max_experiments is not None and max_experiments < 1):
            raise ValueError("A study needs a task and a positive experiment budget")
        study = Study(identity("study"), task, task, max_experiments=max_experiments)
        self.store.create(study)
        return study

    def focus(self, study_id: str, phase: str, next_action: str) -> Study:
        if not phase.strip():
            raise ValueError("A focus label must be nonempty")
        if not next_action.strip():
            raise ValueError("Record what the agent intends to do next")
        return self.store.revise(replace(self.store.get(study_id), phase=phase, next_action=next_action), "plan.focused")

    def decide(self, study_id: str, choice: str, rationale: str, refs: tuple[str, ...],
               next_action: str, *, expected_revision: int | None = None) -> ResearchRecord:
        if choice not in {"continue", "pivot", "park", "report"} or not next_action.strip():
            raise ValueError("Choose continue, pivot, park or report and name the next action")
        return self.notebook.record(study_id, "decision", rationale, refs,
                                    {"choice": choice, "next_action": next_action},
                                    expected_revision=expected_revision)
