"""Dependency boundaries used by the concrete modules, not a plugin registry."""

from __future__ import annotations

from typing import Any, Protocol

from .domain import AgentStep, ExperimentSpec, Job


class ResearchAgent(Protocol):
    def next_step(self, context: dict[str, Any]) -> AgentStep:
        """Return one proposal; the controller owns all side effects."""
        ...


class ExperimentHarness(Protocol):
    def submit(self, study_id: str, plan_version: int, spec: ExperimentSpec,
               request_key: str, *, expected_revision: int | None = None) -> Job: ...

    def get(self, job_id: str) -> Job: ...


class LiteratureProvider(Protocol):
    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]: ...

    def read(self, url: str) -> dict[str, Any]: ...


class SnapshotReader(Protocol):
    def snapshot(self, study_id: str) -> dict[str, Any]: ...
