"""Composition root. Construction starts neither models nor experiment processes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .delivery.reports import ProgressReportWriter
from .experiments.harness import LocalExperimentHarness
from .experiments.worker import LocalPythonWorker, ParallelWorkerPool
from .interaction.commands import UserCommandInbox
from .interaction.queries import ResearchQueries
from .observability.events import ResearchEvents
from .observability.tracker import ExperimentTracker
from .research.controller import ResearchController
from .research.literature import LiteratureReviewPolicy
from .research.notebook import ResearchNotebook
from .runtime.context import ResearchContextBuilder
from .runtime.coordinator import AgentCoordinator
from .storage.artifacts import ArtifactStore
from .storage.sqlite import SqliteResearchStore


@dataclass(frozen=True)
class ResearchSystem:
    root: Path
    store: SqliteResearchStore
    artifacts: ArtifactStore
    notebook: ResearchNotebook
    controller: ResearchController
    harness: LocalExperimentHarness
    inbox: UserCommandInbox
    queries: ResearchQueries
    events: ResearchEvents
    coordinator: AgentCoordinator
    reports: ProgressReportWriter
    tracker: ExperimentTracker
    modules: object = None

    def worker(self, worker_id: str, *, python_executable: str | None = None) -> LocalPythonWorker:
        if self.modules is not None and self.modules.has("experiments"):
            return self.modules.get("experiments").worker(worker_id, python_executable=python_executable)
        return LocalPythonWorker(
            self.store,
            self.artifacts,
            self.root / "executions",
            worker_id,
            python_executable=python_executable,
        )

    def worker_pool(
        self,
        max_workers: int = 2,
        *,
        executor: str = "local",
        python_executable: str | None = None,
    ) -> ParallelWorkerPool:
        if self.modules is not None and self.modules.has("experiments"):
            return self.modules.get("experiments").worker_pool(max_workers, executor=executor,
                                                              python_executable=python_executable)
        if executor == "docker":
            from .experiments.docker_worker import DockerExperimentWorker

            return ParallelWorkerPool(
                self.store,
                self.artifacts,
                self.root / "executions",
                max_workers=max_workers,
                worker_factory=lambda wid: DockerExperimentWorker(
                    self.store,
                    self.artifacts,
                    self.root / "executions",
                    wid,
                    python_executable=python_executable,
                ),
            )
        return ParallelWorkerPool(
            self.store,
            self.artifacts,
            self.root / "executions",
            max_workers=max_workers,
            python_executable=python_executable,
        )


def build_system(
    root: str | Path,
    *,
    experiments_between_reviews: int = 3,
    mlflow_tracking_uri: str | None = None,
    module_specs: dict | None = None,
    module_root: str | Path | None = None,
    host=None,
) -> ResearchSystem:
    root = Path(root).resolve()
    store = SqliteResearchStore(root / "research.sqlite3")
    artifacts = ArtifactStore(root / "artifacts")
    notebook = ResearchNotebook(store, artifacts)
    controller = ResearchController(store, notebook)
    harness = LocalExperimentHarness(store, artifacts)
    inbox = UserCommandInbox(store)
    queries = ResearchQueries(store)
    events = ResearchEvents(store)
    contexts = ResearchContextBuilder(
        LiteratureReviewPolicy(experiments_between_reviews=experiments_between_reviews)
    )
    coordinator = AgentCoordinator(store, controller, notebook, harness, inbox, contexts)
    tracker = ExperimentTracker(root / "tracking", mlflow_tracking_uri=mlflow_tracking_uri)
    system = ResearchSystem(
        root,
        store,
        artifacts,
        notebook,
        controller,
        harness,
        inbox,
        queries,
        events,
        coordinator,
        ProgressReportWriter(queries),
        tracker,
    )
    from ..modules.registry import build_modules
    modules = build_modules(module_root or root / "modules", system, module_specs, host=host)
    object.__setattr__(system, "modules", modules)
    if modules.has("experiments"):
        object.__setattr__(system, "harness", modules.get("experiments"))
        system.coordinator.harness = system.harness
    if modules.has_capability("observability.trace"):
        object.__setattr__(system, "tracker", modules.service("observability.trace"))
    return system
