"""Durable development harness: freeze now, execute later in a separate worker."""

from __future__ import annotations

from pathlib import Path

from ..domain import ExperimentSpec, Job, digest, document, encode, identity
from ..storage.artifacts import ArtifactStore, relative_path
from ..storage.sqlite import SqliteResearchStore


class LocalExperimentHarness:
    def __init__(self, store: SqliteResearchStore, artifacts: ArtifactStore):
        self.store, self.artifacts = store, artifacts

    def submit(self, study_id: str, plan_version: int, spec: ExperimentSpec,
               request_key: str, *, expected_revision: int | None = None,
               expected_manifest: dict | None = None) -> Job:
        if not all(s.strip() for s in (spec.purpose, spec.protocol, spec.data_version, request_key)):
            raise ValueError("Purpose, protocol, data version and request key are required")
        if spec.timeout_seconds is not None and (type(spec.timeout_seconds) is not int or spec.timeout_seconds < 1):
            raise ValueError("Executor timeout must be positive or null for no time limit")
        relative_path(spec.entrypoint)
        metric_path = relative_path(spec.metrics_path)
        if len(metric_path.parts) < 2 or metric_path.parts[0] not in {"outputs", "results"} or metric_path.suffix != ".json":
            raise ValueError("metrics_path must be a JSON file under outputs/ or results/")
        if any(name.startswith("results/") for name in spec.files):
            raise ValueError("Prior results cannot be included in source snapshots")
        if not spec.entrypoint.endswith(".py") or spec.entrypoint not in spec.files:
            raise ValueError("The Python entrypoint must be one of the frozen source files")
        if spec.check_entrypoint:
            from .validation import check_runner
            check_runner(spec.check_entrypoint, spec.check_runner)
            relative_path(spec.check_entrypoint)
            if not spec.check_entrypoint.endswith(".py") or spec.check_entrypoint not in spec.files:
                raise ValueError("The check_entrypoint must be a .py file included in frozen files")
        if any(name.startswith("outputs/") or name in {"stdout.log", "stderr.log"} for name in spec.files):
            raise ValueError("Source snapshots cannot include prior experiment outputs")
        if any(not isinstance(arg, str) for arg in spec.args):
            raise ValueError("Experiment arguments must be strings")
        manifest = self.artifacts.freeze(Path(spec.source_dir), spec.files)
        if expected_manifest is not None and manifest != expected_manifest:
            raise ValueError("Source changed while freezing the approved submission")
        frozen_spec = document(spec)
        frozen_spec.pop("source_dir")
        fingerprint = digest(encode({"plan_version": plan_version, "spec": frozen_spec,
                                     "manifest": manifest}).encode())
        job = Job(identity("exp"), study_id, plan_version, request_key, fingerprint, frozen_spec, manifest)
        return self.store.enqueue(job, expected_revision=expected_revision)

    def get(self, job_id: str) -> Job:
        return self.store.get_job(job_id)
