"""Shared data contracts. No model SDK, executor, UI or database imports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from typing import Any
from uuid import uuid4


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def identity(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def encode(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class Study:
    id: str
    task: str
    direction: str
    revision: int = 0
    plan_version: int = 1
    status: str = "running"
    phase: str = "intake"
    next_action: str = "Review literature and choose a focused question."
    acceptance: str = "pending"
    scientific_outcome: str = "unassessed"
    max_experiments: int | None = 12


@dataclass(frozen=True)
class ResearchRecord:
    id: str
    study_id: str
    kind: str
    text: str
    refs: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=now)


@dataclass(frozen=True)
class ExperimentSpec:
    purpose: str
    source_dir: str
    files: tuple[str, ...]
    entrypoint: str
    protocol: str
    data_version: str
    args: tuple[str, ...] = ()
    seed: int = 0
    timeout_seconds: int | None = 60
    parent_id: str | None = None
    check_entrypoint: str | None = None
    metrics_path: str = "outputs/metrics.json"
    module: bool = False
    check_runner: str = "auto"


@dataclass(frozen=True)
class Job:
    id: str
    study_id: str
    plan_version: int
    request_key: str
    fingerprint: str
    spec: dict[str, Any]
    manifest: dict[str, str]
    status: str = "queued"
    worker_id: str | None = None
    result: dict[str, Any] | None = None
    created_at: str = field(default_factory=now)
    validity: str = "unverified"
    checks: list[dict[str, Any]] = field(default_factory=list)
    invalidated_reason: str = ""


@dataclass(frozen=True)
class ExecutionResult:
    status: str
    returncode: int | None
    metrics: dict[str, Any]
    artifacts: dict[str, str]
    error: str = ""


@dataclass(frozen=True)
class AgentStep:
    """A proposed bounded action, tied to the exact snapshot used by the agent."""

    kind: str
    based_on_revision: int
    payload: dict[str, Any]


def study_from(data: dict[str, Any]) -> Study:
    return Study(**data)


def job_from(data: dict[str, Any]) -> Job:
    return Job(**data)


def document(value: Any) -> dict[str, Any]:
    return asdict(value)
