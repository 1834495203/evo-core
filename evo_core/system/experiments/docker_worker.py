"""Optional Docker container experiment worker with safe local fallback."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any
from uuid import uuid4

from ..domain import ExecutionResult, Job, document, encode
from ..storage.artifacts import ArtifactStore, relative_path
from ..storage.sqlite import SqliteResearchStore
from .worker import LocalPythonWorker


class DockerExperimentWorker:
    """Executes frozen experiments in a network-isolated Docker container when available."""

    def __init__(
        self,
        store: SqliteResearchStore,
        artifacts: ArtifactStore,
        root: Path,
        worker_id: str,
        *,
        image: str = "research-harness-sandbox:0.3",
        cpus: str = "2",
        memory_mb: int = 2048,
        fallback_to_local: bool = True,
        python_executable: str | None = None,
    ) -> None:
        self.store = store
        self.artifacts = artifacts
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.worker_id = worker_id
        self.image = image
        self.cpus = cpus
        self.memory_mb = memory_mb
        self.fallback_to_local = fallback_to_local
        self._local = LocalPythonWorker(
            store,
            artifacts,
            self.root,
            worker_id,
            python_executable=python_executable,
        )

    def is_docker_available(self) -> bool:
        try:
            proc = subprocess.run(
                ["docker", "image", "inspect", self.image, "--format", "{{.Id}}"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            return proc.returncode == 0 and bool(proc.stdout.strip())
        except Exception:
            return False

    def run_once(self, study_id: str | None = None) -> Job | None:
        job = self.store.claim(self.worker_id, study_id)
        if job is None:
            return None
        try:
            if self.is_docker_available():
                result = self.execute_in_docker(job)
            elif self.fallback_to_local:
                result = self._local.execute(job)
                meta = {
                    "executor": "local_fallback",
                    "requested_image": self.image,
                    "seed": job.spec["seed"],
                }
                result = ExecutionResult(
                    status=result.status,
                    returncode=result.returncode,
                    metrics=result.metrics,
                    artifacts={**result.artifacts, "executor.json": self.artifacts.put(encode(meta).encode())},
                    error=result.error,
                )
            else:
                result = ExecutionResult("failed", None, {}, {}, f"Docker image unavailable: {self.image}")
        except Exception as exc:
            result = ExecutionResult("failed", None, {}, {}, f"{type(exc).__name__}: {exc}")

        finished = self.store.finish_job(job.id, self.worker_id, document(result))
        if finished.status == "succeeded" and finished.spec.get("check_entrypoint"):
            check_rec = self._local.run_check_script(finished, finished.spec["check_entrypoint"])
            finished = self.store.verify_job(finished.id, check_rec, self.worker_id)
        return finished

    def execute_in_docker(self, job: Job) -> ExecutionResult:
        workspace = self.root / job.id
        if workspace.exists():
            workspace = self.root / f"{job.id}_{uuid4().hex[:6]}"
        self.artifacts.materialize(job.manifest, workspace)
        stdout_path, stderr_path = workspace / "stdout.log", workspace / "stderr.log"
        name = f"evo-exp-{uuid4().hex[:10]}"
        cmd = [
            "docker",
            "run",
            "--rm",
            "--init",
            "--name",
            name,
            "--network=none",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            f"--cpus={self.cpus}",
            f"--memory={self.memory_mb}m",
            f"--memory-swap={self.memory_mb}m",
            "--pids-limit=64",
            "-e",
            f"EVO_EXPERIMENT_SEED={job.spec['seed']}",
            "-e",
            "MPLBACKEND=Agg",
            "--mount",
            f"type=bind,source={workspace.resolve()},target=/workspace",
            "-w",
            "/workspace",
            "--entrypoint",
            "python",
            self.image,
            *(["-m", job.spec["entrypoint"][:-3].replace("/", ".")] if job.spec.get("module") else [job.spec["entrypoint"]]),
            *job.spec["args"],
        ]
        error = ""
        returncode: int | None = None
        try:
            with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
                proc = subprocess.run(
                    cmd,
                    stdout=out,
                    stderr=err,
                    timeout=job.spec["timeout_seconds"],
                    check=False,
                )
                returncode = proc.returncode
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=15, check=False)
            error = "Docker experiment timed out"

        artifacts = {
            "stdout.log": self.artifacts.put(stdout_path.read_bytes()),
            "stderr.log": self.artifacts.put(stderr_path.read_bytes()),
            "executor.json": self.artifacts.put(
                encode({"executor": "docker", "image": self.image, "command": cmd}).encode()
            ),
        }
        metrics: dict[str, Any] = {}
        if returncode == 0 and not error:
            try:
                metric_file = workspace / relative_path(job.spec.get("metrics_path", "outputs/metrics.json"))
                outputs = workspace / metric_file.relative_to(workspace).parts[0]
                if not metric_file.resolve().is_relative_to(workspace) or metric_file.is_symlink():
                    raise ValueError("Metric output escapes workspace")
                metrics = json.loads(metric_file.read_text(encoding="utf-8"))
                if not isinstance(metrics, dict) or not metrics:
                    raise ValueError("outputs/metrics.json must be a nonempty JSON object")
                encode(metrics)
                for path in sorted(outputs.rglob("*")):
                    if path.is_symlink() or not path.resolve().is_relative_to(workspace):
                        raise ValueError("Output symlinks are not supported")
                    if path.is_file():
                        artifacts[path.relative_to(workspace).as_posix()] = self.artifacts.put(path.read_bytes())
            except (OSError, ValueError) as exc:
                error = f"Invalid experiment output: {exc}"
                metrics = {}
        elif not error:
            error = f"Docker exited with code {returncode}"
        return ExecutionResult("failed" if error else "succeeded", returncode, metrics, artifacts, error)
