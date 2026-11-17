"""Explicit local Python executor for trusted development code, not a container sandbox.

One run_once call claims at most one job. No background thread is implicitly started.
The existing research-harness Docker executor will be a different adapter.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, wait as wait_futures
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any, Callable
from uuid import uuid4

from ..domain import ExecutionResult, Job, document, encode
from ..storage.artifacts import ArtifactStore, relative_path
from ..storage.sqlite import SqliteResearchStore


class LocalPythonWorker:
    def __init__(
        self,
        store: SqliteResearchStore,
        artifacts: ArtifactStore,
        root: Path,
        worker_id: str,
        *,
        python_executable: str | None = None,
    ):
        self.store, self.artifacts = store, artifacts
        self.root, self.worker_id = root.resolve(), worker_id
        self.python_executable = python_executable or sys.executable
        self.root.mkdir(parents=True, exist_ok=True)

    def run_once(self, study_id: str | None = None) -> Job | None:
        job = self.store.claim(self.worker_id, study_id)
        if job is None:
            return None
        try:
            result = self.execute(job)
        except Exception as exc:
            result = ExecutionResult("failed", None, {}, {}, f"{type(exc).__name__}: {exc}")
        finished = self.store.finish_job(job.id, self.worker_id, document(result))
        if finished.status == "succeeded" and finished.spec.get("check_entrypoint"):
            check_rec = self.run_check_script(finished, finished.spec["check_entrypoint"],
                                              timeout_seconds=finished.spec["timeout_seconds"])
            finished = self.store.verify_job(finished.id, check_rec, self.worker_id)
        return finished

    def execute(self, job: Job) -> ExecutionResult:
        workspace = self.root / job.id
        if workspace.exists():
            workspace = self.root / f"{job.id}_{uuid4().hex[:6]}"
        self.artifacts.materialize(job.manifest, workspace)
        stdout_path, stderr_path = workspace / "stdout.log", workspace / "stderr.log"
        entry = ["-m", job.spec["entrypoint"][:-3].replace("/", ".")] if job.spec.get("module") else [job.spec["entrypoint"]]
        command = [self.python_executable, *entry, *job.spec["args"]]
        env = dict(
            os.environ,
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONIOENCODING="utf-8",
            PYTHONUTF8="1",
            MPLBACKEND="Agg",
            EVO_EXPERIMENT_SEED=str(job.spec["seed"]),
        )
        error, returncode = "", None
        with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
            try:
                process = subprocess.run(command, cwd=workspace, stdout=out, stderr=err, env=env,
                                         timeout=job.spec["timeout_seconds"], check=False)
                returncode = process.returncode
            except subprocess.TimeoutExpired:
                error = "Experiment timed out"
        artifacts = {"stdout.log": self.artifacts.put(stdout_path.read_bytes()),
                     "stderr.log": self.artifacts.put(stderr_path.read_bytes())}
        artifacts["execution.json"] = self.artifacts.put(encode({
            "python": sys.version, "executable": self.python_executable, "command": command,
            "seed": job.spec["seed"], "protocol": job.spec["protocol"],
            "data_version": job.spec["data_version"], "manifest": job.manifest,
        }).encode())
        metrics = {}
        if returncode == 0 and not error:
            try:
                metric_file = workspace / relative_path(job.spec.get("metrics_path", "outputs/metrics.json"))
                outputs = workspace / metric_file.relative_to(workspace).parts[0]
                if not metric_file.resolve().is_relative_to(workspace) or metric_file.is_symlink():
                    raise ValueError("Metric output escapes workspace")
                metrics = json.loads(metric_file.read_text(encoding="utf-8"))
                if not isinstance(metrics, dict) or not metrics:
                    raise ValueError("Declared metrics_path must contain a nonempty JSON object")
                encode(metrics)  # Reject NaN and Infinity; do not silently create invalid evidence.
                total = 0
                for path in sorted(outputs.rglob("*")):
                    if path.is_symlink() or not path.resolve().is_relative_to(workspace):
                        raise ValueError("Output symlinks are not supported")
                    if path.is_file():
                        data = path.read_bytes()
                        total += len(data)
                        if total > 40_000_000:
                            raise ValueError("Development output archive exceeds 40 MB")
                        artifacts[path.relative_to(workspace).as_posix()] = self.artifacts.put(data)
            except (OSError, ValueError) as exc:
                error = f"Invalid experiment output: {exc}"
                metrics = {}
        elif not error:
            error = f"Python exited with code {returncode}"
        return ExecutionResult("failed" if error else "succeeded", returncode, metrics, artifacts, error)

    def run_check_script(
        self,
        job: Job,
        check_entrypoint: str,
        *,
        extra_files: dict[str, bytes] | None = None,
        args: tuple[str, ...] = (),
        timeout_seconds: int | None = 60,
        runner: str | None = None,
    ) -> dict[str, Any]:
        rel_entry = relative_path(check_entrypoint)
        check_dir = self.root / f"{job.id}_check_{len(job.checks) + 1}"
        if check_dir.exists():
            check_dir = self.root / f"{job.id}_check_{len(job.checks) + 1}_{uuid4().hex[:6]}"
        self.artifacts.materialize(job.manifest, check_dir)

        result = job.result or {}
        archived = result.get("artifacts") or {}
        for rel_name, blob_hash in archived.items():
            if not rel_name.startswith(("outputs/", "results/")):
                continue
            target = check_dir / relative_path(rel_name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self.artifacts.read(blob_hash))

        inputs_dir = check_dir / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        targets_payload = {
            "job_id": job.id,
            "study_id": job.study_id,
            "status": job.status,
            "spec": job.spec,
            "metrics": result.get("metrics", {}),
            "artifacts": [
                {"path": name, "archive_hash": key}
                for name, key in archived.items()
            ],
        }
        (inputs_dir / "check_targets.json").write_text(encode(targets_payload), encoding="utf-8")

        check_manifest = dict(job.manifest)
        for rel_name, raw_bytes in (extra_files or {}).items():
            if relative_path(rel_name).parts[0] in {"outputs", "results", "inputs"}:
                raise ValueError("Checker source files cannot overwrite archived evidence")
            target = check_dir / relative_path(rel_name)
            if not target.resolve().is_relative_to(check_dir):
                raise ValueError(f"Check file escapes workspace: {rel_name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw_bytes)
            check_manifest[rel_name] = self.artifacts.put(raw_bytes)

        entry_path = check_dir / rel_entry
        if not entry_path.is_file():
            return {
                "entrypoint": check_entrypoint,
                "passed": False,
                "checks": [],
                "errors": [f"Missing check entrypoint: {check_entrypoint}"],
                "returncode": None,
            }

        checks_file = check_dir / "outputs" / "checks.json"
        checks_file.unlink(missing_ok=True)
        from .validation import check_runner, run_pytest
        selected_runner = check_runner(check_entrypoint, runner or (job.spec.get("check_runner", "auto")
                                      if check_entrypoint == job.spec.get("check_entrypoint") else "auto"))
        if selected_runner == "pytest":
            receipt = run_pytest(self.python_executable, check_dir, targets=(check_entrypoint, *args), timeout=timeout_seconds)
            checks_file.parent.mkdir(parents=True, exist_ok=True)
            checks_file.write_text(encode({"checks": receipt["checks"]}), encoding="utf-8")
            return {**{k: v for k, v in receipt.items() if k not in {"stdout", "stderr"}},
                    "entrypoint": check_entrypoint, "source_manifest": check_manifest, "args": list(args),
                    "stdout": receipt["stdout"][-2000:], "stderr": receipt["stderr"][-2000:],
                    "artifacts": {"stdout.log": self.artifacts.put(receipt["stdout"].encode()),
                                  "stderr.log": self.artifacts.put(receipt["stderr"].encode()),
                                  "checks.json": self.artifacts.put(checks_file.read_bytes()),
                                  **({"pytest.xml": self.artifacts.put((check_dir / "outputs/pytest.xml").read_bytes())}
                                     if (check_dir / "outputs/pytest.xml").is_file() else {})}}
        cmd = [self.python_executable, check_entrypoint, *args]
        env = dict(
            os.environ,
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONIOENCODING="utf-8",
            PYTHONUTF8="1",
            MPLBACKEND="Agg",
        )
        errors: list[str] = []
        returncode: int | None = None
        stdout = ""
        stderr = ""
        try:
            proc = subprocess.run(
                cmd,
                cwd=check_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                timeout=timeout_seconds,
                check=False,
            )
            returncode = proc.returncode
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
        except subprocess.TimeoutExpired:
            errors.append(f"Check script timed out after {timeout_seconds}s")

        checks: list[dict[str, Any]] = []
        checks_file = check_dir / "outputs" / "checks.json"
        try:
            if not checks_file.resolve().is_relative_to(check_dir) or checks_file.is_symlink():
                raise ValueError("outputs/checks.json escapes workspace or is a symlink")
            raw_text = checks_file.read_text(encoding="utf-8")
            doc = json.loads(raw_text)
            encode(doc)
            raw_checks = doc.get("checks")
            if not isinstance(raw_checks, list) or not raw_checks:
                raise ValueError("outputs/checks.json must contain a non-empty 'checks' list")
            for item in raw_checks:
                if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"].strip():
                    raise ValueError("Each check entry must have a non-empty string 'name'")
                if type(item.get("passed")) is not bool:
                    raise ValueError(f"Check '{item.get('name')}' must have boolean 'passed'")
                checks.append(item)
            failed = [item["name"] for item in checks if not item["passed"]]
            if failed:
                errors.append("Failed checks: " + ", ".join(failed))
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f"Invalid or missing outputs/checks.json: {exc}")
            checks = []

        if returncode != 0 and not any("timed out" in e for e in errors):
            errors.append(f"Check script exited with code {returncode}")

        return {
            "runner": "checks_json",
            "entrypoint": check_entrypoint,
            "source_manifest": check_manifest,
            "args": list(args),
            "artifacts": {"stdout.log": self.artifacts.put(stdout.encode("utf-8")),
                          "stderr.log": self.artifacts.put(stderr.encode("utf-8")),
                          **({"checks.json": self.artifacts.put(checks_file.read_bytes())}
                             if checks_file.is_file() and not checks_file.is_symlink() else {})},
            "passed": returncode == 0 and not errors and bool(checks),
            "checks": checks,
            "errors": errors,
            "returncode": returncode,
            "stdout": stdout[-2000:],
            "stderr": stderr[-2000:],
        }

    def verify_job(
        self,
        job_id: str,
        check_entrypoint: str,
        *,
        source_dir: Path | None = None,
        files: tuple[str, ...] = (),
        args: tuple[str, ...] = (),
        timeout_seconds: int = 60,
        runner: str | None = None,
    ) -> Job:
        job = self.store.get_job(job_id)
        extra_files: dict[str, bytes] = {}
        if source_dir is not None:
            src_root = Path(source_dir).resolve()
            wanted = list(files)
            if check_entrypoint not in wanted:
                wanted.append(check_entrypoint)
            for name in wanted:
                rel = relative_path(name)
                candidate = src_root / rel
                if candidate.is_file():
                    if candidate.is_symlink() or not candidate.resolve().is_relative_to(src_root):
                        raise ValueError(f"Check file escapes source directory: {name}")
                    extra_files[name] = candidate.read_bytes()
        check_rec = self.run_check_script(
            job,
            check_entrypoint,
            extra_files=extra_files,
            args=args,
            timeout_seconds=timeout_seconds,
            runner=runner,
        )
        return self.store.verify_job(job.id, check_rec, self.worker_id)


class ParallelWorkerPool:
    """Thread-pooled worker coordinator for parallel frozen experiment execution."""

    def __init__(
        self,
        store: SqliteResearchStore,
        artifacts: ArtifactStore,
        root: Path,
        max_workers: int = 2,
        *,
        worker_factory: Callable[[str], Any] | None = None,
        python_executable: str | None = None,
    ):
        self.store = store
        self.artifacts = artifacts
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_workers = max(1, int(max_workers))
        self.worker_factory = worker_factory
        self.python_executable = python_executable
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="evo-worker")
        self._lock = threading.Lock()
        self._futures: list[Future[Job | None]] = []
        self._worker_seq = 0

    def _make_worker(self) -> Any:
        with self._lock:
            self._worker_seq += 1
            wid = f"pool-worker-{self._worker_seq}"
        if self.worker_factory is not None:
            return self.worker_factory(wid)
        return LocalPythonWorker(
            self.store,
            self.artifacts,
            self.root,
            wid,
            python_executable=self.python_executable,
        )

    def submit_async(self, study_id: str | None = None) -> Future[Job | None]:
        worker = self._make_worker()
        fut = self._executor.submit(worker.run_once, study_id)
        with self._lock:
            self._futures.append(fut)
        return fut

    def wait_all(self, timeout: float | None = None) -> list[Job]:
        with self._lock:
            pending = [f for f in self._futures]
        if not pending:
            return []
        done, not_done = wait_futures(pending, timeout=timeout)
        results: list[Job] = []
        for fut in done:
            res = fut.result()
            if res is not None:
                results.append(res)
        with self._lock:
            self._futures = [f for f in not_done]
        return results

    def drain(self, study_id: str | None = None, max_jobs: int = 32) -> list[Job]:
        completed: list[Job] = []
        lock = threading.Lock()
        remaining = [max(1, int(max_jobs))]

        def run_slot() -> None:
            worker = self._make_worker()
            while True:
                with lock:
                    if remaining[0] <= 0:
                        return
                    remaining[0] -= 1
                job = worker.run_once(study_id)
                if job is None:
                    return
                with lock:
                    completed.append(job)

        slot_futs = [self._executor.submit(run_slot) for _ in range(self.max_workers)]
        for fut in slot_futs:
            fut.result()
        return completed

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)
