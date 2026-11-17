"""Workspace-confined filesystem, cross-platform shell execution, and background process tracking."""

from __future__ import annotations

import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

BLOCKED_COMMANDS = frozenset(
    {
        "sudo",
        "chmod",
        "chown",
        "mkfs",
        "dd",
        "shutdown",
        "reboot",
        "pkill",
        "killall",
    }
)
PIPE_NETWORK_RHS = frozenset(
    {"nc", "ncat", "netcat", "ssh", "curl", "wget", "telnet", "socat", "scp", "rsync"}
)
PIPE_SHELL_RHS = frozenset(
    {"sh", "bash", "zsh", "dash", "python", "python3", "node", "powershell", "pwsh"}
)
_DESTRUCTIVE_RE = [
    re.compile(r"(?:^|\s)rm\s+-[^\s]*r[^\s]*f[^\s]*\s+/(\s|$)"),
    re.compile(r"(?:^|\s)rm\s+-[^\s]*r[^\s]*f[^\s]*\s+['\"]?[A-Za-z]:[/\\](\s|$)"),
]
_HOST_ESCAPE_PREFIXES = (
    "/Users/",
    "/home/",
    "/etc/",
    "/usr/",
    "/var/",
    "/root/",
    "/bin/",
    "/sbin/",
)


def _is_usable_bash(bash_path: Path) -> bool:
    """Verify bash.exe can initialize in the current OS/sandbox environment."""
    try:
        res = subprocess.run(
            [str(bash_path), "-c", "true"],
            capture_output=True,
            timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        return res.returncode == 0
    except Exception:
        return False


def discover_git_bash() -> Path | None:
    """Locate a working Git for Windows bash.exe when running on win32."""
    if sys.platform != "win32":
        return None
    candidates: list[Path] = []
    git_exe = shutil.which("git")
    if git_exe:
        candidates.append(Path(git_exe).resolve().parents[1] / "bin" / "bash.exe")
    candidates.extend(
        [
            Path("C:/Program Files/Git/bin/bash.exe"),
            Path("D:/software/git/Git/bin/bash.exe"),
        ]
    )
    for candidate in candidates:
        if candidate.is_file() and _is_usable_bash(candidate):
            return candidate
    return None


def discover_science_python_dir(configured: str | None = None) -> Path | None:
    """Locate an optional scientific Python Scripts/bin directory."""
    if configured:
        p = Path(configured).expanduser().resolve()
        if p.is_file():
            return p.parent
        if p.is_dir():
            return p
    learn_root = Path(__file__).resolve().parents[2]
    candidate = learn_root / "EvoScientist" / ".runtime" / "science-env" / (
        "Scripts" if sys.platform == "win32" else "bin"
    )
    if candidate.is_dir():
        return candidate
    return None


def _build_shell_env(science_python_dir: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "MPLBACKEND": "Agg",
            "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS", "2"),
            "OPENBLAS_NUM_THREADS": env.get("OPENBLAS_NUM_THREADS", "2"),
            "MKL_NUM_THREADS": env.get("MKL_NUM_THREADS", "2"),
            "MSYS_NO_PATHCONV": "1",
        }
    )
    if science_python_dir and science_python_dir.is_dir():
        env["PATH"] = str(science_python_dir) + os.pathsep + env.get("PATH", "")
        env["VIRTUAL_ENV"] = str(science_python_dir.parent)
    return env


def validate_command(command: str, *, dangerous: bool = False) -> str | None:
    """Return an error message if *command* violates sandbox safety checks, else None."""
    text = command.strip()
    if not text:
        return "Command rejected: empty command."

    for pattern in _DESTRUCTIVE_RE:
        if pattern.search(text):
            return "Command blocked: recursive delete of root filesystem."

    segments = re.split(r"\|\||&&|[;\n]", text)
    for segment in segments:
        pipes = segment.split("|")
        for idx, stage in enumerate(pipes):
            stage = stage.strip()
            if not stage:
                continue
            try:
                tokens = shlex.split(stage, posix=True)
            except ValueError:
                tokens = stage.split()
            if not tokens:
                continue
            cmd = Path(tokens[0]).name.lower()
            if cmd in BLOCKED_COMMANDS:
                return f"Command blocked: '{cmd}' is disallowed in the sandbox."
            if idx > 0 and (cmd in PIPE_SHELL_RHS or cmd in PIPE_NETWORK_RHS):
                prev_tokens = pipes[idx - 1].strip().split()
                prev_cmd = Path(prev_tokens[0]).name.lower() if prev_tokens else ""
                if prev_cmd in {"curl", "wget"} and cmd in PIPE_SHELL_RHS:
                    return f"Command blocked: piping download directly into '{cmd}'."
            if not dangerous:
                for tok in tokens[1:]:
                    if any(tok.startswith(prefix) for prefix in _HOST_ESCAPE_PREFIXES):
                        return (
                            f"Command blocked: host path '{tok}' outside workspace. "
                            "Use relative paths inside '.'."
                        )
    return None


def _terminate_tree(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                check=False,
                capture_output=True,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except OSError:
            pass


class WorkspaceSandbox:
    """Confined file access and shell execution inside *workspace*."""

    def __init__(
        self,
        workspace: str | Path,
        *,
        run_dir: str | Path | None = None,
        dangerous: bool = False,
        default_timeout: int = 180,
        max_output_chars: int = 24_000,
        science_python: str | None = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.run_dir = Path(run_dir).resolve() if run_dir else (self.workspace / ".evo_run")
        self.logs_dir = self.run_dir / "tool_logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.dangerous = dangerous
        self.default_timeout = default_timeout
        self.max_output_chars = max_output_chars
        self.git_bash = discover_git_bash()
        self.science_python_dir = discover_science_python_dir(science_python)
        self.env = _build_shell_env(self.science_python_dir)

    def resolve_path(self, rel_or_abs: str) -> Path:
        """Resolve a user/agent path strictly inside workspace unless dangerous=True."""
        raw = rel_or_abs.strip()
        if not raw or raw in {".", "./"}:
            return self.workspace
        # Normalize accidental leading '/' from deepagents habits like '/src/a.py'
        # into workspace-relative 'src/a.py' when not a Windows drive or existing root path.
        if raw.startswith("/") and not re.match(r"^/[A-Za-z]:", raw):
            raw = raw.lstrip("/")
        elif re.match(r"^/[A-Za-z]:", raw):
            raw = raw[1:]

        candidate = Path(raw)
        if not candidate.is_absolute():
            resolved = (self.workspace / candidate).resolve()
        else:
            resolved = candidate.resolve()

        if not self.dangerous:
            try:
                resolved.relative_to(self.workspace)
            except ValueError as exc:
                raise PermissionError(
                    f"Path '{rel_or_abs}' escapes workspace '{self.workspace}'. "
                    "Use workspace-relative paths such as 'src/exp.py'."
                ) from exc
        return resolved

    def rel_path(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.workspace).as_posix()
        except ValueError:
            return str(path)

    def spawn_shell(
        self,
        command: str,
        *,
        stdout_target: Any,
        stderr_target: Any,
    ) -> tuple[subprocess.Popen[str], Path | None]:
        """Spawn *command* in workspace using Git Bash on Windows (via temp script) or /bin/sh."""
        script_path: Path | None = None
        kwargs: dict[str, Any] = {
            "cwd": str(self.workspace),
            "env": self.env,
            "stdout": stdout_target,
            "stderr": stderr_target,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
        }
        if sys.platform == "win32":
            kwargs["creationflags"] = (
                subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            )
            if self.git_bash is not None:
                fd, tmp = tempfile.mkstemp(prefix="evo_cmd_", suffix=".sh", dir=str(self.logs_dir))
                os.close(fd)
                script_path = Path(tmp)
                script_path.write_text(command + "\n", encoding="utf-8", newline="\n")
                # Use relative or POSIX path for Git Bash
                posix_script = script_path.as_posix()
                proc = subprocess.Popen([str(self.git_bash), posix_script], **kwargs)
                return proc, script_path
            proc = subprocess.Popen(command, shell=True, **kwargs)
            return proc, None

        kwargs["start_new_session"] = True
        proc = subprocess.Popen(["/bin/sh", "-c", command], **kwargs)
        return proc, None

    def execute(self, command: str, timeout: int | None = None) -> tuple[int, str]:
        """Run *command* synchronously with timeout and output capping/offload."""
        err = validate_command(command, dangerous=self.dangerous)
        if err:
            return 1, err

        effective_timeout = max(1, min(3600, int(timeout or self.default_timeout)))
        proc, script_path = self.spawn_shell(
            command,
            stdout_target=subprocess.PIPE,
            stderr_target=subprocess.PIPE,
        )
        try:
            stdout, stderr = proc.communicate(timeout=effective_timeout)
            rc = proc.returncode or 0
        except subprocess.TimeoutExpired:
            _terminate_tree(proc)
            try:
                stdout, stderr = proc.communicate(timeout=2)
            except Exception:
                stdout, stderr = "", ""
            combined = self._format_output(stdout, stderr, rc=124, timed_out=effective_timeout)
            return 124, combined
        finally:
            if script_path and script_path.exists():
                script_path.unlink(missing_ok=True)

        return rc, self._format_output(stdout, stderr, rc=rc)

    def _format_output(
        self,
        stdout: str,
        stderr: str,
        *,
        rc: int,
        timed_out: int | None = None,
    ) -> str:
        parts: list[str] = []
        if stdout.strip():
            parts.append(stdout.rstrip())
        if stderr.strip():
            parts.append("[stderr]\n" + stderr.rstrip())
        body = "\n\n".join(parts) if parts else "(no output)"

        if len(body) > self.max_output_chars:
            log_file = self.logs_dir / f"exec_{uuid.uuid4().hex[:8]}.log"
            log_file.write_text(body, encoding="utf-8")
            head = body[: self.max_output_chars // 2]
            tail = body[-(self.max_output_chars // 2) :]
            body = (
                f"{head}\n\n... [truncated {len(body) - self.max_output_chars} chars; "
                f"full output saved to {log_file}] ...\n\n{tail}"
            )

        if timed_out is not None:
            return (
                f"{body}\n\n[Command timed out after {timed_out}s (exit=124). "
                "For long jobs, pass a larger timeout up to 3600 or use run_in_background.]"
            )
        status = "succeeded" if rc == 0 else f"failed with exit code {rc}"
        return f"{body}\n\n[Command {status}]"


@dataclass
class BgProcess:
    process_id: str
    name: str
    command: str
    popen: subprocess.Popen[str]
    pid: int
    log_path: Path
    started_ts: float
    script_path: Path | None = None
    returncode: int | None = None
    finished_ts: float | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class BackgroundProcessManager:
    """Track long-running detached experiment processes per workspace."""

    def __init__(self, sandbox: WorkspaceSandbox) -> None:
        self.sandbox = sandbox
        self._procs: dict[str, BgProcess] = {}
        self._lock = threading.Lock()
        self.bg_dir = self.sandbox.workspace / ".bg_processes"
        self.bg_dir.mkdir(parents=True, exist_ok=True)

    def launch(self, command: str, name: str | None = None) -> tuple[str | None, str]:
        err = validate_command(command, dangerous=self.sandbox.dangerous)
        if err:
            return None, err

        proc_id = uuid.uuid4().hex[:8]
        log_path = self.bg_dir / f"{proc_id}.log"
        log_handle = log_path.open("w", encoding="utf-8", errors="replace")
        try:
            popen, script_path = self.sandbox.spawn_shell(
                command,
                stdout_target=log_handle,
                stderr_target=subprocess.STDOUT,
            )
        except Exception:
            log_handle.close()
            raise

        record = BgProcess(
            process_id=proc_id,
            name=name or proc_id,
            command=command,
            popen=popen,
            pid=popen.pid,
            log_path=log_path,
            started_ts=time.time(),
            script_path=script_path,
        )
        with self._lock:
            self._procs[proc_id] = record

        def _watch() -> None:
            try:
                popen.wait()
            finally:
                log_handle.close()
                if script_path and script_path.exists():
                    script_path.unlink(missing_ok=True)
                with self._lock:
                    if record.returncode is None:
                        record.returncode = popen.poll()
                        record.finished_ts = time.time()

        threading.Thread(target=_watch, daemon=True).start()
        rel_log = self.sandbox.rel_path(log_path)
        return proc_id, (
            f"Started background process '{proc_id}' (name={record.name}, pid={popen.pid}). "
            f"Log: {rel_log}. Poll with check_process('{proc_id}')."
        )

    def _sync_status(self, rec: BgProcess) -> str:
        rc = rec.popen.poll()
        if rc is not None and rec.returncode is None:
            rec.returncode = rc
            rec.finished_ts = time.time()
        if rec.returncode is None:
            return "running"
        return "success" if rec.returncode == 0 else f"error(exit={rec.returncode})"

    @staticmethod
    def _read_tail(log_path: Path, max_bytes: int = 8192) -> str:
        try:
            with log_path.open("rb") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                if size == 0:
                    return "(no output yet)"
                if size > max_bytes:
                    fh.seek(-max_bytes, os.SEEK_END)
                    return "...(truncated)...\n" + fh.read().decode("utf-8", "replace")
                fh.seek(0)
                return fh.read().decode("utf-8", "replace")
        except OSError:
            return "(log unavailable)"

    def check(self, process_id: str) -> str:
        with self._lock:
            rec = self._procs.get(process_id)
            if rec is None:
                return f"Unknown process_id: {process_id}"
            status = self._sync_status(rec)
            end_ts = rec.finished_ts or time.time()
            elapsed = round(end_ts - rec.started_ts, 1)
        tail = self._read_tail(rec.log_path)
        return (
            f"Process {rec.process_id} ({rec.name}): status={status}, elapsed={elapsed}s\n"
            f"--- recent output ---\n{tail}"
        )

    def stop(self, process_id: str) -> str:
        with self._lock:
            rec = self._procs.get(process_id)
            if rec is None:
                return f"Unknown process_id: {process_id}"
        _terminate_tree(rec.popen)
        with self._lock:
            self._sync_status(rec)
        return f"Stopped process {process_id} ({rec.name})."

    def list_all(self) -> list[dict[str, object]]:
        with self._lock:
            rows = []
            for rec in self._procs.values():
                status = self._sync_status(rec)
                elapsed = round((rec.finished_ts or time.time()) - rec.started_ts, 1)
                rows.append(
                    {
                        "process_id": rec.process_id,
                        "name": rec.name,
                        "status": status,
                        "elapsed_s": elapsed,
                        "log": self.sandbox.rel_path(rec.log_path),
                    }
                )
            return rows

    def list_running(self) -> list[dict[str, object]]:
        return [r for r in self.list_all() if r["status"] == "running"]

    def cleanup_all(self) -> None:
        with self._lock:
            procs = list(self._procs.values())
        for rec in procs:
            if rec.popen.poll() is None:
                _terminate_tree(rec.popen)
            try:
                if psutil.pid_exists(rec.pid):
                    p = psutil.Process(rec.pid)
                    for child in p.children(recursive=True):
                        child.kill()
                    p.kill()
            except Exception:
                pass
