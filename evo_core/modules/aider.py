"""Aider CLI adapter. Aider owns editing/context; Harness owns review and execution."""
from contextlib import contextmanager
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from uuid import uuid4

import httpx
import psutil
import yaml

from .local import LocalCoding
from ..llm import redact_secret
from ..runtime.host import CURRENT_CALL, call_scope
from ..system.domain import ExperimentSpec
from ..system.storage.artifacts import relative_path

_IGNORE = {".git", ".research_modules", ".bg_processes", ".pytest_tmp", ".test_tmp",
           "__pycache__", ".venv", "venv", "results", "outputs", "figures", "archive_invalid"}
_SOURCE_SUFFIXES = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".csv"}
HANDOFF = "coder_handoff.json"


def _source_file(root, name):
    path = root / relative_path(name)
    if (not path.resolve().is_relative_to(root.resolve()) or path.is_symlink()
            or any(part in _IGNORE or part.startswith(".") for part in Path(name).parts)
            or name == HANDOFF or path.suffix.lower() not in _SOURCE_SUFFIXES):
        raise ValueError(f"Not a publishable source file: {name}")
    return path


@contextmanager
def model_relay(ctx):
    """Forward non-streaming OpenAI requests; record actual SDK calls and provider usage.

    The child receives a local token. Upstream credentials stay in this process.
    This does not implement a model/agent loop or alter the provider's response.
    """
    key, base = ctx.config.resolved_api_credentials()
    token = uuid4().hex
    parent = CURRENT_CALL.get() or ctx.host.identity
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            if self.path != "/v1/chat/completions" or self.headers.get("Authorization") != f"Bearer {token}":
                self.send_error(403)
                return
            identity = ctx.host.identify("subagent:coder")
            # Explicitly preserve the invoking tool as the parent across the HTTP thread.
            from dataclasses import replace
            identity = replace(identity, parent_call_id=parent.call_id or None)
            with call_scope(identity):
                started = time.perf_counter()
                try:
                    body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                    if body.get("stream"):
                        raise ValueError("Aider relay requires --no-stream")
                    req = ctx.host.artifacts.put(json.dumps(body, ensure_ascii=False).encode())
                    ctx.emit_event("coder_api_start", backend="aider", identity=asdict(identity), request_ref=asdict(req))
                    with httpx.Client(timeout=ctx.config.request_timeout) as client:
                        response = client.post(base.rstrip("/") + "/chat/completions", json=body,
                                               headers={"Authorization": f"Bearer {key}"})
                    raw = response.content
                    safe = redact_secret(raw.decode("utf-8", errors="replace"), key)
                    ref = ctx.host.artifacts.put(safe.encode())
                    elapsed = round((time.perf_counter() - started) * 1000, 2)
                    if response.is_success:
                        usage = response.json().get("usage") or {}
                        cached = usage.get("prompt_cache_hit_tokens") or (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
                        ctx.host.models.report_sdk_call(component="aider", input_tokens=usage.get("prompt_tokens", 0),
                            output_tokens=usage.get("completion_tokens", 0), cached_tokens=cached,
                            duration_ms=elapsed, request_ref=req, response_ref=ref)
                    requests.append({"status": response.status_code, "duration_ms": elapsed})
                    ctx.emit_event("coder_api_end", backend="aider", identity=asdict(identity),
                                   status_code=response.status_code, duration_ms=elapsed, response_ref=asdict(ref))
                    self.send_response(response.status_code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                except Exception as exc:
                    message = redact_secret(str(exc), key)
                    ctx.emit_event("coder_api_error", backend="aider", identity=asdict(identity), error=message)
                    payload = json.dumps({"error": {"message": message, "type": "relay_error"}}).encode()
                    self.send_response(502)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", token, requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class AiderCoding(LocalCoding):
    def __init__(self, root, options):
        super().__init__(root, require_review=options.get("require_review", True))
        self.options = dict(options)
        self.command = list(options.get("command") or ["aider"])
        if not self.command or not shutil.which(self.command[0]):
            raise ValueError("Aider CLI is not installed; install aider-chat or configure coding.options.command")

    def system_prompt_section(self, state, ctx):
        return (
            "# Aider coding module\n"
            "Use create_coding_task and implement_coding_task to send implementation requests to the real Aider CLI. "
            "Aider has its own task-specific chat history and edits a staged project copy. "
            "The adapter freezes the returned files and exact run specification as an unapproved submission. "
            "Inspect it and approve through review_code_submission before submit_experiment. "
            "Include relevant file paths and desired behavior in the request. Give revisions through the same task_id. "
            "All result data and figures must be inside the same top-level output directory as metrics_path; "
            "the Harness archives that directory. Test/check capabilities are supplied by their configured modules."
        )

    def implement(self, task_id, instruction="", *, ctx=None):
        if ctx is None:
            raise ValueError("Aider implementation requires an active module runtime")
        task = self.docs.get(task_id)
        if task["kind"] != "coding_task":
            raise ValueError("Expected coding task")
        workspace = Path(ctx.workspace).resolve()
        session = self.root / "aider_sessions" / task_id
        attempt = session / uuid4().hex[:12]
        tree = attempt / "worktree"
        tree.mkdir(parents=True)
        for source in workspace.rglob("*"):
            name = source.relative_to(workspace).as_posix()
            if not source.is_file() or source.is_symlink():
                continue
            try:
                _source_file(workspace, name)
            except ValueError:
                continue
            target = tree / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        # Aider's repo map works on tracked source without touching the user's repository.
        flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
        subprocess.run(["git", "init", "-q"], cwd=tree, check=True, capture_output=True, **flags)
        subprocess.run(["git", "add", "--all"], cwd=tree, check=True, capture_output=True, **flags)
        template = {"files": [], "test_paths": [], "run": {
            "purpose": "Describe the experiment", "entrypoint": "src/run.py", "protocol": "Describe exact protocol",
            "data_version": "dataset-version", "args": [], "seed": 0,
            "timeout_seconds": ctx.config.plugins.harness.experiment_timeout_seconds,
            "parent_id": None, "check_entrypoint": None, "metrics_path": "outputs/metrics.json",
            "module": False, "check_runner": "auto"}}
        prompt = (
            "Implement this researcher request. Work only within this staged project. Do not run experiments, "
            "install packages, or approve code. Harness performs execution after researcher review. "
            "Use Aider edits to create/update code and the JSON handoff file. Do not merely describe a patch.\n\n"
            f"TASK:\n{task['text']}\n\nREVISION:\n{instruction}\n\n"
            f"Write {HANDOFF} as valid JSON following this exact structure (replace illustrative values):\n"
            + json.dumps(template, ensure_ascii=False, indent=2)
            + "\nfiles must include every runtime dependency and optional unit/check file, using relative POSIX paths. "
            "Do not list the handoff itself. run supports only the keys shown; do not use command/cwd/timeout_s. "
            "entrypoint must be included in files. test_paths can be empty if no unit tests were requested. "
            "All raw results, datasets generated during execution, tables, analysis and figures must be written "
            "under the same top-level directory as metrics_path, normally outputs/. The checker runs later in a "
            "separate restored snapshot and sees only archived outputs, not the original execution working directory. "
            "The default checker receives no CLI arguments, so align its default paths with the submitted run.\n"
        )
        (attempt / "request.md").write_text(prompt, encoding="utf-8")
        # Explicit config prevents home/cwd Aider configuration from changing the contract.
        (attempt / "config.yml").write_text("{}\n", encoding="utf-8")
        (attempt / "empty.env").write_text("", encoding="utf-8")
        model = self.options.get("model") or ctx.config.effective_aux_model()
        model = model if "/" in model else "openai/" + model
        extra = {}
        if ctx.config.thinking in {"enabled", "disabled"} and "deepseek" in model:
            extra["extra_body"] = {"thinking": {"type": ctx.config.thinking}}
            if ctx.config.thinking == "enabled":
                extra["reasoning_effort"] = ctx.config.reasoning_effort
        settings = [{"name": model, "edit_format": self.options.get("edit_format", "diff"),
                     "weak_model_name": model, "use_repo_map": True, "extra_params": extra}]
        (attempt / "model-settings.yml").write_text(yaml.safe_dump(settings), encoding="utf-8")
        (tree / HANDOFF).write_text(json.dumps(template, indent=2), encoding="utf-8")
        command = [*self.command, "--model", model, "--weak-model", model,
                   "--model-settings-file", str(attempt / "model-settings.yml"),
                   "--config", str(attempt / "config.yml"), "--env-file", str(attempt / "empty.env"),
                   "--message-file", str(attempt / "request.md"),
                   "--chat-history-file", str(session / "chat.md"), "--restore-chat-history",
                   "--input-history-file", str(session / "input.history"),
                   "--llm-history-file", str(attempt / "llm.log"),
                   "--no-auto-commits", "--no-dirty-commits", "--no-auto-lint", "--no-auto-test",
                   "--no-suggest-shell-commands", "--no-detect-urls", "--no-check-update",
                   "--no-show-release-notes", "--no-analytics", "--no-pretty", "--no-stream",
                   "--no-show-model-warnings", "--no-check-model-accepts-settings", "--yes-always",
                   "--map-tokens", str(self.options.get("map_tokens", 2048)),
                   "--max-chat-history-tokens", str(self.options.get("max_chat_history_tokens", 24000)),
                   "--encoding", "utf-8",
                   "--file", HANDOFF]
        # Existing code is editable context; the research documents remain read-only.
        for path in sorted(tree.rglob("*.py")):
            command.extend(("--file", path.relative_to(tree).as_posix()))
        for name in ("research_request.md", "literature_notes.md"):
            if (tree / name).is_file():
                command.extend(("--read", name))
        log = attempt / "stdout.log"
        with ctx.host.models.scope("subagent:coder"), model_relay(ctx) as (base, token, requests):
            env = {k: v for k, v in os.environ.items() if not k.startswith("AIDER_")}
            for name in ("CUSTOM_OPENAI_API_KEY", "CUSTOM_OPENAI_BASE_URL", "OPENAI_BASE_URL"):
                env.pop(name, None)
            env.update(OPENAI_API_KEY=token, OPENAI_API_BASE=base, AIDER_OPENAI_API_BASE=base,
                       PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
                       GIT_TERMINAL_PROMPT="0", LITELLM_LOCAL_MODEL_COST_MAP="True")
            started = time.monotonic()
            ctx.emit_event("coder_backend_start", backend="aider", task_id=task_id, model=model,
                           worktree=str(tree), log_path=str(log))
            with log.open("wb") as out:
                process = subprocess.Popen(command, cwd=tree, env=env, stdin=subprocess.DEVNULL,
                                           stdout=out, stderr=subprocess.STDOUT, **flags)
                try:
                    heartbeat = started
                    while process.poll() is None:
                        ctx.host.cancellation.check()
                        limit = self.options.get("timeout_seconds")
                        if limit is not None and time.monotonic() - started > limit:
                            raise TimeoutError("Aider implementation timed out")
                        if time.monotonic() - heartbeat >= 10:
                            ctx.emit_event("coder_backend_progress", backend="aider", task_id=task_id,
                                elapsed_seconds=round(time.monotonic()-started, 1), log_bytes=log.stat().st_size,
                                completed_api_requests=len(requests))
                            heartbeat = time.monotonic()
                        time.sleep(0.25)
                finally:
                    if process.poll() is None:
                        # Windows script launchers can own a second interpreter process.
                        try:
                            parent_process = psutil.Process(process.pid)
                            for child in reversed(parent_process.children(recursive=True)):
                                try:
                                    child.kill()
                                except psutil.NoSuchProcess:
                                    pass
                            parent_process.kill()
                        except psutil.NoSuchProcess:
                            pass
                        process.wait(timeout=10)
            ctx.emit_event("coder_backend_end", backend="aider", task_id=task_id,
                           returncode=process.returncode, api_requests=len(requests), log_path=str(log))
        output = log.read_text(encoding="utf-8", errors="replace")
        if process.returncode:
            raise RuntimeError(f"Aider failed ({process.returncode}); log: {log}\n{output[-3000:]}")
        payload = json.loads((tree / HANDOFF).read_text(encoding="utf-8"))
        files = payload["files"]
        if not isinstance(files, list) or not files or len(set(files)) != len(files):
            raise ValueError(f"Aider did not provide a nonempty unique files list; log: {log}")
        run = dict(payload["run"])
        spec = ExperimentSpec(source_dir=str(tree), files=tuple(files), **run)
        if spec.entrypoint not in files or (spec.check_entrypoint and spec.check_entrypoint not in files):
            raise ValueError("Aider handoff omitted the entrypoint or checker from files")
        for name in files:
            _source_file(tree, name).read_bytes()
            _source_file(workspace, name)
        canonical = asdict(spec)
        canonical.pop("source_dir")
        canonical.pop("files")
        # Freeze before publishing. Publication is implementation, never researcher approval.
        submission = self.submit(task_id, str(tree), files, canonical, actor="subagent:coder",
                                 test_paths=payload.get("test_paths") or None)
        for name in files:
            destination = _source_file(workspace, name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((tree / name).read_bytes())
            if ctx.observe_file and destination.suffix.lower() in {".md", ".txt"}:
                ctx.observe_file(name, destination.read_text(encoding="utf-8"))
        return {"task_id": task_id, "backend": "aider", "submissions": [submission["id"]],
                "output": output[-6000:], "log_path": str(log), "handoff_path": str(tree / HANDOFF)}


def coding_factory(ctx, registry, options):
    return AiderCoding(ctx.root / "coding", options)
