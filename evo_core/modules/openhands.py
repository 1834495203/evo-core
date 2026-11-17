"""OpenHands coding adapter: persistent SDK workspace, observed events, frozen handoff."""
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from uuid import uuid4

import psutil

from .local import LocalCoding
from ..protocol import ToolDef
from ..runtime.host import call_scope
from ..system.domain import ExperimentSpec
from ..system.storage.artifacts import relative_path

_IGNORE = {".git", ".research_modules", ".bg_processes", ".pytest_tmp", ".test_tmp", "__pycache__",
           ".venv", "venv", "results", "outputs", "figures", "archive_invalid"}
_SUFFIXES = {".py", ".md", ".txt", ".rst", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".csv", ".r", ".cpp", ".h"}


def source_path(root, name):
    target = root / relative_path(name)
    if (not target.resolve().is_relative_to(root.resolve()) or target.is_symlink()
        or any(p in _IGNORE or p.startswith(".") for p in Path(name).parts)
        or target.suffix.lower() not in _SUFFIXES):
        raise ValueError(f"Not a source path: {name}")
    return target


def inventory(root):
    result = {}
    if not root.exists():
        return result
    for directory, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in _IGNORE and not d.startswith(".")
                   and not (Path(directory) / d).is_symlink()]
        for name in files:
            relative = (Path(directory) / name).relative_to(root).as_posix()
            try:
                path = source_path(root, relative)
            except ValueError:
                continue
            result[relative] = sha256(path.read_bytes()).hexdigest()
    return result


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


class OpenHandsCoding(LocalCoding):
    def __init__(self, root, options):
        super().__init__(root, require_review=options.get("require_review", True))
        self.options = dict(options)
        relative_python = Path(".module_envs/openhands") / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        default = next((base / relative_python for base in Path(__file__).resolve().parents
                        if (base / relative_python).is_file()), Path(__file__).resolve().parents[2] / relative_python)
        self.python = Path(options.get("python_executable") or os.environ.get("EVO_OPENHANDS_PYTHON")
                           or default)
        if not self.python.is_file():
            raise ValueError("OpenHands environment missing: configure coding.options.python_executable or EVO_OPENHANDS_PYTHON")
        # SDK event filenames include UUIDs and atomic-write suffixes. Keep their
        # directory short enough for native Windows paths, including run snapshots.
        state_base = Path(options.get("state_root") or default.parents[3] / ".module_state/openhands")
        self.state_root = state_base / sha256(str(self.root.resolve()).encode()).hexdigest()[:12]
        self._lock = threading.Lock()

    def system_prompt_section(self, state, ctx):
        return (
            "# OpenHands coding module\n"
            "Use create_coding_task and implement_coding_task for coding. The real OpenHands SDK runs its own "
            "agent and persistent conversation in a task-specific directory. Reuse task_id for revisions, "
            "continuation and recovery; partial edits survive failures. Its file editor can write multiple "
            "files across tool calls, so request concrete implementation changes rather than a giant code response. "
            "After implementation, use prepare_coding_submission(task_id, run, test_paths) to freeze actual files "
            "for inspection and researcher review. You specify the exact run parameters, not shell commands. "
            "Library-only coding tasks can be completed without an executable entrypoint; prepare a submission "
            "when a runnable experiment is ready. Review the returned submission before submit_experiment. "
            "Coder does not execute experiments or approve code. All generated data/plots must share the "
            "top-level directory of metrics_path (normally outputs/), so Harness can archive and restore them."
        )

    def session(self, task_id, workspace):
        if self.docs.get(task_id)["kind"] != "coding_task":
            raise ValueError("Expected coding task")
        root = self.state_root / sha256(task_id.encode()).hexdigest()[:12]
        metadata = root / "session.json"
        tree = root / "workspace"
        if not metadata.exists():
            tree.mkdir(parents=True, exist_ok=True)
            baseline = inventory(workspace)
            for name in baseline:
                destination = source_path(tree, name)
                if destination.exists():
                    continue  # A partially initialized directory must never lose its edits.
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(workspace / name, destination)
            save_json(metadata, {"conversation_id": str(uuid4()), "baseline": baseline, "published": {},
                                 "task_id": task_id, "project": str(workspace.resolve())})
            save_json(self.root / "openhands_sessions" / task_id / "location.json", {"session_root": str(root)})
        data = json.loads(metadata.read_text(encoding="utf-8"))
        if data["project"] != str(workspace.resolve()):
            raise ValueError("Coding session belongs to a different workspace")
        return root, tree, data

    def changes(self, task_id, ctx, file_path=""):
        root, tree, data = self.session(task_id, Path(ctx.workspace))
        if file_path:
            path = source_path(tree, file_path)
            return {"task_id": task_id, "file_path": file_path, "content": path.read_text(encoding="utf-8"),
                    "hash": sha256(path.read_bytes()).hexdigest()}
        current = inventory(tree)
        baseline = data["baseline"]
        return {"task_id": task_id, "conversation_id": data["conversation_id"], "worktree": str(tree),
                "files": list(current), "changed_files": [n for n, h in current.items() if baseline.get(n) != h],
                "deleted_files": sorted(set(baseline) - set(current))}

    def implement(self, task_id, instruction="", *, ctx=None):
        if ctx is None:
            raise ValueError("OpenHands implementation needs an active runtime")
        if not self._lock.acquire(blocking=False):
            raise ValueError("Coding provider is busy; retry after the current task finishes")
        try:
            return self._implement(task_id, instruction, ctx)
        finally:
            self._lock.release()

    def _implement(self, task_id, instruction, ctx):
        root, tree, data = self.session(task_id, Path(ctx.workspace))
        attempt = root / "attempts" / uuid4().hex[:12]
        attempt.mkdir(parents=True)
        events_path, cancel_path = attempt / "events.jsonl", attempt / "cancel"
        key, base = ctx.config.resolved_api_credentials()
        model = self.options.get("model") or ctx.config.effective_aux_model()
        model = model if "/" in model else "openai/" + model
        llm = {"model": model, "api_key": key, "base_url": base, "usage_id": "coder", "api_mode": "chat",
               "max_output_tokens": self.options.get("max_output_tokens", ctx.config.max_output_tokens),
               "timeout": ctx.config.request_timeout, "num_retries": ctx.config.max_retries}
        if "deepseek" in model:
            llm["litellm_extra_body"] = {"thinking": {"type": ctx.config.thinking}}
            llm["reasoning_effort"] = ctx.config.reasoning_effort
        prompt = (
            f"Work only within this coding workspace: {tree}. Implement files with the file editor tool, "
            "using multiple tool calls as needed. All source files must be inside this directory. "
            "Do not run experiments, install packages, approve code, or request secrets. "
            "Harness will execute code after researcher review. Preserve completed edits on continuation. "
            "Finish with a concise factual summary of actual changes, unresolved issues and missing pieces. "
            "Do not print whole files in your final answer; write them using the file editor.\n\n"
            f"Original task:\n{self.docs.get(task_id)['text']}\n\nCurrent request:\n{instruction or 'Implement the task.'}"
        )
        request = {"workspace": str(tree), "conversation_id": data["conversation_id"],
                   "persistence_dir": str(root / "conversations"), "events_path": str(events_path),
                   "cancel_path": str(cancel_path), "llm": llm, "prompt": prompt,
                   "max_iterations": self.options.get("max_iterations")}
        (attempt / "request.md").write_text(prompt, encoding="utf-8")
        env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1",
                   LITELLM_LOCAL_MODEL_COST_MAP="True", OPENHANDS_SUPPRESS_BANNER="1",
                   OH_PERSISTENCE_DIR=str(root / "sdk_home"))
        flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
        identity = ctx.host.identify("subagent:coder")
        seen = 0
        final = None
        def collect():
            nonlocal seen, final
            if not events_path.exists():
                return
            lines = events_path.read_text(encoding="utf-8").splitlines()
            for line in lines[seen:]:
                try:
                    item = json.loads(line)
                except ValueError:
                    break
                seen += 1
                ref = ctx.host.artifacts.put(line.encode())
                if item["kind"] == "model":
                    log = item["log"]
                    usage = log.get("usage_summary") or {}
                    ctx.host.models.report_sdk_call(component="openhands", input_tokens=usage.get("prompt_tokens", 0),
                        output_tokens=usage.get("completion_tokens", 0), cached_tokens=usage.get("cache_read_tokens", 0),
                        duration_ms=max(0, log.get("latency_sec", 0) * 1000), request_ref=ref, response_ref=ref)
                elif item["kind"] in {"result", "error"}:
                    final = item
                event = item.get("event") or {}
                ctx.emit_event("coder_sdk_event", backend="openhands", task_id=task_id, record_kind=item["kind"],
                    sdk_event_type=event.get("kind"), tool=event.get("tool_name"), event_ref=asdict(ref),
                    preview=json.dumps(event.get("action") or event.get("observation") or item.get("message") or item.get("error") or "", ensure_ascii=False)[:600])

        log_path = attempt / "worker.log"
        with call_scope(identity), log_path.open("wb") as log:
            ctx.emit_event("coder_backend_start", backend="openhands", task_id=task_id,
                           conversation_id=data["conversation_id"], worktree=str(tree), log_path=str(log_path))
            process = subprocess.Popen([str(self.python), "-u", "-B", str(Path(__file__).with_name("openhands_worker.py"))],
                stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, cwd=tree, env=env, **flags)
            process.stdin.write(json.dumps(request).encode("utf-8"))
            process.stdin.close()
            started, heartbeat = time.monotonic(), time.monotonic()
            cancel_started = None
            try:
                while process.poll() is None:
                    collect()
                    if ctx.host.cancellation.requested():
                        cancel_path.touch(exist_ok=True)
                        cancel_started = cancel_started or time.monotonic()
                        if time.monotonic() - cancel_started > 3:
                            raise InterruptedError("OpenHands cancelled; partial workspace and SDK events retained")
                    limit = self.options.get("timeout_seconds")
                    if limit is not None and time.monotonic() - started > limit:
                        cancel_path.touch(exist_ok=True)
                        raise TimeoutError("OpenHands timed out; workspace retained")
                    if time.monotonic() - heartbeat >= 10:
                        ctx.emit_event("coder_backend_progress", backend="openhands", task_id=task_id,
                                       elapsed_seconds=round(time.monotonic() - started, 1), sdk_events=seen)
                        heartbeat = time.monotonic()
                    time.sleep(0.2)
            finally:
                if process.poll() is None:
                    try:
                        parent = psutil.Process(process.pid)
                        for child in parent.children(recursive=True):
                            try: child.kill()
                            except psutil.NoSuchProcess: pass
                        parent.kill()
                    except psutil.NoSuchProcess:
                        pass
                    process.wait(timeout=10)
                collect()
                ctx.emit_event("coder_backend_end", backend="openhands", task_id=task_id,
                               returncode=process.returncode, sdk_events=seen, log_path=str(log_path))
        if process.returncode or not final or final.get("kind") == "error" or final.get("status") != "finished":
            error = (final or {}).get("error") or log_path.read_text(encoding="utf-8", errors="replace")[-1800:]
            raise RuntimeError(f"OpenHands could not finish: {error}. Partial edits retained at {tree}; reuse task_id.")
        return {**self.changes(task_id, ctx), "backend": "openhands", "output": final.get("message", ""),
                "sdk_status": final.get("status"), "submissions": [], "log_path": str(log_path),
                "next_action": "Inspect changes; use prepare_coding_submission when a runnable version is ready."}

    def prepare(self, task_id, run, *, test_paths=None, ctx):
        root, tree, data = self.session(task_id, Path(ctx.workspace))
        files = [name for name in inventory(tree) if name not in {"research_request.md", "literature_notes.md"}]
        normalized = dict(run)
        normalized.setdefault("timeout_seconds", ctx.config.plugins.harness.experiment_timeout_seconds)
        spec = ExperimentSpec(source_dir=str(tree), files=tuple(files), **normalized)
        if spec.entrypoint not in files or (spec.check_entrypoint and spec.check_entrypoint not in files):
            raise ValueError("Run entrypoint and checker must exist in the actual coding workspace")
        current = inventory(Path(ctx.workspace))
        staged = inventory(tree)
        changed = [n for n in files if current.get(n) != staged[n]]
        for name in files:
            expected = data["published"].get(name, data["baseline"].get(name))
            if current.get(name) not in (expected, staged[name]):
                raise ValueError(f"Project file changed independently: {name}; merge before publishing")
        canonical = asdict(spec)
        canonical.pop("source_dir"); canonical.pop("files")
        rec = self.submit(task_id, str(tree), files, canonical, actor="subagent:coder", test_paths=test_paths)
        for name in changed:
            target = source_path(Path(ctx.workspace), name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((tree / name).read_bytes())
            data["published"][name] = staged[name]
            if ctx.observe_file and target.suffix in {".md", ".txt", ".rst"}:
                ctx.observe_file(name, target.read_text(encoding="utf-8"))
        save_json(root / "session.json", data)
        return rec

    def tools(self, ctx):
        run_schema = {"type": "object", "properties": {
            "purpose": {"type": "string"}, "entrypoint": {"type": "string"},
            "protocol": {"type": "string"}, "data_version": {"type": "string"},
            "args": {"type": "array", "items": {"type": "string"}}, "seed": {"type": "integer"},
            "timeout_seconds": {"type": ["integer", "null"]}, "parent_id": {"type": ["string", "null"]},
            "check_entrypoint": {"type": ["string", "null"]}, "metrics_path": {"type": "string"},
            "module": {"type": "boolean"}, "check_runner": {"enum": ["auto", "pytest", "checks_json"]}},
            "required": ["purpose", "entrypoint", "protocol", "data_version"], "additionalProperties": False}
        return [ToolDef("prepare_coding_submission", "Freeze actual OpenHands files with a researcher-specified run contract; returns unapproved code.",
            {"type": "object", "properties": {"task_id": {"type": "string"}, "run": run_schema,
              "test_paths": {"type": "array", "items": {"type": "string"}}}, "required": ["task_id", "run"]},
            lambda task_id, run, test_paths=None: json.dumps(self.prepare(task_id, run, test_paths=test_paths, ctx=ctx), ensure_ascii=False),
            permissions=("code.request",)),
            ToolDef("inspect_coding_workspace", "List actual files and changes in a persistent coding task workspace.",
                {"type": "object", "properties": {"task_id": {"type": "string"}, "file_path": {"type": "string"}}, "required": ["task_id"]},
                lambda task_id, file_path="": json.dumps(self.changes(task_id, ctx, file_path), ensure_ascii=False), permissions=("code.read",))]


def coding_factory(ctx, registry, options):
    return OpenHandsCoding(ctx.root / "coding", options)
