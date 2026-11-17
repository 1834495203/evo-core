"""Persistent coding workspaces and reviewed publication, without external model calls."""
import io
import json
from pathlib import Path
import sys

import pytest

from evo_core.config import CoreConfig
from evo_core.harness_server import HarnessSession
from evo_core.modules import openhands
from evo_core.system.domain import ExperimentSpec


def config(root):
    return CoreConfig(workspace=str(root / "ws"), run_dir=str(root / "meta"),
        api_key="test-api-key", base_url="https://example.test/v1", modules={"coding": {
            "factory": "evo_core.modules.openhands:coding_factory",
            "options": {"python_executable": sys.executable, "state_root": str(root / "oh"), "require_review": True}}})


def test_partial_workspace_and_conversation_survive_restart(tmp_path):
    cfg = config(tmp_path)
    first = HarnessSession(cfg, "Persistent coding")
    task = first.invoke("create_coding_task", {"text": "Write a package"}).data
    provider = first.ctx.module_registry.get("coding")
    root, tree, meta = provider.session(task["id"], Path(cfg.workspace))
    (tree / "partial.py").write_text("value = 42")
    first.close()
    second = HarnessSession(cfg, "Persistent coding")
    try:
        provider = second.ctx.module_registry.get("coding")
        next_root, next_tree, restored = provider.session(task["id"], Path(cfg.workspace))
        assert next_root == root and next_tree == tree
        assert restored["conversation_id"] == meta["conversation_id"]
        assert (next_tree / "partial.py").read_text() == "value = 42"
        assert not (Path(cfg.workspace) / "partial.py").exists()
    finally:
        second.close()


def test_scanned_files_are_frozen_but_require_researcher_approval(tmp_path):
    cfg = config(tmp_path); session = HarnessSession(cfg, "Freeze actual files")
    try:
        task = session.invoke("create_coding_task", {"text": "Implement"}).data
        provider = session.ctx.module_registry.get("coding")
        _, tree, _ = provider.session(task["id"], Path(cfg.workspace))
        (tree / "run.py").write_text("print('ready')")
        (tree / "helper.py").write_text("value = 42")
        result = session.invoke("prepare_coding_submission", {"task_id": task["id"], "run": {
            "purpose": "pilot", "entrypoint": "run.py", "protocol": "p", "data_version": "d"}})
        assert result.status == "completed", result.error
        submission = result.data
        assert submission["details"]["approved"] is False
        assert set(submission["details"]["files"]) == {"run.py", "helper.py"}
        assert (Path(cfg.workspace) / "helper.py").read_text() == "value = 42"
        with pytest.raises(ValueError, match="approval"):
            provider.authorize(submission["id"], ExperimentSpec("pilot", cfg.workspace, ("run.py", "helper.py"), "run.py", "p", "d"))
        # A researcher edit must not silently disappear when the Coder republishes.
        (Path(cfg.workspace) / "helper.py").write_text("value = 99")
        with pytest.raises(ValueError, match="independently"):
            provider.prepare(task["id"], {"purpose": "pilot", "entrypoint": "run.py", "protocol": "p", "data_version": "d"}, ctx=session.ctx)
        assert (Path(cfg.workspace) / "helper.py").read_text() == "value = 99"
    finally:
        session.close()


def test_failed_sdk_keeps_edits_and_sdk_calls_are_counted(tmp_path, monkeypatch):
    cfg = config(tmp_path); session = HarnessSession(cfg, "Observe external SDK")
    fail = True
    class Process:
        def __init__(self, *args, **kwargs):
            self.returncode = None
            process = self
            class Input(io.BytesIO):
                def close(self):
                    req = json.loads(self.getvalue())
                    (Path(req["workspace"]) / "partial.py").write_text("value = 42")
                    events = [{"kind": "model", "log": {"usage_summary": {
                        "prompt_tokens": 123, "completion_tokens": 7, "cache_read_tokens": 10}, "latency_sec": 0.1}},
                        {"kind": "error", "error": "simulated provider interruption"} if fail else
                        {"kind": "result", "status": "finished", "message": "Completed partial.py"}]
                    Path(req["events_path"]).write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
                    process.returncode = 1 if fail else 0
                    super().close()
            self.stdin = Input()
        def poll(self):
            return self.returncode
    monkeypatch.setattr(openhands.subprocess, "Popen", Process)
    try:
        task = session.invoke("create_coding_task", {"text": "Implement"}).data
        failed = session.invoke("implement_coding_task", {"task_id": task["id"]})
        assert failed.status == "failed" and "Partial edits retained" in failed.error.message
        provider = session.ctx.module_registry.get("coding")
        _, tree, meta = provider.session(task["id"], Path(cfg.workspace))
        assert (tree / "partial.py").exists()
        fail = False
        resumed = session.invoke("implement_coding_task", {"task_id": task["id"], "instruction": "Continue"})
        assert resumed.status == "completed", resumed.error
        assert resumed.data["conversation_id"] == meta["conversation_id"]
        assert resumed.data["changed_files"] == ["partial.py"]
        assert session.kernel.llm.counts_snapshot()["subagent:coder"] == 2
        assert session.kernel.llm.usage_snapshot()["input_tokens"] == 246
        assert not session.ctx.research_system.queries.snapshot(session.ctx.study_id)["jobs"]
    finally:
        session.close()


@pytest.mark.parametrize("name", ["../outside.py", ".research_modules/approval.json", "outputs/metrics.json"])
def test_publication_rejects_non_source_paths(tmp_path, name):
    with pytest.raises(ValueError):
        openhands.source_path(tmp_path, name)
