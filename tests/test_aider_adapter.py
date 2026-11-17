"""Check the external coder's publication/review boundary without model calls."""
from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import sys

import pytest

from evo_core.config import CoreConfig
from evo_core.harness_server import HarnessSession
from evo_core.modules import aider
from evo_core.system.domain import ExperimentSpec


@pytest.mark.parametrize("name", ["../escape.py", "outputs/result.py", ".research_modules/review.json", "coder_handoff.json"])
def test_external_coder_cannot_publish_runtime_or_result_files(tmp_path, name):
    with pytest.raises(ValueError):
        aider._source_file(tmp_path, name)


def test_external_coder_returns_frozen_unapproved_submission_and_ignores_private_files(tmp_path, monkeypatch):
    cfg = CoreConfig(workspace=str(tmp_path / "ws"), run_dir=str(tmp_path / "meta"),
        modules={"coding": {"factory": "evo_core.modules.aider:coding_factory",
                           "options": {"command": [sys.executable], "require_review": True}}})
    session = HarnessSession(cfg, "Use replacement coder")
    workspace = Path(cfg.workspace)
    (workspace / ".env").write_text("PRIVATE=secret")
    (workspace / "research_request.md").write_text("A research request")

    @contextmanager
    def relay(ctx):
        yield "http://127.0.0.1:1/v1", "local-token", []

    class Process:
        def __init__(self, command, *, cwd, env, stdout, **kwargs):
            assert env["OPENAI_API_KEY"] == "local-token"
            assert "--no-auto-test" in command and "--no-suggest-shell-commands" in command
            assert not (cwd / ".env").exists()
            assert not (cwd / ".research_modules").exists()
            assert (cwd / "research_request.md").is_file()
            (cwd / "run.py").write_text("print('implemented by external coder')")
            spec = asdict(ExperimentSpec("pilot", str(cwd), ("run.py",), "run.py", "p", "d"))
            spec.pop("files"); spec.pop("source_dir")
            (cwd / aider.HANDOFF).write_text(json.dumps({"files": ["run.py"], "run": spec}))
            stdout.write(b"external coder completed")
            self.returncode = 0
        def poll(self):
            return self.returncode

    monkeypatch.setattr(aider, "model_relay", relay)
    monkeypatch.setattr(aider.subprocess, "Popen", Process)
    # Stub Git setup as well; this test only exercises the publication boundary.
    monkeypatch.setattr(aider.subprocess, "run", lambda *a, **k: None)
    try:
        task = session.invoke("create_coding_task", {"text": "Implement"}).data
        result = session.invoke("implement_coding_task", {"task_id": task["id"]})
        assert result.status == "completed", result.error
        assert result.data["backend"] == "aider"
        sid = result.data["submissions"][0]
        submission = session.invoke("inspect_code_submission", {"submission_id": sid}).data
        assert submission["details"]["approved"] is False
        assert submission["details"]["submitted_by"] == "subagent:coder"
        assert (workspace / "run.py").read_text().startswith("print(")
        coding = session.ctx.module_registry.get("coding")
        spec = ExperimentSpec("pilot", str(workspace), ("run.py",), "run.py", "p", "d")
        with pytest.raises(ValueError, match="approval"):
            coding.authorize(sid, spec)
        assert not session.ctx.research_system.queries.snapshot(session.ctx.study_id)["jobs"]
    finally:
        session.close()
