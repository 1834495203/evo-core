"""Regressions for the actual rt-004 failures; no LLM or network calls."""
import json
from pathlib import Path
import pytest

from evo_core.system.bootstrap import build_system
from evo_core.system.domain import ExperimentSpec
from evo_core.config import CoreConfig
from evo_core.harness_server import HarnessSession


@pytest.mark.parametrize("checker,valid", [
    ("from pathlib import Path\nimport json\ndef test_metric(tmp_path):\n    assert tmp_path.exists()\n    assert json.loads(Path('outputs/metrics.json').read_text())['score'] == 3\n", True),
    ("def test_wrong():\n    assert False\n", False),
    ("# No collected tests, despite a zero-exit script when run directly\n", False),
    ("import pytest\n@pytest.mark.skip(reason='not executed')\ndef test_skipped():\n    pass\n", False),
])
def test_pytest_checker_is_executed_and_generates_fresh_checks(tmp_path, checker, valid):
    system = build_system(tmp_path / "system")
    study = system.controller.create("Verify output")
    source = tmp_path / "source"; source.mkdir()
    (source / "run.py").write_text("import json\nfrom pathlib import Path\nPath('outputs').mkdir()\nPath('outputs/metrics.json').write_text(json.dumps({'score':3}))\nPath('outputs/checks.json').write_text('{\"checks\":[{\"name\":\"stale\",\"passed\":true}]}')\n", encoding="utf-8")
    (source / "test_result.py").write_text(checker, encoding="utf-8")
    spec = ExperimentSpec("test", str(source), ("run.py", "test_result.py"), "run.py", "p", "d", check_entrypoint="test_result.py")
    job = system.harness.submit(study.id, 1, spec, "request")
    result = system.worker("worker").run_once(study.id)
    assert result.id == job.id and result.status == "succeeded"
    assert result.validity == ("verified" if valid else "invalid")
    check = result.checks[-1]
    assert check["runner"] == "pytest"
    assert check["passed"] is valid
    assert not any(c["name"] == "stale" for c in check["checks"])
    assert "checks.json" in check["artifacts"]


def test_plain_check_script_still_requires_its_machine_readable_result(tmp_path):
    system = build_system(tmp_path / "system"); study = system.controller.create("Do not accept empty checks")
    source = tmp_path / "source"; source.mkdir()
    (source / "run.py").write_text("from pathlib import Path\nPath('outputs').mkdir()\nPath('outputs/metrics.json').write_text('{\"score\":3}')")
    (source / "check.py").write_text("print('looks fine')")
    spec = ExperimentSpec("test", str(source), ("run.py", "check.py"), "run.py", "p", "d", check_entrypoint="check.py")
    system.harness.submit(study.id, 1, spec, "r")
    result = system.worker("w").run_once(study.id)
    assert result.validity == "invalid"
    assert result.checks[0]["runner"] == "checks_json"


def test_approved_subset_receipt_survives_restart_and_unrelated_edits(tmp_path):
    cfg = CoreConfig(workspace=str(tmp_path / "ws"), run_dir=str(tmp_path / "meta"),
                     modules={"coding": {"options": {"require_review": True}}})
    session = HarnessSession(cfg, "Scoped tests")
    ws = Path(cfg.workspace)
    (ws / "tests").mkdir(); (ws / "checks").mkdir()
    (ws / "run.py").write_text("from pathlib import Path\nPath('outputs').mkdir()\nPath('outputs/metrics.json').write_text('{\"score\":3}')")
    (ws / "tests/test_unit.py").write_text("def test_unit(tmp_path):\n    assert tmp_path.exists()\n")
    (ws / "checks/test_results.py").write_text("from pathlib import Path\ndef test_result():\n    assert '3' in Path('outputs/metrics.json').read_text()\n")
    task = session.invoke("create_coding_task", {"text": "Implement"}).data
    submission = session.kernel.execute_tool_with_hooks(session.state, session.ctx, session.tools, "submit_code",
        {"task_id": task["id"], "files": ["run.py", "tests/test_unit.py", "checks/test_results.py"],
         "run": {"purpose": "pilot", "entrypoint": "run.py", "protocol": "p", "data_version": "d", "check_entrypoint": "checks/test_results.py"}},
         actor="subagent:coder", return_result=True).data
    tested = session.invoke("test_code_submission", {"submission_id": submission["id"]})
    assert tested.data["passed"] and tested.data["test_paths"] == ["tests/test_unit.py"]
    session.invoke("review_code_submission", {"submission_id": submission["id"], "approved": True, "notes": "Reviewed"})
    session.close()
    # A different setup task must not force this submission to include all workspace files.
    (ws / "unrelated_broken.py").write_text("def broken(")
    resumed = HarnessSession(cfg, "Scoped tests")
    try:
        resumed.state.metadata["modified_py_since_test"] = ["unrelated_broken.py"]
        run = {"purpose": "pilot", "entrypoint": "run.py", "protocol": "p", "data_version": "d", "code_submission_id": submission["id"]}
        result = resumed.invoke("submit_experiment", run)
        assert result.status == "completed", result.error
        assert result.data["validity"] == "verified"
        (ws / "run.py").write_text((ws / "run.py").read_text() + "\n# changed\n")
        changed = resumed.invoke("submit_experiment", run)
        assert changed.status == "failed" and "Code changed" in changed.error.message
        assert len(resumed.ctx.research_system.queries.snapshot(resumed.ctx.study_id)["jobs"]) == 1
    finally:
        resumed.close()


def test_execution_environment_uses_the_worker_python_without_a_probe_experiment(tmp_path):
    cfg = CoreConfig(workspace=str(tmp_path / "ws"), run_dir=str(tmp_path / "meta"))
    session = HarnessSession(cfg, "Inspect environment")
    try:
        result = session.invoke("get_execution_environment", {})
        assert result.status == "completed"
        assert result.data["packages"]["pytest"]["available"]
        assert result.data["executable"]
        assert not session.ctx.research_system.queries.snapshot(session.ctx.study_id)["jobs"]
    finally:
        session.close()
