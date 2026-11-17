"""Regression tests for the actual rt-003 failure: research outside the ledger."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from evo_core.config import CoreConfig
from evo_core.engine import ResearchKernel
from evo_core.llm import LLMResponse
from evo_core.plugins.harness import ResearchHarnessPlugin
from evo_core.plugins.sanity_gate import SanityGatePlugin, python_manifest
from evo_core.protocol import RunState
from evo_core.system.research.evidence_contract import delivery_issues, pending_interpretations


class NoNetworkModel:
    api_key = "fake"

    def complete(self, *args, **kwargs):
        raise AssertionError("These tests must not call a model")


@pytest.fixture
def runtime(tmp_path):
    cfg = CoreConfig(workspace=str(tmp_path / "ws"), run_dir=str(tmp_path / "meta"))
    plugin = ResearchHarnessPlugin()
    kernel = ResearchKernel(cfg, plugins=[plugin], llm_client=NoNetworkModel())
    ctx = kernel._build_ctx()
    state = RunState("routing-test", "Study a phenomenon", model_calls=1)
    ctx.state = state
    plugin.before_run(state, ctx)
    plugin.before_model_call(state, [], ctx)
    tools = kernel._collect_tools(state, ctx)
    yield kernel, plugin, state, ctx, tools
    plugin._pool.shutdown()


def prepare_project(ctx):
    src = ctx.workspace / "src"
    src.mkdir()
    (src / "__init__.py").write_text("", encoding="utf-8")
    (src / "value.py").write_text("VALUE = 1.5", encoding="utf-8")
    (src / "calibrate.py").write_text(
        "from .value import VALUE\nimport json\nfrom pathlib import Path\n"
        "Path('results').mkdir()\nPath('results/calibration.json').write_text(json.dumps({'mean':VALUE}))\n",
        encoding="utf-8")
    (src / "check.py").write_text(
        "from pathlib import Path\nimport json\nm=json.loads(Path('results/calibration.json').read_text())\n"
        "Path('outputs').mkdir(exist_ok=True)\n"
        "Path('outputs/checks.json').write_text(json.dumps({'checks':[{'name':'mean','passed':m['mean']==1.5}]}))",
        encoding="utf-8")


def submit(tools, key="calibration"):
    return json.loads(tools["submit_experiment"].handler(
        purpose="Calibration", entrypoint="src/calibrate.py", protocol="mean-v1", data_version="fixed-v1",
        module=True, metrics_path="results/calibration.json", request_key=key, wait=True))


def test_harness_routes_formal_work_and_preserves_setup(runtime):
    kernel, plugin, state, ctx, tools = runtime
    assert "run_in_background" not in tools
    for args in ({"command": "python -m src.calibrate"},
                 {"command": "python -m src.calibrate", "purpose": "experiment"},
                 {"command": "print(1)", "purpose": "analysis"}):
        assert plugin.before_tool_call(state, "execute", args, ctx).action == "block"
    assert plugin.before_tool_call(state, "execute", {"command": "python --version", "purpose": "setup"}, ctx).action == "allow"
    assert plugin.before_tool_call(state, "run_in_background", {"command": "anything"}, ctx).action == "block"
    debug = {"command": "python -c 'print(1)'", "purpose": "debug"}
    assert plugin.before_tool_call(state, "execute", debug, ctx).action == "allow"
    assert debug["timeout"] == 30


def test_existing_module_and_result_path_register_real_evidence(runtime):
    kernel, plugin, state, ctx, tools = runtime
    prepare_project(ctx)
    job = submit(tools)
    assert job["status"] == "succeeded", job
    assert job["metrics"] == {"mean": 1.5}
    snapshot = ctx.research_system.queries.snapshot(ctx.study_id)
    assert snapshot["study"]["phase"] == "intake"
    assert pending_interpretations(snapshot) == [job["job_id"]]
    state.model_calls += 1
    assert plugin.before_tool_call(state, "submit_experiment", {}, ctx).action == "allow"
    tools["record_research_note"].handler(kind="observation", text="Measured mean=1.5", refs=[job["job_id"]])
    assert plugin.before_tool_call(state, "submit_experiment", {}, ctx).action == "allow"
    tools["record_research_note"].handler(kind="decision", text="Validate and report", refs=[job["job_id"]],
                                       choice="report", next_action="Check results")
    checked = json.loads(tools["verify_experiment"].handler(job["job_id"], "src/check.py"))
    assert checked["validity"] == "verified"
    report = f"Measured mean 1.5, based on {job['job_id']}."
    assert delivery_issues(ctx.research_system.queries.snapshot(ctx.study_id), report) == []


def test_shell_files_and_missing_citations_cannot_pass_delivery(runtime):
    kernel, plugin, state, ctx, tools = runtime
    (ctx.workspace / "results").mkdir()
    (ctx.workspace / "results" / "unregistered.json").write_text('{"score":1}')
    assert any("No registered" in x for x in delivery_issues(ctx.research_system.queries.snapshot(ctx.study_id), "We measured score=1"))
    prepare_project(ctx)
    job = submit(tools)
    issues = delivery_issues(ctx.research_system.queries.snapshot(ctx.study_id), "We measured mean=1.5")
    assert any("must cite" in x for x in issues)
    assert not any("observation" in x for x in issues)


def test_shell_edits_invalidate_snapshot_and_fake_sanity_does_not_unlock(runtime):
    kernel, plugin, state, ctx, tools = runtime
    prepare_project(ctx)
    gate = SanityGatePlugin()
    gate.before_run(state, ctx)
    gate.after_tool_call(state, "execute", {"command": "echo sanity", "purpose": "test"},
                         "sanity\n[Command succeeded]", ctx)
    assert gate.before_tool_call(state, "submit_experiment", {}, ctx).action == "block"
    gate.after_tool_call(state, "execute", {"command": "python -m pytest", "purpose": "test"},
                         "16 passed in 0.1s\n[Command succeeded]", ctx)
    assert gate.before_tool_call(state, "submit_experiment", {}, ctx).action == "allow"
    (ctx.workspace / "src/value.py").write_text("VALUE = 2", encoding="utf-8")
    assert gate.before_tool_call(state, "submit_experiment", {}, ctx).action == "block"


def test_direction_change_during_model_call_blocks_obsolete_tool_batch(runtime):
    kernel, plugin, state, ctx, tools = runtime
    ctx.research_system.inbox.send(ctx.study_id, "change_direction", "New question")
    assert plugin.before_tool_call(state, "execute", {"command": "python --version", "purpose": "setup"}, ctx).action == "block"
    plugin.before_model_call(state, [], ctx)
    assert plugin.before_tool_call(state, "execute", {"command": "python --version", "purpose": "setup"}, ctx).action == "allow"


def test_old_plan_result_cannot_overwrite_new_direction_phase(runtime):
    kernel, plugin, state, ctx, tools = runtime
    prepare_project(ctx)
    from evo_core.system.domain import ExperimentSpec
    system = ctx.research_system
    spec = ExperimentSpec("Old experiment", str(ctx.workspace), ("src/value.py",), "src/value.py", "p1", "d1")
    job = system.harness.submit(ctx.study_id, 1, spec, "old-plan")
    system.store.claim("external-worker", ctx.study_id)
    system.inbox.send(ctx.study_id, "change_direction", "New direction")
    system.inbox.apply_pending(ctx.study_id)
    system.store.finish_job(job.id, "external-worker", {"status": "succeeded", "metrics": {}, "artifacts": {}})
    study = system.store.get(ctx.study_id)
    assert study.plan_version == 2
    assert study.phase == "intake"
    assert study.direction == "New direction"


def test_checker_must_produce_fresh_checks_not_reuse_experiment_output(runtime):
    kernel, plugin, state, ctx, tools = runtime
    prepare_project(ctx)
    job = submit(tools)
    system = ctx.research_system
    original = system.harness.get(job["job_id"])
    artifacts = dict(original.result["artifacts"])
    artifacts["outputs/checks.json"] = system.artifacts.put(b'{"checks":[{"name":"stale","passed":true}]}')
    copied_job = replace(original, result={**original.result, "artifacts": artifacts})
    worker = system.worker("checker")
    check = worker.run_check_script(copied_job, "noop.py", extra_files={"noop.py": b"print('no checks written')"})
    assert check["passed"] is False


def test_honest_no_execution_is_inconclusive(runtime):
    kernel, plugin, state, ctx, tools = runtime
    tools["record_research_note"].handler(kind="limitation", text="Resources unavailable; no empirical conclusion.")
    tools["set_research_focus"].handler("reporting", "Deliver limitations", "inconclusive")
    assert delivery_issues(ctx.research_system.queries.snapshot(ctx.study_id), "No experiments executed.") == []


def test_experiments_and_delivery_need_no_stages_sources_summaries_or_decisions(runtime):
    kernel, plugin, state, ctx, tools = runtime
    assert "get_research_workflow" not in tools
    tools["set_research_focus"].handler("free-form investigation", "Choose autonomously")
    prepare_project(ctx)
    first = submit(tools, "autonomous-first")
    state.model_calls += 1
    assert plugin.before_tool_call(state, "submit_experiment", {}, ctx).action == "allow"
    submit(tools, "autonomous-second")
    assert ctx.research_system.store.get(ctx.study_id).phase == "free-form investigation"
    checked = tools["verify_experiment"].handler(first["job_id"], "src/check.py")
    assert json.loads(checked)["validity"] == "verified"
    snapshot = ctx.research_system.queries.snapshot(ctx.study_id)
    assert not any(n["kind"] in {"source", "summary", "decision", "observation"} for n in snapshot["records"])
    report = f"Measured mean=1.5, supported by {first['job_id']}. Other work need not be included."
    assert delivery_issues(snapshot, report) == []
    (ctx.workspace / "final_report.md").write_text(report, encoding="utf-8")
    assert plugin.after_agent_finish(state, report, ctx).action == "accept"
    state.pending_notices.clear()
    plugin.before_model_call(state, [], ctx)
    assert not any("Checkpoint" in n or "before the next" in n for n in state.pending_notices)
