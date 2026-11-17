"""Behavioral checks for the opt-in system, using isolated projects and no API calls."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from evo_core.system import build_system
from evo_core.system.__main__ import DEMO_CODE
from evo_core.system.domain import AgentStep, ExperimentSpec
from evo_core.system.research.literature import LiteratureReviewPolicy
from evo_core.system.storage.sqlite import ConflictError


def setup_study(tmp_path: Path, *, code: str = DEMO_CODE, budget: int = 4):
    system = build_system(tmp_path / "system")
    study = system.controller.create("Inspect a fixed numerical example", max_experiments=budget)
    source = tmp_path / "source"
    source.mkdir()
    (source / "experiment.py").write_text(code, encoding="utf-8")
    spec = ExperimentSpec("Measure a fixed mean", str(source), ("experiment.py",), "experiment.py",
                          "mean-v1", "fixed-list-v1")
    return system, study, spec


def test_frozen_input_runs_and_links_real_evidence(tmp_path):
    system, study, spec = setup_study(tmp_path)
    job = system.harness.submit(study.id, 1, spec, "first")
    (Path(spec.source_dir) / "experiment.py").write_text("raise RuntimeError('changed')", encoding="utf-8")
    completed = system.worker("worker-a").run_once()
    assert completed.id == job.id
    assert completed.status == "succeeded"
    assert completed.result["metrics"] == {"mean": 1.5, "n": 4}
    raw = system.artifacts.read(completed.result["artifacts"]["outputs/metrics.json"])
    assert json.loads(raw) == completed.result["metrics"]
    note = system.notebook.record(study.id, "observation", "The observed mean is 1.5", (job.id,))
    system.controller.decide(study.id, "report", "Prepare a progress export", (note.id,), "Validate the inference")
    report = system.reports.export(study.id, tmp_path / "progress.md")
    assert job.id in report.read_text(encoding="utf-8")
    assert system.store.get(study.id).acceptance == "pending"
    assert system.store.get(study.id).next_action == "Validate the inference"


def test_submit_is_idempotent_and_rejects_changed_inputs(tmp_path):
    system, study, spec = setup_study(tmp_path, budget=1)
    first = system.harness.submit(study.id, 1, spec, "same")
    assert system.harness.submit(study.id, 1, spec, "same").id == first.id
    with pytest.raises(ValueError, match="budget"):
        system.harness.submit(study.id, 1, spec, "second")
    (Path(spec.source_dir) / "experiment.py").write_text("print('other')", encoding="utf-8")
    with pytest.raises(ConflictError, match="different inputs"):
        system.harness.submit(study.id, 1, spec, "same")


def test_pending_direction_command_blocks_claim_then_supersedes_old_queue(tmp_path):
    system, study, spec = setup_study(tmp_path)
    old = system.harness.submit(study.id, 1, spec, "old")
    system.inbox.send(study.id, "change_direction", "Investigate another question")
    assert system.worker("worker-a").run_once() is None
    handled = system.inbox.apply_pending(study.id)
    assert handled[0]["status"] == "applied"
    assert system.store.get(study.id).plan_version == 2
    assert system.worker("worker-a").run_once() is None
    assert system.harness.get(old.id).status == "superseded"
    new = system.harness.submit(study.id, 2, spec, "new")
    assert system.worker("worker-a").run_once().id == new.id


def test_read_queries_and_event_paging_do_not_mutate_state(tmp_path):
    system, study, spec = setup_study(tmp_path)
    system.harness.submit(study.id, 1, spec, "first")
    before = system.queries.snapshot(study.id)
    system.queries.progress(study.id)
    first = system.events.read(study.id, 0, 1)
    rest = system.events.read(study.id, first["next_cursor"])
    assert rest["events"][0]["sequence"] > first["next_cursor"]
    after = system.queries.snapshot(study.id)
    assert after["study"] == before["study"]
    assert after["event_cursor"] == before["event_cursor"]


def test_restart_reconnects_to_queued_job_without_reexecuting_completed_job(tmp_path):
    system, study, spec = setup_study(tmp_path)
    job = system.harness.submit(study.id, 1, spec, "first")
    reopened = build_system(system.root)
    assert reopened.worker("new-process").run_once().id == job.id
    assert build_system(system.root).worker("later-process").run_once() is None


def test_stale_agent_proposal_is_discarded_when_user_changes_direction(tmp_path):
    system, study, spec = setup_study(tmp_path)

    class ChangingAgent:
        def next_step(self, context):
            system.inbox.send(study.id, "change_direction", "New user direction")
            return AgentStep("experiment", context["study"]["revision"],
                             {"spec": asdict(spec), "request_key": "obsolete"})

    outcome = system.coordinator.tick(study.id, ChangingAgent())
    assert outcome["status"] == "refresh_required"
    assert system.queries.snapshot(study.id)["jobs"] == []
    assert system.store.get(study.id).direction == "New user direction"


def test_coordinator_dispatches_agent_step_through_harness(tmp_path):
    system, study, spec = setup_study(tmp_path)

    class Agent:
        def next_step(self, context):
            return AgentStep("experiment", context["study"]["revision"],
                             {"spec": asdict(spec), "request_key": "proposed"})

    outcome = system.coordinator.tick(study.id, Agent())
    assert outcome["status"] == "submitted"
    assert system.worker("worker-a").run_once().id == outcome["job_id"]


@pytest.mark.parametrize("code,error", [
    ("raise RuntimeError('broken')", "exited with code"),
    ("print('no metrics')", "Invalid experiment output"),
    ("from pathlib import Path\nPath('outputs').mkdir()\nPath('outputs/metrics.json').write_text('{\"score\": NaN}')", "Invalid experiment output"),
])
def test_failed_execution_or_metric_contract_never_becomes_scientific_acceptance(tmp_path, code, error):
    system, study, spec = setup_study(tmp_path, code=code)
    system.harness.submit(study.id, 1, spec, "first")
    job = system.worker("worker-a").run_once()
    assert job.status == "failed"
    assert error in job.result["error"]
    assert system.store.get(study.id).acceptance == "pending"
    assert system.store.get(study.id).scientific_outcome == "unassessed"


def test_pause_does_not_erase_an_already_running_experiment(tmp_path):
    system, study, spec = setup_study(tmp_path)
    job = system.harness.submit(study.id, 1, spec, "first")
    system.store.claim("worker-a")
    system.inbox.send(study.id, "pause")
    system.inbox.apply_pending(study.id)
    assert system.harness.get(job.id).status == "running"
    assert system.store.get(study.id).status == "paused"
    system.inbox.send(study.id, "resume")
    system.inbox.apply_pending(study.id)
    assert system.store.get(study.id).status == "running"


def test_references_cannot_use_unfinished_or_foreign_study_experiments(tmp_path):
    system, study, spec = setup_study(tmp_path)
    job = system.harness.submit(study.id, 1, spec, "first")
    with pytest.raises(ValueError, match="not an observed result"):
        system.notebook.record(study.id, "observation", "Premature claim", (job.id,))
    other = system.controller.create("Another study")
    source = system.notebook.source(other.id, "A source", "https://example.org/paper", "Read excerpt", "Section 2")
    with pytest.raises(ValueError):
        system.notebook.record(study.id, "observation", "Wrong evidence", (source.id,))


def test_literature_review_reacts_to_plan_version_and_suggestions_do_not_change_scope(tmp_path):
    system, study, spec = setup_study(tmp_path)
    policy = LiteratureReviewPolicy()
    assert policy.reasons(system.queries.snapshot(study.id))
    system.notebook.source(study.id, "A source", "https://example.org/paper", "Read excerpt", "Section 2")
    assert policy.reasons(system.queries.snapshot(study.id)) == []
    system.inbox.send(study.id, "suggestion", "Consider a different baseline")
    assert system.inbox.apply_pending(study.id)[0]["status"] == "delivered"
    assert system.store.get(study.id).direction == study.direction
    system.inbox.send(study.id, "change_direction", "Another direction")
    system.inbox.apply_pending(study.id)
    assert any("direction changed" in reason for reason in policy.reasons(system.queries.snapshot(study.id)))


def test_concurrent_submission_cannot_overspend_experiment_budget(tmp_path):
    system, study, spec = setup_study(tmp_path, budget=1)

    def submit(key):
        try:
            return system.harness.submit(study.id, 1, spec, key).status
        except ValueError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(submit, ["a", "b"]))
    assert sorted(results) == ["queued", "rejected"]
    assert len(system.queries.snapshot(study.id)["jobs"]) == 1


def test_concurrent_artifact_publication_preserves_complete_original_bytes(tmp_path):
    system = build_system(tmp_path / "system")
    raw = b"same experiment evidence" * 1000
    with ThreadPoolExecutor(max_workers=4) as pool:
        keys = list(pool.map(system.artifacts.put, [raw] * 12))
    assert len(set(keys)) == 1
    assert system.artifacts.read(keys[0]) == raw
    assert not list(system.artifacts.root.glob(".tmp-*"))
