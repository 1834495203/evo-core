"""Comprehensive tests for unified execution boundaries, parallel harness, verification, literature fulltext, Q&A, and live dashboard."""

from __future__ import annotations

from dataclasses import asdict
import json
import sqlite3
from pathlib import Path
from typing import Any
import urllib.request

import pytest

from evo_core.cli import main as cli_main
from evo_core.config import load_config
from evo_core.engine import ResearchKernel
from evo_core.llm import LLMResponse, ToolCallRequest
from evo_core.plugins import (
    BudgetPlugin,
    LiteratureSearchPlugin,
    ResearchHarnessPlugin,
    SanityGatePlugin,
    SubagentsPlugin,
)
from evo_core.plugins.literature import _reconstruct_openalex_abstract
from evo_core.system import build_system
from evo_core.system.domain import ExperimentSpec
from evo_core.system.experiments.docker_worker import DockerExperimentWorker
from evo_core.system.interaction.qa import CompanionQAAgent
from evo_core.system.observability.server import ResearchDashboardServer
from evo_core.system.research.literature import LiteratureReviewPolicy


class ScriptedLLM:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.api_key = "sk-test-key"
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages, *, tools=None, **kwargs) -> LLMResponse:
        self.calls.append({"messages": messages, "tools": tools, **kwargs})
        if self._responses:
            return self._responses.pop(0)
        return LLMResponse(content="Done.", reasoning_content="")


def _tc(tc_id: str, name: str, args: dict[str, Any]) -> ToolCallRequest:
    return ToolCallRequest(
        id=tc_id,
        name=name,
        arguments_raw=json.dumps(args, ensure_ascii=False),
        arguments=args,
    )


def test_before_tool_call_blocks_syntax_error_and_untested_sweep(tmp_path: Path) -> None:
    cfg = load_config(overrides={"workspace": str(tmp_path / "ws"), "run_dir": str(tmp_path / "meta")})
    llm = ScriptedLLM(
        [
            # 1. Write syntactically broken file
            LLMResponse(
                content="",
                reasoning_content="",
                tool_calls=[
                    _tc("c1", "write_file", {"file_path": "src/bad.py", "content": "def broken(:\n  pass\n"}),
                ],
            ),
            # 2. Attempt run_in_background -> blocked due to syntax error
            LLMResponse(
                content="",
                reasoning_content="",
                tool_calls=[
                    _tc("c2", "run_in_background", {"command": "python -c \"print('should_not_run')\"", "name": "bg1"}),
                ],
            ),
            # 3. Fix syntax, attempt run_in_background BEFORE running test -> blocked by untested guard
            LLMResponse(
                content="",
                reasoning_content="",
                tool_calls=[
                    _tc("c3", "write_file", {"file_path": "src/bad.py", "content": "def ok():\n    return 1\n"}),
                    _tc("c4", "run_in_background", {"command": "python -c \"print('still_blocked')\"", "name": "bg2"}),
                ],
            ),
            # 4. Run a sanity/test command then launch background process
            LLMResponse(
                content="",
                reasoning_content="",
                tool_calls=[
                    _tc("c5", "execute", {"command": "python -c \"import src.bad as b; print('test_pass', b.ok())\""}),
                    _tc("c6", "run_in_background", {"command": "python -c \"print('bg_allowed')\"", "name": "bg3"}),
                ],
            ),
            LLMResponse(content="Verified and ran background task.", reasoning_content=""),
        ]
    )
    events: list[dict[str, Any]] = []
    kernel = ResearchKernel(cfg, plugins=[SanityGatePlugin()], llm_client=llm, on_event=events.append)
    res = kernel.run("Verify sanity blocking.")
    assert res.status == "completed"
    blocked_events = [e for e in events if e["event"] == "tool_blocked"]
    assert len(blocked_events) == 2
    assert "SyntaxError" in blocked_events[0]["reason"] or "SyntaxErrors" in blocked_events[0]["reason"]
    assert "not been verified" in blocked_events[1]["reason"]
    # Only the 3rd launch succeeded after test ran
    assert len(list((tmp_path / "ws" / ".bg_processes").glob("*.log"))) == 1


def test_subagent_shares_hooks_and_respects_budget(tmp_path: Path) -> None:
    # Shared hook test: subagent cannot run_in_background with unverified Python edit
    cfg1 = load_config(overrides={"workspace": str(tmp_path / "ws1"), "run_dir": str(tmp_path / "m1")})
    events1: list[dict[str, Any]] = []
    llm1 = ScriptedLLM(
        [
            LLMResponse(
                content="",
                reasoning_content="",
                tool_calls=[_tc("p1", "delegate_task", {"role": "code", "instruction": "Write and launch."})],
            ),
            # Child calls:
            LLMResponse(
                content="",
                reasoning_content="",
                tool_calls=[
                    _tc("s1", "write_file", {"file_path": "src/sub.py", "content": "x = 1\n"}),
                    _tc("s2", "run_in_background", {"command": "python -c \"print('no')\""}),
                ],
            ),
            LLMResponse(content="Subagent finished after hook block.", reasoning_content=""),
            # Parent finish:
            LLMResponse(content="Done.", reasoning_content=""),
        ]
    )
    k1 = ResearchKernel(cfg1, plugins=[SanityGatePlugin(), SubagentsPlugin()], llm_client=llm1, on_event=events1.append)
    res1 = k1.run("Test subagent shared hooks.")
    assert res1.status == "completed"
    blocked = [e for e in events1 if e["event"] == "tool_blocked"]
    assert len(blocked) == 1
    assert blocked[0]["actor"] == "subagent:code"

    # Shared budget test: subagent calls count toward max_calls
    cfg2 = load_config(overrides={"workspace": str(tmp_path / "ws2"), "run_dir": str(tmp_path / "m2")})
    cfg2.plugins.budget.max_calls = 3
    llm2 = ScriptedLLM(
        [
            LLMResponse(
                content="",
                reasoning_content="",
                tool_calls=[_tc("p1", "delegate_task", {"role": "code", "instruction": "Burn budget."})],
            ),
            # Child step 1 (effective=2):
            LLMResponse(content="", reasoning_content="", tool_calls=[_tc("s1", "think", {"reflection": "1"})]),
            # Child step 2 (effective=3):
            LLMResponse(content="", reasoning_content="", tool_calls=[_tc("s2", "think", {"reflection": "2"})]),
            # Child step 3 should be prevented by budget!
            LLMResponse(content="Should not reach here", reasoning_content=""),
        ]
    )
    k2 = ResearchKernel(cfg2, plugins=[BudgetPlugin(), SubagentsPlugin()], llm_client=llm2)
    res2 = k2.run("Test subagent budget.")
    assert res2.status == "stopped"
    assert res2.acceptance == "unverified_budget_stop"


def test_default_plugins_finish_without_reviewer_or_fabricated_acceptance(tmp_path: Path) -> None:
    cfg = load_config(overrides={"workspace": str(tmp_path / "ws"), "run_dir": str(tmp_path / "meta")})
    llm = ScriptedLLM([LLMResponse(content="Delivering incomplete work honestly.", reasoning_content="")])
    events: list[dict[str, Any]] = []
    res = ResearchKernel(cfg, llm_client=llm, on_event=events.append).run("Conduct research.")
    assert res.status == "completed"
    assert res.acceptance == "not_reviewed"
    assert res.scientific_outcome == "unassessed"
    assert res.model_calls == 1
    assert len(llm.calls) == 1
    assert not any(e["event"] in {"revision_requested", "rubric_check"} for e in events)
    summary = json.loads((tmp_path / "meta" / "summary.json").read_text(encoding="utf-8"))
    assert summary["revision_count"] == 0
    assert summary["delivery_issues"]
    assert "rubric_judge" not in summary["enabled_plugins"]


def test_parallel_worker_pool_and_experiment_verification_and_invalidation(tmp_path: Path) -> None:
    system = build_system(tmp_path / "system")
    study = system.controller.create("Parallel test", max_experiments=8)
    src = tmp_path / "src"
    src.mkdir()
    (src / "exp_a.py").write_text(
        "import json, pathlib, time\ntime.sleep(0.05)\npathlib.Path('outputs').mkdir(exist_ok=True)\n"
        "pathlib.Path('outputs/metrics.json').write_text('{\"acc\": 0.91}')\n",
        encoding="utf-8",
    )
    (src / "exp_b.py").write_text(
        "import json, pathlib, time\ntime.sleep(0.05)\npathlib.Path('outputs').mkdir(exist_ok=True)\n"
        "pathlib.Path('outputs/metrics.json').write_text('{\"acc\": 0.12}')\n",
        encoding="utf-8",
    )
    (src / "check_acc.py").write_text(
        "import json, pathlib\n"
        "m = json.loads(pathlib.Path('outputs/metrics.json').read_text())\n"
        "ok = m['acc'] > 0.5\n"
        "pathlib.Path('outputs/checks.json').write_text(json.dumps({'checks': [{'name': 'acc_above_half', 'passed': ok}]}))\n",
        encoding="utf-8",
    )

    job_a = system.harness.submit(
        study.id,
        1,
        ExperimentSpec("Run A", str(src), ("exp_a.py", "check_acc.py"), "exp_a.py", "p1", "d1", check_entrypoint="check_acc.py"),
        "req-a",
    )
    job_b = system.harness.submit(
        study.id,
        1,
        ExperimentSpec("Run B", str(src), ("exp_b.py", "check_acc.py"), "exp_b.py", "p1", "d1", check_entrypoint="check_acc.py"),
        "req-b",
    )

    pool = system.worker_pool(max_workers=2)
    try:
        completed = pool.drain(study.id)
    finally:
        pool.shutdown()

    assert len(completed) == 2
    fresh_a = system.harness.get(job_a.id)
    fresh_b = system.harness.get(job_b.id)
    assert fresh_a.validity == "verified"
    assert fresh_b.validity == "invalid"

    # Invalidated run cannot be cited as normal supporting evidence
    with pytest.raises(ValueError, match="marked invalid"):
        system.notebook.record(study.id, "observation", "Wrong claim", (fresh_b.id,))

    # Disclosed failure observation succeeds when allow_invalid_ref=True
    neg = system.notebook.record(
        study.id,
        "observation",
        "Run B failed check",
        (fresh_b.id,),
        details={"allow_invalid_ref": True},
    )
    assert neg.id.startswith("note_")

    # Verify DockerExperimentWorker local fallback when Docker image is absent
    job_c = system.harness.submit(
        study.id,
        1,
        ExperimentSpec("Fallback test", str(src), ("exp_a.py",), "exp_a.py", "p1", "d1"),
        "req-c",
    )
    docker_worker = DockerExperimentWorker(
        system.store,
        system.artifacts,
        system.root / "executions",
        "docker-w1",
        image="nonexistent-evo-test-image:never",
        fallback_to_local=True,
    )
    finished_c = docker_worker.run_once(study.id)
    assert finished_c is not None and finished_c.id == job_c.id
    assert finished_c.status == "succeeded"
    exec_meta = json.loads(system.artifacts.read(finished_c.result["artifacts"]["executor.json"]).decode("utf-8"))
    assert exec_meta["executor"] == "local_fallback"


def test_literature_policy_triggers_on_consecutive_failures_and_fulltext_cache(tmp_path: Path) -> None:
    system = build_system(tmp_path / "system")
    study = system.controller.create("Failure review test", max_experiments=8)
    src = tmp_path / "src"
    src.mkdir()
    (src / "fail.py").write_text("raise RuntimeError('boom')\n", encoding="utf-8")

    policy = LiteratureReviewPolicy(experiments_between_reviews=5, max_consecutive_failures=2)
    system.notebook.source(study.id, "Seed Paper", "https://example.org/p1", "Initial excerpt", "Sec 1")
    assert policy.reasons(system.queries.snapshot(study.id)) == []

    for key in ("f1", "f2"):
        system.harness.submit(
            study.id,
            1,
            ExperimentSpec("Failing exp", str(src), ("fail.py",), "fail.py", "p1", "d1"),
            key,
        )
    worker = system.worker("w1")
    worker.run_once(study.id)
    worker.run_once(study.id)
    reasons = policy.reasons(system.queries.snapshot(study.id))
    assert any("consecutively" in r for r in reasons)

    # Check method phase trigger after observation
    system.notebook.source(study.id, "Followup Paper", "https://example.org/p2", "Another excerpt", "Sec 2")
    obs = system.notebook.record(study.id, "observation", "Observed bottleneck", (system.queries.snapshot(study.id)["jobs"][0]["id"],), details={"allow_invalid_ref": True})
    system.controller.focus(study.id, "method", "Design fix")
    reasons_method = policy.reasons(system.queries.snapshot(study.id))
    assert any("before finalizing a new method" in r for r in reasons_method)

    # Check OpenAlex abstract reconstruction + read_paper_fulltext local fulltext caching
    abstract = _reconstruct_openalex_abstract({"Conformal": [0], "prediction": [1], "handles": [2], "shift.": [3]})
    assert abstract == "Conformal prediction handles shift."

    ws = tmp_path / "ws"
    (ws / "papers").mkdir(parents=True)
    long_paper = ("# Conformal Baseline Paper\n\n" + ("conformal calibration quantile residual variance " * 60))
    (ws / "papers" / "sample.md").write_text(long_paper, encoding="utf-8")

    from evo_core.protocol import HookContext, RunState
    from evo_core.sandbox import BackgroundProcessManager, WorkspaceSandbox

    sb = WorkspaceSandbox(ws, run_dir=tmp_path / "rm")
    ctx = HookContext(
        config=load_config(overrides={"workspace": str(ws), "run_dir": str(tmp_path / "rm")}),
        workspace=ws,
        run_dir=tmp_path / "rm",
        sandbox=sb,
        bg_manager=BackgroundProcessManager(sb),
        llm_client=None,
        emit_event=lambda *a, **k: None,
        state=RunState(run_id="r", task_prompt="t", study_id=study.id),
        research_system=system,
        study_id=study.id,
    )
    lit = LiteratureSearchPlugin()
    tool_map = {t.name: t for t in lit.tools(ctx)}
    out = tool_map["read_paper_fulltext"].handler(identifier_or_url="papers/sample.md", focus_query="conformal quantile")
    assert "reading_depth: fulltext_read" in out
    assert (ws / "literature_db" / "index.json").is_file()
    idx = json.loads((ws / "literature_db" / "index.json").read_text(encoding="utf-8"))
    assert idx["sample"]["reading_depth"] == "fulltext_read"


def test_kernel_delivery_preserves_experiment_validity_without_final_review(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    meta = tmp_path / "meta"
    (ws / "papers").mkdir(parents=True)
    (ws / "papers" / "seed.md").write_text("# Seed\n\n" + ("baseline quantile method " * 100), encoding="utf-8")
    (ws / "figures").mkdir(parents=True)
    (ws / "figures" / "fig1.png").write_bytes(b"PNGDATA" * 300)

    cfg = load_config(overrides={"workspace": str(ws), "run_dir": str(meta)})

    bad_code = "import json, pathlib\npathlib.Path('outputs').mkdir(exist_ok=True)\npathlib.Path('outputs/metrics.json').write_text('{\"score\": 0.1}')\n"
    good_code = "import json, pathlib\npathlib.Path('outputs').mkdir(exist_ok=True)\npathlib.Path('outputs/metrics.json').write_text('{\"score\": 0.92}')\n"
    check_code = (
        "import json, pathlib\n"
        "m = json.loads(pathlib.Path('outputs/metrics.json').read_text())\n"
        "pathlib.Path('outputs/checks.json').write_text(json.dumps({'checks': [{'name': 'valid_score', 'passed': m['score'] > 0.5}]}))\n"
    )
    test_code = "def test_sanity():\n    assert 1 + 1 == 2\n"

    scripted_turns = [
        # Turn 1: Read seed paper + write code + tests + run sanity test
        LLMResponse(
            content="",
            reasoning_content="",
            tool_calls=[
                _tc("c1", "read_paper_fulltext", {"identifier_or_url": "papers/seed.md"}),
                _tc("c2", "write_file", {"file_path": "src/exp_bad.py", "content": bad_code}),
                _tc("c3", "write_file", {"file_path": "src/exp_good.py", "content": good_code}),
                _tc("c4", "write_file", {"file_path": "src/check_exp.py", "content": check_code}),
                _tc("c5", "write_file", {"file_path": "tests/test_sanity.py", "content": test_code}),
                _tc("c6", "execute", {"command": "python -m pytest tests/test_sanity.py", "purpose": "test"}),
            ],
        ),
        # Turn 2: Launch both experiments asynchronously via harness and poll completion
        LLMResponse(
            content="",
            reasoning_content="",
            tool_calls=[
                _tc("c7", "submit_experiment", {
                    "purpose": "Buggy baseline", "entrypoint": "src/exp_bad.py",
                    "files": ["src/exp_bad.py", "src/check_exp.py"], "protocol": "p1", "data_version": "d1",
                    "check_entrypoint": "src/check_exp.py", "request_key": "k_bad", "wait": False,
                }),
                _tc("c8", "submit_experiment", {
                    "purpose": "Improved method", "entrypoint": "src/exp_good.py",
                    "files": ["src/exp_good.py", "src/check_exp.py"], "protocol": "p1", "data_version": "d1",
                    "check_entrypoint": "src/check_exp.py", "request_key": "k_good", "wait": False,
                }),
                _tc("c9", "poll_experiments", {"wait_seconds": 15}),
            ],
        ),
    ]

    class DynamicReportLLM(ScriptedLLM):
        def __init__(self, responses: list[LLMResponse]) -> None:
            super().__init__(responses)
            self.post_step = 0

        def complete(self, messages, *, tools=None, **kwargs) -> LLMResponse:
            if self._responses:
                return super().complete(messages, tools=tools, **kwargs)
            self.calls.append({"messages": messages, "tools": tools, **kwargs})
            self.post_step += 1

            import re

            all_text = "\n".join(str(m.get("content", "")) for m in messages)
            ids = re.findall(r'"job_id": "(exp_[a-f0-9]+)"', all_text)
            bad_id = ids[0] if ids else "exp_missing"
            good_id = ids[1] if len(ids) > 1 else "exp_missing2"

            if self.post_step == 1:
                bad_report = (
                    "# Final Research Report\n\n"
                    + ("Preliminary experimental summary. " * 25)
                    + f"\n\nExperiment {bad_id} and {good_id} gave great results.\n\n![fig](figures/fig1.png)\n"
                )
                return LLMResponse(
                    content="",
                    reasoning_content="",
                    tool_calls=[_tc("init_w", "write_file", {"file_path": "final_report.md", "content": bad_report})],
                )
            if self.post_step == 2:
                return LLMResponse(content="Initial report complete.", reasoning_content="")
            return LLMResponse(content="Report delivered.", reasoning_content="")

    dyn_llm = DynamicReportLLM(scripted_turns)
    kernel = ResearchKernel(
        cfg,
        plugins=[LiteratureSearchPlugin(), ResearchHarnessPlugin(), SanityGatePlugin()],
        llm_client=dyn_llm,
    )
    res = kernel.run("Conduct verified end-to-end study.")
    assert res.status == "completed"
    assert res.acceptance == "not_reviewed"
    assert res.scientific_outcome == "unassessed"
    summary = json.loads((meta / "summary.json").read_text(encoding="utf-8"))
    assert summary["revision_count"] == 0
    assert res.model_calls == 4
    assert res.final_text == "Initial report complete."
    db = sqlite3.connect(meta / "system" / "research.sqlite3")
    jobs = [json.loads(row[0]) for row in db.execute("SELECT body FROM jobs")]
    db.close()
    assert {job["validity"] for job in jobs} == {"invalid", "verified"}
    assert (meta / "live_snapshot.json").is_file()
    assert (meta / "system" / "tracking" / "experiments.jsonl").is_file()


def test_companion_qa_and_dashboard_server_and_live_steering(tmp_path: Path) -> None:
    run_dir = tmp_path / "meta"
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "notes.txt").write_text("Important empirical observation inside workspace.", encoding="utf-8")

    system = build_system(run_dir / "system")
    study = system.controller.create("Dashboard & QA study", max_experiments=8)
    src = tmp_path / "src"
    src.mkdir()
    (src / "run.py").write_text(
        "import json, pathlib\npathlib.Path('outputs').mkdir(exist_ok=True)\npathlib.Path('outputs/metrics.json').write_text('{\"loss\": 0.05}')\n",
        encoding="utf-8",
    )
    (src / "chk.py").write_text(
        "import json, pathlib\npathlib.Path('outputs/checks.json').write_text('{\"checks\": [{\"name\": \"finite\", \"passed\": true}]}')\n",
        encoding="utf-8",
    )
    job = system.harness.submit(
        study.id,
        1,
        ExperimentSpec("Baseline", str(src), ("run.py", "chk.py"), "run.py", "p1", "d1", check_entrypoint="chk.py"),
        "k1",
    )
    finished = system.worker("w1").run_once(study.id)
    assert finished is not None and finished.validity == "verified"
    system.notebook.source(study.id, "Paper A", "https://example.org/a", "Excerpt A", "Sec 1")
    system.notebook.record(study.id, "observation", "Loss reached 0.05", (job.id,))

    # Test structured + LLM Q&A without touching main study state
    fake_qa_llm = ScriptedLLM(
        [
            LLMResponse(
                content="",
                reasoning_content="",
                tool_calls=[_tc("q1", "read_workspace_file", {"file_path": "notes.txt"})],
            ),
            LLMResponse(content="Based on verified job and notes.txt, loss is 0.05.", reasoning_content=""),
        ]
    )
    qa = CompanionQAAgent(system.queries, system.events, system.artifacts, ws, llm_client=fake_qa_llm)
    srv = ResearchDashboardServer(
        study_id=study.id,
        queries=system.queries,
        inbox=system.inbox,
        events=system.events,
        qa_agent=qa,
        run_dir=run_dir,
        host="127.0.0.1",
        port=0,
    )
    host, port = srv.start_background()
    assert port > 0
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _call(method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, str]:
        try:
            url = f"http://{host}:{port}{path}"
            if method == "GET":
                with opener.open(url, timeout=3) as resp:
                    return resp.status, resp.read().decode("utf-8")
            req = urllib.request.Request(
                url,
                data=json.dumps(body or {}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with opener.open(req, timeout=3) as resp:
                return resp.status, resp.read().decode("utf-8")
        except OSError:
            code, _, raw = srv.dispatch_request(method, path, body)
            return code, raw.decode("utf-8")

    try:
        st_html, html = _call("GET", "/")
        assert st_html == 200
        assert "EvoCore Autonomous Research Dashboard" in html

        st_prog, prog_raw = _call("GET", "/api/progress")
        assert st_prog == 200
        prog = json.loads(prog_raw)
        assert prog["experiments"][0]["validity"] == "verified"

        st_evs, evs_raw = _call("GET", "/api/events?after=0&limit=20")
        assert st_evs == 200
        evs = json.loads(evs_raw)
        assert len(evs["events"]) >= 4

        st_qa, ans_raw = _call("POST", "/api/ask", {"question": "What did notes.txt and experiments show?"})
        assert st_qa == 200
        ans = json.loads(ans_raw)
        assert ans["mode"] == "llm"
        assert "0.05" in ans["answer"]

        st_cmd, cmd_raw = _call(
            "POST",
            "/api/command",
            {"kind": "change_direction", "text": "Switch to robust conformal shifts"},
        )
        assert st_cmd == 200
        cmd_res = json.loads(cmd_raw)
        assert cmd_res["status"] == "pending"
    finally:
        srv.stop()

    # Verify CLI status / steer / ask --no-llm work cleanly
    assert cli_main(["status", str(run_dir)]) == 0
    assert cli_main(["steer", str(run_dir), "--kind", "suggestion", "--text", "Add ablation table"]) == 0
    assert cli_main(["ask", str(run_dir), "--question", "Progress?", "--no-llm"]) == 0

    # Verify ResearchHarnessPlugin applies pending steering notices inside kernel loop
    cfg = load_config(overrides={"workspace": str(ws), "run_dir": str(run_dir)})
    harness_plugin = ResearchHarnessPlugin()
    state_json_path = run_dir / "state.json"
    from evo_core.protocol import RunState
    kernel = ResearchKernel(
        cfg,
        plugins=[harness_plugin],
        llm_client=ScriptedLLM([LLMResponse(content="Acknowledged user guidance.", reasoning_content="")]),
    )
    # Seed state with existing study_id so before_run attaches to same study
    from dataclasses import asdict as dc_asdict
    seeded = RunState(run_id="r_steer", task_prompt="Dashboard & QA study", study_id=study.id, messages=[{"role": "user", "content": "Continue"}])
    state_json_path.write_text(json.dumps(dc_asdict(seeded)), encoding="utf-8")
    res = kernel.run("Continue", resume=True)
    assert res.status == "completed"
    # Confirm change_direction bumped plan_version to 2 and recorded applied commands
    final_snap = system.queries.snapshot(study.id)
    assert final_snap["study"]["plan_version"] == 2
    assert final_snap["study"]["direction"] == "Switch to robust conformal shifts"
