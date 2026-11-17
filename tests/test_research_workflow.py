"""Actual tool-boundary scientific lifecycle with local execution; literature is an explicit fixture."""
import json
from pathlib import Path

import pytest

from evo_core.config import CoreConfig
from evo_core.harness_server import HarnessSession
from evo_core.system.bootstrap import build_system


@pytest.fixture
def session(tmp_path):
    cfg = CoreConfig(workspace=str(tmp_path / "workspace"), run_dir=str(tmp_path / "meta"))
    result = HarnessSession(cfg, "Offline integration fixture: investigate prediction bias; no novel science claimed")
    yield result
    result.close()


def call(session, name, **kwargs):
    result = session.call(name, kwargs)
    assert not result.startswith(("[TOOL ERROR", "[TOOL BLOCKED")), result
    try:
        return json.loads(result)
    except json.JSONDecodeError:
        return result


def note(s, kind, text, refs=(), **details):
    return call(s, "record_research_document", kind=kind, text=text, refs=list(refs), details=details)["record_id"]


def test_full_workflow_through_runtime_neutral_tools(session, monkeypatch):
    s = session
    focus = lambda phase: call(s, "set_research_focus", phase=phase, next_action="Perform " + phase)
    call(s, "set_research_focus", phase="custom research focus", next_action="Choose freely")
    focus("literature")
    question = note(s, "question", "Does an additive prediction bias improve squared error on centered synthetic data?")
    from evo_core.plugins import literature
    fixture_text = "Explicit test fixture, not a real publication. Squared error is (prediction-target)**2. " * 120
    monkeypatch.setattr(literature, "fetch_paper_content", lambda *a, **kw: {
        "id": "fixture", "title": "Offline test fixture", "url": "https://example.test/fixture",
        "text": fixture_text, "locator": "fixture section 1", "reading_depth": "fulltext_read"})
    result = call(s, "read_paper_fulltext", identifier_or_url="https://example.test/fixture")
    source = call(s, "query_research_records", kind="source")["records"][0]["id"]
    # The harness owns full bytes, independent of editable literature_db caches.
    for path in s.ctx.workspace.rglob("*.txt"):
        path.unlink()
    assert call(s, "read_research_document", record_id=source, max_chars=32000)["content"] == fixture_text
    note(s, "review", "Fixture only: compare biases on the same samples; squared error is the metric.", [source])
    focus("reproduction")
    code = """import json, sys, statistics
from pathlib import Path
bias = float(sys.argv[1]) if len(sys.argv)>1 else 0
errors = [(x + bias - x)**2 for x in range(-10,11)]
Path('outputs').mkdir(exist_ok=True)
Path('outputs/metrics.json').write_text(json.dumps({'mse':statistics.mean(errors), 'n':len(errors)}))
"""
    checker = """import json
from pathlib import Path
m=json.loads(Path('outputs/metrics.json').read_text())
Path('outputs/checks.json').write_text(json.dumps({'checks':[{'name':'nonnegative_mse_and_sample_count','passed':m['mse']>=0 and m['n']==21}]}))
"""
    test = """import unittest
class ErrorInvariant(unittest.TestCase):
    def test_squared_error(self):
        for x in range(-10,11):
            self.assertEqual((x-x)**2, 0)
            self.assertGreaterEqual((x+1-x)**2, 0)
if __name__=='__main__': unittest.main()
"""
    for name, content in [("experiment.py", code), ("check.py", checker), ("test_invariant.py", test)]:
        call(s, "write_file", file_path=name, content=content)
    tested = call(s, "execute", command="python -m unittest discover -s . -p test_invariant.py", purpose="test")
    assert "Ran 1 test" in tested
    def submit(key, bias, wait=True):
        return call(s, "submit_experiment", purpose=key, entrypoint="experiment.py",
                    files=["experiment.py", "check.py"], args=[str(bias)], protocol="paired-centered-v1",
                    data_version="integers-minus10-to10", request_key=key, check_entrypoint="check.py", wait=wait)
    reproduction = submit("reproduction", 0)
    rid = reproduction["job_id"]
    assert reproduction["metrics"]["mse"] == 0
    note(s, "reproduction", "Reproduced zero error for the identity predictor.", [rid, source])
    observation = note(s, "observation", "Identity prediction has MSE zero on these deterministic samples.", [rid])
    note(s, "summary", "Reproduction complete; investigate the proposed bias.", [observation])
    call(s, "record_research_note", kind="decision", text="Test the bias despite the expected limitation.",
         refs=[observation], choice="continue", next_action="Compare candidate biases")
    focus("investigation")
    note(s, "hypothesis", "A nonzero additive bias will improve MSE.", [observation])
    note(s, "method", "Compare biases 0, 1 and -1; the latter two are candidate/ablation.", [question])
    call(s, "read_paper_fulltext", identifier_or_url="https://example.test/fixture", focus_query="squared error")
    note(s, "review", "Refreshed fixture review predicts no improvement; retain the falsifiable comparison.", [source, observation])
    note(s, "protocol", "Shared deterministic data; bias=0 baseline, bias=1 method, bias=-1 ablation. No sampling CI is claimed.", [source])
    focus("method")
    # Same batch can run independently in parallel.
    baseline = submit("baseline", 0, False)
    candidate = submit("candidate", 1, False)
    # Poll before interpreting, then record all actual results.
    jobs = call(s, "poll_experiments", wait_seconds=10)
    assert all(j["status"] == "succeeded" for j in jobs)
    for job in jobs:
        if job["job_id"] != rid:
            note(s, "observation", f"Measured MSE={job['metrics']['mse']}", [job["job_id"]])
    note(s, "summary", "Compared baseline/candidate; retain their actual measurements and run ablation.",
         [baseline["job_id"], candidate["job_id"]])
    call(s, "record_research_note", kind="decision", text="Check the ablation before concluding.",
         refs=[candidate["job_id"]], choice="continue", next_action="Ablation")
    ablation = submit("ablation", -1)
    note(s, "observation", "Ablation MSE is also 1.", [ablation["job_id"]])
    focus("validation")
    ids = [rid, baseline["job_id"], candidate["job_id"], ablation["job_id"]]
    for job_id in ids:
        assert call(s, "verify_experiment", job_id=job_id, check_entrypoint="check.py")["validity"] == "verified"
    note(s, "limitation", "Deterministic toy fixture; no novelty, population inference or real literature claim.", ids)
    summary = note(s, "summary", "Hypothesis refuted: baseline MSE=0, candidate/ablation MSE=1; no improvement.", ids)
    call(s, "record_research_note", kind="decision", text="Stop this direction and report the negative result.",
         refs=[summary], choice="report", next_action="Archive results and reproduction instructions")
    call(s, "set_research_focus", phase="reporting", next_action="Deliver", scientific_outcome="refuted")
    # A small SVG visualizes the actual archived measurements; the bundle preserves its report path.
    figure = s.ctx.workspace / "figures" / "mse.svg"
    figure.parent.mkdir()
    figure.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="200" height="120">'
                      '<text x="5" y="15">Measured MSE: baseline=0, candidate=1</text>'
                      '<rect x="100" y="20" width="30" height="90" fill="steelblue"/></svg>', encoding="utf-8")
    report = "# Offline integration report\n\nThis is an engineering fixture, not autonomous scientific discovery.\n" + \
        "\n".join(ids) + "\nBaseline MSE=0; bias candidates MSE=1. Hypothesis refuted.\n" + \
        "No sampling uncertainty is claimed for this deterministic toy problem.\n" + \
        "Reproduce: python experiment.py 0; python experiment.py 1; python experiment.py -1.\n" + \
        "![Measured error](figures/mse.svg)\n"
    archived = call(s, "archive_research_report", content=report, refs=ids+[summary, source])
    assert call(s, "read_research_document", record_id=archived["record_id"])["content"] == report
    rebuilt = build_system(s.ctx.research_system.root)
    snap = rebuilt.queries.snapshot(s.ctx.study_id)
    assert snap["study"]["scientific_outcome"] == "refuted"
    assert len(snap["jobs"]) == 4
    assert s.harness.after_agent_finish(s.state, report, s.ctx).action == "accept"
    import zipfile
    bundle = call(s, "export_research_bundle")["bundle"]
    with zipfile.ZipFile(bundle) as archive:
        assert archive.read("final_report.md").decode() == report
        assert "mse" in archive.read("metrics.csv").decode()
        assert f"experiments/{rid}/source/experiment.py" in archive.namelist()
        assert f"experiments/{rid}/checks/0/source/check.py" in archive.namelist()
        assert f"documents/{source}.md" in archive.namelist()
        assert archive.read("figures/mse.svg") == figure.read_bytes()


def test_review_cannot_be_satisfied_by_an_uncited_or_non_source_note(session):
    question = note(session, "question", "A question is not literature evidence")
    result = session.call("record_research_document", {"kind":"review", "text":"Read nothing", "refs":[question]})
    assert "TOOL ERROR" in result
    assert call(session, "query_research_records", kind="review")["total"] == 0
