import json
from dataclasses import asdict
from pathlib import Path

import pytest

from evo_core.config import CoreConfig
from evo_core.harness_server import HarnessSession
from evo_core.modules.local import LocalLiterature
from evo_core.modules.registry import ModuleRegistry, ModuleContext
from evo_core.system.bootstrap import build_system


def replacement_notebook(ctx, registry, options):
    from evo_core.modules.local import SQLiteDocuments
    return SQLiteDocuments(ctx.root / "replacement.sqlite3")


def extra_module(ctx, registry, options):
    from evo_core.protocol import ToolDef
    class Extra:
        def tools(self, ctx):
            return [ToolDef("extension_ping", "Custom module tool", {"type":"object","properties":{}}, lambda:"pong")]
    return Extra()


def test_additional_module_and_disabled_library_tools(tmp_path):
    cfg = CoreConfig(workspace=str(tmp_path / "ws"), run_dir=str(tmp_path / "meta"),
        modules={"literature":{"enabled":False}, "extra":{"factory":"test_modules:extra_module"}})
    session = HarnessSession(cfg, "Load independent extension")
    try:
        assert "add_paper" not in session.tools
        assert "write_long_term_note" in session.tools
        assert session.call("extension_ping", {}) == "pong"
    finally:
        session.close()


def test_module_replacement_disable_dependencies_and_restart(tmp_path):
    system = build_system(tmp_path)
    modules = system.modules
    modules.get("notebook").put("idea", "Persist this across independent runs")
    assert build_system(tmp_path).modules.get("notebook").search("Persist")
    disabled = build_system(tmp_path / "off", module_specs={"literature":{"enabled":False}})
    assert not disabled.modules.has("literature")
    with pytest.raises(ValueError, match="disabled"):
        disabled.modules.get("literature")
    registry = ModuleRegistry(ModuleContext(tmp_path, system))
    sentinel = object()
    registry.register("custom", lambda *a: sentinel)
    assert registry.get("custom") is sentinel
    registry.register("bad", lambda *a: object(), requires=["missing"])
    with pytest.raises(ValueError, match="missing"):
        registry.get("bad")
    replacement = build_system(tmp_path / "swap", module_specs={"notebook":{
        "factory":"test_modules:replacement_notebook"}})
    assert replacement.modules.get("notebook").path.name == "replacement.sqlite3"
    assert replacement.modules.has("literature")
    closed = []
    class Resource:
        def close(self):
            closed.append(True)
    registry.register("resource", lambda *a: Resource())
    registry.get("resource")
    registry.close()
    assert closed == [True]


def test_invalid_module_fails_early(tmp_path):
    with pytest.raises(ValueError, match="missing methods"):
        build_system(tmp_path, module_specs={"notebook":{"factory":"evo_core.modules.local:context_factory"}})


def test_paper_candidates_admission_and_lazy_content(tmp_path):
    class Provider:
        reads = 0
        def search(self, query, limit):
            return [{"title":"Candidate", "url":"https://example.test/paper"}]
        def read(self, url):
            self.reads += 1
            return {"content":"Archived paper body", "excerpt":"Archived", "locator":"section 1"}
    provider = Provider()
    library = LocalLiterature(tmp_path, provider)
    assert library.search("query")
    assert library.library() == []
    paper = library.add("Title", "Abstract", "My own note", "https://example.test/paper")
    assert provider.reads == 0
    assert "Archived paper body" not in json.dumps(library.library())
    assert library.read(paper["id"])["text"] == "Archived paper body"
    assert library.read(paper["id"])["text"] == "Archived paper body"
    assert provider.reads == 1


def test_reviewer_approval_exact_code_version_and_role_boundary(tmp_path):
    cfg = CoreConfig(workspace=str(tmp_path / "ws"), run_dir=str(tmp_path / "meta"),
                     modules={"coding":{"options":{"require_review":True}}})
    cfg.plugins.sanity_gate.enabled = False
    session = HarnessSession(cfg, "Review a code version")
    session.kernel.plugins = [p for p in session.kernel.plugins if p.name != "sanity_gate"]
    def call(name, **args):
        result = session.kernel.execute_tool_with_hooks(session.state, session.ctx, session.tools, name, args, actor="subagent:coder" if name == "submit_code" else "main")
        assert not result.startswith(("[TOOL ERROR", "[TOOL BLOCKED")), result
        return json.loads(result)
    try:
        source = session.ctx.workspace / "run.py"
        source.write_text("import json\nfrom pathlib import Path\nPath('outputs').mkdir()\nPath('outputs/metrics.json').write_text(json.dumps({'value':2}))")
        task = call("create_coding_task", text="Implement a tiny metric")
        run = {"purpose":"pilot", "entrypoint":"run.py", "protocol":"p1", "data_version":"d1"}
        submission = call("submit_code", task_id=task["id"], files=["run.py"], run=run)
        assert "TOOL ERROR" in session.call("submit_experiment", {**run, "files":["run.py"],"code_submission_id":submission["id"]})
        assert "TOOL BLOCKED" in session.kernel.execute_tool_with_hooks(session.state,session.ctx,session.tools,
            "review_code_submission", {"submission_id":submission["id"],"approved":True,"notes":"approve"},actor="subagent:coder")
        call("review_code_submission", submission_id=submission["id"], approved=True, notes="Reviewed")
        result = call("submit_experiment", **run, files=["run.py"], code_submission_id=submission["id"])
        assert result["metrics"] == {"value":2}
        source.write_text(source.read_text()+"\n# changed")
        assert "Code changed" in session.call("submit_experiment", {**run, "files":["run.py"],"code_submission_id":submission["id"]})
        contexts = session.ctx.research_system.modules.get("context")
        contexts.save("s", "coder", [{"role":"user","content":"private implementation"}])
        assert contexts.load("s", "researcher") == []
        assert contexts.load("s", "coder")[0]["content"] == "private implementation"
    finally:
        session.close()


def test_kernel_runs_modules_without_harness_plugin(tmp_path):
    from evo_core.config import CoreConfig
    from evo_core.engine import ResearchKernel
    from evo_core.llm import LLMResponse, ToolCallRequest

    ws = tmp_path / "ws"
    meta = tmp_path / "meta"
    cfg = CoreConfig(
        workspace=str(ws),
        run_dir=str(meta),
        modules={
            "literature": {"enabled": True},
            "notebook": {"enabled": True},
            "coding": {"enabled": True},
            "experiments": {"enabled": False},
            "context": {"enabled": True},
            "tracing": {"enabled": True},
        },
    )
    cfg.plugins.harness.enabled = False

    def _tc(cid, name, args):
        return ToolCallRequest(cid, name, json.dumps(args), args)

    class ScriptLLM:
        api_key = "fake"

        def __init__(self, responses):
            self.responses = list(responses)
            self.calls = []

        def complete(self, messages, *, tools=None, **kwargs):
            self.calls.append(messages)
            return self.responses.pop(0) if self.responses else LLMResponse("done", "")

    llm1 = ScriptLLM([
        LLMResponse("", "", tool_calls=[
            _tc("1", "add_paper", {
                "title": "Cache Paper", "abstract": "Study of FIFO.",
                "notes": "Key baseline", "url": "https://example.test/cache",
                "authors": "A. Author", "year": "2024", "tags": ["caching"],
            }),
            _tc("2", "write_long_term_note", {
                "kind": "idea", "text": "Test adaptive ghost queue sizing.", "summary": "Ghost queue sizing idea",
            }),
            _tc("3", "create_coding_task", {"text": "Implement baseline simulator"}),
        ]),
        LLMResponse("Saved paper and note.", ""),
    ])
    res1 = ResearchKernel(cfg, plugins=[], llm_client=llm1).run("Explore caching without harness")
    assert res1.status == "completed"
    mod_dir = ws / ".research_modules"
    assert (mod_dir / "literature" / "cards").is_dir()
    assert list((mod_dir / "literature" / "cards").glob("*.md"))
    assert list((mod_dir / "notebook_md").glob("*.md"))

    llm2 = ScriptLLM([LLMResponse("Saw index.", "")])
    res2 = ResearchKernel(cfg, plugins=[], llm_client=llm2).run("Continue work")
    assert res2.status == "completed"
    assert "# Long-Term Project Notes Index" in llm2.calls[0][0]["content"]
    assert "Ghost queue sizing idea" in llm2.calls[0][0]["content"]


def test_strict_researcher_coder_separation_and_implement_handoff(tmp_path):
    import re
    from evo_core.config import CoreConfig
    from evo_core.engine import ResearchKernel
    from evo_core.llm import LLMResponse, ToolCallRequest
    from evo_core.plugins.harness import ResearchHarnessPlugin
    from evo_core.plugins.subagents import SubagentsPlugin

    ws = tmp_path / "ws"
    meta = tmp_path / "meta"
    cfg = CoreConfig(
        workspace=str(ws),
        run_dir=str(meta),
        modules={
            "coding": {"options": {"require_review": True}},
            "context": {"enabled": True},
            "tracing": {"enabled": True},
        },
    )
    cfg.plugins.subagents.roles = ["coder"]
    cfg.plugins.sanity_gate.enabled = False

    def _tc(cid, name, args):
        return ToolCallRequest(cid, name, json.dumps(args), args)

    class HandoffLLM:
        api_key = "fake"

        def __init__(self):
            self.step = 0

        def complete(self, messages, *, tools=None, **kwargs):
            self.step += 1
            sys_txt = messages[0]["content"]
            all_txt = "\n".join(str(m.get("content", "")) for m in messages)
            if "dedicated Coder agent" in sys_txt:
                task_m = re.search(r"Coding task_id:\s*(\S+)", all_txt)
                task_id = task_m.group(1) if task_m else ""
                if not any(m.get("role") == "tool" and m.get("name") == "submit_code" for m in messages):
                    if not any(m.get("role") == "tool" and m.get("name") == "write_file" for m in messages):
                        run_py = (
                            "import json\nfrom pathlib import Path\n"
                            "Path('results').mkdir(exist_ok=True)\n"
                            "Path('results/out.json').write_text(json.dumps({'score': 9}))\n"
                        )
                        chk_py = (
                            "import json\nfrom pathlib import Path\n"
                            "m = json.loads(Path('results/out.json').read_text())\n"
                            "Path('outputs').mkdir(exist_ok=True)\n"
                            "Path('outputs/checks.json').write_text(json.dumps({'checks': [{'name': 'pos', 'passed': m['score'] > 0}]}))\n"
                        )
                        return LLMResponse("", "", tool_calls=[
                            _tc("c1", "write_file", {"file_path": "src/run.py", "content": run_py}),
                            _tc("c2", "write_file", {"file_path": "src/check.py", "content": chk_py}),
                        ])
                    return LLMResponse("", "", tool_calls=[
                        _tc("c3", "submit_code", {
                            "task_id": task_id,
                            "files": ["src/run.py", "src/check.py"],
                            "run": {
                                "purpose": "pilot",
                                "entrypoint": "src/run.py",
                                "check_entrypoint": "src/check.py",
                                "metrics_path": "results/out.json",
                                "protocol": "p1",
                                "data_version": "d1",
                            },
                        })
                    ])
                return LLMResponse("Implemented and submitted code.", "")

            # Researcher turns
            if self.step == 1:
                return LLMResponse("", "", tool_calls=[
                    _tc("r1", "write_file", {"file_path": "src/illegal.py", "content": "x=1"}),
                    _tc("r2", "submit_code", {"task_id": "none", "files": [], "run": {}}),
                ])
            if self.step == 2:
                return LLMResponse("", "", tool_calls=[
                    _tc("r3", "create_coding_task", {"text": "Write src/run.py and src/check.py"}),
                ])
            if self.step == 3:
                task_id = re.findall(r'"id":\s*"(doc_[a-f0-9]+)"', all_txt)[-1]
                return LLMResponse("", "", tool_calls=[
                    _tc("r4", "implement_coding_task", {"task_id": task_id, "instruction": "Follow spec"}),
                ])
            if not any(m.get("role") == "tool" and m.get("name") == "review_code_submission" for m in messages):
                sub_id = re.findall(r'"submissions":\s*\[\s*"(doc_[a-f0-9]+)"', all_txt)[-1]
                return LLMResponse("", "", tool_calls=[
                    _tc("r5", "inspect_code_submission", {"submission_id": sub_id, "file_path": "src/run.py"}),
                    _tc("r6", "review_code_submission", {"submission_id": sub_id, "approved": True, "notes": "Reviewed"}),
                ])
            if not any(m.get("role") == "tool" and m.get("name") == "submit_experiment" for m in messages):
                sub_id = re.findall(r'"submissions":\s*\[\s*"(doc_[a-f0-9]+)"', all_txt)[-1]
                return LLMResponse("", "", tool_calls=[
                    _tc("r7", "submit_experiment", {
                        "purpose": "pilot", "entrypoint": "src/run.py", "protocol": "p1",
                        "data_version": "d1", "code_submission_id": sub_id,
                    }),
                ])
            if not any(m.get("role") == "tool" and m.get("name") == "verify_experiment" for m in messages):
                job_id = re.findall(r'"job_id":\s*"(exp_[a-f0-9]+)"', all_txt)[-1]
                return LLMResponse("", "", tool_calls=[
                    _tc("r8", "verify_experiment", {"job_id": job_id, "check_entrypoint": "src/check.py"}),
                    _tc("r9", "annotate_experiment", {"job_id": job_id, "note": "Verified score=9"}),
                ])
            if not (ws / "final_report.md").is_file():
                job_id = re.findall(r'"job_id":\s*"(exp_[a-f0-9]+)"', all_txt)[-1]
                return LLMResponse("", "", tool_calls=[
                    _tc("r10", "write_file", {"file_path": "final_report.md", "content": f"Verified {job_id[:12]} score=9."}),
                ])
            return LLMResponse("Completed.", "")

    events = []
    kernel = ResearchKernel(cfg, plugins=[ResearchHarnessPlugin(), SubagentsPlugin()], llm_client=HandoffLLM(), on_event=events.append)
    res = kernel.run("Strictly separated research + coding")
    assert res.status == "completed", res.error
    assert res.acceptance == "not_reviewed"
    blocked = [e for e in events if e["event"] == "tool_blocked"]
    assert {e["tool"] for e in blocked} >= {"write_file", "submit_code"}

    contexts = kernel.modules.get("context")
    coder_msgs = contexts.load(res.run_id, "coder")
    researcher_msgs = contexts.load(res.run_id, "researcher")
    assert any(m.get("name") == "submit_code" for m in coder_msgs)
    assert not any(m.get("name") == "submit_code" and m.get("content", "").startswith('{"id"') for m in researcher_msgs)
    assert (meta / "requests" / "subagent-coder-0001-input.json").is_file()

    spans = [json.loads(ln) for ln in (meta / "system" / "tracking" / "spans.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    llm_actors = {s.get("actor") for s in spans if s["kind"] == "llm_call"}
    assert {"main", "subagent:coder"} <= llm_actors
    assert not any(s["kind"] == "tool_call" for s in spans)


def test_evoscientist_style_context_editing_and_history_offload(tmp_path):
    from evo_core.llm import LLMResponse
    from evo_core.modules.local import LocalContexts

    ctx_mod = LocalContexts(tmp_path / "context")
    offloaded = ctx_mod.offload_tool_output("s1", "coder", "execute", "Z" * 4000, max_chars=200)
    assert "offloaded 4000 chars" in offloaded
    assert list((tmp_path / "context" / "tool_logs").glob("s1_coder_execute_*.txt"))

    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Build feature"},
    ]
    for i in range(7):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "name": "read_file", "content": ("A" * 700) + f"_{i}"})
    msgs.append({"role": "assistant", "content": "", "tool_calls": [{"id": "ct", "type": "function", "function": {"name": "think", "arguments": "{}"}}]})
    msgs.append({"role": "tool", "tool_call_id": "ct", "name": "think", "content": "X" * 700})

    # Trigger stage-1 tool output clearing (50%), not stage-2 compaction yet
    ctx_mod.edit_and_compact("s1", "coder", msgs, context_limit_tokens=200, edit_fraction=0.5, compact_fraction=999.0, keep_recent_tools=5)
    tool_msgs = [m for m in msgs if m.get("role") == "tool"]
    assert tool_msgs[0]["content"].startswith("[Cleared older read_file output")
    assert tool_msgs[1]["content"].startswith("[Cleared older read_file output")
    assert tool_msgs[-1]["name"] == "think" and tool_msgs[-1]["content"].startswith("X" * 100)

    class CompLLM:
        def complete(self, *args, **kwargs):
            return LLMResponse("- Compacted earlier coder turns", "")

    # Trigger stage-2 summarization + conversation_history offload
    ctx_mod.edit_and_compact("s1", "coder", msgs, llm_client=CompLLM(), context_limit_tokens=150, compact_fraction=0.5, keep_recent_messages=4)
    assert any("<compacted_history>" in str(m.get("content", "")) for m in msgs)
    hist_file = tmp_path / "context" / "conversation_history" / "s1_coder.md"
    assert hist_file.is_file()
    hist_txt = hist_file.read_text(encoding="utf-8")
    assert '<message type="ai">' in hist_txt and '<tool_call ' in hist_txt


def test_unified_notes_experiment_annotation_and_short_exp_id_delivery(tmp_path):
    from evo_core.config import CoreConfig
    from evo_core.harness_server import HarnessSession
    from evo_core.system.research.evidence_contract import delivery_issues

    cfg = CoreConfig(workspace=str(tmp_path / "ws"), run_dir=str(tmp_path / "meta"))
    cfg.plugins.sanity_gate.enabled = False
    session = HarnessSession(cfg, "Annotate experiment and deliver with short ID")
    session.kernel.plugins = [p for p in session.kernel.plugins if p.name != "sanity_gate"]
    try:
        ws = session.ctx.workspace
        (ws / "run.py").write_text("import json\nfrom pathlib import Path\nPath('outputs').mkdir()\nPath('outputs/metrics.json').write_text(json.dumps({'val': 7}))")
        (ws / "chk.py").write_text("import json\nfrom pathlib import Path\nPath('outputs').mkdir(exist_ok=True)\nPath('outputs/checks.json').write_text(json.dumps({'checks': [{'name': 'ok', 'passed': True}]}))")
        job = json.loads(session.call("submit_experiment", {
            "purpose": "test", "entrypoint": "run.py", "files": ["run.py", "chk.py"],
            "check_entrypoint": "chk.py", "protocol": "p1", "data_version": "d1",
        }))
        job_id = job["job_id"]
        session.call("annotate_experiment", {"job_id": job_id, "note": "Measured val=7 cleanly"})
        polled = json.loads(session.call("poll_experiments", {"job_ids": [job_id]}))
        assert polled[0]["notes"] and polled[0]["notes"][0]["text"] == "Measured val=7 cleanly"
        session.call("verify_experiment", {"job_id": job_id, "check_entrypoint": "chk.py"})

        short_id = job_id[:12]
        report = f"# Report\n\nExperiment {short_id} yielded val=7.\n"
        snap = session.ctx.research_system.queries.snapshot(session.ctx.study_id)
        assert delivery_issues(snap, report) == []
        archived = json.loads(session.call("archive_research_report", {"content": report, "refs": [job_id]}))
        assert archived["record_id"].startswith("note_")
    finally:
        session.close()
