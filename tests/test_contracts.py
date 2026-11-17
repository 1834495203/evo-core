"""Contract acceptance tests: real dispatch, persistence, role boundaries and replacement."""
import json
from dataclasses import replace
from pathlib import Path
import pytest

from evo_core.config import CoreConfig, load_config
from evo_core.engine import ResearchKernel
from evo_core.llm import LLMResponse, ToolCallRequest
from evo_core.modules.adapters import native_factory, configure_host
from evo_core.modules.registry import ModuleRegistry, ModuleContext, build_modules
from evo_core.runtime import Capability, Requirement, Manifest, Module, Export, Result, Error, TaskRef, ObjectRef
from evo_core.runtime.host import Policy

OBJECT = {"type": "object"}


def registry(root):
    return ModuleRegistry(ModuleContext(root, None))


class Model:
    api_key = "secret-fixture"
    def __init__(self, responses=None):
        self.responses = list(responses or [LLMResponse("done", "")]); self.calls = []
    def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        assert self.responses, "Unexpected extra model call"
        return self.responses.pop(0)


def test_unknown_native_module_runs_in_real_kernel_and_resolves_after_restart(tmp_path):
    cfg = CoreConfig(workspace=str(tmp_path / "ws"), run_dir=str(tmp_path / "run"), strict_roles=True,
                     modules={"datasets": {"contract": "native", "factory": "evo_core.modules.example_catalog:make_catalog"}})
    args = {"title": "Example", "uri": "https://example.test/data"}
    response = LLMResponse("", "", tool_calls=[ToolCallRequest("t1", "register_dataset", json.dumps(args), args)])
    model = Model([response, LLMResponse("Registered, finished.", "")])
    kernel = ResearchKernel(cfg, plugins=[], llm_client=model)
    result = kernel.run("Register a dataset")
    assert result.status == "completed" and result.model_calls == 2
    call = next(m for m in model.calls[1][0] if m.get("role") == "tool")
    data = json.loads(call["content"])
    assert data["status"] == "completed"
    ref = ObjectRef(**data["refs"][0])
    assert ref.owner == "example.dataset_catalog"
    fresh = build_modules(Path(cfg.workspace) / ".research_modules", None, cfg.modules)
    assert fresh.resolve(ref).data["uri"] == args["uri"]
    assert fresh.get("notebook").search("Example")
    events = [json.loads(line) for line in (Path(cfg.run_dir) / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    starts = [e for e in events if e["event"] == "tool_start"]
    assert starts[0]["identity"]["run_id"] == result.run_id
    assert starts[0]["identity"]["actor_id"] == "main"
    assert starts[0]["identity"]["parent_call_id"]
    recorded_ids = {e["identity"]["call_id"] for e in events if e["event"] in {"llm_start", "tool_start", "capability_start", "task_start", "reference_start"}}
    for e in events:
        if e["event"] in {"llm_start", "tool_start", "capability_start", "task_start", "reference_start"}:
            parent = e["identity"]["parent_call_id"]
            assert parent is None or parent in recorded_ids
    assert any(e["event"] == "module_event" and e["event_type"] == "dataset.registered" for e in events)


def test_capability_versions_cycles_and_unambiguous_providers(tmp_path):
    r = registry(tmp_path)
    @native_factory(Manifest("provider", provides=(Capability("cache", 2),)))
    def provider(ctx): return Module(provider.manifest)
    r.register("provider", provider, contract="native")
    @native_factory(Manifest("consumer", requires=(Requirement("cache", 1),)))
    def consumer(ctx): raise AssertionError("Must reject before construction")
    r.register("consumer", consumer, contract="native")
    with pytest.raises(ValueError, match="Incompatible"): r.get("consumer")
    @native_factory(Manifest("duplicate", provides=(Capability("cache", 2),)))
    def duplicate(ctx): return Module(duplicate.manifest)
    with pytest.raises(ValueError, match="Ambiguous"): r.register("duplicate", duplicate, contract="native")
    assert not r.has("duplicate")
    @native_factory(Manifest("missing", requires=(Requirement("absent"),)))
    def missing(ctx): return Module(missing.manifest)
    r.register("missing", missing, contract="native")
    with pytest.raises(ValueError, match="Missing required"): r.get("missing")
    @native_factory(Manifest("a", provides=(Capability("a"),), requires=(Requirement("b"),)))
    def a(ctx): return Module(a.manifest)
    @native_factory(Manifest("b", provides=(Capability("b"),), requires=(Requirement("a"),)))
    def b(ctx): return Module(b.manifest)
    r.register("a", a, contract="native"); r.register("b", b, contract="native")
    with pytest.raises(ValueError, match="Circular"): r.get("a")


def test_input_output_validation_and_identity_cannot_be_supplied_by_caller(tmp_path):
    r = registry(tmp_path); seen = []
    @native_factory(Manifest("math", provides=(Capability("math"),)))
    def factory(ctx):
        assert not any(hasattr(ctx, k) for k in ("services", "research_system", "state", "llm_client", "root"))
        with pytest.raises(ValueError): ctx.storage.put("../escape.json", {})
        with pytest.raises(PermissionError): ctx.secrets("UNDECLARED")
        def double(call, args):
            seen.append(call.identity)
            return Result(data="wrong" if args["n"] == 0 else args["n"] * 2)
        return Module(factory.manifest, (Export("math", "double", double,
            {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"], "additionalProperties": False},
            {"type": "integer"}, ("workspace.read",)),))
    r.register("math", factory, contract="native")
    assert r.invoke("math", "double", {"n": "bad"}).error.code == "schema_invalid"
    assert r.invoke("math", "double", {"n": 1, "actor_id": "host"}).error.code == "schema_invalid"
    assert seen == []
    assert r.invoke("math", "double", {"n": 0}).error.code == "schema_invalid"
    assert r.invoke("math", "double", {"n": 3}, actor="qa").data == 6
    assert seen[-1].actor_id == "qa"
    with pytest.raises(Exception): seen[-1].actor_id = "host"
    with pytest.raises(ValueError): Result(status="running").validate()
    with pytest.raises(ValueError): Result(status="failed", error="untyped").validate()


def test_declared_dependencies_do_not_escalate_actor_permissions(tmp_path):
    r = registry(tmp_path); configure_host(r.host, CoreConfig(strict_roles=True)); approved = []
    @native_factory(Manifest("approver", provides=(Capability("approval"),)))
    def approve(ctx):
        def handler(call, args): approved.append(call.identity.actor_id); return Result(data={})
        return Module(approve.manifest, (Export("approval", "approve", handler, OBJECT, OBJECT, ("code.approve",)),))
    @native_factory(Manifest("consumer", provides=(Capability("requester"),), requires=(Requirement("approval"),)))
    def consumer(ctx):
        def handler(call, args):
            assert ctx.capabilities.call("unlisted", "get", {}).error.code == "dependency_undeclared"
            return ctx.capabilities.call("approval", "approve", {})
        return Module(consumer.manifest, (Export("requester", "try", handler, OBJECT, OBJECT, ("workspace.read",)),))
    r.register("a", approve, contract="native"); r.register("b", consumer, contract="native")
    assert r.invoke("requester", "try", {}, actor="subagent:coder").error.code == "permission_denied"
    assert approved == []
    assert r.invoke("requester", "try", {}, actor="main").status == "completed"
    assert approved == ["main"]


TASKS = Manifest("jobs", provides=(Capability("jobs"),), task_operations=("get", "result", "cancel"))
@native_factory(TASKS)
def task_provider(ctx):
    def submit(call, args):
        ctx.storage.put("job.json", {"status": "queued"})
        return Result(status="running", task=TaskRef("jobs", "job"))
    def task(call, operation, ref):
        if operation == "cancel": ctx.storage.put("job.json", {"status": "cancelled"})
        state = ctx.storage.get("job.json")
        return Result(status="cancelled" if state["status"] == "cancelled" else "completed", data=state)
    return Module(TASKS, (Export("jobs", "submit", submit, OBJECT, OBJECT, ("experiment.submit",)),), task_handler=task)


def test_tasks_restore_query_cancel_and_unsupported_operations(tmp_path):
    r = registry(tmp_path); configure_host(r.host, CoreConfig(strict_roles=True))
    r.register("any_name", task_provider, contract="native")
    submitted = r.invoke("jobs", "submit", {}, actor="main")
    ref = submitted.task
    fresh = registry(tmp_path); configure_host(fresh.host, CoreConfig(strict_roles=True))
    fresh.register("different_slot", task_provider, contract="native")
    assert fresh.task(ref, "get", actor="qa").data == {"status": "queued"}
    assert fresh.task(ref, "cancel", actor="qa").error.code == "permission_denied"
    assert fresh.task(ref, "cancel", actor="main").status == "cancelled"
    assert fresh.task(ref, "result", actor="main").status == "cancelled"
    assert fresh.task(ref, "retry").error.code == "operation_unsupported"


def test_strict_permissions_without_harness_and_frozen_test_entrypoint(tmp_path):
    cfg = CoreConfig(workspace=str(tmp_path / "ws"), run_dir=str(tmp_path / "run"),
                     modules={"coding": {"options": {"require_review": True}}})
    cfg.plugins.harness.enabled = False
    kernel = ResearchKernel(cfg, plugins=[], llm_client=Model())
    kernel.run("Initialize independent modules")
    state = kernel._build_initial_state("probe", None, False); ctx = kernel._build_ctx(); ctx.state = state
    ctx.module_registry = kernel.modules
    tools = kernel._collect_tools(state, ctx)
    def call(name, args, actor="main"):
        return kernel.execute_tool_with_hooks(state, ctx, tools, name, args, actor=actor, return_result=True)
    assert call("execute", {"command": "python -c \"print(1)\"", "purpose": "setup"}).error.code == "permission_denied"
    assert call("write_file", {"file_path": "src/bypass.py", "content": "x=1"}).error.code == "permission_denied"
    assert not (Path(cfg.workspace) / "src/bypass.py").exists()
    assert call("write_file", {"file_path": "README.md", "content": "Notes"}).status == "completed"
    assert call("write_file", {"file_path": "test_small.py", "content": "def test_ok():\n    assert 2 + 2 == 4\n"}, "subagent:coder").status == "completed"
    assert call("review_code_submission", {"submission_id": "fake", "approved": True, "notes": ""}, "subagent:coder").error.code == "permission_denied"
    task = call("create_coding_task", {"text": "Write test"}).data
    submission = call("submit_code", {"task_id": task["id"], "files": ["test_small.py"],
        "run": {"purpose": "pilot", "entrypoint": "test_small.py", "protocol": "p", "data_version": "d"}}, "subagent:coder").data
    assert call("test_code_submission", {"submission_id": submission["id"]}).data["passed"]
    assert state.metadata["tests_executed"] == 1


def broken_tracer(ctx, registry, options):
    class Broken:
        def log_span(self, *args, **kwargs): raise RuntimeError("exporter unavailable")
        def log_job(self, *args, **kwargs): raise RuntimeError("exporter unavailable")
    return Broken()


def test_exporter_failure_isolated_and_all_model_calls_counted(tmp_path):
    cfg = CoreConfig(workspace=str(tmp_path / "ws"), run_dir=str(tmp_path / "run"),
                     modules={"tracing": {"factory": "test_contracts:broken_tracer"}})
    model = Model([LLMResponse("done", ""), LLMResponse("summary", "")])
    kernel = ResearchKernel(cfg, plugins=[], llm_client=model)
    assert kernel.run("Finish despite failed tracing").status == "completed"
    with kernel.llm.scope("compactor:researcher"):
        kernel.llm.complete([{"role": "user", "content": "Summarize"}])
    assert kernel.llm.counts_snapshot() == {"main": 1, "compactor:researcher": 1}
    assert (Path(cfg.run_dir) / "export_errors.jsonl").is_file()


def test_remote_schema_refs_rejected_without_network_access():
    export = Export("x", "read", lambda *_: Result(data={}), {"$ref": "https://example.test/schema"}, {})
    with pytest.raises(ValueError, match="Remote JSON"): export.validate()


HISTORY = Manifest("alternate.history", provides=(Capability("conversation.context"),))
@native_factory(HISTORY)
def history_provider(ctx):
    def save(call, args):
        ctx.storage.put("history.json", list(args["messages"]))
        return Result()
    def load(call, args):
        try: history = ctx.storage.get("history.json")
        except FileNotFoundError: history = []
        return Result(data=history)
    common = {"session": {"type": "string"}, "role": {"type": "string"}}
    return Module(HISTORY, (
        Export("conversation.context", "save", save, {"type": "object", "properties": {**common, "messages": {"type": "array"}},
            "required": ["session", "role", "messages"]}, {"type": "null"}, ("conversation.manage",)),
        Export("conversation.context", "load", load, {"type": "object", "properties": common, "required": ["session", "role"]},
               {"type": "array"}, ("conversation.manage",)),
    ))


TRACE = Manifest("alternate.trace", provides=(Capability("observability.trace"),))
@native_factory(TRACE)
def trace_provider(ctx):
    def log(call, args):
        try: rows = ctx.storage.get("trace.json")
        except FileNotFoundError: rows = []
        rows.append(dict(args)); ctx.storage.put("trace.json", rows)
        return Result()
    return Module(TRACE, (
        Export("observability.trace", "log_span", log, {"type": "object", "properties": {"kind": {"type": "string"}, "payload": OBJECT},
            "required": ["kind", "payload"]}, {"type": "null"}, ("trace.write",)),
    ))


def test_context_and_trace_can_be_replaced_under_unrelated_slot_names(tmp_path):
    cfg = CoreConfig(workspace=str(tmp_path / "ws"), run_dir=str(tmp_path / "run"), strict_roles=True, modules={
        "context": {"enabled": False}, "tracing": {"enabled": False},
        "remote_a": {"contract": "native", "factory": "test_contracts:history_provider"},
        "remote_b": {"contract": "native", "factory": "test_contracts:trace_provider"},
    })
    first = ResearchKernel(cfg, plugins=[], llm_client=Model())
    assert first.run("Persistence via a native provider").status == "completed"
    # These backends implement no .save/.log_span Python methods and expose no legacy context.
    assert isinstance(first.modules.get("remote_a"), Module)
    with first.llm.scope("host"):
        assert first.modules.service("conversation.context").load("s", "researcher")
    trace_path = first.modules._contexts["remote_b"].storage.path("trace.json")
    rows = json.loads(trace_path.read_text(encoding="utf-8"))
    assert any(r["kind"] == "llm_completed" for r in rows)
    second = ResearchKernel(cfg, plugins=[], llm_client=Model())
    result = second.run("Persistence via a native provider", resume=True)
    assert result.status == "completed"
    assert second.llm.counts_snapshot()["main"] == 2


def test_provider_exception_is_not_retried_by_signature_compatibility():
    from evo_core.modules.tools import _call_supported
    calls = []
    def mutate(value):
        calls.append(value)
        raise ValueError("The service failed after a side effect")
    with pytest.raises(ValueError): _call_supported(mutate, value=1)
    assert calls == [1]


def test_delegated_sdk_runs_as_coder_and_cannot_promote_itself(tmp_path):
    r = registry(tmp_path); configure_host(r.host, CoreConfig(strict_roles=True))
    actors = []
    @native_factory(Manifest("sdk.coder", provides=(Capability("sdk.coder"),)))
    def factory(ctx):
        def handler(call, args):
            actors.append(call.identity.actor_id)
            ctx.workspace.write("src/created.py", "x=1")
            assert not r.host.policy.permits(call.identity.actor_id, ("code.approve",))
            return Result(data={})
        return Module(factory.manifest, (
            Export("sdk.coder", "implement", handler, OBJECT, OBJECT, ("code.request",), execution_role="coder"),
            Export("sdk.coder", "elevate", handler, OBJECT, OBJECT, ("code.request",), execution_role="host"),
        ))
    r.register("sdk", factory, contract="native")
    assert r.invoke("sdk.coder", "implement", {}, actor="main").status == "completed"
    assert actors == ["coder"]
    assert (tmp_path / "src/created.py").is_file()
    assert r.invoke("sdk.coder", "elevate", {}, actor="main").error.code == "delegation_denied"


def test_sdk_usage_report_and_owner_qualified_note_projection(tmp_path):
    from evo_core.harness_server import HarnessSession
    cfg = CoreConfig(workspace=str(tmp_path / "ws"), run_dir=str(tmp_path / "meta"))
    session = HarnessSession(cfg, "Owner-qualified references")
    try:
        first = session.invoke("write_long_term_note", {"kind": "hypothesis", "text": "Hypothesis A"})
        ref = first.refs[0]
        second = session.invoke("write_long_term_note", {"kind": "observation", "text": "Observed A", "refs": [ref.__dict__]})
        assert second.status == "completed" and "projection_warning" not in second.data
        records = session.ctx.research_system.queries.snapshot(session.ctx.study_id)["records"]
        assert [r["kind"] for r in records] == ["hypothesis", "observation"]
        assert records[1]["refs"] == [records[0]["id"]]
        assert session.ctx.module_registry.resolve(ref).data["text"] == "Hypothesis A"
        with session.kernel.llm.scope("coder"):
            session.kernel.llm.report_sdk_call(input_tokens=20, output_tokens=5, duration_ms=1, component="external.sdk")
        assert session.kernel.llm.counts_snapshot() == {"coder": 1}
        assert session.kernel.llm.usage_snapshot()["input_tokens"] == 20
        trace = session.ctx.module_registry.invoke("observability.trace", "query", {"actor": "coder"}, actor="qa")
        assert trace.status == "completed"
        assert any(e.get("source") == "sdk" for e in trace.data["events"])
    finally:
        session.close()


def test_invalid_provider_is_closed_during_startup_failure(tmp_path):
    r = registry(tmp_path); closed = []
    @native_factory(Manifest("bad", provides=(Capability("bad"),)))
    def bad(ctx):
        return Module(bad.manifest, (Export("bad", "get", lambda *_: Result(),
            {"type": "unknown-type"}, {}),), close_handler=lambda: closed.append(True))
    r.register("bad", bad, contract="native")
    with pytest.raises(Exception): r.get("bad")
    assert closed == [True]
    r.close()
    assert closed == [True]
