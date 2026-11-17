"""Compatibility boundary for the original local services, not kernel policy."""
import inspect
import json
from pathlib import Path
from dataclasses import asdict, is_dataclass

from ..runtime.contracts import Capability, Manifest, Export, Result, native_factory

# Business interfaces live here. The loader/dispatcher never switch on module names.
PORTS = {
    "literature": ("literature.library", ("search", "add", "library", "read", "annotate")),
    "notebook": ("notes.store", ("put", "get", "search")),
    "coding": ("code.implementation", ("task", "submit", "review", "authorize", "inspect")),
    "experiments": ("experiment.execution", ("submit", "get", "worker", "worker_pool")),
    "context": ("conversation.context", ("load", "save")),
    "tracing": ("observability.trace", ("log_span", "log_job")),
}


def plain(value):
    return json.loads(json.dumps(asdict(value) if is_dataclass(value) else value,
                                default=lambda v: asdict(v) if is_dataclass(v) else str(v), allow_nan=False))


def local_manifest(name, factory):
    cap = PORTS.get(name, (f"extension.{name}", ()))[0]
    kinds = {"literature": ("paper",), "notebook": ("note",), "coding": ("code_submission", "coding_task"),
             "experiments": ("experiment",)}.get(name, ())
    return Manifest(f"local.{name}", provides=(Capability(cap),), reference_kinds=kinds,
                    task_operations=("get", "result") if name == "experiments" else ())


def local_task(instance, ref, operation):
    from ..runtime.contracts import Error
    job = plain(instance.get(ref.id))
    if operation == "get": return Result(data=job)
    if job["status"] in {"queued", "running"}: return Result(status="running", task=ref)
    if job["status"] == "failed":
        return Result(status="failed", data=job, error=Error("task_failed", (job.get("result") or {}).get("error", "Experiment failed")))
    return Result(data=job)


def local_reference(name, instance, ref):
    if name == "literature": return Result(data=instance.documents.get(ref.id))
    if name == "notebook": return Result(data=instance.get(ref.id))
    if name == "coding": return Result(data=instance.docs.get(ref.id))
    if name == "experiments": return Result(data=plain(instance.get(ref.id)))
    return Result.failure("reference_unsupported", ref.kind)


def validate_legacy(name, instance):
    methods = PORTS.get(name, ("", ()))[1]
    missing = [method for method in methods if not callable(getattr(instance, method, None))]
    if missing:
        raise ValueError(f"Invalid module {name}; missing methods: {missing}")


def local_exports(name, instance):
    cap = PORTS.get(name, (f"extension.{name}", ()))[0]
    operations = {
        "literature": ("search", "add", "library", "read", "annotate"),
        "notebook": ("put", "get", "search", "index"),
        "coding": ("task", "inspect", "review"),
        "experiments": ("get", "list_jobs", "invalidate"),
        "context": ("load", "save", "offload_tool_output"),
        "tracing": ("log_span", "log_job", "query"),
    }.get(name, ())
    for operation in operations:
        fn = getattr(instance, operation, None)
        if not callable(fn): continue
        parameters = inspect.signature(fn).parameters
        properties = {p: {} for p, v in parameters.items()
                      if v.kind not in {v.VAR_POSITIONAL, v.VAR_KEYWORD}}
        required = [p for p, v in parameters.items() if v.default is v.empty and p in properties]
        permission = {"literature": "literature.read", "notebook": "notes.read", "coding": "code.read",
                      "experiments": "experiment.read", "context": "conversation.manage", "tracing": "trace.write"}[name]
        if operation in {"add", "annotate"}: permission = "literature.write"
        if operation == "put": permission = "notes.write"
        if operation == "task": permission = "code.request"
        if operation == "review": permission = "code.approve"
        if operation == "invalidate": permission = "experiment.invalidate"
        if name == "tracing" and operation == "query": permission = "trace.read"
        def handler(ctx, args, fn=fn, name=name, operation=operation):
            params = dict(args)
            if name == "coding" and operation == "review": params["actor"] = ctx.identity.actor_id
            return Result(data=plain(fn(**params)))
        yield Export(cap, operation, handler,
                     {"type": "object", "properties": properties, "required": required,
                      "additionalProperties": any(v.kind is v.VAR_KEYWORD for v in parameters.values())}, {}, (permission,))




def configure_host(host, config):
    from ..runtime.host import Policy
    strict = config.strict_roles
    if strict is None:
        strict = bool(config.modules.get("coding", {}).get("options", {}).get("require_review", False))
    researcher = {"workspace.read", "workspace.write", "code.request", "code.read", "code.approve", "code.test",
        "notes.read", "notes.write", "literature.read", "literature.write", "experiment.submit", "experiment.read",
        "experiment.verify", "experiment.invalidate", "tasks.read", "tasks.cancel", "objects.read", "trace.read", "runtime.plan"}
    grants = {
        "host": {"*"},
        "researcher": researcher if strict else {"*"},
        "coder": {"workspace.read", "workspace.write", "code.read", "code.edit", "code.submit", "notes.read", "notes.write", "literature.read", "tasks.read", "objects.read", "trace.read", "runtime.plan"},
        "qa": {"workspace.read", "notes.read", "literature.read", "experiment.read", "tasks.read", "objects.read", "trace.read"},
    }
    grants.update(config.role_permissions)
    aliases = {"main": "researcher", "mcp": "researcher", "subagent:coder": "coder", "subagent:code": "coder", "observer": "qa"}
    delegations = {"host": ["*"], "researcher": ["coder"]}
    delegations.update(config.role_delegations)
    host.policy = Policy(grants, strict=strict, aliases=aliases, delegations=delegations)
    host.write_permissions = file_permissions


def file_permissions(arguments):
    name = str(arguments.get("file_path", "")).replace("\\", "/").lower()
    if any(part in {".research_modules", ".runtime", ".evo_run"} for part in name.split("/")):
        return ("runtime.admin",)
    suffix = Path(name).suffix
    if suffix not in {".md", ".txt", ".csv", ".png", ".svg", ".jpg", ".pdf"}:
        return ("code.edit",)
    return ("workspace.write",)


def annotate_tool(tool, registry=None):
    """Legacy tool permissions are declared by the application adapter, not the kernel."""
    permissions = {
        "ls": "workspace.read", "read_file": "workspace.read", "write_file": "workspace.write", "edit_file": "workspace.write",
        "execute": "process.execute", "run_in_background": "process.execute", "stop_process": "process.execute",
        "check_process": "workspace.read", "list_processes": "workspace.read", "write_todos": "runtime.plan", "think": "runtime.plan",
        "create_coding_task": "code.request", "implement_coding_task": "code.request", "delegate_task": "code.request",
        "submit_code": "code.submit", "inspect_code_submission": "code.read", "review_code_submission": "code.approve",
        "test_code_submission": "code.test",
        "find_papers": "literature.read", "list_papers": "literature.read", "read_library_paper": "literature.read",
        "add_paper": "literature.write", "annotate_paper": "literature.write",
        "admit_paper": "literature.write", "list_paper_searches": "literature.read", "read_paper_search": "literature.read",
        "write_long_term_note": "notes.write", "search_long_term_notes": "notes.read", "read_long_term_note": "notes.read",
        "record_research_note": "notes.write", "record_research_document": "notes.write", "query_research_records": "notes.read",
        "read_research_document": "notes.read", "set_research_focus": "runtime.plan", "get_research_status": "experiment.read",
        "submit_experiment": "experiment.submit", "poll_experiments": "experiment.read", "verify_experiment": "experiment.verify",
        "get_execution_environment": "workspace.read",
        "invalidate_experiment": "experiment.invalidate", "annotate_experiment": "notes.write", "list_modules": "workspace.read",
        "archive_research_report": "notes.write", "export_research_bundle": "notes.read",
    }
    if not tool.permissions and not tool.wire_result:
        tool.permissions = (permissions.get(tool.name, "extension.call"),)
    if tool.name in {"write_file", "edit_file"} and tool.owner == "host":
        tool.permission_resolver = file_permissions
        # Code editors need code.edit, document writers need workspace.write; don't require both.
        tool.permissions = ()
    references = {
        "add_paper": ("literature.library", "paper"), "annotate_paper": ("literature.library", "paper"),
        "admit_paper": ("literature.library", "paper"),
        "write_long_term_note": ("notes.store", "note"), "read_long_term_note": ("notes.store", "note"),
        "create_coding_task": ("code.implementation", "coding_task"), "submit_code": ("code.implementation", "code_submission"),
        "review_code_submission": ("code.implementation", "code_submission"),
        "submit_experiment": ("experiment.execution", "experiment"),
    }
    if registry and not tool.wire_result and tool.name in references:
        cap, kind = references[tool.name]
        provider = registry._capabilities.get(cap)
        if provider:
            tool.owner = registry._manifests[provider[0]].id
            tool.reference_kind = kind
    return tool


def bind_runtime_tools(registry, tools, invoke):
    """Local service operations use the same authorized path as agent tools.

    This is the application compatibility adapter. Native providers keep their own exports.
    Worker construction and authorization internals are deliberately not public RPC methods.
    """
    aliases = {
        "literature": {"search": "find_papers", "add": "add_paper", "library": "list_papers",
                       "read": "read_library_paper", "annotate": "annotate_paper", "history": "list_paper_searches",
                       "read_search": "read_paper_search", "admit": "admit_paper"},
        "notebook": {"put": "write_long_term_note", "get": "read_long_term_note", "search": "search_long_term_notes"},
        "coding": {"task": "create_coding_task", "implement": "implement_coding_task", "submit": "submit_code",
                   "inspect": "inspect_code_submission", "review": "review_code_submission", "test": "test_code_submission"},
        "experiments": {"submit": "submit_experiment", "verify": "verify_experiment", "invalidate": "invalidate_experiment",
                        "annotate": "annotate_experiment", "environment": "get_execution_environment"},
    }
    record = {"type": "object", "required": ["id", "kind", "text", "details"]}
    for slot, operations in aliases.items():
        if not registry.has_legacy(slot): continue
        capability = PORTS[slot][0]
        for operation, name in operations.items():
            tool = tools.get(name)
            if tool is None: continue
            def handler(ctx, args, name=name):
                return invoke(name, dict(args), ctx.identity.actor_id)
            output = {}
            if name in {"add_paper", "annotate_paper", "write_long_term_note", "read_long_term_note", "create_coding_task", "submit_code", "review_code_submission"}:
                output = record
            elif name in {"find_papers", "list_papers", "search_long_term_notes"}:
                output = {"type": "array", "items": {"type": "object"}}
            elif name == "read_library_paper":
                output = {"type": "object", "required": ["paper_id", "text"]}
            tool.output_schema = output
            export = Export(capability, operation, handler, tool.parameters, output, tool.permissions)
            export.validate()
            registry._exports[(capability, operation)] = (slot, export)


def ledger_refs(system, study_id, refs):
    """Translate project references only at the legacy ledger boundary."""
    records = system.queries.snapshot(study_id)["records"]
    translated = []
    for ref in refs:
        identifier = ref.get("id", "") if isinstance(ref, dict) else str(ref)
        if isinstance(ref, dict):
            candidates = [r for r in records
                if ((r["details"].get("module_ref") or {}).get("owner") == ref["owner"]
                    and (r["details"].get("module_ref") or {}).get("id") == identifier)
                or (r["details"].get("module_record_owner") == ref["owner"] and r["details"].get("module_record_id") == identifier)
                or (ref["owner"] == "local.notebook" and r["id"] == identifier)]
        else:
            candidates = [r for r in records if r["id"] == identifier
                          or r["details"].get("module_record_id") == identifier
                          or (r["details"].get("module_ref") or {}).get("id") == identifier]
        translated.append(candidates[-1]["id"] if candidates else identifier)
    return tuple(translated)
