"""Versioned, JSON-serializable wire contracts. No scientific domain dependencies."""
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Mapping
import json
from jsonschema import Draft202012Validator


@dataclass(frozen=True)
class Capability:
    name: str
    version: int = 1


@dataclass(frozen=True)
class Requirement:
    name: str
    version: int = 1
    optional: bool = False


@dataclass(frozen=True)
class Manifest:
    id: str
    version: str = "1.0.0"
    contract_version: int = 1
    provides: tuple[Capability, ...] = ()
    requires: tuple[Requirement, ...] = ()
    events: tuple[str, ...] = ()
    task_operations: tuple[str, ...] = ()
    reference_kinds: tuple[str, ...] = ()

    def validate(self):
        if not self.id or self.contract_version != 1 or not self.version:
            raise ValueError("Unsupported or incomplete module manifest")
        names = [c.name for c in self.provides]
        if len(names) != len(set(names)) or any(not c.name or type(c.version) is not int or c.version < 1 for c in self.provides):
            raise ValueError("Invalid/duplicate capability exports")
        if any(not r.name or type(r.version) is not int or r.version < 1 for r in self.requires):
            raise ValueError("Invalid capability requirement")
        if set(self.task_operations) - {"get", "result", "cancel"}:
            raise ValueError("Unknown task operation")


@dataclass(frozen=True)
class Identity:
    project_id: str
    run_id: str
    session_id: str
    actor_id: str
    call_id: str
    parent_call_id: str | None = None


@dataclass(frozen=True)
class ObjectRef:
    kind: str
    owner: str
    id: str
    version: str | None = None


@dataclass(frozen=True)
class TaskRef:
    owner: str
    id: str


@dataclass(frozen=True)
class Error:
    code: str
    message: str
    retryable: bool = False
    details: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Result:
    status: str = "completed"
    data: Any = None
    refs: tuple[ObjectRef, ...] = ()
    artifacts: tuple[ObjectRef, ...] = ()
    task: TaskRef | None = None
    error: Error | None = None

    @classmethod
    def failure(cls, code, message, *, retryable=False):
        return cls(status="failed", error=Error(code, str(message), retryable))

    def wire(self):
        self.validate()
        return asdict(self)

    def validate(self):
        if self.status not in {"completed", "running", "failed", "cancelled"}:
            raise ValueError("Invalid call status")
        if self.status == "running" and not self.task:
            raise ValueError("A running result requires a task reference")
        if self.status == "failed" and not self.error:
            raise ValueError("A failed result requires an error")
        if self.error and (not isinstance(self.error, Error) or not isinstance(self.error.code, str) or not self.error.code
                           or not isinstance(self.error.message, str) or type(self.error.retryable) is not bool
                           or not isinstance(self.error.details, dict)):
            raise ValueError("Invalid structured error")
        if self.status in {"completed", "running"} and self.error:
            raise ValueError("Successful/running results cannot include errors")
        for ref in (*self.refs, *self.artifacts):
            if not isinstance(ref, ObjectRef) or not all(isinstance(v, str) and v for v in (ref.kind, ref.owner, ref.id)):
                raise ValueError("Invalid object reference")
        if self.task and (not isinstance(self.task, TaskRef) or not self.task.owner or not self.task.id):
            raise ValueError("Invalid task reference")
        json.dumps(asdict(self), allow_nan=False)


@dataclass(frozen=True)
class Export:
    """A method within a capability, optionally advertised as an agent tool."""
    capability: str
    operation: str
    handler: Callable  # (ModuleCallContext, arguments: Mapping) -> Result
    input_schema: dict
    output_schema: dict
    permissions: tuple[str, ...] = ()
    tool_name: str = ""
    description: str = ""
    execution_role: str | None = None

    def validate(self):
        if not self.capability or not self.operation or not callable(self.handler):
            raise ValueError("Invalid capability operation")
        Draft202012Validator.check_schema(self.input_schema)
        Draft202012Validator.check_schema(self.output_schema)
        # External schemas must be self-contained; no network schema resolution.
        def check(value):
            if isinstance(value, dict):
                if "$ref" in value and not str(value["$ref"]).startswith("#"):
                    raise ValueError("Remote JSON Schema references are not permitted")
                for v in value.values(): check(v)
            elif isinstance(value, list):
                for v in value: check(v)
        check(self.input_schema); check(self.output_schema)


@dataclass
class Module:
    manifest: Manifest
    exports: tuple[Export, ...] = ()
    task_handler: Callable | None = None  # (context, operation, TaskRef) -> Result
    close_handler: Callable | None = None
    resolve_handler: Callable | None = None

    def close(self):
        if self.close_handler:
            self.close_handler()


def validate_payload(schema: dict, value):
    Draft202012Validator(schema).validate(value)


def native_factory(manifest):
    """Publish a factory's versioned requirements before initialization."""
    def decorate(factory):
        factory.manifest = manifest
        return factory
    return decorate
