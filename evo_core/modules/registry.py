"""Small composition mechanism. Factory paths are trusted application configuration."""
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable
from ..runtime.contracts import Module, Result, Manifest, TaskRef, validate_payload
from ..runtime.host import Host, CallContext, call_scope
from .adapters import local_manifest, local_exports, validate_legacy


@dataclass(frozen=True)
class ModuleContext:
    root: Path
    services: Any
    host: Any = None


class ModuleRegistry:
    def __init__(self, context: ModuleContext):
        self.context = context
        self._instances = {}
        self._factories = {}
        self._building = set()
        self._manifests = {}
        self._formats = {}
        self._capabilities = {}
        self._exports = {}
        self._contexts = {}
        from ..system.storage.artifacts import ArtifactStore
        self.host = context.host or Host(context.root, ArtifactStore(context.root / "host_artifacts"))

    def register(self, name: str, factory: Callable, *, requires=(), options=None, contract="legacy"):
        if name in self._factories or name in self._instances:
            raise ValueError(f"Duplicate module: {name}")
        if contract not in {"legacy", "native"}:
            raise ValueError("contract must be native or legacy")
        manifest = getattr(factory, "manifest", None) if contract == "native" else local_manifest(name, factory)
        if not isinstance(manifest, Manifest):
            raise ValueError("Native factory must declare a Manifest before initialization")
        manifest.validate()
        if any(m.id == manifest.id for m in self._manifests.values()):
            raise ValueError(f"Duplicate module identity: {manifest.id}")
        for cap in manifest.provides:
            if cap.name in self._capabilities:
                raise ValueError(f"Ambiguous capability provider: {cap.name}")
        self._factories[name] = (factory, tuple(requires), options or {})
        for cap in manifest.provides:
            self._capabilities[cap.name] = (name, cap.version)
        self._manifests[name] = manifest
        self._formats[name] = contract

    def get(self, name: str):
        if name in self._instances:
            return self._instances[name]
        if name not in self._factories:
            raise ValueError(f"Module is disabled or missing: {name}")
        if name in self._building:
            raise ValueError(f"Circular module dependency: {name}")
        factory, dependencies, options = self._factories[name]
        self._building.add(name)
        instance = None
        try:
            for dep in dependencies:
                self.get(dep)
            manifest = self._manifests[name]
            for req in manifest.requires:
                provider = self._capabilities.get(req.name)
                if provider is None:
                    if req.optional: continue
                    raise ValueError(f"Missing required capability: {req.name}@{req.version}")
                if provider[1] != req.version:
                    raise ValueError(f"Incompatible capability: {req.name}@{req.version}")
                self.get(provider[0])
            if self._formats[name] == "native":
                ctx = self.host.context(manifest, options, CapabilityAccess(self, manifest.requires))
                self._contexts[name] = ctx
                instance = factory(ctx)
                if not isinstance(instance, Module) or instance.manifest != manifest:
                    raise ValueError("Factory returned an incompatible module manifest")
                exports = tuple(instance.exports)
                if manifest.task_operations and not callable(instance.task_handler):
                    raise ValueError("Declared task operations require a handler")
                if manifest.reference_kinds and not callable(instance.resolve_handler):
                    raise ValueError("Declared reference kinds require a resolver")
            else:
                instance = factory(self.context, self, options)
                validate_legacy(name, instance)
                self._contexts[name] = self.host.context(manifest, options, CapabilityAccess(self, manifest.requires))
                exports = tuple(local_exports(name, instance))
            if instance is None:
                raise ValueError(f"Module factory returned no implementation: {name}")
            exported = set()
            for export in exports:
                export.validate()
                key = (export.capability, export.operation)
                if export.capability not in {c.name for c in manifest.provides} or key in self._exports or key in exported:
                    raise ValueError(f"Undeclared/duplicate export: {key}")
                exported.add(key)
            for export in exports:
                self._exports[(export.capability, export.operation)] = (name, export)
            self._instances[name] = instance
            return instance
        except Exception:
            self._contexts.pop(name, None)
            close = getattr(instance, "close", None)
            if callable(close):
                try: close()
                except Exception as exc:
                    self.host.emit("module_close_error", module=self._manifests[name].id, error=str(exc))
            raise
        finally:
            self._building.remove(name)

    def has(self, name):
        return name in self._factories

    def has_legacy(self, name):
        return self.has(name) and not self.is_native(name)

    def has_capability(self, name, version=1):
        return name in self._capabilities and self._capabilities[name][1] == version

    def service(self, capability, *, actor=None):
        provider = self._capabilities.get(capability)
        if provider is None: raise ValueError(f"Missing capability: {capability}")
        instance = self.get(provider[0])
        if not self.is_native(provider[0]): return instance
        return CapabilityPort(self, capability, actor)

    def describe(self):
        from dataclasses import asdict
        return [{"name": name, "implementation": f"{factory.__module__}:{factory.__name__}",
                 "requires": list(deps), "contract": self._formats[name],
                 "manifest": asdict(self._manifests[name]),
                 "operations": [{"capability": e.capability, "operation": e.operation, "tool": e.tool_name,
                                 "input_schema": e.input_schema, "output_schema": e.output_schema,
                                 "permissions": list(e.permissions), "execution_role": e.execution_role}
                                for owner, e in self._exports.values() if owner == name]}
                for name, (factory, deps, _) in self._factories.items()]

    def is_native(self, name):
        return self._formats.get(name) == "native"

    def invoke(self, capability, operation, arguments, *, actor=None):
        provider = self._capabilities.get(capability)
        if provider is None:
            return Result.failure("capability_missing", capability)
        self.get(provider[0])
        entry = self._exports.get((capability, operation))
        if entry is None:
            return Result.failure("operation_missing", f"{capability}.{operation}")
        name, export = entry
        identity = self.host.identify(actor)
        if not self.host.policy.permits(identity.actor_id, export.permissions):
            return Result.failure("permission_denied", f"Missing permissions: {export.permissions}")
        if export.execution_role:
            if not self.host.policy.can_delegate(identity.actor_id, export.execution_role):
                return Result.failure("delegation_denied", export.execution_role)
            from dataclasses import replace
            identity = replace(identity, actor_id=export.execution_role)
        if self.host.cancellation.requested():
            return Result(status="cancelled")
        with call_scope(identity):
            self.host.emit("capability_start", identity=identity.__dict__, module=self._manifests[name].id,
                           capability=capability, operation=operation)
            try:
                validate_payload(export.input_schema, arguments)
                from types import MappingProxyType
                from .adapters import plain
                result = export.handler(CallContext(identity, self._contexts[name]), MappingProxyType(plain(arguments)))
                if not isinstance(result, Result):
                    raise ValueError("Capability handlers must return Result")
                result.validate()
                if result.status == "completed": validate_payload(export.output_schema, result.data)
            except Exception as exc:
                from jsonschema import ValidationError
                code = "schema_invalid" if isinstance(exc, ValidationError) else "provider_error"
                result = Result.failure(code, str(exc))
            self.host.emit("capability_end", identity=identity.__dict__, result=result.wire())
            return result

    def native_tools(self):
        from ..protocol import ToolDef
        tools = []
        for (cap, operation), (name, export) in self._exports.items():
            if not self.is_native(name) or not export.tool_name: continue
            def call(cap=cap, operation=operation, **arguments):
                return self.invoke(cap, operation, arguments)
            tools.append(ToolDef(export.tool_name, export.description, export.input_schema, call,
                                 permissions=export.permissions, owner=self._manifests[name].id,
                                 wire_result=True, output_schema=export.output_schema))
        return tools

    def task(self, ref: TaskRef, operation: str, *, actor=None):
        found = next((name for name, m in self._manifests.items() if m.id == ref.owner), None)
        if found is None: return Result.failure("task_owner_missing", ref.owner)
        manifest = self._manifests[found]
        if operation not in manifest.task_operations:
            return Result.failure("operation_unsupported", operation)
        identity = self.host.identify(actor)
        permissions = ("tasks.cancel",) if operation == "cancel" else ("tasks.read",)
        if not self.host.policy.permits(identity.actor_id, permissions):
            return Result.failure("permission_denied", operation)
        try:
            with call_scope(identity):
                self.host.emit("task_start", identity=identity.__dict__, owner=ref.owner, task_id=ref.id, operation=operation)
                if self.is_native(found):
                    result = self.get(found).task_handler(CallContext(identity, self._contexts[found]), operation, ref)
                else:
                    from .adapters import local_task
                    result = local_task(self.get(found), ref, operation)
                result.validate()
                self.host.emit("task_end", identity=identity.__dict__, status=result.status)
                return result
        except Exception as exc:
            self.host.emit("task_end", identity=identity.__dict__, status="failed", error=str(exc))
            return Result.failure("provider_error", str(exc))

    def resolve(self, ref, *, actor=None):
        identity = self.host.identify(actor)
        if not self.host.policy.permits(identity.actor_id, ("objects.read",)):
            return Result.failure("permission_denied", "objects.read")
        if ref.owner == "host.artifacts" and ref.kind == "artifact":
            try:
                import base64
                return Result(data={"encoding": "base64", "content": base64.b64encode(self.host.artifacts.read(ref)).decode()})
            except Exception as exc: return Result.failure("reference_missing", str(exc))
        found = next((name for name, m in self._manifests.items() if m.id == ref.owner), None)
        if found is None: return Result.failure("reference_owner_missing", ref.owner)
        if ref.kind not in self._manifests[found].reference_kinds:
            return Result.failure("reference_unsupported", ref.kind)
        try:
            with call_scope(identity):
                self.host.emit("reference_start", identity=identity.__dict__, owner=ref.owner, object_id=ref.id)
                instance = self.get(found)
                if self.is_native(found): result = instance.resolve_handler(CallContext(identity, self._contexts[found]), ref)
                else:
                    from .adapters import local_reference
                    result = local_reference(found, instance, ref)
                result.validate()
                self.host.emit("reference_end", identity=identity.__dict__, status=result.status)
                return result
        except Exception as exc:
            self.host.emit("reference_end", identity=identity.__dict__, status="failed", error=str(exc))
            return Result.failure("reference_missing", str(exc))

    def system_prompt_sections(self, state: Any, ctx: Any) -> list[str]:
        sections: list[str] = []
        for instance in self._instances.values():
            hook = getattr(instance, "system_prompt_section", None)
            if callable(hook):
                sec = hook(state, ctx)
                if sec and str(sec).strip():
                    sections.append(str(sec).strip())
        return sections

    def before_model_call(self, state: Any, messages: list[dict], ctx: Any) -> list[dict]:
        current = messages
        for instance in self._instances.values():
            hook = getattr(instance, "before_model_call", None)
            if callable(hook):
                current = hook(state, current, ctx)
        return current

    def before_tool_call(self, state: Any, tool_name: str, arguments: dict, ctx: Any) -> Any:
        from ..protocol import ToolCallDecision

        for name, instance in self._instances.items():
            hook = getattr(instance, "before_tool_call", None)
            if callable(hook):
                decision = hook(state, tool_name, arguments, ctx)
                if getattr(decision, "action", "allow") == "block":
                    if not getattr(decision, "source", ""):
                        return ToolCallDecision(action="block", feedback=decision.feedback, source=name)
                    return decision
        return ToolCallDecision()

    def after_tool_call(self, state: Any, tool_name: str, arguments: dict, result: str, ctx: Any) -> None:
        for instance in self._instances.values():
            hook = getattr(instance, "after_tool_call", None)
            if callable(hook):
                hook(state, tool_name, arguments, result, ctx)

    def after_agent_finish(self, state: Any, final_text: str, ctx: Any) -> list[Any]:
        from ..protocol import TurnDecision

        decisions: list[Any] = []
        for name, instance in self._instances.items():
            hook = getattr(instance, "after_agent_finish", None)
            if callable(hook):
                try:
                    decisions.append(hook(state, final_text, ctx))
                except Exception as exc:
                    decisions.append(TurnDecision(source=name, feedback=f"Local finish check failed: {type(exc).__name__}: {exc}"))
        return decisions

    def close(self):
        for name, instance in reversed(list(self._instances.items())):
            close = getattr(instance, "close", None)
            if close:
                try: close()
                except Exception as exc:
                    self.host.emit("module_close_error", module=self._manifests[name].id, error=str(exc))


def import_factory(path):
    module, name = path.split(":", 1)
    return getattr(import_module(module), name)


DEFAULTS = {
    "literature": ("literature_factory", ()),
    "notebook": ("notebook_factory", ()),
    "coding": ("coding_factory", ()),
    "experiments": ("experiments_factory", ()),
    "context": ("context_factory", ()),
    "tracing": ("tracing_factory", ()),
}


def build_modules(root, services, specs=None, *, host=None):
    registry = ModuleRegistry(ModuleContext(Path(root), services, host))
    entries = {**DEFAULTS, **{name:(None, ()) for name in (specs or {}) if name not in DEFAULTS}}
    for name, (default, dependencies) in entries.items():
        spec = dict((specs or {}).get(name, {}))
        if not spec.get("enabled", True):
            continue
        path = spec.get("factory", f"evo_core.modules.local:{default}" if default else "")
        if not path:
            raise ValueError(f"Additional module {name} needs a factory path")
        registry.register(name, import_factory(path), requires=spec.get("requires", dependencies),
                          options=spec.get("options", {}), contract=spec.get("contract", "legacy"))
    # Validate selected implementations/dependencies at startup, not during research.
    try:
        for item in registry.describe():
            registry.get(item["name"])
    except Exception:
        registry.close()
        raise
    return registry


class CapabilityAccess:
    """Only declared dependencies are visible to a native module."""
    def __init__(self, registry, requirements):
        self.__registry = registry
        self.__requirements = {r.name: r for r in requirements}

    def call(self, capability, operation, arguments):
        if capability not in self.__requirements:
            return Result.failure("dependency_undeclared", capability)
        return self.__registry.invoke(capability, operation, arguments)


class CapabilityPort:
    """Compatibility facade for existing in-process consumers of a native capability."""
    def __init__(self, registry, capability, actor=None):
        self.__registry = registry
        self.__capability = capability
        self.__actor = actor

    def __getattr__(self, operation):
        entry = self.__registry._exports.get((self.__capability, operation))
        if entry is None: raise AttributeError(operation)
        def invoke(*args, **kwargs):
            names = list(entry[1].input_schema.get("properties", {}))
            if len(args) > len(names): raise TypeError("Too many positional arguments")
            from .adapters import plain
            params = plain({**dict(zip(names, args)), **{k: v for k, v in kwargs.items() if k in names}})
            result = self.__registry.invoke(self.__capability, operation, params, actor=self.__actor)
            if result.error: raise RuntimeError(f"{result.error.code}: {result.error.message}")
            return result.data
        return invoke
