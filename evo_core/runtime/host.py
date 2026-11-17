"""Narrow host interfaces. New modules never receive ResearchSystem/HookContext."""
from contextvars import ContextVar
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from threading import Event
from uuid import uuid4
import hashlib
import json
import os
import re

from .contracts import Identity, Result, ObjectRef, TaskRef

CURRENT_CALL = ContextVar("evo_call", default=None)
EXPORTING_TRACE = ContextVar("evo_exporting_trace", default=False)


@contextmanager
def call_scope(identity):
    token = CURRENT_CALL.set(identity)
    try:
        yield
    finally:
        CURRENT_CALL.reset(token)


class NamespaceStorage:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, name):
        target = (self.root / name).resolve()
        if not target.is_relative_to(self.root):
            raise ValueError("Storage path escapes module namespace")
        return target

    def put(self, key, value):
        target = self.path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{uuid4().hex[:12]}.tmp")
        try:
            temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    def get(self, key):
        return json.loads(self.path(key).read_text(encoding="utf-8"))


class Artifacts:
    def __init__(self, store):
        self.__store = store

    def put(self, data: bytes):
        return ObjectRef("artifact", "host.artifacts", self.__store.put(data))

    def read(self, ref: ObjectRef):
        if ref.kind != "artifact" or ref.owner != "host.artifacts":
            raise ValueError("Artifact belongs to another store")
        return self.__store.read(ref.id)


class Cancellation:
    def __init__(self):
        self.__event = Event()

    def request(self):
        self.__event.set()

    def requested(self):
        return self.__event.is_set()

    def check(self):
        if self.requested():
            raise InterruptedError("Call cancelled")


class CancellationView:
    def __init__(self, token): self.__token = token
    def requested(self): return self.__token.requested()
    def check(self): return self.__token.check()


class Workspace:
    def __init__(self, root, host):
        self.__root, self.__host = Path(root).resolve(), host

    def __path(self, name):
        target = (self.__root / name).resolve()
        if not target.is_relative_to(self.__root): raise PermissionError("Workspace path escapes root")
        return target

    def read(self, name):
        identity = CURRENT_CALL.get() or self.__host.identity
        if not self.__host.policy.permits(identity.actor_id, ("workspace.read",)): raise PermissionError("workspace.read")
        return self.__path(name).read_text(encoding="utf-8")

    def write(self, name, text):
        identity = CURRENT_CALL.get() or self.__host.identity
        if not self.__host.policy.permits(identity.actor_id, self.__host.write_permissions({"file_path": name})):
            raise PermissionError("Workspace write not granted")
        target = self.__path(name); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        if self.__host.observe_file: self.__host.observe_file(name, text)


class ModelPort:
    def __init__(self, host): self.__host = host
    def complete(self, *args, **kwargs):
        if self.__host.models is None: raise RuntimeError("Model service unavailable")
        return self.__host.models.complete(*args, **kwargs)

    def report_sdk_call(self, **usage):
        if self.__host.models is None: raise RuntimeError("Model service unavailable")
        return self.__host.models.report_sdk_call(**usage)


class Policy:
    """Role grants belong to the host, never to a provider implementation."""
    def __init__(self, grants=None, *, strict=False, aliases=None, delegations=None):
        self.grants = grants or {}
        self.strict = strict
        self.aliases = aliases or {}
        self.delegations = delegations or {}

    def permits(self, actor, permissions):
        role = self.aliases.get(actor, actor)
        allowed = self.grants.get(role, set() if self.strict else {"*"})
        return "*" in allowed or set(permissions).issubset(allowed)

    def can_delegate(self, actor, target):
        role = self.aliases.get(actor, actor)
        allowed = self.delegations.get(role, ())
        return role == target or "*" in allowed or target in allowed


@dataclass(frozen=True)
class ModuleContext:
    module_id: str
    config: object
    storage: NamespaceStorage
    artifacts: Artifacts
    capabilities: object
    cancellation: Cancellation
    models: object
    emit: object
    secrets: object
    workspace: Workspace


@dataclass(frozen=True)
class CallContext:
    identity: Identity
    module: ModuleContext


class Host:
    def __init__(self, root, artifacts, *, emit=None, models=None, secret_provider=None, workspace=None):
        self.root = Path(root).resolve()
        self.artifacts = Artifacts(artifacts)
        self.models = models
        self.emit = emit or (lambda *a, **k: None)
        self.secret_provider = secret_provider or os.environ.get
        self.cancellation = Cancellation()
        self.policy = Policy()
        self.write_permissions = lambda arguments: ("workspace.write",)
        self.identity = Identity(hashlib.sha256(str(self.root).encode()).hexdigest()[:16], "", "", "main", "")
        self.workspace = Workspace(workspace or self.root, self)
        self.observe_file = None

    def identify(self, actor=None):
        parent = CURRENT_CALL.get()
        return replace(parent or self.identity, actor_id=actor or (parent.actor_id if parent else self.identity.actor_id),
                       call_id=f"call_{uuid4().hex}", parent_call_id=parent.call_id if parent and parent.call_id else None)

    def context(self, manifest, options, capabilities):
        namespace = hashlib.sha256(manifest.id.encode()).hexdigest()[:20]
        permitted_secrets = frozenset(options.get("secret_names", ()))
        def secret(name):
            if name not in permitted_secrets:
                raise PermissionError("Secret not granted to this module")
            return self.secret_provider(name)
        def emit(kind, **payload):
            if kind not in manifest.events:
                raise ValueError(f"Undeclared event: {kind}")
            identity = CURRENT_CALL.get() or self.identity
            self.emit("module_event", module=manifest.id, event_type=kind, identity=identity.__dict__, payload=payload)
        return ModuleContext(manifest.id, MappingProxyType(json.loads(json.dumps(options))), NamespaceStorage(self.root / "providers" / namespace),
                             self.artifacts, capabilities, CancellationView(self.cancellation), ModelPort(self), emit, secret, self.workspace)
