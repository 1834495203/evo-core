"""A runnable native extension: persists dataset cards and uses a declared notes capability.

This is intentionally a small local module, not an external dataset platform integration.
"""
from uuid import uuid4
from ..runtime import Manifest, Capability, Requirement, Export, Module, Result, ObjectRef, native_factory

MANIFEST = Manifest("example.dataset_catalog", provides=(Capability("datasets.catalog"),),
                    requires=(Requirement("notes.store"),), events=("dataset.registered",), reference_kinds=("dataset",))


@native_factory(MANIFEST)
def make_catalog(context):
    def register(call, arguments):
        identifier = "dataset_" + uuid4().hex
        record = {"id": identifier, "title": arguments["title"], "uri": arguments["uri"]}
        context.storage.put(identifier + ".json", record)
        note = context.capabilities.call("notes.store", "put", {"kind": "dataset", "text": f"Dataset: {record['title']} ({record['uri']})"})
        if note.error:
            return note
        context.emit("dataset.registered", dataset_id=identifier)
        return Result(data=record, refs=(ObjectRef("dataset", MANIFEST.id, identifier),))

    def resolve(call, ref):
        return Result(data=context.storage.get(ref.id + ".json"))

    return Module(MANIFEST, exports=(Export("datasets.catalog", "register", register,
        {"type": "object", "properties": {"title": {"type": "string", "minLength": 1}, "uri": {"type": "string", "minLength": 1}},
         "required": ["title", "uri"], "additionalProperties": False},
        {"type": "object", "properties": {"id": {"type": "string"}, "title": {"type": "string"}, "uri": {"type": "string"}},
         "required": ["id", "title", "uri"], "additionalProperties": False},
        permissions=("notes.write",), tool_name="register_dataset", description="Register a dataset card with a source URI; does not download data."),),
        resolve_handler=resolve)
