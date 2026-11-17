"""A single typed call boundary, shared by native and compatibility tools."""
import json
from dataclasses import asdict
from pathlib import Path
from .contracts import Result, ObjectRef, validate_payload
from .host import call_scope


def invoke_tool(host, tool, arguments, invoke):
    """Return (canonical result, legacy presentation). Identity is never user supplied."""
    identity = host.identify()
    with call_scope(identity):
        permissions = list(tool.permissions)
        # Resource-specific policy is attached to file tools at their adapter boundary.
        resolver = getattr(tool, "permission_resolver", None)
        if resolver:
            permissions.extend(resolver(arguments))
        if not host.policy.permits(identity.actor_id, permissions):
            result = Result.failure("permission_denied", f"Missing permissions: {permissions}")
            return result, f"[TOOL BLOCKED by permissions] {result.error.message}"
        if host.cancellation.requested():
            return Result(status="cancelled"), "[TOOL BLOCKED by cancellation] Call cancelled"
        try:
            validate_payload(tool.parameters, arguments)
        except Exception as exc:
            result = Result.failure("schema_invalid", str(exc))
            return result, f"Tool call argument error: {exc}"
        encoded = json.dumps(arguments, ensure_ascii=False)
        argument_ref = host.artifacts.put(encoded.encode())
        host.emit("tool_start", identity=asdict(identity), tool=tool.name, owner=tool.owner,
                  arguments_preview=encoded[:600], arguments_ref=asdict(argument_ref))
        try:
            raw = invoke()
            if isinstance(raw, Result):
                raw.validate()
                result = raw
                presentation = json.dumps(result.wire(), ensure_ascii=False)
            else:
                presentation = str(raw)
                if presentation.startswith(("[TOOL ERROR]", "Error:", "[TOOL BLOCKED", "Tool call argument error")):
                    result = Result.failure("tool_failed", presentation)
                else:
                    try: data = json.loads(presentation)
                    except (ValueError, TypeError): data = presentation
                    refs = ()
                    if tool.reference_kind and isinstance(data, dict) and (data.get("id") or data.get("job_id")):
                        refs = (ObjectRef(tool.reference_kind, tool.owner, data.get("id") or data["job_id"]),)
                    result = Result(data=data, refs=refs)
            if result.status == "completed" and tool.output_schema is not None:
                validate_payload(tool.output_schema, result.data)
        except Exception as exc:
            result = Result.failure("provider_error", f"{type(exc).__name__}: {exc}")
            presentation = f"[TOOL ERROR] {tool.name}: {result.error.message}"
        result_ref = host.artifacts.put(json.dumps(result.wire(), ensure_ascii=False).encode())
        host.emit("tool_completed", identity=asdict(identity), tool=tool.name, owner=tool.owner,
                  status=result.status, result_ref=asdict(result_ref))
        return result, presentation
