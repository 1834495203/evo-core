# Module contract v1

The runtime supplies identities, permission checks, capability routing, schema validation,
owner-qualified references, task routing, cancellation, model telemetry and namespaced storage.
It does not prescribe scientific stages or require a final reviewer.

## Native module initialization

Use `contract: native` and a `factory: package.module:function` in application configuration.
Decorate the factory with `native_factory(Manifest(...))`. The manifest is inspected before
initialization, so dependencies, versions, duplicate providers and cycles fail at startup.
Capability versions are positive integer interface versions; module package versions are strings.
Only one provider per capability is selected. Replace/disable the previous provider explicitly.

The factory accepts exactly one `evo_core.runtime.host.ModuleContext` and returns `Module`.
It receives no kernel, ResearchSystem, mutable RunState, raw model client or other modules' options.

| Facility | Public interface |
| --- | --- |
| Own configuration | `context.config`, an independent configuration mapping |
| Own durable namespace | `context.storage.get(key)`, `put(key, json_value)`, `path(relative_name)` |
| Artifacts | `context.artifacts.put(bytes) -> ObjectRef`, `read(ref) -> bytes` |
| Declared dependencies | `context.capabilities.call(capability, operation, arguments) -> Result` |
| Workspace text files | `context.workspace.read(path)`, `write(path, text)`, checked against caller grants |
| Cancellation | `context.cancellation.requested()`, `check()`; modules cannot request global cancellation |
| Shared model interface | `context.models.complete(messages, ...)`; unavailable services raise explicitly |
| Events | `context.emit(declared_event_name, **payload)`; host attaches actual identity |
| Secrets | `context.secrets(name)` only for configured `options.secret_names` |

Secrets default to the host environment. They are not injected into ordinary tool results.
Native SDK adapters can use their own backend, filesystem, database and model client. They must
propagate identity/trace context and report native SDK calls when not using the shared model port;
the runtime cannot observe opaque external calls automatically.
Use `context.models.report_sdk_call(input_tokens=..., output_tokens=..., cached_tokens=...,
duration_ms=..., request_ref=..., response_ref=...)` from the adapter's SDK observer callback.
The host stamps the effective actor/parent identity and includes these calls in total usage/counts.

## Exports and calls

An `Export` declares capability, operation, input JSON Schema, output JSON Schema, required
permissions and `(call_context, arguments) -> Result` handler. Set `tool_name` and `description`
to expose it to an agent. Tools use the same authorized dispatcher as internal calls.
Schemas use JSON Schema 2020-12 through `jsonschema`; remote schema references are rejected.
Output schemas describe `Result.data`. Define `additionalProperties: false` to reject unexpected
arguments. Input validation runs before handler execution; success output is validated afterward.
For delegated SDK agents, set `Export.execution_role="coder"`. The host first checks the caller's
permission to request the operation and its allowed delegation, then executes the handler and
nested model/tool calls under the Coder identity. A provider cannot delegate itself to `host`.
`role_delegations` overrides the application's default researcher-to-Coder delegation.
Malformed provider output is a structured error, not a request to rewrite research.

`call_context.identity` is immutable and supplied by the host:
`project_id`, `run_id`, `session_id`, `actor_id`, `call_id`, `parent_call_id`.
Arguments cannot replace it. Dependency calls preserve the actor and create child call identities.

Result wire format:

```json
{
  "status": "completed",
  "data": {"id": "dataset_123"},
  "refs": [{"kind": "dataset", "owner": "example.dataset_catalog", "id": "dataset_123", "version": null}],
  "artifacts": [],
  "task": null,
  "error": null
}
```

Statuses are `completed`, `running`, `failed`, `cancelled`. A running result requires a
`TaskRef(owner,id)`; failed results require `Error(code,message,retryable,details)`.
All payloads must be finite JSON values. Typical host error codes are `schema_invalid`,
`permission_denied`, `dependency_undeclared`, `capability_missing`, `operation_missing`,
`operation_unsupported`, `provider_error`, `reference_missing`.
Retryability is information; it does not trigger automatic retries. Adapters own operation
idempotency and must not relaunch remote jobs simply because a request is retried.

## References and background tasks

Declare `reference_kinds` and implement `Module.resolve_handler(context, ObjectRef)`.
`registry.resolve(ref)` and `resolve_module_ref` route by owner, not by knowledge of a database.
Artifact references use owner `host.artifacts` and are byte-addressed by hash.
The generic reference tool encodes bytes as base64; providers should return concise cards or
excerpt references instead of large document payloads.

Declare supported `task_operations` (`get`, `result`, `cancel`) and implement
`Module.task_handler(context, operation, TaskRef)`. Generic tools are `get_module_task`,
`get_module_result`, `cancel_module_task`. Unsupported operations return a structured error.
Providers own durable task state, workers, cancellation semantics and recovery. A host task
reference does not imply a durable scheduler or guaranteed cancellation.

## Permissions and trusted adapters

`strict_roles` and `role_permissions` select host-owned grants. Explicit per-role grants replace
that role's defaults. The generic ACL implementation does not know scientific modules;
application defaults are in `modules/adapters.py`.

In strict mode the researcher can request/review/test code and submit experiments, but cannot
edit code, call `submit_code` or execute raw commands. The Coder can edit/submit but cannot approve
or execute. Q&A has read permissions. This applies with Harness enabled or disabled and through
MCP. File tools check resource permissions; Python files and other executable/config formats
require `code.edit`, internal module paths require `runtime.admin`.

Use `test_code_submission` to run pytest in a frozen submission directory. Test results authorize
formal launch for the exact submission manifest and survive restart. Unrelated workspace Python
files do not invalidate that receipt; changes to a submitted file still require resubmission and
approval. `submit_code.test_paths` declares unit-test files; `test_code_submission.test_paths`
can select them explicitly. Post-run checks that need experiment outputs are separate. The default
discovery excludes the result-check file when other unit-test files are present.

`run.check_runner` is `auto`, `pytest` or `checks_json`. Auto recognizes `test_*.py`/`*_test.py` and
executes pytest, rather than running the file as an ordinary script. The host translates fresh
JUnit results into checks.json; zero collected tests, all-skipped suites, errors and failures do
not pass. Other scripts must produce a nonempty outputs/checks.json. Both code tests and result
tests use host-managed temporary directories inside their execution snapshots. `wait=True` waits
for the configured result checker as well as the experiment process.

`get_execution_environment` reports packages in the actual worker interpreter without installing
anything or registering a probe experiment.
The local executor and in-process adapters are trusted development code, not an OS security
sandbox. Use an isolated backend for untrusted code. A trusted Python plugin is not isolated
from the Python process by these API permissions.

## Existing modules and migration

The six existing services are wrapped by manifests and capability exports:

| Existing slot | Capability |
| --- | --- |
| literature | `literature.library@1` |
| notebook | `notes.store@1` |
| coding | `code.implementation@1` |
| experiments | `experiment.execution@1` |
| context | `conversation.context@1` |
| tracing | `observability.trace@1` |

Paper searches are durably journaled separately from the admitted library. `list_paper_searches`
and `read_paper_search` retrieve candidates; `admit_paper(candidate_id, notes)` explicitly selects
one without retyping bibliographic information. Search records are not admitted papers or proof
that full text was read. Search responses are also archived in research bundles.

Workspace Markdown/text writing and editing is mirrored through notes.store as versioned document
drafts, including native workspace writes. Generated results and internal module files are
excluded. This bridge works with a replaced notes provider, preserves owner-qualified references,
and records errors visibly. Capturing a document does not infer scientific claims, admit papers,
or approve the report. There are no required writing or literature checkpoints.

`list_modules` describes selected providers, operations, schemas and permissions. Scientific
tools and old record projections live in application adapters, not in the generic dispatcher.
Native extensions may use any slot name. Context and trace runtime consumers resolve capabilities.
Worker factories and other in-process legacy methods are not JSON RPC operations.

Old factories remain in the explicit `legacy` compatibility mode and still receive their old
context. `get(name)` and old runtime hooks are compatibility APIs; new modules should use native
contracts. Old tool string presentations are retained, while `HarnessSession.invoke` or
`execute_tool_with_hooks(..., return_result=True)` returns the canonical `Result` for every tool.
Native tool results are always returned as envelopes. A migration of stored data is a separate
adapter responsibility; a new provider does not automatically understand the old provider's IDs.

The legacy local experiment backend still uses an in-process worker pool and the research ledger's
normalized job schema. Replacing that backend inside the old Harness requires a facade/projection
adapter for its queue and evidence records. A native provider may instead expose independent task
tools, references and events through the generic protocol, without pretending to implement Python
worker factory objects over JSON. The optional [OpenHands coder adapter](OPENHANDS_CODER.md)
uses the real SDK through the existing coding boundary; no Zotero/ClearML integration is claimed.

## Tracing

The shared model gateway records researcher, Coder, compaction and companion Q&A calls. Summary
fields `total_model_calls` and `model_calls_by_actor` include gateway calls; the older `model_calls`
field remains the main-loop count. Request/response data is archived and events carry references.
Tool events include start, completion, input preview, references and parent identity. Trace exporter
errors are isolated into `run_dir/export_errors.jsonl`. No independent reviewer is installed.

## Runnable example

`modules/example_catalog.py` adds `datasets.catalog@1` with no kernel changes. It declares its
dependency on `notes.store`, validates dataset metadata, persists a card, creates a note, emits
`dataset.registered` and returns a resolvable dataset reference. It does not download data.

```powershell
# From evo-core/; uses the application's configured model credentials.
..\EvoScientist\.venv\Scripts\python.exe -m evo_core run --config examples/native_modules.yaml --prompt "Register a dataset card and describe where it is stored."
```

`tests/test_contracts.py` also runs this module without a network/model, validates dependency and
permission failures, exercises task restoration/cancellation, and replaces context/trace providers
under unrelated slot names. Interface tests do not establish long-running scientific quality.
