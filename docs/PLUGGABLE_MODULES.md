# EvoCore module framework

Versioned native manifests, schemas, restricted contexts, permissions and task/reference contracts
are documented in [Module contracts](MODULE_CONTRACTS.md). The original factories below remain
compatibility adapters; their broad context is not the interface for new native modules.

`ResearchKernel` owns model/tool execution, event delivery and cancellation, and loads `ModuleRegistry`
at startup (`workspace/.research_modules`) even when `ResearchHarnessPlugin` is disabled.
`ModuleRegistry` supplies independently replaceable business services that may also expose runtime
hooks (`system_prompt_section`, `before_model_call`, `before_tool_call`, `after_tool_call`,
`after_agent_finish`, `tools`). No scientific stages or action order are enforced.
The researcher decides when to finish. Finish hooks only record local diagnostics; they cannot
restart the agent. There is no independent final LLM reviewer. Completion records
`acceptance="not_reviewed"`; unresolved local checks appear in `summary.json.delivery_issues`.
Legacy YAML reviewer settings are ignored on load.

## Composition

`system/bootstrap.py` creates the existing store/artifact infrastructure and passes it to
module factories. `modules/registry.py` loads factories from trusted application configuration,
checks required methods, detects missing/circular dependencies and caches module instances.
`modules/contracts.py` defines the public interfaces. `modules/tools.py` adapts these interfaces
to tool schemas. The existing Harness plugin bridges the services to the runtime; its legacy
study-scoped tools remain available for compatibility.

```yaml
modules:
  literature:
    enabled: true
    factory: evo_core.modules.local:literature_factory
    options: {timeout: 25}
  notebook: {enabled: true}
  coding:
    enabled: true
    factory: evo_core.modules.openhands:coding_factory
    options: {require_review: true}
  experiments: {enabled: true, requires: [coding]}
  context: {enabled: true}
  tracing: {enabled: true}
```

Replace a factory with `my_package.adapters:make_module`; the callable accepts
`(ModuleContext, ModuleRegistry, options)` and returns the documented interface.
Additional module names can be configured with a factory and optional dependencies.
An implementation may expose `tools(ctx) -> list[ToolDef]` to contribute new tools.
Factories are trusted code, not model-controlled inputs. Missing interfaces/dependencies
fail at startup. Duplicate module tool names fail during assembly.

## Initial implementations

| Module | Implementation and tools |
|---|---|
| literature | Existing search/full-text provider adapted to a SQLite paper-card library, markdown cards (`cards/`), metadata (`authors`, `year`, `venue`, `doi`, `tags`), `focus_query` excerpts and CAS content cache. `find_papers`, `add_paper`, `list_papers`, `annotate_paper`, `read_library_paper`. Admitted/read papers also sync to study `source` records when Harness is active. |
| notebook | Project-scoped SQLite documents with revision history, markdown mirrors (`notebook_md/`), compact zero-cold-start prompt index (`system_prompt_section`), and bidirectional sync with Harness study notes. `write_long_term_note`, `search_long_term_notes`, `read_long_term_note`. |
| coding | Default YAML uses the real OpenHands SDK Agent/Conversation and official file editor, with persistent task workspaces, SDK state and context condensation. Existing CAS snapshots, normalized run specifications and researcher approval are retained. See [OpenHands](OPENHANDS_CODER.md). |
| experiments | Self-contained adapter supporting `submit`, `get`, `list_jobs`, `verify`, `invalidate`, `annotate`, and local/Docker workers (with standalone store fallback when Harness is disabled). `annotate_experiment` binds observations/notes directly to experiment IDs; `delivery_issues` accepts full or unambiguous 12+ char short `exp_*` prefixes. |
| context | Independent SQLite role sessions (`load`/`save`) plus EvoScientist-style `offload_tool_output` (`tool_logs/`) and two-stage `edit_and_compact` (clearing older tool results beyond 50% window + archiving evicted turns to `conversation_history/{session}_{role}.md` and summarizing at 72%) for both `researcher` and `coder`. |
| tracing | Adapter around JSONL/optional MLflow tracker (`log_span`, `log_job`, `log_llm_call`) covering both `main` researcher calls and `subagent:*` calls (plus `subagent-*.json` request dumps under `run_dir/requests/`). |

Project modules live under `<workspace>/.research_modules/`, independently of run_meta.
Library and long-term notes survive new runs in the same workspace; session histories use
run ID plus role to avoid mixing unrelated conversations. Use a new workspace for an isolated project.
Disabling a module removes its new tools. Legacy experiment tools explicitly refuse submissions
when experiments is disabled. Disabling experiments with coding enabled still permits code review.
With default dependencies, disabling coding requires disabling experiments or explicitly selecting
another execution-authorization arrangement. Harness still has its legacy run ledger and event log.

## Researcher and coder

The main EvoCore agent acts as researcher. Default YAML disables the original local subagent
plugin and sends `implement_coding_task` to OpenHands, with independent task-specific history.
The legacy local delegation roles and ContextModule-managed coder dialogue remain opt-in;
neither coding backend copies the researcher's message history.
Coder has editing/submission tools but cannot approve code or execute commands/experiments.
When `coding.options.require_review: true` is active together with the `coder` subagent role, the
main researcher agent is also blocked from writing/editing `.py` files or calling `submit_code`
directly, forcing real separation between design/review (researcher) and implementation (coder).
The researcher inspects frozen source, approves it via `review_code_submission`, and calls
`submit_experiment` with `code_submission_id` (inheriting approved `files` and `run` defaults when
omitted). The default YAML selects the real OpenHands SDK via
`evo_core.modules.openhands:coding_factory`; see [OpenHands adapter](OPENHANDS_CODER.md).
It scans actual source with `prepare_coding_submission` rather than relying on a model-written file manifest.
Select `evo_core.modules.local:coding_factory` to restore the old local Coder.
Other third-party agents may replace `CodingModule.implement` or use a runner.

The default YAML requires review. Bare `CoreConfig()` retains the pre-module direct execution
behavior for old API callers; select `coding.options.require_review: true` to use the new policy.
Passing unit-test receipts are persisted per frozen submission, not per whole working directory.
Use test_paths to distinguish unit tests from result-dependent check files. Pytest checkers are
executed by the shared host runner and translated to the common checks.json format.
The MCP adapter exposes the same services for optional integrations but is not required to run EvoCore.

## Deliberate first-version limits

These are simple local implementations, not claims of PaperQA/Temporal integration. Existing
code is adapted rather than duplicated. SQLite document search is lexical, not semantic.
The default OpenHands coder uses its SDK context condenser and persisted conversation. Experiment
worker pools remain in process, and factory replacement occurs at startup, not while tasks are running.
The main and coder use sequential delegation, not concurrent autonomous execution. Legacy
Harness study-record tools still coexist with the new project-level services.

`tests/test_modules.py` tests replacement, disabled modules, startup validation, lazy paper
reading, persistence, researcher/coder isolation, unauthorized execution and reviewed-version
execution against actual Python subprocesses. Existing workflow and MCP regressions remain.
