# OpenHands coding module

The default configuration uses the real OpenHands Software Agent SDK 1.62.0 and
official tools 1.62.0. Its Agent/Conversation run the coding loop, file edits,
task tracker, persistence and context condenser. The adapter does not implement
another model loop. Aider and the original Coder remain opt-in factories; no
Codex runtime or account is used.

```yaml
modules:
  coding:
    factory: evo_core.modules.openhands:coding_factory
    options:
      require_review: true
      # Optional overrides:
      # python_executable: /path/to/openhands/python
      # state_root: /short/path/to/openhands-state
      # model: openai/model-name
      # max_output_tokens: 24000
      timeout_seconds: null
      max_iterations: null
```

Install the isolated Python 3.12+ dependency environment on Windows:

```powershell
.\evo-core\scripts\install_openhands.ps1
```

On other platforms, create `.module_envs/openhands` under evo-core and install
`openhands-sdk==1.62.0 openhands-tools==1.62.0` together. The adapter searches its
ancestors for that environment, including when launched from a source snapshot.
`EVO_OPENHANDS_PYTHON` overrides the interpreter. Main-agent and science Python
environments are unchanged. Missing SDK dependencies fail explicitly.

The existing model API credentials and auxiliary/main model are used. Output
tokens are explicitly passed to OpenHands from its module override or the main
configuration; unset per-task iteration limits map to the SDK's largest integer,
and there is no configured cost/time budget. These settings cannot remove a
provider's physical context/output limits. The SDK critic is disabled.

## Persistence and review

One coding task has one persistent workspace and SDK conversation UUID. A new
worker process restores that conversation and edits the same files. Failed,
cancelled and incomplete edits are retained. The short module state directory
`.module_state/openhands/<project-hash>/<task-hash>/` avoids Windows event filename
limits; the project coding directory stores a `location.json` pointer. Do not
delete either directory when intending to resume a task.

`implement_coding_task` returns an actual source inventory, changes, summary,
conversation ID and log location. It supports library-only tasks without forcing
an invented runnable entrypoint or a model-written handoff JSON.
`inspect_coding_workspace(task_id, file_path?)` reads the staged source.

When ready, `prepare_coding_submission(task_id, run, test_paths?)` takes the
researcher's exact ExperimentSpec fields, scans actual source, freezes it and
publishes code as an **unapproved** submission. Concurrent project edits cause
an explicit conflict. Existing review, source-version authorization and Harness
experiment tools remain responsible for formal execution. Research phases are
not prescribed by this adapter.

Only the official file editor, official task tracker and built-in FinishTool are
exposed. The file-editor executor is wrapped to enforce workspace bounds and
create parent directories. The coding worker has no terminal, experiment or
approval tool. This is a tool boundary, not an OS sandbox for a hostile SDK.

## Tracking

Native SDK callbacks save action/observation/error events and request/response
logs. The adapter forwards events with tool names and previews into Harness;
actual usage and latency from SDK completion telemetry update model-call totals
under `subagent:coder`. Unknown model pricing is not treated as a known cost.
Cancellation requests pause the SDK, with process-tree termination as a fallback;
already persisted events and source remain available.

Live multi-file implementation and process-restart check:

```powershell
$env:PYTHONPATH='evo-core'
.\EvoScientist\.venv\Scripts\python.exe -B evo-core/scripts/smoke_openhands.py --run-root evo-core/runs/openhands-check
.\EvoScientist\.venv\Scripts\python.exe -B evo-core/scripts/check_openhands_recovery.py --run-root evo-core/runs/openhands-recovery-check
```

Validation on 2026-11-17: a seven-file implementation was restored in a new
runtime, corrected using actual failed-test feedback, passed 57 generated unit
cases, and completed an archived Harness experiment and its result checks.
An independent live fixture interrupted the SDK after writing a marker file;
a new runtime restored the same conversation UUID, preserved the marker bytes,
and completed three remaining package files. These are integration fixtures,
not evidence that an autonomous research project will succeed.

References: [SDK](https://github.com/OpenHands/software-agent-sdk),
[persistence](https://docs.openhands.dev/sdk/guides/convo-persistence),
[context condenser](https://docs.openhands.dev/sdk/guides/context-condenser).
