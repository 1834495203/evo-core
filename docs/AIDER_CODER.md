# Aider coding backend

This optional legacy adapter selects the installed Aider CLI. The default application
now selects [OpenHands](OPENHANDS_CODER.md). Aider owns its code-editing dialogue, diff application,
repository map and history. EvoCore only adapts its input/output to the existing
code-submission and experiment interfaces. No kernel changes are needed.

```yaml
modules:
  coding:
    factory: evo_core.modules.aider:coding_factory
    options:
      require_review: true
      command: [aider]
      map_tokens: 2048
      timeout_seconds: null
```

Install Aider separately (`aider-chat`; validated here with 0.86.1) and ensure
`aider` and Git are on PATH. Select `evo_core.modules.local:coding_factory` to
switch back. Missing Aider fails explicitly; there is no silent local fallback.
The model defaults to the configured auxiliary/main model, so replacing the
coding backend does not require replacing the model.
The default YAML disables the original `subagents` plugin, so `delegate_task`
cannot accidentally invoke the old handwritten Coder loop. Enable it explicitly
when switching back to the local coding factory or adding other local roles.

Each task has independent Aider chat history. Each invocation edits a fresh copy
of project source, with an isolated Git index for Aider's repository map. Internal
Harness state, .env files, prior results and generated artifacts are not copied.
The CLI runs with explicit configuration, no automatic lint/test/shell suggestions,
no commits and no update/analytics requests. This staged directory is a file
publication boundary, not an OS sandbox for an untrusted executable.

Aider writes `coder_handoff.json` with `files`, optional `test_paths`, and the exact
`ExperimentSpec` fields under `run`. The adapter validates and freezes source,
publishes it to the project, and returns an **unapproved** submission. Existing
researcher review, exact-version authorization, optional checkers and experiment
execution stay in their modules. Aider cannot approve a submission or register
an experiment through this interface.

An ephemeral loopback OpenAI relay forwards Aider's non-streaming requests to the
configured provider. Only a local token is given to the CLI. Actual requests,
responses, timings and provider usage are archived in Harness traces as SDK calls
under `subagent:coder`; API failures are recorded explicitly. CLI output, original
request, handoff and task chat history are retained under
`.research_modules/coding/aider_sessions/<task_id>/`. Progress events include the
log path, elapsed time and actual completed HTTP request count.

Runtime artifacts must share the top-level directory of `metrics_path`, normally
`outputs/`, because that directory is restored for post-run verification. The
adapter communicates this existing experiment contract to Aider.

Official references: [CLI scripting](https://aider.chat/docs/scripting.html),
[OpenAI-compatible APIs](https://aider.chat/docs/llms/openai-compat.html),
[CLI options](https://aider.chat/docs/config/options.html).

Live integration check (uses the configured model credentials):

```powershell
$env:PYTHONPATH='evo-core'
.\EvoScientist\.venv\Scripts\python.exe -B evo-core/scripts/smoke_aider.py --run-root evo-core/runs/aider-smoke
```
