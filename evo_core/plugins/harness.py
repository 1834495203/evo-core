"""Pluggable ResearchHarnessPlugin bridging ResearchSystem into ResearchKernel."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
from uuid import uuid4

from ..llm import redact_secret
from ..protocol import BasePlugin, HookContext, RunState, ToolCallDecision, ToolDef, TurnDecision
from ..system.bootstrap import ResearchSystem, build_system
from ..system.domain import ExperimentSpec, Job
from ..system.research.evidence_contract import delivery_issues
from ..system.research.instructions import HARNESS_INSTRUCTIONS
from ..system.research.tools import archive_tools


def _rel_posix(raw: str) -> str:
    return raw.strip().lstrip("/").replace("\\", "/")


def _resolve_science_python(ctx: HookContext) -> str:
    sci_dir = getattr(ctx.sandbox, "science_python_dir", None)
    if sci_dir:
        for name in ("python.exe", "python3", "python"):
            cand = Path(sci_dir) / name
            if cand.is_file():
                return str(cand)
    return sys.executable


class ResearchHarnessPlugin(BasePlugin):
    """Unifies externalized experiment harness, parallel execution, research ledger, and user steering."""

    name = "harness"

    def __init__(self) -> None:
        self._pool: Any = None
        self._futures: dict[str, Any] = {}
        self._py_exe: str = sys.executable

    def enabled(self, cfg: Any) -> bool:
        return bool(cfg.plugins.harness.enabled)

    def system_prompt_section(self, state: RunState, ctx: HookContext) -> str | None:
        coding_mod = (
            ctx.module_registry.get("coding")
            if ctx.module_registry is not None and ctx.module_registry.has("coding")
            else None
        )
        strict_coder = (
            coding_mod is not None
            and bool(getattr(coding_mod, "require_review", False))
            and bool(ctx.config.plugins.subagents.enabled)
            and "coder" in (ctx.config.plugins.subagents.roles or [])
        )
        coder_rule = (
            " When coding review with `coder` is enabled, the Researcher must NOT write/edit `.py` files or call `submit_code` directly: "
            "use `create_coding_task` + `implement_coding_task` (or `delegate_task(role='coder', ...)`), inspect and `review_code_submission`, "
            "verify the frozen submission with `test_code_submission`, then launch `submit_experiment(code_submission_id=...)`."
            if strict_coder
            else ""
        )
        return (
            "# Research Harness & Evidence Ledger\n" + HARNESS_INSTRUCTIONS + "\n"
            "Persist documents when useful with record_research_document; query_research_records/read_research_document recover history. archive_research_report stores a report.\n"
            "Project modules: find_papers returns candidates; add_paper explicitly admits paper cards; read_library_paper retrieves full text on demand (supports focus_query). "
            "Use write_long_term_note/search_long_term_notes/read_long_term_note for cross-run tasks, goals, ideas and experiment notes (`refs` link IDs), or annotate_experiment to attach notes directly to an experiment. "
            "Delegate implementations to coder via create_coding_task + implement_coding_task (or delegate_task); coder calls submit_code, researcher reviews via review_code_submission, then submit_experiment uses code_submission_id. Use list_modules to inspect implementations."
            + coder_rule + "\n"
            "Formal experiments, literature notes, observations, hypotheses, and decisions are tracked externally:\n"
            "- Use `record_research_note` for sources, reproductions, observations, hypotheses, methods, decisions, and limitations.\n"
            "- Optional: use set_research_focus to record a free-form focus label and next action.\n"
            "- Use `submit_experiment` (with `outputs/metrics.json` or custom `metrics_path`) to run frozen, reproducible experiments (`wait=false` allows parallel execution, then collect via `poll_experiments`).\n"
            "- Use verify_experiment with pytest tests or a check script. Auto mode runs test_*.py / *_test.py through pytest and generates checks.json; other scripts must write outputs/checks.json. Choose check_runner explicitly when needed. Invalidate results when bugs make them unusable.\n"
            "- Invalidated experiments CANNOT be cited as supporting evidence."
            "\n- With harness enabled, ALL reportable measurements (including calibration, pilots, baselines and sweeps) "
            "MUST use submit_experiment. execute is only for declared setup/test/debug/analysis; raw background launches are unavailable."
            "\n- Scripts may keep their own JSON result path: supply metrics_path to submit_experiment. "
            "Omit files when passing code_submission_id to inherit the reviewed submission manifest, or omit files without submission to snapshot src/**/*.py plus entrypoint/checker; explicitly include data/config dependencies when needed."
        )

    def before_run(self, state: RunState, ctx: HookContext) -> None:
        hcfg = ctx.config.plugins.harness
        system = build_system(
            ctx.run_dir / "system",
            experiments_between_reviews=hcfg.experiments_between_reviews,
            mlflow_tracking_uri=hcfg.mlflow_tracking_uri,
            module_specs=ctx.config.modules,
            module_root=ctx.workspace / ".research_modules",
            host=ctx.host,
        )
        if state.study_id:
            try:
                existing = system.store.get(state.study_id)
                system.store.revise(replace(existing, status="running", acceptance="pending"), "study.resumed")
            except KeyError:
                state.study_id = None
        if not state.study_id:
            study = system.controller.create(state.task_prompt, max_experiments=hcfg.max_experiments)
            state.study_id = study.id

        ctx.research_system = system
        ctx.module_registry = system.modules
        ctx.study_id = state.study_id
        self._py_exe = _resolve_science_python(ctx)
        self._pool = system.worker_pool(
            max_workers=hcfg.max_parallel_workers,
            executor=hcfg.executor,
            python_executable=self._py_exe,
        )
        self._futures.clear()
        self._write_live_snapshot(state, ctx)

    def _sync_job_artifacts(self, job: Job | dict[str, Any], state: RunState, ctx: HookContext) -> str:
        system: ResearchSystem = ctx.research_system
        job_id = job["id"] if isinstance(job, dict) else job.id
        status = job["status"] if isinstance(job, dict) else job.status
        validity = job.get("validity", "unverified") if isinstance(job, dict) else job.validity
        result = (job.get("result") if isinstance(job, dict) else job.result) or {}

        results_dir = ctx.workspace / "results" / job_id
        if status in {"succeeded", "failed"}:
            artifacts = result.get("artifacts") or {}
            if artifacts:
                results_dir.mkdir(parents=True, exist_ok=True)
                for rel_name, blob_hash in artifacts.items():
                    short = rel_name[len("outputs/"):] if rel_name.startswith("outputs/") else rel_name
                    target = (results_dir / short).resolve()
                    try:
                        target.relative_to(results_dir.resolve())
                    except ValueError:
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        target.write_bytes(system.artifacts.read(blob_hash))
                    except Exception:
                        pass

        sig_map = dict(state.metadata.get("logged_job_sigs") or {})
        sig = f"{status}:{validity}"
        if status in {"succeeded", "failed"} and sig_map.get(job_id) != sig:
            sig_map[job_id] = sig
            state.metadata["logged_job_sigs"] = sig_map
            try:
                system.tracker.log_job(job)
            except Exception as exc:
                ctx.emit_event("tracking_error", source="experiment", error=str(exc))

        return ctx.sandbox.rel_path(results_dir) if results_dir.exists() else ""

    def _write_live_snapshot(self, state: RunState, ctx: HookContext) -> None:
        system: ResearchSystem | None = ctx.research_system
        study_id = ctx.study_id
        if system is None or not study_id:
            return
        try:
            progress = system.queries.progress(study_id)
        except Exception:
            return
        payload = {
            **progress,
            "runtime": {
                "run_id": state.run_id,
                "status": state.status,
                "acceptance": state.acceptance,
                "scientific_outcome": state.scientific_outcome,
                "model_calls": state.model_calls,
                "subagent_model_calls": int(state.metadata.get("subagent_model_calls") or 0),
                "tool_calls_total": state.tool_calls_total,
                "input_tokens": state.input_tokens,
                "output_tokens": state.output_tokens,
                "cached_input_tokens": state.cached_input_tokens,
                "elapsed_seconds": round(time.time() - state.started_at, 2) if state.started_at else 0.0,
                "revision_count": state.revision_count,
                "todos": state.todos,
            },
        }
        secret = getattr(ctx.llm_client, "api_key", None)
        text = redact_secret(json.dumps(payload, ensure_ascii=False, indent=2, default=str), secret)
        snap_file = ctx.run_dir / "live_snapshot.json"
        temporary = ctx.run_dir / f".live-{uuid4().hex[:8]}.tmp"
        try:
            temporary.write_text(text, encoding="utf-8")
            os.replace(temporary, snap_file)
        finally:
            temporary.unlink(missing_ok=True)

    def _format_job_summary(self, job: Job | dict[str, Any], results_rel: str) -> dict[str, Any]:
        if isinstance(job, dict):
            res = job.get("result") or {}
            return {
                "job_id": job["id"],
                "purpose": (job.get("spec") or {}).get("purpose", ""),
                "status": job["status"],
                "validity": job.get("validity", "unverified"),
                "invalidated_reason": job.get("invalidated_reason", ""),
                "plan_version": job.get("plan_version"),
                "metrics": res.get("metrics", {}),
                "error": res.get("error", ""),
                "results_dir": results_rel,
                "checks": job.get("checks", []),
            }
        res = job.result or {}
        return {
            "job_id": job.id,
            "purpose": job.spec.get("purpose", ""),
            "status": job.status,
            "validity": job.validity,
            "invalidated_reason": job.invalidated_reason,
            "plan_version": job.plan_version,
            "metrics": res.get("metrics", {}),
            "error": res.get("error", ""),
            "results_dir": results_rel,
            "checks": job.checks,
        }

    def tools(self, ctx: HookContext) -> list[ToolDef]:
        def _require_system() -> tuple[ResearchSystem, str]:
            if ctx.research_system is None or not ctx.study_id:
                raise RuntimeError("ResearchHarnessPlugin is not initialized for this run.")
            return ctx.research_system, ctx.study_id

        def record_research_note(
            kind: str,
            text: str,
            refs: list[str] | None = None,
            choice: str = "",
            next_action: str = "",
            allow_invalid_ref: bool = False,
            url: str = "",
            excerpt: str = "",
            locator: str = "",
        ) -> str:
            system, study_id = _require_system()
            ref_tuple = tuple(str(r) for r in (refs or []))
            if kind == "decision":
                rec = system.controller.decide(study_id, choice, text, ref_tuple, next_action)
            elif kind == "source":
                if url and url.startswith(("http://", "https://")):
                    rec = system.notebook.source(
                        study_id,
                        text,
                        url,
                        excerpt or text,
                        locator or "cited source",
                    )
                else:
                    study = system.store.get(study_id)
                    rec = system.notebook.record(
                        study_id,
                        "source",
                        text,
                        ref_tuple,
                        details={"url": url, "excerpt": excerpt or text, "locator": locator or "note", "plan_version": study.plan_version},
                    )
            else:
                details = {"allow_invalid_ref": True} if allow_invalid_ref else None
                rec = system.notebook.record(study_id, kind, text, ref_tuple, details=details)

            if system.modules is not None and system.modules.has("notebook"):
                try:
                    system.modules.get("notebook").put(
                        kind, text, record_id=rec.id, refs=list(ref_tuple), study_id=study_id, ledger_record_id=rec.id
                    )
                except Exception:
                    pass

            if isinstance(ctx.state, RunState):
                self._write_live_snapshot(ctx.state, ctx)
            return json.dumps(
                {"record_id": rec.id, "kind": rec.kind, "refs": list(rec.refs)},
                ensure_ascii=False,
            )

        def set_research_focus(phase: str, next_action: str, scientific_outcome: str = "") -> str:
            system, study_id = _require_system()
            if scientific_outcome and scientific_outcome not in {"unassessed", "supported", "refuted", "inconclusive"}:
                raise ValueError("Unknown scientific outcome")
            study = system.controller.focus(study_id, phase, next_action)
            system.notebook.record(study_id, "stage", next_action, details={"phase": phase})
            study = system.store.get(study_id)
            if scientific_outcome:
                if scientific_outcome not in {"unassessed", "supported", "refuted", "inconclusive"}:
                    raise ValueError("scientific_outcome must be unassessed, supported, refuted, or inconclusive")
                study = system.store.revise(
                    replace(study, scientific_outcome=scientific_outcome),
                    "study.outcome_updated",
                )
                if isinstance(ctx.state, RunState):
                    ctx.state.scientific_outcome = scientific_outcome
            if isinstance(ctx.state, RunState):
                self._write_live_snapshot(ctx.state, ctx)
            return json.dumps(
                {
                    "study_id": study.id,
                    "phase": study.phase,
                    "plan_version": study.plan_version,
                    "next_action": study.next_action,
                    "scientific_outcome": study.scientific_outcome,
                },
                ensure_ascii=False,
            )

        def submit_experiment(
            purpose: str,
            entrypoint: str,
            protocol: str,
            data_version: str,
            files: list[str] | None = None,
            args: list[str] | None = None,
            seed: int | None = None,
            timeout_seconds: int | None = None,
            check_entrypoint: str | None = None,
            parent_id: str | None = None,
            request_key: str = "",
            wait: bool = True,
            metrics_path: str | None = None,
            module: bool | None = None,
            code_submission_id: str = "",
            check_runner: str | None = None,
        ) -> str:
            system, study_id = _require_system()
            if not system.modules.has("experiments"):
                raise ValueError("Experiment module is disabled")

            approved_run: dict[str, Any] = {}
            approved_files: list[str] | None = None
            if code_submission_id and system.modules.has("coding"):
                coding_mod = system.modules.get("coding")
                try:
                    sub_rec = coding_mod.inspect(code_submission_id)
                    sub_details = sub_rec.get("details") or {}
                    approved_run = dict(sub_details.get("run") or {})
                    approved_files = list(sub_details.get("files") or (sub_details.get("manifest") or {}).keys())
                except Exception:
                    pass

            clean_entry = _rel_posix(entrypoint)
            eff_check = approved_run.get("check_entrypoint") or "" if check_entrypoint is None else check_entrypoint
            clean_check = _rel_posix(eff_check) if eff_check else ""
            clean_files: list[str] = []
            if files is not None:
                selected = files
            elif approved_files is not None:
                selected = approved_files
            else:
                selected = [p.relative_to(ctx.workspace).as_posix() for p in (ctx.workspace / "src").rglob("*.py")]
            for f in selected:
                cf = _rel_posix(f)
                if cf and cf not in clean_files:
                    clean_files.append(cf)
            if clean_entry and clean_entry not in clean_files:
                clean_files.append(clean_entry)
            if clean_check and clean_check not in clean_files:
                clean_files.append(clean_check)

            eff_args = approved_run.get("args", []) if args is None else args
            eff_seed = approved_run.get("seed", 0) if seed is None else seed
            eff_timeout = (
                approved_run.get("timeout_seconds", ctx.config.plugins.harness.experiment_timeout_seconds)
                if timeout_seconds is None
                else timeout_seconds
            )
            eff_parent = approved_run.get("parent_id") if parent_id is None else (parent_id or None)
            eff_metrics = approved_run.get("metrics_path", "outputs/metrics.json") if metrics_path is None else metrics_path
            eff_module = approved_run.get("module", False) if module is None else module

            study = system.store.get(study_id)
            spec = ExperimentSpec(
                purpose=purpose,
                source_dir=str(ctx.workspace),
                files=tuple(clean_files),
                entrypoint=clean_entry,
                protocol=protocol,
                data_version=data_version,
                args=tuple(str(a) for a in (eff_args or [])),
                seed=int(eff_seed),
                timeout_seconds=eff_timeout,
                parent_id=eff_parent,
                check_entrypoint=(clean_check or None),
                metrics_path=_rel_posix(eff_metrics),
                module=bool(eff_module),
                check_runner=check_runner or approved_run.get("check_runner", "auto"),
            )
            req_key = request_key.strip() or f"req_{uuid4().hex[:10]}"
            approval = {}
            if system.modules.has("coding"):
                coding = system.modules.get("coding")
                if code_submission_id or bool(ctx.config.modules.get("coding", {}).get("options", {}).get("require_review", False)):
                    approval["expected_manifest"] = coding.authorize(code_submission_id, spec)
            job = system.harness.submit(study_id, study.plan_version, spec, req_key, **approval)
            if isinstance(ctx.state, RunState):
                ctx.state.metadata["last_submission_call"] = ctx.state.metadata.get("tool_batch_id", ctx.state.model_calls)

            if self._pool is None:
                self._pool = system.worker_pool(
                    max_workers=ctx.config.plugins.harness.max_parallel_workers,
                    executor=ctx.config.plugins.harness.executor,
                    python_executable=self._py_exe,
                )
            fut = self._pool.submit_async(study_id)
            self._futures[job.id] = fut

            if wait:
                deadline = time.monotonic() + spec.timeout_seconds + 15.0 if spec.timeout_seconds is not None else None
                def ready(job):
                    return job.status in {"succeeded", "failed", "superseded"} and not (
                        job.status == "succeeded" and spec.check_entrypoint and job.validity == "unverified")
                while True:
                    cur = system.harness.get(job.id)
                    if ready(cur):
                        job = cur
                        break
                    if fut.done():
                        fut.result()  # Surface a worker crash even when no experiment timeout is configured.
                        job = system.harness.get(job.id)
                        if ready(job):
                            break
                    self._consume_commands(ctx.state, ctx)
                    if ctx.state.metadata.get("user_pause_requested") or (deadline is not None and time.monotonic() >= deadline):
                        job = system.harness.get(job.id)
                        break
                    time.sleep(0.05)

            results_rel = ""
            if isinstance(ctx.state, RunState):
                results_rel = self._sync_job_artifacts(job, ctx.state, ctx)
                self._write_live_snapshot(ctx.state, ctx)

            return json.dumps(self._format_job_summary(job, results_rel), ensure_ascii=False, indent=2)

        def poll_experiments(job_ids: list[str] | None = None, wait_seconds: int = 5) -> str:
            system, study_id = _require_system()
            target_ids = set(job_ids) if job_ids else None
            deadline = time.monotonic() + max(0, min(300, int(wait_seconds)))

            while True:
                all_jobs = system.harness.list_jobs(study_id) if hasattr(system.harness, "list_jobs") else system.queries.snapshot(study_id)["jobs"]
                relevant = [
                    j for j in all_jobs
                    if target_ids is None or j["id"] in target_ids
                ]
                pending = [j for j in relevant if j["status"] in {"queued", "running"} or (
                    j["status"] == "succeeded" and j["spec"].get("check_entrypoint") and j.get("validity") == "unverified")]
                if not pending or time.monotonic() >= deadline:
                    break
                time.sleep(0.05)

            all_jobs = system.harness.list_jobs(study_id) if hasattr(system.harness, "list_jobs") else system.queries.snapshot(study_id)["jobs"]
            items = []
            jobs_to_show = [
                j for j in all_jobs
                if target_ids is None or j["id"] in target_ids
            ]
            if target_ids is None:
                jobs_to_show = jobs_to_show[-15:]
            for j in jobs_to_show:
                rel = ""
                if isinstance(ctx.state, RunState):
                    rel = self._sync_job_artifacts(j, ctx.state, ctx)
                item = self._format_job_summary(j, rel)
                if system.modules.has("notebook"):
                    item["notes"] = [n for n in system.modules.get("notebook").search()
                                     if j["id"] in n["details"].get("refs", [])]
                items.append(item)
            if isinstance(ctx.state, RunState):
                self._write_live_snapshot(ctx.state, ctx)
            return json.dumps(items, ensure_ascii=False, indent=2)

        def verify_experiment(
            job_id: str,
            check_entrypoint: str,
            files: list[str] | None = None,
            args: list[str] | None = None,
            timeout_seconds: int | None = None,
            check_runner: str | None = None,
        ) -> str:
            system, study_id = _require_system()
            if system.harness.get(job_id).study_id != study_id:
                raise ValueError("Experiment belongs to another study")
            clean_check = _rel_posix(check_entrypoint)
            clean_files = tuple(_rel_posix(f) for f in (files or []) if _rel_posix(f))
            eff_timeout = (timeout_seconds if timeout_seconds is not None
                           else ctx.config.plugins.harness.experiment_timeout_seconds)
            if hasattr(system.harness, "verify"):
                updated = system.harness.verify(
                    job_id,
                    clean_check,
                    source_dir=ctx.workspace,
                    files=clean_files,
                    args=tuple(str(a) for a in (args or [])),
                    timeout_seconds=eff_timeout,
                    python_executable=self._py_exe,
                    runner=check_runner,
                )
            else:
                worker = system.worker("checker", python_executable=self._py_exe)
                updated = worker.verify_job(
                    job_id,
                    clean_check,
                    source_dir=ctx.workspace,
                    files=clean_files,
                    args=tuple(str(a) for a in (args or [])),
                    timeout_seconds=eff_timeout,
                    runner=check_runner,
                )
            rel = ""
            if isinstance(ctx.state, RunState):
                rel = self._sync_job_artifacts(updated, ctx.state, ctx)
                self._write_live_snapshot(ctx.state, ctx)
            return json.dumps(self._format_job_summary(updated, rel), ensure_ascii=False, indent=2)

        def invalidate_experiment(job_id: str, reason: str) -> str:
            system, study_id = _require_system()
            if system.harness.get(job_id).study_id != study_id:
                raise ValueError("Experiment belongs to another study")
            if hasattr(system.harness, "invalidate"):
                updated = system.harness.invalidate(job_id, reason, actor="agent")
            else:
                updated = system.store.invalidate_job(job_id, reason, actor="agent")
            rel = ""
            if isinstance(ctx.state, RunState):
                rel = self._sync_job_artifacts(updated, ctx.state, ctx)
                self._write_live_snapshot(ctx.state, ctx)
            return json.dumps(self._format_job_summary(updated, rel), ensure_ascii=False, indent=2)

        def get_research_status() -> str:
            system, study_id = _require_system()
            progress = system.queries.progress(study_id)
            snap = system.queries.snapshot(study_id)
            return json.dumps(progress, ensure_ascii=False, indent=2, default=str)

        from ..modules.tools import module_tools
        return module_tools(ctx) + archive_tools(ctx) + [
            ToolDef(
                name="record_research_note",
                description=(
                    "Record an evidence-linked research note into the durable ledger "
                    "(source, question, reproduction, observation, hypothesis, method, decision, limitation)."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": ["source", "question", "reproduction", "observation", "hypothesis", "method", "decision", "limitation"],
                        },
                        "text": {"type": "string", "description": "Explicit scientific statement or finding."},
                        "refs": {"type": "array", "items": {"type": "string"}, "description": "Referenced exp_* or note_* IDs."},
                        "choice": {"type": "string", "description": "For decision: continue | pivot | park | report."},
                        "next_action": {"type": "string", "description": "For decision: concrete next action."},
                        "allow_invalid_ref": {"type": "boolean", "description": "Set true only when documenting negative/invalidated failures."},
                        "url": {"type": "string", "description": "For source: paper/code URL."},
                        "excerpt": {"type": "string", "description": "For source: read passage excerpt."},
                        "locator": {"type": "string", "description": "For source: page/section locator."},
                    },
                    "required": ["kind", "text"],
                },
                handler=record_research_note,
            ),
            ToolDef(
                name="set_research_focus",
                description="Optionally describe current focus and next action; labels impose no execution order.",
                parameters={
                    "type": "object",
                    "properties": {
                        "phase": {
                            "type": "string",
                            "description": "Optional free-form focus label; does not control execution.",
                        },
                        "next_action": {"type": "string"},
                        "scientific_outcome": {
                            "type": "string",
                            "enum": ["unassessed", "supported", "refuted", "inconclusive"],
                        },
                    },
                    "required": ["phase", "next_action"],
                },
                handler=set_research_focus,
            ),
            ToolDef(
                name="submit_experiment",
                description=(
                    "Freeze workspace source files and execute a reproducible experiment in the external harness. "
                    "The entrypoint writes non-empty JSON at metrics_path (default outputs/metrics.json)."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "purpose": {"type": "string"},
                        "entrypoint": {"type": "string", "description": "Relative .py entrypoint inside workspace."},
                        "files": {"type": "array", "items": {"type": "string"}, "description": "Source/data files to freeze."},
                        "protocol": {"type": "string", "description": "Evaluation protocol version/identifier."},
                        "data_version": {"type": "string", "description": "Dataset/generator version identifier."},
                        "args": {"type": "array", "items": {"type": "string"}},
                        "seed": {"type": "integer"},
                        "timeout_seconds": {"type": ["integer", "null"]},
                        "check_entrypoint": {"type": "string", "description": "Optional pytest test file or checks.json verification script."},
                        "check_runner": {"type": "string", "enum": ["auto", "pytest", "checks_json"]},
                        "parent_id": {"type": "string"},
                        "request_key": {"type": "string", "description": "Optional idempotency key."},
                        "wait": {"type": "boolean", "description": "Wait for completion (true) or launch asynchronously for parallel runs (false)."},
                        "metrics_path": {"type": "string", "description": "JSON output under outputs/ or results/; allows existing scripts without rewriting their result path."},
                        "module": {"type": "boolean", "description": "Run the .py entrypoint as python -m (for package-relative imports)."},
                        "code_submission_id": {"type": "string", "description": "Researcher-approved code submission for this exact code and run specification."},
                    },
                    "required": ["purpose", "entrypoint", "protocol", "data_version"],
                },
                handler=submit_experiment,
            ),
            ToolDef(
                name="poll_experiments",
                description="Wait for and collect status, metrics, and materialized results/ dirs of submitted harness experiments.",
                parameters={
                    "type": "object",
                    "properties": {
                        "job_ids": {"type": "array", "items": {"type": "string"}},
                        "wait_seconds": {"type": "integer", "description": "Seconds to wait for running jobs (default 5)."},
                    },
                },
                handler=poll_experiments,
            ),
            ToolDef(
                name="verify_experiment",
                description=(
                    "Run an independent checker script (must write outputs/checks.json with {checks:[{name,passed}]}) "
                    "against the frozen archive of a finished experiment."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "job_id": {"type": "string"},
                        "check_entrypoint": {"type": "string"},
                        "check_runner": {"type": "string", "enum": ["auto", "pytest", "checks_json"]},
                        "files": {"type": "array", "items": {"type": "string"}},
                        "args": {"type": "array", "items": {"type": "string"}},
                        "timeout_seconds": {"type": "integer"},
                    },
                    "required": ["job_id", "check_entrypoint"],
                },
                handler=verify_experiment,
            ),
            ToolDef(
                name="invalidate_experiment",
                description="Mark an experiment as invalid so stale or buggy numbers cannot be cited as supporting evidence.",
                parameters={
                    "type": "object",
                    "properties": {
                        "job_id": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["job_id", "reason"],
                },
                handler=invalidate_experiment,
            ),
            ToolDef(
                name="get_research_status",
                description="Inspect structured study state, experiments, notes, pending user messages, and literature triggers.",
                parameters={"type": "object", "properties": {}},
                handler=get_research_status,
            ),
        ]

    def _consume_commands(self, state: RunState, ctx: HookContext) -> None:
        if ctx.research_system is None or not ctx.study_id:
            return
        for command in ctx.research_system.inbox.apply_pending(ctx.study_id):
            if command["status"] == "rejected":
                continue
            if command["kind"] in {"suggestion", "change_direction"}:
                state.add_notice(f"[User {command['kind']}] {command['text']}")
        study = ctx.research_system.store.get(ctx.study_id)
        state.metadata["user_pause_requested"] = study.status == "paused"

    def before_tool_call(self, state: RunState, tool_name: str, arguments: dict[str, Any],
                         ctx: HookContext) -> ToolCallDecision:
        if ctx.research_system is None or not ctx.study_id:
            return ToolCallDecision()
        self._consume_commands(state, ctx)
        study = ctx.research_system.store.get(ctx.study_id)
        readonly = {"ls", "read_file", "get_research_status", "poll_experiments", "check_process", "list_processes",
                    "query_research_records", "read_research_document", "list_papers", "read_library_paper",
                    "search_long_term_notes", "read_long_term_note", "list_modules"}
        tool_spec = ctx.tool_specs.get(tool_name)
        if tool_spec and tool_spec.permissions and all(p.endswith(".read") or p == "tasks.cancel" for p in tool_spec.permissions):
            readonly.add(tool_name)
        reason = ""
        if tool_name not in readonly and (study.status != "running" or
                state.metadata.get("model_plan_version", study.plan_version) != study.plan_version):
            reason = "User paused or changed the plan. Refresh research status and replan before further execution."
        elif tool_name == "run_in_background":
            reason = "Raw background execution is disabled while harness is enabled. Submit formal work with submit_experiment."
        elif tool_name == "execute":
            purpose = arguments.get("purpose")
            if purpose not in {"setup", "test", "debug", "analysis"}:
                reason = "Declare purpose=setup/test/debug/analysis. Measurements, pilots, calibration and sweeps must use submit_experiment."
            elif purpose == "debug":
                if int(arguments.get("timeout") or 30) > 30:
                    reason = "Debug commands are limited to 30 seconds; submit longer/reportable experiments through the harness."
                else:
                    arguments["timeout"] = 30
            elif purpose == "analysis":
                refs = arguments.get("experiment_ids") or []
                jobs = {j["id"]: j for j in ctx.research_system.queries.snapshot(ctx.study_id)["jobs"]}
                if not refs or any(ref not in jobs or jobs[ref]["status"] not in {"succeeded", "failed"} for ref in refs):
                    reason = "Analysis must cite completed ledger experiment_ids; unregistered shell outputs are not research evidence."
        return ToolCallDecision(action="block", source=self.name, feedback=reason) if reason else ToolCallDecision()

    def before_model_call(
        self,
        state: RunState,
        messages: list[dict[str, Any]],
        ctx: HookContext,
    ) -> list[dict[str, Any]]:
        system: ResearchSystem | None = ctx.research_system
        study_id = ctx.study_id
        if system is None or not study_id:
            return messages

        handled = system.inbox.apply_pending(study_id)
        for cmd in handled:
            kind = cmd.get("kind")
            if kind == "suggestion":
                state.add_notice(f"[User Suggestion ({cmd['id']})] {cmd['text']}")
            elif kind == "change_direction":
                state.add_notice(
                    f"[User Direction Change (plan_version={cmd.get('applied_plan_version')})] "
                    f"New direction: {cmd['text']}. Queued experiments from the old plan were superseded; "
                    "Choose your next action for the updated direction."
                )
            elif kind == "pause":
                state.metadata["user_pause_requested"] = True
            elif kind == "resume":
                state.metadata.pop("user_pause_requested", None)

        snap = system.queries.snapshot(study_id)
        state.metadata["model_plan_version"] = snap["study"]["plan_version"]
        if snap["study"]["status"] == "paused":
            state.metadata["user_pause_requested"] = True
        elif snap["study"]["status"] == "running":
            state.metadata.pop("user_pause_requested", None)

        for job in snap["jobs"]:
            if job["status"] in {"succeeded", "failed"}:
                self._sync_job_artifacts(job, state, ctx)

        self._write_live_snapshot(state, ctx)
        return messages

    def after_tool_call(
        self,
        state: RunState,
        tool_name: str,
        arguments: dict[str, Any],
        result: str,
        ctx: HookContext,
    ) -> None:
        system: ResearchSystem | None = ctx.research_system
        study_id = ctx.study_id
        if system is None or not study_id:
            return

        if tool_name in {"write_file", "edit_file"} and not result.startswith(("Error:", "[TOOL")):
            rel = _rel_posix(str(arguments.get("file_path") or ""))
            if rel.endswith(".py"):
                try:
                    target = ctx.sandbox.resolve_path(rel)
                    new_hash = (
                        hashlib.sha256(target.read_bytes()).hexdigest()
                        if target.is_file()
                        else ""
                    )
                except Exception:
                    new_hash = ""
                if new_hash:
                    snap = system.queries.snapshot(study_id)
                    stale_jobs = [
                        j["id"]
                        for j in snap["jobs"]
                        if j["status"] == "succeeded"
                        and j.get("validity") != "invalid"
                        and j.get("manifest", {}).get(rel)
                        and j["manifest"][rel] != new_hash
                    ]
                    if stale_jobs:
                        state.add_notice(
                            f"[Evidence Integrity Notice] Modified '{rel}', which was frozen in completed "
                            f"experiment(s) {stale_jobs}. Re-run and verify updated experiments, or call "
                            "invalidate_experiment if the edit fixed a bug in prior runs."
                        )

        self._write_live_snapshot(state, ctx)

    def after_agent_finish(
        self,
        state: RunState,
        final_text: str,
        ctx: HookContext,
    ) -> TurnDecision:
        system: ResearchSystem | None = ctx.research_system
        study_id = ctx.study_id
        if system is None or not study_id:
            return TurnDecision(action="accept")

        if self._pool is not None:
            self._pool.wait_all(timeout=5.0)

        snap = system.queries.snapshot(study_id)
        for job in snap["jobs"]:
            if job["status"] in {"succeeded", "failed"}:
                self._sync_job_artifacts(job, state, ctx)

        in_flight = [j["id"] for j in snap["jobs"] if j["status"] in {"queued", "running"}]
        if in_flight and not state.metadata.get("budget_stop_requested"):
            return TurnDecision(
                action="revise",
                source=self.name,
                feedback=(
                    f"Experiments {in_flight} are still queued or running in the harness. "
                    "Results are incomplete at delivery time."
                ),
            )

        system.reports.export(study_id, system.root / "reports" / f"{study_id}.md")
        report_path = ctx.workspace / "final_report.md"
        report = report_path.read_text(encoding="utf-8") if report_path.is_file() else ""
        issues = delivery_issues(snap, report)
        if issues:
            return TurnDecision(action="revise", source=self.name, feedback="\n".join(issues))
        self._write_live_snapshot(state, ctx)
        return TurnDecision(action="accept")

    def after_run(self, state: RunState, ctx: HookContext) -> None:
        if self._pool is not None:
            try:
                self._pool.shutdown()
            except Exception:
                pass
        system: ResearchSystem | None = ctx.research_system
        study_id = ctx.study_id
        if system is not None and study_id:
            try:
                study = system.store.get(study_id)
                updated_status = state.status if state.status in {"completed", "stopped", "paused", "failed"} else study.status
                system.store.revise(
                    replace(
                        study,
                        status=updated_status,
                        acceptance=state.acceptance,
                        scientific_outcome=state.scientific_outcome,
                    ),
                    "study.finalized",
                )
            except Exception:
                pass
            self._write_live_snapshot(state, ctx)
