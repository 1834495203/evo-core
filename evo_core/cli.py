"""CLI entrypoint for evo-core (`evocore run`, `evocore probe`, `evocore inspect`)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console

from .config import load_config
from .engine import ResearchKernel
from .llm import ChatClient
from .system.bootstrap import ResearchSystem, build_system
from .system.interaction.qa import CompanionQAAgent
from .system.observability.server import ResearchDashboardServer

console = Console()


def _load_system_from_run_dir(run_dir: Path) -> tuple[ResearchSystem, str, Path | None, Path | None]:
    root = Path(run_dir).resolve()
    actual_run_dir: Path | None = root
    if (root / "system" / "research.sqlite3").is_file():
        sys_root = root / "system"
    elif (root / "research.sqlite3").is_file():
        sys_root = root
        actual_run_dir = root.parent if (root.parent / "state.json").is_file() else None
    else:
        raise FileNotFoundError(f"No research.sqlite3 found under {root} or {root / 'system'}")

    system = build_system(sys_root)
    study_id: str | None = None
    workspace_path: Path | None = None

    if actual_run_dir is not None:
        summary_file = actual_run_dir / "summary.json"
        if summary_file.is_file():
            try:
                summary_data = json.loads(summary_file.read_text(encoding="utf-8"))
                study_id = summary_data.get("study_id") or study_id
                if summary_data.get("workspace"):
                    workspace_path = Path(summary_data["workspace"]).resolve()
            except Exception:
                pass
        state_file = actual_run_dir / "state.json"
        if not study_id and state_file.is_file():
            try:
                state_data = json.loads(state_file.read_text(encoding="utf-8"))
                study_id = state_data.get("study_id") or study_id
            except Exception:
                pass
        snap_file = actual_run_dir / "live_snapshot.json"
        if not study_id and snap_file.is_file():
            try:
                snap_data = json.loads(snap_file.read_text(encoding="utf-8"))
                study_id = (snap_data.get("study") or {}).get("id") or study_id
            except Exception:
                pass

    if not study_id:
        studies = system.store.list_studies()
        if not studies:
            raise ValueError(f"No studies found in {sys_root}")
        study_id = str(studies[0]["id"])

    return system, study_id, actual_run_dir, workspace_path


def _cmd_probe(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.model:
        cfg.model = args.model
    api_key, base_url = cfg.resolved_api_credentials()
    client = ChatClient(
        api_key=api_key,
        base_url=base_url,
        default_model=cfg.model,
        thinking=cfg.thinking,
        reasoning_effort=cfg.reasoning_effort,
        timeout=30.0,
    )
    tool_schema = [
        {
            "type": "function",
            "function": {
                "name": "ping",
                "description": "Connectivity verification tool.",
                "parameters": {
                    "type": "object",
                    "properties": {"status": {"type": "string"}},
                    "required": ["status"],
                },
            },
        }
    ]
    resp = client.complete(
        [{"role": "user", "content": "Call ping with status='READY'."}],
        tools=tool_schema,
        max_tokens=256,
    )
    ok = any(tc.name == "ping" for tc in resp.tool_calls)
    console.print(
        f"[bold]Probe[/bold] model={cfg.model} thinking={cfg.thinking} "
        f"tool_ok={ok} in={resp.input_tokens} out={resp.output_tokens} "
        f"reasoning_chars={len(resp.reasoning_content)}"
    )
    return 0 if ok else 1


def _cmd_run(args: argparse.Namespace) -> int:
    overrides: dict = {}
    if args.workspace:
        overrides["workspace"] = args.workspace
    if args.run_dir:
        overrides["run_dir"] = args.run_dir
    if args.model:
        overrides["model"] = args.model
    if args.max_calls is not None:
        overrides.setdefault("plugins", {}).setdefault("budget", {})["max_calls"] = (
            None if args.max_calls <= 0 else args.max_calls
        )
    if args.disable_plugin:
        pdict = overrides.setdefault("plugins", {})
        for pname in args.disable_plugin:
            pdict[pname] = {"enabled": False}

    cfg = load_config(args.config, overrides=overrides)
    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    if args.topic_file and args.topic_id:
        for line in Path(args.topic_file).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("id") == args.topic_id:
                prompt = row["prompt"]
                break
    if not prompt:
        console.print("[red]Error: provide --prompt, --prompt-file, or --topic-file + --topic-id[/red]")
        return 2

    dash_holder: list[ResearchDashboardServer] = []

    def on_event(evt: dict) -> None:
        kind = evt.get("event")
        if kind == "run_start":
            console.print(
                f"[cyan]Run {evt['run_id']}[/cyan] model={evt['model']} "
                f"plugins={evt['plugins']} sys_chars={evt['system_prompt_chars']}"
            )
            dash_port = getattr(args, "dashboard_port", None)
            if dash_port is not None and dash_port >= 0 and evt.get("study_id"):
                try:
                    sys_obj = build_system(Path(cfg.run_dir).resolve() / "system")
                    qa = CompanionQAAgent(
                        sys_obj.queries,
                        sys_obj.events,
                        sys_obj.artifacts,
                        Path(cfg.workspace).resolve(),
                        llm_client=kernel.llm,
                        model=cfg.effective_aux_model(),
                    )
                    srv = ResearchDashboardServer(
                        study_id=evt["study_id"],
                        queries=sys_obj.queries,
                        inbox=sys_obj.inbox,
                        events=sys_obj.events,
                        qa_agent=qa,
                        run_dir=Path(cfg.run_dir).resolve(),
                        port=dash_port,
                    )
                    host, port = srv.start_background()
                    dash_holder.append(srv)
                    console.print(f"[cyan]Dashboard[/cyan] http://{host}:{port}/ (study={evt['study_id']})")
                except Exception as exc:
                    console.print(f"[yellow]Dashboard failed to start: {exc}[/yellow]")
        elif kind == "model_end":
            console.print(
                f"  [dim]call {evt['call']}[/dim] tools={evt['tools']} "
                f"in={evt['input_tokens']} (cached={evt['cached_tokens']}) out={evt['output_tokens']}"
            )
        elif kind == "tool_blocked":
            console.print(f"  [yellow]blocked {evt['tool']}[/yellow] ({evt.get('source')}): {evt.get('reason')}")
        elif kind == "delivery_check":
            for issue in evt.get("issues", []):
                console.print(f"  [yellow]delivery note[/yellow]: {issue}")
        elif kind == "context_compacted":
            console.print(
                f"  [magenta]compacted[/magenta] prev_est={evt['prev_tokens']} "
                f"kept_msgs={evt['kept_msgs']}"
            )
        elif kind == "run_end":
            console.print(
                f"[bold green]Finished[/bold green] status={evt['status']} "
                f"acceptance={evt.get('acceptance', 'pending')} "
                f"outcome={evt.get('scientific_outcome', 'unassessed')} "
                f"calls={evt['model_calls']} tools={evt['tool_calls']} "
                f"elapsed={evt['elapsed_seconds']}s"
            )

    kernel = ResearchKernel(cfg, on_event=on_event)
    try:
        result = kernel.run(prompt, rubric=args.rubric, resume=args.resume)
    finally:
        for srv in dash_holder:
            srv.stop()
    console.print(
        json.dumps(
            {
                "run_id": result.run_id,
                "study_id": result.study_id,
                "status": result.status,
                "acceptance": result.acceptance,
                "scientific_outcome": result.scientific_outcome,
                "model_calls": result.model_calls,
                "tool_calls_total": result.tool_calls_total,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "cached_input_tokens": result.cached_input_tokens,
                "elapsed_seconds": result.elapsed_seconds,
                "workspace": str(result.workspace),
                "run_dir": str(result.run_dir),
                "error": result.error,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.status in {"completed", "paused"} else 1


def _cmd_inspect(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    summary = run_dir / "summary.json"
    if not summary.is_file():
        console.print(f"[red]No summary.json in {run_dir}[/red]")
        return 1
    console.print(summary.read_text(encoding="utf-8"))
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    system, study_id, actual_run_dir, _ = _load_system_from_run_dir(Path(args.run_dir))
    prog = system.queries.progress(study_id)
    if actual_run_dir is not None:
        snap_file = actual_run_dir / "live_snapshot.json"
        if snap_file.is_file():
            try:
                live = json.loads(snap_file.read_text(encoding="utf-8"))
                if isinstance(live, dict) and "runtime" in live:
                    prog["runtime"] = live["runtime"]
            except Exception:
                pass
    console.print(json.dumps(prog, ensure_ascii=False, indent=2, default=str))
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    system, study_id, _, workspace = _load_system_from_run_dir(Path(args.run_dir))
    if args.workspace:
        workspace = Path(args.workspace).resolve()
    llm_client = None
    model_name = None
    use_llm = not args.no_llm
    if use_llm:
        cfg = load_config(args.config)
        api_key, base_url = cfg.resolved_api_credentials()
        model_name = args.model or cfg.effective_aux_model()
        llm_client = ChatClient(
            api_key=api_key,
            base_url=base_url,
            default_model=model_name,
            thinking="disabled",
            timeout=60.0,
        )
    qa = CompanionQAAgent(
        system.queries,
        system.events,
        system.artifacts,
        workspace,
        llm_client=llm_client,
        model=model_name,
    )
    ans = qa.answer(study_id, args.question, use_llm=use_llm)
    console.print(json.dumps(ans, ensure_ascii=False, indent=2, default=str))
    return 0


def _cmd_steer(args: argparse.Namespace) -> int:
    system, study_id, _, _ = _load_system_from_run_dir(Path(args.run_dir))
    cmd_id = system.inbox.send(study_id, args.kind, args.text or "")
    console.print(json.dumps({"study_id": study_id, "command_id": cmd_id, "kind": args.kind, "status": "pending"}, ensure_ascii=False, indent=2))
    return 0


def _cmd_dashboard(args: argparse.Namespace) -> int:
    system, study_id, actual_run_dir, workspace = _load_system_from_run_dir(Path(args.run_dir))
    if args.workspace:
        workspace = Path(args.workspace).resolve()
    llm_client = None
    model_name = None
    if not args.no_llm:
        try:
            cfg = load_config(args.config)
            api_key, base_url = cfg.resolved_api_credentials()
            model_name = cfg.effective_aux_model()
            llm_client = ChatClient(
                api_key=api_key,
                base_url=base_url,
                default_model=model_name,
                thinking="disabled",
                timeout=60.0,
            )
        except Exception:
            llm_client = None
    qa = CompanionQAAgent(
        system.queries,
        system.events,
        system.artifacts,
        workspace,
        llm_client=llm_client,
        model=model_name,
    )
    srv = ResearchDashboardServer(
        study_id=study_id,
        queries=system.queries,
        inbox=system.inbox,
        events=system.events,
        qa_agent=qa,
        run_dir=actual_run_dir,
        host=args.host,
        port=args.port,
    )
    host, port = srv.start_background()
    console.print(f"[cyan]Dashboard serving[/cyan] http://{host}:{port}/ (study={study_id}). Ctrl+C to stop.")
    try:
        while True:
            import time
            time.sleep(1.0)
    except KeyboardInterrupt:
        srv.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evocore", description="EvoCore Autonomous Research Kernel")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_probe = sub.add_parser("probe", help="Test LLM API connectivity and tool calling")
    p_probe.add_argument("--config", help="Path to YAML config")
    p_probe.add_argument("--model", help="Override model name")
    p_probe.set_defaults(func=_cmd_probe)

    p_run = sub.add_parser("run", help="Run an autonomous research task")
    p_run.add_argument("--config", help="Path to YAML config")
    p_run.add_argument("--prompt", help="Research prompt text")
    p_run.add_argument("--prompt-file", help="Path to prompt file")
    p_run.add_argument("--topic-file", help="Path to research_topics.jsonl")
    p_run.add_argument("--topic-id", help="Topic ID in topic-file")
    p_run.add_argument("--workspace", help="Workspace output directory")
    p_run.add_argument("--run-dir", help="Run metadata/logs directory")
    p_run.add_argument("--model", help="Override model name")
    p_run.add_argument("--max-calls", type=int, help="Max LLM calls (<=0 for unlimited)")
    p_run.add_argument("--rubric", help="Optional research and delivery requirements (no automatic reviewer)")
    p_run.add_argument("--resume", action="store_true", help="Resume saved run state")
    p_run.add_argument("--disable-plugin", action="append", default=[], help="Plugin name to disable")
    p_run.add_argument("--dashboard-port", type=int, default=None, help="Serve live HTTP dashboard during run")
    p_run.set_defaults(func=_cmd_run)

    p_ins = sub.add_parser("inspect", help="Inspect a run summary")
    p_ins.add_argument("run_dir", help="Path to run_meta directory")
    p_ins.set_defaults(func=_cmd_inspect)

    p_stat = sub.add_parser("status", help="Inspect consistent live study status & experiments")
    p_stat.add_argument("run_dir", help="Path to run_meta or system root directory")
    p_stat.set_defaults(func=_cmd_status)

    p_ask = sub.add_parser("ask", help="Ask the non-blocking read-only Companion Q&A Agent")
    p_ask.add_argument("run_dir", help="Path to run_meta or system root directory")
    p_ask.add_argument("--question", required=True, help="Question to ask about research progress or evidence")
    p_ask.add_argument("--no-llm", action="store_true", help="Return deterministic structured summary without calling an LLM")
    p_ask.add_argument("--config", help="Optional YAML config for Q&A model")
    p_ask.add_argument("--model", help="Override Q&A model name")
    p_ask.add_argument("--workspace", help="Optional workspace path")
    p_ask.set_defaults(func=_cmd_ask)

    p_steer = sub.add_parser("steer", help="Send a suggestion, direction change, pause, or resume command")
    p_steer.add_argument("run_dir", help="Path to run_meta or system root directory")
    p_steer.add_argument("--kind", required=True, choices=["suggestion", "change_direction", "pause", "resume"])
    p_steer.add_argument("--text", default="", help="Instruction or suggestion text")
    p_steer.set_defaults(func=_cmd_steer)

    p_dash = sub.add_parser("dashboard", help="Launch live web observability & Q&A dashboard")
    p_dash.add_argument("run_dir", help="Path to run_meta or system root directory")
    p_dash.add_argument("--host", default="127.0.0.1")
    p_dash.add_argument("--port", type=int, default=8765)
    p_dash.add_argument("--config", help="Optional YAML config for Q&A model")
    p_dash.add_argument("--workspace", help="Optional workspace path")
    p_dash.add_argument("--no-llm", action="store_true", help="Use structured Q&A responses without calling LLM")
    p_dash.set_defaults(func=_cmd_dashboard)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
