"""Explicit entrypoint for the opt-in skeleton; never resumes an old evo-core run."""

from __future__ import annotations

import argparse
from pathlib import Path

from .bootstrap import build_system
from .domain import ExperimentSpec, encode


DEMO_CODE = '''import json
from pathlib import Path
values = [0, 1, 2, 3]
result = {"mean": sum(values) / len(values), "n": len(values)}
Path("outputs").mkdir()
Path("outputs/metrics.json").write_text(json.dumps(result), encoding="utf-8")
print(result)
'''


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Opt-in research-system skeleton")
    parser.add_argument("--root", required=True, help="Independent system database/artifact directory")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("demo", help="Run an engineering demonstration, not autonomous research")
    status = commands.add_parser("status")
    status.add_argument("study_id")
    events = commands.add_parser("events")
    events.add_argument("study_id")
    events.add_argument("--after", type=int, default=0)
    worker = commands.add_parser("worker", help="Claim and run one trusted local Python experiment")
    worker.add_argument("--id", default="local-worker")
    command = commands.add_parser("send")
    command.add_argument("study_id")
    command.add_argument("kind", choices=["suggestion", "change_direction", "pause", "resume"])
    command.add_argument("--text", default="")
    apply = commands.add_parser("apply-commands")
    apply.add_argument("study_id")
    args = parser.parse_args(argv)
    system = build_system(args.root)
    if args.command == "demo":
        study = system.controller.create("Engineering demo: verify the experiment evidence pipeline.")
        project = system.root / "demo-inputs" / study.id
        project.mkdir(parents=True)
        (project / "experiment.py").write_text(DEMO_CODE, encoding="utf-8")
        spec = ExperimentSpec("Compute the mean of four fixed values.", str(project),
                              ("experiment.py",), "experiment.py", "mean-check-v1", "fixed-list-v1")
        job = system.harness.submit(study.id, study.plan_version, spec, "first-experiment")
        result = system.worker("demo-worker").run_once(study.id)
        if result is None or result.id != job.id:
            raise RuntimeError("Demo experiment could not be claimed")
        if result.status != "succeeded":
            raise RuntimeError(encode(result.result))
        observation = system.notebook.record(study.id, "observation",
            f"Executed output reports mean={result.result['metrics']['mean']} for n=4.", (job.id,))
        system.controller.decide(study.id, "report", "The engineering pipeline completed; this is not a scientific finding.",
                                 (observation.id,), "Connect real research adapters and validation before scientific use.")
        report = system.reports.export(study.id, system.root / "reports" / f"{study.id}.md")
        print(encode({"study_id": study.id, "job_id": job.id, "job_status": result.status,
                      "acceptance": system.store.get(study.id).acceptance, "progress_report": str(report)}))
    elif args.command == "status":
        print(encode(system.queries.progress(args.study_id)))
    elif args.command == "events":
        print(encode(system.events.read(args.study_id, args.after)))
    elif args.command == "worker":
        job = system.worker(args.id).run_once()
        print(encode({"job_id": job.id, "status": job.status} if job else {"status": "idle"}))
    elif args.command == "send":
        print(encode({"command_id": system.inbox.send(args.study_id, args.kind, args.text)}))
    elif args.command == "apply-commands":
        print(encode({"handled": system.inbox.apply_pending(args.study_id)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
