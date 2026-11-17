"""Live Aider -> frozen submission -> researcher approval -> Harness smoke check."""
import argparse
import json
from pathlib import Path

from evo_core.config import load_config
from evo_core.harness_server import HarnessSession


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    cfg = load_config(Path(__file__).resolve().parents[1] / "config.default.yaml", overrides={
        "workspace": str(root / "workspace"), "run_dir": str(root / "run_meta"),
        "science_python": str(Path(__file__).resolve().parents[2] / "EvoScientist/.venv/Scripts")})
    session = HarnessSession(cfg, "Verify real Aider integration")
    def call(name, **arguments):
        result = session.invoke(name, arguments)
        if result.status != "completed":
            raise RuntimeError(f"{name}: {result.error}")
        return result.data
    try:
        task = call("create_coding_task", text=(
            "Small integration task, not scientific research. Create src/value.py with twice(x) returning x*2. "
            "Create src/run.py that imports value.twice, creates outputs/, and writes outputs/metrics.json "
            "as {\"answer\":twice(21)}. Create tests/test_value.py that imports via src.value and asserts "
            "twice(0)==0 and twice(21)==42. Create src/check.py reading outputs/metrics.json and writing "
            "outputs/checks.json with {\"checks\":[{\"name\":\"answer\",\"passed\":<answer equals 42>}]} "
            "and exiting nonzero if false. Use only stdlib and pytest. "
            "Handoff run: purpose=integration-smoke, entrypoint=src/run.py, protocol=twice-21-v1, "
            "data_version=synthetic-v1, check_entrypoint=src/check.py, metrics_path=outputs/metrics.json, "
            "check_runner=checks_json. List all four files and test_paths=[tests/test_value.py]."))
        implemented = call("implement_coding_task", task_id=task["id"])
        sid = implemented["submissions"][0]
        receipt = call("test_code_submission", submission_id=sid)
        if not receipt["passed"]:
            raise RuntimeError(receipt["output"])
        call("review_code_submission", submission_id=sid, approved=True,
             notes="Integration fixture reviewed: unit assertions and declared output/check contract.")
        result = call("submit_experiment", code_submission_id=sid, purpose="integration-smoke",
                      entrypoint="src/run.py", protocol="twice-21-v1", data_version="synthetic-v1")
        if result["status"] != "succeeded" or result["validity"] != "verified":
            raise RuntimeError(json.dumps(result))
        summary = {"backend": "aider", "submission_id": sid, "experiment": result,
                   "model_calls_by_actor": session.kernel.llm.counts_snapshot()}
        (root / "smoke-result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        session.close()


if __name__ == "__main__":
    main()
