"""Live multi-file implementation, process restart, revision and reviewed execution."""
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
    core = Path(__file__).resolve().parents[1]
    cfg = load_config(core / "config.default.yaml", overrides={"workspace": str(root / "workspace"),
        "run_dir": str(root / "run_meta"), "science_python": str(core.parent / "EvoScientist/.venv/Scripts")})
    def call(session, name, **arguments):
        result = session.invoke(name, arguments)
        if result.status != "completed":
            raise RuntimeError(f"{name}: {result.error}")
        return result.data
    session = HarnessSession(cfg, "OpenHands live integration validation")
    try:
        previous = root / "initial-result.json"
        if previous.exists():
            saved = json.loads(previous.read_text(encoding="utf-8"))
            task = {"id": saved["task_id"]}
            first_id = saved["conversation_id"]
        else:
            task = call(session, "create_coding_task", text=(
            "Implement a small Python package as files, not an essay. Only stdlib and pytest are allowed. "
            "Create src/calibration/__init__.py, src/calibration/quantile.py, src/calibration/intervals.py, "
            "src/run.py, checks/check_result.py, tests/test_calibration.py. "
            "quantile.py provides conformal_quantile(scores, alpha), sorts finite scores, uses the clipped "
            "1-based ceil((n+1)*(1-alpha)) order statistic, rejects empty input and alpha outside (0,1). "
            "intervals.py provides interval(prediction,q) returning (prediction-q,prediction+q), rejecting q<0. "
            "src/run.py creates outputs/metrics.json with {quantile:3, lower:7, upper:13} from scores=[1,2,3,4], "
            "alpha=0.5, prediction=10, and saves outputs/details.csv. "
            "checks/check_result.py reads those archived outputs and writes outputs/checks.json with "
            "a nonempty checks list using boolean passed fields; exits nonzero when any check fails. "
            "tests/test_calibration.py imports src.calibration and checks quantile, intervals, invalid alpha, "
            "empty input and negative q. Add no shell or experiment execution. Use your file editor. "
                "This is an integration fixture, not a research contribution."))
            first = call(session, "implement_coding_task", task_id=task["id"])
            first_id = first["conversation_id"]
            save = {"phase": "initial_implementation", "task_id": task["id"], "conversation_id": first_id,
                    "files": first["files"], "changes": first["changed_files"]}
            (root / "initial-result.json").write_text(json.dumps(save, indent=2), encoding="utf-8")
    finally:
        session.close()
    # A new runtime/module instance must restore the SDK conversation and actual source.
    session = HarnessSession(cfg, "OpenHands live integration validation")
    try:
        second = call(session, "implement_coding_task", task_id=task["id"], instruction=(
            "Continue from existing code without recreating the workspace. Add robust rejection of NaN "
            "and infinity in conformal_quantile and corresponding tests. Add tests proving the alpha=0.5 "
            "result stays 3. Preserve all other files. Finish with a concise list of actual changes."))
        assert second["conversation_id"] == first_id
        run = {"purpose": "openhands-integration", "entrypoint": "src/run.py", "protocol": "finite-sample-quantile-v1",
               "data_version": "fixture-v1", "check_entrypoint": "checks/check_result.py", "check_runner": "checks_json"}
        submission = call(session, "prepare_coding_submission", task_id=task["id"], run=run,
                          test_paths=["tests/test_calibration.py"])
        receipt = call(session, "test_code_submission", submission_id=submission["id"])
        if not receipt["passed"]:
            (root / "failed-test-feedback.txt").write_text(receipt["output"], encoding="utf-8")
            repaired = call(session, "implement_coding_task", task_id=task["id"], instruction=(
                "The Harness tested the frozen source and found this failure:\n" + receipt["output"] +
                "\nFix the actual cause in this same workspace. Preserve the documented order-statistic "
                "formula ceil((n+1)*(1-alpha)), clipped to [1,n]. With scores=[10,20,30], alpha=0.99 "
                "implies rank=1 and result=10, while alpha=0.01 implies clipped rank=3 and result=30. "
                "Correct reversed test expectations instead of changing the correct formula to satisfy them. "
                "Inspect neighboring boundary cases, preserve the other modules and describe the correction."))
            assert repaired["conversation_id"] == first_id
            submission = call(session, "prepare_coding_submission", task_id=task["id"], run=run,
                              test_paths=["tests/test_calibration.py"])
            receipt = call(session, "test_code_submission", submission_id=submission["id"])
            if not receipt["passed"]:
                raise RuntimeError(receipt["output"])
        call(session, "review_code_submission", submission_id=submission["id"], approved=True,
             notes="Integration fixture: restored multi-file implementation and tested numeric edge cases.")
        result = call(session, "submit_experiment", code_submission_id=submission["id"],
                      purpose=run["purpose"], entrypoint=run["entrypoint"], protocol=run["protocol"], data_version=run["data_version"])
        if result["status"] != "succeeded" or result["validity"] != "verified":
            raise RuntimeError(json.dumps(result))
        output = {"backend": "openhands", "conversation_id": first_id, "restored_after_restart": True,
                  "submission_id": submission["id"], "unit_tests": receipt["tests_run"], "experiment": result,
                  "model_calls": session.kernel.llm.counts_snapshot()}
        (root / "smoke-result.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(output, ensure_ascii=False, indent=2))
    finally:
        session.close()


if __name__ == "__main__":
    main()
