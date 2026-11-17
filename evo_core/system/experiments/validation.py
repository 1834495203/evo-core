"""Shared pytest execution and machine-readable receipts for immutable snapshots."""
from pathlib import Path
import os
import subprocess
import sys
import xml.etree.ElementTree as ET


def check_runner(entrypoint: str, requested: str = "auto") -> str:
    if requested not in {"auto", "pytest", "checks_json"}:
        raise ValueError("check_runner must be auto, pytest or checks_json")
    if requested != "auto": return requested
    name = Path(entrypoint).name
    return "pytest" if name.startswith("test_") or name.endswith("_test.py") else "checks_json"


def run_pytest(python: str, workspace: Path, *, targets=(), timeout=60) -> dict:
    workspace = workspace.resolve()
    outputs = workspace / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    report = outputs / "pytest.xml"
    report.unlink(missing_ok=True)
    temporary = workspace / ".test_tmp"
    temporary.mkdir(exist_ok=True)
    command = [python, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
               "--basetemp", str(workspace / ".pytest_tmp"), "--junitxml", str(report), *targets]
    env = dict(os.environ, TMP=str(temporary), TEMP=str(temporary), TMPDIR=str(temporary),
               PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1", MPLBACKEND="Agg")
    errors, checks, skipped = [], [], 0
    stdout, stderr, returncode = "", "", None
    try:
        proc = subprocess.run(command, cwd=workspace, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              env=env, timeout=timeout, check=False,
                              creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        stdout, stderr, returncode = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        errors.append(f"pytest timed out after {timeout}s")
    try:
        if report.is_symlink() or not report.resolve().is_relative_to(workspace):
            raise ValueError("pytest report escapes snapshot")
        root = ET.parse(report).getroot()
        for case in root.iter("testcase"):
            if case.find("skipped") is not None:
                skipped += 1
                continue
            failure = case.find("failure")
            error = case.find("error")
            checks.append({"name": (case.get("classname", "") + "::" + case.get("name", "test")).strip(":"),
                           "passed": failure is None and error is None})
        if not checks: errors.append("pytest executed no non-skipped tests")
        if any(not c["passed"] for c in checks): errors.append("pytest assertions or collection failed")
    except (OSError, ET.ParseError, ValueError) as exc:
        errors.append(f"Invalid or missing pytest report: {exc}")
    if returncode != 0: errors.append(f"pytest exited with code {returncode}")
    return {"runner": "pytest", "passed": returncode == 0 and bool(checks) and not errors,
            "returncode": returncode, "checks": checks, "errors": errors, "tests_run": len(checks),
            "tests_skipped": skipped, "stdout": stdout, "stderr": stderr}
