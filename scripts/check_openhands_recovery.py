"""Interrupt a real SDK task after a persisted edit, then restore in a new runtime."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from evo_core.config import load_config
from evo_core.harness_server import HarnessSession


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    root = Path(parser.parse_args().run_root).resolve()
    core = Path(__file__).resolve().parents[1]
    cfg = load_config(core / "config.default.yaml", overrides={
        "workspace": str(root / "workspace"), "run_dir": str(root / "run_meta")})
    session = HarnessSession(cfg, "OpenHands cancellation recovery fixture")
    task = session.invoke("create_coding_task", {"text": (
        "Create marker.py first with exactly MARKER = 1 and a newline. In a subsequent tool call, "
        "read marker.py back. Then create arithmetic/double.py providing double(x)=2*x and "
        "arithmetic/triple.py providing triple(x)=3*x, plus arithmetic/__init__.py exports. "
        "Do not run code. Preserve files if interrupted. This is an integration fixture.")}).data
    provider = session.ctx.module_registry.get("coding")
    session_root, tree, state = provider.session(task["id"], Path(cfg.workspace))
    marker = tree / "marker.py"
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(session.invoke, "implement_coding_task", {"task_id": task["id"]})
        deadline = time.monotonic() + 120
        while not marker.exists() and not future.done() and time.monotonic() < deadline:
            time.sleep(0.1)
        if not marker.exists():
            session.kernel.host.cancellation.request()
            future.result(timeout=15)
            raise RuntimeError("No persisted edit was available for the interruption check")
        original = marker.read_bytes()
        session.kernel.host.cancellation.request()
        first = future.result(timeout=15)
        if first.status == "completed":
            raise RuntimeError("Task finished before it could be interrupted; repeat fixture")
    session.close()
    restored = HarnessSession(cfg, "OpenHands cancellation recovery fixture")
    try:
        result = restored.invoke("implement_coding_task", {"task_id": task["id"], "instruction": (
            "Continue after an intentional Harness interruption. Preserve marker.py byte-for-byte. "
            "Read existing files and finish the arithmetic package. Missing files must be created "
            "in their real directories; do not restart from a blank workspace.")})
        if result.status != "completed":
            raise RuntimeError(str(result.error))
        assert result.data["conversation_id"] == state["conversation_id"]
        assert marker.read_bytes() == original
        for name in ("arithmetic/double.py", "arithmetic/triple.py", "arithmetic/__init__.py"):
            assert (tree / name).is_file(), name
        output = {"interrupted": True, "restored_in_new_runtime": True, "marker_preserved": True,
                  "conversation_id": state["conversation_id"], "task_id": task["id"],
                  "workspace": str(tree), "files": result.data["files"]}
        (root / "recovery-result.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(json.dumps(output, indent=2))
    finally:
        restored.close()


if __name__ == "__main__":
    main()
