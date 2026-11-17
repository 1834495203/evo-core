"""Isolated dependency worker using the real OpenHands SDK agent loop."""
import json
import os
from pathlib import Path
import sys
import threading
import uuid

# The adapter filename is openhands.py; direct script execution must not shadow
# the external openhands namespace package with its own sibling.
sys.path[:] = [p for p in sys.path if Path(p).resolve() != Path(__file__).resolve().parent]
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")


def main():
    request = json.loads(sys.stdin.read())
    from openhands.sdk import Agent, Conversation, LLM, Tool
    from openhands.sdk.context.condenser import LLMSummarizingCondenser
    from openhands.sdk.tool import ToolExecutor, register_tool
    from openhands.tools.file_editor import FileEditorTool
    from openhands.tools.file_editor.definition import FileEditorObservation
    from openhands.tools.task_tracker import TaskTrackerTool

    root = Path(request["workspace"]).resolve()
    events_path = Path(request["events_path"])
    lock = threading.Lock()
    def emit(kind, **data):
        # Credentials are only received on stdin; never emit the request/config.
        line = json.dumps({"kind": kind, **data}, ensure_ascii=False, default=str)
        line = line.replace(request["llm"]["api_key"], "[REDACTED]")
        with lock, events_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    class ScopedExecutor(ToolExecutor):
        def __init__(self, delegate):
            self.delegate = delegate
        def __call__(self, action, conversation=None):
            target = Path(action.path).resolve()
            if not target.is_relative_to(root) or target.is_symlink():
                return FileEditorObservation.from_text(text="Path outside coding workspace is denied.",
                    command=action.command, path=action.path, is_error=True)
            if action.command == "create":
                # The SDK editor writes files but expects the parent directory to
                # exist. This bridge supplies that filesystem operation without
                # granting the coding agent a general execution tool.
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                except OSError as exc:
                    return FileEditorObservation.from_text(text=str(exc), command=action.command,
                        path=str(target), is_error=True)
            return self.delegate(action.model_copy(update={"path": str(target)}), conversation)

    class EvoFileEditorTool(FileEditorTool):
        @classmethod
        def create(cls, conv_state):
            tools = super().create(conv_state)
            return [tool.set_executor(ScopedExecutor(tool.executor)) for tool in tools]

    register_tool(EvoFileEditorTool.name, EvoFileEditorTool)
    llm = LLM(**request["llm"], log_completions=True,
              log_completions_folder=str(events_path.parent / "completion_logs"))
    condenser_llm = llm.model_copy(update={"usage_id": "coder_context"})
    agent = Agent(llm=llm,
        tools=[Tool(name=EvoFileEditorTool.name), Tool(name=TaskTrackerTool.name)],
        include_default_tools=["FinishTool"], critic=None,
        condenser=LLMSummarizingCondenser(llm=condenser_llm, max_size=80, keep_first=2))

    def completion_log(filename, data):
        emit("model", filename=filename, log=json.loads(data))
    def callback(event):
        data = event.model_dump(mode="json", exclude_none=True)
        # Keep public actions, observations, messages and errors; hidden reasoning is not required for progress.
        data.pop("reasoning_content", None)
        data.pop("thinking_blocks", None)
        emit("event", event=data)

    conversation = Conversation(agent=agent, workspace=str(root), callbacks=[callback],
        persistence_dir=request["persistence_dir"], conversation_id=uuid.UUID(request["conversation_id"]),
        max_iteration_per_run=request.get("max_iterations") or sys.maxsize,
        max_budget_per_run=None, visualizer=None, delete_on_close=False)
    # The SDK restores its agent/config. Attach telemetry to the actual resumed LLM objects.
    conversation.agent.llm.telemetry.set_log_completions_callback(completion_log)
    if conversation.agent.condenser:
        conversation.agent.condenser.llm.telemetry.set_log_completions_callback(completion_log)
    stopped = threading.Event()
    def watch_cancel():
        while not stopped.wait(0.25):
            if Path(request["cancel_path"]).exists():
                conversation.pause()
                return
    watcher = threading.Thread(target=watch_cancel, daemon=True)
    watcher.start()
    try:
        conversation.send_message(request["prompt"])
        conversation.run()
        state = str(getattr(conversation.state.execution_status, "value", conversation.state.execution_status))
        last = ""
        for event in conversation.state.events:
            if getattr(event, "tool_name", "") == "finish":
                last = getattr(getattr(event, "action", None), "message", "")
        emit("result", status=state, message=last, conversation_id=request["conversation_id"])
    except Exception as exc:
        emit("error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        stopped.set()
        watcher.join(timeout=1)
        conversation.close()


if __name__ == "__main__":
    main()
