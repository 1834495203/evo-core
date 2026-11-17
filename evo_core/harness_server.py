"""Expose the same research tools to EvoScientist (or any MCP runtime) over stdio.

No model loop or API credentials required in this process. Stdout is MCP only.
"""
import argparse
import asyncio
import json
from threading import RLock

from .config import load_config
from .engine import ResearchKernel
from .plugins.harness import ResearchHarnessPlugin
from .plugins.literature import LiteratureSearchPlugin
from .plugins.sanity_gate import SanityGatePlugin
from .system.research.instructions import HARNESS_INSTRUCTIONS


class NoModel:
    api_key = None

    def complete(self, *args, **kwargs):
        raise RuntimeError("The harness does not own a model; reasoning belongs to the connected agent runtime")


class HarnessSession:
    """Shared tool boundary, state and checks; no duplicated experimental implementation."""
    def __init__(self, config, task):
        self.lock = RLock()
        self.harness = ResearchHarnessPlugin()
        self.kernel = ResearchKernel(config, plugins=[self.harness, LiteratureSearchPlugin(), SanityGatePlugin()],
                                     llm_client=NoModel())
        self.ctx = self.kernel._build_ctx()
        self.state = self.kernel._build_initial_state(task, None, resume=True)
        from dataclasses import replace
        self.kernel.host.identity = replace(self.kernel.host.identity, run_id=self.state.run_id, session_id=self.state.run_id)
        self.ctx.state = self.state
        for plugin in self.kernel.plugins:
            plugin.before_run(self.state, self.ctx)
        self.kernel.modules = self.ctx.module_registry
        self.tools = self.kernel._collect_tools(self.state, self.ctx)
        # Core filesystem and execution tools are shared with the normal runtime;
        # every invocation still passes the same purpose/test/evidence gates.
        self.tools.pop("delegate_task", None)
        self.harness.before_model_call(self.state, [], self.ctx)
        self.kernel._save_state(self.state)

    def call(self, name, arguments):
        return self._call(name, arguments, typed=False)

    def invoke(self, name, arguments):
        """Typed API; call() retains the original string presentation for MCP clients."""
        return self._call(name, arguments, typed=True)

    def _call(self, name, arguments, *, typed):
        with self.lock:
            self.state.metadata["external_tool_sequence"] = self.state.metadata.get("external_tool_sequence", 0) + 1
            self.state.metadata["tool_batch_id"] = self.state.metadata["external_tool_sequence"]
            result = self.kernel.execute_tool_with_hooks(self.state, self.ctx, self.tools, name, arguments,
                         call_num=self.state.metadata["external_tool_sequence"], actor="mcp", run_after_hooks=True, return_result=typed)
            if name == "get_research_status":
                self.state.metadata["model_plan_version"] = self.ctx.research_system.store.get(self.ctx.study_id).plan_version
            self.kernel._save_state(self.state)
            return result

    def close(self):
        self.harness._pool.shutdown()
        self.kernel._save_state(self.state)
        self.ctx.module_registry.close()


async def serve(config, task):
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    session = HarnessSession(config, task)
    server = Server("research-harness", instructions=HARNESS_INSTRUCTIONS)

    @server.list_tools()
    async def list_tools():
        return [Tool(name=t.name, description=t.description, inputSchema=t.parameters)
                for t in session.tools.values()]

    @server.call_tool()
    async def call_tool(name, arguments):
        result = await asyncio.to_thread(session.call, name, arguments or {})
        return [TextContent(type="text", text=result)]

    try:
        async with stdio_server() as (reader, writer):
            await server.run(reader, writer, server.create_initialization_options())
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    config = load_config(args.config, overrides={"workspace": args.workspace, "run_dir": args.run_dir})
    asyncio.run(serve(config, args.task))


if __name__ == "__main__":
    main()
