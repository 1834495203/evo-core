"""Use the same MCP adapter as upstream EvoScientist against a real stdio subprocess."""
import asyncio
import json
from pathlib import Path
import sys

import pytest


def test_upstream_mcp_adapter_discovers_tools_and_persists_notes(tmp_path):
    pytest.importorskip("mcp")
    client_module = pytest.importorskip("langchain_mcp_adapters.client")
    root = Path(__file__).resolve().parents[1]
    connection = {"transport": "stdio", "command": sys.executable,
                  "args": [str(root / "serve_harness.py"), "--config", str(root / "config.default.yaml"),
                           "--workspace", str(tmp_path / "workspace"), "--run-dir", str(tmp_path / "meta"),
                           "--task", "MCP integration fixture"]}

    async def check():
        client = client_module.MultiServerMCPClient({"research": connection})
        async with client.session("research") as session:
            tools = await session.list_tools()
            assert {"get_research_status", "read_research_document", "submit_experiment"} <= {t.name for t in tools.tools}
            result = await session.call_tool("record_research_document", {"kind":"question", "text":"Persist across MCP restart?"})
            assert not result.isError
            record_id = json.loads(result.content[0].text)["record_id"]
        # Reconnect through a new server process: no duplicate study and no lost document.
        async with client.session("research") as session:
            result = await session.call_tool("query_research_records", {"kind":"question"})
            records = json.loads(result.content[0].text)
            assert records["total"] == 1
            assert records["records"][0]["id"] == record_id
    asyncio.run(asyncio.wait_for(check(), timeout=25))
