"""The tools an MCP client is offered, pinned: names, descriptions, schemas.

UPDATE_TOOL_SNAPSHOT=1 rewrites the snapshot after a deliberate change.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import anyio
from mcp import Client

from vectrify.mcp.server import Vectrify, build_server, build_window_server

SNAPSHOT = Path(__file__).with_name("tools.json")


async def offered(server) -> list[dict]:
    async with Client(server) as client:
        tools = (await client.list_tools()).tools
    return [t.model_dump(mode="json", by_alias=True, exclude_none=True) for t in tools]


def test_tools_match_snapshot():
    tools = anyio.run(offered, build_server())
    if os.environ.get("UPDATE_TOOL_SNAPSHOT"):
        SNAPSHOT.write_text(json.dumps(tools, indent=1) + "\n")
    pinned = json.loads(SNAPSHOT.read_text())
    assert [t["name"] for t in tools] == [t["name"] for t in pinned]
    for tool, expected in zip(tools, pinned, strict=True):
        # Through JSON, as a client gets it.
        assert json.loads(json.dumps(tool)) == expected, tool["name"]

    # The window offers the same tools, less opening and connecting.
    window = anyio.run(offered, build_window_server(Vectrify()))
    assert window == [t for t in tools if t["name"] not in {"open", "connect"}]
