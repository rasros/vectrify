"""The MCP server the editor hosts over Streamable HTTP while agents may edit."""

from __future__ import annotations

import socket
import stat
from pathlib import Path

import anyio
import httpx2
import pytest
from mcp import Client, MCPError
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client

from tests.mcp.helpers import data, error, free_port, images, png_size
from vectrify.mcp.target import read_discovery
from vectrify.ui.agent import token_file
from vectrify.ui.server import Backend

SAMPLE = (Path(__file__).parents[1] / "ui" / "sample.svg").read_text()


@pytest.fixture
def editor():
    """An editor with one window, hosting MCP on a port of the test's own."""
    backend = Backend(SAMPLE, "sample.svg")
    backend.agents.mcp_port = free_port()
    _, state = backend.handle("/api/session", {}, None)
    yield backend, state["session"]
    backend.agents.close()


def client(url: str, token: str | None) -> Client:
    headers = {} if token is None else {"Authorization": f"Bearer {token}"}
    return Client(
        streamable_http_client(url, http_client=create_mcp_http_client(headers))
    )


def post(
    url: str, token: str | None, host: str | None = None, origin: str | None = None
) -> httpx2.Response:
    """A bare initialize request, as a client's first."""
    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    if host is not None:
        headers["Host"] = host
    if origin is not None:
        headers["Origin"] = origin
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "0"},
        },
    }
    return httpx2.post(url, json=body, headers=headers, timeout=5)


def test_a_client_over_http_edits_the_window(editor):
    backend, session_id = editor
    status, allowed = backend.handle("/api/agent", {"enabled": True}, session_id)
    assert status == 200
    url = allowed["mcp"]["url"]
    port = backend.agents.mcp_port
    assert url == f"http://127.0.0.1:{port}/mcp"
    token = token_file().read_text().strip()
    assert stat.S_IMODE(token_file().stat().st_mode) == 0o600
    assert allowed["mcp"]["command"] == (
        f"claude mcp add --transport http --scope user vectrify {url} "
        f'--header "Authorization: Bearer {token}"'
    )
    found = read_discovery()
    assert found is not None
    assert found["token"] == token
    window = backend.sessions[session_id]

    async def run():
        async with client(url, token) as mcp:
            tools = {t.name for t in (await mcp.list_tools()).tools}
            assert {"describe", "render", "paint", "history"} <= tools
            # Always this window: nothing to open or connect to.
            assert not {"open", "connect"} & tools
            refused = await mcp.call_tool("paint", {"ids": ["sun"], "fill": "#0f0"})
            assert "describe() first" in error(refused)
            described = data(await mcp.call_tool("describe", {}))
            assert described["target"].startswith("the editor window")
            assert "sun" in [o["id"] for o in described["objects"]]
            painted = data(
                await mcp.call_tool("paint", {"ids": ["sun"], "fill": "#00ff00"})
            )
            assert painted["step"] == "Agent: Change paint"
            rendered = await mcp.call_tool("render", {"max_side": 120})
            assert png_size(images(rendered)[0]) == (120, 80)
            history = data(await mcp.call_tool("history", {}))
            newest = history["undo"][0]
            assert (newest["label"], newest["author"]) == (
                "Agent: Change paint",
                "agent",
            )

    anyio.run(run)
    with window.lock:
        assert window.editor.undo_labels == ("Agent: Change paint",)
    # The footer's status counts the HTTP client like any agent.
    _, poll = backend.handle("/api/poll", {}, session_id)
    assert poll["agent"]["connected"]
    assert poll["agent"]["last_action"] == "Agent: Change paint"


def test_the_token_is_kept_until_regenerated(editor):
    backend, session_id = editor
    first = backend.handle("/api/agent", {"enabled": True}, session_id)[1]
    backend.handle("/api/agent", {"enabled": False}, session_id)
    again = backend.handle("/api/agent", {"enabled": True}, session_id)[1]
    assert again["mcp"]["command"] == first["mcp"]["command"]
    old = token_file().read_text().strip()
    regenerated = backend.handle("/api/agent", {"regenerate": True}, session_id)[1]
    new = token_file().read_text().strip()
    assert new != old
    assert new in regenerated["mcp"]["command"]
    url = regenerated["mcp"]["url"]
    assert post(url, old).status_code == 401
    assert post(url, new).status_code == 200


def test_a_wrong_or_missing_token_or_host_is_refused(editor):
    backend, session_id = editor
    url = backend.handle("/api/agent", {"enabled": True}, session_id)[1]["mcp"]["url"]
    token = token_file().read_text().strip()
    assert post(url, None).status_code == 401
    assert post(url, "wrong").status_code == 401
    # A page from elsewhere: through a renamed host, or from its own origin.
    assert post(url, token, host="evil.example:80").status_code == 403
    assert post(url, token, origin="http://evil.example").status_code == 403
    assert post(url, token).status_code == 200

    async def run() -> str:
        try:
            async with client(url, "wrong") as mcp:
                await mcp.list_tools()
        except MCPError as exc:
            return str(exc)
        except BaseException as exc:  # The SDK wraps it in task groups.
            return repr(exc)
        return "connected"

    assert "error response" in anyio.run(run)


def test_turning_agents_off_stops_it(editor):
    backend, session_id = editor
    url = backend.handle("/api/agent", {"enabled": True}, session_id)[1]["mcp"]["url"]
    token = token_file().read_text().strip()
    off = backend.handle("/api/agent", {"enabled": False}, session_id)[1]
    assert "mcp" not in off
    with pytest.raises(httpx2.ConnectError):
        post(url, token)
    # Off but not yet stopped refuses too.
    backend.handle("/api/agent", {"enabled": True}, session_id)
    agents = backend.agents
    agents.agent = None
    assert post(url, token).status_code == 403


def test_a_taken_port_moves_to_the_next_free_one(editor):
    backend, session_id = editor
    port = backend.agents.mcp_port
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", port))
        busy.listen(1)
        allowed = backend.handle("/api/agent", {"enabled": True}, session_id)[1]
    url = allowed["mcp"]["url"]
    assert url != f"http://127.0.0.1:{port}/mcp"
    assert int(url.split(":")[2].split("/")[0]) > port
    token = token_file().read_text().strip()
    assert post(url, token).status_code == 200
