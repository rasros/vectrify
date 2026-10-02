"""The live channel: an MCP client editing the drawing a window shows."""

from __future__ import annotations

import http.client
import json
import stat
from pathlib import Path
from threading import Thread
from typing import Any

import anyio
import pytest
from mcp import Client

from tests.mcp.helpers import data, error, free_port, images, png_size
from vectrify.mcp.server import Vectrify, build_server
from vectrify.mcp.target import LiveTarget, TargetError, read_discovery
from vectrify.ui.agent import discovery_file
from vectrify.ui.server import Backend, EditorServer

SAMPLE = (Path(__file__).parents[1] / "ui" / "sample.svg").read_text()


@pytest.fixture
def server():
    server = EditorServer(("127.0.0.1", 0), SAMPLE)
    server.backend.agents.mcp_port = free_port()
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.backend.agents.close()
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def page(server, path: str, payload: dict, session: str = "") -> tuple[int, Any]:
    """A request as the editor's page sends it."""
    client = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    client.request(
        "POST",
        path,
        body=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "X-Vectrify-Session": session},
    )
    response = client.getresponse()
    body = json.loads(response.read())
    client.close()
    return response.status, body


def agent(server, token: str | None, payload: dict) -> tuple[int, Any]:
    client = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    client.request("POST", "/agent/call", body=json.dumps(payload), headers=headers)
    response = client.getresponse()
    body = json.loads(response.read())
    client.close()
    return response.status, body


def test_agents_edit_the_window_only_when_allowed_and_with_the_token(server):
    _, state = page(server, "/api/session", {})
    session = state["session"]
    hello = {"tool": "hello", "args": {}}
    status, body = agent(server, "anything", hello)
    assert status == 403
    assert "Allow agents" in body["error"]
    assert read_discovery() is None

    status, allowed = page(server, "/api/agent", {"enabled": True}, session)
    assert status == 200
    assert allowed["enabled"]
    assert not allowed["connected"]
    found = read_discovery()
    assert found is not None
    assert found["url"] == f"http://127.0.0.1:{server.server_port}"
    assert stat.S_IMODE(discovery_file().stat().st_mode) == 0o600

    assert agent(server, None, hello)[0] == 401
    assert agent(server, "wrong", hello)[0] == 401
    status, body = agent(server, found["token"], hello)
    assert status == 200
    assert body["data"]["revision"] == 0

    status, _ = page(server, "/api/agent", {"enabled": False}, session)
    assert status == 200
    assert read_discovery() is None
    assert agent(server, found["token"], hello)[0] == 403


def test_an_mcp_client_edits_live_and_the_window_sees_it(server):
    _, state = page(server, "/api/session", {})
    session_id = state["session"]
    page(server, "/api/agent", {"enabled": True}, session_id)
    window = server.backend.sessions[session_id]
    vectrify = Vectrify()

    async def run():
        async with Client(build_server(vectrify)) as client:
            # With an editor allowing agents, the server attaches by itself.
            described = data(await client.call_tool("describe", {}))
            assert described["target"].startswith("the editor window")
            assert "sun" in [o["id"] for o in described["objects"]]

            painted = data(
                await client.call_tool("paint", {"ids": ["sun"], "fill": "#00ff00"})
            )
            assert painted["step"] == "Agent: Change paint"
            # Images travel as PNG bodies and arrive as image content.
            rendered = await client.call_tool("render", {"max_side": 120})
            assert png_size(images(rendered)[0]) == (120, 80)

            # The page's poll sees an agent edit to fetch, and who made it.
            status, poll = page(server, "/api/poll", {}, session_id)
            assert status == 200
            assert poll["agent"]["connected"]
            assert poll["agent"]["changes"] >= 1
            assert poll["revision"] == painted["revision"]
            _, shown = page(server, "/api/session", {"session": session_id})
            assert "#00ff00" in shown["svg"]
            assert shown["undo"] == ["Agent: Change paint"]

            # The person edits; the agent, not having looked, is refused.
            page(
                server,
                "/api/action",
                {
                    "command": "select",
                    "objects": ["sun"],
                    "epoch": shown["epoch"],
                    "revision": shown["revision"],
                },
                session_id,
            )
            page(
                server,
                "/api/action",
                {
                    "command": "paint",
                    "changes": {"fill": "#0000ff"},
                    "epoch": shown["epoch"],
                    "revision": shown["revision"],
                },
                session_id,
            )
            stale = await client.call_tool("move", {"ids": ["sun"], "dx": 5, "dy": 0})
            assert "changed since you last looked" in error(stale)
            history = data(await client.call_tool("history", {}))
            assert [(e["label"], e["author"]) for e in history["undo"]] == [
                ("Change paint", "person"),
                ("Agent: Change paint", "agent"),
            ]
            data(await client.call_tool("undo", {}))

    anyio.run(run)
    with window.lock:
        assert window.editor.undo_labels == ("Agent: Change paint",)
        assert window.editor.redo_labels == ("Change paint",)


def test_the_persons_selection_stays_and_the_poll_names_what_the_agent_touched(
    server,
):
    _, state = page(server, "/api/session", {})
    session_id = state["session"]
    page(server, "/api/agent", {"enabled": True}, session_id)
    found = read_discovery()
    assert found is not None
    token = found["token"]
    _, geometry = page(
        server,
        "/api/nodes",
        {"objects": ["river"], "epoch": state["epoch"], "revision": 0},
        session_id,
    )
    river = [
        n["id"]
        for sp in geometry["geometries"]["river"]["subpaths"]
        for n in sp["nodes"]
    ]
    # The person is in Nodes with two of the river's points selected.
    _, chosen = page(
        server,
        "/api/action",
        {
            "command": "select",
            "objects": ["river"],
            "nodes": river[:2],
            "epoch": state["epoch"],
            "revision": state["revision"],
        },
        session_id,
    )
    seen = [state["epoch"], state["revision"]]
    _, painted = agent(
        server,
        token,
        {"tool": "paint", "args": {"seen": seen, "ids": ["sun"], "fill": "#00f"}},
    )
    _, shown = page(server, "/api/session", {"session": session_id})
    assert shown["selection"] == chosen["selection"]
    assert shown["undo"] == ["Agent: Change paint"]
    _, poll = page(server, "/api/poll", {}, session_id)
    assert poll["revision"] == painted["data"]["revision"]
    assert poll["agent"]["touched"] == [{"change": 1, "ids": ["sun"]}]

    # The agent deletes the person's selected path: it leaves their selection.
    seen = [painted["data"]["epoch"], painted["data"]["revision"]]
    agent(server, token, {"tool": "delete", "args": {"seen": seen, "ids": ["river"]}})
    _, shown = page(server, "/api/session", {"session": session_id})
    assert shown["selection"] == {"objects": [], "nodes": []}
    _, poll = page(server, "/api/poll", {}, session_id)
    # Nothing left to show of a deletion.
    assert [t["change"] for t in poll["agent"]["touched"]] == [1]
    assert poll["agent"]["changes"] == 2


def test_the_desktop_window_opens_a_port_of_its_own_while_allowed():
    backend = Backend(SAMPLE, "sample.svg")
    backend.agents.mcp_port = free_port()
    _, state = backend.handle("/api/session", {}, None)
    session = state["session"]
    status, allowed = backend.handle("/api/agent", {"enabled": True}, session)
    assert status == 200
    assert allowed["enabled"]
    found = read_discovery()
    assert found is not None
    target = LiveTarget(found["url"], found["token"])
    assert target.call("hello", {}).data["name"] == "sample.svg"
    backend.handle("/api/agent", {"enabled": False}, session)
    assert read_discovery() is None
    with pytest.raises(TargetError):
        target.call("hello", {})


def test_the_window_reports_its_view_with_its_poll_for_view(server):
    _, state = page(server, "/api/session", {})
    session_id = state["session"]
    page(server, "/api/agent", {"enabled": True}, session_id)
    found = read_discovery()
    assert found is not None
    page(
        server,
        "/api/action",
        {
            "command": "select",
            "objects": ["sun"],
            "epoch": state["epoch"],
            "revision": 0,
        },
        session_id,
    )
    shown = {
        "region": [10, 20, 60, 40],
        "zoom": 4,
        "pixels": [240, 160],
        "tool": "select",
        "entered": None,
        "reference_view": None,
        "reference_opacity": None,
    }
    status, _ = page(server, "/api/poll", {"view": shown}, session_id)
    assert status == 200
    status, bad = page(server, "/api/poll", {"view": {"zoom": 2}}, session_id)
    assert status == 400
    assert "region" in bad["error"]
    status, body = agent(server, found["token"], {"tool": "view", "args": {}})
    assert status == 200
    view = body["data"]
    assert view["window"] is True
    assert view["region"] == [10, 20, 60, 40]
    assert view["zoom"] == 4
    assert view["tool"] == "select"
    assert view["selection"]["objects"] == ["sun"]
    status, body = agent(
        server, found["token"], {"tool": "render", "args": {"region": "view"}}
    )
    assert status == 200
    assert body["data"]["pixels"] == [240, 160]
    # Looking changes neither the person's selection nor the drawing.
    _, after = page(server, "/api/session", {"session": session_id})
    assert after["selection"]["objects"] == ["sun"]
    assert after["revision"] == 0
