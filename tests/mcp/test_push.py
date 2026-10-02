"""The page hears of each agent call as it happens: server-sent events from
the editor's server, and in the desktop window a script the window runs."""

from __future__ import annotations

import http.client
import json
import queue
import time
from pathlib import Path
from threading import Thread
from typing import Any

import pytest

from tests.mcp.helpers import free_port
from vectrify.mcp.target import LiveTarget, TargetError, read_discovery
from vectrify.ui.agent import discovery_file
from vectrify.ui.desktop import Api
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


def listen(server, session: str) -> tuple[http.client.HTTPConnection, queue.Queue]:
    """An EventSource's request, its events (data lines) read into a queue."""
    client = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=30)
    client.request("GET", f"/api/events?session={session}")
    response = client.getresponse()
    assert response.status == 200
    assert response.getheader("Content-Type") == "text/event-stream"
    events: queue.Queue = queue.Queue()

    def read() -> None:
        try:
            while line := response.fp.readline():
                if line.startswith(b"data: "):
                    events.put((time.perf_counter(), json.loads(line[6:])))
        except (OSError, ValueError):
            pass

    Thread(target=read, daemon=True).start()
    return client, events


def test_an_agent_edit_is_pushed_to_the_page_at_once(server):
    _, state = page(server, "/api/session", {})
    session_id = state["session"]
    page(server, "/api/agent", {"enabled": True}, session_id)
    found = read_discovery()
    assert found is not None
    target = LiveTarget(found["url"], found["token"])
    connection, events = listen(server, session_id)
    try:
        # The pulse as it is now, first.
        _, first = events.get(timeout=5)
        assert first["session"] == session_id
        assert first["revision"] == state["revision"]
        assert first["agent"]["enabled"]
        seen = target.call("describe", {}).data
        # A look is pushed too: the footer shows the agent connected.
        _, looked = events.get(timeout=5)
        assert looked["agent"]["connected"]
        assert looked["revision"] == state["revision"]
        sent = time.perf_counter()
        painted = target.call(
            "properties",
            {
                "ids": ["sun"],
                "fill": "#00ff00",
                "seen": [seen["epoch"], seen["revision"]],
            },
        ).data
        arrived, pushed = events.get(timeout=5)
        while pushed["revision"] != painted["revision"]:
            arrived, pushed = events.get(timeout=5)
        # Pushed as the call returns, not on the next poll.
        assert arrived - sent < 1.0
        assert pushed["agent"]["last_action"] == "Agent: Change paint"
        assert [t["ids"] for t in pushed["agent"]["touched"]] == [["sun"]]
        # Turning agents off is pushed as well.
        page(server, "/api/agent", {"enabled": False}, session_id)
        while pushed["agent"]["enabled"]:
            _, pushed = events.get(timeout=5)
    finally:
        connection.close()


def test_events_need_a_live_session_and_the_page_origin(server):
    client = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    client.request("GET", "/api/events?session=nope")
    response = client.getresponse()
    assert response.status == 401
    response.read()
    client.request(
        "GET", "/api/events?session=nope", headers={"Origin": "http://evil.example"}
    )
    response = client.getresponse()
    assert response.status == 403
    client.close()


def test_the_desktop_window_reaches_agents_on_the_mcp_port_alone():
    """No window opens: the bridge object the page calls, and the script the
    window would run, stand in for pywebview."""
    backend = Backend(SAMPLE, "sample.svg")
    backend.agents.mcp_port = free_port()
    api = Api(backend)
    scripts: queue.Queue = queue.Queue()
    stop = api._start_pushing(scripts.put)
    try:
        opened = api.request("/api/session", {}, None)
        assert opened["status"] == 200
        session = opened["body"]["session"]
        allowed = api.request("/api/agent", {"enabled": True}, session)["body"]
        url = allowed["mcp"]["url"]
        assert url == f"http://127.0.0.1:{backend.agents.mcp_port}/mcp"
        found = read_discovery()
        assert found is not None
        # One port, one token: connect()'s JSON channel is on the MCP port.
        assert found["url"] == url.removesuffix("/mcp")
        assert json.loads(discovery_file().read_text())["mcp"] == url
        target = LiveTarget(found["url"], found["token"])
        seen = target.call("describe", {}).data
        painted = target.call(
            "properties",
            {
                "ids": ["sun"],
                "fill": "#00ff00",
                "seen": [seen["epoch"], seen["revision"]],
            },
        ).data
        # The window is told to run the page's handler with the new pulse.
        deadline = time.monotonic() + 5
        pulse: dict = {}
        while pulse.get("revision") != painted["revision"]:
            script = scripts.get(timeout=max(0.01, deadline - time.monotonic()))
            prefix = "window.vectrifyPulse && window.vectrifyPulse("
            assert script.startswith(prefix)
            assert script.endswith(")")
            pulse = json.loads(script[len(prefix) : -1])
        assert pulse["session"] == session
        assert pulse["agent"]["last_action"] == "Agent: Change paint"
        state = api.request("/api/session", {"session": session}, None)["body"]
        assert "#00ff00" in state["svg"]
        # A wrong token is refused on that port as on the editor's own.
        with pytest.raises(TargetError, match="token"):
            LiveTarget(found["url"], "wrong").call("hello", {})
    finally:
        stop.set()
        backend.agents.close()
