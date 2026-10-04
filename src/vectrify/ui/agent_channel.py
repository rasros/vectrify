"""One editor window's authenticated agent channel and server lifecycle."""

from __future__ import annotations

import atexit
import contextlib
import hmac
import json
import os
import secrets
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from vectrify.document import StaleRevisionError
from vectrify.ui.agent_setup import (
    _write_private,
    agent_token,
    app_setup,
    claude_command,
    discovery_file,
    local_host,
    mcp_url,
    port_file,
    remembered_port,
)

if TYPE_CHECKING:
    from vectrify.ui.agent import Agent

# How long after its last call an agent still shows as connected.
CONNECTED = 120.0
# Where the editor hosts the MCP server while a window allows agents, unless
# it hosted it elsewhere before (or --mcp-port says), and how many ports after
# it to try when it is taken.
MCP_PORT = 8770
MCP_PORTS = 20
OFF = (
    "Agent editing is off in this editor. Turn on 'Allow agents to edit' "
    "('Agents') in its footer."
)


class AgentChannel:
    """One window's door for agents: HTTP on localhost, with a token.

    Off until the window allows agents to edit. Then the editor hosts the MCP
    server itself (Streamable HTTP at ``/mcp``): on *mcp_port* when one is
    given (``--mcp-port``), else on the port it used last time, else on
    ``MCP_PORT``; the next free port when that is taken, saying the URL
    moved. It keeps the JSON channel ``vectrify-mcp``'s ``connect()`` uses
    (``/agent/call``): in ``--serve`` mode the editor's own server carries it
    (*url* is that server's), in the desktop app the MCP port does, with the
    same token.

    Each agent call (and each change of the channel) is a *beat*: the page
    waits on it (``wait``) to show the agent's edits at once.
    """

    def __init__(self, backend, mcp_port: int | None = None):
        self.backend = backend
        self.url: str | None = None
        self.mcp_port = mcp_port
        self.mcp_url: str | None = None
        self.mcp_error: str | None = None
        # {old, new}: the URL clients were given, and the one it moved to.
        self.mcp_moved: dict[str, str] | None = None
        self.session_id: str | None = None
        self.token: str | None = None
        self.agent: Agent | None = None
        self._hosted: Any = None
        self._images: OrderedDict[str, bytes] = OrderedDict()
        self._lock = threading.Lock()
        self._apps: dict[str, str] | None = None
        self._pulse = threading.Condition()
        self.beat = 0
        atexit.register(self.close)

    @property
    def apps(self) -> dict[str, str]:
        """The snippets that add ``vectrify-mcp`` to other apps, found once."""
        if self._apps is None:
            self._apps = app_setup()
        return self._apps

    def changed(self) -> None:
        """Wake whatever waits for the agent's next call."""
        with self._pulse:
            self.beat += 1
            self._pulse.notify_all()

    def wait(self, beat: int, timeout: float) -> int:
        """The beat after *beat*, or *beat* again after *timeout* seconds."""
        with self._pulse:
            self._pulse.wait_for(lambda: self.beat != beat, timeout)
            return self.beat

    def status(self, session_id: str | None) -> dict[str, Any]:
        enabled = session_id is not None and session_id == self.session_id
        agent = self.agent if enabled else None
        connected = bool(
            agent and agent.last_time and time.monotonic() - agent.last_time < CONNECTED
        )
        result: dict[str, Any] = {
            "enabled": enabled,
            "connected": connected,
            "last_action": agent.last_action if agent else None,
            "changes": agent.changes if agent else 0,
            # What the agent's recent changes touched, for the window to show.
            "touched": list(agent.touched) if agent else [],
        }
        if enabled and self.token is not None:
            result["apps"] = self.apps
            if self.mcp_url is not None:
                mcp: dict[str, Any] = {
                    "url": self.mcp_url,
                    "command": claude_command(self.mcp_url, self.token),
                }
                if self.mcp_moved is not None:
                    mcp["moved"] = dict(self.mcp_moved)
                result["mcp"] = mcp
            elif self.mcp_error is not None:
                result["mcp_error"] = self.mcp_error
        return result

    def enable(self, session_id: str) -> dict[str, Any]:
        from vectrify.ui.agent import Agent

        session = self.backend.sessions[session_id]
        with self._lock:
            # One window of this editor at a time.
            if self.session_id != session_id or self.agent is None:
                self._forget()
                self.agent = Agent(session)
                self.agent.on_call = self.changed
                self.session_id = session_id
            self.token = agent_token()
            self._host()
            url = self._call_url()
            if url is not None:
                self._write(url, self.token)
        self.changed()
        return self.status(session_id)

    def _call_url(self) -> str | None:
        """Where ``/agent/call`` is: the editor's server, else the MCP port."""
        if self.url is not None:
            return self.url
        if self.mcp_url is not None:
            return self.mcp_url.removesuffix("/mcp")
        return None

    def _host(self) -> None:
        """Serve MCP over HTTP from this editor, if the SDK is installed."""
        if self._hosted is not None:
            return
        try:
            from vectrify.mcp.hosted import HostedMCP
        except ImportError:
            self.mcp_error = (
                "Install vectrify[mcp] to host the MCP server in the editor"
                + (
                    "; vectrify-mcp can still connect()."
                    if self.url is not None
                    else "."
                )
            )
            return
        remembered = remembered_port()
        preferred = self.mcp_port or remembered or MCP_PORT
        hosted = HostedMCP(self)
        try:
            url = hosted.start(preferred, MCP_PORTS)
        except OSError as exc:
            self.mcp_error = f"Could not host the MCP server: {exc}"
            return
        self._hosted, self.mcp_url, self.mcp_error = hosted, url, None
        # A client may hold the URL it had: given --mcp-port, or used before.
        expected = mcp_url(preferred)
        given = self.mcp_port is not None or remembered is not None
        self.mcp_moved = (
            {"old": expected, "new": url} if given and url != expected else None
        )
        with contextlib.suppress(OSError):
            _write_private(port_file(), url.rsplit(":", 1)[1].split("/")[0] + "\n")

    def regenerate(self, session_id: str) -> dict[str, Any]:
        """Make a new token; clients holding the old one are refused."""
        with self._lock:
            self._forget()
            token = agent_token(regenerate=True)
            if self.token is not None:
                self.token = token
                url = self._call_url()
                if url is not None:
                    self._write(url, token)
        self.changed()
        return self.status(session_id)

    def disable(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            if self.session_id == session_id:
                self._forget()
                if self.agent is not None:
                    self.agent.on_call = None
                self.session_id = self.token = self.agent = None
                self._unhost()
        self.changed()
        return self.status(session_id)

    def _unhost(self) -> None:
        hosted, self._hosted, self.mcp_url = self._hosted, None, None
        self.mcp_moved = None
        if hosted is not None:
            hosted.stop()

    def close(self) -> None:
        """At quit: forget the discovery file and stop hosting MCP."""
        self._forget()
        self._unhost()

    def _write(self, url: str, token: str) -> None:
        info = {"url": url, "token": token, "pid": os.getpid()}
        if self.mcp_url is not None:
            info["mcp"] = self.mcp_url
        _write_private(discovery_file(), json.dumps(info))

    def _forget(self) -> None:
        """Remove the discovery file, if it still names this editor."""
        path = discovery_file()
        with contextlib.suppress(OSError, ValueError):
            if json.loads(path.read_text()).get("pid") == os.getpid():
                path.unlink()

    def http(self, method: str, path: str, headers: Any, body: bytes) -> tuple:
        """Answer one agent request as (status, content type, body)."""

        from vectrify.ui.agent import REFUSALS, RefusedError, reason

        def error(status: int, message: str) -> tuple:
            return status, "application/json", json.dumps({"error": message}).encode()

        token = self.token
        given = headers.get("Authorization", "")
        if token is None or self.agent is None:
            return error(403, OFF)
        if not hmac.compare_digest(given.encode(), f"Bearer {token}".encode()):
            return error(401, "Wrong agent token; read editor.json again")
        agent = self.agent
        route = urlparse(path).path
        if method == "GET" and route.startswith("/agent/image/"):
            image = self._images.get(route.rsplit("/", 1)[1])
            if image is None:
                return error(404, "This image has expired; render again")
            return 200, "image/png", image
        if method != "POST" or route != "/agent/call":
            return error(404, "Not found")
        try:
            request = json.loads(body)
            if not isinstance(request, dict) or not isinstance(
                request.get("tool"), str
            ):
                raise ValueError("Expected {tool, args}")
            reply = agent.call(request["tool"], request.get("args") or {})
        except StaleRevisionError as exc:
            return error(409, str(exc))
        except RefusedError as exc:
            return (
                400,
                "application/json",
                json.dumps({"error": str(exc), "where": exc.where}).encode(),
            )
        except REFUSALS as exc:
            return error(400, reason(exc))
        names = []
        with self._lock:
            for name, png in reply.images:
                key = secrets.token_urlsafe(12)
                self._images[key] = png
                names.append([name, key])
            while len(self._images) > 32:
                self._images.popitem(last=False)
        return (
            200,
            "application/json",
            json.dumps({"data": reply.data, "images": names}, allow_nan=False).encode(),
        )


# How big an agent's request may be.
MAX_CALL = 300 * 1024 * 1024


def answer(handler: BaseHTTPRequestHandler, channel: AgentChannel, port: int) -> None:
    """Serve one agent request on *handler*'s connection."""
    if not local_host(handler.headers, port):
        status, kind, body = (
            403,
            "application/json",
            b'{"error": "Agents reach the editor on 127.0.0.1 only"}',
        )
    else:
        data = b""
        if handler.command == "POST":
            length = int(handler.headers.get("Content-Length", "0") or 0)
            if not 0 <= length <= MAX_CALL:
                length = 0
            data = handler.rfile.read(length)
        status, kind, body = channel.http(
            handler.command, handler.path, handler.headers, data
        )
    handler.send_response(status)
    handler.send_header("Content-Type", kind)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)
