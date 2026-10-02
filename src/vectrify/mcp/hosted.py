"""The MCP server the editor hosts itself while a window allows agents.

The same tools as ``vectrify-mcp``, served over the MCP Streamable HTTP
transport at ``http://127.0.0.1:<port>/mcp`` from a background thread
(uvicorn running the SDK's ASGI app), on the window's session in process. A
client adds the URL once, with the editor's stable token as a bearer header.
The same port, with the same token, carries ``/agent/call``, the JSON channel
``vectrify-mcp``'s ``connect()`` uses, for the desktop window, which has no
HTTP server of its own.
"""

from __future__ import annotations

import hmac
import json
import socket
import threading
from typing import TYPE_CHECKING

import anyio.to_thread
import uvicorn
from mcp.server.transport_security import TransportSecuritySettings
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from vectrify.mcp.server import Vectrify, build_window_server
from vectrify.mcp.target import WindowTarget
from vectrify.ui.agent import MAX_CALL, OFF

if TYPE_CHECKING:
    from vectrify.ui.agent import AgentChannel

PATH = "/mcp"


class Guard:
    """Refuse a request not addressed to this machine, made while agents are
    off, or without the token; pass the rest (and the lifespan) to *app*."""

    def __init__(self, app: ASGIApp, channel: AgentChannel, port: int):
        self.app = app
        self.channel = channel
        self.hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            headers = Headers(scope=scope)
            token = self.channel.token
            given = headers.get("authorization", "")
            if headers.get("host", "") not in self.hosts:
                refusal = 403, "Agents reach the editor on 127.0.0.1 only"
            elif token is None or self.channel.agent is None:
                refusal = 403, OFF
            elif not hmac.compare_digest(given.encode(), f"Bearer {token}".encode()):
                refusal = (
                    401,
                    (
                        "Wrong or missing agent token. Copy the command from the "
                        "editor's Agents popover again."
                    ),
                )
            else:
                refusal = None
            if refusal is not None:
                status, message = refusal
                body = json.dumps({"error": message}).encode()
                await respond(send, status, "application/json", body)
                return
            if scope["path"].startswith("/agent/"):
                await self.agent_call(scope, receive, send, headers)
                return
        await self.app(scope, receive, send)

    async def agent_call(
        self, scope: Scope, receive: Receive, send: Send, headers: Headers
    ) -> None:
        """``vectrify-mcp``'s JSON channel (``connect()``), on this same port."""
        body = bytearray()
        more = True
        while more:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body += message.get("body", b"")
            more = message.get("more_body", False)
            if len(body) > MAX_CALL:
                await respond(
                    send,
                    413,
                    "application/json",
                    b'{"error": "Request exceeds the editor limit"}',
                )
                return
        path = scope["path"] + (
            "?" + scope["query_string"].decode() if scope.get("query_string") else ""
        )
        # An agent call may wait (a job's status up to two minutes): off the
        # event loop, so other clients keep being answered.
        status, kind, answer = await anyio.to_thread.run_sync(
            self.channel.http, scope["method"], path, headers, bytes(body)
        )
        await respond(send, status, kind, answer)


async def respond(send: Send, status: int, kind: str, body: bytes) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", kind.encode()),
                (b"content-length", str(len(body)).encode()),
                (b"cache-control", b"no-store"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def bind(port: int, tries: int) -> socket.socket:
    """A listening socket on 127.0.0.1 at *port*, or the next free one."""
    error: OSError | None = None
    for candidate in range(port, port + max(tries, 1)):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", candidate))
        except OSError as exc:
            sock.close()
            error = exc
            continue
        sock.listen(128)
        return sock
    assert error is not None
    raise error


class HostedMCP:
    """One run of the hosted server: started when allowed, stopped when not."""

    def __init__(self, channel: AgentChannel):
        self.channel = channel
        self.url: str | None = None
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._sock: socket.socket | None = None

    def start(self, port: int, tries: int = 1) -> str:
        sock = self._sock = bind(port, tries)
        port = sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}{PATH}"
        # A fresh MCP server each run: its session manager runs only once.
        mcp = build_window_server(Vectrify(target=WindowTarget(self.channel, self.url)))
        app = mcp.streamable_http_app(
            streamable_http_path=PATH,
            transport_security=TransportSecuritySettings(
                allowed_hosts=[f"127.0.0.1:{port}", f"localhost:{port}"],
                allowed_origins=[
                    f"http://127.0.0.1:{port}",
                    f"http://localhost:{port}",
                ],
            ),
        )
        config = uvicorn.Config(
            Guard(app, self.channel, port),
            log_level="warning",
            lifespan="on",
            timeout_graceful_shutdown=1,
        )
        server = uvicorn.Server(config)
        self._server = server
        self._thread = threading.Thread(
            target=server.run,
            kwargs={"sockets": [sock]},
            daemon=True,
            name="vectrify-mcp-http",
        )
        self._thread.start()
        # Answer from the first request on.
        while not server.started and self._thread.is_alive():
            self._thread.join(0.02)
        if not server.started:
            sock.close()
            raise OSError("The MCP server did not start")
        return self.url

    def stop(self, timeout: float = 3.0) -> None:
        server, thread = self._server, self._thread
        self._server = self._thread = None
        if server is None or thread is None:
            return
        server.should_exit = True
        thread.join(timeout)
        if thread.is_alive():
            server.force_exit = True
            thread.join(1.0)
        if self._sock is not None:
            self._sock.close()
            self._sock = None
