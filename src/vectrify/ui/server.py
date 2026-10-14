"""The editor's backend, and the loopback-only server that serves it.

``Backend`` answers the page's requests; the HTTP handler here and the desktop
window (``vectrify.ui.desktop``) are two ways of reaching it. No frontend build
step is required.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import secrets
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from vectrify.document import DocumentError, import_svg
from vectrify.document.editor import PinnedDeletionError
from vectrify.project_file import MAX_SOURCE, decode_source
from vectrify.ui.agent_channel import MCP_PORT, AgentChannel, answer
from vectrify.ui.session import Session
from vectrify.ui.shared import REFUSALS, png, refusal

STATIC = Path(__file__).with_name("static")
# How often an idle push channel says it is still there, in seconds.
KEEPALIVE = 15.0
# What the editor opens with when it is given no drawing: an empty artboard.
BLANK = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="800" '
    'viewBox="0 0 1200 800"/>'
)


class Backend:
    """Editor sessions over one starting document, and the page's requests."""

    def __init__(
        self,
        initial: str | bytes | None = None,
        name: str = "Untitled.svg",
        reference: dict | None = None,
    ):
        source = decode_source(initial or BLANK)
        self.project_source = source if source.lstrip().startswith("{") else None
        self.document = import_svg(BLANK if self.project_source else source)
        self.name = name
        self.reference = reference
        self.sessions: dict[str, Session] = {}
        # The door agents edit a window's session through, once it allows it.
        self.agents = AgentChannel(self)

    def pulse(self, session: Session, session_id: str) -> dict[str, Any]:
        """What the page is told of its session's drawing and agents."""
        return {
            "session": session_id,
            **session.content_version(),
            "agent": self.agents.status(session_id),
        }

    def events(self, session_id: str) -> Iterator[dict[str, Any] | None]:
        """The page's push channel: its pulse now and after each agent call,
        or None when nothing happened for a while (to keep the line open).
        Ends when the session does."""
        channel = self.agents
        beat = channel.beat
        while True:
            session = self.sessions.get(session_id)
            if session is None:
                return
            with session.lock:
                pulse = self.pulse(session, session_id)
            yield pulse
            while (new := channel.wait(beat, KEEPALIVE)) == beat:
                yield None
            beat = new

    def handle(self, path: str, data: Any, session_id: str | None) -> tuple[int, dict]:
        """Answer one request as (status, body), whatever carried it."""
        try:
            if not isinstance(data, dict):
                raise DocumentError("Expected an editor request")
            if path == "/api/session":
                return 200, self._session(data)
            session = self.sessions.get(session_id or "")
            if session is None:
                return 401, {"error": "Editor session expired; reload this page"}
            # The comparison captures its own snapshot under the lock, then
            # rasterizes it without holding up the person's next edit.
            if path in {"/api/reference-error", "/api/reference-error-cache"}:
                return 200, ROUTES[path](self, session, data, session_id or "")
            with session.lock:
                route = ROUTES.get(path)
                if route is None:
                    return 404, {"error": "Not found"}
                return 200, route(self, session, data, session_id or "")
        except PinnedDeletionError as exc:
            if path == "/api/action" and data.get("command") == "delete":
                return 200, {
                    "confirmation": "delete_pinned",
                    "message": (
                        "The selected objects contain pinned points. "
                        "Delete them anyway?"
                    ),
                }
            return 400, {"error": str(exc)}
        except REFUSALS as exc:
            return refusal(exc)

    def _session(self, data: dict) -> dict[str, Any]:
        session_id = data.get("session")
        if session_id not in self.sessions:
            session_id = secrets.token_urlsafe(24)
            session = Session(
                self.document,
                self.name,
                self.reference if not self.project_source else None,
            )
            if self.project_source:
                session.open(self.project_source, self.name)
                if self.reference:
                    session.reference = session.fit_reference(self.reference)
            self.sessions[session_id] = session
        session = self.sessions[session_id]
        with session.lock:
            result = session.state()
            result["session"] = session_id
        return result

    def _nodes(self, session: Session, data: dict, _session_id: str) -> dict:
        session.check_revision(data)
        if "objects" in data:
            return session.geometries(data["objects"])
        return session.nodes(data["object"])

    def _poll(self, session: Session, data: dict, session_id: str) -> dict:
        # What the page checks for agents' edits, and the footer (pushed to it
        # as they happen, polled now and then); it says what it shows, for an
        # agent's describe().
        if data.get("view") is not None:
            session.set_view(data["view"])
        return self.pulse(session, session_id)

    def _agent(self, _session: Session, data: dict, session_id: str) -> dict:
        if data.get("regenerate"):
            return self.agents.regenerate(session_id)
        if data.get("enabled"):
            return self.agents.enable(session_id)
        return self.agents.disable(session_id)

    def _reference(self, session: Session, _data: dict, _session_id: str) -> dict:
        return {**session.content_version(), "reference": session.reference}

    def _reference_error(self, session: Session, data: dict, _session_id: str) -> dict:
        with session.lock:
            session.check_revision(data)
            if data.get("view") is not None:
                session.set_view(data["view"])
        return {
            "image": "data:image/png;base64,"
            + base64.b64encode(
                png(session.reference_error(data), compress_level=1)
            ).decode(),
        }


# The page's requests on its session, by path: (backend, session, data,
# session id) to the answer's body.
ROUTES: dict[str, Callable[[Backend, Session, dict, str], Any]] = {
    "/api/action": lambda _backend, session, data, _id: session.action(data),
    "/api/operation": lambda _backend, session, data, _id: session.operation(data),
    "/api/holes": lambda _backend, session, data, _id: session.holes(data),
    "/api/region": lambda _backend, session, data, _id: session.region(data),
    "/api/sample": lambda _backend, session, data, _id: session.sample(data),
    "/api/nodes": Backend._nodes,
    "/api/poll": Backend._poll,
    "/api/agent": Backend._agent,
    "/api/reference": Backend._reference,
    "/api/reference-error": Backend._reference_error,
    "/api/reference-error-cache": lambda _backend, session, data, _id: (
        session.cache_reference_error(data)
    ),
    "/api/export": lambda _backend, session, data, _id: session.export(data),
}


class EditorServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        initial: str | bytes | None = None,
        name: str = "Untitled.svg",
        reference: dict | None = None,
        backend: Backend | None = None,
    ):
        super().__init__(address, Handler)
        self.backend = backend or Backend(initial, name, reference)
        # Agents reach a window served here on this same port.
        self.backend.agents.url = f"http://127.0.0.1:{self.server_port}"


class Handler(BaseHTTPRequestHandler):
    server: Any

    def log_message(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def respond(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
            "connect-src 'self'; object-src 'none'; frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(body)

    def json(self, status: int, value: Any) -> None:
        self.respond(
            status, json.dumps(value, allow_nan=False).encode(), "application/json"
        )

    def same_origin(self) -> bool:
        host = self.headers.get("Host", "")
        expected = f"127.0.0.1:{self.server.server_port}"
        localhost = f"localhost:{self.server.server_port}"
        return (
            host in {expected, localhost}
            and self.headers.get("Origin", f"http://{host}") == f"http://{host}"
        )

    def do_GET(self) -> None:
        if self.path.startswith("/agent/"):
            answer(self, self.server.backend.agents, self.server.server_port)
            return
        if not self.same_origin():
            self.json(
                403, {"error": "This editor accepts local same-origin requests only"}
            )
            return
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/events":
            self.events(parse_qs(parsed.query).get("session", [""])[0])
            return
        # Every module in static/ is servable by name, so new controllers load
        # without a route of their own; only plain top-level names qualify.
        files = {
            "/": "index.html",
            "/style.css": "style.css",
            "/favicon.svg": "favicon.svg",
        } | {f"/{file.name}": file.name for file in STATIC.glob("*.js")}
        if path not in files:
            self.json(404, {"error": "Not found"})
            return
        file = STATIC / files[path]
        self.respond(
            200, file.read_bytes(), mimetypes.guess_type(file.name)[0] or "text/plain"
        )

    def events(self, session_id: str) -> None:
        """Server-sent events: the page's pulse after each agent call."""
        backend = self.server.backend
        if session_id not in backend.sessions:
            self.json(401, {"error": "Editor session expired; reload this page"})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            for pulse in backend.events(session_id):
                if pulse is None:
                    self.wfile.write(b": still here\n\n")
                else:
                    data = json.dumps(pulse, allow_nan=False)
                    self.wfile.write(f"data: {data}\n\n".encode())
                self.wfile.flush()
        except OSError:
            # The page went away.
            pass
        self.close_connection = True

    def do_POST(self) -> None:
        # Agents carry a token rather than the page's origin.
        if self.path.startswith("/agent/"):
            answer(self, self.server.backend.agents, self.server.server_port)
            return
        if (
            not self.same_origin()
            or self.headers.get_content_type() != "application/json"
        ):
            self.json(403, {"error": "Use the local editor to send commands"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_SOURCE * 2:
                self.json(413, {"error": "Request exceeds the editor limit"})
                return
            data = json.loads(self.rfile.read(length))
        except (ValueError, OSError) as exc:
            self.json(400, {"error": str(exc)})
            return
        status, result = self.server.backend.handle(
            self.path, data, self.headers.get("X-Vectrify-Session")
        )
        self.json(status, result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the Vectrify SVG editor")
    parser.add_argument(
        "svg", nargs="?", type=Path, help="SVG or .vectrify project to open initially"
    )
    parser.add_argument("--reference", type=Path, help="PNG/JPEG/WebP reference image")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Serve the editor to a browser instead of opening a desktop window",
    )
    parser.add_argument("--port", type=int, default=8765, help="Port for --serve")
    parser.add_argument(
        "--mcp-port",
        type=int,
        default=None,
        help="Where to host the MCP server while agents are allowed (default: "
        f"the port used last time, else {MCP_PORT}; the next free port if taken)",
    )
    args = parser.parse_args()
    reference = None
    if args.reference:
        mime = mimetypes.guess_type(args.reference.name)[0]
        reference = Session.validate_reference(
            {
                "name": args.reference.name,
                "data_url": f"data:{mime};base64,"
                + base64.b64encode(args.reference.read_bytes()).decode(),
                "opacity": 0.5,
            }
        )
    backend = Backend(
        args.svg.read_bytes() if args.svg else None,
        args.svg.name if args.svg else "Untitled.svg",
        reference,
    )
    backend.agents.mcp_port = args.mcp_port
    if not args.serve:
        from vectrify.ui import desktop

        if desktop.available():
            desktop.run(backend, args.svg)
            return
        print(
            "Desktop window unavailable (install vectrify[desktop]); "
            "serving to a browser instead.",
            flush=True,
        )
    server = EditorServer(("127.0.0.1", args.port), backend=backend)
    print(f"Vectrify editor: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
