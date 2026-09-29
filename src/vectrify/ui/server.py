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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from vectrify.document import DocumentError, StaleRevisionError, import_svg
from vectrify.llm import keys
from vectrify.ui.session import MAX_SOURCE, Session

STATIC = Path(__file__).with_name("static")
# What the editor opens with when it is given no drawing: an empty artboard.
BLANK = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="800" '
    'viewBox="0 0 1200 800"/>'
)


class Backend:
    """Editor sessions over one starting document, and the page's requests."""

    def __init__(
        self,
        initial: str | None = None,
        name: str = "Untitled.svg",
        reference: dict | None = None,
    ):
        self.document = import_svg(initial or BLANK)
        self.name = name
        self.reference = reference
        self.sessions: dict[str, Session] = {}

    def handle(self, path: str, data: Any, session_id: str | None) -> tuple[int, dict]:
        """Answer one request as (status, body), whatever carried it."""
        try:
            if not isinstance(data, dict):
                raise DocumentError("Expected an editor request")
            if path == "/api/session":
                session_id = data.get("session")
                if session_id not in self.sessions:
                    session_id = secrets.token_urlsafe(24)
                    self.sessions[session_id] = Session(
                        self.document, self.name, self.reference
                    )
                session = self.sessions[session_id]
                with session.lock:
                    result = session.state()
                    result["session"] = session_id
                return 200, result
            session = self.sessions.get(session_id or "")
            if session is None:
                return 401, {"error": "Editor session expired; reload this page"}
            with session.lock:
                if path == "/api/action":
                    result = session.action(data)
                elif path == "/api/operation":
                    result = session.operation(data)
                elif path == "/api/holes":
                    result = session.holes(data)
                elif path == "/api/nodes":
                    session.check_revision(data)
                    result = session.nodes(data["object"])
                elif path == "/api/settings":
                    if {"api_keys", "local", "models"} & data.keys():
                        keys.save(
                            data.get("api_keys"), data.get("local"), data.get("models")
                        )
                    result = keys.summary()
                elif path == "/api/reference":
                    result = {"reference": session.reference}
                elif path == "/api/export":
                    session.check_revision(data)
                    result = {
                        "content": session.project()
                        if data.get("project")
                        else session.state()["svg"]
                    }
                else:
                    return 404, {"error": "Not found"}
            return 200, result
        except StaleRevisionError as exc:
            return 409, {"error": str(exc)}
        except (
            DocumentError,
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            AttributeError,
            OSError,
        ) as exc:
            return 400, {"error": str(exc)}


class EditorServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        initial: str | None = None,
        name: str = "Untitled.svg",
        reference: dict | None = None,
        backend: Backend | None = None,
    ):
        super().__init__(address, Handler)
        self.backend = backend or Backend(initial, name, reference)


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
        if not self.same_origin():
            self.json(
                403, {"error": "This editor accepts local same-origin requests only"}
            )
            return
        path = urlparse(self.path).path
        files = {
            "/": "index.html",
            "/app.js": "app.js",
            "/style.css": "style.css",
            "/favicon.svg": "favicon.svg",
        }
        if path not in files:
            self.json(404, {"error": "Not found"})
            return
        file = STATIC / files[path]
        self.respond(
            200, file.read_bytes(), mimetypes.guess_type(file.name)[0] or "text/plain"
        )

    def do_POST(self) -> None:
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
    parser.add_argument("svg", nargs="?", type=Path, help="SVG to open initially")
    parser.add_argument("--reference", type=Path, help="PNG/JPEG/WebP reference image")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Serve the editor to a browser instead of opening a desktop window",
    )
    parser.add_argument("--port", type=int, default=8765, help="Port for --serve")
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
        args.svg.read_text() if args.svg else None,
        args.svg.name if args.svg else "Untitled.svg",
        reference,
    )
    if not args.serve:
        from vectrify.ui import desktop

        if desktop.available():
            desktop.run(backend)
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
