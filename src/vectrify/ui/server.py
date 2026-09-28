"""Loopback-only local editor server. No frontend build step is required."""

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
from vectrify.ui.session import MAX_SOURCE, Session

STATIC = Path(__file__).with_name("static")


class EditorServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        initial: str | None = None,
        name: str = "Mountain study.svg",
        reference: dict | None = None,
    ):
        super().__init__(address, Handler)
        self.document = import_svg(initial or (STATIC / "sample.svg").read_text())
        self.name = name
        self.reference = reference
        self.sessions: dict[str, Session] = {}


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
            if not isinstance(data, dict):
                raise DocumentError("Expected an editor request")
            if self.path == "/api/session":
                session_id = data.get("session")
                if session_id not in self.server.sessions:
                    session_id = secrets.token_urlsafe(24)
                    self.server.sessions[session_id] = Session(
                        self.server.document, self.server.name, self.server.reference
                    )
                session = self.server.sessions[session_id]
                with session.lock:
                    result = session.state()
                    result["session"] = session_id
                self.json(200, result)
                return
            session = self.server.sessions.get(self.headers.get("X-Vectrify-Session"))
            if session is None:
                self.json(401, {"error": "Editor session expired; reload this page"})
                return
            with session.lock:
                if self.path == "/api/action":
                    result = session.action(data)
                elif self.path == "/api/contact":
                    result = session.contact(data)
                elif self.path == "/api/simplify":
                    result = session.simplify(data)
                elif self.path == "/api/improve":
                    result = session.improve(data)
                elif self.path == "/api/holes":
                    result = session.holes(data)
                elif self.path == "/api/nodes":
                    session.check_revision(data)
                    result = session.nodes(data["object"])
                elif self.path == "/api/reference":
                    result = {"reference": session.reference}
                elif self.path == "/api/export":
                    session.check_revision(data)
                    result = {
                        "content": session.project()
                        if data.get("project")
                        else session.state()["svg"]
                    }
                else:
                    self.json(404, {"error": "Not found"})
                    return
            self.json(200, result)
        except StaleRevisionError as exc:
            self.json(409, {"error": str(exc)})
        except (
            DocumentError,
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            AttributeError,
            OSError,
        ) as exc:
            self.json(400, {"error": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the Vectrify local SVG editor")
    parser.add_argument("svg", nargs="?", type=Path, help="SVG to open initially")
    parser.add_argument("--reference", type=Path, help="PNG/JPEG/WebP reference image")
    parser.add_argument("--port", type=int, default=8765)
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
    server = EditorServer(
        ("127.0.0.1", args.port),
        args.svg.read_text() if args.svg else None,
        args.svg.name if args.svg else "Mountain study.svg",
        reference,
    )
    print(f"Vectrify editor: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
