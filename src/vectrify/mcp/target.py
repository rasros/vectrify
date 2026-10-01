"""What the MCP server edits: a file it opened, or the running editor's window.

Both answer the same calls with the same ``Agent``; a file's lives in this
process, the editor's in the editor, reached over HTTP on localhost with the
token it wrote to its discovery file.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from vectrify.document import StaleRevisionError
from vectrify.ui.agent import (
    REFUSALS,
    Agent,
    RefusedError,
    Reply,
    discovery_file,
    reason,
)
from vectrify.ui.server import Backend


class TargetError(Exception):
    """A call the target refused, in words the agent can act on.

    *where* is the drawing's epoch and revision when the refusal itself made
    a revision (by taking back the steps of the edit already done).
    """

    def __init__(self, message: str, where: dict[str, Any] | None = None):
        super().__init__(message)
        self.where = where


class FileTarget:
    """A drawing opened from disk, edited headlessly in this process."""

    kind = "file"

    def __init__(self, path: Path):
        source = path.read_text(encoding="utf-8")
        self.path = path
        self.backend = Backend(name=path.name)
        status, state = self.backend.handle("/api/session", {}, None)
        assert status == 200
        self.session = self.backend.sessions[state["session"]]
        with self.session.lock:
            self.session.open(source, path.name)
        self.agent = Agent(self.session)

    def call(self, tool: str, args: dict[str, Any]) -> Reply:
        try:
            return self.agent.call(tool, args)
        except StaleRevisionError as exc:
            raise TargetError(str(exc)) from None
        except RefusedError as exc:
            raise TargetError(str(exc), exc.where) from None
        except REFUSALS as exc:
            raise TargetError(reason(exc)) from None

    def describe(self) -> str:
        return f"the file {self.path}"


def read_discovery() -> dict[str, str] | None:
    """The running editor's {url, token}, if one allows agents."""
    try:
        data = json.loads(discovery_file().read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not {"url", "token"} <= data.keys():
        return None
    return {"url": str(data["url"]), "token": str(data["token"])}


class LiveTarget:
    """The drawing in a running editor window, which shows each edit."""

    kind = "live"
    # Through no proxy: the editor is on this machine.
    _opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def __init__(self, url: str, token: str):
        self.url = url.rstrip("/")
        self.token = token

    def _send(self, path: str, body: bytes | None = None, timeout: float = 30):
        request = urllib.request.Request(
            self.url + path,
            data=body,
            method="POST" if body is not None else "GET",
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with self._opener.open(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            try:
                answer = json.loads(exc.read())
            except ValueError:
                answer = {}
            raise TargetError(
                answer.get("error") or f"The editor refused ({exc.code})",
                answer.get("where"),
            ) from None
        except (urllib.error.URLError, OSError) as exc:
            raise TargetError(
                f"The editor at {self.url} is not answering ({exc}). Is it still "
                "open with agents allowed? connect() again, or open() a file."
            ) from None

    def call(self, tool: str, args: dict[str, Any]) -> Reply:
        body = json.dumps({"tool": tool, "args": args}).encode()
        # job_status may wait up to two minutes for a job.
        answer = json.loads(self._send("/agent/call", body, timeout=180))
        images = [
            (str(name), self._send(f"/agent/image/{key}"))
            for name, key in answer.get("images", [])
        ]
        return Reply(answer["data"], images)

    def describe(self) -> str:
        return f"the editor window at {self.url}"
