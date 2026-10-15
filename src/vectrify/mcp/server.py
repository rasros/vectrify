"""``vectrify-mcp``: a stdio MCP server for looking at and editing a drawing.

Every tool is a thin wrapper over one ``vectrify.ui.agent.Agent`` call, made
from that call's signature and docstring, so a file opened here and the
drawing in a running editor window answer alike.
``register_tools`` adds them to a server against one ``Vectrify`` state: the
stdio server's (``build_server``, which can also open files and connect), or
the one the editor hosts over HTTP on its window (``build_window_server``).
The server remembers the revision each client last saw (per Streamable HTTP
session; a stdio server has one client) and sends it with each edit; the
session refuses an edit of a drawing that changed since.
"""

from __future__ import annotations

import argparse
import base64
import inspect
import json
import mimetypes
from collections import OrderedDict
from collections.abc import Hashable, Sequence
from contextvars import ContextVar
from pathlib import Path
from typing import Annotated, Any, get_type_hints

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, ImageContent, TextContent, ToolAnnotations
from pydantic import Field

from vectrify.mcp.guide import GUIDE, INSTRUCTIONS, WINDOW_INSTRUCTIONS
from vectrify.mcp.target import (
    FileTarget,
    LiveTarget,
    TargetError,
    WindowTarget,
    read_discovery,
)
from vectrify.project_file import encode_project
from vectrify.ui.agent import CALLS, LOOKS

Target = FileTarget | LiveTarget | WindowTarget

# The agent's argument types as the schemas constrain them: an edit's
# targets and points are never empty; taper controls match the UI's bounds.
_Point = Annotated[list[str], Field(min_length=2, max_length=2)]
CONSTRAINED: dict[str, Any] = {
    "Ids": Annotated[list[str], Field(min_length=1)],
    "Point": _Point,
    "Points": Annotated[list[_Point], Field(min_length=1)],
    "TaperLength": Annotated[float, Field(ge=0, allow_inf_nan=False)],
    "TaperWidth": Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)],
}
# Agent calls no tool offers: save, load_reference and attaching send them.
UNOFFERED = frozenset({"hello", "export", "set_reference"})
# Agent call arguments a tool leaves out.
HIDDEN = {"pick": {"limit"}}
LOOK = ToolAnnotations(read_only_hint=True)
NO_TARGET = (
    "No drawing yet. open(path) a .svg or .vectrify file, or turn on 'Agents' "
    "in the running editor's footer and call connect()."
)


# Which client the current call comes from: its Streamable HTTP session, or
# for a sessionless request the client it names. Each remembers what it saw.
CLIENT: ContextVar[Hashable] = ContextVar("vectrify_client", default=None)
# How many clients' last looks are remembered.
CLIENTS = 64


def client_key(ctx: Any) -> Hashable:
    """The client a request comes from, as far as the transport tells."""
    request = getattr(ctx, "request", None)
    headers = getattr(request, "headers", None)
    session = headers.get("mcp-session-id") if headers is not None else None
    if session:
        return ("session", session)
    params = getattr(getattr(ctx, "session", None), "client_params", None)
    info = getattr(params, "client_info", None)
    if info is not None:
        return ("client", info.name, info.version)
    # stdio: the process is the client.
    return None


async def per_client(ctx: Any, call_next: Any) -> Any:
    """Middleware: run the request as its client's."""
    token = CLIENT.set(client_key(ctx))
    try:
        return await call_next(ctx)
    finally:
        CLIENT.reset(token)


class Vectrify:
    """The server's one target and, for each client, the revision it last saw
    there: one client's look is not another's."""

    def __init__(self, target: Target | None = None) -> None:
        self.target: Target | None = target
        # client -> (what it looked at, [epoch, revision]).
        self._seen: OrderedDict[Hashable, tuple[object, list]] = OrderedDict()

    @property
    def seen(self) -> list | None:
        """What the current client last saw of the current target."""
        entry = self._seen.get(CLIENT.get())
        target = self.target
        if entry is None or target is None or entry[0] is not target.identity():
            return None
        return entry[1]

    def _saw(self, where: dict[str, Any]) -> None:
        assert self.target is not None
        key = CLIENT.get()
        self._seen[key] = (self.target.identity(), [where["epoch"], where["revision"]])
        self._seen.move_to_end(key)
        while len(self._seen) > CLIENTS:
            self._seen.popitem(last=False)

    def require(self) -> Target:
        if self.target is None:
            # A running editor that allows agents is the default target.
            found = read_discovery()
            if found is None:
                raise ToolError(NO_TARGET)
            try:
                self.attach(LiveTarget(found["url"], found["token"]))
            except TargetError as exc:
                raise ToolError(f"{NO_TARGET} ({exc})") from None
        assert self.target is not None
        return self.target

    def attach(self, target: Target) -> dict[str, Any]:
        reply = target.call("hello", {})
        self.target = target
        # What anyone saw was of the target before.
        self._seen.clear()
        self._saw(reply.data)
        return reply.data

    def call(self, tool: str, args: dict[str, Any] | None = None) -> CallToolResult:
        target = self.require()
        # Unset arguments take the tool's defaults.
        payload = {k: v for k, v in (args or {}).items() if v is not None}
        payload["seen"] = self.seen
        try:
            reply = target.call(tool, payload)
        except TargetError as exc:
            if exc.where:
                self._saw(exc.where)
            raise ToolError(str(exc)) from None
        data = reply.data
        if "epoch" in data and "revision" in data:
            self._saw(data)
        return result(data, reply.images)


def result(
    data: dict[str, Any], images: Sequence[tuple[str, bytes]] = ()
) -> CallToolResult:
    content: list[Any] = [
        TextContent(type="text", text=json.dumps(data, separators=(",", ":")))
    ]
    for name, png in images:
        content.append(TextContent(type="text", text=f"Image: {name}"))
        content.append(
            ImageContent(
                type="image",
                data=base64.b64encode(png).decode(),
                mime_type="image/png",
            )
        )
    return CallToolResult(content=content)


def build_server(state: Vectrify | None = None) -> MCPServer:
    """The stdio server: a file it opens, or a running editor's window."""
    state = state or Vectrify()
    server = MCPServer("vectrify", instructions=INSTRUCTIONS, middleware=[per_client])
    tool = server.tool

    @tool(structured_output=False)
    def open(path: str) -> CallToolResult:  # noqa: A001
        """Open an .svg or .vectrify file to edit headlessly (no window).

        Replaces the current target; the file is written only by save().
        """
        file = Path(path).expanduser()
        try:
            state.attach(FileTarget(file))
        except (OSError, UnicodeDecodeError) as exc:
            raise ToolError(f"Could not read {file}: {exc}") from None
        except TargetError as exc:
            raise ToolError(str(exc)) from None
        except Exception as exc:
            raise ToolError(f"Could not open {file}: {exc}") from None
        return state.call("describe", {"page_size": 50})

    @tool(structured_output=False)
    def connect(url: str | None = None, token: str | None = None) -> CallToolResult:
        """Attach to the running editor window, which shows each edit live.

        The person allows it with 'Agents' in the editor's footer; url and
        token default to what that wrote to ~/.local/state/vectrify/editor.json.
        """
        if url is None or token is None:
            found = read_discovery()
            if found is None:
                raise ToolError(
                    "No editor allows agents. Ask the person to turn on "
                    "'Agents' in the editor's footer."
                )
            url, token = url or found["url"], token or found["token"]
        try:
            state.attach(LiveTarget(url, token))
        except TargetError as exc:
            raise ToolError(str(exc)) from None
        return state.call("describe", {"page_size": 50})

    register_tools(server, state)
    return server


def build_window_server(state: Vectrify) -> MCPServer:
    """The server the editor hosts, on the window that allows agents."""
    # Quiet: it runs inside the editor, whose terminal is the person's.
    server = MCPServer(
        "vectrify",
        instructions=WINDOW_INSTRUCTIONS,
        log_level="WARNING",
        middleware=[per_client],
    )
    register_tools(server, state)
    return server


def register_tools(server: MCPServer, state: Vectrify) -> None:
    """Add the guide and every drawing tool to *server*, acting on *state*."""
    tool = server.tool

    @server.resource("vectrify://guide", mime_type="text/markdown")
    def guide() -> str:
        """How to work on a drawing: the look, edit, look again loop."""
        return GUIDE

    # Files -------------------------------------------------------------

    def write(path: Path, project: bool) -> CallToolResult:
        reply = state.call("export", {"project": project})
        text = reply.content[0]
        assert isinstance(text, TextContent)
        data = json.loads(text.text)
        try:
            if project:
                path.write_bytes(encode_project(data["content"]))
            else:
                path.write_text(data["content"], encoding="utf-8")
        except OSError as exc:
            raise ToolError(f"Could not write {path}: {exc}") from None
        return result({"saved": str(path), "revision": data["revision"]})

    @tool(structured_output=False)
    def save(path: str | None = None) -> CallToolResult:
        """Save the drawing: to the opened file, or to path.

        A .vectrify path writes a project (locks, pins, the reference and the
        selection kept); a .svg path, or any other, plain SVG. The editor
        window needs a path.
        """
        target = state.require()
        if path is None:
            if not isinstance(target, FileTarget):
                raise ToolError("Give a path to save the editor window's drawing to")
            file = target.path
        else:
            file = Path(path).expanduser()
        saved = write(file, file.suffix.lower() == ".vectrify")
        if isinstance(target, FileTarget):
            target.path = file
        return saved

    @tool(structured_output=False)
    def load_reference(path: str | None = None) -> CallToolResult:
        """Load a PNG, JPEG or WebP as the reference image the drawing traces,
        stretched over the artboard as the editor shows it; without a path,
        remove the reference."""
        if path is None:
            return state.call("set_reference", {})
        file = Path(path).expanduser()
        mime = mimetypes.guess_type(file.name)[0]
        if mime not in {"image/png", "image/jpeg", "image/webp"}:
            raise ToolError("Choose a PNG, JPEG or WebP reference")
        try:
            data = file.read_bytes()
        except OSError as exc:
            raise ToolError(f"Could not read {file}: {exc}") from None
        return state.call(
            "set_reference",
            {
                "name": file.name,
                "data_url": f"data:{mime};base64," + base64.b64encode(data).decode(),
            },
        )

    for name, call in CALLS.items():
        if name not in UNOFFERED:
            offer(server, state, name, call)


def offer(server: MCPServer, state: Vectrify, name: str, call: Any) -> None:
    """Add the agent call *call* to *server* as the tool *name*: its
    signature, less self and seen, and its docstring."""
    hints = get_type_hints(call, localns=CONSTRAINED, include_extras=True)
    parameters = [
        p.replace(annotation=hints[p.name])
        for p in list(inspect.signature(call).parameters.values())[2:]
        if p.name not in HIDDEN.get(name, ())
    ]

    def tool(**args: Any) -> CallToolResult:
        reply = state.call(name, args)
        if name != "describe":
            return reply
        text = reply.content[0]
        assert isinstance(text, TextContent)
        data = json.loads(text.text)
        target = state.target
        assert target is not None
        data["target"] = target.describe()
        return result(data)

    tool.__name__ = tool.__qualname__ = name
    tool.__doc__ = call.__doc__
    signature = inspect.Signature(parameters, return_annotation=CallToolResult)
    vars(tool)["__signature__"] = signature
    annotations = LOOK if name in LOOKS else None
    server.tool(annotations=annotations, structured_output=False)(tool)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MCP server through which an agent edits Vectrify drawings"
    )
    parser.add_argument(
        "file", nargs="?", type=Path, help="An .svg or .vectrify file to open"
    )
    args = parser.parse_args()
    state = Vectrify()
    if args.file is not None:
        state.attach(FileTarget(args.file.expanduser()))
    build_server(state).run()
