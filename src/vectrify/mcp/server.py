"""``vectrify-mcp``: a stdio MCP server for looking at and editing a drawing.

Every tool is a thin wrapper over one ``vectrify.ui.agent.Agent`` call, so a
file opened here and the drawing in a running editor window answer alike.
``register_tools`` adds them to a server against one ``Vectrify`` state: the
stdio server's (``build_server``, which can also open files and connect), or
the one the editor hosts over HTTP on its window (``build_window_server``).
The server remembers the revision the agent last saw and sends it with each
edit; the session refuses an edit of a drawing that changed since.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, Literal

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

Target = FileTarget | LiveTarget | WindowTarget

# An edit's targets, always given: no tool acts on the current selection.
Ids = Annotated[list[str], Field(min_length=1)]
Point = Annotated[list[str], Field(min_length=2, max_length=2)]
Points = Annotated[list[Point], Field(min_length=1)]
Region = list[float]
LOOK = ToolAnnotations(read_only_hint=True)
NO_TARGET = (
    "No drawing yet. open(path) a .svg or .vectrify file, or turn on 'Agents' "
    "in the running editor's footer and call connect()."
)


class Vectrify:
    """The server's one target and the revision the agent last saw there."""

    def __init__(self, target: Target | None = None) -> None:
        self.target: Target | None = target
        self.seen: list | None = None

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
        self.seen = [reply.data["epoch"], reply.data["revision"]]
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
                self.seen = [exc.where["epoch"], exc.where["revision"]]
            raise ToolError(str(exc)) from None
        data = reply.data
        if "epoch" in data and "revision" in data:
            self.seen = [data["epoch"], data["revision"]]
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
    server = MCPServer("vectrify", instructions=INSTRUCTIONS)
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
        "vectrify", instructions=WINDOW_INSTRUCTIONS, log_level="WARNING"
    )
    register_tools(server, state)
    return server


def register_tools(server: MCPServer, state: Vectrify) -> None:
    """Add the guide and every drawing tool to *server*, acting on *state*."""
    tool = server.tool
    look = server.tool(annotations=LOOK, structured_output=False)

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
            path.write_text(data["content"], encoding="utf-8")
        except OSError as exc:
            raise ToolError(f"Could not write {path}: {exc}") from None
        return result({"saved": str(path), "revision": data["revision"]})

    @tool(structured_output=False)
    def save(path: str | None = None) -> CallToolResult:
        """Save the drawing: to the opened file, or to path.

        A .vectrify path keeps locks, pins, the reference and the selection;
        any other is written as plain SVG. The editor window needs a path.
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
    def export_svg(path: str) -> CallToolResult:
        """Write the drawing as plain SVG to path."""
        state.require()
        return write(Path(path).expanduser(), False)

    @tool(structured_output=False)
    def load_reference(path: str) -> CallToolResult:
        """Load a PNG, JPEG or WebP as the reference image the drawing traces.

        It is stretched over the artboard, as the editor shows it.
        """
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

    @tool(structured_output=False)
    def remove_reference() -> CallToolResult:
        """Remove the reference image."""
        return state.call("set_reference", {})

    # Looking -----------------------------------------------------------

    @look
    def describe(
        page: int = 0, page_size: int = 100, within: str | None = None
    ) -> CallToolResult:
        """The drawing: artboard, reference, selection and objects.

        Objects (id, label, tag, parent, paint, bounds [x, y, w, h], locks)
        come a page at a time, in document order (later is in front);
        within lists one group's contents.
        """
        reply = state.call(
            "describe", {"page": page, "page_size": page_size, "within": within}
        )
        target = state.target
        assert target is not None
        text = reply.content[0]
        assert isinstance(text, TextContent)
        data = json.loads(text.text)
        data["target"] = target.describe()
        return result(data)

    @look
    def render(
        region: Region | None = None,
        overlay: Literal["none", "side", "over"] = "none",
        max_side: int | None = None,
    ) -> CallToolResult:
        """A PNG of the drawing, optionally of region [x, y, w, h] only.

        overlay="side" puts the reference beside it, "over" blends the two.
        max_side caps the longer side in pixels (default 1024, at most 2048).
        """
        return state.call(
            "render", {"region": region, "overlay": overlay, "max_side": max_side}
        )

    @look
    def reference(
        region: Region | None = None, max_side: int | None = None
    ) -> CallToolResult:
        """A PNG of the reference image, optionally of region [x, y, w, h]."""
        return state.call("reference", {"region": region, "max_side": max_side})

    @look
    def compare(
        region: Region | None = None, max_side: int | None = None
    ) -> CallToolResult:
        """How far the drawing is from the reference, over region or all.

        Gives the mean squared error (0 is identical), the worst cells of a
        4 x 4 grid as regions to look at next, and a heat map (black agrees).
        """
        return state.call("compare", {"region": region, "max_side": max_side})

    @look
    def get_svg(ids: list[str] | None = None) -> CallToolResult:
        """The SVG of the whole drawing, or of the objects ids."""
        return state.call("get_svg", {"ids": ids})

    @look
    def points(id: str) -> CallToolResult:  # noqa: A002
        """A path's contours and nodes: ids, commands, values and pins.

        Values are [x, y] for M and L nodes and [c1x, c1y, c2x, c2y, x, y]
        for C, in the path's own coordinates.
        """
        return state.call("points", {"id": id})

    @look
    def holes(id: str) -> CallToolResult:  # noqa: A002
        """The holes of a path: ids, areas and bounds."""
        return state.call("holes", {"id": id})

    # History -----------------------------------------------------------

    @look
    def history(limit: int = 30) -> CallToolResult:
        """The undo and redo stacks, newest first: label, author, revision.

        author is "agent" for edits made through this server, else "person".
        """
        return state.call("history", {"limit": limit})

    @tool(structured_output=False)
    def undo(steps: int = 1) -> CallToolResult:
        """Undo the last steps, whoever made them; check history() first."""
        return state.call("undo", {"steps": steps})

    @tool(structured_output=False)
    def redo(steps: int = 1) -> CallToolResult:
        """Redo steps undone."""
        return state.call("redo", {"steps": steps})

    # Objects -----------------------------------------------------------

    @tool(structured_output=False)
    def paint(
        ids: Ids,
        fill: str | None = None,
        stroke: str | None = None,
        stroke_width: float | None = None,
        opacity: float | None = None,
        fill_opacity: float | None = None,
        stroke_opacity: float | None = None,
    ) -> CallToolResult:
        """Set the paint of the objects ids: colours as CSS colours or
        "none"."""
        return state.call(
            "paint",
            {
                "ids": ids,
                "fill": fill,
                "stroke": stroke,
                "stroke_width": stroke_width,
                "opacity": opacity,
                "fill_opacity": fill_opacity,
                "stroke_opacity": stroke_opacity,
            },
        )

    @tool(structured_output=False)
    def rename(id: str, name: str) -> CallToolResult:  # noqa: A002
        """Name an object; an empty name clears it."""
        return state.call("rename", {"id": id, "name": name})

    @tool(structured_output=False)
    def locks(id: str, locks: list[str]) -> CallToolResult:  # noqa: A002
        """Set an object's locks: edit kinds (geometry, paint, structure,
        transform) or properties (fill, stroke, d, ...). Empty unlocks."""
        return state.call("locks", {"id": id, "locks": locks})

    @tool(structured_output=False)
    def move(ids: Ids, dx: float, dy: float) -> CallToolResult:
        """Move the objects ids by dx, dy document units."""
        return state.call("move", {"dx": dx, "dy": dy, "ids": ids})

    @tool(structured_output=False)
    def resize(
        ids: Ids,
        scale: list[float] | None = None,
        anchor: str | list[float] = "center",
        box: Region | None = None,
    ) -> CallToolResult:
        """Scale objects by scale [sx, sy] about anchor (a point, or center,
        top-left, top-right, bottom-left, bottom-right of their bounds), or
        fit their painted bounds to box [x, y, w, h]."""
        return state.call(
            "resize", {"ids": ids, "scale": scale, "anchor": anchor, "box": box}
        )

    @tool(structured_output=False)
    def reorder(
        ids: Ids, to: Literal["front", "back", "forward", "backward"]
    ) -> CallToolResult:
        """Restack objects within their group; forward and backward move
        one object one step."""
        return state.call("reorder", {"to": to, "ids": ids})

    @tool(structured_output=False)
    def move_into(ids: Ids, parent: str, index: int) -> CallToolResult:
        """Move objects into group parent (the root id for the top level)
        at index among its children (0 is the back)."""
        return state.call("move_into", {"ids": ids, "parent": parent, "index": index})

    @tool(structured_output=False)
    def group(ids: Ids) -> CallToolResult:
        """Group two or more objects."""
        return state.call("group", {"ids": ids})

    @tool(structured_output=False)
    def ungroup(ids: Ids) -> CallToolResult:
        """Ungroup groups, keeping their children."""
        return state.call("ungroup", {"ids": ids})

    @tool(structured_output=False)
    def join(ids: Ids, color_source: str | None = None) -> CallToolResult:
        """Merge paths into one outline, its colour mixed by area or taken
        from the path color_source."""
        return state.call("join", {"ids": ids, "color_source": color_source})

    @tool(structured_output=False)
    def join_ends(
        ids: Ids,
        reach: float = 4.0,
        bridge: Literal["curve", "line"] = "curve",
    ) -> CallToolResult:
        """Join the ends of stroked lines that lie within reach of each other."""
        return state.call("join_ends", {"ids": ids, "reach": reach, "bridge": bridge})

    @tool(structured_output=False)
    def join_points(a: Point, b: Point) -> CallToolResult:
        """Join two points, [object id, node id] each, cutting a contour
        open first where one is not a free end."""
        return state.call("join_points", {"a": a, "b": b})

    @tool(structured_output=False)
    def split_parts(ids: Ids) -> CallToolResult:
        """Split paths into their disconnected parts; holes stay with theirs."""
        return state.call("split_parts", {"ids": ids})

    @tool(structured_output=False)
    def cut_hole(ids: Ids) -> CallToolResult:
        """With two paths, cut the inner (or overlapping) one out of the
        other as a hole."""
        return state.call("cut_hole", {"ids": ids})

    @tool(structured_output=False)
    def fill_holes(contours: Points, delete_enclosed: bool = False) -> CallToolResult:
        """Fill holes, given as [object id, hole id] pairs from holes();
        delete_enclosed also deletes shapes lying wholly inside them."""
        return state.call(
            "fill_holes", {"contours": contours, "delete_enclosed": delete_enclosed}
        )

    @tool(structured_output=False)
    def holes_to_shapes(contours: Points) -> CallToolResult:
        """Turn holes ([object id, hole id] pairs) into shapes of their own."""
        return state.call("holes_to_shapes", {"contours": contours})

    @tool(structured_output=False)
    def detach(ids: Ids) -> CallToolResult:
        """Give an instance (use) or a path sharing its geometry an
        editable geometry of its own."""
        return state.call("detach", {"ids": ids})

    @tool(structured_output=False)
    def convert(
        ids: Ids,
        to: Literal["line", "fill", "either"] = "either",
    ) -> CallToolResult:
        """Turn thin filled shapes into centre lines (to="line"), stroked
        lines into filled outlines (to="fill"), or each into the other."""
        return state.call("convert", {"ids": ids, "to": to})

    @tool(structured_output=False)
    def delete(ids: Ids) -> CallToolResult:
        """Delete the objects ids."""
        return state.call("delete", {"ids": ids})

    @tool(structured_output=False)
    def add_path(
        d: str,
        fill: str | None = None,
        stroke: str | None = None,
        stroke_width: float | None = None,
        parent: str | None = None,
        index: int | None = None,
        name: str | None = None,
    ) -> CallToolResult:
        """Draw a new path from SVG path data with one subpath (close it with
        Z for a shape), in front, or in parent at index; its id is in created."""
        return state.call(
            "add_path",
            {
                "d": d,
                "fill": fill,
                "stroke": stroke,
                "stroke_width": stroke_width,
                "parent": parent,
                "index": index,
                "name": name,
            },
        )

    @tool(structured_output=False)
    def knife(
        start: list[float],
        end: list[float],
        ids: list[str] | None = None,
        within: str | None = None,
    ) -> CallToolResult:
        """Cut paths along the line start to end ([x, y] each): the paths
        ids, or every unlocked path it crosses (within a group if given)."""
        return state.call(
            "knife", {"start": start, "end": end, "ids": ids, "within": within}
        )

    @tool(structured_output=False)
    def redraw_outline(
        id: str,  # noqa: A002
        points: list[list[float]],
        long_way: bool = False,
        pixel: float = 1.0,
    ) -> CallToolResult:
        """Redraw the stretch of a path's outline that the stroke points
        ([x, y] each) runs along; both ends must lie on the same contour,
        within 10 pixel sizes. long_way takes the other way round a loop."""
        return state.call(
            "redraw_outline",
            {"id": id, "points": points, "long_way": long_way, "pixel": pixel},
        )

    # Points ------------------------------------------------------------

    @tool(structured_output=False)
    def set_points(changes: dict[str, dict[str, list[float]]]) -> CallToolResult:
        """Set node values: {object id: {node id: values}}, values as
        points() lists them (moving a node, move its handles with it)."""
        return state.call("set_points", {"changes": changes})

    @tool(structured_output=False)
    def handles(points: Points, count: Literal[0, 1, 2]) -> CallToolResult:
        """Give points 0, 1 or 2 curve handles."""
        return state.call("handles", {"points": points, "count": count})

    @tool(structured_output=False)
    def pin(points: Points, pinned: bool = True) -> CallToolResult:
        """Pin points so no edit or operation moves them, or unpin them."""
        return state.call("pin", {"points": points, "pinned": pinned})

    @tool(structured_output=False)
    def break_points(points: Points) -> CallToolResult:
        """Cut lines or closed contours open at points."""
        return state.call("break_points", {"points": points})

    @tool(structured_output=False)
    def delete_segment(points: Points) -> CallToolResult:
        """Delete the segment between two neighbouring points."""
        return state.call("delete_segment", {"points": points})

    @tool(structured_output=False)
    def split_edge(points: Points) -> CallToolResult:
        """Add a point halfway along the edge leading into each point."""
        return state.call("split_edge", {"points": points})

    @tool(structured_output=False)
    def delete_points(points: Points) -> CallToolResult:
        """Delete points; a contour or path left too small goes too."""
        return state.call("delete_points", {"points": points})

    @tool(structured_output=False)
    def delete_contours(points: Points) -> CallToolResult:
        """Delete the whole contours these points are on."""
        return state.call("delete_contours", {"points": points})

    # Operations --------------------------------------------------------

    @tool(structured_output=False)
    def generate(
        method: Literal["cel", "colour-regions", "samvg"] = "cel",
        settings: dict[str, Any] | None = None,
        group: str | None = None,
    ) -> CallToolResult:
        """Trace the reference into new shapes, as a job to preview.

        cel: flat colour regions with ink lines (settings regions,
        tolerance, line_width, strokes, outline); colour-regions: posterised
        regions (colours, min_pixels, tolerance, ...); samvg needs a GPU.
        Over the whole drawing, or into the area of the group group.
        """
        return state.call(
            "generate", {"method": method, "settings": settings, "group": group}
        )

    @tool(structured_output=False)
    def tidy(
        ids: Ids,
        settings: dict[str, Any] | None = None,
        rounds: int | None = None,
    ) -> CallToolResult:
        """Tidy paths as a job: snap to the reference, simplify, fit (steps
        shape, snap, detail, simplify; tolerance, seconds, ...)."""
        return state.call("tidy", {"ids": ids, "settings": settings, "rounds": rounds})

    @tool(structured_output=False)
    def fit_colours(
        ids: Ids,
        fill: Literal["flat", "linear"] = "flat",
        passes: int | None = None,
        resolution: int | None = None,
    ) -> CallToolResult:
        """Fit the fills of objects to the reference, flat or as linear
        gradients, as a job."""
        return state.call(
            "fit_colours",
            {"ids": ids, "fill": fill, "passes": passes, "resolution": resolution},
        )

    @tool(structured_output=False)
    def snap_edges(ids: Ids, tolerance: float = 1.0) -> CallToolResult:
        """Snap the touching edges of two or more paths together, as a job."""
        return state.call("snap_edges", {"ids": ids, "tolerance": tolerance})

    @tool(structured_output=False)
    def cleanup(ids: Ids) -> CallToolResult:
        """Merge duplicate paths and drop redundant points, as a job."""
        return state.call("cleanup", {"ids": ids})

    @look
    def job_status(id: str, wait_seconds: float = 30) -> CallToolResult:  # noqa: A002
        """A job's progress; waits up to wait_seconds (at most 120) for it to
        finish, then gives its metrics and before/after previews."""
        return state.call("job_status", {"id": id, "wait_seconds": wait_seconds})

    @tool(structured_output=False)
    def apply(id: str, choice: int = 0) -> CallToolResult:  # noqa: A002
        """Keep a finished job's result (or alternative choice) as one undo
        step."""
        return state.call("apply", {"id": id, "choice": choice})

    @tool(structured_output=False)
    def discard(id: str) -> CallToolResult:  # noqa: A002
        """Drop a job and its result."""
        return state.call("discard", {"id": id})

    @tool(structured_output=False)
    def stop(id: str) -> CallToolResult:  # noqa: A002
        """Stop a running job early, keeping its best result so far."""
        return state.call("stop", {"id": id})


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
