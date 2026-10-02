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
# A region: [x, y, width, height] in document units.
Region = list[float]
# What the person's window shows, as view() reports it.
View = Literal["view"]
# A region, or a polygon [[x, y], ...] around it.
Area = list[float] | list[list[float]]
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

    # Looking -----------------------------------------------------------

    @look
    def describe(
        page: int = 0,
        page_size: int = 100,
        within: str | None = None,
        region: Region | None = None,
    ) -> CallToolResult:
        """The drawing: artboard, reference, selection and objects.

        Objects (id, label, tag, parent, paint, bounds [x, y, w, h], locks)
        come a page at a time, in document order (later is in front);
        within lists one group's contents. With region [x, y, w, h], only the
        objects that paint inside it (not just their bounds), front to back,
        each path with the contours of it that do.
        """
        reply = state.call(
            "describe",
            {"page": page, "page_size": page_size, "within": within, "region": region},
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
        region: Region | View | None = None,
        overlay: Literal["none", "side", "over", "reference"] = "none",
        max_side: int | None = None,
        grid: bool = False,
    ) -> CallToolResult:
        """A PNG of the drawing, optionally of region [x, y, w, h] only, or
        region="view": exactly what the person's window shows.

        overlay="side" puts the reference beside it, "over" blends the two,
        "reference" shows the reference alone. max_side caps the longer side
        in pixels (default 1024, at most 2048). grid=true draws lines
        labelled with document coordinates. The text says how pixels map to
        document coordinates.
        """
        return state.call(
            "render",
            {"region": region, "overlay": overlay, "max_side": max_side, "grid": grid},
        )

    @look
    def compare(
        region: Region | View | None = None,
        max_side: int | None = None,
        grid: bool = False,
    ) -> CallToolResult:
        """How far the drawing is from the reference, over region or all.

        Gives the mean squared error (0 is identical), the worst cells of a
        4 x 4 grid as regions to look at next, and a heat map (black agrees).
        """
        return state.call(
            "compare", {"region": region, "max_side": max_side, "grid": grid}
        )

    @look
    def pick(x: float, y: float, radius: float = 0) -> CallToolResult:
        """What paints at document point (x, y), or within radius of it,
        front to back: each object with its groups and, for a path, the
        contours (index, first node, bounds) whose own stroke or fill is
        there. Stroke widths and transforms count; bounds alone do not.
        colour gives the drawing's and the reference's mean colour there and
        their difference (0 to 1)."""
        return state.call("pick", {"x": x, "y": y, "radius": radius})

    @look
    def trace_reference(
        region: Region | View | None = None,
        colour: str | None = None,
        dark: bool = True,
        tolerance: float | None = None,
        min_area: float | None = None,
    ) -> CallToolResult:
        """Outlines of the reference's dark areas in region (luminance at
        most tolerance, default 0.35), or of the areas near colour (RGB
        distance 0-1, default 0.12), as closed path data in document
        coordinates, largest first, holes included. Areas under min_area
        square units are left out. Adjust the drawing to these instead of
        reading coordinates off an image."""
        return state.call(
            "trace_reference",
            {
                "region": region,
                "colour": colour,
                "dark": dark,
                "tolerance": tolerance,
                "min_area": min_area,
            },
        )

    @look
    def view() -> CallToolResult:
        """What the person is looking at: their selection (objects and
        points) and, in the editor window, the visible region [x, y, w, h],
        the zoom (screen pixels per unit), the active tool, the entered
        group and the reference view. Read-only: you never change them.
        render(region="view") renders the same."""
        return state.call("view", {})

    @look
    def get_svg(ids: list[str] | None = None) -> CallToolResult:
        """The SVG of the whole drawing, or of the objects ids."""
        return state.call("get_svg", {"ids": ids})

    @look
    def points(
        id: str,  # noqa: A002
        region: Region | None = None,
        contours: list[int] | None = None,
        coords: Literal["document", "local", "both"] = "document",
        nodes: bool = True,
        page: int = 0,
        page_size: int = 300,
    ) -> CallToolResult:
        """A path's contours (index, id, first node, count, closed, bounds,
        hole) with their nodes (id, i, command, values, pinned), a page at a
        time.

        A contour with hole=true is a hole of a filled path, with its area;
        holes() takes [path id, contour id]. region [x, y, w, h] keeps the
        contours crossing it and the nodes inside it; contours keeps those
        indices; nodes=false lists contours only. Values are [x, y] for M and
        L, [c1x, c1y, c2x, c2y, x, y] for C, in document coordinates
        (coords="local": the path's own, "both": both). "more" says when
        there is another page.
        """
        return state.call(
            "points",
            {
                "id": id,
                "region": region,
                "contours": contours,
                "coords": coords,
                "nodes": nodes,
                "page": page,
                "page_size": page_size,
            },
        )

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
    def properties(
        ids: Ids,
        fill: str | None = None,
        stroke: str | None = None,
        stroke_width: float | None = None,
        opacity: float | None = None,
        fill_opacity: float | None = None,
        stroke_opacity: float | None = None,
        name: str | None = None,
        locks: list[str] | None = None,
    ) -> CallToolResult:
        """Set the objects' paint (colours as CSS colours or "none"), name
        (one object; empty clears it) and locks: edit kinds (geometry, paint,
        structure, transform) or properties (fill, stroke, d, ...); an empty
        list unlocks. One undo step."""
        return state.call(
            "properties",
            {
                "ids": ids,
                "fill": fill,
                "stroke": stroke,
                "stroke_width": stroke_width,
                "opacity": opacity,
                "fill_opacity": fill_opacity,
                "stroke_opacity": stroke_opacity,
                "name": name,
                "locks": locks,
            },
        )

    @tool(structured_output=False)
    def transform(
        ids: Ids,
        dx: float | None = None,
        dy: float | None = None,
        scale: list[float] | None = None,
        anchor: str | list[float] = "center",
        box: Region | None = None,
    ) -> CallToolResult:
        """Move objects by dx, dy document units and/or scale them by scale
        [sx, sy] about anchor (a point, or center, top-left, top-right,
        bottom-left, bottom-right of their bounds); or fit their painted
        bounds to box [x, y, w, h]."""
        return state.call(
            "transform",
            {
                "ids": ids,
                "dx": dx,
                "dy": dy,
                "scale": scale,
                "anchor": anchor,
                "box": box,
            },
        )

    @tool(structured_output=False)
    def arrange(
        ids: Ids,
        to: Literal["front", "back", "forward", "backward"] | None = None,
        parent: str | None = None,
        index: int | None = None,
    ) -> CallToolResult:
        """Restack objects within their group (to; forward and backward move
        one step), or move them into group parent (the root id for the top
        level) at index among its children (0 is the back; default the
        front)."""
        return state.call(
            "arrange", {"ids": ids, "to": to, "parent": parent, "index": index}
        )

    @tool(structured_output=False)
    def group(ids: Ids) -> CallToolResult:
        """Group two or more objects."""
        return state.call("group", {"ids": ids})

    @tool(structured_output=False)
    def ungroup(ids: Ids) -> CallToolResult:
        """Ungroup groups, keeping their children."""
        return state.call("ungroup", {"ids": ids})

    @tool(structured_output=False)
    def join(
        ids: list[str] | None = None,
        points: Points | None = None,
        reach: float | None = None,
        bridge: Literal["curve", "line"] | None = None,
        color_source: str | None = None,
    ) -> CallToolResult:
        """Join, as the editor's Join does: two points ([object id, node id]
        each) join each other, cutting a contour open first where one is not
        a free end; stroked lines (ids) join their ends lying within reach
        (default 4) of each other, bridged by a curve or a line; filled paths
        (ids) merge into one outline, its colour mixed by area or taken from
        the path color_source. "joined" says which."""
        return state.call(
            "join",
            {
                "ids": ids,
                "points": points,
                "reach": reach,
                "bridge": bridge,
                "color_source": color_source,
            },
        )

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
    def holes(
        contours: Points,
        action: Literal["fill", "shape"] = "fill",
        delete_enclosed: bool = False,
    ) -> CallToolResult:
        """Fill holes (action="fill"), or turn them into shapes of their own
        ("shape"); contours are [path id, contour id] pairs of the contours
        points() marks hole=true. delete_enclosed also deletes the shapes
        lying wholly inside the filled holes."""
        return state.call(
            "holes",
            {
                "contours": contours,
                "action": action,
                "delete_enclosed": delete_enclosed,
            },
        )

    @tool(structured_output=False)
    def convert(
        ids: Ids,
        to: Literal["line", "fill", "either", "path"] = "either",
    ) -> CallToolResult:
        """Turn thin filled shapes into centre lines (to="line"), stroked
        lines into filled outlines (to="fill"), or each into the other; or
        give an instance (use) or a path sharing its geometry an editable
        path of its own (to="path")."""
        return state.call("convert", {"ids": ids, "to": to})

    @tool(structured_output=False)
    def delete(
        ids: list[str] | None = None,
        points: Points | None = None,
        region: Area | None = None,
        contours: bool = False,
        cut: bool = False,
    ) -> CallToolResult:
        """Delete objects (ids); points ([object id, node id] pairs; a
        contour or path left too small goes too), or with contours=true the
        whole contours they are on; or the contours lying inside region
        ([x, y, w, h] or a polygon), of the paths ids or every unlocked path
        painting there (cut=true also cuts away the parts of contours
        crossing into it). One undo step."""
        return state.call(
            "delete",
            {
                "ids": ids,
                "points": points,
                "region": region,
                "contours": contours,
                "cut": cut,
            },
        )

    @tool(structured_output=False)
    def extract(
        region: Area, ids: list[str] | None = None, cut: bool = True
    ) -> CallToolResult:
        """Take the contours lying inside region ([x, y, w, h] or a polygon
        [[x, y], ...], document coordinates) out of their paths, each path's
        into a new path with its paint, just above it in its group.

        The paths ids, or every unlocked path painting there. cut=true cuts
        strokes crossing the region's edge there (and splits fills along it);
        cut=false takes only whole contours. One undo step.
        """
        return state.call("extract", {"region": region, "ids": ids, "cut": cut})

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
        """Draw a new path from SVG path data in document coordinates (close
        it with Z for a shape; more subpaths are holes or parts).

        It goes into the group of what is drawn under it, just above that
        (placed says where and why), or into parent at index; its id is in
        created.
        """
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
    def set_points(
        changes: dict[str, dict[str, list[float]]],
        coords: Literal["document", "local"] = "document",
    ) -> CallToolResult:
        """Set node values: {object id: {node id: values}}, values as
        points() lists them, in document coordinates unless coords="local"
        (moving a node, move its handles with it)."""
        return state.call("set_points", {"changes": changes, "coords": coords})

    @tool(structured_output=False)
    def point_style(
        points: Points,
        handles: Literal[0, 1, 2] | None = None,
        pinned: bool | None = None,
    ) -> CallToolResult:
        """Give points 0, 1 or 2 curve handles, and pin them (so no edit or
        operation moves them) or unpin them."""
        return state.call(
            "point_style", {"points": points, "handles": handles, "pinned": pinned}
        )

    @tool(structured_output=False)
    def break_points(points: Points) -> CallToolResult:
        """Cut lines or closed contours open at points. Given the two points
        at the ends of a segment, delete that segment instead, as the
        editor's Break does."""
        return state.call("break_points", {"points": points})

    @tool(structured_output=False)
    def split_edge(points: Points) -> CallToolResult:
        """Add a point halfway along the edge leading into each point."""
        return state.call("split_edge", {"points": points})

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

    @tool(structured_output=False)
    def job(
        id: str,  # noqa: A002
        action: Literal["status", "apply", "discard", "stop"] = "status",
        wait_seconds: float = 30,
        choice: int = 0,
    ) -> CallToolResult:
        """A job's progress (action="status"): waits up to wait_seconds (at
        most 120) for it to finish, then gives its metrics and before/after
        previews. "apply" keeps a finished job's result (or alternative
        choice) as one undo step, "discard" drops the job, "stop" stops it
        early keeping its best result so far."""
        args: dict[str, Any] = {"id": id, "action": action}
        if action == "status":
            args["wait_seconds"] = wait_seconds
        elif action == "apply":
            args["choice"] = choice
        return state.call("job", args)


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
