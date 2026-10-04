"""What an agent may do to an editor session, and the channel it reaches it by.

``Agent`` answers an agent's calls (``describe``, ``render``, ``paint``,
``generate``…) on one ``Session``. Every edit goes through ``Session.action``
or ``Session.operation``, so selection scope, locks, pins, permissions and
revision checks hold exactly as for a person, each call is one undo step, and
the history labels it "Agent: …". Each call names its own targets and works on
a selection of its own: the person's selection is given back, cleaned of what
the call deleted, within the same locked step. The MCP server
(``vectrify.mcp``) holds an ``Agent`` in process for a file it opened, or
reaches the one of a running editor through ``AgentChannel``: HTTP on
localhost with a token, opened when the window allows agents to edit.

The agent's calls carry *seen*, the (epoch, revision) it last looked at; an
edit of a drawing that changed since is refused, so it never overwrites a
person's edit it has not seen.
"""

from __future__ import annotations

import atexit
import base64
import contextlib
import hmac
import io
import itertools
import json
import math
import os
import secrets
import shlex
import shutil
import sys
import threading
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict, deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cairosvg
import numpy as np
from cairosvg.colors import color as css_colour
from PIL import Image
from shapely.geometry import LineString, Polygon
from shapely.geometry import Point as ShapelyPoint

from vectrify.document import (
    Document,
    DocumentError,
    HitIndex,
    Selection,
    StaleRevisionError,
    export_svg,
)
from vectrify.document.hit_test import IDENTITY, Matrix
from vectrify.document.holes import document_hole_shape, find_holes
from vectrify.document.join import path_style
from vectrify.document.model import EditKind, Subpath
from vectrify.document.regions import object_matrix, region_polygon, samples
from vectrify.document.svg import parse_path
from vectrify.document.topology import inverse_matrix, mapped_point
from vectrify.image_utils import on_white
from vectrify.operations.generate import frame
from vectrify.ui import agent_look

# What the history puts before an agent's edits.
AGENT_PREFIX = "Agent: "
# Rendered images: the longest side by default, and at most.
DEFAULT_SIDE = 1024
MAX_SIDE = 2048
MIN_SIDE = 16
# How many objects describe() lists at a time by default, and at most.
PAGE_SIZE = 100
MAX_PAGE = 500
# How long job(action="status") may wait for a job.
MAX_WAIT = 120.0
# How long after its last call an agent still shows as connected.
CONNECTED = 120.0
# How many of the agent's recent changes the page is told the objects of.
TOUCHED = 20
# How many nodes points() lists at a time by default, and at most.
NODES = 300
MAX_NODES = 2000
# How many objects pick() lists, and contours of each.
PICKED = 20
CONTOURS = 30

# Each agent tool that edits, and the editor commands it sends. The MCP
# server has one tool of the same name for each. Each also sends "select",
# choosing its own targets for the session's checks; the agent keeps no
# selection between calls, so there is no select tool.
EDITS: dict[str, tuple[str, ...]] = {
    "properties": ("paint", "rename", "locks"),
    "transform": ("resize", "move"),
    "arrange": ("reorder", "move_objects"),
    "group": ("group",),
    "ungroup": ("ungroup",),
    "join": ("join_paths", "join_ends", "join_two_ends"),
    "split_parts": ("split_disconnected",),
    "cut_hole": ("cut_hole",),
    "holes": ("fill_holes", "holes_to_shapes"),
    "convert": ("fill_to_line", "line_to_fill", "convert_lines", "detach"),
    "delete": ("delete", "delete_node", "delete_contour", "extract", "detach"),
    "add_path": ("add_path", "paint", "move_objects", "rename"),
    "set_points": ("move_nodes",),
    "point_style": ("node_handles", "pin"),
    "break_points": ("break_points", "delete_segment"),
    "split_edge": ("split",),
    "extract": ("extract", "detach", "move_objects", "group"),
    "knife": ("knife",),
    "redraw_outline": ("redraw_outline",),
    "set_reference": ("reference",),
    "undo": ("undo",),
    "redo": ("redo",),
}

# Editor commands no agent tool sends, and why.
LEFT_OUT = {
    "open": "An agent opens a file as its own headless target; it never "
    "replaces the drawing in the person's window.",
    "node": "The one-point drag; set_points sends move_nodes, which moves one "
    "point or many.",
    "to_front": "Internal: arrange(to='front') arrives as reorder and is "
    "renamed to this inside the session.",
    "to_back": "Internal: arrange(to='back') arrives as reorder and is "
    "renamed to this inside the session.",
}

# Calls that only look, and so need no refresh of the window.
LOOKS = frozenset(
    {
        "hello",
        "describe",
        "render",
        "compare",
        "get_svg",
        "points",
        "pick",
        "trace_reference",
        "view",
        "history",
        "export",
    }
)

Box = tuple[float, float, float, float]
# The paint describe() and pick() list of an object.
PAINT_KEYS = (
    "fill",
    "stroke",
    "stroke-width",
    "opacity",
    "fill-opacity",
    "stroke-opacity",
)
# What pick() looks through to the objects inside.
CONTAINERS = frozenset({"svg", "g", "defs", "clipPath", "linearGradient", "stop"})
# Between the drawing and the reference set side by side.
SIDE_GAP = 4
# One step of an edit: a payload, one made once the steps before are done,
# or None to skip it.
Step = dict | Callable[[], dict | None] | None
# What a region edit acts on: paths, the basic shapes it turns into paths
# where it cuts them, and (detached first) instances.
REGION_TAGS = frozenset({"path", "rect", "circle", "ellipse", "line", "use"})


# What a session refuses an edit with, as Backend.handle answers them.
REFUSALS = (
    DocumentError,
    ValueError,
    TypeError,
    KeyError,
    IndexError,
    AttributeError,
    OSError,
)


def reason(exc: BaseException) -> str:
    """A refusal's message for the agent."""
    if isinstance(exc, KeyError):
        return (
            f"No such object, point or field: {exc.args[0] if exc.args else ''} "
            "(pick(x, y) gives the ids at a spot, points(id) a path's node ids)"
        )
    return str(exc) or type(exc).__name__


class RefusedError(DocumentError):
    """A refused edit whose earlier steps were taken back.

    Taking them back is a new revision, so *where* says which: the drawing
    is otherwise as the agent last saw it.
    """

    def __init__(self, message: str, where: dict[str, Any]):
        super().__init__(message)
        self.where = where


@dataclass
class Reply:
    """An agent call's answer: data, and PNG images by name."""

    data: dict[str, Any]
    images: list[tuple[str, bytes]] = field(default_factory=list)


def _finite(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise DocumentError(f"{what} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise DocumentError(f"{what} must be finite")
    return result


def _box(value: Any, what: str = "The region") -> Box:
    if not isinstance(value, list | tuple) or len(value) != 4:
        raise DocumentError(f"{what} is [x, y, width, height] in document units")
    x, y, w, h = (_finite(v, what) for v in value)
    if w <= 0 or h <= 0:
        raise DocumentError(f"{what} needs a positive width and height")
    return x, y, w, h


def _ids(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str) or not isinstance(value, list | tuple):
        raise DocumentError("Give object ids as a list")
    return [str(v) for v in value]


def _targets(value: Any) -> list[str]:
    """The ids an edit acts on: given, never the current selection."""
    ids = _ids(value)
    if not ids:
        raise DocumentError("Give ids, the objects to act on")
    return ids


def _points(value: Any) -> list[list[str]]:
    if (
        not isinstance(value, list | tuple)
        or not value
        or any(not isinstance(p, list | tuple) or len(p) != 2 for p in value)
    ):
        raise DocumentError("Give points as a list of [object id, node id] pairs")
    return [[str(o), str(n)] for o, n in value]


def kept_selection(selection: Selection, document: Document) -> Selection:
    """*selection* without the objects and points *document* no longer has."""
    existing = {e.id for e in document.elements()}
    objects = selection.object_ids & existing
    nodes = selection.node_ids
    if nodes:
        present: set[str] = set()
        for oid in objects:
            for element in Document(document.element(oid)).elements():
                with contextlib.suppress(DocumentError):
                    geometry = document.geometry_for(element.id)
                    present.update(n.id for sp in geometry.subpaths for n in sp.nodes)
        nodes = nodes & present
    kept = Selection(objects, nodes, selection.whole_document)
    try:
        document.selection_ids(kept)
    except DocumentError:
        return Selection()
    return kept


def changed_objects(before: Document, after: Document) -> set[str]:
    """The objects of *after* that are new or differ from *before* in tag,
    attributes, name, locks, parent or geometry (not in their children)."""

    def signatures(document: Document) -> dict[str, tuple]:
        found: dict[str, tuple] = {}
        stack = [(child, "") for child in document.root.children]
        while stack:
            element, parent = stack.pop()
            geometry = None
            if element.geometry_id is not None:
                with contextlib.suppress(DocumentError):
                    geometry = document.geometry(element.geometry_id)
            found[element.id] = (
                (element.tag, element.attributes, element.name, element.locks, parent),
                geometry,
            )
            stack.extend((child, element.id) for child in element.children)
        return found

    old = signatures(before)
    changed = set()
    for oid, (own, geometry) in signatures(after).items():
        was = old.get(oid)
        # Geometries are immutable: an unchanged one is the same object.
        if was is None or was[0] != own or was[1] is not geometry:
            changed.add(oid)
    return changed


def _map_values(values: Sequence[float], matrix: Matrix) -> list[float]:
    """A node's values, pairs of coordinates, mapped through *matrix*."""
    if tuple(matrix) == IDENTITY:
        return list(values)
    mapped = []
    for x, y in zip(values[::2], values[1::2], strict=True):
        mx, my = mapped_point((x, y), matrix)
        mapped.extend((round(mx, 6), round(my, 6)))
    return mapped


def _contour_summary(subpath: Subpath, index: int, matrix: Matrix) -> dict[str, Any]:
    """A contour's index, ID, first node, node count, closed and bounds
    [x, y, w, h] in document units."""
    points = np.asarray([mapped_point(p, matrix) for p in samples(subpath)])
    left, top = points.min(axis=0)
    right, bottom = points.max(axis=0)
    return {
        "index": index,
        "id": subpath.id,
        "first_node": subpath.nodes[0].id,
        "count": len(subpath.nodes),
        "closed": subpath.closed,
        "bounds": [
            round(float(left), 3),
            round(float(top), 3),
            round(float(right - left), 3),
            round(float(bottom - top), 3),
        ],
    }


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def _size(region: Box, side: int) -> tuple[int, int]:
    scale = side / max(region[2], region[3])
    return max(1, round(region[2] * scale)), max(1, round(region[3] * scale))


def render_document(svg: str, region: Box, size: tuple[int, int]) -> Image.Image:
    """*svg* over *region*, stretched to *size* pixels, on white."""
    root = ET.fromstring(svg)
    frame(root, region, size)
    png = cairosvg.svg2png(bytestring=ET.tostring(root), background_color="white")
    assert png is not None
    with Image.open(io.BytesIO(png)) as image:
        return image.convert("RGB")


def crop_reference(
    reference: Image.Image, artboard: Box, region: Box, size: tuple[int, int]
) -> Image.Image:
    """The part of *reference*, stretched over the artboard as the editor
    shows it, that lies under *region*; white outside it."""
    vx, vy, vw, vh = artboard
    sx, sy = reference.width / vw, reference.height / vh
    x, y, w, h = region
    box = ((x - vx) * sx, (y - vy) * sy, (x + w - vx) * sx, (y + h - vy) * sy)
    return reference.transform(
        size,
        Image.Transform.EXTENT,
        box,
        Image.Resampling.BILINEAR,
        fillcolor="white",
    )


def heat_map(difference: np.ndarray) -> Image.Image:
    """Black where the images agree, through red and yellow to white."""
    d = np.clip(difference * 3, 0, 3)
    rgb = np.stack([np.clip(d, 0, 1), np.clip(d - 1, 0, 1), np.clip(d - 2, 0, 1)], -1)
    return Image.fromarray((rgb * 255).astype(np.uint8), "RGB")


class Agent:
    """An agent's calls on one editor session."""

    def __init__(self, session):
        self.session = session
        self._renders: OrderedDict[tuple, Image.Image] = OrderedDict()
        self._index: tuple[tuple, HitIndex] | None = None
        self._svg: tuple[tuple, str] | None = None
        self._reference: tuple[str, Image.Image] | None = None
        # For the window: how many calls changed something, and the last one.
        self.changes = 0
        self.last_action: str | None = None
        self.last_time = 0.0
        # The objects each recent change touched, as {change, ids}, for the
        # window to show; and the current call's, once it has made them.
        self.touched: deque[dict[str, Any]] = deque(maxlen=TOUCHED)
        self._touched: list[str] = []
        # Called after each call, for the window to show it at once.
        self.on_call: Callable[[], None] | None = None

    # The call boundary --------------------------------------------------

    def call(self, tool: str, args: dict[str, Any] | None = None) -> Reply:
        """Answer one call; refusals raise DocumentError with the reason."""
        args = dict(args or {})
        handler = getattr(self, f"tool_{tool}", None)
        if handler is None:
            raise DocumentError(f"Unknown agent call: {tool}")
        seen = args.pop("seen", None)
        self._touched = []
        try:
            return self._answer(tool, handler, seen, args)
        finally:
            # Even a refusal may have made a revision, taking back its steps.
            if self.on_call is not None:
                self.on_call()

    def _answer(
        self, tool: str, handler: Callable[..., Reply], seen: Any, args: dict
    ) -> Reply:
        try:
            reply = handler(seen, **args)
        except TypeError as exc:
            # A wrong or missing argument, said in the tool's own terms.
            raise DocumentError(f"{tool}: {exc}") from None
        finally:
            self.last_time = time.monotonic()
        # A job's status only looks; its other actions change something.
        looks = tool in LOOKS or (
            tool == "job" and args.get("action", "status") == "status"
        )
        if not looks:
            self.changes += 1
            self.last_action = reply.data.get("step") or (
                f"{args['action']} job" if tool == "job" else tool.replace("_", " ")
            )
            if self._touched:
                self.touched.append({"change": self.changes, "ids": self._touched})
        return reply

    def _key(self) -> tuple:
        return (self.session.epoch, self.session.editor.snapshot.revision)

    def _where(self) -> dict[str, Any]:
        snapshot = self.session.editor.snapshot
        return {"epoch": self.session.epoch, "revision": snapshot.revision}

    def _check_seen(self, seen: Any) -> None:
        if not isinstance(seen, list | tuple) or len(seen) != 2:
            raise StaleRevisionError(
                "Look at the drawing before editing it: call describe() first."
            )
        epoch, revision = seen
        session = self.session
        if epoch != session.epoch:
            raise StaleRevisionError(
                "Another drawing was opened since you last looked. Call "
                "describe() to see it before editing."
            )
        now = session.editor.snapshot.revision
        if revision != now:
            raise StaleRevisionError(
                f"The drawing changed since you last looked (revision {revision}, "
                f"now {now}); someone else may have edited it. Call describe() "
                "and render() to see it as it is, then make this edit again."
            )

    def _edit(
        self,
        seen: Any,
        steps: Sequence[Step],
        label: str | None = None,
    ) -> Reply:
        """Send *steps* to Session.action as one undo step, as the agent.

        A step is a payload, or a function giving one (or None to skip)
        once the steps before it are done. A refusal takes back the steps
        already done.
        """
        session = self.session
        with session.lock, self._own_selection() as touched:
            self._check_seen(seen)
            editor = session.editor
            before = {e.id for e in editor.snapshot.document.elements()}
            revision = editor.snapshot.revision
            since = len(editor.undo_entries)
            editor.label_prefix = AGENT_PREFIX
            try:
                for step in steps:
                    payload = step() if callable(step) else step
                    if payload is None:
                        continue
                    session.action(
                        {
                            **payload,
                            "epoch": session.epoch,
                            "revision": editor.snapshot.revision,
                        }
                    )
                    # What each step selects, its targets and its results.
                    touched.update(editor.snapshot.selection.object_ids)
            except Exception as exc:
                editor.rollback(since)
                self._settle(revision)
                if editor.snapshot.revision != revision:
                    raise RefusedError(reason(exc), self._where()) from exc
                raise
            finally:
                editor.label_prefix = ""
            if label is not None:
                label = AGENT_PREFIX + label
            editor.squash(since, label)
            self._settle(revision)
            snapshot = editor.snapshot
            after = {e.id for e in snapshot.document.elements()}
            data: dict[str, Any] = {
                **self._where(),
                "changed": snapshot.revision != revision,
                "result": self._selection(),
            }
            if len(editor.undo_entries) > since:
                data["step"] = editor.undo_labels[-1]
            created, removed = sorted(after - before), sorted(before - after)
            if created:
                data["created"] = created[:200]
            if removed:
                data["removed"] = removed[:200]
            return Reply(data)

    def _settle(self, revision: int) -> None:
        """Make the call one revision past *revision*, however many edits it
        took: what was cached at the revisions in between is dropped, as
        the next edits reuse their numbers."""
        if self.session.editor.snapshot.revision <= revision + 1:
            return
        self.session.settle(revision)
        self._svg = self._index = None
        self._renders = OrderedDict(
            (key, image) for key, image in self._renders.items() if key[2] <= revision
        )

    @contextlib.contextmanager
    def _own_selection(self) -> Iterator[set[str]]:
        """Run a call on a selection of the agent's own, then give the person
        theirs back, without the objects and points the call deleted.

        Held under the session lock, so the window never shows the agent's.
        The selection is not a revision, and the call's undo step selects
        the person's too. Yields the set of ids the call touched, to add to;
        the objects it changed are added after it.
        """
        editor = self.session.editor
        person = editor.snapshot.selection
        document = editor.snapshot.document
        since = len(editor.undo_entries)
        touched: set[str] = set()
        try:
            yield touched
            after = editor.snapshot.document
            if after is not document:
                touched |= changed_objects(document, after)
            existing = {e.id for e in after.elements()} - {after.root.id}
            self._touched = sorted(touched & existing)[:200]
        finally:
            kept = kept_selection(person, editor.snapshot.document)
            editor.select(kept)
            editor.reselect(since, person, kept)

    @staticmethod
    def _select(ids: list[str]) -> dict:
        """The step selecting *ids*, the call's own targets."""
        return {"command": "select", "objects": ids}

    @staticmethod
    def _select_points(points: list[list[str]]) -> dict:
        return {
            "command": "select",
            "objects": sorted({o for o, _ in points}),
            "nodes": sorted({n for _, n in points}),
        }

    def _selection(self) -> dict[str, Any]:
        snapshot = self.session.editor.snapshot
        document = snapshot.document
        nodes = snapshot.selection.node_ids
        points = []
        if nodes:
            for oid in sorted(snapshot.selection.object_ids):
                with contextlib.suppress(DocumentError):
                    geometry = document.geometry_for(oid)
                    points.extend(
                        [oid, n.id]
                        for sp in geometry.subpaths
                        for n in sp.nodes
                        if n.id in nodes
                    )
        return {"objects": sorted(snapshot.selection.object_ids), "points": points}

    # Looking ------------------------------------------------------------

    def _document_svg(self) -> str:
        key = self._key()
        if self._svg is None or self._svg[0] != key:
            self._svg = (key, export_svg(self.session.editor.snapshot.document))
        return self._svg[1]

    def _hits(self) -> HitIndex:
        key = self._key()
        if self._index is None or self._index[0] != key:
            self._index = (key, HitIndex(self.session.editor.snapshot.document))
        return self._index[1]

    def _reference_image(self) -> Image.Image | None:
        reference = self.session.reference
        if not reference:
            return None
        url = reference["data_url"]
        if self._reference is None or self._reference[0] != url:
            image = self.session.reference_image()
            assert image is not None
            self._reference = (url, on_white(image))
        return self._reference[1]

    def _region(self, region: Any) -> Box:
        if region is None:
            return self.session.editor.snapshot.document.artboard()
        return _box(region)

    @staticmethod
    def _side(max_side: Any) -> int:
        if max_side is None:
            return DEFAULT_SIDE
        if isinstance(max_side, bool) or not isinstance(max_side, int):
            raise DocumentError("max_side is a whole number of pixels")
        return max(MIN_SIDE, min(MAX_SIDE, max_side))

    def _cached(self, kind: str, region: Box, size: tuple[int, int]) -> Image.Image:
        """The drawing or the reference over *region*, cached per revision."""
        with self.session.lock:
            reference = self.session.reference
            key = (
                kind,
                *self._key(),
                region,
                size,
                hash(reference["data_url"]) if reference and kind == "reference" else 0,
            )
            image = self._renders.get(key)
            if image is not None:
                self._renders.move_to_end(key)
                return image
            if kind == "drawing":
                image = render_document(self._document_svg(), region, size)
            else:
                picture = self._reference_image()
                if picture is None:
                    raise DocumentError(
                        "No reference image is loaded; load_reference() one"
                    )
                artboard = self.session.editor.snapshot.document.artboard()
                image = crop_reference(picture, artboard, region, size)
            self._renders[key] = image
            while len(self._renders) > 24:
                self._renders.popitem(last=False)
            return image

    def _view(self) -> dict[str, Any]:
        view = self.session.view
        if view is None:
            raise DocumentError(
                "No editor window has said what it shows: this is a file opened "
                "headlessly, or the window has not reported yet. Give a region "
                "[x, y, w, h] instead; view() says what the person sees."
            )
        return view

    def _look(self, region: Any, max_side: Any, default: int = DEFAULT_SIDE):
        """The region to look at and the long side to render it at: region
        "view" is what the person's window shows, at its size."""
        if region == "view":
            view = self._view()
            pixels = view["pixels"]
            side = (
                self._side(max_side)
                if max_side is not None
                else max(MIN_SIDE, min(MAX_SIDE, max(pixels)))
            )
            return tuple(view["region"]), side
        return self._region(region), self._side(
            max_side if max_side is not None else default
        )

    def tool_hello(self, _seen: Any) -> Reply:
        return Reply({**self._where(), "name": self.session.name})

    @staticmethod
    def _object_row(row: dict[str, Any], hits: HitIndex) -> dict[str, Any]:
        """An object as describe() and pick() list it."""
        attributes = row["attributes"]
        bounds = hits.bounds(frozenset({row["id"]}))
        item: dict[str, Any] = {
            "id": row["id"],
            "label": row["label"],
            "tag": row["tag"],
            "parent": row["parent"],
            "depth": row["depth"],
            "paint": {k: attributes[k] for k in PAINT_KEYS if k in attributes},
            "bounds": [
                round(bounds[0], 3),
                round(bounds[1], 3),
                round(bounds[2] - bounds[0], 3),
                round(bounds[3] - bounds[1], 3),
            ]
            if bounds
            else None,
        }
        if row["name"]:
            item["name"] = row["name"]
        if attributes.get("transform"):
            item["transform"] = attributes["transform"]
        for key in ("locks", "inherited_locks"):
            if row[key]:
                item[key] = row[key]
        if row["shared"]:
            item["shared_geometry"] = True
        if row["resource"]:
            item["definition"] = True
        return item

    def _covering(self, shape: Any) -> list[tuple[str, list[tuple[int, str]]]]:
        """The drawn objects that paint within *shape*, front to back, each
        with the contours (index, "stroke" or "fill") of it that do."""
        hits = self._hits()
        document = self.session.editor.snapshot.document
        left, top, right, bottom = shape.bounds
        found = []
        for element in reversed(document.elements()):
            if element.children or element.tag in CONTAINERS:
                continue
            area = hits.area(element.id)
            if area is None:
                continue
            x0, y0, x1, y1 = area.bounds
            if x0 > right or x1 < left or y0 > bottom or y1 < top:
                continue
            if not area.intersects(shape):
                continue
            contours = (
                hits.contours_in(element.id, shape) if element.tag == "path" else []
            )
            found.append((element.id, contours))
        return found

    def _contour_rows(
        self, oid: str, contours: Sequence[tuple[int, str]]
    ) -> list[dict[str, Any]]:
        document = self.session.editor.snapshot.document
        geometry = document.geometry_for(oid)
        matrix = object_matrix(document, oid)
        rows = []
        for index, paint in contours[:CONTOURS]:
            row = _contour_summary(geometry.subpaths[index], index, matrix)
            row["paints"] = paint
            rows.append(row)
        return rows

    def _rows_by_id(self) -> dict[str, dict[str, Any]]:
        return {r["id"]: r for r in self.session.state(svg=False)["objects"]}

    def tool_describe(
        self,
        _seen: Any,
        page: int = 0,
        page_size: int = PAGE_SIZE,
        within: str | None = None,
        region: Any = None,
    ) -> Reply:
        session = self.session
        if type(page) is not int or page < 0:
            raise DocumentError("page is a whole number from 0")
        if type(page_size) is not int or not 1 <= page_size <= MAX_PAGE:
            raise DocumentError(f"page_size is from 1 to {MAX_PAGE}")
        with session.lock:
            state = session.state(svg=False)
            document = session.editor.snapshot.document
            rows = state["objects"]
            if within is not None:
                inside = {e.id for e in Document(document.element(within)).elements()}
                rows = [r for r in rows if r["id"] in inside and r["id"] != within]
            contours: dict[str, list[tuple[int, str]]] = {}
            if region is not None:
                found = self._covering(Polygon(region_polygon(region)))
                contours = dict(found)
                by_id = {r["id"]: r for r in rows}
                rows = [by_id[oid] for oid, _ in found if oid in by_id]
            chosen = rows[page * page_size : (page + 1) * page_size]
            hits = self._hits() if chosen else None
            objects = []
            for row in chosen:
                assert hits is not None
                item = self._object_row(row, hits)
                if contours.get(row["id"]):
                    item["contours"] = self._contour_rows(
                        row["id"], contours[row["id"]]
                    )
                    if len(contours[row["id"]]) > CONTOURS:
                        item["contours_total"] = len(contours[row["id"]])
                objects.append(item)
            picture = self._reference_image()
            undo, redo = session.editor.undo_labels, session.editor.redo_labels
            data = {
                **self._where(),
                "name": session.name,
                "artboard": state["bounds"],
                "reference": {
                    "name": state["reference"]["name"],
                    "pixels": [picture.width, picture.height] if picture else None,
                }
                if state["reference"]
                else None,
                "selection": self._selection(),
                "root": state["root"],
                "objects": objects,
                "page": page,
                "pages": max(1, math.ceil(len(rows) / page_size)),
                "total": len(rows),
                "undo": undo[-1] if undo else None,
                "redo": redo[0] if redo else None,
            }
            if region is not None:
                data["order"] = (
                    "front to back: the objects that paint inside the region, "
                    "with the contours of each that do"
                )
            elif len(rows) > page_size:
                data["next"] = (
                    "To find what is at a spot, pick(x, y) or "
                    "describe(region=[x, y, w, h]) rather than paging"
                )
            return Reply(data)

    def tool_pick(
        self,
        _seen: Any,
        x: float,
        y: float,
        radius: float = 0.0,
        limit: int = PICKED,
    ) -> Reply:
        """What paints at (x, y), or within radius of it, front to back, and
        the drawing's and the reference's colour there."""
        at = (_finite(x, "x"), _finite(y, "y"))
        reach = _finite(radius, "radius")
        if reach < 0:
            raise DocumentError("radius must not be negative")
        if type(limit) is not int or limit < 1:
            raise DocumentError("limit is a whole number from 1")
        spot = ShapelyPoint(at)
        shape = spot.buffer(reach) if reach > 0 else spot
        with self.session.lock:
            document = self.session.editor.snapshot.document
            found = self._covering(shape)
            rows = self._rows_by_id() if found else {}
            hits = self._hits()
            objects = []
            for oid, contours in found[:limit]:
                item = self._object_row(rows[oid], hits)
                item["groups"] = [a.id for a in document.ancestry(oid)[1:-1]]
                if contours:
                    item["contours"] = self._contour_rows(oid, contours)
                    if len(contours) > CONTOURS:
                        item["contours_total"] = len(contours)
                objects.append(item)
            data: dict[str, Any] = {
                **self._where(),
                "at": list(at),
                "radius": reach,
                "colour": self._colours(at, reach),
                "order": "front to back",
                "objects": objects,
                "total": len(found),
            }
            if not found:
                data["note"] = (
                    "Nothing paints here; a larger radius, or describe(region=...)"
                )
            else:
                data["next"] = (
                    "points(id, contours=[index]) lists a contour's nodes; "
                    "extract(region) takes what lies in a region into a path of "
                    "its own; delete(region=...) deletes it."
                )
            return Reply(data)

    def tool_view(self, _seen: Any) -> Reply:
        """What the person is looking at: their selection, and what the
        window shows."""
        with self.session.lock:
            session = self.session
            data: dict[str, Any] = {**self._where(), "selection": self._selection()}
            view = session.view
            if view is None:
                data.update(
                    window=False,
                    region=list(session.editor.snapshot.document.artboard()),
                    note="No editor window has said what it shows (a file opened "
                    "headlessly, or a window that has not reported yet): the "
                    "region is the whole artboard.",
                )
                return Reply(data)
            data.update(
                window=True,
                region=[round(v, 3) for v in view["region"]],
                zoom=round(view["zoom"], 4),
                pixels=view["pixels"],
                tool=view["tool"],
                entered_group=view["entered"],
                reference_view=view["reference_view"],
                reported_seconds_ago=round(time.monotonic() - session.view_time, 1),
                note="zoom is screen pixels per document unit; region is "
                "[x, y, w, h] of the drawing the window shows; "
                'render(region="view") renders it.',
            )
            return Reply(data)

    def _compose(
        self, box: Box, side: int, overlay: str, opacity: float = 0.5
    ) -> tuple[Image.Image, tuple[int, int]]:
        """The drawing over *box*, with the reference beside it, blended over
        it or instead of it; and the size of one of them."""
        if overlay == "side":
            # Two images side by side share the pixel budget.
            size = _size(box, side // 2 if box[2] >= box[3] else side)
        else:
            size = _size(box, side)
        if overlay == "reference":
            return self._cached("reference", box, size), size
        drawing = self._cached("drawing", box, size)
        if overlay == "none":
            return drawing, size
        reference = self._cached("reference", box, size)
        if overlay == "over":
            return Image.blend(drawing, reference, opacity), size
        image = Image.new("RGB", (2 * size[0] + SIDE_GAP, size[1]), "#808080")
        image.paste(drawing, (0, 0))
        image.paste(reference, (size[0] + SIDE_GAP, 0))
        return image, size

    def tool_render(
        self,
        _seen: Any,
        region: Any = None,
        overlay: str = "none",
        max_side: int | None = None,
        grid: bool = False,
    ) -> Reply:
        if overlay not in {"none", "side", "over", "reference"}:
            raise DocumentError(
                "overlay is none, side (by side), over or reference (alone)"
            )
        box, side = self._look(region, max_side)
        opacity = 0.5
        if region == "view":
            view = self._view()
            mode = view["reference_view"] if self.session.reference else None
            if overlay == "none" and mode in {"overlay", "reference"}:
                # As the window shows it: the reference over the drawing at its
                # opacity there, or the reference alone.
                overlay = "over" if mode == "overlay" else "reference"
                opacity = view.get("reference_opacity") or 0.5
        image, size = self._compose(box, side, overlay, opacity)
        if grid:
            image = agent_look.draw_grid(image, box, 0, size[0])
            if overlay == "side":
                image = agent_look.draw_grid(image, box, size[0] + SIDE_GAP, size[0])
        data = {
            **self._where(),
            "region": list(box),
            "pixels": list(image.size),
            "mapping": agent_look.mapping(
                box, size, gap=SIDE_GAP if overlay == "side" else 0
            ),
        }
        if region == "view":
            data["shows"] = (
                "what the person's window shows"
                + (", the reference over the drawing" if overlay == "over" else "")
                + (", the reference alone" if overlay == "reference" else "")
            )
        name = "reference" if overlay == "reference" else "render"
        return Reply(data, [(name, _png(image))])

    def tool_compare(
        self,
        _seen: Any,
        region: Any = None,
        max_side: int | None = None,
        grid: bool = False,
    ) -> Reply:
        """The mean squared error against the reference, a heat map of where
        they differ, and the worst cells of a 4 by 4 grid over the region."""
        box, side = self._look(region, max_side, 512)
        size = _size(box, side)
        drawing = np.asarray(self._cached("drawing", box, size), dtype=np.float64)
        reference = np.asarray(self._cached("reference", box, size), dtype=np.float64)
        squared = ((drawing - reference) / 255) ** 2
        per_pixel = squared.mean(axis=2)
        cells = []
        height, width = per_pixel.shape
        for row in range(4):
            for col in range(4):
                top, bottom = row * height // 4, (row + 1) * height // 4
                left, right = col * width // 4, (col + 1) * width // 4
                if bottom <= top or right <= left:
                    continue
                x = box[0] + box[2] * left / width
                y = box[1] + box[3] * top / height
                cells.append(
                    {
                        "region": [
                            round(x, 3),
                            round(y, 3),
                            round(box[2] * (right - left) / width, 3),
                            round(box[3] * (bottom - top) / height, 3),
                        ],
                        "mse": round(
                            float(per_pixel[top:bottom, left:right].mean()), 6
                        ),
                    }
                )
        cells.sort(key=lambda c: -c["mse"])
        heat = heat_map(np.sqrt(per_pixel))
        if grid:
            heat = agent_look.draw_grid(heat, box)
        return Reply(
            {
                **self._where(),
                "region": list(box),
                "mse": round(float(per_pixel.mean()), 6),
                "worst_cells": cells[:4],
                "mapping": agent_look.mapping(box, size),
            },
            [("difference", _png(heat))],
        )

    def _colours(self, at: tuple[float, float], reach: float) -> dict[str, Any]:
        """The drawing's and the reference's mean colour within *reach* of
        *at* (half a unit at least), and how far apart they are."""
        half = max(reach, 0.5)
        box = (at[0] - half, at[1] - half, 2 * half, 2 * half)
        size = (17, 17)
        drawing = agent_look.disc_mean(self._cached("drawing", box, size))
        colours: dict[str, Any] = {
            "drawing": agent_look.hex_colour(drawing),
            "reference": None,
        }
        if self.session.reference:
            reference = agent_look.disc_mean(self._cached("reference", box, size))
            colours["reference"] = agent_look.hex_colour(reference)
            colours["difference"] = round(
                math.dist(drawing, reference) / math.sqrt(3), 4
            )
        return colours

    def tool_trace_reference(
        self,
        _seen: Any,
        region: Any = None,
        colour: str | None = None,
        dark: bool = True,
        tolerance: float | None = None,
        min_area: float | None = None,
    ) -> Reply:
        """Outlines, in document units, of the reference's dark areas (or
        those near *colour*) in a region."""
        box, _ = self._look(region, None)
        picture = self._reference_image()
        if picture is None:
            raise DocumentError("No reference image is loaded; load_reference() one")
        rgb: tuple[float, float, float] | None = None
        if colour is not None:
            try:
                r, g, b, _alpha = css_colour(str(colour))
            except Exception:
                raise DocumentError(f"Not a CSS colour: {colour}") from None
            rgb = (r, g, b)
        elif not dark:
            raise DocumentError("Give a colour to trace, or dark=true")
        limit = (
            _finite(tolerance, "tolerance")
            if tolerance is not None
            else (0.35 if rgb is None else 0.12)
        )
        artboard = self.session.editor.snapshot.document.artboard()
        # The reference's own resolution there, within reason.
        native = max(
            box[2] * picture.width / artboard[2], box[3] * picture.height / artboard[3]
        )
        size = _size(box, round(min(1024, max(256, native))))
        image = self._cached("reference", box, size)
        mask = agent_look.area_mask(image, colour=rgb, tolerance=limit)
        unit = (box[2] / size[0]) * (box[3] / size[1])
        min_pixels = (
            max(1, math.ceil(_finite(min_area, "min_area") / unit))
            if min_area is not None
            else 6
        )
        traced = agent_look.trace_areas(mask, box, min_pixels=min_pixels)
        data = {
            **self._where(),
            "region": list(box),
            "pixels": list(size),
            "traced": f"luminance at most {limit}"
            if rgb is None
            else f"within {limit} of {agent_look.hex_colour(rgb)}",
            **traced,
            "next": "Each shape's d is closed path data in document units, holes "
            "included: compare it with points(id, region=...) and move nodes "
            "with set_points, or draw it with add_path(d).",
        }
        if not traced["shapes"]:
            data["note"] = (
                "Nothing matched: a higher tolerance, or pick(x, y) for the colour"
            )
        return Reply(data)

    def tool_get_svg(self, _seen: Any, ids: Any = None) -> Reply:
        wanted = _ids(ids)
        with self.session.lock:
            svg = self._document_svg()
            where = self._where()
        if wanted is None:
            return Reply({**where, "svg": svg})
        root = ET.fromstring(svg)
        found = {e.get("id"): e for e in root.iter() if e.get("id") in wanted}
        missing = [i for i in wanted if i not in found]
        if missing:
            raise DocumentError(f"No object with id {missing[0]}")
        return Reply(
            {
                **where,
                "elements": {
                    i: ET.tostring(found[i], encoding="unicode").replace(
                        ' xmlns:ns0="http://www.w3.org/2000/svg"', ""
                    )
                    for i in wanted
                },
            }
        )

    def tool_points(
        self,
        _seen: Any,
        id: str,  # noqa: A002
        region: Any = None,
        contours: Any = None,
        coords: str = "document",
        nodes: bool = True,
        page: int = 0,
        page_size: int = NODES,
    ) -> Reply:
        """A path's contours, and their nodes a page at a time: all, those of
        the contours *contours*, or those inside *region*."""
        if coords not in {"document", "local", "both"}:
            raise DocumentError("coords is document, local or both")
        if type(page) is not int or page < 0:
            raise DocumentError("page is a whole number from 0")
        if type(page_size) is not int or not 1 <= page_size <= MAX_NODES:
            raise DocumentError(f"page_size is from 1 to {MAX_NODES}")
        with self.session.lock:
            document = self.session.editor.snapshot.document
            geometry = document.geometry_for(id)
            matrix = object_matrix(document, id)
            subpaths = geometry.subpaths
            if contours is not None:
                if not isinstance(contours, list | tuple) or any(
                    type(i) is not int or not 0 <= i < len(subpaths) for i in contours
                ):
                    raise DocumentError(
                        f"contours are indices from 0 to {len(subpaths) - 1}"
                    )
                chosen = sorted(set(contours))
            else:
                chosen = list(range(len(subpaths)))
            inside = None
            if region is not None:
                polygon = Polygon(region_polygon(region))
                inside = polygon.covers

                def meets(subpath: Subpath) -> bool:
                    line = [mapped_point(p, matrix) for p in samples(subpath)]
                    shape = (
                        LineString(line)
                        if len(set(line)) > 1
                        else ShapelyPoint(line[0])
                    )
                    return polygon.intersects(shape)

                chosen = [i for i in chosen if meets(subpaths[i])]
            # What this page lists: whole contours, or nodes of them.
            if nodes:
                entries = [
                    (i, n, node)
                    for i in chosen
                    for n, node in enumerate(subpaths[i].nodes)
                    if inside is None
                    or inside(ShapelyPoint(mapped_point(node.endpoint, matrix)))
                ]
            else:
                entries = [(i, -1, None) for i in chosen]
            listed = entries[page * page_size : (page + 1) * page_size]
            rows: dict[int, dict[str, Any]] = {}
            for i, n, node in listed:
                row = rows.get(i)
                if row is None:
                    row = rows[i] = _contour_summary(subpaths[i], i, matrix)
                if node is None:
                    continue
                values = list(node.values)
                item: dict[str, Any] = {"id": node.id, "i": n, "command": node.command}
                if coords == "local":
                    item["values"] = values
                else:
                    item["values"] = _map_values(values, matrix)
                    if coords == "both":
                        item["local"] = values
                if node.pinned:
                    item["pinned"] = True
                row.setdefault("nodes", []).append(item)
            if page == 0 and nodes:
                # Contours crossing the region with no node inside it.
                listing = {i for i, _, _ in entries}
                for i in chosen:
                    if i not in listing:
                        rows[i] = _contour_summary(subpaths[i], i, matrix)
                rows = dict(sorted(rows.items()))
            # Holes: a filled path's contours that cut out of the ones around
            # them, by the ids holes() takes.
            found: dict[str, Any] = {}
            with contextlib.suppress(DocumentError):
                found = {h.id: h for h in find_holes(document, id)}
            for row in rows.values():
                hole = found.get(row["id"])
                row["hole"] = hole is not None
                if hole is not None:
                    row["area"] = round(
                        document_hole_shape(document, id, (hole,)).area, 3
                    )
            pages = max(1, math.ceil(len(entries) / page_size))
            data: dict[str, Any] = {
                **self._where(),
                "object": id,
                "geometry": geometry.id,
                "users": [
                    e.id for e in document.elements() if e.geometry_id == geometry.id
                ],
                "coords": coords,
                "contours_total": len(subpaths),
                "contours_chosen": len(chosen),
                "holes_total": len(found),
                "contours": list(rows.values()),
                "page": page,
                "pages": pages,
            }
            if any(abs(a - b) > 1e-12 for a, b in zip(matrix, IDENTITY, strict=True)):
                data["transform"] = (
                    "matrix(" + " ".join(f"{v:.9g}" for v in matrix) + ")"
                )
                data["coords_note"] = (
                    "values are document coordinates (the path's own mapped "
                    "through its transform); set_points takes them so, or "
                    'coords="local" for its own'
                )
            if nodes:
                data["nodes_total"] = len(entries)
                if region is not None:
                    data["nodes_shown"] = (
                        "those inside the region; i is the node's place in its contour"
                    )
            if found:
                data["holes_note"] = (
                    "A contour with hole=true is a hole; holes(contours=[[object, "
                    'contour id]], action="fill" or "shape") fills it or makes it '
                    "a shape of its own"
                )
            if page + 1 < pages:
                data["more"] = (
                    f"{len(entries) - (page + 1) * page_size} more "
                    f"{'nodes' if nodes else 'contours'}: call points again with "
                    f"page={page + 1}, or narrow it with region or contours"
                )
            return Reply(data)

    def tool_history(self, _seen: Any, limit: int = 30) -> Reply:
        if type(limit) is not int or limit < 1:
            raise DocumentError("limit is a whole number from 1")
        with self.session.lock:
            editor = self.session.editor

            def entry(e) -> dict[str, Any]:
                agent = e.label.startswith(AGENT_PREFIX)
                return {
                    "label": e.label,
                    "author": "agent" if agent else "person",
                    "revision": e.revision,
                }

            undo = [entry(e) for e in editor.undo_entries]
            redo = [entry(e) for e in editor.redo_entries]
            return Reply(
                {
                    **self._where(),
                    # Newest first, so undo(1) takes back undo[0].
                    "undo": undo[::-1][:limit],
                    "redo": redo[:limit],
                    "undo_total": len(undo),
                    "redo_total": len(redo),
                }
            )

    def tool_export(self, _seen: Any, project: bool = False) -> Reply:
        with self.session.lock:
            content = self.session.project() if project else self._document_svg()
            return Reply(
                {**self._where(), "name": self.session.name, "content": content}
            )

    # Editing ------------------------------------------------------------

    def tool_properties(
        self,
        seen: Any,
        ids: Any,
        fill: str | None = None,
        stroke: str | None = None,
        stroke_width: float | None = None,
        opacity: float | None = None,
        fill_opacity: float | None = None,
        stroke_opacity: float | None = None,
        name: str | None = None,
        locks: Any = None,
    ) -> Reply:
        """Set the paint, the name (of one object) and the locks of *ids*."""
        targets = _targets(ids)
        changes = {
            key: str(value)
            for key, value in (
                ("fill", fill),
                ("stroke", stroke),
                ("stroke-width", stroke_width),
                ("opacity", opacity),
                ("fill-opacity", fill_opacity),
                ("stroke-opacity", stroke_opacity),
            )
            if value is not None
        }
        if not changes and name is None and locks is None:
            raise DocumentError(
                "Give a fill, stroke, stroke width, opacity, name or locks"
            )
        if name is not None and len(targets) != 1:
            raise DocumentError("Give one id to name")
        wanted: frozenset[str] | None = None
        if locks is not None:
            if not isinstance(locks, list | tuple):
                raise DocumentError("Give the locks as a list, empty to unlock")
            wanted = frozenset(str(v) for v in locks)

        def lock(oid: str, adding: bool) -> Callable[[], dict | None]:
            # Unlocking comes before the other changes, locking after them,
            # so a lock lifted here does not refuse them.
            def step() -> dict | None:
                assert wanted is not None
                element = self.session.editor.snapshot.document.element(oid)
                now = frozenset(str(getattr(v, "value", v)) for v in element.locks)
                after = wanted if adding else now & wanted
                if after == now:
                    return None
                return {"command": "locks", "object": oid, "locks": sorted(after)}

            return step

        steps: list[Step] = [self._select(targets)]
        if wanted is not None:
            steps.extend(lock(oid, adding=False) for oid in targets)
        if changes:
            steps.append({"command": "paint", "changes": changes})
        if name is not None:
            steps.append({"command": "rename", "object": targets[0], "name": name})
        if wanted is not None:
            steps.extend(lock(oid, adding=True) for oid in targets)
        kinds = sum(x is not None for x in (changes or None, name, wanted))
        return self._edit(seen, steps, "Properties" if kinds > 1 else None)

    def _selected_bounds(self) -> Box:
        selection = self.session.editor.snapshot.selection.object_ids
        if not selection:
            raise DocumentError("Give ids, the objects to measure")
        bounds = self._hits().bounds(frozenset(selection))
        if bounds is None:
            raise DocumentError("The objects paint nothing to measure")
        return bounds[0], bounds[1], bounds[2] - bounds[0], bounds[3] - bounds[1]

    def tool_transform(
        self,
        seen: Any,
        ids: Any,
        dx: float | None = None,
        dy: float | None = None,
        scale: Any = None,
        anchor: Any = "center",
        box: Any = None,
    ) -> Reply:
        """Scale the objects *ids* by *scale* about *anchor* (a point, or
        center, top-left, top-right, bottom-left, bottom-right) and move them
        by *dx*, *dy*; or fit their painted bounds to *box*."""
        offset = (
            _finite(dx if dx is not None else 0.0, "dx"),
            _finite(dy if dy is not None else 0.0, "dy"),
        )
        moving = dx is not None or dy is not None
        if box is not None and (scale is not None or moving):
            raise DocumentError("Give box alone, or scale and dx, dy")
        if box is None and scale is None and not moving:
            raise DocumentError("Give dx, dy to move, scale [sx, sy], or box")
        moved = {"dx": offset[0], "dy": offset[1]}

        def resize() -> dict | None:
            if box is None and scale is None:
                return None
            bx, by, bw, bh = self._selected_bounds()
            if box is not None:
                x, y, w, h = _box(box, "The box")
                if bw <= 0 or bh <= 0:
                    raise DocumentError("The objects have no area to resize")
                moved.update(dx=x - bx, dy=y - by)
                return {
                    "command": "resize",
                    "anchor": [bx, by],
                    "scale": [w / bw, h / bh],
                }
            if not isinstance(scale, list | tuple) or len(scale) != 2:
                raise DocumentError("scale is [sx, sy]")
            names = {
                "center": (bx + bw / 2, by + bh / 2),
                "top-left": (bx, by),
                "top-right": (bx + bw, by),
                "bottom-left": (bx, by + bh),
                "bottom-right": (bx + bw, by + bh),
            }
            if isinstance(anchor, str):
                if anchor not in names:
                    raise DocumentError(
                        "anchor is a point [x, y] or one of " + ", ".join(names)
                    )
                point = names[anchor]
            else:
                if not isinstance(anchor, list | tuple) or len(anchor) != 2:
                    raise DocumentError("anchor is a point [x, y]")
                point = (_finite(anchor[0], "anchor"), _finite(anchor[1], "anchor"))
            return {
                "command": "resize",
                "anchor": list(point),
                "scale": [_finite(scale[0], "scale"), _finite(scale[1], "scale")],
            }

        def move() -> dict | None:
            if abs(moved["dx"]) < 1e-9 and abs(moved["dy"]) < 1e-9:
                return None
            return {"command": "move", "dx": moved["dx"], "dy": moved["dy"]}

        label = None
        if box is not None or scale is not None:
            label = "Transform" if moving else "Resize"
        return self._edit(seen, [self._select(_targets(ids)), resize, move], label)

    def tool_arrange(
        self,
        seen: Any,
        ids: Any,
        to: str | None = None,
        parent: str | None = None,
        index: int | None = None,
    ) -> Reply:
        """Restack *ids* within their group (*to*), or move them into the
        group *parent* at *index* among its children (0 is the back; the
        front by default)."""
        targets = _targets(ids)
        if (to is None) == (parent is None):
            raise DocumentError(
                "Give to (front, back, forward, backward) to restack, or parent "
                "(and index) to move into a group"
            )
        if parent is not None:

            def into() -> dict:
                document = self.session.editor.snapshot.document
                at = len(document.element(parent).children) if index is None else index
                return {
                    "command": "move_objects",
                    "objects": targets,
                    "parent": parent,
                    "index": at,
                }

            return self._edit(seen, [into])
        if index is not None:
            raise DocumentError("index goes with parent, not with to")
        steps = {"forward": 1, "backward": -1}
        if to in {"front", "back"}:
            payload: dict[str, Any] = {"command": "reorder", "to": to}
        elif to in steps:
            payload = {"command": "reorder", "step": steps[str(to)]}
        else:
            raise DocumentError("to is front, back, forward or backward")
        return self._edit(seen, [self._select(targets), payload])

    def _simple(self, command: str, seen: Any, ids: Any, **extra: Any) -> Reply:
        return self._edit(
            seen, [self._select(_targets(ids)), {"command": command, **extra}]
        )

    def tool_group(self, seen: Any, ids: Any) -> Reply:
        return self._simple("group", seen, ids)

    def tool_ungroup(self, seen: Any, ids: Any) -> Reply:
        return self._simple("ungroup", seen, ids)

    def _join_candidates(self, ids: list[str]) -> list[str]:
        """The paths *ids* name, themselves or inside the groups among them,
        as the editor's Join counts them."""
        document = self.session.editor.snapshot.document
        found: list[str] = []
        for oid in ids:
            for element in Document(document.element(oid)).elements():
                if element.tag == "path" and element.id not in found:
                    found.append(element.id)
        return found

    def tool_join(
        self,
        seen: Any,
        ids: Any = None,
        points: Any = None,
        reach: float | None = None,
        bridge: str | None = None,
        color_source: str | None = None,
    ) -> Reply:
        """As the editor's Join: two points join each other; stroked lines
        join their ends within reach; filled paths merge into one outline,
        its colour mixed by area or taken from the path color_source."""
        if (ids is None) == (points is None):
            raise DocumentError(
                "Give ids (the paths to join) or points (two [object id, node "
                "id] pairs to join), not both"
            )
        if points is not None:
            if reach is not None or bridge is not None or color_source is not None:
                raise DocumentError(
                    "Two points join as they are: reach, bridge and color_source "
                    "go with ids"
                )
            pairs = _points(points)
            if len(pairs) != 2:
                raise DocumentError("Give two points to join")
            reply = self._edit(
                seen,
                [
                    self._select_points(pairs),
                    {"command": "join_two_ends", "points": pairs},
                ],
            )
            reply.data["joined"] = "points"
            return reply
        targets = _targets(ids)
        with self.session.lock:
            document = self.session.editor.snapshot.document
            paths = self._join_candidates(targets)
            styles = [path_style(document, document.element(p)) for p in paths]
        lines = [s for s in styles if s["fill"] == "none" and s.get("stroke") != "none"]
        if len(paths) > 1 and len(lines) < len(paths):
            if reach is not None or bridge is not None:
                raise DocumentError(
                    "These are filled paths, which merge by area: reach and "
                    "bridge go with stroked lines"
                )
            options: dict[str, Any] = (
                {"colors": "mix"}
                if color_source is None
                else {"colors": "source", "color_source": color_source}
            )
            reply = self._simple("join_paths", seen, targets, options=options)
            reply.data["joined"] = "outlines"
            return reply
        if color_source is not None:
            raise DocumentError(
                "These are stroked lines, which join at their ends: color_source "
                "goes with filled paths"
            )
        bridge = "curve" if bridge is None else bridge
        if bridge not in {"curve", "line"}:
            raise DocumentError("bridge is curve or line")
        reply = self._simple(
            "join_ends",
            seen,
            targets,
            reach=4.0 if reach is None else _finite(reach, "reach"),
            bridge=bridge,
        )
        reply.data["joined"] = "line ends"
        return reply

    def tool_split_parts(self, seen: Any, ids: Any) -> Reply:
        return self._simple("split_disconnected", seen, ids)

    def tool_cut_hole(self, seen: Any, ids: Any) -> Reply:
        return self._simple("cut_hole", seen, ids)

    def tool_holes(
        self,
        seen: Any,
        contours: Any,
        action: str = "fill",
        delete_enclosed: bool = False,
    ) -> Reply:
        """Fill holes, or turn them into shapes of their own; *contours* are
        [object id, hole id] pairs, the ids points() marks as holes."""
        try:
            pairs = _points(contours)
        except DocumentError:
            raise DocumentError(
                "Give holes as a list of [object id, hole id] pairs"
            ) from None
        objects = sorted({o for o, _ in pairs})
        if action == "shape":
            if delete_enclosed:
                raise DocumentError("delete_enclosed goes with action='fill'")
            return self._edit(
                seen,
                [
                    self._select(objects),
                    {"command": "holes_to_shapes", "contours": pairs},
                ],
            )
        if action != "fill":
            raise DocumentError("action is fill or shape")

        def fill() -> dict:
            payload: dict[str, Any] = {"command": "fill_holes", "contours": pairs}
            if delete_enclosed:
                if len(objects) != 1:
                    raise DocumentError("Delete enclosed shapes with one path's holes")
                found = self.session.holes(
                    {
                        "epoch": self.session.epoch,
                        "revision": self.session.editor.snapshot.revision,
                        "object": objects[0],
                        "holes": [h for _, h in pairs],
                        "find_enclosed": True,
                    }
                )
                payload["delete_objects"] = found["enclosed"]
            return payload

        return self._edit(seen, [self._select(objects), fill])

    def tool_convert(self, seen: Any, ids: Any, to: str = "either") -> Reply:
        commands = {
            "line": "fill_to_line",
            "fill": "line_to_fill",
            "either": "convert_lines",
            "path": "detach",
        }
        if to not in commands:
            raise DocumentError("to is line, fill, either or path")
        return self._simple(commands[to], seen, ids)

    def tool_delete(
        self,
        seen: Any,
        ids: Any = None,
        points: Any = None,
        region: Any = None,
        contours: bool = False,
        cut: bool = False,
        detach: bool = False,
    ) -> Reply:
        """Delete the objects *ids*; the *points* (or, with *contours*, the
        contours they are on); or the contours inside *region*, of the paths
        and shapes *ids* or every unlocked one painting there."""
        if points is not None and region is not None:
            raise DocumentError("Give points or a region, not both")
        if (cut or detach) and region is None:
            raise DocumentError("cut and detach go with a region")
        if region is not None:
            return self._in_region(seen, region, ids, cut, delete=True, detach=detach)
        if points is not None:
            if ids is not None:
                raise DocumentError(
                    "Give ids (objects) or points ([object id, node id] pairs), "
                    "not both"
                )
            command = "delete_contour" if contours else "delete_node"
            return self._on_points(command, seen, points)
        if ids is None:
            raise DocumentError(
                "Give ids (objects to delete), points ([object id, node id] "
                "pairs), or a region to delete the contours inside it"
            )
        if contours:
            raise DocumentError(
                "contours goes with points: the contours those points are on"
            )
        return self._simple("delete", seen, ids)

    def _spot(self, d: str) -> dict[str, Any] | None:
        """Where a new path drawn from *d* belongs: in the group of what is
        drawn under it, just above that, as {parent, index, above}."""
        try:
            geometry = parse_path(str(d))
        except (DocumentError, ValueError):
            return None
        points = [n.endpoint for s in geometry.subpaths for n in s.nodes]
        if not points:
            return None
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
        centre = ShapelyPoint((left + right) / 2, (top + bottom) / 2)
        found = self._covering(centre)
        if not found and right > left and bottom > top:
            found = self._covering(
                Polygon([(left, top), (right, top), (right, bottom), (left, bottom)])
            )
        document = self.session.editor.snapshot.document
        for oid, _ in found:
            ancestry = document.ancestry(oid)
            groups = ancestry[1:-1]
            # A move into the group must keep the path's look and be allowed.
            if (
                any(
                    EditKind.STRUCTURE in a.locks
                    or float(a.get("opacity", "1") or "1") != 1
                    or a.get("clip-path", "none") != "none"
                    for a in groups
                )
                or EditKind.STRUCTURE in ancestry[0].locks
            ):
                continue
            parent = ancestry[-2]
            index = next(i for i, c in enumerate(parent.children) if c.id == oid)
            return {"parent": parent.id, "index": index + 1, "above": oid}
        return None

    def tool_add_path(
        self,
        seen: Any,
        d: str,
        fill: str | None = None,
        stroke: str | None = None,
        stroke_width: float | None = None,
        parent: str | None = None,
        index: int | None = None,
        name: str | None = None,
    ) -> Reply:
        placed: dict[str, Any] = {}

        def new() -> str:
            (oid,) = self.session.editor.snapshot.selection.object_ids
            return oid

        def where() -> None:
            # Before the path exists: what is drawn under it.
            if parent is None and index is None:
                spot = self._spot(d)
                if spot is not None:
                    placed.update(spot)

        def paint() -> dict | None:
            changes = {
                key: str(value)
                for key, value in (("fill", fill), ("stroke", stroke))
                if value is not None
            }
            return {"command": "paint", "changes": changes} if changes else None

        def place() -> dict | None:
            document = self.session.editor.snapshot.document
            root = document.root
            if parent is None and index is None:
                if not placed:
                    placed.update(
                        parent=root.id,
                        index=len(root.children) - 1,
                        why="nothing is drawn under it, so it went in front at "
                        "the top level",
                    )
                    return None
                placed["why"] = (
                    f"it lies over {placed['above']}, so it went just above it, "
                    "in its group; give parent and index to put it elsewhere"
                )
                target, at = placed["parent"], placed["index"]
                if target == root.id and at >= len(root.children) - 1:
                    return None
            else:
                target = parent or root.id
                count = len(document.element(target).children)
                at = count if index is None else index
                placed.update(parent=target, index=at, why="as given")
            return {
                "command": "move_objects",
                "objects": [new()],
                "parent": target,
                "index": at,
            }

        def rename() -> dict | None:
            return (
                None
                if name is None
                else {"command": "rename", "object": new(), "name": name}
            )

        reply = self._edit(
            seen,
            [
                where,
                {
                    "command": "add_path",
                    "d": d,
                    "stroke_width": 2 if stroke_width is None else stroke_width,
                },
                paint,
                place,
                rename,
            ],
            "Draw path",
        )
        if placed.get("why") == "as given":
            placed.pop("above", None)
        reply.data["placed"] = placed
        return reply

    def tool_set_points(
        self, seen: Any, changes: Any, coords: str = "document"
    ) -> Reply:
        if not isinstance(changes, dict) or not changes:
            raise DocumentError(
                "changes is {object id: {node id: [values]}}, the values as "
                "points() lists them"
            )
        if coords not in {"document", "local"}:
            raise DocumentError("coords is document or local")
        points = [[str(o), str(n)] for o, nodes in changes.items() for n in nodes]

        def move() -> dict:
            document = self.session.editor.snapshot.document
            local: dict[str, dict[str, list[float]]] = {}
            for oid, nodes in changes.items():
                if not isinstance(nodes, dict):
                    raise DocumentError("Give each path's nodes as {node id: values}")
                inverse = (
                    inverse_matrix(object_matrix(document, str(oid)))
                    if coords == "document"
                    else IDENTITY
                )
                local[str(oid)] = {}
                for nid, values in nodes.items():
                    if not isinstance(values, list | tuple) or len(values) not in {
                        2,
                        6,
                    }:
                        raise DocumentError(
                            "Node values are [x, y], or [c1x, c1y, c2x, c2y, x, y]"
                        )
                    numbers = [_finite(v, "A node value") for v in values]
                    local[str(oid)][str(nid)] = _map_values(numbers, inverse)
            return {"command": "move_nodes", "changes": local}

        return self._edit(seen, [self._select_points(points), move])

    def _on_points(self, command: str, seen: Any, points: Any, **extra: Any) -> Reply:
        pairs = _points(points)
        return self._edit(
            seen,
            [
                self._select_points(pairs),
                {"command": command, "points": pairs, **extra},
            ],
        )

    def tool_point_style(
        self,
        seen: Any,
        points: Any,
        handles: int | None = None,
        pinned: bool | None = None,
    ) -> Reply:
        """Give *points* 0, 1 or 2 curve handles, and pin or unpin them."""
        pairs = _points(points)
        if handles is None and pinned is None:
            raise DocumentError("Give handles (0, 1 or 2) or pinned")
        if handles is not None and handles not in {0, 1, 2}:
            raise DocumentError("handles is 0, 1 or 2")
        steps: list[Step] = [self._select_points(pairs)]
        pin = (
            None
            if pinned is None
            else {"command": "pin", "points": pairs, "pinned": bool(pinned)}
        )
        # Unpin before changing the handles, pin after.
        if pin is not None and not pinned:
            steps.append(pin)
        if handles is not None:
            steps.append({"command": "node_handles", "points": pairs, "count": handles})
        if pin is not None and pinned:
            steps.append(pin)
        both = handles is not None and pinned is not None
        return self._edit(seen, steps, "Point style" if both else None)

    def _segment_among(self, pairs: list[list[str]]) -> bool:
        """Whether two of the points are the ends of one segment, as the
        editor's Break tells deleting a segment from breaking at points."""
        document = self.session.editor.snapshot.document
        chosen: dict[str, set[str]] = {}
        for oid, nid in pairs:
            chosen.setdefault(oid, set()).add(nid)
        for oid, nids in chosen.items():
            try:
                subpaths = document.geometry_for(oid).subpaths
            except DocumentError:
                continue
            for subpath in subpaths:
                ids = [n.id for n in subpath.nodes]
                # A closed contour ending on its moveto shows one point for
                # the two nodes: the last stands for both.
                twins = (
                    subpath.closed
                    and len(ids) > 2
                    and subpath.nodes[0].endpoint == subpath.nodes[-1].endpoint
                )
                shown = ids[1:] if twins else ids
                among = {ids[-1] if twins and n == ids[0] else n for n in nids}
                ends = list(itertools.pairwise(shown))
                if subpath.closed and len(shown) > 1:
                    ends.append((shown[-1], shown[0]))
                if any(a != b and a in among and b in among for a, b in ends):
                    return True
        return False

    def tool_break_points(self, seen: Any, points: Any) -> Reply:
        """Cut lines or closed contours open at points; given the two points
        at a segment's ends, delete that segment instead, as Break does."""
        pairs = _points(points)
        with self.session.lock:
            segment = self._segment_among(pairs)
        command = "delete_segment" if segment else "break_points"
        reply = self._on_points(command, seen, pairs)
        reply.data["broke"] = "segment deleted" if segment else "at the points"
        return reply

    def tool_split_edge(self, seen: Any, points: Any) -> Reply:
        return self._on_points("split", seen, points)

    def _region_paths(
        self, polygon: list[tuple[float, float]], ids: Any, detach: bool
    ) -> list[str]:
        """What a region edit acts on: *ids*, or every drawn path, basic shape
        or (with *detach*) instance that paints inside the region and whose
        geometry and structure are not locked. An instance draws shared
        geometry, so without *detach* it is refused rather than left out."""
        document = self.session.editor.snapshot.document
        if ids is not None:
            found = _targets(ids)
        else:
            found = [
                oid
                for oid, _ in self._covering(Polygon(polygon))
                if document.element(oid).tag in REGION_TAGS
                and not any(
                    a.locks & {EditKind.GEOMETRY, EditKind.STRUCTURE}
                    for a in document.ancestry(oid)
                )
            ]
            if not found:
                raise DocumentError(
                    "No unlocked path or shape paints inside the region; "
                    "pick(x, y) or describe(region=...) shows what is there"
                )
        instances = [oid for oid in found if document.element(oid).tag == "use"]
        if instances and not detach:
            raise DocumentError(
                f"{', '.join(instances[:10])} "
                + ("is an instance" if len(instances) == 1 else "are instances")
                + " (use) of shared geometry. Pass detach=true to give "
                + ("it a path" if len(instances) == 1 else "them paths")
                + f" of {'its' if len(instances) == 1 else 'their'} own first, "
                f'or convert(ids={instances[:10]}, to="path"), or name the '
                "paths to act on in ids"
            )
        return found

    def _in_region(
        self,
        seen: Any,
        region: Any,
        ids: Any,
        cut: bool,
        delete: bool,
        group: bool = False,
        detach: bool = False,
    ) -> Reply:
        if not isinstance(region, list | tuple):
            raise DocumentError("region is [x, y, w, h] or a polygon [[x, y], ...]")
        polygon = region_polygon(region)
        session = self.session
        targets: list[str] = []
        before: set[str] = set()
        pieces: list[dict[str, Any]] = []

        def document() -> Document:
            return session.editor.snapshot.document

        def choose() -> dict:
            targets[:] = self._region_paths(polygon, ids, detach)
            before.update(e.id for e in document().elements())
            return self._select(targets)

        def instances() -> list[str]:
            return [t for t in targets if document().element(t).tag == "use"]

        def taken() -> dict | None:
            """The pieces the region took, each with the path it came from
            (the one just below it), before any grouping moves them."""
            for element in document().elements():
                if element.id in before or element.tag != "path":
                    continue
                siblings = document().ancestry(element.id)[-2].children
                at = next(i for i, c in enumerate(siblings) if c.id == element.id)
                pieces.append(
                    {
                        "path": element.id,
                        "from": siblings[at - 1].id if at else None,
                        "contours": len(document().geometry_for(element.id).subpaths),
                    }
                )
            return None

        def gather() -> dict | None:
            """Move the pieces together, just above the path the frontmost
            came from, to group them there."""
            if not pieces:
                return None
            front = pieces[-1]
            parent = document().ancestry(front["path"])[-2]
            moving = {p["path"] for p in pieces}
            staying = [c.id for c in parent.children if c.id not in moving]
            index = staying.index(front["from"]) + 1 if front["from"] else 0
            return {
                "command": "move_objects",
                "objects": [p["path"] for p in pieces],
                "parent": parent.id,
                "index": index,
            }

        steps: list[Step] = [
            choose,
            # Instances get paths of their own, keeping their ids.
            lambda: self._select(instances()) if instances() else None,
            lambda: {"command": "detach"} if instances() else None,
            lambda: self._select(targets),
            {
                "command": "extract",
                "region": [list(p) for p in polygon],
                "cut": bool(cut),
                "delete": delete,
            },
            taken,
        ]
        if group:
            steps += [
                gather,
                lambda: self._select([p["path"] for p in pieces]),
                {"command": "group"},
            ]
        reply = self._edit(
            seen,
            steps,
            label=("Extract region into a group" if group else None),
        )
        if not delete:
            reply.data["extracted"] = pieces
            if group and pieces:
                reply.data["group"] = document().ancestry(pieces[0]["path"])[-2].id
        return reply

    def tool_extract(
        self,
        seen: Any,
        region: Any,
        ids: Any = None,
        cut: bool = True,
        group: bool = False,
        detach: bool = False,
    ) -> Reply:
        return self._in_region(
            seen, region, ids, cut, delete=False, group=bool(group), detach=detach
        )

    def tool_knife(
        self,
        seen: Any,
        start: Any,
        end: Any,
        ids: Any = None,
        within: str | None = None,
    ) -> Reply:
        payload: dict[str, Any] = {"command": "knife", "start": start, "end": end}
        if within is not None:
            payload["within"] = within
        return self._edit(
            seen,
            [{"command": "select", "objects": _ids(ids) or []}, payload],
        )

    def tool_redraw_outline(
        self,
        seen: Any,
        id: str,  # noqa: A002
        points: Any,
        long_way: bool = False,
        pixel: float = 1.0,
    ) -> Reply:
        return self._edit(
            seen,
            [
                self._select([id]),
                {
                    "command": "redraw_outline",
                    "object": id,
                    "points": points,
                    "long_way": bool(long_way),
                    "pixel": pixel,
                },
            ],
        )

    def tool_set_reference(
        self, seen: Any, name: str | None = None, data_url: str | None = None
    ) -> Reply:
        reference = (
            None
            if data_url is None
            else {"name": name or "Reference", "data_url": data_url, "opacity": 0.5}
        )
        reply = self._edit(seen, [{"command": "reference", "reference": reference}])
        reply.data["step"] = "Load reference" if reference else "Remove reference"
        return reply

    def _history(self, seen: Any, command: str, steps: int) -> Reply:
        if type(steps) is not int or steps < 1:
            raise DocumentError("steps is a whole number from 1")
        with self.session.lock, self._own_selection():
            self._check_seen(seen)
            editor = self.session.editor
            stack = editor.undo_labels if command == "undo" else editor.redo_labels
            if not stack:
                raise DocumentError(f"Nothing to {command}")
            revision = editor.snapshot.revision
            labels = []
            for _ in range(min(steps, len(stack))):
                labels.append(
                    editor.undo_labels[-1]
                    if command == "undo"
                    else editor.redo_labels[0]
                )
                self.session.action({"command": command, **self._where()})
            self._settle(revision)
            return Reply(
                {
                    **self._where(),
                    "changed": True,
                    "undone" if command == "undo" else "redone": labels,
                    "step": f"{command.capitalize()} {labels[0].lower()}",
                }
            )

    def tool_undo(self, seen: Any, steps: int = 1) -> Reply:
        return self._history(seen, "undo", steps)

    def tool_redo(self, seen: Any, steps: int = 1) -> Reply:
        return self._history(seen, "redo", steps)

    # Operations ---------------------------------------------------------

    def _start(
        self,
        seen: Any,
        ids: list[str],
        action: str,
        method: str,
        permissions: dict[str, bool],
        settings: Any,
        scope: str | None = None,
        budget: dict | None = None,
        bounds: bool = False,
    ) -> Reply:
        if settings is not None and not isinstance(settings, dict):
            raise DocumentError("settings is an object of setting names and values")
        session = self.session
        with session.lock, self._own_selection():
            self._check_seen(seen)
            # The job works on the selection it starts with: the agent's own,
            # of its targets, which the person's then replaces again.
            session.action({**self._select(ids), **self._where()})
            payload: dict[str, Any] = {
                "command": "start",
                **self._where(),
                "action": action,
                "method": method,
                "permissions": permissions,
                "settings": settings or {},
            }
            if scope is not None:
                payload["scope"] = scope
            if budget:
                payload["budget"] = budget
            if bounds:
                x, y, w, h = self._selected_bounds()
                pad = max(8.0, max(w, h) * 0.06)
                payload["bounds"] = [x - pad, y - pad, w + 2 * pad, h + 2 * pad]
            job = session.operation(payload)
        return self._job_reply(job)

    def _job_reply(self, job: dict) -> Reply:
        """A job's state, its recommended result's previews as images."""
        images: list[tuple[str, bytes]] = []
        data = {k: v for k, v in job.items() if v is not None}
        result = data.get("result")
        if result:
            previews = result.pop("previews", None) or {}
            for name, url in previews.items():
                if isinstance(url, str) and url.startswith("data:image/png;base64,"):
                    images.append((name, base64.b64decode(url.split(",", 1)[1])))
            data["previews"] = [name for name, _ in images]
            data["alternatives"] = [
                {k: v for k, v in a.items() if k != "previews"}
                for a in data.get("alternatives", [])
            ]
        if data.get("status") == "ready":
            data["next"] = (
                'job(id, action="apply") keeps the recommended result as one '
                "undo step"
                + (", choice=n an alternative" if data["alternatives"] else "")
                + '; job(id, action="discard") drops it'
            )
        return Reply(data, images)

    def tool_generate(
        self,
        seen: Any,
        method: str = "cel",
        settings: Any = None,
        group: str | None = None,
    ) -> Reply:
        """Over the whole drawing, or into the area of the group *group*."""
        return self._start(
            seen,
            [] if group is None else [str(group)],
            "generate",
            method,
            {"structure": True},
            settings,
            scope="drawing" if group is None else "selection",
        )

    def tool_tidy(
        self,
        seen: Any,
        ids: Any = None,
        settings: Any = None,
        rounds: int | None = None,
        region: Any = None,
    ) -> Reply:
        """Tidy the paths *ids*; within *region*, only the points inside it,
        of *ids* or every unlocked path painting there."""
        from vectrify.operations.methods.nodes import SETTINGS

        if region is None and ids is None:
            raise DocumentError(
                "Give ids (paths to tidy), or a region to tidy what is inside it"
            )
        chosen = {k: v.default for k, v in SETTINGS.items()}
        if isinstance(settings, dict):
            chosen.update(settings)
        if region is not None:
            if not isinstance(region, list | tuple):
                raise DocumentError("region is [x, y, w, h] or a polygon [[x, y], ...]")
            region_polygon(region)
            settings = {**(settings if isinstance(settings, dict) else {})}
            settings["region"] = [
                list(p) if isinstance(p, list | tuple) else p for p in region
            ]
        structure = bool((chosen["snap"] and chosen["detail"]) or chosen["simplify"])
        return self._start(
            seen,
            _targets(ids) if ids is not None or region is None else [],
            "improve",
            "nodes",
            {"geometry": True, "structure": structure, "paint": True},
            settings,
            scope="selection",
            budget={"steps": rounds} if rounds is not None else None,
        )

    def tool_fit_colours(
        self,
        seen: Any,
        ids: Any,
        fill: str = "flat",
        passes: int | None = None,
        resolution: int | None = None,
    ) -> Reply:
        settings: dict[str, Any] = {"fill": fill}
        if passes is not None:
            settings["passes"] = passes
        if resolution is not None:
            settings["resolution"] = resolution
        return self._start(
            seen, _targets(ids), "improve", "colours", {"paint": True}, settings
        )

    def tool_snap_edges(self, seen: Any, ids: Any, tolerance: float = 1.0) -> Reply:
        return self._start(
            seen,
            _targets(ids),
            "snap",
            "edges",
            {"geometry": True, "structure": True},
            {"tolerance": tolerance},
            bounds=True,
        )

    def tool_cleanup(self, seen: Any, ids: Any) -> Reply:
        return self._start(
            seen,
            _targets(ids),
            "simplify",
            "cleanup",
            {"geometry": True, "structure": True},
            None,
            bounds=True,
        )

    def _job(self, command: str, job: str, **extra: Any) -> dict:
        with self.session.lock:
            return self.session.operation({"command": command, "job": job, **extra})

    def tool_job(
        self,
        seen: Any,
        id: str,  # noqa: A002
        action: str = "status",
        wait_seconds: float = 0,
        choice: int = 0,
    ) -> Reply:
        """A job's status (waiting up to *wait_seconds* for it), or apply
        its result (or alternative *choice*), discard it or stop it."""
        if action == "status":
            return self._job_status(id, wait_seconds)
        if action == "apply":
            return self._apply(seen, id, choice)
        if action == "discard":
            return Reply(self._job("discard", id))
        if action == "stop":
            return self._job_reply(self._job("stop", id))
        raise DocumentError("action is status, apply, discard or stop")

    def _job_status(self, id: str, wait_seconds: Any) -> Reply:  # noqa: A002
        wait = max(0.0, min(MAX_WAIT, _finite(wait_seconds, "wait_seconds")))
        deadline = time.monotonic() + wait
        state = self._job("status", id)
        # The session is not held while waiting, so the person keeps editing.
        while state["status"] == "running" and time.monotonic() < deadline:
            time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
            state = self._job("status", id)
        if state["status"] != "running":
            state = self._job("status", id, preview=True)
        return self._job_reply(state)

    def _apply(self, seen: Any, id: str, choice: int) -> Reply:  # noqa: A002
        session = self.session
        with session.lock, self._own_selection() as touched:
            self._check_seen(seen)
            editor = session.editor
            since = len(editor.undo_entries)
            editor.label_prefix = AGENT_PREFIX
            try:
                session.operation({"command": "apply", "job": id, "choice": choice})
            finally:
                editor.label_prefix = ""
            touched.update(editor.snapshot.selection.object_ids)
            data: dict[str, Any] = {
                **self._where(),
                "changed": True,
                "result": self._selection(),
            }
            if len(editor.undo_entries) > since:
                data["step"] = editor.undo_labels[-1]
            return Reply(data)


# The live channel ------------------------------------------------------

# Where the editor hosts the MCP server while a window allows agents, unless
# it hosted it elsewhere before (or --mcp-port says), and how many ports after
# it to try when it is taken.
MCP_PORT = 8770
MCP_PORTS = 20
OFF = (
    "Agent editing is off in this editor. Turn on 'Allow agents to edit' "
    "('Agents') in its footer."
)


def state_dir() -> Path:
    """Where the editor keeps what agents need to find it."""
    state = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(state) / "vectrify"


def discovery_file() -> Path:
    """Where a running editor tells agents how to reach it."""
    return state_dir() / "editor.json"


def token_file() -> Path:
    """The agent token, kept across runs so a client is added only once."""
    return state_dir() / "agent-token"


def port_file() -> Path:
    """The port the MCP server was last hosted on, preferred next time so the
    URL clients were given keeps working."""
    return state_dir() / "mcp-port"


def _write_private(path: Path, text: str) -> None:
    """Write *path* owner-only (0600 in a 0700 directory), atomically."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(".tmp")
    with contextlib.suppress(FileNotFoundError):
        temporary.unlink()
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as file:
        file.write(text)
    temporary.replace(path)


def agent_token(regenerate: bool = False) -> str:
    """The stable agent token: read, or made once (or anew) and kept."""
    path = token_file()
    if not regenerate:
        with contextlib.suppress(OSError, UnicodeDecodeError):
            token = path.read_text().strip()
            if len(token) >= 32:
                return token
    token = secrets.token_urlsafe(32)
    _write_private(path, token + "\n")
    return token


def remembered_port() -> int | None:
    """The port the MCP server was last hosted on, if any."""
    with contextlib.suppress(OSError, ValueError):
        port = int(port_file().read_text().strip())
        if 0 < port < 65536:
            return port
    return None


def mcp_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/mcp"


def local_host(headers: Any, port: int) -> bool:
    """Whether a request was addressed to this machine, not via a renamed host."""
    return headers.get("Host", "") in {f"127.0.0.1:{port}", f"localhost:{port}"}


def claude_command(url: str, token: str) -> str:
    """The Claude Code command that adds the editor's MCP server."""
    return (
        f"claude mcp add --transport http --scope user vectrify {url} "
        f'--header "Authorization: Bearer {token}"'
    )


def mcp_executable() -> str:
    """This install's ``vectrify-mcp``: beside the running Python, else on PATH."""
    name = "vectrify-mcp.exe" if os.name == "nt" else "vectrify-mcp"
    beside = Path(sys.executable).parent / name
    if beside.is_file():
        return str(beside.absolute())
    found = shutil.which("vectrify-mcp")
    return str(Path(found).absolute()) if found else "vectrify-mcp"


def _shell_word(word: str) -> str:
    if os.name == "nt":
        return f'"{word}"' if any(c in word for c in ' \t"&()^') else word
    return shlex.quote(word)


def codex_snippet(command: str) -> str:
    """The ``config.toml`` table that adds ``vectrify-mcp`` to Codex.

    Starting it imports the vision stack, so it gets more than Codex's default
    10 s. A JSON string is also a valid TOML basic string.
    """
    return (
        "[mcp_servers.vectrify]\n"
        f"command = {json.dumps(command)}\n"
        "startup_timeout_sec = 30\n"
    )


def codex_command(command: str) -> str:
    """The Codex CLI command that adds the same server (default timeout)."""
    return f"codex mcp add vectrify -- {_shell_word(command)}"


def claude_desktop_snippet(command: str) -> str:
    """The ``claude_desktop_config.json`` entry that adds ``vectrify-mcp``."""
    return json.dumps({"mcpServers": {"vectrify": {"command": command}}}, indent=2)


def claude_desktop_config() -> str:
    """Where Claude Desktop keeps its MCP servers on this OS."""
    if sys.platform == "darwin":
        return "~/Library/Application Support/Claude/claude_desktop_config.json"
    if os.name == "nt":
        return r"%APPDATA%\Claude\claude_desktop_config.json"
    # Claude Desktop has no official Linux build; unofficial ones read this.
    return "~/.config/Claude/claude_desktop_config.json"


def app_setup() -> dict[str, str]:
    """What other agent apps need to start ``vectrify-mcp``, which then
    attaches to the window with Agents on through ``editor.json``: no token."""
    command = mcp_executable()
    codex_home = os.environ.get("CODEX_HOME")
    return {
        "executable": command,
        "codex_config": str(Path(codex_home) / "config.toml")
        if codex_home
        else "~/.codex/config.toml",
        "codex": codex_snippet(command),
        "codex_command": codex_command(command),
        "claude_desktop": claude_desktop_snippet(command),
        "claude_desktop_config": claude_desktop_config(),
    }


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
