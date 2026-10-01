"""What an agent may do to an editor session, and the channel it reaches it by.

``Agent`` answers an agent's calls (``describe``, ``render``, ``paint``,
``generate``…) on one ``Session``. Every edit goes through ``Session.action``
or ``Session.operation``, so selection scope, locks, pins, permissions and
revision checks hold exactly as for a person, each call is one undo step, and
the history labels it "Agent: …". The MCP server (``vectrify.mcp``) holds an
``Agent`` in process for a file it opened, or reaches the one of a running
editor through ``AgentChannel``: HTTP on localhost with a token, opened when
the window allows agents to edit.

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
import json
import math
import os
import secrets
import threading
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cairosvg
import numpy as np
from PIL import Image

from vectrify.document import (
    Document,
    DocumentError,
    HitIndex,
    StaleRevisionError,
    export_svg,
)
from vectrify.document.holes import document_hole_shape, find_holes
from vectrify.image_utils import on_white
from vectrify.operations.generate import frame

# What the history puts before an agent's edits.
AGENT_PREFIX = "Agent: "
# Rendered images: the longest side by default, and at most.
DEFAULT_SIDE = 1024
MAX_SIDE = 2048
MIN_SIDE = 16
# How many objects describe() lists at a time by default, and at most.
PAGE_SIZE = 100
MAX_PAGE = 500
# How long job_status() may wait for a job.
MAX_WAIT = 120.0
# How long after its last call an agent still shows as connected.
CONNECTED = 120.0

# Each agent tool that edits, and the editor commands it sends. The MCP
# server has one tool of the same name for each.
EDITS: dict[str, tuple[str, ...]] = {
    "select": ("select",),
    "paint": ("paint",),
    "rename": ("rename",),
    "locks": ("locks",),
    "move": ("move",),
    "resize": ("resize", "move"),
    "reorder": ("reorder",),
    "move_into": ("move_objects",),
    "group": ("group",),
    "ungroup": ("ungroup",),
    "join": ("join_paths",),
    "join_ends": ("join_ends",),
    "join_points": ("join_two_ends",),
    "split_parts": ("split_disconnected",),
    "cut_hole": ("cut_hole",),
    "fill_holes": ("fill_holes",),
    "holes_to_shapes": ("holes_to_shapes",),
    "detach": ("detach",),
    "convert": ("fill_to_line", "line_to_fill", "convert_lines"),
    "delete": ("delete",),
    "add_path": ("add_path", "paint", "move_objects", "rename"),
    "set_points": ("move_nodes",),
    "handles": ("node_handles",),
    "pin": ("pin",),
    "break_points": ("break_points",),
    "delete_segment": ("delete_segment",),
    "split_edge": ("split",),
    "delete_points": ("delete_node",),
    "delete_contours": ("delete_contour",),
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
    "to_front": "Internal: reorder(to='front') arrives as reorder and is "
    "renamed to this inside the session.",
    "to_back": "Internal: reorder(to='back') arrives as reorder and is "
    "renamed to this inside the session.",
}

# Session.operation commands no agent tool sends, and why.
OPERATIONS_LEFT_OUT = {
    "check": "The dialog's probe of whether a start would be accepted; an "
    "agent starts the job and reads the refusal instead.",
}

# Calls that only look, and so need no refresh of the window.
LOOKS = frozenset(
    {
        "hello",
        "describe",
        "render",
        "reference",
        "compare",
        "get_svg",
        "points",
        "holes",
        "history",
        "job_status",
        "export",
    }
)

Box = tuple[float, float, float, float]
# One step of an edit: a payload, one made once the steps before are done,
# or None to skip it.
Step = dict | Callable[[], dict | None] | None


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
        return f"No such object, point or field: {exc.args[0] if exc.args else ''}"
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


def _points(value: Any) -> list[list[str]]:
    if (
        not isinstance(value, list | tuple)
        or not value
        or any(not isinstance(p, list | tuple) or len(p) != 2 for p in value)
    ):
        raise DocumentError("Give points as a list of [object id, node id] pairs")
    return [[str(o), str(n)] for o, n in value]


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

    # The call boundary --------------------------------------------------

    def call(self, tool: str, args: dict[str, Any] | None = None) -> Reply:
        """Answer one call; refusals raise DocumentError with the reason."""
        args = dict(args or {})
        handler = getattr(self, f"tool_{tool}", None)
        if handler is None:
            raise DocumentError(f"Unknown agent call: {tool}")
        seen = args.pop("seen", None)
        try:
            reply = handler(seen, **args)
        except TypeError as exc:
            # A wrong or missing argument, said in the tool's own terms.
            raise DocumentError(f"{tool}: {exc}") from None
        finally:
            self.last_time = time.monotonic()
        if tool not in LOOKS:
            self.changes += 1
            self.last_action = reply.data.get("step") or tool.replace("_", " ")
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
        with session.lock:
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
            except Exception as exc:
                editor.rollback(since)
                if editor.snapshot.revision != revision:
                    raise RefusedError(reason(exc), self._where()) from exc
                raise
            finally:
                editor.label_prefix = ""
            if label is not None:
                label = AGENT_PREFIX + label
            editor.squash(since, label)
            snapshot = editor.snapshot
            after = {e.id for e in snapshot.document.elements()}
            data: dict[str, Any] = {
                **self._where(),
                "changed": snapshot.revision != revision,
                "selection": self._selection(),
            }
            if len(editor.undo_entries) > since:
                data["step"] = editor.undo_labels[-1]
            created, removed = sorted(after - before), sorted(before - after)
            if created:
                data["created"] = created[:200]
            if removed:
                data["removed"] = removed[:200]
            return Reply(data)

    @staticmethod
    def _select(ids: list[str] | None) -> dict | None:
        """The step selecting *ids*, or None to keep the selection."""
        return None if ids is None else {"command": "select", "objects": ids}

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
        reference = self.session.reference
        key = (
            kind,
            *self._key(),
            region,
            size,
            hash(reference["data_url"]) if reference and kind == "reference" else 0,
        )
        image = self._renders.get(key)
        if image is None:
            with self.session.lock:
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
        else:
            self._renders.move_to_end(key)
        return image

    def tool_hello(self, _seen: Any) -> Reply:
        return Reply({**self._where(), "name": self.session.name})

    def tool_describe(
        self,
        _seen: Any,
        page: int = 0,
        page_size: int = PAGE_SIZE,
        within: str | None = None,
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
            chosen = rows[page * page_size : (page + 1) * page_size]
            hits = self._hits() if chosen else None
            objects = []
            for row in chosen:
                attributes = row["attributes"]
                bounds = hits.bounds(frozenset({row["id"]})) if hits else None
                item: dict[str, Any] = {
                    "id": row["id"],
                    "label": row["label"],
                    "tag": row["tag"],
                    "parent": row["parent"],
                    "depth": row["depth"],
                    "paint": {
                        k: attributes[k]
                        for k in (
                            "fill",
                            "stroke",
                            "stroke-width",
                            "opacity",
                            "fill-opacity",
                            "stroke-opacity",
                        )
                        if k in attributes
                    },
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
                objects.append(item)
            picture = self._reference_image()
            undo, redo = session.editor.undo_labels, session.editor.redo_labels
            return Reply(
                {
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
            )

    def tool_render(
        self,
        _seen: Any,
        region: Any = None,
        overlay: str = "none",
        max_side: int | None = None,
    ) -> Reply:
        box = self._region(region)
        side = self._side(max_side)
        if overlay not in {"none", "side", "over"}:
            raise DocumentError("overlay is none, side (by side) or over")
        if overlay == "side":
            # Two images side by side share the pixel budget.
            size = _size(box, side // 2 if box[2] >= box[3] else side)
        else:
            size = _size(box, side)
        drawing = self._cached("drawing", box, size)
        if overlay == "none":
            image = drawing
        else:
            reference = self._cached("reference", box, size)
            if overlay == "over":
                image = Image.blend(drawing, reference, 0.5)
            else:
                gap = 4
                image = Image.new("RGB", (2 * size[0] + gap, size[1]), "#808080")
                image.paste(drawing, (0, 0))
                image.paste(reference, (size[0] + gap, 0))
        return Reply(
            {**self._where(), "region": list(box), "pixels": list(image.size)},
            [("render", _png(image))],
        )

    def tool_reference(
        self, _seen: Any, region: Any = None, max_side: int | None = None
    ) -> Reply:
        box = self._region(region)
        image = self._cached("reference", box, _size(box, self._side(max_side)))
        return Reply(
            {**self._where(), "region": list(box), "pixels": list(image.size)},
            [("reference", _png(image))],
        )

    def tool_compare(
        self, _seen: Any, region: Any = None, max_side: int | None = None
    ) -> Reply:
        """The mean squared error against the reference, a heat map of where
        they differ, and the worst cells of a 4 by 4 grid over the region."""
        box = self._region(region)
        size = _size(box, self._side(max_side if max_side is not None else 512))
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
        return Reply(
            {
                **self._where(),
                "region": list(box),
                "mse": round(float(per_pixel.mean()), 6),
                "worst_cells": cells[:4],
            },
            [("difference", _png(heat_map(np.sqrt(per_pixel))))],
        )

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

    def tool_points(self, _seen: Any, id: str) -> Reply:  # noqa: A002
        with self.session.lock:
            geometry = self.session.geometries([id])["geometries"][str(id)]
            return Reply({**self._where(), "object": id, "geometry": geometry})

    def tool_holes(self, _seen: Any, id: str) -> Reply:  # noqa: A002
        with self.session.lock:
            document = self.session.editor.snapshot.document
            holes = [
                {
                    "id": h.id,
                    "area": round(document_hole_shape(document, id, (h,)).area, 3),
                    "bounds": [round(v, 3) for v in h.shape.bounds],
                }
                for h in find_holes(document, id)
            ]
            return Reply({**self._where(), "object": id, "holes": holes})

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

    def tool_select(self, seen: Any, objects: Any = None, points: Any = None) -> Reply:
        ids = _ids(objects) or []
        pairs = _points(points) if points else []
        return self._edit(
            seen,
            [
                {
                    "command": "select",
                    "objects": sorted(set(ids) | {o for o, _ in pairs}),
                    "nodes": sorted({n for _, n in pairs}),
                }
            ],
        )

    def tool_paint(
        self,
        seen: Any,
        ids: Any = None,
        fill: str | None = None,
        stroke: str | None = None,
        stroke_width: float | None = None,
        opacity: float | None = None,
        fill_opacity: float | None = None,
        stroke_opacity: float | None = None,
    ) -> Reply:
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
        if not changes:
            raise DocumentError("Give a fill, stroke, stroke width or opacity")
        return self._edit(
            seen, [self._select(_ids(ids)), {"command": "paint", "changes": changes}]
        )

    def tool_rename(self, seen: Any, id: str, name: str) -> Reply:  # noqa: A002
        return self._edit(
            seen,
            [self._select([id]), {"command": "rename", "object": id, "name": name}],
        )

    def tool_locks(self, seen: Any, id: str, locks: Any) -> Reply:  # noqa: A002
        if not isinstance(locks, list | tuple):
            raise DocumentError("Give the locks as a list, empty to unlock")
        return self._edit(
            seen,
            [
                self._select([id]),
                {"command": "locks", "object": id, "locks": [str(v) for v in locks]},
            ],
        )

    def tool_move(self, seen: Any, dx: float, dy: float, ids: Any = None) -> Reply:
        return self._edit(
            seen,
            [
                self._select(_ids(ids)),
                {
                    "command": "move",
                    "dx": _finite(dx, "dx"),
                    "dy": _finite(dy, "dy"),
                },
            ],
        )

    def _selected_bounds(self) -> Box:
        selection = self.session.editor.snapshot.selection.object_ids
        if not selection:
            raise DocumentError("Select an object first")
        bounds = self._hits().bounds(frozenset(selection))
        if bounds is None:
            raise DocumentError("The selection paints nothing to measure")
        return bounds[0], bounds[1], bounds[2] - bounds[0], bounds[3] - bounds[1]

    def tool_resize(
        self,
        seen: Any,
        ids: Any = None,
        scale: Any = None,
        anchor: Any = "center",
        box: Any = None,
    ) -> Reply:
        """Scale the selection about *anchor* (a point, or center, top-left,
        top-right, bottom-left, bottom-right), or fit its painted bounds to
        *box*."""
        if (scale is None) == (box is None):
            raise DocumentError("Give either scale [sx, sy] or box [x, y, w, h]")
        moved: dict[str, float] = {}

        def resize() -> dict:
            bx, by, bw, bh = self._selected_bounds()
            if box is not None:
                x, y, w, h = _box(box, "The box")
                if bw <= 0 or bh <= 0:
                    raise DocumentError("The selection has no area to resize")
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
            if not moved or (abs(moved["dx"]) < 1e-9 and abs(moved["dy"]) < 1e-9):
                return None
            return {"command": "move", "dx": moved["dx"], "dy": moved["dy"]}

        return self._edit(seen, [self._select(_ids(ids)), resize, move], "Resize")

    def tool_reorder(self, seen: Any, to: str, ids: Any = None) -> Reply:
        steps = {"forward": 1, "backward": -1}
        if to in {"front", "back"}:
            payload: dict[str, Any] = {"command": "reorder", "to": to}
        elif to in steps:
            payload = {"command": "reorder", "step": steps[to]}
        else:
            raise DocumentError("to is front, back, forward or backward")
        return self._edit(seen, [self._select(_ids(ids)), payload])

    def tool_move_into(self, seen: Any, ids: Any, parent: str, index: int) -> Reply:
        return self._edit(
            seen,
            [
                {
                    "command": "move_objects",
                    "objects": _ids(ids),
                    "parent": parent,
                    "index": index,
                }
            ],
        )

    def _simple(self, command: str, seen: Any, ids: Any, **extra: Any) -> Reply:
        return self._edit(
            seen, [self._select(_ids(ids)), {"command": command, **extra}]
        )

    def tool_group(self, seen: Any, ids: Any = None) -> Reply:
        return self._simple("group", seen, ids)

    def tool_ungroup(self, seen: Any, ids: Any = None) -> Reply:
        return self._simple("ungroup", seen, ids)

    def tool_join(
        self, seen: Any, ids: Any = None, color_source: str | None = None
    ) -> Reply:
        options: dict[str, Any] = (
            {"colors": "mix"}
            if color_source is None
            else {"colors": "source", "color_source": color_source}
        )
        return self._simple("join_paths", seen, ids, options=options)

    def tool_join_ends(
        self, seen: Any, ids: Any = None, reach: float = 4.0, bridge: str = "curve"
    ) -> Reply:
        if bridge not in {"curve", "line"}:
            raise DocumentError("bridge is curve or line")
        return self._simple("join_ends", seen, ids, reach=reach, bridge=bridge)

    def tool_join_points(self, seen: Any, a: Any, b: Any) -> Reply:
        points = _points([a, b])
        return self._edit(
            seen,
            [
                self._select_points(points),
                {"command": "join_two_ends", "points": points},
            ],
        )

    def tool_split_parts(self, seen: Any, ids: Any = None) -> Reply:
        return self._simple("split_disconnected", seen, ids)

    def tool_cut_hole(self, seen: Any, ids: Any = None) -> Reply:
        return self._simple("cut_hole", seen, ids)

    def _contours(self, contours: Any) -> list[list[str]]:
        try:
            return _points(contours)
        except DocumentError:
            raise DocumentError(
                "Give holes as a list of [object id, hole id] pairs"
            ) from None

    def tool_fill_holes(
        self, seen: Any, contours: Any, delete_enclosed: bool = False
    ) -> Reply:
        pairs = self._contours(contours)
        objects = sorted({o for o, _ in pairs})

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

    def tool_holes_to_shapes(self, seen: Any, contours: Any) -> Reply:
        pairs = self._contours(contours)
        return self._edit(
            seen,
            [
                self._select(sorted({o for o, _ in pairs})),
                {"command": "holes_to_shapes", "contours": pairs},
            ],
        )

    def tool_detach(self, seen: Any, ids: Any = None) -> Reply:
        return self._simple("detach", seen, ids)

    def tool_convert(self, seen: Any, ids: Any = None, to: str = "either") -> Reply:
        commands = {
            "line": "fill_to_line",
            "fill": "line_to_fill",
            "either": "convert_lines",
        }
        if to not in commands:
            raise DocumentError("to is line, fill or either")
        return self._simple(commands[to], seen, ids)

    def tool_delete(self, seen: Any, ids: Any = None) -> Reply:
        return self._simple("delete", seen, ids)

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
        def new() -> str:
            (oid,) = self.session.editor.snapshot.selection.object_ids
            return oid

        def paint() -> dict | None:
            changes = {
                key: str(value)
                for key, value in (("fill", fill), ("stroke", stroke))
                if value is not None
            }
            return {"command": "paint", "changes": changes} if changes else None

        def place() -> dict | None:
            if parent is None and index is None:
                return None
            document = self.session.editor.snapshot.document
            target = parent or document.root.id
            count = len(document.element(target).children)
            return {
                "command": "move_objects",
                "objects": [new()],
                "parent": target,
                "index": count if index is None else index,
            }

        def rename() -> dict | None:
            return (
                None
                if name is None
                else {"command": "rename", "object": new(), "name": name}
            )

        return self._edit(
            seen,
            [
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

    def tool_set_points(self, seen: Any, changes: Any) -> Reply:
        if not isinstance(changes, dict) or not changes:
            raise DocumentError(
                "changes is {object id: {node id: [values]}}, the values as "
                "points() lists them"
            )
        points = [[str(o), str(n)] for o, nodes in changes.items() for n in nodes]
        return self._edit(
            seen,
            [
                self._select_points(points),
                {"command": "move_nodes", "changes": changes},
            ],
        )

    def _on_points(self, command: str, seen: Any, points: Any, **extra: Any) -> Reply:
        pairs = _points(points)
        return self._edit(
            seen,
            [
                self._select_points(pairs),
                {"command": command, "points": pairs, **extra},
            ],
        )

    def tool_handles(self, seen: Any, points: Any, count: int) -> Reply:
        if count not in {0, 1, 2}:
            raise DocumentError("count is 0, 1 or 2 handles")
        return self._on_points("node_handles", seen, points, count=count)

    def tool_pin(self, seen: Any, points: Any, pinned: bool = True) -> Reply:
        return self._on_points("pin", seen, points, pinned=bool(pinned))

    def tool_break_points(self, seen: Any, points: Any) -> Reply:
        return self._on_points("break_points", seen, points)

    def tool_delete_segment(self, seen: Any, points: Any) -> Reply:
        return self._on_points("delete_segment", seen, points)

    def tool_split_edge(self, seen: Any, points: Any) -> Reply:
        return self._on_points("split", seen, points)

    def tool_delete_points(self, seen: Any, points: Any) -> Reply:
        return self._on_points("delete_node", seen, points)

    def tool_delete_contours(self, seen: Any, points: Any) -> Reply:
        return self._on_points("delete_contour", seen, points)

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
        with self.session.lock:
            self._check_seen(seen)
            editor = self.session.editor
            stack = editor.undo_labels if command == "undo" else editor.redo_labels
            if not stack:
                raise DocumentError(f"Nothing to {command}")
            labels = []
            for _ in range(min(steps, len(stack))):
                labels.append(
                    editor.undo_labels[-1]
                    if command == "undo"
                    else editor.redo_labels[0]
                )
                self.session.action({"command": command, **self._where()})
            return Reply(
                {
                    **self._where(),
                    "changed": True,
                    "undone" if command == "undo" else "redone": labels,
                    "step": f"{command.capitalize()} {labels[0].lower()}",
                    "selection": self._selection(),
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
        ids: Any,
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
        with session.lock:
            # Choosing what to work on is a selection, like the person's.
            if ids is not None:
                self._edit(seen, [self._select(_ids(ids))])
            else:
                self._check_seen(seen)
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
                "apply(id) keeps the recommended result as one undo step"
                + (
                    ", apply(id, choice=n) an alternative"
                    if data["alternatives"]
                    else ""
                )
                + "; discard(id) drops it"
            )
        return Reply(data, images)

    def tool_generate(
        self,
        seen: Any,
        method: str = "cel",
        settings: Any = None,
        scope: str = "drawing",
        group: str | None = None,
    ) -> Reply:
        if scope not in {"drawing", "selection"}:
            raise DocumentError("scope is drawing, or selection with one group")
        return self._start(
            seen,
            [group] if group is not None else None,
            "generate",
            method,
            {"structure": True},
            settings,
            scope=scope,
        )

    def tool_tidy(
        self,
        seen: Any,
        ids: Any = None,
        settings: Any = None,
        rounds: int | None = None,
    ) -> Reply:
        from vectrify.operations.methods.nodes import SETTINGS

        chosen = {k: v.default for k, v in SETTINGS.items()}
        if isinstance(settings, dict):
            chosen.update(settings)
        structure = bool((chosen["snap"] and chosen["detail"]) or chosen["simplify"])
        return self._start(
            seen,
            ids,
            "improve",
            "nodes",
            {"geometry": True, "structure": structure},
            settings,
            scope="selection",
            budget={"steps": rounds} if rounds is not None else None,
        )

    def tool_fit_colours(
        self,
        seen: Any,
        ids: Any = None,
        fill: str = "flat",
        passes: int | None = None,
        resolution: int | None = None,
    ) -> Reply:
        settings: dict[str, Any] = {"fill": fill}
        if passes is not None:
            settings["passes"] = passes
        if resolution is not None:
            settings["resolution"] = resolution
        return self._start(seen, ids, "improve", "colours", {"paint": True}, settings)

    def tool_snap_edges(
        self, seen: Any, ids: Any = None, tolerance: float = 1.0
    ) -> Reply:
        return self._start(
            seen,
            ids,
            "snap",
            "edges",
            {"geometry": True, "structure": True},
            {"tolerance": tolerance},
            bounds=True,
        )

    def tool_cleanup(self, seen: Any, ids: Any = None) -> Reply:
        return self._start(
            seen,
            ids,
            "simplify",
            "cleanup",
            {"geometry": True, "structure": True},
            None,
            bounds=True,
        )

    def _job(self, command: str, job: str, **extra: Any) -> dict:
        with self.session.lock:
            return self.session.operation({"command": command, "job": job, **extra})

    def tool_job_status(self, _seen: Any, id: str, wait_seconds: float = 0) -> Reply:  # noqa: A002
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

    def tool_apply(self, seen: Any, id: str, choice: int = 0) -> Reply:  # noqa: A002
        session = self.session
        with session.lock:
            self._check_seen(seen)
            editor = session.editor
            since = len(editor.undo_entries)
            editor.label_prefix = AGENT_PREFIX
            try:
                session.operation({"command": "apply", "job": id, "choice": choice})
            finally:
                editor.label_prefix = ""
            data: dict[str, Any] = {
                **self._where(),
                "changed": True,
                "selection": self._selection(),
            }
            if len(editor.undo_entries) > since:
                data["step"] = editor.undo_labels[-1]
            return Reply(data)

    def tool_discard(self, _seen: Any, id: str) -> Reply:  # noqa: A002
        return Reply(self._job("discard", id))

    def tool_stop(self, _seen: Any, id: str) -> Reply:  # noqa: A002
        return self._job_reply(self._job("stop", id))


# The live channel ------------------------------------------------------


def discovery_file() -> Path:
    """Where a running editor tells agents how to reach it."""
    state = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(state) / "vectrify" / "editor.json"


def local_host(headers: Any, port: int) -> bool:
    """Whether a request was addressed to this machine, not via a renamed host."""
    return headers.get("Host", "") in {f"127.0.0.1:{port}", f"localhost:{port}"}


class AgentChannel:
    """One window's door for agents: HTTP on localhost, with a token.

    Off until the window allows agents to edit. In ``--serve`` mode the
    editor's own server carries it (*url* is that server's); in the desktop
    app it opens a port of its own when allowed.
    """

    def __init__(self, backend):
        self.backend = backend
        self.url: str | None = None
        self.session_id: str | None = None
        self.token: str | None = None
        self.agent: Agent | None = None
        self._server: ThreadingHTTPServer | None = None
        self._images: OrderedDict[str, bytes] = OrderedDict()
        self._lock = threading.Lock()
        atexit.register(self._forget)

    def status(self, session_id: str | None) -> dict[str, Any]:
        enabled = session_id is not None and session_id == self.session_id
        agent = self.agent if enabled else None
        connected = bool(
            agent and agent.last_time and time.monotonic() - agent.last_time < CONNECTED
        )
        return {
            "enabled": enabled,
            "connected": connected,
            "last_action": agent.last_action if agent else None,
            "changes": agent.changes if agent else 0,
        }

    def enable(self, session_id: str) -> dict[str, Any]:
        session = self.backend.sessions[session_id]
        with self._lock:
            # One window of this editor at a time; allowing it again keeps
            # the token an agent may already hold.
            if self.session_id != session_id or self.token is None:
                self._forget()
                self.agent = Agent(session)
                self.session_id = session_id
                self.token = secrets.token_urlsafe(32)
            url = self.url
            if url is None:
                if self._server is None:
                    self._server = AgentServer(self)
                    threading.Thread(
                        target=self._server.serve_forever,
                        daemon=True,
                        name="vectrify-agents",
                    ).start()
                url = f"http://127.0.0.1:{self._server.server_port}"
            self._write(url, self.token)
        return self.status(session_id)

    def disable(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            if self.session_id == session_id:
                self._forget()
                self.session_id = self.token = self.agent = None
                if self._server is not None:
                    server, self._server = self._server, None

                    def close() -> None:
                        server.shutdown()
                        server.server_close()

                    threading.Thread(target=close, daemon=True).start()
        return self.status(session_id)

    def _write(self, url: str, token: str) -> None:
        path = discovery_file()
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = path.with_suffix(".tmp")
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as file:
            json.dump({"url": url, "token": token, "pid": os.getpid()}, file)
        temporary.replace(path)

    def _forget(self) -> None:
        """Remove the discovery file, if it still names this window."""
        path = discovery_file()
        with contextlib.suppress(OSError, ValueError):
            if json.loads(path.read_text()).get("token") == self.token:
                path.unlink()

    def http(self, method: str, path: str, headers: Any, body: bytes) -> tuple:
        """Answer one agent request as (status, content type, body)."""

        def error(status: int, message: str) -> tuple:
            return status, "application/json", json.dumps({"error": message}).encode()

        token = self.token
        given = headers.get("Authorization", "")
        if token is None or self.agent is None:
            return error(
                403,
                "Agent editing is off in this editor. Turn on 'Allow agents to "
                "edit' in its footer.",
            )
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
            if not 0 <= length <= 300 * 1024 * 1024:
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


class AgentHandler(BaseHTTPRequestHandler):
    server: Any

    def log_message(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def do_GET(self) -> None:
        answer(self, self.server.channel, self.server.server_port)

    def do_POST(self) -> None:
        answer(self, self.server.channel, self.server.server_port)


class AgentServer(ThreadingHTTPServer):
    """The desktop window's agent port, open while agents are allowed."""

    daemon_threads = True

    def __init__(self, channel: AgentChannel):
        super().__init__(("127.0.0.1", 0), AgentHandler)
        self.channel = channel
