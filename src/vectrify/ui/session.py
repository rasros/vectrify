"""Small UI-facing command adapter; document constraints remain authoritative."""

from __future__ import annotations

import base64
import contextlib
import io
import json
import math
import re
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from threading import RLock
from typing import Any
from uuid import uuid4

from PIL import Image
from shapely.geometry import Polygon

from vectrify.document import (
    Document,
    DocumentError,
    Editor,
    EditRejectedError,
    Element,
    Geometry,
    Selection,
    StaleRevisionError,
    Subpath,
    export_svg,
    import_svg,
    load_project,
    save_project,
)
from vectrify.document.arrows import ARROWS, is_open
from vectrify.document.clipboard import copy_objects, paste_objects
from vectrify.document.clipping import apply_clip
from vectrify.document.editor import Transaction
from vectrify.document.hit_test import HitIndex
from vectrify.document.holes import document_hole_shape, enclosed_objects, find_holes
from vectrify.document.join import path_style
from vectrify.document.lines import stroke_outline
from vectrify.document.model import EditKind, new_id, paint_server
from vectrify.document.paint import GradientStop, LinearGradient
from vectrify.document.redraw import attachment
from vectrify.document.regions import SHAPES, region_polygon
from vectrify.document.svg import NUMBER, SOLID, parse_path
from vectrify.document.topology import inverse_matrix
from vectrify.document.transforms import (
    REFLECTIONS,
    ancestry_matrix,
    multiply,
    object_matrix,
    rotation,
)
from vectrify.operations import (
    Budget,
    Job,
    Method,
    OperationRequest,
    Permissions,
    method,
)
from vectrify.project_file import MAX_SOURCE, decode_source, encode_project
from vectrify.refine.centreline import centreline
from vectrify.refine.redraw import redraw_stretch

# Aspect ratios this close (relatively) count as the same shape.
ASPECT_SLACK = 0.005
# The most pixels the reference is drawn over a grown artboard at; past
# this its resolution drops instead (see Session.reference_image).
REFERENCE_PIXELS = 64_000_000
# How near, in screen pixels, a redraw stroke's ends attach to a point of the
# outline, else to the outline itself.
NODE_REACH = 6
REACH = 10


# What Draw path and the shape tools paint with unless told otherwise.
DRAWING_PAINT = "#83b899"
# The basic shapes the shape tools draw, with their attributes (a rectangle's
# corner radius optional, last), and what their undo entries call them.
SHAPE_ATTRIBUTES = {
    "rect": ("x", "y", "width", "height", "rx"),
    "circle": ("cx", "cy", "r"),
    "ellipse": ("cx", "cy", "rx", "ry"),
    "line": ("x1", "y1", "x2", "y2"),
}
SHAPE_NAMES = {
    "rect": "rectangle",
    "circle": "circle",
    "ellipse": "ellipse",
    "line": "line",
}
# The sizes that must be positive for a shape to show.
SHAPE_SIZES = {"rect": ("width", "height"), "circle": ("r",), "ellipse": ("rx", "ry")}


# The edits the knife makes to a path it cuts: with nothing selected, it
# leaves paths locked against them alone.
KNIFE_EDITS = frozenset({EditKind.GEOMETRY, EditKind.STRUCTURE})
# Why a region edit takes nothing (Transaction.extract_region).
NO_CONTOUR = "No contour of these paths lies inside the region"


# Commands on points, possibly in several selected paths: their undo labels
# and handlers.
POINT_COMMANDS = {
    "node": ("Edit node", "_point_node"),
    "move_nodes": ("Move points", "_point_move_nodes"),
    "split": ("Add node", "_point_split"),
    "node_handles": ("Change handles", "_point_handles"),
    "delete_handle": ("Delete handle", "_point_delete_handle"),
    "straighten_handles": ("Toggle straight handles", "_point_handles"),
    "delete_node": ("Delete node", "_point_delete_node"),
    "delete_contour": ("Delete contour", "_point_delete_contour"),
    "break_points": ("Break at point", "_point_break"),
    "delete_segment": ("Delete segment", "_point_break"),
    "join_two_ends": ("Join points", "_point_join_two_ends"),
}


@dataclass(frozen=True)
class Command:
    """A session action's handler, undo label, and selection requirement."""

    handler: str
    label: str = ""
    needs_selection: bool = False


# The dispatch table is also the command inventory used by MCP coverage tests.
ACTION_COMMANDS = {
    "open": Command("_command_open"),
    "select": Command("_command_select"),
    "undo": Command("_command_history"),
    "redo": Command("_command_history"),
    "rename": Command("_command_rename"),
    "locks": Command("_command_locks"),
    "visibility": Command("_command_visibility"),
    "pin": Command("_command_pin"),
    "reference": Command("_command_reference"),
    "artboard": Command("_command_artboard"),
    **{
        name: Command("_command_points", label)
        for name, (label, _) in POINT_COMMANDS.items()
    },
    "add_path": Command("_command_add_path", "Draw path"),
    "add_shape": Command("_command_add_shape", "Draw shape"),
    "copy": Command("_command_copy", needs_selection=True),
    "paste": Command("_command_paste", "Paste objects"),
    "move_objects": Command("_command_move_objects"),
    "knife": Command("_knife", "Cut with knife"),
    "redraw_outline": Command("_redraw_outline", "Redraw outline"),
    "join_ends": Command("_command_lines", "Join line ends", True),
    "fill_to_line": Command("_command_lines", "Fill to centerline", True),
    "line_to_fill": Command("_command_lines", "Stroke to path", True),
    "convert_lines": Command("_command_lines", "Convert line/fill", True),
    "extract": Command("_command_extract"),
    "fill_holes": Command("_command_fill_holes", "Fill holes", True),
    "holes_to_shapes": Command("_command_holes_to_shapes", "Holes to shapes", True),
    "cut_hole": Command("_command_cut_hole", "Cut out as hole", True),
    "union": Command("_command_boolean", "Union", True),
    "subtract": Command("_command_boolean", "Subtract", True),
    "intersect": Command("_command_boolean", "Intersect", True),
    "exclude": Command("_command_boolean", "Exclude", True),
    "paint": Command("_command_paint", "Change paint", True),
    "move": Command("_command_move", "Move selection", True),
    "align": Command("_command_align", "Align objects", True),
    "distribute": Command("_command_align", "Distribute objects", True),
    "resize": Command("_command_resize", "Resize", True),
    "rotate": Command("_command_rotate", "Rotate", True),
    "flip_horizontal": Command("_command_flip", "Flip horizontal", True),
    "flip_vertical": Command("_command_flip", "Flip vertical", True),
    "group": Command("_command_group", "Group objects", True),
    "ungroup": Command("_command_ungroup", "Ungroup objects", True),
    "delete": Command("_command_delete", "Delete selection", True),
    "reorder": Command("_command_reorder", "Change stacking", True),
    "to_front": Command("_command_restack", "Bring to front", True),
    "to_back": Command("_command_restack", "Send to back", True),
    "detach": Command("_command_detach", "Detach geometry", True),
    "convert_to_path": Command("_command_convert_to_path", "Convert to path", True),
    "set_clip": Command("_command_set_clip", "Set clip", True),
    "remove_clip": Command("_command_remove_clip", "Remove clip", True),
    "split_disconnected": Command(
        "_command_split_disconnected", "Split disconnected parts", True
    ),
    "join_paths": Command("_command_join_paths", "Merge shapes", True),
    "combine_paths": Command("_command_combine_paths", "Combine paths", True),
}

OPERATION_COMMANDS = {
    "start": "_operation_start",
    "status": "_operation_status",
    "stop": "_operation_stop",
    "discard": "_operation_discard",
    "apply": "_operation_apply",
}


# Names of elements whose tag does not read as one.
# Ids the editor makes itself (new_id: a prefix and 32 hex digits) are not
# names: such objects are labelled by kind and number, like "Group 3".
GENERATED_ID = re.compile(r"[a-z]+_[0-9a-f]{32}")
LABELS = {"g": "Group", "linearGradient": "Linear gradient", "stop": "Gradient stop"}


def has_node(document: Document, object_id: str, node_id: str) -> bool:
    try:
        document.geometry_for(object_id).node(node_id)
    except DocumentError:
        return False
    return True


def number(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise DocumentError("Enter a finite number")
    return result


def length_text(value: float) -> str:
    """A coordinate or length as an attribute, without float noise."""
    return f"{round(value, 9) + 0.0:.12g}"


def drawing_frame(document: Document, parent: str) -> tuple[float, ...]:
    """The matrix from root user space into the frame of *parent*'s children:
    what is drawn into a transformed group lands where it was drawn."""
    return multiply(
        inverse_matrix(object_matrix(document, parent)),
        object_matrix(document, document.root.id),
    )


def drawing_paint(
    payload: dict, fill: str, stroke: str, scale: float = 1.0
) -> dict[str, str]:
    """The fill, stroke and stroke width a drawing tool asks for: by default
    *fill* and *stroke*, two units wide. The width is in root user space and
    is drawn *scale* times as wide in the frame it goes into."""
    paint = {}
    for name, default in (("fill", fill), ("stroke", stroke)):
        value = str(payload.get(name) or default)
        if not SOLID.fullmatch(value):
            raise DocumentError(f"The drawing {name} must be a colour or none")
        paint[name] = value
    width = number(payload.get("stroke_width", 2))
    if width <= 0:
        raise DocumentError("Stroke width must be positive")
    paint["stroke-width"] = length_text(width * scale)
    return paint


def parent_offset(
    ancestry: tuple[Element, ...], dx: float, dy: float
) -> tuple[float, float]:
    """A move by *dx*, *dy* in document coordinates, in the frame of the last
    of *ancestry*'s parent: as far on the page inside transformed groups."""
    a, b, c, d, _, _ = inverse_matrix(ancestry_matrix(ancestry[:-1]))
    return round(a * dx + c * dy, 12) + 0.0, round(b * dx + d * dy, 12) + 0.0


# A transform that starts with a translation, which a further move adds to.
LEADING_TRANSLATE = re.compile(
    rf"\s*translate\(\s*({NUMBER})(?:\s*,?\s*({NUMBER}))?\s*\)\s*(.*)", re.S
)


def translate(
    tx: Transaction, ancestry: tuple[Element, ...], dx: float, dy: float
) -> None:
    """Move the last of *ancestry* by *dx*, *dy* in its parent's frame.

    A transform that already starts with a translation takes the move into
    it, so repeated moves, such as arrow-key nudges, leave one translate.
    """
    previous = ancestry[-1].get("transform", "") or ""
    match = LEADING_TRANSLATE.fullmatch(previous)
    if match:
        dx = round(dx + float(match[1]), 12) + 0.0
        dy = round(dy + float(match[2] or 0), 12) + 0.0
        previous = match[3]
    transform = f"translate({dx} {dy}) {previous}".strip()
    tx.set_attributes(ancestry[-1].id, {"transform": transform})


def matrix_text(matrix: tuple[float, ...]) -> str:
    # Full precision: a transform undoing a group's must undo it exactly.
    return "matrix(" + " ".join(repr(v + 0.0) for v in matrix) + ")"


class Session:
    def __init__(
        self,
        document: Document,
        name: str = "Untitled.svg",
        reference: dict | None = None,
    ):
        self.editor = Editor(document)
        self.clipboard: Document | None = None
        self.name = name
        self._reference = self.fit_reference(reference) if reference else None
        self.reference_revision = 0
        self.epoch = uuid4().hex
        self.lock = RLock()
        self._svg_revision = -1
        self._svg = ""
        self._objects_document: Document | None = None
        self._objects: list[dict] = []
        self._reference_error: tuple[tuple, Image.Image] | None = None
        self.jobs: dict[str, Job] = {}
        # What the person's window last said it shows (see set_view), and when.
        self.view: dict[str, Any] | None = None
        self.view_time = 0.0
        # The last arrow-key nudge: its drawing, run and the revision it made.
        self._nudge: tuple[str, str, int] | None = None

    @property
    def reference(self) -> dict | None:
        return self._reference

    @reference.setter
    def reference(self, value: dict | None) -> None:
        if value != self._reference:
            self.reference_revision += 1
        self._reference = value

    def restore_reference(self, reference: dict | None, revision: int) -> None:
        """Restore a failed edit's reference checkpoint without a new version."""
        self._reference = reference
        self.reference_revision = revision

    @property
    def project_version(self) -> str:
        """Persisted content within this epoch, excluding selection and view."""
        return f"{self.editor.snapshot.revision}:{self.reference_revision}"

    def content_version(self) -> dict:
        return {
            "epoch": self.epoch,
            "revision": self.editor.snapshot.revision,
            "reference_revision": self.reference_revision,
            "project_version": self.project_version,
        }

    def set_view(self, view: Any) -> None:
        """Keep what the page reports it shows: the visible document region
        [x, y, w, h], the zoom (screen pixels per document unit), its size in
        pixels, the active tool, the entered group and the reference view."""
        if not isinstance(view, dict):
            raise DocumentError("A view is an object")
        region = view.get("region")
        pixels = view.get("pixels")
        if (
            not isinstance(region, list)
            or len(region) != 4
            or not isinstance(pixels, list)
            or len(pixels) != 2
        ):
            raise DocumentError("A view needs its region and pixel size")
        x, y, w, h = (number(v) for v in region)
        zoom = number(view.get("zoom"))
        width, height = (int(number(v)) for v in pixels)
        if w <= 0 or h <= 0 or zoom <= 0 or width <= 0 or height <= 0:
            raise DocumentError("A view needs a positive size and zoom")
        entered = view.get("entered")
        reference = view.get("reference_view")
        opacity = view.get("reference_opacity")
        self.view = {
            "region": [x, y, w, h],
            "zoom": zoom,
            "pixels": [width, height],
            "tool": str(view.get("tool", ""))[:40],
            "entered": str(entered) if entered else None,
            "reference_view": reference
            if reference in {"drawing", "overlay", "reference", "error"}
            else None,
            "reference_opacity": min(1.0, max(0.0, number(opacity)))
            if opacity is not None
            else None,
        }
        self.view_time = time.monotonic()

    def operation(self, payload: dict) -> dict:
        """Start, poll, stop, apply or discard one automated operation."""
        command = payload.get("command")
        handler = OPERATION_COMMANDS.get(command) if isinstance(command, str) else None
        if handler is None:
            raise DocumentError("Unknown operation command")
        return getattr(self, handler)(payload)

    def _operation_job(self, payload: dict) -> Job:
        job = self.jobs.get(str(payload.get("job")))
        if job is None:
            raise DocumentError("This operation preview has expired")
        return job

    def _operation_start(self, payload: dict) -> dict:
        self.check_revision(payload)
        chosen = method(str(payload.get("action")), str(payload.get("method")))
        request = self._request(chosen, payload)
        if chosen.background and any(
            j.method.background and j.status == "running" for j in self.jobs.values()
        ):
            raise DocumentError("Another operation is already running")
        job = Job(chosen, request, context_key=self._job_key(chosen))
        job.start()
        self.jobs = {k: v for k, v in self.jobs.items() if v.status == "running"}
        self.jobs[job.id] = job
        return job.state(preview=True)

    def _operation_status(self, payload: dict) -> dict:
        job = self._operation_job(payload)
        return job.state(preview=bool(payload.get("preview")))

    def _operation_stop(self, payload: dict) -> dict:
        job = self._operation_job(payload)
        job.stop.set()
        return job.state()

    def _operation_discard(self, payload: dict) -> dict:
        job = self._operation_job(payload)
        job.stop.set()
        del self.jobs[job.id]
        return {"discarded": True}

    def _operation_apply(self, payload: dict) -> dict:
        job = self._operation_job(payload)
        if job.context_key != self._job_key(job.method):
            raise StaleRevisionError(
                "The drawing or reference changed. Run the operation again."
            )
        job.apply(payload.get("choice", 0))
        del self.jobs[job.id]
        return self.state()

    def _request(self, chosen: Method, payload: dict) -> OperationRequest:
        bounds = payload.get("bounds", self.state(svg=False)["bounds"])
        snapshot = self.editor.snapshot_at(
            payload.get("revision", self.editor.snapshot.revision)
        )
        if "selection" in payload:
            snapshot = replace(snapshot, selection=self._payload_selection(payload))
        # Operations act on whole paths, whichever of their points are selected.
        snapshot = replace(
            snapshot, selection=replace(snapshot.selection, node_ids=frozenset())
        )
        if payload.get("scope") == "drawing":
            snapshot = replace(snapshot, selection=Selection(whole_document=True))
        elif payload.get("scope") not in {None, "selection"}:
            raise DocumentError("Choose the selection or the whole drawing")
        return OperationRequest(
            action=chosen.action,
            method=chosen.name,
            snapshot=snapshot,
            editor=self.editor,
            permissions=Permissions.parse(payload.get("permissions")),
            settings=payload.get("settings") or {},
            budget=Budget.parse(payload.get("budget")),
            reference=self.reference_image() if chosen.needs_reference else None,
            bounds=tuple(bounds) if isinstance(bounds, list) else bounds,
        )

    def _job_key(self, method: Method) -> tuple:
        """What a result depends on besides the revision the commit checks."""
        reference = self.reference["data_url"] if self.reference else None
        return (self.epoch, reference if method.needs_reference else None)

    def reference_image(self) -> Image.Image | None:
        """The reference over the artboard, as the editor and every method
        stretch it.

        The image keeps its own place in the drawing, its *box*, which is
        the artboard it was fitted to. After the artboard changes it is
        cropped, or padded with transparency, to cover the new artboard at
        the same resolution, so it stays under the artwork it was traced
        into.
        """
        if not self.reference:
            return None
        data = base64.b64decode(self.reference["data_url"].split(",", 1)[1])
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            picture = image.copy()
        ax, ay, aw, ah = self.editor.snapshot.document.artboard()
        bx, by, bw, bh = self.reference.get("box") or (ax, ay, aw, ah)
        if (ax, ay, aw, ah) == (bx, by, bw, bh):
            return picture
        sx, sy = picture.width / bw, picture.height / bh
        shrink = min(1.0, math.sqrt(REFERENCE_PIXELS / (aw * sx * ah * sy)))
        size = (max(1, round(aw * sx * shrink)), max(1, round(ah * sy * shrink)))
        extent = (
            (ax - bx) * sx,
            (ay - by) * sy,
            (ax + aw - bx) * sx,
            (ay + ah - by) * sy,
        )
        return picture.convert("RGBA").transform(
            size,
            Image.Transform.EXTENT,
            extent,
            Image.Resampling.BILINEAR,
            fillcolor=(0, 0, 0, 0),
        )

    def _reference_error_key(self) -> tuple | None:
        if not self.reference:
            return None
        return (
            self.epoch,
            self.editor.snapshot.revision,
            self.reference["data_url"],
            tuple(
                self.reference.get("box") or self.editor.snapshot.document.artboard()
            ),
        )

    def reference_error(self, expected: dict | None = None) -> Image.Image:
        """One error map shared by the canvas and MCP's current-view render."""
        import numpy as np

        from vectrify.image_utils import on_white
        from vectrify.svg_render import render_image
        from vectrify.ui.agent import _size, crop_reference, heat_map

        with self.lock:
            if expected is not None:
                self.check_revision(expected)
            key = self._reference_error_key()
            if key is None:
                raise DocumentError("Load a reference image first")
            document = self.editor.snapshot.document
            if self._reference_error and self._reference_error[0] == key:
                return self._reference_error[1]
            reference = self.reference_image()
        # Documents are immutable snapshots. Compare the captured pair without
        # blocking edits, selection, or polling on the rasterizer and PNG work.
        assert reference is not None
        bounds = document.artboard()
        size = _size(bounds, min(2048, max(reference.size)))
        drawing = render_image(export_svg(document), bounds, size)
        target = crop_reference(on_white(reference), bounds, bounds, size)
        difference = (
            np.asarray(drawing, dtype=np.float64) - np.asarray(target, dtype=np.float64)
        ) / 255
        image = heat_map(np.sqrt((difference**2).mean(axis=2)))
        with self.lock:
            if self._reference_error_key() == key:
                self._reference_error = (key, image)
        return image

    def sample(self, payload: dict) -> dict:
        """The Eyedropper's colour at *payload*'s point: the artwork's or
        the reference's, sampled as MCP's pick samples it (see
        agent_look.sample_colour) over *radius* document units.

        Nothing changes; the page applies the colour with a paint edit.
        """
        from vectrify.image_utils import on_white
        from vectrify.ui import agent_look
        from vectrify.ui.agent import crop_reference, render_document

        self.check_revision(payload)
        source = payload.get("source")
        if source not in {"artwork", "reference"}:
            raise DocumentError("Sample the artwork or the reference")
        try:
            x, y = (number(v) for v in payload["at"])
            radius = number(payload.get("radius", 0))
        except (KeyError, TypeError, ValueError) as exc:
            raise DocumentError("Sample at a point [x, y] in the drawing") from exc
        if radius < 0:
            raise DocumentError("A sample's radius must not be negative")
        with self.lock:
            document = self.editor.snapshot.document
            artboard = document.artboard()
            if source == "artwork":
                svg = self._exported_svg()

                def render(box: tuple, size: tuple[int, int]) -> Image.Image:
                    return render_document(svg, box, size)

            else:
                reference = self.reference_image()
                if reference is None:
                    raise DocumentError(
                        "No reference is loaded: load an image under Reference "
                        "to sample it"
                    )
                ax, ay, aw, ah = artboard
                if not (ax <= x <= ax + aw and ay <= y <= ay + ah):
                    raise DocumentError(
                        "The reference covers only the artboard: sample inside it"
                    )
                picture = on_white(reference)

                def render(box: tuple, size: tuple[int, int]) -> Image.Image:
                    return crop_reference(picture, artboard, box, size)

            colour = agent_look.sample_colour(render, (x, y), radius)
            return {
                **self.content_version(),
                "source": source,
                "at": [x, y],
                "radius": max(radius, agent_look.MIN_SAMPLE_RADIUS),
                "colour": agent_look.hex_colour(colour),
            }

    def _object_state(self, document: Document) -> list[dict]:
        # Documents are immutable. Selection changes keep this metadata
        # valid, including labels, hierarchy, paint and inherited locks.
        if self._objects_document is document:
            return self._objects
        objects = []
        counters: dict[str, int] = {}
        # How many elements draw each geometry: more than one shares it.
        users = Counter(
            e.geometry_id for e in document.elements() if e.geometry_id is not None
        )
        hidden = {
            child.id
            for gradient in document.elements()
            if gradient.paint_owner is not None
            for child in Document(gradient).elements()
        }
        # A defs holding only private paint is an SVG storage detail.
        hidden.update(
            e.id
            for e in document.elements()
            if e.tag == "defs"
            and all(c.id in hidden for c in e.children)
            and not e.attributes
            and not e.locks
            and not e.name
        )
        gradient_users: dict[str, int] = {}
        for element in document.elements():
            if element.tag == "svg" or element.id in hidden:
                continue
            counters[element.tag] = counters.get(element.tag, 0) + 1
            ancestors = document.ancestry(element.id)
            server = paint_server(path_style(document, element)["fill"])
            gradient = (
                element
                if element.tag == "linearGradient"
                else document.element(server)
                if server
                else None
            )
            if gradient and gradient.id not in gradient_users:
                gradient_users[gradient.id] = (
                    1
                    if gradient.paint_owner is not None
                    else sum(
                        document.element(oid).tag
                        in {"path", "rect", "circle", "ellipse", "line", "use"}
                        for oid in document.dependents({gradient.id})
                    )
                )
            objects.append(
                {
                    "id": element.id,
                    "tag": element.tag,
                    "name": element.name,
                    "label": element.name
                    or (
                        element.id
                        if not GENERATED_ID.fullmatch(element.id)
                        else "Definitions"
                        if element.tag == "defs"
                        else f"{LABELS.get(element.tag, element.tag.capitalize())} "
                        f"{counters[element.tag]}"
                    ),
                    "parent": ancestors[-2].id,
                    "depth": len(ancestors) - 2,
                    "resource": any(e.tag in {"defs", "clipPath"} for e in ancestors),
                    "open_path": is_open(document, element),
                    "shared": element.geometry_id is not None
                    and users[element.geometry_id] > 1,
                    "attributes": dict(element.attributes),
                    "fill_gradient": {
                        "id": gradient.id,
                        "private": gradient.paint_owner == element.id,
                        "users": gradient_users[gradient.id],
                        "attributes": dict(gradient.attributes),
                        "stops": [dict(c.attributes) for c in gradient.children],
                    }
                    if gradient
                    else None,
                    "locks": sorted(element.locks),
                    "inherited_locks": sorted(
                        set().union(*(a.locks for a in ancestors))
                    ),
                    # Hidden itself (display="none"), and hidden by itself
                    # or a group around it: not drawn or hit, but kept.
                    "hidden": element.get("display") == "none",
                    "inherited_hidden": any(
                        a.get("display") == "none" for a in ancestors
                    ),
                }
            )
        self._objects_document = document
        self._objects = objects
        return objects

    def state(self, *, svg: bool = True) -> dict:
        snapshot = self.editor.snapshot
        objects = self._object_state(snapshot.document)
        root = snapshot.document.root
        bounds = list(snapshot.document.artboard())
        result = {
            **self.content_version(),
            "name": self.name,
            "root": root.id,
            "clipboard": self.clipboard is not None,
            "bounds": bounds,
            "objects": objects,
            "selection": {
                "objects": sorted(snapshot.selection.object_ids),
                "nodes": sorted(snapshot.selection.node_ids),
            },
            "undo": [e.label for e in self.editor.undo_entries if e.author == "person"],
            "redo": [e.label for e in self.editor.redo_entries if e.author == "person"],
            "undo_ids": [
                e.id for e in self.editor.undo_entries if e.author == "person"
            ],
            "redo_ids": [
                e.id for e in self.editor.redo_entries if e.author == "person"
            ],
            "reference": {
                "name": self.reference["name"],
                "opacity": self.reference["opacity"],
            }
            if self.reference
            else None,
        }
        if svg:
            result["svg"] = self._exported_svg()
        return result

    def _exported_svg(self) -> str:
        """The drawing as SVG, exported once per revision."""
        snapshot = self.editor.snapshot
        if self._svg_revision != snapshot.revision:
            self._svg = export_svg(snapshot.document)
            self._svg_revision = snapshot.revision
        return self._svg

    def nodes(self, object_id: str) -> dict:
        document = self.editor.snapshot.document
        geometry = document.geometry_for(object_id)
        return {
            "epoch": self.epoch,
            "revision": self.editor.snapshot.revision,
            "object": object_id,
            "geometry": asdict(geometry),
        }

    def geometries(self, object_ids: list) -> dict:
        """The geometry of each of several paths, for editing their points.

        Each also lists *users*, the paths drawing that same geometry: an
        edit to its points changes all of them.
        """
        document = self.editor.snapshot.document
        users: dict[str, list[str]] = {}
        for element in document.elements():
            if element.geometry_id is not None:
                users.setdefault(element.geometry_id, []).append(element.id)
        result = {}
        for oid in map(str, object_ids):
            geometry = document.geometry_for(oid)
            result[oid] = {**asdict(geometry), "users": users.get(geometry.id, [])}
        return {
            "epoch": self.epoch,
            "revision": self.editor.snapshot.revision,
            "geometries": result,
        }

    def _in_selection(self, object_id: str) -> bool:
        """Whether *object_id* is selected, or inside a selected group."""
        snapshot = self.editor.snapshot
        return object_id in snapshot.document.selection_ids(
            replace(snapshot.selection, node_ids=frozenset())
        )

    def holes(self, payload: dict) -> dict:
        self.check_revision(payload)
        document = self.editor.snapshot.document
        oid = payload["object"]
        if not self._in_selection(oid):
            raise DocumentError("Select the path before inspecting holes")
        holes = find_holes(document, oid)
        requested = frozenset(payload.get("holes", []))
        if not requested <= {h.id for h in holes}:
            raise DocumentError("The hole selection is no longer valid")
        selected = tuple(h for h in holes if h.id in requested)
        return {
            "object": oid,
            "holes": [
                {
                    "id": h.id,
                    "area": document_hole_shape(document, oid, (h,)).area,
                    "d": h.path_data,
                    "bounds": list(h.shape.bounds),
                }
                for h in holes
            ],
            "enclosed": sorted(enclosed_objects(document, oid, selected))
            if payload.get("find_enclosed")
            else [],
        }

    def settle(self, revision: int) -> None:
        """Count the edits since *revision* as one revision (Editor.settle),
        forgetting the SVG made at one in between."""
        self.editor.settle(revision)
        self._svg_revision = -1

    def check_revision(self, payload: dict) -> None:
        if payload.get("epoch") != self.epoch:
            raise StaleRevisionError(
                "Another drawing was opened. This edit belongs to the previous drawing."
            )
        revision = payload.get("revision")
        if (
            type(revision) is not int
            or not 0 <= revision <= self.editor.snapshot.revision
        ):
            raise DocumentError("Invalid document revision")

    def _payload_selection(self, payload: dict) -> Selection:
        selected = payload.get("selection")
        if selected is None:
            return self.editor.snapshot.selection
        return Selection(
            frozenset(selected.get("objects", [])), frozenset(selected.get("nodes", []))
        )

    @contextlib.contextmanager
    def planning(
        self, revision: int, selection: Selection | None = None
    ) -> Iterator[None]:
        """Plan stale commands privately, then merge one edit into live state.

        The caller holds the session lock. A failed plan/merge never changes
        the live drawing, history, selection, or reference.
        """
        live = self.editor
        if revision == live.snapshot.revision:
            yield
            return
        base = live.snapshot_at(revision)
        fork = live.fork(
            revision, selection if selection is not None else base.selection
        )
        reference = self.reference
        reference_revision = self.reference_revision
        self.editor = fork
        self._svg_revision = -1
        self._objects_document = None
        try:
            yield
            edited = fork.snapshot
            # Handlers may issue several transactions; this is one atomic
            # edit against the live drawing, authored by the caller.
            if fork.undo_entries:
                entry = fork.undo_entries[-1]
                author = live.author
                live.author = entry.author
                try:
                    live.merge(base, edited.document, entry.label, edited.selection)
                finally:
                    live.author = author
        except Exception:
            self.restore_reference(reference, reference_revision)
            raise
        finally:
            self.editor = live
            self._svg_revision = -1
            self._objects_document = None

    def open(self, source: str | bytes, name: str) -> None:
        source = decode_source(source)
        reference = None
        if source.lstrip().startswith("{"):
            data = json.loads(source)
            if data.get("vectrify_editor") == 1:
                document, selection = load_project(json.dumps(data["document"]))
                if data.get("reference"):
                    reference = self.validate_reference(data["reference"])
            else:
                document, selection = load_project(source)
        else:
            document, selection = import_svg(source), Selection()
        editor = Editor(document, selection=selection)
        if reference is not None and "box" not in reference:
            # Projects saved before references kept their place had it
            # stretched over the artboard.
            reference["box"] = list(document.artboard())
        for job in self.jobs.values():
            job.stop.set()
        self.jobs = {}
        self.editor, self.name, self.reference = editor, name, reference
        self.reference_revision = 0
        self.epoch = uuid4().hex
        self._svg_revision = -1

    def fit_reference(self, reference: dict) -> dict:
        """*reference* fitted to the artboard (see _fitted_reference), placed
        over it: its *box* is the artboard it now covers. Later artboard
        changes leave it there (see reference_image)."""
        fitted = self._fitted_reference(reference)
        return {**fitted, "box": list(self.editor.snapshot.document.artboard())}

    def _fitted_reference(self, reference: dict) -> dict:
        """*reference* fitted to the artboard without distortion.

        The editor and every method stretch the reference over the artboard,
        so their shapes have to agree. An empty drawing takes the image's
        size as its artboard; a drawing with content keeps its artboard, and
        the image is centred on a transparent canvas of the artboard's shape.
        Returns the reference to keep, which may be a new, padded image.
        """
        data = base64.b64decode(reference["data_url"].split(",", 1)[1])
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            width, height = image.size
            document = self.editor.snapshot.document
            _, _, aw, ah = document.artboard()
            if (
                aw <= 0
                or ah <= 0
                or abs(width / height - aw / ah) <= ASPECT_SLACK * (aw / ah)
            ):
                return reference
            empty = not any(e.tag not in {"svg", "defs"} for e in document.elements())
            if empty:
                root = document.root.id
                with self.editor.transaction(
                    "Fit artboard to reference", selection=Selection.all()
                ) as tx:
                    tx.set_attributes(
                        root,
                        {
                            "width": str(width),
                            "height": str(height),
                            "viewBox": f"0 0 {width} {height}",
                        },
                    )
                return reference
            # Pad to the artboard's shape, keeping every image pixel.
            if width / height > aw / ah:
                canvas = (width, round(width * ah / aw))
            else:
                canvas = (round(height * aw / ah), height)
            padded = Image.new("RGBA", canvas, (0, 0, 0, 0))
            padded.paste(
                image.convert("RGBA"),
                ((canvas[0] - width) // 2, (canvas[1] - height) // 2),
            )
        buffer = io.BytesIO()
        padded.save(buffer, "PNG")
        url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        return {**reference, "data_url": url}

    @staticmethod
    def validate_reference(value: dict) -> dict:
        url = value["data_url"]
        if not isinstance(url, str) or len(url) > MAX_SOURCE:
            raise DocumentError("Reference image exceeds the editor limit")
        prefix, encoded = url.split(",", 1)
        formats = {
            "data:image/png;base64": "PNG",
            "data:image/jpeg;base64": "JPEG",
            "data:image/webp;base64": "WEBP",
        }
        if prefix not in formats:
            raise DocumentError("Choose a PNG, JPEG or WebP reference")
        data = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(data)) as image:
            if image.format != formats[prefix]:
                raise DocumentError("Reference content does not match its image type")
            image.verify()
        opacity = number(value.get("opacity", 0.5))
        if not 0 <= opacity <= 1:
            raise DocumentError("Reference opacity must be between zero and one")
        reference: dict[str, Any] = {
            "name": str(value.get("name", "Reference")),
            "data_url": url,
            "opacity": opacity,
        }
        # Where the image lies in the drawing; without it, over the artboard.
        box = value.get("box")
        if box is not None:
            if not isinstance(box, list) or len(box) != 4:
                raise DocumentError("A reference's box is [x, y, width, height]")
            box = [number(v) for v in box]
            if box[2] <= 0 or box[3] <= 0:
                raise DocumentError("A reference's box needs a positive size")
            reference["box"] = box
        return reference

    def project(self) -> str:
        return json.dumps(
            {
                "vectrify_editor": 1,
                "document": json.loads(
                    save_project(
                        self.editor.snapshot.document, self.editor.snapshot.selection
                    )
                ),
                "reference": self.reference,
            },
            allow_nan=False,
            separators=(",", ":"),
        )

    def export(self, payload: dict) -> dict[str, Any]:
        """The drawing to save: its SVG, or the project, compressed with a
        summary of what it holds when *compressed* asks."""
        self.check_revision(payload)
        if (
            "project_version" in payload
            and payload["project_version"] != self.project_version
        ):
            raise StaleRevisionError("The project changed while saving. Save again.")
        content = self.project() if payload.get("project") else self.state()["svg"]
        result = {**self.content_version(), "content": content}
        if payload.get("project") and payload.get("compressed"):
            result["content"] = base64.b64encode(
                encode_project(result["content"])
            ).decode("ascii")
            result["encoding"] = "base64"
            document = self.editor.snapshot.document
            geometry_ids = {element.geometry_id for element in document.elements()}
            result["summary"] = {
                "objects": len(document.elements()) - 1,
                "nodes": sum(
                    len(subpath.nodes)
                    for geometry in document.geometries
                    if geometry.id in geometry_ids
                    for subpath in geometry.subpaths
                ),
            }
        return result

    def action(self, payload: dict) -> dict:
        command = payload["command"]
        # Selection is UI state. A concurrent document edit must not
        # prevent the person selecting surviving identities.
        self.check_revision(payload)
        if (
            command == "delete"
            and payload.get("confirm_pinned") is True
            and payload["revision"] != self.editor.snapshot.revision
        ):
            raise StaleRevisionError(
                "The drawing changed while confirming deletion. Try deleting again."
            )
        if (
            command not in {"select", "undo", "redo", "open"}
            and payload["revision"] != self.editor.snapshot.revision
        ):
            with self.planning(payload["revision"], self._payload_selection(payload)):
                self.action(payload)
            # A nudge merged from an earlier revision starts a step of its own.
            self._nudge = None
            return self.state()
        spec = ACTION_COMMANDS.get(command) if isinstance(command, str) else None
        if spec is None:
            raise DocumentError("Unknown editor command")
        if spec.needs_selection and not self.editor.snapshot.selection.object_ids:
            raise DocumentError("Select an object first")
        before = self.editor.snapshot.revision
        getattr(self, spec.handler)(payload)
        if command in {"move", "move_nodes"}:
            self._group_nudge(payload.get("nudge"), before)
        return self.state(
            svg=command == "open"
            or self.editor.snapshot.revision != before
            or payload.get("revision") != before
        )

    def _group_nudge(self, run: Any, before: int) -> None:
        """Make the nudges of one run one undo step.

        Arrow-key nudges name their *run*: the window starts a new one when
        the selection changes or a second passes without a nudge. A nudge
        joins the step of the one before it when that was of the same run,
        on this drawing, and nothing else has changed the drawing since.
        """
        after = self.editor.snapshot.revision
        if not isinstance(run, str) or not run or after == before:
            self._nudge = None
            return
        previous, self._nudge = self._nudge, (self.epoch, run, after)
        entries = self.editor.undo_entries
        if (
            previous == (self.epoch, run, before)
            and len(entries) >= 2
            and entries[-2].revision == before
            and entries[-2].author == entries[-1].author
            and entries[-2].label == entries[-1].label
        ):
            self.editor.squash(len(entries) - 2)

    def _command_select(self, payload: dict) -> None:
        self.editor.select(
            Selection(
                object_ids=frozenset(payload.get("objects", [])),
                node_ids=frozenset(payload.get("nodes", [])),
            )
        )

    def _command_open(self, payload: dict) -> None:
        source = payload["source"]
        encoding = payload.get("encoding")
        if encoding == "base64":
            if len(source) > 4 * ((MAX_SOURCE + 2) // 3):
                raise DocumentError("File exceeds the 128 MB editor limit")
            source = base64.b64decode(source, validate=True)
        elif encoding is not None:
            raise DocumentError("Unsupported file encoding")
        self.open(source, str(payload.get("name", "Untitled.svg")))

    def _command_rename(self, payload: dict) -> None:
        if self.editor.snapshot.selection.object_ids != frozenset({payload["object"]}):
            raise DocumentError("Select one object to rename")
        self.editor.rename_object(payload["object"], payload["name"])

    def _command_locks(self, payload: dict) -> None:
        object_id = payload["object"]
        if object_id not in self.editor.snapshot.document.selection_ids(
            self.editor.snapshot.selection
        ):
            raise DocumentError("Select the object before changing its locks")
        self.editor.set_locks(object_id, frozenset(payload["locks"]))

    def _command_visibility(self, payload: dict) -> None:
        """Hide or show objects as one undoable edit.

        The objects are those given, as the tree's eye names its row, or
        else the selection; the selection stays as it is. With "all", every
        hidden object is shown. Hiding is saved in the drawing (see
        Transaction.set_hidden).
        """
        document = self.editor.snapshot.document
        hidden = payload.get("hidden")
        if not isinstance(hidden, bool):
            raise DocumentError("Say whether to hide or show the objects")
        if payload.get("all"):
            if hidden:
                raise DocumentError("Hide all is not offered: select what to hide")
            ids = [
                e.id
                for e in document.elements()
                if e.get("display") == "none"
                and not any(
                    a.tag in {"defs", "clipPath"} for a in document.ancestry(e.id)
                )
            ]
            label = "Show all objects"
        else:
            given = payload.get("objects")
            if given is None:
                given = sorted(self._object_selection().object_ids)
            if not isinstance(given, list) or not all(
                isinstance(oid, str) for oid in given
            ):
                raise DocumentError("Give the objects to hide or show")
            if not given:
                raise DocumentError("Select objects first")
            ids = [e.id for e in document.elements() if e.id in set(given)]
            if len(ids) != len(set(given)):
                raise DocumentError("Some of the objects were removed meanwhile")
            label = "Hide objects" if hidden else "Show objects"
        ids = [
            oid
            for oid in ids
            if (document.element(oid).get("display") == "none") != hidden
        ]
        if not ids:
            return
        with self.editor.transaction(
            label, selection=Selection(object_ids=frozenset(ids))
        ) as tx:
            for oid in ids:
                tx.set_hidden(oid, hidden)

    def _command_pin(self, payload: dict) -> None:
        self.editor.pin_nodes(self._points(payload), pinned=bool(payload["pinned"]))

    def _command_reference(self, payload: dict) -> None:
        reference = (
            self.validate_reference(payload["reference"])
            if payload.get("reference")
            else None
        )
        # A new image is fitted to the artboard; an opacity change sends
        # the fitted one back and is kept as it is.
        if reference and reference["data_url"] != (self.reference or {}).get(
            "data_url"
        ):
            reference = self.fit_reference(reference)
        elif reference and self.reference:
            # The image keeps its place: only artboard fitting places it.
            box = self.reference.get("box")
            artboard = self.editor.snapshot.document.artboard()
            reference = {**reference, "box": box or list(artboard)}
        self.reference = reference

    def _command_artboard(self, payload: dict) -> None:
        """Set the artboard to *box* [x, y, width, height], or with *fit*
        "artwork" to the painted bounds of everything drawn, as one undoable
        edit (Transaction.set_artboard). The artwork stays where it is, and
        so does the reference (see reference_image)."""
        document = self.editor.snapshot.document
        if payload.get("fit") == "artwork":
            painted = HitIndex(document).bounds(
                frozenset(e.id for e in document.elements())
            )
            if painted is None:
                raise DocumentError("Nothing is drawn to fit the artboard to")
            left, top, right, bottom = painted
            box = (left, top, right - left, bottom - top)
            label = "Fit artboard to artwork"
        else:
            raw = payload.get("box")
            if not isinstance(raw, list) or len(raw) != 4:
                raise DocumentError("Give the artboard as [x, y, width, height]")
            box = tuple(number(v) for v in raw)
            label = "Change artboard"
        with self.editor.transaction(label) as tx:
            tx.set_artboard(*box)

    def _command_history(self, payload: dict) -> None:
        getattr(self.editor, payload["command"])(payload.get("ids"), author="person")

    def _command_points(self, payload: dict) -> None:
        chosen = self.editor.snapshot.selection
        self._edit_points(payload["command"], payload)
        # Deleting the last selected points keeps their paths selected.
        if chosen.node_ids and self.editor.snapshot.selection == Selection():
            existing = {e.id for e in self.editor.snapshot.document.elements()}
            kept = chosen.object_ids & existing
            if kept:
                self.editor.select(Selection(object_ids=kept))

    def _points(self, payload: dict) -> list[tuple[str, str]]:
        """The (object, node) pairs a point command acts on, in selected paths.

        *points* lists them, possibly across several paths, selected or
        inside a selected group; a single point may be given as *object* and
        *node* instead. A node of geometry several paths share is one point:
        it is kept once, in the first path naming it.
        """
        raw = payload.get("points")
        if raw is None:
            raw = [[payload["object"], payload["node"]]]
        if (
            not isinstance(raw, list)
            or not raw
            or any(not isinstance(p, (list, tuple)) or len(p) != 2 for p in raw)
        ):
            raise DocumentError("Choose the points to edit")
        points = list(dict.fromkeys((str(o), str(n)) for o, n in raw))
        if not all(self._in_selection(o) for o in {o for o, _ in points}):
            raise DocumentError("Select the path before editing its points")
        document = self.editor.snapshot.document
        unique = {}
        for oid, nid in points:
            try:
                geometry = document.geometry_for(oid).id
            except DocumentError:
                geometry = oid
            unique.setdefault((geometry, nid), (oid, nid))
        return list(unique.values())

    def _edit_points(self, command: str, payload: dict) -> None:
        """Edit points of one or more selected paths as one undoable step."""
        document = self.editor.snapshot.document
        if command == "move_nodes":
            changes = payload.get("changes") or {}
            if (
                not isinstance(changes, dict)
                or not changes
                or any(not isinstance(v, dict) for v in changes.values())
            ):
                raise DocumentError("Choose the points to move")
            points = self._points(
                {"points": [[o, n] for o, nodes in changes.items() for n in nodes]}
            )
        else:
            points = self._points(payload)
        if command in {"node", "delete_handle"} and len(points) != 1:
            raise DocumentError("Drag one point at a time, or move them together")
        objects = frozenset(o for o, _ in points)
        # Editing a path's nodes changes every object drawing its geometry.
        scope = objects.union(
            *(document.geometry_users(document.geometry_for(o).id) for o in objects)
        )
        label, handler = POINT_COMMANDS[command]
        with self.editor.transaction(
            label, selection=Selection(object_ids=scope)
        ) as tx:
            getattr(self, handler)(tx, document, points, payload)

    @staticmethod
    def _contour_of(geometry: Geometry, nid: str) -> Subpath:
        """The contour of *geometry* holding node *nid*."""
        return next(s for s in geometry.subpaths if any(n.id == nid for n in s.nodes))

    def _point_node(
        self,
        tx: Transaction,
        _document: Document,
        points: list[tuple[str, str]],
        payload: dict,
    ) -> None:
        oid, nid = points[0]
        values = tuple(number(v) for v in payload["values"])
        if "handle_offset" in payload:
            offset = payload["handle_offset"]
            if offset not in {0, 2} or len(values) != 6:
                raise DocumentError("Choose a curve handle to move")
            tx.move_handle(oid, nid, offset, (values[offset], values[offset + 1]))
        else:
            tx.update_node(oid, nid, values)

    def _point_move_nodes(
        self,
        tx: Transaction,
        _document: Document,
        points: list[tuple[str, str]],
        payload: dict,
    ) -> None:
        # The values already carry the handles each point takes along.
        # A node of shared geometry moves once, as its first path has it.
        kept = set(points)
        for oid, nodes in payload["changes"].items():
            moved = {
                str(nid): tuple(number(v) for v in values)
                for nid, values in nodes.items()
                if (str(oid), str(nid)) in kept
            }
            if moved:
                tx.update_nodes(str(oid), moved)

    def _point_split(
        self,
        tx: Transaction,
        document: Document,
        points: list[tuple[str, str]],
        _payload: dict,
    ) -> None:
        edges = {}
        for oid, nid in points:
            geometry = document.geometry_for(oid)
            node = geometry.node(nid)
            contour = self._contour_of(geometry, nid)
            if node.command == "M" and not contour.closed:
                if len(contour.nodes) < 2:
                    raise DocumentError("This contour has no edge to add a node on")
                nid = contour.nodes[1].id
            # Both ends of the first edge may be selected. Split each
            # original edge once, regardless of selection order.
            edges.setdefault((geometry.id, nid), (oid, nid))
        for oid, nid in edges.values():
            tx.split_edge(oid, nid)

    def _point_delete_handle(
        self,
        tx: Transaction,
        _document: Document,
        points: list[tuple[str, str]],
        payload: dict,
    ) -> None:
        oid, nid = points[0]
        tx.delete_handle(oid, nid, payload["offset"])

    def _point_handles(
        self,
        tx: Transaction,
        document: Document,
        points: list[tuple[str, str]],
        payload: dict,
    ) -> None:
        seen = set()
        for oid, nid in points:
            geometry = document.geometry_for(oid)
            geometry.node(nid)
            contour = self._contour_of(geometry, nid)
            first, last = contour.nodes[0], contour.nodes[-1]
            # A box selection can include both nodes at a closed
            # contour's join. Switch its one handle only once.
            canonical = (
                first.id
                if contour.closed and first.endpoint == last.endpoint and nid == last.id
                else nid
            )
            key = geometry.id, canonical
            if key in seen:
                continue
            seen.add(key)
            if payload["command"] == "node_handles":
                tx.set_node_handles(oid, nid, int(payload["count"]))
            else:
                tx.set_node_handle_alignment(
                    oid, nid, bool(payload.get("aligned", True)), payload.get("side")
                )

    def _point_delete_node(
        self,
        tx: Transaction,
        _document: Document,
        points: list[tuple[str, str]],
        _payload: dict,
    ) -> None:
        # Deleting one point can take its contour, or its path, along.
        for oid, nid in points:
            if has_node(tx.preview, oid, nid):
                tx.delete_node(oid, nid)

    def _point_break(
        self,
        tx: Transaction,
        _document: Document,
        points: list[tuple[str, str]],
        payload: dict,
    ) -> None:
        by_object: dict[str, list[str]] = {}
        for oid, nid in points:
            by_object.setdefault(oid, []).append(nid)
        for oid, nids in by_object.items():
            if payload["command"] == "break_points":
                tx.break_points(oid, nids)
            else:
                tx.delete_segments(oid, frozenset(nids))

    def _point_join_two_ends(
        self,
        tx: Transaction,
        _document: Document,
        points: list[tuple[str, str]],
        _payload: dict,
    ) -> None:
        if len(points) != 2 or points[0] == points[1]:
            raise DocumentError("Select the two points to join")
        # Any two points join: one that is not a free end yet is cut there
        # first, opening a closed contour or splitting a line, so it becomes one.
        ends = []
        for oid, nid in points:
            copies: dict[str, set[str]] = {}
            geometry = tx.preview.geometry_for(oid)
            free_end = any(
                not sp.closed and nid in {sp.nodes[0].id, sp.nodes[-1].id}
                for sp in geometry.subpaths
            )
            if not free_end:
                copies = tx.break_points(oid, [nid])
            # A closed contour's start opens under other IDs.
            found: set[str] = set()
            grown = {nid}
            while grown != found:
                found = grown
                grown = found.union(*(copies.get(i, ()) for i in found))
            geometry = tx.preview.geometry_for(oid)
            present = {n.id for sp in geometry.subpaths for n in sp.nodes}
            alive = sorted(found & present)
            ends.append((oid, nid if nid in present or not alive else alive[0]))
        tx.join_ends(frozenset(o for o, _ in points), ends=ends)

    def _point_delete_contour(
        self,
        tx: Transaction,
        document: Document,
        points: list[tuple[str, str]],
        _payload: dict,
    ) -> None:
        contours = {}
        for oid, nid in points:
            geometry = document.geometry_for(oid)
            subpath = self._contour_of(geometry, nid)
            contours.setdefault((geometry.id, subpath.id), (oid, nid))
        for oid, nid in contours.values():
            if has_node(tx.preview, oid, nid):
                tx.delete_contour(oid, nid)

    def _object_selection(self) -> Selection:
        # Object commands act on whole objects, whichever points are selected.
        return replace(self.editor.snapshot.selection, node_ids=frozenset())

    def _object_transaction(self, command: str) -> Transaction:
        return self.editor.transaction(
            ACTION_COMMANDS[command].label, selection=self._object_selection()
        )

    def _command_lines(self, payload: dict) -> None:
        self._lines(payload["command"], payload, self._object_selection().object_ids)

    def _command_add_path(self, payload: dict) -> None:
        document = self.editor.snapshot.document
        geometry = parse_path(str(payload.get("d", "")))
        # One contour as drawn, or several (a shape and its holes) as an
        # agent gives them.
        if not geometry.subpaths or any(len(s.nodes) < 2 for s in geometry.subpaths):
            raise DocumentError("Draw at least two path points")
        closed = all(s.closed for s in geometry.subpaths)
        paint = drawing_paint(
            payload,
            DRAWING_PAINT if closed else "none",
            "none" if closed else DRAWING_PAINT,
        )
        element = Element(
            new_id("object"), "path", tuple(paint.items()), geometry_id=geometry.id
        )
        parent = str(payload.get("parent") or document.root.id)
        if parent != document.root.id:
            # Creation inside a group intentionally adopts its clipping/opacity.
            # Preserve the same root-space coordinates as drawing at the top level.
            element = replace(
                element,
                attributes=(
                    *element.attributes,
                    ("transform", matrix_text(drawing_frame(document, parent))),
                ),
            )
        # Creation explicitly targets its container, independently of selection.
        with self.editor.transaction("Draw path", selection=Selection.all()) as tx:
            tx.insert_object(
                parent, element, index=payload.get("index"), geometries=(geometry,)
            )
        self.editor.select(Selection(object_ids=frozenset({element.id})))

    def _command_add_shape(self, payload: dict) -> None:
        """Draw a native rect, circle, ellipse or line, given in root user
        space, into the container *parent*: the shape stays one until
        converted to a path."""
        document = self.editor.snapshot.document
        tag = str(payload.get("shape", ""))
        if tag not in SHAPE_ATTRIBUTES:
            raise DocumentError("Draw a rect, circle, ellipse or line")
        values: dict[str, float] = {}
        for name in SHAPE_ATTRIBUTES[tag]:
            if payload.get(name) is None:
                if (tag, name) == ("rect", "rx"):
                    continue
                raise DocumentError(f"A {SHAPE_NAMES[tag]} needs {name}")
            try:
                values[name] = number(payload[name])
            except (TypeError, ValueError):
                raise DocumentError(f"{name} must be a number") from None
        if tag == "line":
            if (values["x1"], values["y1"]) == (values["x2"], values["y2"]):
                raise DocumentError("Drag out a line with some length")
        elif any(values[n] <= 0 for n in SHAPE_SIZES[tag]):
            raise DocumentError(
                f"Drag out a {SHAPE_NAMES[tag]} with some width and height"
            )
        if tag == "rect" and "rx" in values:
            # Corners round at most to a half side, as SVG draws them.
            values["rx"] = min(
                max(values["rx"], 0.0), values["width"] / 2, values["height"] / 2
            )
            if not values["rx"]:
                del values["rx"]
        parent = str(payload.get("parent") or document.root.id)
        a, b, c, d, e, f = drawing_frame(document, parent)
        scale = math.sqrt(abs(a * d - b * c))
        transform = None
        if abs(b) + abs(c) <= 1e-9 * scale and abs(abs(a) - abs(d)) <= 1e-9 * scale:
            # Moved, evenly scaled or mirrored: the group's frame takes the
            # shape's own attributes, so its numbers read as drawn.
            def at(x: float, y: float) -> tuple[float, float]:
                return a * x + e, d * y + f

            if tag == "line":
                values["x1"], values["y1"] = at(values["x1"], values["y1"])
                values["x2"], values["y2"] = at(values["x2"], values["y2"])
            elif tag == "rect":
                x1, y1 = at(values["x"], values["y"])
                x2, y2 = at(
                    values["x"] + values["width"], values["y"] + values["height"]
                )
                values.update(
                    x=min(x1, x2),
                    y=min(y1, y2),
                    width=abs(x2 - x1),
                    height=abs(y2 - y1),
                )
                if "rx" in values:
                    values["rx"] *= scale
            else:
                values["cx"], values["cy"] = at(values["cx"], values["cy"])
                for name in ("r", "rx", "ry"):
                    if name in values:
                        values[name] *= scale
        else:
            # Rotated or skewed: the shape keeps root coordinates under a
            # transform undoing the group's.
            transform = matrix_text((a, b, c, d, e, f))
            scale = 1.0
        if tag == "line":
            paint = drawing_paint(payload, "none", DRAWING_PAINT, scale)
            paint["fill"] = "none"
            if paint["stroke"] == "none":
                raise DocumentError("A line needs a stroke colour")
        else:
            paint = drawing_paint(payload, DRAWING_PAINT, "none", scale)
            if paint["fill"] == paint["stroke"] == "none":
                raise DocumentError("Give the shape a fill or a stroke colour")
        attributes = (
            *((name, length_text(value)) for name, value in values.items()),
            *paint.items(),
            *((("transform", transform),) if transform else ()),
        )
        element = Element(new_id("object"), tag, attributes)
        label = f"Draw {SHAPE_NAMES[tag]}"
        # Creation explicitly targets its container, independently of selection.
        with self.editor.transaction(label, selection=Selection.all()) as tx:
            tx.insert_object(parent, element, index=payload.get("index"))
        self.editor.select(Selection(object_ids=frozenset({element.id})))

    def _command_copy(self, _payload: dict) -> None:
        snapshot = self.editor.snapshot
        self.clipboard = copy_objects(snapshot.document, snapshot.selection.object_ids)

    def _command_paste(self, _payload: dict) -> None:
        if self.clipboard is None:
            raise DocumentError("Copy objects first")
        copied = paste_objects(self.clipboard)
        document = self.editor.snapshot.document
        previous = self.editor.snapshot.selection
        history = len(self.editor.undo_entries)
        with self.editor.transaction("Paste objects", selection=Selection.all()) as tx:
            tx.insert_objects(
                document.root.id, copied.root.children, geometries=copied.geometries
            )
        self.editor.select(
            Selection(
                object_ids=frozenset(
                    e.id for e in copied.root.children if e.tag != "defs"
                )
            )
        )
        self.editor.reselect(history, previous, self.editor.snapshot.selection)

    def _command_move_objects(self, payload: dict) -> None:
        document = self.editor.snapshot.document
        # Dragging rows in the tree names its objects, selected or not.
        objects = payload.get("objects")
        if not isinstance(objects, list) or not objects:
            raise DocumentError("Choose objects to move")
        moving = frozenset(str(oid) for oid in objects)
        parent = str(payload["parent"])
        if document.root.id in moving:
            raise DocumentError("Cannot move the document root")
        regrouped = any(
            ancestry[-2].id != parent
            for ancestry in map(document.ancestry, moving)
            if not any(a.id in moving for a in ancestry[:-1])
        )
        with self.editor.transaction(
            "Move into group" if regrouped else "Change stacking",
            selection=Selection(object_ids=moving),
        ) as tx:
            tx.move_objects(moving, parent, int(payload["index"]))
        self.editor.select(Selection(object_ids=moving))

    def _command_extract(self, payload: dict) -> None:
        # The contours inside a region, taken into paths of their own, or
        # deleted: the selected paths', or with scope "region" those of
        # every unlocked, shown path and shape there.
        polygon = self._region_polygon(payload)
        selection = self._region_scope(payload, polygon)
        delete = bool(payload.get("delete"))
        with self.editor.transaction(
            "Delete in region" if delete else "Extract region", selection=selection
        ) as tx:
            pairs = self._extract_region(tx, payload, polygon, delete=delete)
        existing = {e.id for e in self.editor.snapshot.document.elements()}
        result = {new or old for old, new in pairs} & existing
        self.editor.select(Selection(object_ids=frozenset(result)))

    @staticmethod
    def _region_polygon(payload: dict) -> list[tuple[float, float]]:
        region = payload.get("region")
        if not isinstance(region, list | tuple):
            raise DocumentError("Give the region as [x, y, w, h] or a polygon")
        return region_polygon(region)

    def _region_scope(
        self, payload: dict, polygon: list[tuple[float, float]]
    ) -> Selection:
        """What a region edit acts on: the selected objects (scope
        "selection", the default), or every path and basic shape inside the
        entered group *within* whose geometry and structure are unlocked
        (scope "region"), as the MCP extract tool chooses without ids.

        An instance draws shared geometry, so, as in the MCP tool, one the
        edit would reach is refused rather than left out: a selected one, or
        with scope "region" one painting inside the region.
        """
        document = self.editor.snapshot.document
        scope = payload.get("scope", "selection")
        if scope == "selection":
            selection = self._object_selection()
            if not selection.object_ids:
                raise DocumentError("Select an object first")
            instances = sorted(
                i for i in selection.object_ids if document.element(i).tag == "use"
            )
        elif scope == "region":
            within = payload.get("within") or document.root.id
            inside = {e.id for e in Document(document.element(str(within))).elements()}
            found = [
                e
                for e in document.elements()
                if e.id in inside
                and e.tag in SHAPES | {"path", "use"}
                and not any(
                    a.tag in {"defs", "clipPath"}
                    or a.locks & KNIFE_EDITS
                    or a.get("display") == "none"
                    for a in document.ancestry(e.id)
                )
            ]
            area = Polygon(polygon).buffer(0)
            hits = HitIndex(document) if any(e.tag == "use" for e in found) else None
            instances = [
                e.id
                for e in found
                if e.tag == "use"
                and hits is not None
                and (painted := hits.area(e.id)) is not None
                and painted.intersects(area)
            ]
            selection = Selection(
                object_ids=frozenset(e.id for e in found if e.tag != "use")
            )
        else:
            raise DocumentError('The scope is "selection" or "region"')
        if instances:
            labels = {o["id"]: o["label"] for o in self._object_state(document)}
            names = ", ".join(labels.get(i, i) for i in instances[:5])
            one = len(instances) == 1
            raise DocumentError(
                f"{names} {'is an instance' if one else 'are instances'} of shared"
                f" geometry: detach {'it' if one else 'them'} to editable paths first"
            )
        return selection

    @staticmethod
    def _extract_region(
        tx: Transaction,
        payload: dict,
        polygon: list[tuple[float, float]],
        *,
        cut: bool | None = None,
        delete: bool,
    ) -> tuple[tuple[str, str | None], ...]:
        if cut is None:
            cut = bool(payload.get("cut", True))
        try:
            return tx.extract_region(polygon, cut=cut, delete=delete)
        except EditRejectedError as exc:
            if payload.get("scope") != "region" or not str(exc).startswith(NO_CONTOUR):
                raise
            raise EditRejectedError(
                "No unlocked path or shape here has a contour inside the region"
                + ("" if cut else ", and cutting is off")
            ) from exc

    def region(self, payload: dict) -> dict:
        """What Extract region and Delete region would do with *payload*'s
        region, scope and cut, found by making each edit and dropping it.

        Each operation gives its targets, the paths it changes, each with
        *outcome* "cut" (pieces cut along the region's edge), "contours"
        (whole contours taken) or "whole" (the object goes), or the reason
        it is refused. *crossing* lists the objects the edge crosses, which
        are cut only with cut on (None when cutting would be refused).
        """
        self.check_revision(payload)
        polygon = self._region_polygon(payload)
        cut = payload.get("cut", True)
        if not isinstance(cut, bool):
            raise DocumentError("Choose whether to cut crossing objects")
        result: dict[str, Any] = {
            "epoch": self.epoch,
            "revision": self.editor.snapshot.revision,
        }
        try:
            selection = self._region_scope(payload, polygon)
        except DocumentError as exc:
            refused = {"targets": [], "error": str(exc)}
            return {**result, "extract": refused, "delete": refused, "crossing": None}
        # Cutting finds what the edge crosses, whether or not this edit cuts.
        probe = self._region_trial(selection, payload, polygon, cut=True, delete=False)
        result["extract"] = (
            probe
            if cut
            else self._region_trial(
                selection, payload, polygon, cut=False, delete=False
            )
        )
        result["delete"] = self._region_trial(
            selection, payload, polygon, cut=cut, delete=True
        )
        result["crossing"] = (
            None
            if probe["error"]
            else [t["id"] for t in probe["targets"] if t["outcome"] == "cut"]
        )
        return result

    def _region_trial(
        self,
        selection: Selection,
        payload: dict,
        polygon: list[tuple[float, float]],
        *,
        cut: bool,
        delete: bool,
    ) -> dict:
        """One region edit made and dropped: its targets, or why it is refused."""
        from vectrify.document.features import require_preserved

        tx = self.editor.transaction("Region preview", selection=selection)
        try:
            pairs = self._extract_region(tx, payload, polygon, cut=cut, delete=delete)
            require_preserved(tx.base_document, tx.preview)
        except DocumentError as exc:
            error = str(exc)
            if not cut and error.startswith(NO_CONTOUR):
                # Said with the Region panel's choice, not the MCP argument.
                error = (
                    f"{NO_CONTOUR}: turn on Cut objects crossing the edge to "
                    "take the parts inside"
                )
            return {"targets": [], "error": error}
        finally:
            tx.abort()
        before, after = tx.base_document, tx.preview
        present = {e.id for e in after.elements()}
        targets = []
        for old, new in pairs:
            if old not in present:
                outcome = "whole"
            else:
                # Contours taken whole keep their IDs; pieces cut along the
                # edge are new. A basic shape had no contours of its own.
                element = before.element(old)
                kept = (
                    set()
                    if element.tag in SHAPES
                    else {s.id for s in before.geometry_for(old).subpaths}
                )
                changed = after.geometry_for(old).subpaths + (
                    after.geometry_for(new).subpaths if new else ()
                )
                outcome = (
                    "cut" if any(s.id not in kept for s in changed) else "contours"
                )
            targets.append({"id": old, "outcome": outcome})
        return {"targets": targets, "error": None}

    def _command_fill_holes(self, payload: dict) -> None:
        selection = self._object_selection()
        selected = selection.object_ids
        document = self.editor.snapshot.document
        targets = self._hole_targets(payload, "be filled")
        cleanup = frozenset(payload.get("delete_objects", []))
        if cleanup:
            if len(targets) != 1:
                raise DocumentError("Delete enclosed shapes with one path's holes")
            ((oid, requested),) = targets.items()
            holes = tuple(h for h in find_holes(document, oid) if h.id in requested)
            if not cleanup <= enclosed_objects(document, oid, holes):
                raise DocumentError(
                    "Only shapes fully inside the chosen holes can be deleted"
                )
        with self.editor.transaction(
            "Fill holes", selection=Selection(object_ids=selected | cleanup)
        ) as tx:
            for oid, requested in targets.items():
                tx.fill_holes(oid, requested)
            if cleanup:
                tx.delete_objects(cleanup)

    def _command_holes_to_shapes(self, payload: dict) -> None:
        selection = self._object_selection()
        targets = self._hole_targets(payload, "become shapes")
        shapes: list[str] = []
        with self.editor.transaction("Holes to shapes", selection=selection) as tx:
            for oid, requested in targets.items():
                shapes.extend(tx.holes_to_shapes(oid, requested))
        self.editor.select(Selection(object_ids=frozenset(shapes)))

    def _command_cut_hole(self, payload: dict) -> None:
        selected = self._object_selection().object_ids
        with self._object_transaction(payload["command"]) as tx:
            outer = tx.cut_out_hole(selected)
        self.editor.select(Selection(object_ids=frozenset({outer})))

    def _command_boolean(self, payload: dict) -> None:
        # The backmost selected path is the target: it keeps its id and
        # paint, takes the result, and stays selected.
        selected = self._object_selection().object_ids
        with self._object_transaction(payload["command"]) as tx:
            target = tx.boolean_paths(selected, payload["command"])
        self.editor.select(Selection(object_ids=frozenset({target})))

    def _command_paint(self, payload: dict) -> None:
        if "gradient" in payload:
            self._paint_gradient(payload["gradient"])
            return
        selected = self._object_selection().object_ids
        with self._object_transaction(payload["command"]) as tx:
            changes = payload["changes"]
            if (
                not isinstance(changes, dict)
                or not changes.keys()
                <= {
                    "fill",
                    "stroke",
                    "stroke-width",
                    "opacity",
                    "fill-opacity",
                    "stroke-opacity",
                    "stroke-linecap",
                    "stroke-linejoin",
                    "stroke-miterlimit",
                }
                | ARROWS
            ):
                raise DocumentError("Unsupported paint property")
            fill = changes.get("fill")
            if isinstance(fill, dict):
                try:
                    x1, y1 = (number(v) for v in fill["start"])
                    x2, y2 = (number(v) for v in fill["end"])
                    fill = LinearGradient(
                        (x1, y1),
                        (x2, y2),
                        tuple(
                            GradientStop(
                                number(stop["offset"]),
                                stop["colour"],
                                number(stop.get("opacity", 1)),
                            )
                            for stop in fill["stops"]
                        ),
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise DocumentError("Invalid gradient fill") from exc
            elif fill is not None and not isinstance(fill, str):
                raise DocumentError("A fill needs a colour or gradient")
            others = {k: v for k, v in changes.items() if k != "fill"}
            for oid in selected:
                if others:
                    tx.set_attributes(oid, others)
                if "fill" in changes:
                    # Its own gradient, if it had one, goes with a new fill.
                    old = tx.preview.element(oid)
                    follows = (
                        isinstance(fill, LinearGradient)
                        and old.get("stroke") not in {None, "none"}
                        and old.get("stroke") == old.get("fill")
                    )
                    tx.set_fill(oid, fill)
                    if follows:
                        tx.set_attributes(
                            oid, {"stroke": tx.preview.element(oid).get("fill")}
                        )

    def _paint_gradient(self, spec: dict) -> None:
        """An explicit shared-paint edit authorizes its users without selecting them.

        An edit without *stops*, as dragging an endpoint on the canvas sends,
        keeps the stops as they are.
        """
        document = self.editor.snapshot.document
        selected = self._object_selection().object_ids
        try:
            gradient_id = spec["id"]
            attributes = spec["attributes"]
            stops = (
                tuple(
                    GradientStop(
                        number(stop["offset"]),
                        stop["colour"],
                        number(stop.get("opacity", 1)),
                    )
                    for stop in spec["stops"]
                )
                if "stops" in spec
                else None
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DocumentError("Invalid gradient edit") from exc
        if not isinstance(attributes, dict) or any(
            not isinstance(value, str) and value is not None
            for value in attributes.values()
        ):
            raise DocumentError("Invalid gradient attributes")
        # A shape may edit its inherited fill, or the resource can be selected.
        if not any(
            oid == gradient_id
            or paint_server(path_style(document, document.element(oid))["fill"])
            == gradient_id
            for oid in selected
        ):
            raise DocumentError("Select the gradient or a shape using its fill")
        affected = document.dependents({gradient_id})
        with self.editor.transaction(
            "Edit gradient", selection=Selection(object_ids=affected)
        ) as tx:
            tx.set_gradient(gradient_id, attributes, stops)

    def _command_move(self, payload: dict) -> None:
        """Move the selection by *dx*, *dy* in document coordinates.

        Each object takes the move in its parent's frame, so objects inside
        transformed groups move as far on the page as the others. A drag gives
        *offsets* instead: each object's own, already in its parent's frame.
        """
        selected = self._object_selection().object_ids
        document = self.editor.snapshot.document
        with self._object_transaction(payload["command"]) as tx:
            dx, dy = number(payload["dx"]), number(payload["dy"])
            offsets = payload.get("offsets", {})
            # Avoid translating a selected child twice if its group is selected.
            for oid in selected:
                ancestry = document.ancestry(oid)
                if any(a.id in selected for a in ancestry[:-1]):
                    continue
                if oid in offsets:
                    move_x, move_y = (number(v) for v in offsets[oid])
                    translate(tx, ancestry, move_x, move_y)
                else:
                    translate(tx, ancestry, *parent_offset(ancestry, dx, dy))

    def _command_align(self, payload: dict) -> None:
        """Move each object by its own offset as one edit.

        *moves* maps selected ids to [dx, dy] in document coordinates: Align
        and Distribute measure the objects in the window and send how far
        each goes. Selected objects without a move, such as the key object,
        stay where they are, locked or not.
        """
        selected = self._object_selection().object_ids
        document = self.editor.snapshot.document
        moves = payload.get("moves")
        if not isinstance(moves, dict) or not moves:
            raise DocumentError("Nothing to move: the objects are already in place")
        with self._object_transaction(payload["command"]) as tx:
            for oid, offset in moves.items():
                if oid not in selected:
                    raise DocumentError("Align and distribute move selected objects")
                ancestry = document.ancestry(oid)
                if any(a.id in selected for a in ancestry[:-1]):
                    raise DocumentError(
                        f"{oid} moves with its selected group, not on its own"
                    )
                locked = next(
                    (
                        e
                        for e in (*ancestry, *Document(ancestry[-1]).elements())
                        if EditKind.TRANSFORM in e.locks
                    ),
                    None,
                )
                if locked is not None:
                    raise DocumentError(
                        f"{locked.name or locked.id}: position is locked; "
                        "unlock it to move"
                    )
                dx, dy = (number(v) for v in offset)
                translate(tx, ancestry, *parent_offset(ancestry, dx, dy))

    def _command_resize(self, payload: dict) -> None:
        selected = self._object_selection().object_ids
        with self._object_transaction(payload["command"]) as tx:
            anchor = tuple(number(v) for v in payload["anchor"])
            scale = tuple(number(v) for v in payload["scale"])
            if len(anchor) != 2 or len(scale) != 2:
                raise DocumentError("A resize needs an anchor and two scales")
            tx.scale_objects(selected, (anchor[0], anchor[1]), (scale[0], scale[1]))

    def _pivot(self, payload: dict) -> tuple[float, float]:
        pivot = tuple(number(v) for v in payload["pivot"])
        if len(pivot) != 2:
            raise DocumentError("A pivot is a point [x, y]")
        return pivot[0], pivot[1]

    def _command_rotate(self, payload: dict) -> None:
        """Rotate the selection by *angle* degrees, clockwise on screen, about
        *pivot* in document coordinates."""
        selected = self._object_selection().object_ids
        angle = number(payload["angle"])
        if angle % 360 == 0:
            raise DocumentError("Rotate by an angle that is not a whole turn")
        with self._object_transaction(payload["command"]) as tx:
            tx.transform_objects(
                selected, self._pivot(payload), rotation(angle), verb="rotate"
            )

    def _command_flip(self, payload: dict) -> None:
        """Mirror the selection across the vertical (flip_horizontal) or
        horizontal (flip_vertical) line through *pivot*."""
        selected = self._object_selection().object_ids
        axis = payload["command"].removeprefix("flip_")
        with self._object_transaction(payload["command"]) as tx:
            tx.transform_objects(
                selected, self._pivot(payload), REFLECTIONS[axis], verb="flip"
            )

    def _command_group(self, payload: dict) -> None:
        selected = self._object_selection().object_ids
        with self._object_transaction(payload["command"]) as tx:
            group_id = tx.group_objects(selected)
        self.editor.select(Selection(object_ids=frozenset({group_id})))

    def _command_ungroup(self, payload: dict) -> None:
        selected = self._object_selection().object_ids
        with self._object_transaction(payload["command"]) as tx:
            for oid in selected:
                tx.ungroup_object(oid)

    def _command_delete(self, payload: dict) -> None:
        selected = self._object_selection().object_ids
        with self._object_transaction(payload["command"]) as tx:
            tx.delete_objects(
                selected, allow_pinned=payload.get("confirm_pinned") is True
            )

    def _command_restack(self, payload: dict) -> None:
        selected = self._object_selection().object_ids
        document = self.editor.snapshot.document
        with self._object_transaction(payload["command"]) as tx:
            command = payload["command"]
            # Each container's selected children go to its front or back
            # together, in their current order.
            groups: dict[str, set[str]] = {}
            for oid in selected:
                ancestry = document.ancestry(oid)
                if len(ancestry) < 2:
                    raise DocumentError("Cannot restack the document root")
                if not any(a.id in selected for a in ancestry[:-1]):
                    groups.setdefault(ancestry[-2].id, set()).add(oid)
            for parent, children in groups.items():
                others = len(document.element(parent).children) - len(children)
                tx.move_objects(
                    frozenset(children),
                    parent,
                    others if command == "to_front" else 0,
                )

    def _command_reorder(self, payload: dict) -> None:
        if payload.get("to") in {"front", "back"}:
            self._command_restack({**payload, "command": f"to_{payload['to']}"})
            return
        selected = self._object_selection().object_ids
        document = self.editor.snapshot.document
        with self._object_transaction(payload["command"]) as tx:
            if len(selected) != 1:
                raise DocumentError("Select one object to change its stacking order")
            oid = next(iter(selected))
            siblings = document.ancestry(oid)[-2].children
            index = next(i for i, item in enumerate(siblings) if item.id == oid)
            target = max(0, min(len(siblings) - 1, index + int(payload["step"])))
            tx.reorder_object(oid, target)

    def _command_detach(self, payload: dict) -> None:
        selected = self._object_selection().object_ids
        with self._object_transaction(payload["command"]) as tx:
            for oid in selected:
                tx.detach_geometry(oid)

    def _command_convert_to_path(self, payload: dict) -> None:
        # Basic shapes only: instances and shared geometry are Detach's, so
        # the name says what the edit does.
        selected = self._object_selection().object_ids
        document = self.editor.snapshot.document
        if others := sorted(
            i for i in selected if document.element(i).tag not in SHAPES
        ):
            raise DocumentError(
                f"{others[0]}: Convert to path takes rectangles, circles, "
                "ellipses and lines; detach instances and shared geometry"
            )
        with self._object_transaction(payload["command"]) as tx:
            for oid in sorted(selected):
                tx.detach_geometry(oid)

    def _command_set_clip(self, payload: dict) -> None:
        # The person names each selected object's role, so which is clipped
        # and which is the boundary is never guessed here.
        selected = self._object_selection().object_ids
        targets, boundaries = (
            self._clip_role(payload, "targets"),
            self._clip_role(payload, "boundaries"),
        )
        if set(targets) & set(boundaries):
            raise DocumentError("An object is either clipped or a boundary, not both")
        if set(targets) | set(boundaries) != selected:
            raise DocumentError(
                "Give each selected object one role: clipped or boundary"
            )
        linked = payload.get("linked", True)
        if not isinstance(linked, bool):
            raise DocumentError("Choose linked or copied boundary geometry")
        apply_clip(
            self.editor,
            sorted(targets),
            boundaries,
            linked=linked,
            label=ACTION_COMMANDS[payload["command"]].label,
        )

    @staticmethod
    def _clip_role(payload: dict, role: str) -> list[str]:
        ids = payload.get(role)
        if (
            not isinstance(ids, list)
            or not ids
            or not all(isinstance(i, str) for i in ids)
        ):
            name = "objects to clip" if role == "targets" else "boundary paths"
            raise DocumentError(f"Choose the {name}")
        return ids

    def _command_remove_clip(self, payload: dict) -> None:
        # Only the clipped objects change: an unclipped one in the selection
        # neither blocks the edit with its locks nor leaves the selection.
        selection = self._object_selection()
        document = self.editor.snapshot.document
        clipped = sorted(
            i
            for i in selection.object_ids
            if document.element(i).get("clip-path", "none") != "none"
        )
        if not clipped:
            raise DocumentError("None of the selected objects is clipped")
        apply_clip(
            self.editor,
            clipped,
            action="remove",
            label=ACTION_COMMANDS[payload["command"]].label,
        )
        self.editor.select(selection)

    def _command_split_disconnected(self, payload: dict) -> None:
        selected = self._object_selection().object_ids
        with self._object_transaction(payload["command"]) as tx:
            for oid in sorted(selected):
                tx.split_disconnected(oid)

    def _command_combine_paths(self, payload: dict) -> None:
        selected = self._object_selection().object_ids
        source = payload.get("paint_source")
        if source is not None and (not isinstance(source, str) or not source):
            raise DocumentError("Choose a paint source from the selected paths")
        with self._object_transaction(payload["command"]) as tx:
            tx.combine_paths(selected, paint_source=source)

    def _command_join_paths(self, payload: dict) -> None:
        selected = self._object_selection().object_ids
        with self._object_transaction(payload["command"]) as tx:
            options = payload.get("options", {})
            if not isinstance(options, dict) or set(options) - {
                "colors",
                "color_source",
            }:
                raise DocumentError("Invalid join options")
            mode = options.get("colors", "mix")
            source = options.get("color_source")
            if (
                not isinstance(mode, str)
                or mode not in {"mix", "source"}
                or (mode == "source" and (not isinstance(source, str) or not source))
                or (mode == "mix" and source is not None)
            ):
                raise DocumentError("Choose mixed colors or a selected color source")
            tx.join_paths(selected, color_source=source)

    def _knife(self, payload: dict) -> None:
        """Cut the paths the knife line crosses: the selected ones, or with
        nothing selected every shown drawing path whose geometry and structure
        are unlocked, inside the entered group *within* if one is given."""
        start, end = (
            tuple(number(v) for v in payload[key]) for key in ("start", "end")
        )
        if len(start) != 2 or len(end) != 2:
            raise DocumentError("A knife line needs two points")
        document = self.editor.snapshot.document
        selection = replace(self.editor.snapshot.selection, node_ids=frozenset())
        chosen = bool(selection.object_ids)
        if not chosen:
            within = payload.get("within") or document.root.id
            inside = {e.id for e in Document(document.element(str(within))).elements()}
            selection = Selection(
                object_ids=frozenset(
                    e.id
                    for e in document.elements()
                    if e.tag == "path"
                    and e.id in inside
                    and not any(
                        a.tag in {"defs", "clipPath"}
                        or a.locks & KNIFE_EDITS
                        or a.get("display") == "none"
                        for a in document.ancestry(e.id)
                    )
                )
            )
        try:
            with self.editor.transaction("Cut with knife", selection=selection) as tx:
                pieces = tx.cut_paths((start[0], start[1]), (end[0], end[1]))
        except EditRejectedError as exc:
            if chosen or "Drag the knife across" not in str(exc):
                raise
            raise EditRejectedError(
                "The knife line crosses no line, and no filled shape from "
                "outside to outside, that it can cut"
            ) from exc
        self.editor.select(Selection(object_ids=frozenset(pieces)))

    def _lines(self, command: str, payload: dict, selected: frozenset[str]) -> None:
        """Join the selected lines' ends, or turn thin fills into strokes and
        strokes into fills, as one undoable edit."""
        document = self.editor.snapshot.document
        selection = Selection(object_ids=selected)
        paths = [
            e
            for e in document.elements()
            if e.tag == "path"
            and e.id in document.selection_ids(selection)
            and not any(a.tag in {"defs", "clipPath"} for a in document.ancestry(e.id))
        ]
        if command == "join_ends":
            reach = number(payload.get("reach", 0))
            if reach < 0:
                raise DocumentError("The reach must not be negative")
            with self.editor.transaction(
                ACTION_COMMANDS[command].label, selection=selection
            ) as tx:
                joined = tx.join_ends(
                    selected, reach, curve=payload.get("bridge", "curve") == "curve"
                )
            self.editor.select(Selection(object_ids=frozenset(joined)))
            return
        # Convert flips each path: a fill becomes a line and a line a fill.
        wanted = {
            "fill_to_line": {True},
            "line_to_fill": {False},
            "convert_lines": {True, False},
        }[command]

        def is_fill(path) -> bool | None:
            style = path_style(document, path)
            if style["fill"] != "none":
                return True
            return False if style["stroke"] != "none" else None

        chosen = [p for p in paths if is_fill(p) in wanted]
        if not chosen:
            raise DocumentError(
                {
                    "fill_to_line": "Select filled paths to turn into lines",
                    "line_to_fill": "Select stroked paths without a fill to turn "
                    "into fills",
                    "convert_lines": "Select filled paths or stroked lines",
                }[command]
            )
        label = ACTION_COMMANDS[command].label
        with self.editor.transaction(label, selection=selection) as tx:
            for path in chosen:
                style = path_style(document, path)
                geometry = document.geometry_for(path.id)
                if is_fill(path):
                    line, width = centreline(geometry, style["fill-rule"])
                    changes = {
                        "fill": "none",
                        "stroke": style["fill"],
                        "stroke-opacity": style["fill-opacity"],
                        "stroke-width": f"{width:.4g}",
                        "stroke-linecap": "round",
                        "stroke-linejoin": "round",
                        "fill-opacity": None,
                        "fill-rule": None,
                    }
                else:
                    line = stroke_outline(geometry, style)
                    changes = {
                        "fill": style["stroke"],
                        "fill-opacity": style["stroke-opacity"],
                        "fill-rule": "nonzero",
                        "stroke": "none",
                        "stroke-opacity": None,
                        "stroke-width": None,
                        "stroke-linecap": None,
                        "stroke-linejoin": None,
                        "stroke-miterlimit": None,
                        **dict.fromkeys(ARROWS),
                    }
                tx.replace_geometry(path.id, line)
                tx.set_attributes(path.id, changes)

    def _hole_targets(self, payload: dict, verb: str) -> dict[str, frozenset[str]]:
        """The holes a hole command acts on, by path.

        *contours* lists (object, hole) pairs, possibly across several paths
        selected or inside a selected group; one path's holes may be given
        as *object* and *holes* instead. Paths sharing geometry share their
        holes, so each geometry's holes are taken once, by its first path.
        """
        raw = payload.get("contours")
        if raw is None:
            raw = [[payload.get("object"), hole] for hole in payload.get("holes", [])]
        if (
            not isinstance(raw, list)
            or not raw
            or any(not isinstance(p, (list, tuple)) or len(p) != 2 for p in raw)
        ):
            raise DocumentError(f"Choose the holes that should {verb}")
        document = self.editor.snapshot.document
        by_geometry: dict[str, tuple[str, set[str]]] = {}
        for oid, hole in raw:
            oid = str(oid)
            if not self._in_selection(oid):
                raise DocumentError(f"Select the path whose holes should {verb}")
            geometry = document.geometry_for(oid).id
            by_geometry.setdefault(geometry, (oid, set()))[1].add(str(hole))
        return {oid: frozenset(holes) for oid, holes in by_geometry.values()}

    def _redraw_outline(self, payload: dict) -> None:
        """Redraw the stretch of a selected path's contour a stroke runs along.

        The stroke is in root user space; *pixel* is a screen pixel's size
        there. Each end attaches to the nearest point within NODE_REACH
        screen pixels, else the nearest place on the outline within REACH,
        both on the contour the start attaches to.
        """
        document = self.editor.snapshot.document
        oid = payload["object"]
        # The stroke picks the path it starts on; a selection narrows that to
        # the selected paths.
        if self.editor.snapshot.selection.object_ids and not self._in_selection(oid):
            raise DocumentError("Start on the outline of a selected path")
        if any(a.tag in {"defs", "clipPath"} for a in document.ancestry(oid)):
            raise DocumentError("Redraw drawing paths, not definitions")
        if document.element(oid).tag != "path":
            raise DocumentError("Redraw works on a path; convert the shape first")
        stroke = payload.get("points")
        if not isinstance(stroke, list) or len(stroke) < 2:
            raise DocumentError("Draw along the outline to redraw it")
        if any(not isinstance(p, (list, tuple)) or len(p) != 2 for p in stroke):
            raise DocumentError("Stroke points need two coordinates")
        points = [(number(x), number(y)) for x, y in stroke]
        pixel = number(payload.get("pixel", 1))
        if pixel <= 0:
            raise DocumentError("The screen pixel size must be positive")
        start = attachment(document, oid, points[0], REACH * pixel, NODE_REACH * pixel)
        end = start and attachment(
            document,
            oid,
            points[-1],
            REACH * pixel,
            NODE_REACH * pixel,
            start.subpath_id,
        )
        if start is None or end is None:
            raise DocumentError(
                "Start and end the stroke on the same outline of the selected path"
            )
        # Check explicit features before fitting, which may be expensive.
        from vectrify.document.redraw import redrawn
        from vectrify.document.regions import object_matrix

        geometry = document.geometry_for(oid)
        _, removed = redrawn(
            geometry,
            start.subpath_id,
            (start.node_id, start.t),
            (end.node_id, end.t),
            [("L", (0, 0))],
            matrix=object_matrix(document, oid),
            long_way=bool(payload.get("long_way")),
        )
        for nid in removed:
            node = geometry.node(nid)
            if node.feature is not None:
                raise DocumentError(
                    f"Protected {node.feature} {nid} would be removed; "
                    "redraw around it or release protection"
                )
        stretch = redraw_stretch(
            document,
            oid,
            [start.point, *points[1:-1], end.point],
            self.reference_image(),
            pixel,
        )
        gid = document.geometry_for(oid).id
        scope = frozenset({oid}) | document.geometry_users(gid)
        # The other selected paths, and a selected group, stay selected.
        kept = self.editor.snapshot.selection.object_ids
        with self.editor.transaction(
            "Redraw outline", selection=Selection(object_ids=scope)
        ) as tx:
            tx.redraw_outline(
                oid,
                start.subpath_id,
                (start.node_id, start.t),
                (end.node_id, end.t),
                stretch,
                long_way=bool(payload.get("long_way")),
            )
        existing = {e.id for e in self.editor.snapshot.document.elements()}
        self.editor.select(Selection(object_ids=(kept & existing) or frozenset({oid})))
