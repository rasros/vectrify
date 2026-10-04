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
from dataclasses import asdict, dataclass, replace
from threading import RLock
from typing import Any
from uuid import uuid4

from PIL import Image

from vectrify.document import (
    Document,
    DocumentError,
    Editor,
    EditRejectedError,
    Element,
    Selection,
    StaleRevisionError,
    export_svg,
    import_svg,
    load_project,
    save_project,
)
from vectrify.document.editor import Transaction
from vectrify.document.holes import document_hole_shape, enclosed_objects, find_holes
from vectrify.document.join import path_style
from vectrify.document.lines import stroke_outline
from vectrify.document.model import EditKind, new_id
from vectrify.document.redraw import attachment
from vectrify.document.regions import region_polygon
from vectrify.document.svg import parse_path
from vectrify.operations import (
    Budget,
    Job,
    Method,
    OperationRequest,
    Permissions,
    method,
)
from vectrify.refine.centreline import centreline
from vectrify.refine.redraw import redraw_stretch

MAX_SOURCE = 128 * 1024 * 1024
# Aspect ratios this close (relatively) count as the same shape.
ASPECT_SLACK = 0.005
# How near, in screen pixels, a redraw stroke's ends attach to a point of the
# outline, else to the outline itself.
NODE_REACH = 6
REACH = 10


# The edits the knife makes to a path it cuts: with nothing selected, it
# leaves paths locked against them alone.
KNIFE_EDITS = frozenset({EditKind.GEOMETRY, EditKind.STRUCTURE})


# Commands on points, possibly in several selected paths, and their undo labels.
POINT_COMMANDS = {
    "node": "Edit node",
    "move_nodes": "Move points",
    "split": "Split edge",
    "node_handles": "Change handles",
    "delete_node": "Delete node",
    "delete_contour": "Delete contour",
    "break_points": "Break at point",
    "delete_segment": "Delete segment",
    "join_two_ends": "Join ends",
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
    "pin": Command("_command_pin"),
    "reference": Command("_command_reference"),
    **{
        name: Command("_command_points", label)
        for name, label in POINT_COMMANDS.items()
    },
    "add_path": Command("_command_add_path", "Draw path"),
    "move_objects": Command("_command_move_objects"),
    "knife": Command("_knife", "Cut with knife"),
    "redraw_outline": Command("_redraw_outline", "Redraw outline"),
    "join_ends": Command("_command_lines", "Join ends", True),
    "fill_to_line": Command("_command_lines", "Fill to line", True),
    "line_to_fill": Command("_command_lines", "Line to fill", True),
    "convert_lines": Command("_command_lines", "Convert line/fill", True),
    "extract": Command("_command_extract", needs_selection=True),
    "fill_holes": Command("_command_fill_holes", "Fill holes", True),
    "holes_to_shapes": Command("_command_holes_to_shapes", "Holes to shapes", True),
    "cut_hole": Command("_command_cut_hole", "Cut out as hole", True),
    "paint": Command("_command_paint", "Change paint", True),
    "move": Command("_command_move", "Move selection", True),
    "resize": Command("_command_resize", "Resize", True),
    "group": Command("_command_group", "Group objects", True),
    "ungroup": Command("_command_ungroup", "Ungroup objects", True),
    "delete": Command("_command_delete", "Delete selection", True),
    "reorder": Command("_command_reorder", "Change stacking", True),
    "to_front": Command("_command_restack", "Bring to front", True),
    "to_back": Command("_command_restack", "Send to back", True),
    "detach": Command("_command_detach", "Detach geometry", True),
    "split_disconnected": Command(
        "_command_split_disconnected", "Split disconnected parts", True
    ),
    "join_paths": Command("_command_join_paths", "Join outlines", True),
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


class Session:
    def __init__(
        self,
        document: Document,
        name: str = "Untitled.svg",
        reference: dict | None = None,
    ):
        self.editor = Editor(document)
        self.name = name
        self.reference = self.fit_reference(reference) if reference else None
        self.epoch = uuid4().hex
        self.lock = RLock()
        self._svg_revision = -1
        self._svg = ""
        self._objects_document: Document | None = None
        self._objects: list[dict] = []
        self.jobs: dict[str, Job] = {}
        # What the person's window last said it shows (see set_view), and when.
        self.view: dict[str, Any] | None = None
        self.view_time = 0.0

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
            if reference in {"drawing", "overlay", "reference"}
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
        snapshot = self.editor.snapshot
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
        if not self.reference:
            return None
        data = base64.b64decode(self.reference["data_url"].split(",", 1)[1])
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            return image.copy()

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
        for element in document.elements():
            if element.tag == "svg":
                continue
            counters[element.tag] = counters.get(element.tag, 0) + 1
            ancestors = document.ancestry(element.id)
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
                    "shared": element.geometry_id is not None
                    and users[element.geometry_id] > 1,
                    "attributes": dict(element.attributes),
                    "locks": sorted(element.locks),
                    "inherited_locks": sorted(
                        set().union(*(a.locks for a in ancestors))
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
            "epoch": self.epoch,
            "revision": snapshot.revision,
            "name": self.name,
            "root": root.id,
            "bounds": bounds,
            "objects": objects,
            "selection": {
                "objects": sorted(snapshot.selection.object_ids),
                "nodes": sorted(snapshot.selection.node_ids),
            },
            "undo": list(self.editor.undo_labels),
            "redo": list(self.editor.redo_labels),
            "reference": {
                "name": self.reference["name"],
                "opacity": self.reference["opacity"],
            }
            if self.reference
            else None,
        }
        if svg:
            if self._svg_revision != snapshot.revision:
                self._svg = export_svg(snapshot.document)
                self._svg_revision = snapshot.revision
            result["svg"] = self._svg
        return result

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
        if (
            payload.get("epoch") != self.epoch
            or payload.get("revision") != self.editor.snapshot.revision
        ):
            raise StaleRevisionError(
                "The drawing changed. Refresh before applying this edit."
            )

    def open(self, source: str, name: str) -> None:
        if len(source.encode()) > MAX_SOURCE:
            raise DocumentError("File exceeds the 128 MB editor limit")
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
        for job in self.jobs.values():
            job.stop.set()
        self.jobs = {}
        self.editor, self.name, self.reference = editor, name, reference
        self.epoch = uuid4().hex
        self._svg_revision = -1

    def fit_reference(self, reference: dict) -> dict:
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
        return {
            "name": str(value.get("name", "Reference")),
            "data_url": url,
            "opacity": opacity,
        }

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
        )

    def action(self, payload: dict) -> dict:
        self.check_revision(payload)
        command = payload["command"]
        spec = ACTION_COMMANDS.get(command) if isinstance(command, str) else None
        if spec is None:
            raise DocumentError("Unknown editor command")
        if spec.needs_selection and not self.editor.snapshot.selection.object_ids:
            raise DocumentError("Select an object first")
        before = self.editor.snapshot.revision
        getattr(self, spec.handler)(payload)
        return self.state(
            svg=command == "open" or self.editor.snapshot.revision != before
        )

    def _command_select(self, payload: dict) -> None:
        self.editor.select(
            Selection(
                object_ids=frozenset(payload.get("objects", [])),
                node_ids=frozenset(payload.get("nodes", [])),
            )
        )

    def _command_open(self, payload: dict) -> None:
        self.open(payload["source"], str(payload.get("name", "Untitled.svg")))

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
        self.reference = reference

    def _command_history(self, payload: dict) -> None:
        getattr(self.editor, payload["command"])()

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
        changes: dict = {}
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
        if command == "node" and len(points) != 1:
            raise DocumentError("Drag one point at a time, or move them together")
        objects = frozenset(o for o, _ in points)
        # Editing a path's nodes changes every object drawing its geometry.
        scope = objects.union(
            *(document.geometry_users(document.geometry_for(o).id) for o in objects)
        )
        with self.editor.transaction(
            POINT_COMMANDS[command], selection=Selection(object_ids=scope)
        ) as tx:
            if command == "node":
                oid, nid = points[0]
                tx.update_node(oid, nid, tuple(number(v) for v in payload["values"]))
            elif command == "move_nodes":
                # The values already carry the handles each point takes along.
                # A node of shared geometry moves once, as its first path has it.
                kept = set(points)
                for oid, nodes in changes.items():
                    moved = {
                        str(nid): tuple(number(v) for v in values)
                        for nid, values in nodes.items()
                        if (str(oid), str(nid)) in kept
                    }
                    if moved:
                        tx.update_nodes(str(oid), moved)
            elif command == "split":
                for oid, nid in points:
                    tx.split_edge(oid, nid)
            elif command == "node_handles":
                for oid, nid in points:
                    tx.set_node_handles(oid, nid, int(payload["count"]))
            elif command == "delete_node":
                # Deleting one point can take its contour, or its path, along.
                for oid, nid in points:
                    if has_node(tx.preview, oid, nid):
                        tx.delete_node(oid, nid)
            elif command in {"break_points", "delete_segment"}:
                by_object: dict[str, list[str]] = {}
                for oid, nid in points:
                    by_object.setdefault(oid, []).append(nid)
                for oid, nids in by_object.items():
                    if command == "break_points":
                        tx.break_points(oid, nids)
                    else:
                        tx.delete_segments(oid, frozenset(nids))
            elif command == "join_two_ends":
                if len(points) != 2 or points[0] == points[1]:
                    raise DocumentError("Select the two points to join")
                # Any two points join: one that is not a free end yet is
                # cut there first, opening a closed contour or splitting a
                # line, so it becomes one.
                ends = []
                for oid, nid in points:
                    copies: dict[str, set[str]] = {}
                    with contextlib.suppress(EditRejectedError):
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
                tx.join_ends(objects, ends=ends)
            elif command == "delete_contour":
                contours = {}
                for oid, nid in points:
                    geometry = document.geometry_for(oid)
                    subpath = next(
                        s
                        for s in geometry.subpaths
                        if any(n.id == nid for n in s.nodes)
                    )
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
        width = number(payload.get("stroke_width", 2))
        if width <= 0:
            raise DocumentError("Stroke width must be positive")
        element = Element(
            new_id("object"),
            "path",
            (
                ("fill", "#83b899" if closed else "none"),
                ("stroke", "none" if closed else "#83b899"),
                ("stroke-width", str(width)),
            ),
            geometry_id=geometry.id,
        )
        # Creation explicitly targets the drawing, independently of selection.
        with self.editor.transaction("Draw path", selection=Selection.all()) as tx:
            tx.insert_object(document.root.id, element, geometries=(geometry,))
        self.editor.select(Selection(object_ids=frozenset({element.id})))

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
        selection = self._object_selection()
        # The selected paths' contours inside a region, taken into paths
        # of their own, or deleted.
        region = payload.get("region")
        if not isinstance(region, list | tuple):
            raise DocumentError("Give the region as [x, y, w, h] or a polygon")
        delete = bool(payload.get("delete"))
        with self.editor.transaction(
            "Delete in region" if delete else "Extract region", selection=selection
        ) as tx:
            pairs = tx.extract_region(
                region_polygon(region),
                cut=bool(payload.get("cut", True)),
                delete=delete,
            )
        existing = {e.id for e in self.editor.snapshot.document.elements()}
        result = {new or old for old, new in pairs} & existing
        self.editor.select(Selection(object_ids=frozenset(result)))

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

    def _command_paint(self, payload: dict) -> None:
        selected = self._object_selection().object_ids
        with self._object_transaction(payload["command"]) as tx:
            changes = payload["changes"]
            if not isinstance(changes, dict) or not changes.keys() <= {
                "fill",
                "stroke",
                "stroke-width",
                "opacity",
                "fill-opacity",
                "stroke-opacity",
            }:
                raise DocumentError("Unsupported paint property")
            others = {k: v for k, v in changes.items() if k != "fill"}
            for oid in selected:
                if others:
                    tx.set_attributes(oid, others)
                if "fill" in changes:
                    # Its own gradient, if it had one, goes with a new fill.
                    tx.set_fill(oid, changes["fill"])

    def _command_move(self, payload: dict) -> None:
        selected = self._object_selection().object_ids
        document = self.editor.snapshot.document
        with self._object_transaction(payload["command"]) as tx:
            dx, dy = number(payload["dx"]), number(payload["dy"])
            # Avoid translating a selected child twice if its group is selected.
            for oid in selected:
                if any(a.id in selected for a in document.ancestry(oid)[:-1]):
                    continue
                offset = payload.get("offsets", {}).get(oid, (dx, dy))
                move_x, move_y = (number(v) for v in offset)
                previous = document.element(oid).get("transform", "") or ""
                transform = f"translate({move_x} {move_y}) {previous}".strip()
                tx.set_attributes(oid, {"transform": transform})

    def _command_resize(self, payload: dict) -> None:
        selected = self._object_selection().object_ids
        with self._object_transaction(payload["command"]) as tx:
            anchor = tuple(number(v) for v in payload["anchor"])
            scale = tuple(number(v) for v in payload["scale"])
            if len(anchor) != 2 or len(scale) != 2:
                raise DocumentError("A resize needs an anchor and two scales")
            tx.scale_objects(selected, (anchor[0], anchor[1]), (scale[0], scale[1]))

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
            tx.delete_objects(selected)

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

    def _command_split_disconnected(self, payload: dict) -> None:
        selected = self._object_selection().object_ids
        with self._object_transaction(payload["command"]) as tx:
            for oid in sorted(selected):
                tx.split_disconnected(oid)

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
        nothing selected every drawing path whose geometry and structure are
        unlocked, inside the entered group *within* if one is given."""
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
                        a.tag in {"defs", "clipPath"} or a.locks & KNIFE_EDITS
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
            with self.editor.transaction("Join ends", selection=selection) as tx:
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
        label = {
            "fill_to_line": "Fill to line",
            "line_to_fill": "Line to fill",
            "convert_lines": "Convert line/fill",
        }[command]
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
