"""Small UI-facing command adapter; document constraints remain authoritative."""

from __future__ import annotations

import base64
import io
import json
import math
from dataclasses import asdict, replace
from threading import RLock
from typing import Any
from uuid import uuid4

from PIL import Image

from vectrify.document import (
    Document,
    DocumentError,
    Editor,
    Element,
    Rect,
    Selection,
    StaleRevisionError,
    export_svg,
    import_svg,
    load_project,
    save_project,
)
from vectrify.document.holes import document_hole_shape, enclosed_objects, find_holes
from vectrify.document.model import new_id
from vectrify.document.svg import parse_path
from vectrify.operations import Budget, Job, OperationRequest, Permissions, method

MAX_SOURCE = 128 * 1024 * 1024


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
        self.reference = reference
        self.epoch = uuid4().hex
        self.lock = RLock()
        self._svg_revision = -1
        self._svg = ""
        self.jobs: dict[str, Job] = {}

    def operation(self, payload: dict) -> dict:
        """Start, poll, stop, apply or discard one automated operation."""
        command = payload.get("command")
        if command == "start":
            self.check_revision(payload)
            chosen = method(str(payload.get("action")), str(payload.get("method")))
            if chosen.background and any(
                j.method.background and j.status == "running"
                for j in self.jobs.values()
            ):
                raise DocumentError("Another operation is already running")
            bounds = payload.get("bounds", self.state(svg=False)["bounds"])
            snapshot = self.editor.snapshot
            if payload.get("scope") == "drawing":
                # An explicit whole-drawing request; the focus region still applies.
                snapshot = replace(
                    snapshot,
                    selection=Selection(
                        whole_document=True, focus=snapshot.selection.focus
                    ),
                )
            elif payload.get("scope") not in {None, "selection"}:
                raise DocumentError("Choose the selection or the whole drawing")
            job = Job(
                chosen,
                OperationRequest(
                    action=chosen.action,
                    method=chosen.name,
                    snapshot=snapshot,
                    editor=self.editor,
                    permissions=Permissions.parse(payload.get("permissions")),
                    settings=payload.get("settings") or {},
                    budget=Budget.parse(payload.get("budget")),
                    reference=(
                        self.reference_image() if chosen.needs_reference else None
                    ),
                    bounds=tuple(bounds) if isinstance(bounds, list) else bounds,
                ),
            )
            job.context_key = self._job_key(job)
            job.start()
            self.jobs = {k: v for k, v in self.jobs.items() if v.status == "running"}
            self.jobs[job.id] = job
            return job.state(preview=True)
        job = self.jobs.get(str(payload.get("job")))
        if job is None:
            raise DocumentError("This operation preview has expired")
        if command == "status":
            return job.state(preview=bool(payload.get("preview")))
        if command == "stop":
            job.stop.set()
            return job.state()
        if command == "discard":
            job.stop.set()
            del self.jobs[job.id]
            return {"discarded": True}
        if command == "apply":
            if job.context_key != self._job_key(job):
                raise StaleRevisionError(
                    "The drawing or reference changed. Run the operation again."
                )
            job.apply(payload.get("choice", 0))
            del self.jobs[job.id]
            return self.state()
        raise DocumentError("Unknown operation command")

    def _job_key(self, job: Job) -> tuple:
        """What a result depends on besides the revision the commit checks."""
        reference = self.reference["data_url"] if self.reference else None
        return (self.epoch, reference if job.method.needs_reference else None)

    def reference_image(self) -> Image.Image | None:
        if not self.reference:
            return None
        data = base64.b64decode(self.reference["data_url"].split(",", 1)[1])
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            return image.copy()

    def state(self, *, svg: bool = True) -> dict:
        snapshot = self.editor.snapshot
        objects = []
        counters: dict[str, int] = {}
        for element in snapshot.document.elements():
            if element.tag == "svg":
                continue
            counters[element.tag] = counters.get(element.tag, 0) + 1
            ancestors = snapshot.document.ancestry(element.id)
            objects.append(
                {
                    "id": element.id,
                    "tag": element.tag,
                    "shared_edges": sum(
                        m.geometry_id == element.geometry_id
                        for b in snapshot.document.boundaries
                        for m in b.members
                    ),
                    "name": element.name,
                    "label": element.name
                    or (
                        element.id
                        if not element.id.startswith("object_")
                        else "Definitions"
                        if element.tag == "defs"
                        else f"{element.tag.capitalize()} {counters[element.tag]}"
                    ),
                    "parent": ancestors[-2].id,
                    "depth": len(ancestors) - 2,
                    "resource": any(e.tag in {"defs", "clipPath"} for e in ancestors),
                    "attributes": dict(element.attributes),
                    "locks": sorted(element.locks),
                    "inherited_locks": sorted(
                        set().union(*(a.locks for a in ancestors))
                    ),
                }
            )
        root = snapshot.document.root
        viewbox = root.get("viewBox")
        bounds = (
            [float(v) for v in viewbox.replace(",", " ").split()]
            if viewbox
            else [
                0,
                0,
                float(root.get("width", "1024") or 1024),
                float(root.get("height", "768") or 768),
            ]
        )
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

    def holes(self, payload: dict) -> dict:
        self.check_revision(payload)
        document = self.editor.snapshot.document
        oid = payload["object"]
        if oid not in self.editor.snapshot.selection.object_ids:
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
        before = self.editor.snapshot.revision
        if command == "open":
            self.open(payload["source"], str(payload.get("name", "Untitled.svg")))
            return self.state()
        if command == "select":
            self.editor.select(
                Selection(
                    object_ids=frozenset(payload.get("objects", [])),
                    node_ids=frozenset(payload.get("nodes", [])),
                    focus=self.editor.snapshot.selection.focus,
                )
            )
        elif command == "rectangle":
            rectangle = Rect(*(number(v) for v in payload["rectangle"]))
            mode = payload.get("mode", "contain")
            if mode not in {"intersect", "contain"}:
                raise DocumentError("Choose intersection or containment selection")
            previous = self.editor.snapshot.selection
            selected = self.editor.select_rectangle(
                rectangle, mode=mode, tolerance=0.25
            )
            if payload.get("add"):
                self.editor.select(
                    replace(
                        selected, object_ids=selected.object_ids | previous.object_ids
                    )
                )
        elif command in {"undo", "redo"}:
            getattr(self.editor, command)()
        elif command == "rename":
            if self.editor.snapshot.selection.object_ids != frozenset(
                {payload["object"]}
            ):
                raise DocumentError("Select one object to rename")
            self.editor.rename_object(payload["object"], payload["name"])
        elif command == "locks":
            object_id = payload["object"]
            if object_id not in self.editor.snapshot.document.selection_ids(
                self.editor.snapshot.selection
            ):
                raise DocumentError("Select the object before changing its locks")
            self.editor.set_locks(object_id, frozenset(payload["locks"]))
        elif command == "pin":
            if payload["object"] not in self.editor.snapshot.selection.object_ids:
                raise DocumentError("Select the path before pinning a node")
            self.editor.pin_node(
                payload["object"], payload["node"], pinned=bool(payload["pinned"])
            )
        elif command == "reference":
            reference = (
                self.validate_reference(payload["reference"])
                if payload.get("reference")
                else None
            )
            self.reference = reference
        else:
            self._edit(command, payload)
        return self.state(svg=self.editor.snapshot.revision != before)

    def _edit(self, command: str, payload: dict) -> None:
        selection = self.editor.snapshot.selection
        selected = selection.object_ids
        document = self.editor.snapshot.document
        if command == "add_path":
            geometry = parse_path(str(payload.get("d", "")))
            if len(geometry.subpaths) != 1 or len(geometry.subpaths[0].nodes) < 2:
                raise DocumentError("Draw at least two path points")
            closed = geometry.subpaths[0].closed
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
            return
        if not selected:
            raise DocumentError("Select an object first")
        if command == "fill_holes":
            oid = payload["object"]
            if oid not in selected:
                raise DocumentError("Select the path whose holes should be filled")
            requested = frozenset(payload.get("holes", []))
            cleanup = frozenset(payload.get("delete_objects", []))
            if cleanup:
                holes = tuple(h for h in find_holes(document, oid) if h.id in requested)
                if not cleanup <= enclosed_objects(document, oid, holes):
                    raise DocumentError(
                        "Only shapes fully inside the chosen holes can be deleted"
                    )
            with self.editor.transaction(
                "Fill holes", selection=Selection(object_ids=selected | cleanup)
            ) as tx:
                tx.fill_holes(oid, requested)
                if cleanup:
                    tx.delete_objects(cleanup)
            return
        if command in {"node", "split"}:
            # The user explicitly linked these edges. Direct node edits include
            # linked peers, while their locks and pins remain authoritative.
            gids = {
                document.geometry_for(oid).id
                for oid in selected
                if document.element(oid).tag == "path"
            }
            while True:
                linked = {
                    m.geometry_id
                    for b in document.boundaries
                    if any(m.geometry_id in gids for m in b.members)
                    for m in b.members
                }
                if linked <= gids:
                    break
                gids |= linked
            scope = selected | frozenset(
                oid for gid in gids for oid in document.geometry_users(gid)
            )
            selection = Selection(object_ids=scope)
        if command == "move":
            gids = {
                document.geometry_for(e.id).id
                for e in document.elements()
                if e.tag == "path"
                and any(a.id in selected for a in document.ancestry(e.id))
            }
            if any(
                m.geometry_id in gids for b in document.boundaries for m in b.members
            ):
                raise DocumentError(
                    "Unlink shared boundaries before moving these regions"
                )
        group_id = None
        with self.editor.transaction(
            {
                "unlink_boundaries": "Unlink boundaries",
                "paint": "Change paint",
                "move": "Move selection",
                "node": "Edit node",
                "group": "Group objects",
                "ungroup": "Ungroup objects",
                "delete": "Delete selection",
                "reorder": "Change stacking",
                "split": "Split edge",
                "delete_node": "Delete node",
                "detach": "Detach geometry",
                "split_disconnected": "Split disconnected parts",
                "join_paths": "Join outlines",
            }.get(command, command),
            selection=selection,
        ) as tx:
            if command == "unlink_boundaries":
                gids = {
                    document.geometry_for(oid).id
                    for oid in selected
                    if document.element(oid).tag == "path"
                }
                for boundary in document.boundaries:
                    for member in boundary.members:
                        if member.geometry_id in gids:
                            tx.detach_boundary(member)
            elif command == "paint":
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
                for oid in selected:
                    tx.set_attributes(oid, changes)
            elif command == "move":
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
            elif command == "node":
                tx.update_node(
                    payload["object"],
                    payload["node"],
                    tuple(number(v) for v in payload["values"]),
                )
            elif command == "group":
                group_id = tx.group_objects(selected)
            elif command == "ungroup":
                for oid in selected:
                    tx.ungroup_object(oid)
            elif command == "delete":
                tx.delete_objects(selected)
            elif command == "reorder":
                if len(selected) != 1:
                    raise DocumentError(
                        "Select one object to change its stacking order"
                    )
                oid = next(iter(selected))
                siblings = document.ancestry(oid)[-2].children
                index = next(i for i, item in enumerate(siblings) if item.id == oid)
                target = max(0, min(len(siblings) - 1, index + int(payload["step"])))
                tx.reorder_object(oid, target)
            elif command == "split":
                tx.split_edge(payload["object"], payload["node"])
            elif command == "delete_node":
                tx.delete_node(payload["object"], payload["node"])
            elif command == "detach":
                for oid in selected:
                    tx.detach_geometry(oid)
            elif command == "split_disconnected":
                for oid in sorted(selected):
                    tx.split_disconnected(oid)
            elif command == "join_paths":
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
                    or (
                        mode == "source" and (not isinstance(source, str) or not source)
                    )
                    or (mode == "mix" and source is not None)
                ):
                    raise DocumentError(
                        "Choose mixed colors or a selected color source"
                    )
                tx.join_paths(selected, color_source=source)
            else:
                raise DocumentError("Unknown editor command")

        if group_id is not None:
            self.editor.select(Selection(object_ids=frozenset({group_id})))
