"""Clipping drawn objects to closed paths, and the clip geometry derived from them.

The editable boundary always retains its source: a clip is built from a
boundary path without moving or repainting it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import pathops
from shapely.errors import GEOSException

from vectrify.document.model import (
    Document,
    DocumentError,
    Element,
    Geometry,
    Selection,
    new_id,
)

if TYPE_CHECKING:
    from vectrify.document.editor import Editor

INSET = "data-vectrify-clip-inset"
BASE = "data-vectrify-clip-base"
SOURCE = "data-vectrify-clip-source"


@lru_cache(maxsize=256)
def inset_geometry(
    geometry: Geometry,
    distance: float,
    rule: str,
    join: str = "miter",
    miter_limit: float = 4,
) -> Geometry:
    """Remove an SVG-style boundary band before placement, expanding holes.

    Resolve fill rules first so overlapping contours leave no interior cuts.
    SVG bevels miters exceeding its limit; GEOS's clipped miters instead leave
    unshaded tabs at acute inward corners. Stroke the resolved perimeter with
    SVG's joins and subtract it, keeping the editable source untouched.
    """
    from vectrify.document.hit_test import _filled, _flatten
    from vectrify.document.join import curve_path, path_geometry

    try:
        area = pathops.simplify(curve_path(geometry, rule))
        left, top, right, bottom = area.bounds
        # A huge but finite inset is an empty clip, not a float32 overflow.
        if not area.verbs or distance >= min(right - left, bottom - top) / 2:
            return Geometry(new_id("geometry"), ())
        # Very wide curved strokes can produce inverted inner offset loops in
        # Skia. Check collapse independently before subtracting the stroke.
        contours = []
        for sub in geometry.subpaths:
            points = [sub.nodes[0].endpoint]
            for node in sub.nodes[1:]:
                if node.command == "C":
                    v = node.values
                    points.extend(
                        _flatten(
                            (points[-1], (v[0], v[1]), (v[2], v[3]), node.endpoint),
                            min(0.01, max(0.0001, distance / 16)),
                        )
                    )
                else:
                    points.append(node.endpoint)
            contours.append((tuple(points), sub.closed))
        if _filled(tuple(contours), rule).buffer(-distance).is_empty:
            return Geometry(new_id("geometry"), ())
        # Skia's stroker approximates curved offsets at a fixed coordinate
        # tolerance. At artwork scale this can exceed a thin outline's inset,
        # leaving visible slivers even though the source geometry is shared.
        # Stroke in enlarged, origin-relative coordinates, then map back.
        # Cap enlargement to keep very small distances numerically bounded.
        scale = max(1.0, min(1024.0, 128.0 / distance))
        enlarged = area.transform(scale, 0, 0, scale, -left * scale, -top * scale)
        band = pathops.Path(enlarged)
        band.stroke(
            distance * 2 * scale,
            pathops.LineCap.BUTT_CAP,
            {
                "miter": pathops.LineJoin.MITER_JOIN,
                "round": pathops.LineJoin.ROUND_JOIN,
                "bevel": pathops.LineJoin.BEVEL_JOIN,
            }[join],
            miter_limit,
        )
        band.convertConicsToQuads(0.001 * distance * scale)
        result = pathops.op(enlarged, band, pathops.PathOp.DIFFERENCE)
        return path_geometry(result.transform(1 / scale, 0, 0, 1 / scale, left, top))
    except (pathops.PathOpsError, GEOSException, OverflowError, ValueError) as exc:
        raise DocumentError("Cannot inset this clipping boundary") from exc


def clip_geometry(document: Document, element: Element) -> Geometry:
    from vectrify.document.join import path_style

    geometry = document.geometry_for(element.id)
    distance = float(element.get(INSET, "0") or 0)
    if not distance:
        return geometry
    # Linked clips follow inherited corner settings as well as node edits.
    # Missing or detached sources fall back to the settings captured on the
    # boundary, so a copied clip keeps its original appearance.
    source = element.get(SOURCE)
    try:
        original = document.element(source) if source else None
    except DocumentError:
        original = None
    style = path_style(
        document,
        original
        if original is not None and original.geometry_id == geometry.id
        else element,
    )
    return inset_geometry(
        geometry,
        distance,
        element.get("clip-rule", "nonzero"),
        style["stroke-linejoin"],
        float(style["stroke-miterlimit"]),
    )


def apply_clip(
    editor: Editor,
    targets: Sequence[str],
    boundaries: Sequence[str] = (),
    *,
    action: str = "set",
    linked: bool = True,
    inset: float = 0.0,
    label: str = "Clipping",
) -> list[dict[str, Any]]:
    """Clip each target to the union of the boundary paths, or remove its clip.

    Every target gets a clipPath of its own, placed in its coordinate frame.
    A linked clip shares each boundary's geometry, so later point edits to the
    boundary reshape the clip; a copied one keeps the outline it had. Either
    way the clip stays where it is when the boundary path is later moved, and
    the boundary is never edited. The whole edit is one undoable step, or
    nothing changes. Returns what each target is now clipped to.
    """
    from vectrify.document.hit_test import multiply
    from vectrify.document.join import path_style
    from vectrify.document.regions import SHAPES, object_matrix
    from vectrify.document.topology import inverse_matrix

    if action not in {"set", "remove"}:
        raise DocumentError("Clip action is set or remove")
    if not isinstance(linked, bool):
        raise DocumentError("linked must be true or false")
    if not math.isfinite(inset):
        raise DocumentError("inset must be finite")
    if inset < 0:
        raise DocumentError("Clipping inset cannot be negative")
    if action == "set" and not boundaries:
        raise DocumentError("Give boundary paths to clip to")
    if action == "remove" and boundaries:
        raise DocumentError("Remove clipping without a boundary")
    if action == "remove" and inset:
        raise DocumentError("Remove clipping without an inset")
    targets = list(targets)
    document = editor.snapshot.document
    for oid in targets:
        if any(a.id in targets for a in document.ancestry(oid)[:-1]):
            raise DocumentError(
                "Choose targets without selecting both parent and child"
            )
    sources = []
    for oid in boundaries:
        element = document.element(oid)
        if element.tag in SHAPES:
            raise DocumentError(
                f"{oid}: a {element.tag} cannot be a clipping boundary until "
                "it is a path; use Convert to path first"
            )
        if element.tag != "path" or element.geometry_id is None:
            raise DocumentError("Boundaries must be paths; convert shapes first")
        geometry = document.geometry_for(oid)
        if not geometry.subpaths or any(not s.closed for s in geometry.subpaths):
            raise DocumentError("Boundary paths must have closed contours")
        if inset:
            style = path_style(document, element)
            inset_geometry(
                geometry,
                inset,
                style["fill-rule"],
                style["stroke-linejoin"],
                float(style["stroke-miterlimit"]),
            )
        sources.append((element, geometry, object_matrix(document, oid)))
    result: list[dict[str, Any]] = []
    # Scope is explicit; the transaction's structural command checks target
    # dependents and locks. Boundaries are read, never moved or repainted.
    tx = editor.transaction(label, selection=Selection(frozenset(targets)))
    with tx:
        for oid in targets:
            if action == "remove":
                tx.set_clip(oid, None)
                result.append({"target": oid, "clip": None})
                continue
            frame = inverse_matrix(object_matrix(document, oid))
            children = []
            geometries = []
            for element, geometry, matrix in sources:
                if not linked:
                    geometry = _copied(geometry)
                    geometries.append(geometry)
                style = path_style(document, element)
                rule = style["fill-rule"]
                attrs = {
                    "fill": "#ffffff",
                    "stroke": "none",
                    "clip-rule": rule,
                    "fill-rule": rule,
                    "transform": "matrix("
                    + " ".join(str(v) for v in multiply(frame, matrix))
                    + ")",
                }
                if linked:
                    attrs[SOURCE] = element.id
                if inset:
                    attrs[INSET] = str(inset)
                    attrs["stroke-linejoin"] = style["stroke-linejoin"]
                    attrs["stroke-miterlimit"] = style["stroke-miterlimit"]
                children.append(
                    Element(
                        new_id("boundary"),
                        "path",
                        attributes=tuple(attrs.items()),
                        geometry_id=geometry.id,
                        name="Clipping shape",
                    )
                )
            clip = Element(
                new_id("clip"),
                "clipPath",
                attributes=(("clipPathUnits", "userSpaceOnUse"),),
                children=tuple(children),
                name="Clipping boundary",
            )
            tx.set_clip(oid, clip, tuple(geometries))
            result.append(
                {
                    "target": oid,
                    "clip": clip.id,
                    "sources": list(boundaries),
                    "linked": linked,
                    "inset": inset,
                }
            )
    editor.select(Selection(frozenset(targets)))
    return result


def _copied(geometry: Geometry) -> Geometry:
    """An independent copy of *geometry*, with identities of its own."""
    return replace(
        geometry,
        id=new_id("geometry"),
        subpaths=tuple(
            replace(
                subpath,
                id=new_id("subpath"),
                nodes=tuple(replace(node, id=new_id("node")) for node in subpath.nodes),
            )
            for subpath in geometry.subpaths
        ),
    )
