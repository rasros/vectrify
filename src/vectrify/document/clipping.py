"""Derived clip geometry; the editable boundary always retains its source."""

from __future__ import annotations

from functools import lru_cache

import pathops
from shapely.errors import GEOSException

from vectrify.document.model import (
    Document,
    DocumentError,
    Element,
    Geometry,
    new_id,
)

INSET = "data-vectrify-clip-inset"
BASE = "data-vectrify-clip-base"
SOURCE = "data-vectrify-clip-source"


@lru_cache(maxsize=256)
def inset_geometry(geometry: Geometry, distance: float, rule: str) -> Geometry:
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
        band = pathops.Path(area)
        band.stroke(
            distance * 2, pathops.LineCap.BUTT_CAP, pathops.LineJoin.MITER_JOIN, 4
        )
        return path_geometry(pathops.op(area, band, pathops.PathOp.DIFFERENCE))
    except (pathops.PathOpsError, GEOSException, OverflowError, ValueError) as exc:
        raise DocumentError("Cannot inset this clipping boundary") from exc


def clip_geometry(document: Document, element: Element) -> Geometry:
    geometry = document.geometry_for(element.id)
    distance = float(element.get(INSET, "0") or 0)
    if not distance:
        return geometry
    return inset_geometry(geometry, distance, element.get("clip-rule", "nonzero"))
