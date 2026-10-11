"""Derived clip geometry; the editable boundary always retains its source."""

from __future__ import annotations

from functools import lru_cache

from shapely.errors import GEOSException
from shapely.geometry import Polygon

from vectrify.document.model import (
    Document,
    DocumentError,
    Element,
    Geometry,
    PathNode,
    Subpath,
    new_id,
)

INSET = "data-vectrify-clip-inset"
BASE = "data-vectrify-clip-base"
SOURCE = "data-vectrify-clip-source"


@lru_cache(maxsize=256)
def inset_geometry(geometry: Geometry, distance: float, rule: str) -> Geometry:
    """Remove a boundary band before placement, expanding holes as well.

    Resolve fill rules before offsetting, so overlapping contours do not leave
    interior cuts. Only the derived clip is flattened, within 0.01 local units;
    the editable source's curves, IDs and shared geometry remain untouched.
    """
    from vectrify.document.hit_test import _filled, _flatten

    try:
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
        area = _filled(tuple(contours), rule)
        # A huge but finite inset is a valid empty clip, not an overflow in GEOS.
        if area.is_empty or distance >= max(
            area.bounds[2] - area.bounds[0], area.bounds[3] - area.bounds[1]
        ):
            return Geometry(new_id("geometry"), ())
        inset = area.buffer(-distance, join_style="mitre", mitre_limit=4)
        polygons = [inset] if isinstance(inset, Polygon) else list(inset.geoms)
        subpaths = []
        for polygon in polygons:
            if polygon.is_empty:
                continue
            for ring in (polygon.exterior, *polygon.interiors):
                subpaths.append(
                    Subpath(
                        new_id("subpath"),
                        tuple(
                            PathNode(new_id("node"), "M" if i == 0 else "L", (x, y))
                            for i, (x, y) in enumerate(list(ring.coords)[:-1])
                        ),
                        closed=True,
                    )
                )
        return Geometry(new_id("geometry"), tuple(subpaths))
    except (GEOSException, OverflowError, ValueError) as exc:
        raise DocumentError("Cannot inset this clipping boundary") from exc


def clip_geometry(document: Document, element: Element) -> Geometry:
    geometry = document.geometry_for(element.id)
    distance = float(element.get(INSET, "0") or 0)
    if not distance:
        return geometry
    return inset_geometry(geometry, distance, element.get("clip-rule", "nonzero"))
