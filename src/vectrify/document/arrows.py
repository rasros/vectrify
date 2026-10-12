"""Endpoint decorations derived from the editable line, never extra objects."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vectrify.document.model import Document, Element, Geometry

START = "data-vectrify-arrow-start"
END = "data-vectrify-arrow-end"
ARROWS = frozenset({START, END})
OWNER = "data-vectrify-arrow-owner"


def is_open(document: Document, element: Element) -> bool:
    if element.tag == "line":
        return True
    return element.tag == "path" and any(
        not sub.closed and len(sub.nodes) > 1
        for sub in document.geometry_for(element.id).subpaths
    )


def triangles(document: Document, element: Element, width: float):
    """(Contour index, triangle) in source-local coordinates, for hit and export."""
    if width <= 0 or not any(element.get(key) == "arrow" for key in ARROWS):
        return []
    if element.tag == "line":
        from vectrify.document.svg import parse_path

        x1, y1, x2, y2 = (element.get(key, "0") for key in ("x1", "y1", "x2", "y2"))
        geometry = parse_path(f"M{x1} {y1} L{x2} {y2}")
    elif element.tag == "path":
        geometry = document.geometry_for(element.id)
    else:
        return []
    return geometry_triangles(geometry, width, dict(element.attributes))


def geometry_triangles(geometry: Geometry, width: float, style: dict[str, str]):
    result = []
    for index, sub in enumerate(geometry.subpaths):
        if sub.closed or len(sub.nodes) < 2:
            continue
        # Controls supply exact endpoint tangents; skip coincident controls and
        # zero-length segments before falling back to the next segment.
        first = sub.nodes[0].endpoint
        assert first is not None
        points = [first]
        for node in sub.nodes[1:]:
            points.extend(zip(node.values[::2], node.values[1::2], strict=True))
        for key, tip, candidates in (
            (START, points[0], points[1:]),
            (END, points[-1], list(reversed(points[:-1]))),
        ):
            if style.get(key) != "arrow":
                continue
            vector = next(
                ((p[0] - tip[0], p[1] - tip[1]) for p in candidates if p != tip),
                None,
            )
            if vector is None:
                continue
            length = math.hypot(*vector)
            dx, dy = vector[0] / length, vector[1] / length
            base = (tip[0] + 4 * width * dx, tip[1] + 4 * width * dy)
            result.append(
                (
                    index,
                    (
                        tip,
                        (base[0] - 2 * width * dy, base[1] + 2 * width * dx),
                        (base[0] + 2 * width * dy, base[1] - 2 * width * dx),
                    ),
                )
            )
    return result
