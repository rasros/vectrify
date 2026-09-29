"""Before/after previews of a document region."""

from __future__ import annotations

import base64
import math
import xml.etree.ElementTree as ET

import cairosvg

from vectrify.document import DocumentError, export_svg
from vectrify.operations.generate import frame


def render_previews(before, after, bounds, *, highlight=False):
    if not isinstance(bounds, list | tuple) or len(bounds) != 4:
        raise DocumentError("Expected preview bounds")
    x, y, width, height = (float(v) for v in bounds)
    if (
        not all(math.isfinite(v) for v in (x, y, width, height))
        or width <= 0
        or height <= 0
    ):
        raise DocumentError("Preview bounds must be finite with positive dimensions")
    scale = 900 / max(width, height)

    def render(document):
        root = ET.fromstring(export_svg(document))
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        frame(root, (x, y, width, height), size)
        if highlight and document is after:
            from vectrify.document.hit_test import IDENTITY, multiply, transform
            from vectrify.document.topology import edge, inverse_matrix

            for boundary in document.boundaries:
                ref = boundary.members[0]
                owner = next(iter(document.geometry_users(ref.geometry_id)))
                matrix = IDENTITY
                for ancestor in document.ancestry(owner):
                    matrix = multiply(matrix, transform(ancestor.get("transform")))
                points = edge(document, ref).points
                matrix = multiply(matrix, inverse_matrix(ref.matrix))
                data = f"M{points[0][0]} {points[0][1]} "
                data += "C" if len(points) == 4 else "L"
                data += " ".join(str(v) for point in points[1:] for v in point)
                ET.SubElement(
                    root,
                    "{http://www.w3.org/2000/svg}path",
                    {
                        "d": data,
                        "transform": "matrix(" + " ".join(map(str, matrix)) + ")",
                        "fill": "none",
                        "stroke": "#00c8ff",
                        "stroke-width": str(2 / scale),
                    },
                )
        png = cairosvg.svg2png(bytestring=ET.tostring(root), background_color="white")
        assert png is not None
        return "data:image/png;base64," + base64.b64encode(png).decode()

    return {"before": render(before), "after": render(after)}
