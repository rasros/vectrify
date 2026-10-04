"""Before/after previews of a document region."""

from __future__ import annotations

import base64
import math
import xml.etree.ElementTree as ET

from vectrify.document import DocumentError, export_svg
from vectrify.svg_render import render_png


def render_previews(before, after, bounds, *, highlight=()):
    """Both documents over *bounds*; *highlight*'s edges, which map into root
    user space, are outlined on the after image."""
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
    size = (max(1, round(width * scale)), max(1, round(height * scale)))

    def overlay(root):
        from vectrify.document.topology import edge

        for ref in highlight:
            points = edge(after, ref).points
            data = f"M{points[0][0]} {points[0][1]} "
            data += "C" if len(points) == 4 else "L"
            data += " ".join(str(v) for point in points[1:] for v in point)
            ET.SubElement(
                root,
                "{http://www.w3.org/2000/svg}path",
                {
                    "d": data,
                    "fill": "none",
                    "stroke": "#00c8ff",
                    "stroke-width": str(2 / scale),
                },
            )

    def render(document):
        png = render_png(
            export_svg(document),
            (x, y, width, height),
            size,
            overlay=overlay if document is after else None,
        )
        return "data:image/png;base64," + base64.b64encode(png).decode()

    return {"before": render(before), "after": render(after)}
