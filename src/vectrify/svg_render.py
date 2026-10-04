"""Render SVG user-space regions to PNG with a shared viewport convention."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Callable

import cairosvg


def frame(
    root: ET.Element,
    box: tuple[float, float, float, float],
    size: tuple[int, int],
) -> None:
    """Aim *root*'s viewBox at *box*, stretched over *size* pixels."""
    root.set("viewBox", " ".join(str(v) for v in box))
    root.set("width", str(size[0]))
    root.set("height", str(size[1]))
    root.set("preserveAspectRatio", "none")


def render_png(
    svg: str,
    box: tuple[float, float, float, float],
    size: tuple[int, int],
    *,
    overlay: Callable[[ET.Element], None] | None = None,
) -> bytes:
    """Render *box* stretched to *size* pixels on white.

    The optional overlay adds elements to the framed SVG before rendering,
    so preview marks use the same user space as the original drawing.
    """
    root = ET.fromstring(svg)
    frame(root, box, size)
    if overlay is not None:
        overlay(root)
    png = cairosvg.svg2png(bytestring=ET.tostring(root), background_color="white")
    assert png is not None
    return png
