"""Render SVG user-space regions to PNG with a shared viewport convention."""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from collections import OrderedDict
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import cairosvg
import numpy as np
from cairosvg import surface as surfaces
from cairosvg.parser import Tree
from cairosvg.path import path
from PIL import Image

# Cairo paths are compiled in their viewport's transform. Keep the cache local
# to an operation, bounded, and separate from CairoSVG's normal path handler.
_PATH_TAG = "_vectrify_cached_path"
_CACHE_SIZE = 1024
_cache: ContextVar[OrderedDict | None] = ContextVar("svg_paths", default=None)


@contextmanager
def cached_rendering():
    """Reuse unchanged Cairo paths while one operation judges its candidates."""
    if _cache.get() is not None:
        yield
        return
    token = _cache.set(OrderedDict())
    try:
        yield
    finally:
        _cache.reset(token)


def _cached_path(surface: Any, node: Any) -> None:
    if not hasattr(surface, "path_cache"):
        return
    context = surface.context
    key = (
        node.get("d", ""),
        tuple(context.get_matrix()),
        surface.context_width,
        surface.context_height,
        surface.font_size,
        context.get_tolerance(),
        context.get_current_point(),
    )
    cache = surface.path_cache
    compiled = cache.get(key)
    if compiled is None:
        # Clip contours can accumulate several paths. Compile just this one,
        # then append it to whatever the surrounding artwork already drew.
        previous = context.copy_path()
        current = context.get_current_point()
        context.new_path()
        context.move_to(*current)
        path(surface, node)
        compiled = (context.copy_path(), context.get_tolerance())
        context.new_path()
        context.append_path(previous)
        cache[key] = compiled
        if len(cache) > _CACHE_SIZE:
            cache.popitem(last=False)
    else:
        cache.move_to_end(key)
    context.set_tolerance(compiled[1])
    context.append_path(compiled[0])
    # Only paths with no markers use this handler. CairoSVG otherwise walks
    # every vertex even when there is no marker to draw.
    node.vertices = []


surfaces.TAGS[_PATH_TAG] = _cached_path


class _Surface(surfaces.PNGSurface):
    def __init__(self, *args, path_cache, **kwargs):
        self.path_cache = path_cache
        super().__init__(*args, **kwargs)

    def draw(self, node):
        # Paint servers and markers ask CairoSVG for a path's bounding box.
        # Let its ordinary handler keep all of those tag-dependent semantics.
        references = (
            "fill",
            "stroke",
            "clip-path",
            "filter",
            "mask",
            "marker",
            "marker-start",
            "marker-mid",
            "marker-end",
        )
        if node.tag == "path" and not any(
            "url(" in node.get(key, "") for key in references
        ):
            node.tag = _PATH_TAG
            try:
                super().draw(node)
            finally:
                node.tag = "path"
        else:
            super().draw(node)


def render_image(
    svg: str | bytes,
    box: tuple[float, float, float, float] | None = None,
    size: tuple[int, int] | None = None,
) -> Image.Image:
    """Cairo's RGB pixels directly, without a PNG encode/decode between scores."""
    if box is not None:
        assert size is not None
        root = ET.fromstring(svg)
        frame(root, box, size)
        svg = ET.tostring(root)
    tree = Tree(bytestring=svg.encode() if isinstance(svg, str) else svg)
    cache = _cache.get()
    surface = _Surface(
        tree,
        None,
        96,
        background_color="white",
        path_cache=cache if cache is not None else OrderedDict(),
    )
    assert surface.cairo is not None
    surface.cairo.flush()
    pixels = (
        np.frombuffer(surface.cairo.get_data(), dtype=np.uint8)
        .reshape(surface.height, surface.cairo.get_stride())[:, : surface.width * 4]
        .reshape(surface.height, surface.width, 4)
    )
    channels = [2, 1, 0] if sys.byteorder == "little" else [1, 2, 3]
    return Image.fromarray(pixels[..., channels])


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
