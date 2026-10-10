"""Keep an interior fill from painting across an existing sibling outline."""

from __future__ import annotations

import numpy as np
import shapely
from cairosvg.colors import color

from vectrify.document.hit_test import _filled
from vectrify.document.join import path_style
from vectrify.document.transforms import root_matrix
from vectrify.refine.crossings import bezier
from vectrify.refine.snap import _Frame


def _contours(geometry, frame):
    contours = []
    for subpath in geometry.subpaths:
        previous = frame.pixels(subpath.nodes[0].values)[-1]
        points = [previous]
        for node in subpath.nodes[1:]:
            controls = np.vstack((previous, frame.pixels(node.values)))
            sampled = bezier(controls, np.linspace(0, 1, 33))[0]
            points.extend(sampled[1:])
            previous = controls[-1]
        contours.append((tuple(map(tuple, points)), True))
    return tuple(contours)


def protected_pixels(document, oid, geometry, frame: _Frame, size):
    """Pixels outside a containing earlier sibling fill or on its stroke.

    This is a conservative geometric relation, independent of reference blur:
    the enclosing path remains fixed while its interior shading is fitted.
    Uncontained, translucent, clipped and filtered artwork is left alone.
    The loss can retain the existing artwork there without freezing the
    interior curves. Return None when no such enclosure can be inferred.
    """
    ancestry = document.ancestry(oid)
    if len(ancestry) < 2 or any(
        a.get("clip-path")
        or a.get("filter")
        or a.get("mask")
        or float(a.get("opacity", "1") or "1") != 1
        for a in ancestry
    ):
        return None
    target_style = path_style(document, ancestry[-1])
    if (
        target_style["fill"] == "none"
        or target_style["stroke"] != "none"
        or "url(" in target_style["fill"]
        or color(target_style["fill"])[3] != 1
        or float(target_style["fill-opacity"]) != 1
    ):
        return None
    interior = _filled(_contours(geometry, frame), target_style["fill-rule"])
    if interior.is_empty:
        return None
    support, reach, smallest = None, 0.0, float("inf")
    for sibling in ancestry[-2].children:
        if sibling.id == oid:
            break
        if (
            sibling.tag != "path"
            or sibling.get("filter")
            or sibling.get("clip-path")
            or sibling.get("mask")
        ):
            continue
        style = path_style(document, sibling)
        if (
            style["fill"] == "none"
            or style["stroke"] == "none"
            or "url(" in style["fill"]
            or "url(" in style["stroke"]
            or float(style["fill-opacity"]) != 1
            or float(style["stroke-opacity"]) != 1
            or float(sibling.get("opacity", "1") or "1") != 1
            or float(style["stroke-width"]) <= 0
            or color(style["fill"])[3] != 1
            or color(style["stroke"])[3] != 1
        ):
            continue
        a, b, c, d, e, f = root_matrix(document, sibling.id)
        # The target frame maps local coordinates into the working crop.
        ta, tb, tc, td, te, tf = root_matrix(document, oid)
        target_linear = np.array([[ta, tc], [tb, td]])
        pixels = frame.matrix @ np.linalg.inv(target_linear)
        linear = pixels @ np.array([[a, c], [b, d]])
        offset = frame.offset + pixels @ (np.array([e, f]) - [te, tf])
        source_frame = _Frame(linear, offset)
        source = _contours(document.geometry_for(sibling.id), source_frame)
        enclosing = _filled(source, style["fill-rule"])
        if enclosing.is_empty or enclosing.area >= smallest:
            continue
        radius = float(style["stroke-width"]) * np.linalg.norm(linear, ord=2) / 2 + 0.25
        # Hand-drawn shading can already protrude slightly across the stroke.
        # Recognize a near enclosure too, retaining that baseline rather than
        # allowing the optimizer to enlarge the protrusion.
        if interior.difference(enclosing.buffer(radius)).area > max(
            1.0, 0.02 * interior.area
        ):
            continue
        support, reach, smallest = enclosing, radius, enclosing.area
    if support is None:
        return None
    # The inset includes the existing outline's stroke. Protecting those
    # pixels prevents a blurred reference from rewarding overpainted ink.
    inset = support.buffer(-reach)
    y, x = np.mgrid[: size[1], : size[0]]
    return ~shapely.contains_xy(inset, x + 0.5, y + 0.5)
