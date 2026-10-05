"""Fit stroked lines to the reference: centrelines onto the ink's middle,
widths to the ink's cover.

A traced line is a stroke down the middle of the ink it stands for. Its ink
is read off the reference as each pixel's cover by it: how much darker it is
than the surface around it (a black top-hat of its brightest channel, as cel
finds lines), over how much darker the stroke's own colour is. Across the
line at each point and segment middle, the ink joined to the middle is
weighed: its centre is where the centreline belongs, its sum the line's
width there. Each point moves onto the centre, at most `SHIFT` px, taking
its handles along, and each curve's handles then fit its middle. An additional
whole-span proposal fits the handles independently from multiple readings;
Tidy judges it against the actual render after individual fits. Straight
segments stay straight. The path's stroke width
becomes the median width measured along it, when that differs by more than
`WIDTH_STEP`, and only for opaque strokes (a faint line's cover says how
faint, not how wide). Pinned points stay. All of it in the reference's
pixels.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
from cairosvg.colors import color
from scipy.ndimage import grey_closing, map_coordinates

from vectrify.document import Document, Geometry
from vectrify.document.join import path_style
from vectrify.operations.generate import Region
from vectrify.refine.frozen import Frozen
from vectrify.refine.snap import _bezier, _Frame, _frame

# The most a point or segment middle moves, in reference pixels.
SHIFT = 1.5
# How far past half the line's width the ink is read either way, in pixels.
REACH = 2.0
# The step across the line its ink is read at, in pixels.
STEP = 0.5
# Cover below this is not ink.
INKED = 0.05
# How much a measured width has to differ, as a share, to be taken.
WIDTH_STEP = 0.1
# The least a line's surface has to be lighter than its ink to read it.
CONTRAST = 0.08
# Read the span of a curve, not just its midpoint: unequal handle errors can
# cancel at the middle while leaving both quarters off the reference line.
SAMPLE_SPACING = 8.0
MIN_SAMPLES = 5
MAX_SAMPLES = 33


def is_line(document: Document, oid: str) -> bool:
    """Whether path *oid* is a stroked line: a stroke and no fill."""
    style = path_style(document, document.element(oid))
    return style["fill"] == "none" and style["stroke"] not in {"none", ""}


def fit_lines(
    document: Document,
    oids,
    region: Region,
    fixed: Frozen,
    widths: bool = True,
    *,
    span: bool = False,
) -> Document:
    """*document* with the stroked lines *oids* fitted to *region*'s image:
    their centrelines on their ink's middle and, with *widths*, their stroke
    widths to its cover. With *span*, fit each curve's handles independently
    from readings along it rather than a single midpoint reading."""
    image = np.asarray(region.image.convert("RGB"), dtype=np.float64) / 255
    light = image.max(-1)
    for oid in oids:
        element = document.element(oid)
        style = path_style(document, element)
        if "url(" in style["stroke"]:
            continue
        frame = _frame(document, oid, region, region.image.size)
        if frame is None:
            continue
        scale = math.sqrt(abs(np.linalg.det(frame.matrix)))
        width = float(style["stroke-width"]) * scale
        ink = max(color(style["stroke"])[:3])
        radius = max(2, math.ceil(width) + 2)
        surface = grey_closing(light, size=2 * radius + 1)
        cover = np.clip(
            (surface - light) / np.maximum(surface - ink, CONTRAST), 0.0, 1.0
        )
        cover = np.where(surface - ink >= CONTRAST, cover, 0.0)
        reader = _Reader(cover, width)
        geometry, measured = _fitted(
            document.geometry_for(oid), frame, reader, fixed, span=span
        )
        document = document.replace_geometry(geometry)
        opaque = (
            float(style["stroke-opacity"]) >= 1
            and float(style["opacity"]) >= 1
            and color(style["stroke"])[3] >= 1
        )
        if widths and opaque and measured:
            found = float(np.median(measured))
            if found > 0 and abs(found - width) > WIDTH_STEP * width:
                attributes = dict(element.attributes)
                attributes["stroke-width"] = f"{found / scale:.4g}"
                document = document.replace_element(
                    replace(element, attributes=tuple(attributes.items()))
                )
    return document


class _Reader:
    """Reads a line's ink across it."""

    def __init__(self, cover: np.ndarray, width: float):
        self.cover = cover
        reach = width / 2 + REACH
        self.offsets = np.arange(-reach, reach + 1e-9, STEP)

    def across(self, points: np.ndarray, normals: np.ndarray):
        """(how far the ink's middle is along each normal, the ink's width
        there), as arrays; no ink reads as no shift and no width."""
        xs = points[:, 0, None] + self.offsets * normals[:, 0, None] - 0.5
        ys = points[:, 1, None] + self.offsets * normals[:, 1, None] - 0.5
        weight = map_coordinates(
            self.cover, [ys.ravel(), xs.ravel()], order=1, mode="constant"
        ).reshape(xs.shape)
        # Only the ink joined to the middle, not the next line over.
        middle = len(self.offsets) // 2
        inked = weight > INKED
        joined = np.ones(weight.shape, dtype=bool)
        joined[:, middle:] = np.cumprod(inked[:, middle:], axis=1).astype(bool)
        joined[:, : middle + 1] = np.cumprod(inked[:, middle::-1], axis=1)[
            :, ::-1
        ].astype(bool)
        weight = np.where(joined, weight, 0.0)
        total = weight.sum(1)
        shift = np.where(
            total > 0, (weight * self.offsets).sum(1) / np.maximum(total, 1e-9), 0.0
        )
        return np.clip(shift, -SHIFT, SHIFT), total * STEP


def _unit(vector: np.ndarray) -> np.ndarray | None:
    size = float(np.linalg.norm(vector))
    return vector / size if size > 1e-9 else None


def _normal(tangent: np.ndarray | None) -> np.ndarray | None:
    return None if tangent is None else np.array([-tangent[1], tangent[0]])


def _fitted(
    geometry: Geometry,
    frame: _Frame,
    reader: _Reader,
    fixed: Frozen,
    *,
    span: bool = False,
) -> tuple[Geometry, list[float]]:
    """*geometry* with its points and curves on the ink's middle, and
    the widths measured along it, in pixels."""
    measured: list[float] = []
    subpaths = []
    for subpath in geometry.subpaths:
        nodes = list(subpath.nodes)
        # Each node's controls in pixels: its handles, then its point.
        controls = [frame.pixels(n.values) for n in nodes]
        anchors = np.array([c[-1] for c in controls])
        count = len(nodes)
        closed = subpath.closed and count > 2
        # The points: each along the normal of the line through it.
        tangents = []
        for i in range(count):
            before = anchors[i - 1] if i > 0 else (anchors[-1] if closed else None)
            after = (
                anchors[i + 1] if i < count - 1 else (anchors[0] if closed else None)
            )
            if i < count - 1 and len(controls[i + 1]) == 3:
                after = controls[i + 1][0]
            if i > 0 and len(controls[i]) == 3:
                before = controls[i][1]
            if before is None and after is None:
                tangents.append(None)
                continue
            ahead = after if after is not None else anchors[i]
            behind = before if before is not None else anchors[i]
            tangents.append(_unit(ahead - behind))
        normals = [_normal(t) for t in tangents]
        movable = [
            i
            for i in range(count)
            if normals[i] is not None and nodes[i].id not in fixed.endpoints
        ]
        if movable:
            shift, width = reader.across(
                anchors[movable], np.array([normals[i] for i in movable])
            )
            # An open line's ends cover half as much: their width is not read.
            measured += [
                w
                for i, w in zip(movable, width, strict=True)
                if w > 0 and (closed or 0 < i < count - 1)
            ]
            for i, s in zip(movable, shift, strict=True):
                normal = normals[i]
                assert normal is not None
                delta = s * normal
                controls[i] = controls[i].copy()
                controls[i][-1] = controls[i][-1] + delta
                if len(controls[i]) == 3:
                    controls[i][1] = controls[i][1] + delta
                if i + 1 < count and len(controls[i + 1]) == 3:
                    controls[i + 1] = controls[i + 1].copy()
                    controls[i + 1][0] = controls[i + 1][0] + delta
        # The additional whole-span fit can move the handles independently;
        # the midpoint initializer keeps its established proposal.
        for i in range(1, count):
            if len(controls[i]) != 3:
                continue
            cubic = np.vstack([controls[i - 1][-1], controls[i]])
            delta, widths = _curve_shift(cubic, reader, span=span)
            measured.extend(widths)
            controls[i] = controls[i].copy()
            controls[i][:2] += delta
        subpaths.append(
            replace(
                subpath,
                nodes=tuple(
                    replace(n, values=frame.local(c))
                    for n, c in zip(nodes, controls, strict=True)
                ),
            )
        )
    return replace(geometry, subpaths=tuple(subpaths)), measured


def _curve_shift(cubic: np.ndarray, reader: _Reader, *, span: bool):
    """Least-squares handle motion from the ink at several points on a curve."""
    if not span:
        point, tangent = _bezier(cubic, np.array([0.5]))
        assert tangent is not None
        normal = _normal(_unit(tangent[0]))
        if normal is None:
            return np.zeros((2, 2)), []
        shift, widths = reader.across(point, np.array([normal]))
        delta = 4 / 3 * shift[0] * normal
        return np.array([delta, delta]), [w for w in widths if w > 0]
    hull = np.linalg.norm(np.diff(cubic, axis=0), axis=1).sum()
    count = min(MAX_SAMPLES, max(MIN_SAMPLES, math.ceil(hull / SAMPLE_SPACING)))
    # Leave the end caps out: the anchors already followed their own readings.
    t = np.linspace(0.125, 0.875, count)
    points, tangents = _bezier(cubic, t)
    assert tangents is not None
    length = np.linalg.norm(tangents, axis=1)
    usable = length > 1e-9
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0])) / np.maximum(
        length[:, None], 1e-9
    )
    shift, widths = reader.across(points, normals)
    usable &= widths > 0
    if np.count_nonzero(usable) < 2:
        return np.zeros((2, 2)), []
    basis = np.column_stack((3 * (1 - t) ** 2 * t, 3 * (1 - t) * t**2))
    weight = np.sqrt(widths[usable] / max(float(widths[usable].max()), 1e-9))
    delta = np.linalg.lstsq(
        basis[usable] * weight[:, None],
        shift[usable, None] * normals[usable] * weight[:, None],
        rcond=None,
    )[0]
    # Keep the old midpoint fit's maximum control motion. Anchors and path
    # structure stay untouched here; the caller judges the exact render.
    bound = 4 / 3 * SHIFT
    length = np.linalg.norm(delta, axis=1)
    delta *= np.minimum(1, bound / np.maximum(length, 1e-9))[:, None]
    return delta, widths[usable].tolist()
