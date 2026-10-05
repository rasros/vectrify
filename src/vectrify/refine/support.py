"""Simplify a fill boundary using a nearby selected stroke above it.

A stroke supplies an existing curve, rather than another independently fitted
approximation of the same edge. No edge labels or closed enclosure are needed.
Only shorter, bounded replacements are proposed; the caller judges their exact
render against its reference before any replacement is retained.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import shapely
from cairosvg.colors import color
from scipy.optimize import minimize_scalar

from vectrify.document import Document, Geometry
from vectrify.document.join import path_style
from vectrify.document.model import EditKind
from vectrify.operations.generate import Region
from vectrify.refine.crossings import bezier, crossings
from vectrify.refine.frozen import Frozen, Paths
from vectrify.refine.snap import _Frame, _frame

SAMPLES = 128


def _spans(geometry: Geometry, frame: _Frame):
    rows = []
    for subpath in geometry.subpaths:
        row = []
        previous = frame.pixels(subpath.nodes[0].values)[-1]
        for node in subpath.nodes[1:]:
            points = frame.pixels(node.values)
            row.append(np.vstack((previous, points)))
            previous = points[-1]
        rows.append(row)
    return rows


def _line(controls):
    if not controls:
        return shapely.LineString()
    return shapely.LineString(
        np.vstack([bezier(c, np.linspace(0, 1, SAMPLES + 1))[0] for c in controls])
    )


def _closest(point, controls, samples) -> float:
    index = int(np.argmin(np.sum((samples - point) ** 2, axis=1)))
    segment = min(index // SAMPLES, len(controls) - 1)
    t = (index - segment * SAMPLES) / SAMPLES
    control = controls[segment]

    def distance(value):
        return float(np.sum((bezier(control, np.array([value]))[0][0] - point) ** 2))

    result = minimize_scalar(
        distance,
        bounds=(max(0, t - 2 / SAMPLES), min(1, t + 2 / SAMPLES)),
        method="bounded",
        options={"xatol": 1e-12},
    )
    value = min((result.fun, result.x), (distance(0), 0), (distance(1), 1))[1]
    return segment + value


def _split(control, t):
    levels = [control]
    while len(levels[-1]) > 1:
        previous = levels[-1]
        levels.append((1 - t) * previous[:-1] + t * previous[1:])
    return np.array([level[0] for level in levels]), np.array(
        [level[-1] for level in reversed(levels)]
    )


def _pieces(controls, start, end):
    low, high = sorted((start, end))
    result = []
    for index, control in enumerate(controls):
        a, b = max(0, low - index), min(1, high - index)
        if b <= a + 1e-9:
            continue
        portion = _split(control, b)[0] if b < 1 else control
        if a > 0:
            portion = _split(portion, a / b)[1]
        result.append(portion)
    return [c[::-1] for c in result[::-1]] if end < start else result


def _point(controls, value):
    index = min(int(value), len(controls) - 1)
    return bezier(controls[index], np.array([value - index]))[0][0]


def _eligible(document: Document, oid: str, kind: str) -> bool:
    ancestry = document.ancestry(oid)
    if any(
        a.tag in {"defs", "clipPath", "mask"}
        or a.get("clip-path")
        or a.get("filter")
        or a.get("mask")
        or float(a.get("opacity", "1") or "1") != 1
        or EditKind.GEOMETRY in a.locks
        for a in ancestry
    ):
        return False
    style = path_style(document, ancestry[-1])
    other = "fill" if kind == "stroke" else "stroke"
    paint = style[kind]
    return (
        paint != "none"
        and "url(" not in paint
        and color(paint)[3] == 1
        and float(style[kind + "-opacity"]) == 1
        and style[other] == "none"
        and (kind != "stroke" or float(style["stroke-width"]) > 0)
        and document.geometry_users(document.geometry_for(oid).id) == {oid}
    )


def supported(
    document: Document,
    paths: Paths,
    region: Region,
    fixed: Frozen,
    tolerance: float,
    deadline: float = float("inf"),
    *,
    accept: Callable[[Paths], bool],
) -> Paths:
    """Propose fewer boundary segments using a later selected sibling stroke.

    Work in reference pixels, including transformed paths and nonzero viewboxes.
    Each run must follow the stroke monotonically and stay within tolerance in
    both directions. Pins and held nodes stay untouched, including explicit
    closing aliases. Unsupported paint/context and ambiguous runs are skipped.
    ``accept`` sees the accumulated candidate and must check its actual render.
    """
    if tolerance <= 0 or time.monotonic() >= deadline:
        return paths
    order = [e.id for e in document.elements() if e.id in paths.geometries]
    parents = {oid: document.ancestry(oid)[-2].id for oid in order}
    strokes = {}
    for oid in order:
        if time.monotonic() >= deadline:
            return paths
        if not _eligible(document, oid, "stroke"):
            continue
        frame = _frame(document, oid, region, region.image.size)
        if frame is None:
            continue
        strokes[oid] = []
        for subpath, row in zip(
            paths.geometries[oid].subpaths,
            _spans(paths.geometries[oid], frame),
            strict=True,
        ):
            # Wrapped correspondences on closed strokes need a separate model.
            if not row or subpath.closed:
                continue
            samples = np.vstack(
                [bezier(c, np.linspace(0, 1, SAMPLES + 1)[:-1])[0] for c in row]
                + [row[-1][-1:]]
            )
            strokes[oid].append((row, samples, shapely.LineString(samples)))
    result = paths
    for index, oid in enumerate(order):
        if not _eligible(document, oid, "fill"):
            continue
        frame = _frame(document, oid, region, region.image.size)
        if frame is None:
            continue
        # Prefer the topmost selected stroke when several strokes cover an edge.
        for stroke_id in reversed(order[index + 1 :]):
            if parents[stroke_id] != parents[oid]:
                continue
            for controls, samples, line in strokes.get(stroke_id, ()):
                if time.monotonic() >= deadline:
                    return result
                geometry = result.geometries[oid]
                for sub_index, row in enumerate(_spans(geometry, frame)):
                    matched = [
                        bool(
                            np.max(
                                shapely.distance(
                                    shapely.points(bezier(c, np.linspace(0, 1, 25))[0]),
                                    line,
                                )
                            )
                            <= tolerance
                        )
                        for c in row
                    ]
                    runs = []
                    a = 0
                    while a < len(row):
                        if not matched[a]:
                            a += 1
                            continue
                        b = a + 1
                        while b < len(row) and matched[b]:
                            b += 1
                        if b - a > 1:
                            runs.append((a, b))
                        a = b
                    for a, b in reversed(runs):
                        if time.monotonic() >= deadline:
                            return result
                        current = result.geometries[oid]
                        contour = current.subpaths[sub_index]
                        nodes = list(contour.nodes)
                        touched = nodes[a : b + 1]
                        alias = (
                            contour.closed
                            and np.linalg.norm(
                                np.array(nodes[0].endpoint) - nodes[-1].endpoint
                            )
                            < 1e-9
                        )
                        if alias and (a == 0 or b == len(nodes) - 1):
                            touched = [*touched, nodes[0], nodes[-1]]
                        if any(n.id in fixed.endpoints or n.pinned for n in touched):
                            continue
                        points = np.vstack(
                            [bezier(c, np.linspace(0, 1, 5)[:-1])[0] for c in row[a:b]]
                            + [row[b - 1][-1:]]
                        )
                        at = np.array([_closest(p, controls, samples) for p in points])
                        # Small backtracking can be the redundant noise we are
                        # removing. A monotone correspondence must still exist
                        # within the pixel tolerance, rather than cutting across
                        # an actual turn of the source contour.
                        monotone = (
                            np.maximum.accumulate(at)
                            if at[-1] >= at[0]
                            else np.minimum.accumulate(at)
                        )
                        if any(
                            np.linalg.norm(_point(controls, value) - point) > tolerance
                            for point, value in zip(points, monotone, strict=True)
                        ):
                            continue
                        pieces = _pieces(controls, at[0], at[-1])
                        if not pieces or len(pieces) >= b - a:
                            continue
                        if (
                            shapely.hausdorff_distance(_line(row[a:b]), _line(pieces))
                            > tolerance
                        ):
                            continue
                        start, end = (
                            frame.local(pieces[0][:1]),
                            frame.local(pieces[-1][-1:]),
                        )
                        nodes[a] = replace(
                            nodes[a], values=(*nodes[a].values[:-2], *start)
                        )
                        previous = nodes[a + 1 : b + 1]
                        ids = [
                            *(n.id for n in previous[: len(pieces) - 1]),
                            previous[-1].id,
                        ]
                        nodes[a + 1 : b + 1] = [
                            replace(
                                previous[i],
                                id=nid,
                                command="C" if len(piece) == 4 else "L",
                                values=frame.local(piece[1:]),
                            )
                            for i, (piece, nid) in enumerate(
                                zip(pieces, ids, strict=True)
                            )
                        ]
                        if alias and a == 0:
                            nodes[-1] = replace(
                                nodes[-1], values=(*nodes[-1].values[:-2], *start)
                            )
                        elif alias and b == len(contour.nodes) - 1:
                            nodes[0] = replace(nodes[0], values=end)
                        if contour.closed and len(nodes) < 3:
                            continue
                        subpaths = list(current.subpaths)
                        subpaths[sub_index] = replace(contour, nodes=tuple(nodes))
                        proposal = replace(current, subpaths=tuple(subpaths))
                        if crossings(proposal) > crossings(paths.geometries[oid]):
                            continue
                        # Multiple supports cannot accumulate more than the cap.
                        if any(
                            shapely.hausdorff_distance(_line(old), _line(new))
                            > tolerance
                            for old, new in zip(
                                _spans(paths.geometries[oid], frame),
                                _spans(proposal, frame),
                                strict=True,
                            )
                        ):
                            continue
                        candidate = replace(
                            result, geometries={**result.geometries, oid: proposal}
                        )
                        if accept(candidate):
                            result = candidate
    return result
