"""Simplify: remove the points the outline does not need.

A point goes when the two segments either side of it can be drawn as one
without the outline moving more than a tolerance, in the reference's pixels.
The joined segment keeps the directions the outline leaves and arrives in
and picks its handle lengths to follow the old outline most closely. The
point whose removal moves the outline least goes first, and it repeats
until every remaining point would move it too far.

Pinned points stay, as does anything a contour needs to
stay a shape: two points when open, three when closed.
"""

from __future__ import annotations

import time
from dataclasses import replace
from itertools import pairwise

import numpy as np

from vectrify.document import Document, Geometry, PathNode
from vectrify.operations.generate import Region
from vectrify.refine.frozen import Frozen, Paths
from vectrify.refine.snap import _bezier, _Frame, _frame

# Points each segment is drawn with to measure how far the outline moved, and
# the most a join is measured against.
SAMPLES = 24
MEASURED = 64


def simplify(
    document: Document,
    paths: Paths,
    region: Region,
    fixed: Frozen,
    tolerance: float,
    deadline: float = float("inf"),
) -> Paths:
    """*paths* with the points removed that move no outline over *tolerance* px.

    Past *deadline* (time.monotonic()) no more points go, and each path
    keeps those removed so far.
    """
    geometries = dict(paths.geometries)
    for oid, geometry in paths.geometries.items():
        frame = _frame(document, oid, region, region.image.size)
        if frame is None:
            continue
        # Points go first: a curve drawn as a line joins its neighbours worse.
        geometry = _simplified(geometry, frame, fixed, tolerance, deadline)
        geometries[oid] = straightened(geometry, tolerance, frame)
    return replace(paths, geometries=geometries)


def curved(geometry: Geometry) -> Geometry:
    """*geometry* with its straight segments as curves, still straight, their
    handles at the thirds, so a fit can bend them."""
    subpaths = []
    for subpath in geometry.subpaths:
        nodes = list(subpath.nodes)
        for i in range(1, len(nodes)):
            node = nodes[i]
            if node.command != "L":
                continue
            a = np.asarray(nodes[i - 1].values[-2:], dtype=np.float64)
            b = np.asarray(node.values[-2:], dtype=np.float64)
            values = (*(a + (b - a) / 3), *(a + 2 * (b - a) / 3), *b)
            nodes[i] = replace(
                node, command="C", values=tuple(float(v) for v in values)
            )
        subpaths.append(replace(subpath, nodes=tuple(nodes)))
    return replace(geometry, subpaths=tuple(subpaths))


def straightened(
    geometry: Geometry, tolerance: float, frame: _Frame | None = None
) -> Geometry:
    """*geometry* with every curve whose handles lie within *tolerance* of the
    line between its ends drawn as that line: the handles do nothing there.
    In pixels through *frame*, else in the path's own units."""
    frame = frame or _Frame(np.eye(2), np.zeros(2))
    subpaths = []
    for subpath in geometry.subpaths:
        nodes = list(subpath.nodes)
        for i in range(1, len(nodes)):
            node = nodes[i]
            if node.command != "C":
                continue
            start = frame.pixels(nodes[i - 1].values)[-1]
            c1, c2, end = frame.pixels(node.values)
            if (
                _off_line(start, end, c1) <= tolerance
                and _off_line(start, end, c2) <= tolerance
            ):
                nodes[i] = replace(node, command="L", values=node.values[-2:])
        subpaths.append(replace(subpath, nodes=tuple(nodes)))
    return replace(geometry, subpaths=tuple(subpaths))


def _off_line(start: np.ndarray, end: np.ndarray, point: np.ndarray) -> float:
    """How far *point* is from the segment between *start* and *end*."""
    step = end - start
    length = float(step @ step)
    if length < 1e-12:
        return float(np.linalg.norm(point - start))
    along = np.clip(float((point - start) @ step) / length, 0.0, 1.0)
    return float(np.linalg.norm(start + along * step - point))


def simplified_geometry(
    geometry: Geometry, fixed: Frozen, tolerance: float
) -> Geometry:
    """*geometry*, already in pixels, simplified to *tolerance* pixels."""
    return _simplified(geometry, _Frame(np.eye(2), np.zeros(2)), fixed, tolerance)


def _simplified(
    geometry: Geometry,
    frame: _Frame,
    fixed: Frozen,
    tolerance: float,
    deadline: float = float("inf"),
) -> Geometry:
    subpaths = []
    for subpath in geometry.subpaths:
        if time.monotonic() >= deadline:
            subpaths.append(subpath)
            continue
        nodes = list(subpath.nodes)
        least = 3 if subpath.closed else 2
        # The outline each segment stands for, as it was before any point
        # went: a join is fitted to and measured against it, so removals
        # never drift further than the tolerance from the original.
        spans = _spans(nodes, frame)
        costs = []
        for i in range(len(nodes)):
            if time.monotonic() >= deadline:
                break
            costs.append(_cost(nodes, spans, i, fixed, frame))
        if len(costs) < len(nodes):
            subpaths.append(subpath)
            continue
        while len(nodes) > least and time.monotonic() < deadline:
            choices = [(cost[0], i) for i, cost in enumerate(costs) if cost is not None]
            best = min(choices, default=None)
            closing = _closing(nodes, subpath.closed, fixed, frame)
            if (
                closing is not None
                and closing <= tolerance
                and (best is None or closing < best[0])
            ):
                nodes.pop()
                spans.pop()
                costs.pop()
                for j in (len(nodes) - 2, len(nodes) - 1):
                    costs[j] = _cost(nodes, spans, j, fixed, frame)
                continue
            if best is None or best[0] > tolerance:
                break
            i = best[1]
            cost = costs[i]
            assert cost is not None
            nodes[i : i + 2] = [cost[1]]
            spans[i : i + 2] = [np.vstack([spans[i], spans[i + 1][1:]])]
            costs[i : i + 2] = [None]
            # Only the joins beside the new segment changed.
            for j in (i - 1, i):
                if 0 <= j < len(nodes):
                    costs[j] = _cost(nodes, spans, j, fixed, frame)
        subpaths.append(replace(subpath, nodes=tuple(nodes)))
    return replace(geometry, subpaths=tuple(subpaths))


def _spans(nodes: list[PathNode], frame: _Frame) -> list[np.ndarray]:
    """Points along the segment ending at each node; none before the first."""
    t = np.linspace(0, 1, SAMPLES)
    spans = [np.empty((0, 2))]
    for before, node in pairwise(nodes):
        start = frame.pixels(before.values)[-1]
        spans.append(_bezier(_controls(start, node, frame), t)[0])
    return spans


def _cost(nodes, spans, i: int, fixed: Frozen, frame: _Frame):
    """(how far removing point *i* moves the outline, the joined node), or
    None where the point has to stay."""
    # The first and last points end the contour, or close it.
    if not 0 < i < len(nodes) - 1 or _stays(nodes[i], fixed):
        return None
    old = np.vstack([spans[i], spans[i + 1][1:]])
    # A long merged span measures just as well from fewer of its points.
    if len(old) > MEASURED:
        old = old[np.linspace(0, len(old) - 1, MEASURED).round().astype(int)]
    return _join(nodes[i - 1], nodes[i], nodes[i + 1], frame, old)


def _closing(nodes, closed: bool, fixed: Frozen, frame: _Frame) -> float | None:
    """How far the outline moves if a closed contour's last point goes, the
    closing line then running from the one before it; None if it cannot."""
    last = nodes[-1]
    if not closed or _stays(last, fixed):
        return None
    first = frame.pixels(nodes[0].values)[-1]
    end = frame.pixels(last.values)[-1]
    if np.linalg.norm(end - first) < 1e-6:
        # The contour closes on its own last point: that point is the first.
        return None
    start = frame.pixels(nodes[-2].values)[-1]
    t = np.linspace(0, 1, SAMPLES)
    old = np.vstack([_bezier(_controls(start, last, frame), t)[0], first])
    return _apart(old, np.vstack([start, first]))


def _stays(node: PathNode, fixed: Frozen) -> bool:
    return node.id in fixed.endpoints


def _controls(start: np.ndarray, node: PathNode, frame: _Frame) -> np.ndarray:
    return np.vstack([start, frame.pixels(node.values)])


def _join(
    before: PathNode, middle: PathNode, after: PathNode, frame: _Frame, old
) -> tuple[float, PathNode] | None:
    """The segment replacing the two either side of *middle*, and how far it
    strays from the outline *old* they stand for, in pixels; *after* keeps
    its ID."""
    start = frame.pixels(before.values)[-1]
    first = _controls(start, middle, frame)
    second = _controls(first[-1], after, frame)
    end = second[-1]
    if middle.command == "L" and after.command == "L":
        control = np.vstack([start, end])
        command = "L"
    else:
        control = _cubic(start, first, second, end, old)
        if control is None:
            return None
        command = "C"
    moved = _apart(old, _bezier(control, np.linspace(0, 1, 2 * SAMPLES))[0])
    values = frame.local(control[1:])
    return moved, replace(after, command=command, values=values)


def _apart(a: np.ndarray, b: np.ndarray) -> float:
    """How far apart two polylines are: the furthest either strays from the other."""
    return max(_furthest(a, b), _furthest(b, a))


def _furthest(points: np.ndarray, line: np.ndarray) -> float:
    # Each coordinate on its own, which spares the temporaries of a third
    # axis; the square root, which keeps order, is taken of the result only.
    x, y = points[:, 0, None], points[:, 1, None]
    sx, sy = line[None, :-1, 0], line[None, :-1, 1]
    dx, dy = np.diff(line[:, 0])[None], np.diff(line[:, 1])[None]
    length = np.maximum(dx**2 + dy**2, 1e-12)
    along = np.clip(((x - sx) * dx + (y - sy) * dy) / length, 0, 1)
    gx = x - (sx + along * dx)
    gy = y - (sy + along * dy)
    return float(np.sqrt((gx * gx + gy * gy).min(axis=1).max()))


def _cubic(start, first, second, end, old) -> np.ndarray | None:
    """One cubic from *start* to *end* leaving and arriving as the old two did,
    with the handle lengths that follow *old* most closely."""
    leave = _direction(first[1:], start)
    arrive = _direction(second[-2::-1], end)
    if leave is None or arrive is None:
        return None
    # Parameters along the old outline by length.
    steps = np.linalg.norm(np.diff(old, axis=0), axis=1)
    total = steps.sum()
    if total < 1e-9:
        return None
    t = np.concatenate([[0], np.cumsum(steps)]) / total
    u = 1 - t
    fixed = (u**3 + 3 * u**2 * t)[:, None] * start + (3 * u * t**2 + t**3)[
        :, None
    ] * end
    basis = np.stack(
        [
            (3 * u**2 * t)[:, None] * leave,
            (3 * u * t**2)[:, None] * arrive,
        ],
        axis=-1,
    ).reshape(-1, 2)
    lengths, *_ = np.linalg.lstsq(basis, (old - fixed).reshape(-1), rcond=None)
    chord = float(np.linalg.norm(end - start))
    if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
        lengths = np.array([chord / 3, chord / 3])
    return np.vstack(
        [start, start + lengths[0] * leave, end + lengths[1] * arrive, end]
    )


def _direction(points: np.ndarray, anchor: np.ndarray) -> np.ndarray | None:
    """The unit direction from *anchor* to the first of *points* apart from it."""
    for point in points:
        offset = point - anchor
        size = float(np.linalg.norm(offset))
        if size > 1e-6:
            return offset / size
    return None
