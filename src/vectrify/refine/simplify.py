"""Simplify: remove the points the outline does not need.

A point goes when the two segments either side of it can be drawn as one
without the outline moving more than a tolerance, in the reference's pixels.
The joined segment keeps the directions the outline leaves and arrives in
and picks its handle lengths to follow the old outline most closely. The
point whose removal moves the outline least goes first, and it repeats
until every remaining point would move it too far.

Pinned points and linked edges stay, as does anything a contour needs to
stay a shape: two points when open, three when closed.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from vectrify.document import Document, Geometry, PathNode
from vectrify.operations.generate import Region
from vectrify.refine.frozen import Frozen, Paths
from vectrify.refine.snap import _bezier, _Frame, _frame

# Points each segment is drawn with to measure how far the outline moved.
SAMPLES = 24


def simplify(
    document: Document, paths: Paths, region: Region, fixed: Frozen, tolerance: float
) -> Paths:
    """*paths* with the points removed that move no outline over *tolerance* px."""
    geometries = dict(paths.geometries)
    for oid, geometry in paths.geometries.items():
        frame = _frame(document, oid, region, region.image.size)
        if frame is None:
            continue
        geometries[oid] = _simplified(geometry, frame, fixed, tolerance)
    return replace(paths, geometries=geometries)


def _simplified(
    geometry: Geometry, frame: _Frame, fixed: Frozen, tolerance: float
) -> Geometry:
    subpaths = []
    for subpath in geometry.subpaths:
        nodes = list(subpath.nodes)
        least = 3 if subpath.closed else 2
        while len(nodes) > least:
            best: tuple[float, int, PathNode] | None = None
            # The first and last points end the contour, or close it.
            for i in range(1, len(nodes) - 1):
                if _stays(nodes[i], fixed):
                    continue
                joined = _join(nodes[i - 1], nodes[i], nodes[i + 1], frame)
                if joined is None:
                    continue
                moved, node = joined
                if moved <= tolerance and (best is None or moved < best[0]):
                    best = (moved, i, node)
            closing = _closing(nodes, subpath.closed, fixed, frame)
            if (
                closing is not None
                and closing <= tolerance
                and (best is None or closing < best[0])
            ):
                nodes.pop()
                continue
            if best is None:
                break
            _moved, i, node = best
            nodes[i : i + 2] = [node]
        subpaths.append(replace(subpath, nodes=tuple(nodes)))
    return replace(geometry, subpaths=tuple(subpaths))


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
    return node.id in fixed.nodes or node.id in fixed.endpoints


def _controls(start: np.ndarray, node: PathNode, frame: _Frame) -> np.ndarray:
    return np.vstack([start, frame.pixels(node.values)])


def _join(
    before: PathNode, middle: PathNode, after: PathNode, frame: _Frame
) -> tuple[float, PathNode] | None:
    """The segment replacing the two either side of *middle*, and how far it
    strays from them in pixels; *after* keeps its ID."""
    start = frame.pixels(before.values)[-1]
    first = _controls(start, middle, frame)
    second = _controls(first[-1], after, frame)
    t = np.linspace(0, 1, SAMPLES)
    old = np.vstack([_bezier(first, t)[0], _bezier(second, t)[0][1:]])
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
    start, step = line[:-1], np.diff(line, axis=0)
    length = np.maximum((step**2).sum(axis=1), 1e-12)
    along = ((points[:, None, :] - start[None]) * step[None]).sum(axis=2) / length
    nearest = start[None] + np.clip(along, 0, 1)[..., None] * step[None]
    gaps = np.linalg.norm(points[:, None, :] - nearest, axis=2)
    return float(gaps.min(axis=1).max())


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
