"""Count where an outline crosses itself.

Each contour is drawn as a polyline, bounding each cubic's chord error,
and every pair of non-neighbouring lines that cross counts once. A bow-tie
crosses once; a cubic whose handles tie it into a loop crosses where the loop
closes. Lines that only touch or lie along each other do not count, so a
concave outline that comes close to itself, or meets itself at a point,
counts none. Straight segments are drawn as they are, so two of them crossing
midway are not missed for crossing exactly at a sample.
"""

from __future__ import annotations

import numpy as np
import shapely

from vectrify.document import Geometry


def bezier(control: np.ndarray, t: np.ndarray):
    """Points and tangents along a line (two controls) or cubic (four)."""
    t = t[:, None]
    if len(control) == 2:
        a, b = control
        return a + (b - a) * t, np.repeat((b - a)[None], len(t), axis=0)
    a, b, c, d = control
    u = 1 - t
    points = u**3 * a + 3 * u**2 * t * b + 3 * u * t**2 * c + t**3 * d
    tangents = 3 * u**2 * (b - a) + 6 * u * t * (c - b) + 3 * t**2 * (d - c)
    return points, tangents


def crossing_pairs(points: np.ndarray, closed: bool) -> np.ndarray:
    """(i, j) for every pair of lines of the polyline through *points* that
    cross, line i running from point i to point i + 1."""
    p = np.asarray(points, dtype=np.float64)
    if closed and len(p) and np.any(p[0] != p[-1]):
        p = np.vstack([p, p[:1]])
    if len(p) < 4:
        return np.zeros((0, 2), dtype=int)
    a, b = p[:-1], p[1:]
    n = len(a)
    low, high = np.minimum(a, b), np.maximum(a, b)
    # Lines whose boxes miss each other cannot cross: a tree of the boxes,
    # a hair larger so flat ones stay boxes, finds the pairs whose meet.
    boxes = shapely.box(*(low - 1e-9).T, *(high + 1e-9).T)
    i, j = shapely.STRtree(boxes).query(boxes)
    # Only lines after this one and not next to it, so each pair counts
    # once and neighbours, which always meet, not at all.
    later = j > i + 1
    if closed:
        later &= ~((i == 0) & (j == n - 1))
    later &= np.all(low[i] <= high[j], axis=-1) & np.all(low[j] <= high[i], axis=-1)
    i, j = i[later], j[later]
    order = np.lexsort((j, i))
    i, j = i[order], j[order]

    # Orientation of r against the line from p to q.
    def side(px, py, qx, qy, rx, ry):
        return (qx - px) * (ry - py) - (qy - py) * (rx - px)

    ax, ay, bx, by = a[i, 0], a[i, 1], b[i, 0], b[i, 1]
    cx, cy, dx, dy = a[j, 0], a[j, 1], b[j, 0], b[j, 1]
    d1 = side(ax, ay, bx, by, cx, cy)
    d2 = side(ax, ay, bx, by, dx, dy)
    d3 = side(cx, cy, dx, dy, ax, ay)
    d4 = side(cx, cy, dx, dy, bx, by)
    hit = (d1 * d2 < 0) & (d3 * d4 < 0)
    return np.column_stack([i[hit], j[hit]]).astype(int)


def polyline_crossings(points: np.ndarray, closed: bool) -> int:
    """How many times the polyline through *points* crosses itself."""
    return len(crossing_pairs(points, closed))


def contour_line(controls: list[np.ndarray], samples: int | None = None):
    """The polyline a contour's segments (each 2 or 4 controls) are drawn as,
    and the segment each of its lines is part of."""
    if not controls:
        return np.zeros((0, 2)), np.zeros(0, dtype=int)
    curve = np.array([len(control) == 4 for control in controls])
    counts = np.ones(len(controls), dtype=int)
    if curve.any():
        cubics = np.asarray(
            [c for c, bent in zip(controls, curve, strict=True) if bent]
        )
        if samples is not None:
            counts[curve] = samples
        else:
            # A fixed count per segment misses small loops and changes its
            # verdict after exact subdivision. Bound the chord approximation
            # error instead, using the cubic's second derivative. Straight
            # monotone cubics need only their endpoints.
            extent = np.ptp(np.concatenate(controls), axis=0).max()
            tolerance = max(1e-7, float(extent) * 1e-5)
            derivative = 6 * np.linalg.norm(np.diff(cubics, n=2, axis=1), axis=2).max(1)
            required = np.sqrt(derivative / (8 * tolerance)).clip(2, 512)
            count = 2 ** np.ceil(np.log2(required))
            chord = cubics[:, 3] - cubics[:, 0]
            length2 = (chord**2).sum(1).clip(1e-24)
            handles = cubics[:, 1:3] - cubics[:, :1]
            along = (handles * chord[:, None]).sum(2) / length2[:, None]
            off = handles - along[..., None] * chord[:, None]
            straight = (np.linalg.norm(off, axis=2).max(1) <= tolerance) & (
                (along[:, 0] >= 0) & (along[:, 1] <= 1) & (along[:, 0] <= along[:, 1])
            )
            counts[curve] = np.where(straight, 1, count).astype(int)
    owners = np.repeat(np.arange(len(controls)), counts)
    line = np.empty((1 + len(owners), 2))
    line[0] = np.asarray(controls[0][0], dtype=np.float64)
    # Where each segment's points start in the line, after the first point.
    starts = 1 + np.cumsum(counts) - counts
    ends = [
        np.asarray(c[-1], dtype=np.float64)
        for c, bent in zip(controls, curve, strict=True)
        if not bent
    ]
    if ends:
        line[starts[~curve]] = ends
    if curve.any():
        control = np.asarray(
            [c for c, bent in zip(controls, curve, strict=True) if bent],
            dtype=np.float64,
        )
        for count in np.unique(counts[curve]):
            selected = counts[curve] == count
            t = np.linspace(0, 1, count + 1)[1:, None]
            u = 1 - t
            basis = np.hstack([u**3, 3 * u**2 * t, 3 * u * t**2, t**3])
            points = np.einsum("sk,nkc->nsc", basis, control[selected])
            at = starts[curve][selected, None] + np.arange(count)[None]
            line[at.ravel()] = points.reshape(-1, 2)
    return line, owners


def crossed_nodes(
    geometry: Geometry, samples: int | None = None
) -> tuple[int, set[str]]:
    """How many times *geometry*'s contours cross themselves, in all, and the
    nodes at the ends of the segments that cross.

    One contour crossing another does not count: holes and separate pieces
    are normal, and only a contour folding over itself is a fault.
    """
    total, nodes = 0, set()
    for subpath in geometry.subpaths:
        controls = []
        start = np.array(subpath.nodes[0].endpoint)
        for node in subpath.nodes[1:]:
            points = np.asarray(node.values, dtype=np.float64).reshape(-1, 2)
            controls.append(np.vstack([start, points]))
            start = points[-1]
        line, owners = contour_line(controls, samples)
        pairs = crossing_pairs(line, subpath.closed)
        total += len(pairs)
        # Segment s runs from node s to node s + 1; the closing line, which
        # is not among the drawn ones, back to node 0.
        ids = [n.id for n in subpath.nodes]
        owners = np.append(owners, len(ids) - 1)
        for segment in set(owners[pairs].ravel().tolist()):
            nodes.update((ids[segment], ids[(segment + 1) % len(ids)]))
    return total, nodes


def crossings(geometry: Geometry, samples: int | None = None) -> int:
    """How many times *geometry*'s contours cross themselves, in all."""
    return crossed_nodes(geometry, samples)[0]
