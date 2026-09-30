"""Count where an outline crosses itself.

Each contour is drawn as a polyline, every segment sampled at a few points,
and every pair of non-neighbouring lines that cross counts once. A bow-tie
crosses once; a cubic whose handles tie it into a loop crosses where the loop
closes. Lines that only touch or lie along each other do not count, so a
concave outline that comes close to itself, or meets itself at a point,
counts none. Straight segments are drawn as they are, so two of them crossing
midway are not missed for crossing exactly at a sample.
"""

from __future__ import annotations

import numpy as np

from vectrify.document import Geometry

# How many points each segment is drawn with.
SAMPLES = 8
# How many lines are tested against all the others at once.
_BLOCK = 512


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
    index = np.arange(n)

    # Orientation of r against the line from p to q.
    def side(px, py, qx, qy, rx, ry):
        return (qx - px) * (ry - py) - (qy - py) * (rx - px)

    found = [np.zeros((0, 2), dtype=int)]
    for start in range(0, n, _BLOCK):
        rows = slice(start, min(n, start + _BLOCK))
        # Only lines after this one and not next to it, so each pair counts
        # once and neighbours, which always meet, not at all.
        later = index[None] > index[rows, None] + 1
        if closed:
            later &= ~((index[rows, None] == 0) & (index[None] == n - 1))
        # Lines whose boxes miss each other cannot cross.
        later &= np.all(low[rows, None] <= high[None], axis=-1)
        later &= np.all(low[None] <= high[rows, None], axis=-1)
        i, j = np.nonzero(later)
        if not len(i):
            continue
        i += start
        ax, ay, bx, by = a[i, 0], a[i, 1], b[i, 0], b[i, 1]
        cx, cy, dx, dy = a[j, 0], a[j, 1], b[j, 0], b[j, 1]
        d1 = side(ax, ay, bx, by, cx, cy)
        d2 = side(ax, ay, bx, by, dx, dy)
        d3 = side(cx, cy, dx, dy, ax, ay)
        d4 = side(cx, cy, dx, dy, bx, by)
        hit = (d1 * d2 < 0) & (d3 * d4 < 0)
        found.append(np.column_stack([i[hit], j[hit]]))
    return np.vstack(found)


def polyline_crossings(points: np.ndarray, closed: bool) -> int:
    """How many times the polyline through *points* crosses itself."""
    return len(crossing_pairs(points, closed))


def contour_line(controls: list[np.ndarray], samples: int = SAMPLES):
    """The polyline a contour's segments (each 2 or 4 controls) are drawn as,
    and the segment each of its lines is part of."""
    t = np.linspace(0, 1, samples + 1)[1:]
    line = [np.asarray(controls[0][0], dtype=np.float64)[None]] if controls else []
    owners = []
    for segment, control in enumerate(controls):
        control = np.asarray(control, dtype=np.float64)
        points = control[1:] if len(control) == 2 else bezier(control, t)[0]
        line.append(points)
        owners.extend([segment] * len(points))
    if not line:
        return np.zeros((0, 2)), np.zeros(0, dtype=int)
    return np.vstack(line), np.array(owners, dtype=int)


def crossed_nodes(geometry: Geometry, samples: int = SAMPLES) -> tuple[int, set[str]]:
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


def crossings(geometry: Geometry, samples: int = SAMPLES) -> int:
    """How many times *geometry*'s contours cross themselves, in all."""
    return crossed_nodes(geometry, samples)[0]
