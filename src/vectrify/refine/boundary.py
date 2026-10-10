"""Coherent subpixel reference contours, fitted without discarding knots."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from itertools import pairwise

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.spatial import KDTree

from vectrify.document import Geometry
from vectrify.refine.snap import _Frame


def loops(field: np.ndarray) -> list[np.ndarray]:
    """March the zero crossings between pixel centres, including saddle cells."""
    points, adjacent = {}, {}

    def connect(a, b):
        adjacent.setdefault(a, []).append(b)
        adjacent.setdefault(b, []).append(a)

    negative = field < 0
    cases = (
        negative[:-1, :-1].astype(np.uint8)
        + 2 * negative[:-1, 1:]
        + 4 * negative[1:, 1:]
        + 8 * negative[1:, :-1]
    )
    for y, x in np.argwhere((cases > 0) & (cases < 15)):
        values = [field[y, x], field[y, x + 1], field[y + 1, x + 1], field[y + 1, x]]
        keys = [(0, y, x), (1, y, x + 1), (0, y + 1, x), (1, y, x)]
        corners = np.array([[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1]]) + 0.5
        crossed = []
        for k in range(4):
            v, w = values[k], values[(k + 1) % 4]
            if (v < 0) == (w < 0):
                continue
            key = keys[k]
            crossed.append(key)
            fraction = v / (v - w)
            points[key] = (1 - fraction) * corners[k] + fraction * corners[(k + 1) % 4]
        if len(crossed) == 2:
            connect(*crossed)
        else:
            pairs = (
                [(0, 1), (2, 3)]
                if (values[0] < 0) == (sum(values) < 0)
                else [(0, 3), (1, 2)]
            )
            for a, b in pairs:
                connect(keys[a], keys[b])
    visited, result = set(), []
    for head in adjacent:
        if head in visited:
            continue
        path, previous, current = [], None, head
        while current not in visited:
            visited.add(current)
            path.append(points[current])
            following = adjacent[current]
            next_point = following[0] if following[0] != previous else following[-1]
            previous, current = current, next_point
        if current == head and len(path) > 3:
            result.append(np.asarray(path))
    return result


def field_loops(signal, power, coverage, padding):
    """Preserve the silhouette outside the crop and beneath opaque artwork."""
    visible = power > max(1e-8, float(power.max()) * 0.02)
    if not visible.any():
        return []
    field = (1 - 2 * coverage) * float(np.median(power[visible]))
    h, w = signal.shape
    inner = field[padding : padding + h, padding : padding + w]
    inner[visible] = signal[visible]
    return [p - padding for p in loops(gaussian_filter(field, 0.6))]


def _area(points):
    return (
        points[:, 0] * np.roll(points[:, 1], -1)
        - points[:, 1] * np.roll(points[:, 0], -1)
    ).sum()


def _correspondence(points, target, arc):
    """Assign retained knots to the reference in contour order."""
    walked = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    expected = walked / max(walked[-1], 1e-9) * arc[-1]
    distance = np.square(points[:, None] - target[None]).sum(-1)
    distance += 0.02 * np.square(arc[None] - expected[:, None])
    parent = np.zeros(distance.shape, np.int32)
    best = np.full(len(arc), np.inf)
    best[0] = distance[0, 0]
    for j in range(1, len(points)):
        minimum = np.minimum.accumulate(best)
        parent[j] = np.maximum.accumulate(
            np.where(best <= minimum, np.arange(len(arc)), 0)
        )
        best = minimum + distance[j]
    positions = np.zeros(len(points), int)
    positions[-1] = len(arc) - 1
    for j in range(len(points) - 1, 0, -1):
        positions[j - 1] = parent[j, positions[j]]
    return arc[positions]


def reconstructed(
    geometry: Geometry,
    baseline: Geometry,
    frame: _Frame,
    contours: list[np.ndarray],
    pixel_scale: float,
    displacement: float,
    held: frozenset[str],
    accept: Callable[[Geometry], bool],
    stopped: Callable[[], bool],
) -> Geometry:
    """Try sections of a shared smooth contour with the original point IDs.

    The callback judges exact rendering and crossings. A section includes each
    knot's two arms, with a shared movement cap, so clipping cannot make a spike.
    Protected contours and implicit straight closures use the local cleanup.
    """
    if not contours or displacement <= 0:
        return geometry
    original = {n.id: n for s in baseline.subpaths for n in s.nodes}
    for si, subpath in enumerate(geometry.subpaths):
        nodes = subpath.nodes
        if (
            stopped()
            or not subpath.closed
            or not 4 <= len(nodes) <= 1024
            or nodes[0].endpoint != nodes[-1].endpoint
            or any(n.id in held or n.pinned or n.feature is not None for n in nodes)
            or any(n.command != "C" for n in nodes[1:])
        ):
            continue
        points = frame.pixels(tuple(v for n in nodes for v in n.endpoint))
        loop = min(contours, key=lambda p: np.median(KDTree(p).query(points)[0]))
        if _area(loop) * _area(points) < 0:
            loop = loop[::-1]
        loop = np.vstack((loop, loop[:1]))
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(loop, axis=0), axis=1))]
        keep = np.r_[True, np.diff(arc) > 1e-6]
        loop, arc = loop[keep], arc[keep]
        if len(arc) < 4 or arc[-1] < 1:
            continue
        grid = np.linspace(
            0, arc[-1], min(4096, max(16, int(arc[-1] * pixel_scale * 4)))
        )
        uniform = np.column_stack([np.interp(grid, arc, loop[:, k]) for k in range(2)])
        walked = (
            np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
            * pixel_scale
        )
        for radius in (0.4, 0.8):
            if stopped():
                return geometry
            target = gaussian_filter1d(
                uniform[:-1],
                radius / pixel_scale / (grid[1] - grid[0]),
                axis=0,
                mode="wrap",
            )
            target = np.roll(target, -int(KDTree(target).query(points[0])[1]), axis=0)
            target = np.vstack((target, target[:1]))
            positions = _correspondence(points, target, grid)
            spline = CubicSpline(grid, target, bc_type="periodic")
            destination = [spline(positions[0])[None]]
            for low, high in pairwise(positions):
                start, end = spline(low), spline(high)
                destination.append(
                    np.array(
                        [
                            start + spline(low, 1) * (high - low) / 3,
                            end - spline(high, 1) * (high - low) / 3,
                            end,
                        ]
                    )
                )
            destination = [
                np.asarray(frame.local(v)).reshape(-1, 2) for v in destination
            ]
            origins = [np.asarray(original[n.id].values).reshape(-1, 2) for n in nodes]
            shifts = [
                v - origin for v, origin in zip(destination, origins, strict=True)
            ]
            fractions: list[float] = []
            for j in range(len(nodes)):
                group = [shifts[j][-1]]
                if j:
                    group.append(shifts[j][1])
                if j + 1 < len(nodes):
                    group.append(shifts[j + 1][0])
                fractions.append(
                    min(
                        1,
                        displacement / max(1e-9, max(np.linalg.norm(v) for v in group)),
                    )
                )
            fractions[0] = fractions[-1] = min(fractions[0], fractions[-1])
            proposed = [origins[0] + fractions[0] * shifts[0]]
            for j in range(1, len(nodes)):
                proposed.append(
                    origins[j]
                    + np.array([fractions[j - 1], fractions[j], fractions[j]])[:, None]
                    * shifts[j]
                )
            subpaths = list(geometry.subpaths)
            subpaths[si] = replace(
                geometry.subpaths[si],
                nodes=tuple(
                    replace(n, values=tuple(v.ravel()))
                    for n, v in zip(nodes, proposed, strict=True)
                ),
            )
            candidate = replace(geometry, subpaths=tuple(subpaths))
            if not stopped() and candidate != geometry and accept(candidate):
                geometry = candidate
            # Judge bounded runs independently: a real notch or an occluded
            # section cannot veto smoothing elsewhere along the same contour.
            for start in np.arange(0, walked[-1], 8):
                if stopped():
                    return geometry
                indices = np.flatnonzero((walked >= start) & (walked < start + 8))
                current = geometry.subpaths[si]
                changed = list(current.nodes)
                for j in indices:
                    values = np.asarray(changed[j].values).reshape(-1, 2).copy()
                    values[-1] = proposed[j][-1]
                    if j:
                        values[1] = proposed[j][1]
                    changed[j] = replace(changed[j], values=tuple(values.ravel()))
                    following = j + 1
                    if following == len(nodes):
                        changed[0] = replace(
                            changed[0], values=tuple(proposed[0].ravel())
                        )
                        following = 1
                    values = np.asarray(changed[following].values).reshape(-1, 2).copy()
                    values[0] = proposed[following][0]
                    changed[following] = replace(
                        changed[following], values=tuple(values.ravel())
                    )
                    if j == 0:
                        values = np.asarray(changed[-1].values).reshape(-1, 2).copy()
                        values[1:] = proposed[-1][1:]
                        changed[-1] = replace(changed[-1], values=tuple(values.ravel()))
                subpaths = list(geometry.subpaths)
                subpaths[si] = replace(current, nodes=tuple(changed))
                candidate = replace(geometry, subpaths=tuple(subpaths))
                if candidate != geometry and accept(candidate):
                    geometry = candidate
    return geometry
