"""Trace a raster mask's outlines as cubic Bezier paths.

Each pixel-boundary loop of a mask, holes included, is optionally smoothed,
split at its sharpest turns and fitted with one cubic per stretch; the dense
path can then be simplified to a tolerance.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import cast

import numpy as np
from scipy import ndimage

from vectrify.refine.bezier import derivative, evaluate, fit_cubic

# Outlines are traced with one curve per DENSITY pixels of their length.
DENSITY = 6


def _loops(mask: np.ndarray) -> list[list[tuple[float, float]]]:
    """Trace pixel-boundary loops, retaining exterior and hole contours."""
    edges: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    height, width = mask.shape
    # Only a pixel with a neighbour outside has an edge; walking just those,
    # in the same order, leaves a large region's interior out of the loop.
    padded = np.pad(np.asarray(mask, dtype=bool), 1)
    inner = padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2]
    inner &= padded[1:-1, 2:]
    for y, x in zip(*np.nonzero(mask & ~inner), strict=True):
        if y == 0 or not mask[y - 1, x]:
            edges[(x, y)].append((x + 1, y))
        if x == width - 1 or not mask[y, x + 1]:
            edges[(x + 1, y)].append((x + 1, y + 1))
        if y == height - 1 or not mask[y + 1, x]:
            edges[(x + 1, y + 1)].append((x, y + 1))
        if x == 0 or not mask[y, x - 1]:
            edges[(x, y + 1)].append((x, y))
    loops: list[list[tuple[float, float]]] = []
    while edges:
        start = next(iter(edges))
        current, loop = start, [cast(tuple[float, float], tuple(map(float, start)))]
        while current in edges:
            following = edges[current].pop()
            if not edges[current]:
                del edges[current]
            current = following
            if current == start:
                break
            loop.append(cast(tuple[float, float], tuple(map(float, current))))
        if current == start and len(loop) >= 3:
            loops.append(loop)
    return loops


def _curvature_scores(loop: list[tuple[float, float]]) -> np.ndarray:
    """Return a scale-aware cosine curvature score for a contour."""
    points = np.asarray(loop, dtype=np.float32)
    size = len(points)
    step = max(1, size // 12)
    before = points - np.roll(points, step, axis=0)
    after = np.roll(points, -step, axis=0) - points
    denom = np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1)
    return np.divide(
        (before * after).sum(axis=1), denom, out=np.ones(size), where=denom > 0
    )


def _curviest(loop: list[tuple[float, float]], count: int) -> list[int]:
    """Global curvature maxima each excluding its close neighbours."""
    size = len(loop)
    count = min(count, size)
    score = _curvature_scores(loop)
    blocked = np.zeros(size, dtype=bool)
    chosen: list[int] = []
    exclusion = max(1, size // (count * 2))
    for _ in range(count):
        available = np.where(~blocked)[0]
        if len(available) == 0:
            break
        index = int(available[np.argmin(score[available])])
        chosen.append(index)
        offsets = (np.arange(index - exclusion, index + exclusion + 1) % size).astype(
            int
        )
        blocked[offsets] = True
    return sorted(chosen)


def _fit_cubic(
    points: np.ndarray, *, reparameterize: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Fit fixed-endpoint cubic controls, refining the samples' parameters.

    The fit starts with uniformly spaced ``t`` values, then applies a
    Newton--Raphson reparameterisation before a final least-squares
    control-point fit.  Pixel contours have highly uneven
    arc-length samples around corners, so this matters even with a fixed number
    of curves.
    """
    t = np.linspace(0.0, 1.0, len(points), dtype=np.float64)
    controls = fit_cubic(points, t)
    if reparameterize and len(points) > 2:
        # The endpoints must remain exactly 0 and 1.  Keeping interior values
        # ordered avoids a folded parameterisation on jagged raster contours.
        epsilon = 1e-5
        for _iteration in range(8):
            start, p0, p1, end = controls
            omt = 1 - t
            first = derivative(controls, t)
            second = 6 * omt[:, None] * (p1 - 2 * p0 + start) + 6 * t[:, None] * (
                end - 2 * p1 + p0
            )
            offset = evaluate(controls, t) - points
            numerator = (offset * first).sum(axis=1)
            denominator = (first * first).sum(axis=1) + (offset * second).sum(axis=1)
            updated = t.copy()
            valid = np.abs(denominator[1:-1]) > 1e-10
            # Raster corners can make an unconstrained Newton step enormous;
            # a short, damped step retains the convergence benefit without
            # collapsing several samples onto one parameter value.
            delta = np.clip(
                numerator[1:-1][valid] / denominator[1:-1][valid], -0.05, 0.05
            )
            interior = updated[1:-1]
            interior[valid] -= delta
            updated[1:-1] = interior
            updated[0], updated[-1] = 0.0, 1.0
            updated[1:-1] = np.clip(updated[1:-1], epsilon, 1 - epsilon)
            updated = np.maximum.accumulate(updated)
            updated[-1] = 1.0
            if np.max(np.abs(updated - t)) < 1e-4:
                break
            t = updated
            controls = fit_cubic(points, t)
    return controls[1], controls[2]


def _fit_cubics(samples: list[np.ndarray]) -> np.ndarray:
    """:func:`_fit_cubic` of each of *samples* at once: their two control
    points, by sample. Each comes out exactly as it would alone, the same
    arithmetic on each point; only the least-squares solves go one by one."""
    count = len(samples)
    sizes = np.array([len(sample) for sample in samples])
    width = int(sizes.max())
    rows = np.arange(count)
    # Each sample padded with its last point at parameter 1, kept finite.
    points = np.empty((count, width, 2), dtype=np.float32)
    t = np.ones((count, width), dtype=np.float64)
    for row, sample in enumerate(samples):
        points[row, : len(sample)] = sample
        points[row, len(sample) :] = sample[-1]
        t[row, : len(sample)] = np.linspace(0.0, 1.0, len(sample), dtype=np.float64)
    controls = np.empty((count, 4, 2), dtype=np.float64)

    def solve(which: np.ndarray) -> None:
        for row in which.tolist():
            controls[row] = fit_cubic(points[row, : sizes[row]], t[row, : sizes[row]])

    solve(rows)
    # As _fit_cubic refines each sample's parameters, until they settle.
    epsilon = 1e-5
    index = np.arange(width)
    active = rows[sizes > 2]
    for _iteration in range(8):
        if not active.size:
            break
        control = controls[active]
        first_point, p0, p1, last_point = (control[:, k, None] for k in range(4))
        before = t[active]
        last = sizes[active] - 1
        omt = 1 - before
        first = derivative(control, before)
        second = 6 * omt[..., None] * (p1 - 2 * p0 + first_point) + 6 * before[
            ..., None
        ] * (last_point - 2 * p1 + p0)
        offset = evaluate(control, before) - points[active]
        numerator = (offset * first).sum(axis=-1)
        denominator = (first * first).sum(axis=-1) + (offset * second).sum(axis=-1)
        inner = (index >= 1) & (index < last[:, None])
        valid = inner & (np.abs(denominator) > 1e-10)
        with np.errstate(divide="ignore", invalid="ignore"):
            delta = np.clip(numerator / denominator, -0.05, 0.05)
        updated = np.where(valid, before - delta, before)
        updated[:, 0] = 0.0
        each = np.arange(len(active))
        updated[each, last] = 1.0
        updated = np.where(inner, np.clip(updated, epsilon, 1 - epsilon), updated)
        updated = np.maximum.accumulate(updated, axis=1)
        updated[each, last] = 1.0
        change = np.where(index <= last[:, None], np.abs(updated - before), 0)
        unsettled = change.max(axis=1) >= 1e-4
        active = active[unsettled]
        t[active] = updated[unsettled]
        solve(active)
    return controls[:, 1:3]


def _smoothed(loop: list[tuple[float, float]], sigma: float):
    """*loop* smoothed along its length by a Gaussian of *sigma* pixels.

    A pixel loop walks every step of a raster staircase, and with enough
    curves the fit follows each one: a mask made at a lower resolution
    than the image comes out as steps one of its pixels wide. Smoothing over about
    that width takes the steps out and rounds a real corner by as much.
    """
    if sigma <= 0 or len(loop) < 3:
        return loop
    points = np.asarray(loop, dtype=np.float64)
    reach = min(int(np.ceil(3 * sigma)), (len(points) - 1) // 2)
    smooth = ndimage.gaussian_filter1d(points, sigma, axis=0, mode="wrap", radius=reach)
    return [(float(x), float(y)) for x, y in smooth]


def _cubic_loop(
    loop: list[tuple[float, float]], segments: int, *, smooth: float = 0.0
) -> str | None:
    size = len(loop)
    if size < 3:
        return None
    loop = _smoothed(loop, smooth)
    corners = _curviest(loop, segments)
    if len(corners) < 3:
        return None
    points = np.asarray(loop, dtype=np.float32)
    parts = [f"M {points[corners[0], 0]:.2f} {points[corners[0], 1]:.2f}"]
    pairs = list(zip(corners, [*corners[1:], corners[0]], strict=True))
    # Each sample's indices already include its endpoint. Repeating it adds
    # an artificial least-squares weight at every selected corner and bends
    # each fitted cubic toward its end point rather than the contour data.
    samples = [
        points[
            np.arange(first, second + 1 if second >= first else second + size + 1)
            % size
        ]
        for first, second in pairs
    ]
    for (_, second), (control_a, control_b) in zip(
        pairs, _fit_cubics(samples), strict=True
    ):
        end = points[second]
        parts.append(
            f"C {control_a[0]:.2f} {control_a[1]:.2f} "
            f"{control_b[0]:.2f} {control_b[1]:.2f} {end[0]:.2f} {end[1]:.2f}"
        )
    return " ".join(parts) + " Z"


def mask_path(
    mask: np.ndarray, *, smooth: float = 0.0, density: int = DENSITY
) -> str | None:
    """Fit every mask contour with cubic Beziers, one per *density* pixels of
    its length, each contour first smoothed over *smooth* pixels."""
    parts = [
        piece
        for loop in _loops(mask)
        if (piece := _cubic_loop(loop, max(4, len(loop) // density), smooth=smooth))
    ]
    return " ".join(parts) or None


def _simplified_data(data: str, tolerance: float) -> str:
    """Path *data*, in pixels, with the points it does not need removed."""
    from vectrify.document.svg import parse_path
    from vectrify.refine.frozen import Frozen
    from vectrify.refine.simplify import simplified_geometry

    geometry = simplified_geometry(parse_path(data), Frozen(frozenset()), tolerance)
    # Hundredths of a pixel, as the tracer writes them.
    return re.sub(
        r"-?\d+\.\d+(?:e-?\d+)?",
        lambda number: f"{float(number.group()):.2f}",
        geometry.path_data(),
    )
