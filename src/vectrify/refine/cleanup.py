"""Reference-judged local bridges for traced dents and handle wiggles."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from itertools import pairwise

import numpy as np

from vectrify.document import Geometry, PathNode
from vectrify.refine.crossings import bezier
from vectrify.refine.snap import _Frame


def _sample(nodes: tuple[PathNode, ...] | list[PathNode], frame: _Frame):
    """Sample each segment at quarters, with arc distances in reference pixels."""
    points, tangents = [], []
    for previous, node in pairwise(nodes):
        control = np.vstack((previous.endpoint, np.asarray(node.values).reshape(-1, 2)))
        p, v = bezier(control, np.linspace(0, 1, 5))
        points.extend(p[:-1])
        tangents.extend(v[:-1])
    last = nodes[-1]
    endpoint = np.asarray(last.endpoint)
    points.append(endpoint)
    tangents.append(
        3 * (endpoint - last.values[2:4])
        if last.command == "C"
        else endpoint - nodes[-2].endpoint
    )
    points = np.asarray(points)
    pixels = frame.pixels(tuple(points.ravel()))
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pixels, axis=0), axis=1))]
    return points, np.asarray(tangents), arc


def _fitted(points: np.ndarray, target: np.ndarray, arc: np.ndarray):
    """Fit a cubic to the observed edge, fixing the span's two endpoints."""
    t = (arc / max(arc[-1], 1e-9))[:, None]
    u = 1 - t
    basis = np.hstack((3 * u * u * t, 3 * u * t * t))
    fixed = u**3 * points[0] + t**3 * points[-1]
    controls, *_ = np.linalg.lstsq(basis, target - fixed, rcond=None)
    return controls, fixed + basis @ controls


def cleaned(
    geometry: Geometry,
    baseline: Geometry,
    frame: _Frame,
    displacement: float,
    held: frozenset[str],
    accept: Callable[[Geometry], bool],
    stopped: Callable[[], bool],
    *,
    guide: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> Geometry:
    """Bridge short noisy spans without deleting their knots or changing IDs.

    Each bridge is judged separately in the actual artwork. A real tip or
    notch need not be marked: removing it must improve the reference match.
    Pins, protected features, open ends and the total movement cap still hold.
    """
    if displacement <= 0:
        return geometry
    original = {n.id: n for s in baseline.subpaths for n in s.nodes}
    spans = []
    for si, subpath in enumerate(geometry.subpaths):
        nodes = subpath.nodes
        endpoints = frame.pixels(tuple(v for n in nodes for v in n.endpoint))
        walked = np.r_[0, np.cumsum(np.linalg.norm(np.diff(endpoints, axis=0), axis=1))]
        reference_samples = None
        if guide is not None and len(nodes) > 1:
            samples, tangents, arc = _sample(nodes, frame)
            reference_samples = samples, guide(samples, tangents), arc
        last_start = -float("inf")
        for start in range(len(nodes) - 1):
            if stopped():
                return geometry
            if walked[start] - last_start < 0.75:
                continue
            last_start = walked[start]
            # Work in physical distances, so adding detail cannot shrink a
            # cleanup window to a few redundant subpixel knots.
            ends = {start + 1}
            for distance in (2, 4, 8, 12, 16, 24, 32):
                end = (
                    int(np.searchsorted(walked, walked[start] + distance, side="right"))
                    - 1
                )
                if end > start:
                    ends.add(end)
            for end in sorted(ends):
                window = nodes[start : end + 1]
                if any(n.id in held or n.pinned or n.feature for n in window):
                    break
                pixels = frame.pixels(
                    nodes[start].endpoint
                    + tuple(v for n in window[1:] for v in n.values)
                )
                a = frame.pixels(nodes[start].endpoint)[0]
                b = frame.pixels(nodes[end].endpoint)[0]
                chord = b - a
                length = np.linalg.norm(chord)
                if length < 0.1 or length > 32:
                    continue
                # Include controls: a single cubic may hide a tiny inward hook.
                relative = pixels - a
                along = relative @ chord / length**2
                nearest = a + along.clip(0, 1)[:, None] * chord
                deviation = float(np.linalg.norm(pixels - nearest, axis=1).max())
                if deviation > 0.15:
                    hull = float(np.linalg.norm(np.diff(pixels, axis=0), axis=1).sum())
                    priority = deviation * max(0.1, (hull - length) / max(length, 0.5))
                    if reference_samples is not None:
                        # Prefer spans that a smooth reference fit can repair.
                        # A large, genuine bend must not consume the budget
                        # simply because it has the largest control hull.
                        samples, targets, arc = reference_samples
                        section = slice(4 * start, 4 * end + 1)
                        p, target = samples[section], targets[section]
                        walked_section = arc[section] - arc[4 * start]
                        _, fitted = _fitted(p, target, walked_section)
                        gain = (
                            np.square(p - target).sum()
                            - np.square(fitted - target).sum()
                        )
                        priority = max(0, float(gain))
                    spans.append((-priority, si, start, end))

    for _, si, start, end in sorted(spans)[:128]:
        if stopped():
            break
        subpath = geometry.subpaths[si]
        nodes = list(subpath.nodes)
        a, b = np.array(nodes[start].endpoint), np.array(nodes[end].endpoint)
        endpoints = np.array([n.endpoint for n in nodes[start : end + 1]])
        distances = np.linalg.norm(np.diff(endpoints, axis=0), axis=1)
        total = distances.sum()
        if total <= 1e-9:
            continue
        fractions = np.r_[0, np.cumsum(distances) / total]
        chord = b - a
        length = np.linalg.norm(chord)
        if length < 1e-9:
            continue
        incoming = (
            a - np.array(nodes[start].values[2:4])
            if nodes[start].command == "C"
            else chord
        )
        outgoing = (
            np.array(nodes[end + 1].values[:2]) - b
            if end + 1 < len(nodes) and nodes[end + 1].command == "C"
            else chord
        )

        def arm(direction, chord=chord, length=length):
            norm = np.linalg.norm(direction)
            return (
                direction * length / (3 * norm)
                if norm > 1e-9 and np.dot(direction, chord) > 0
                else chord / 3
            )

        # Fit the reference first; the outside tangents and straight bridge
        # remain alternatives. Subdivide at every retained knot.
        models = [
            (a + arm(incoming), b - arm(outgoing)),
            (a + chord / 3, b - chord / 3),
        ]
        if guide is not None:
            samples, tangents, arc = _sample(nodes[start : end + 1], frame)
            target = guide(samples, tangents)
            fitted, _ = _fitted(samples, target, arc)
            fractions = arc[::4] / max(arc[-1], 1e-9)
            models.insert(0, tuple(fitted))
        for c1, c2 in models:

            def at(t, a=a, b=b, c1=c1, c2=c2):
                u = 1 - t
                point = u**3 * a + 3 * u * u * t * c1 + 3 * u * t * t * c2 + t**3 * b
                tangent = (
                    3 * u * u * (c1 - a) + 6 * u * t * (c2 - c1) + 3 * t * t * (b - c2)
                )
                return point, tangent

            changed = False
            for j in range(start + 1, end + 1):
                node = subpath.nodes[j]
                low, high = fractions[j - start - 1 : j - start + 1]
                p0, t0 = at(low)
                p1, t1 = at(high)
                points = (
                    np.array(
                        [p0 + t0 * (high - low) / 3, p1 - t1 * (high - low) / 3, p1]
                    )
                    if node.command == "C"
                    else p1[None]
                )
                previous = np.asarray(node.values).reshape(-1, 2)
                origin = np.asarray(original[node.id].values).reshape(-1, 2)
                # Clipping separate arms would recreate the jagged corner.
                if np.linalg.norm(points - origin, axis=1).max() > displacement + 1e-9:
                    break
                changed |= bool(np.linalg.norm(points - previous, axis=1).max() > 1e-6)
                nodes[j] = replace(node, values=tuple(points.ravel()))
            else:
                if changed:
                    subpaths = list(geometry.subpaths)
                    subpaths[si] = replace(subpath, nodes=tuple(nodes))
                    candidate = replace(geometry, subpaths=tuple(subpaths))
                    if accept(candidate):
                        geometry = candidate
    return geometry
