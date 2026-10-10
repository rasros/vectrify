"""Reference-judged local bridges for traced dents and handle wiggles."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np

from vectrify.document import Geometry
from vectrify.refine.snap import _Frame


def cleaned(
    geometry: Geometry,
    baseline: Geometry,
    frame: _Frame,
    displacement: float,
    held: frozenset[str],
    accept: Callable[[Geometry], bool],
    stopped: Callable[[], bool],
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

        # Try a curve with the outside tangents first, then a straight bridge.
        # Subdivide it at every retained knot instead of deleting those knots.
        for c1, c2 in (
            (a + arm(incoming), b - arm(outgoing)),
            (a + chord / 3, b - chord / 3),
        ):

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
