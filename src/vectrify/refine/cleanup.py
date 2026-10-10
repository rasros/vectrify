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
        for start in range(len(nodes) - 1):
            if stopped():
                return geometry
            for end in range(start + 1, min(len(nodes), start + 7)):
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
                if length < 0.1 or length > 16:
                    continue
                # Include controls: a single cubic may hide a tiny inward hook.
                relative = pixels - a
                along = relative @ chord / length**2
                nearest = a + along.clip(0, 1)[:, None] * chord
                deviation = float(np.linalg.norm(pixels - nearest, axis=1).max())
                if deviation > 0.15:
                    spans.append((-deviation, si, start, end))

    for _, si, start, end in sorted(spans):
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
        changed = False
        for j in range(start + 1, end + 1):
            node = nodes[j]
            low, high = fractions[j - start - 1 : j - start + 1]
            t = (
                np.array([low + (high - low) / 3, high - (high - low) / 3, high])
                if node.command == "C"
                else np.array([high])
            )
            points = a + t[:, None] * (b - a)
            previous = np.asarray(node.values).reshape(-1, 2)
            origin = np.asarray(original[node.id].values).reshape(-1, 2)
            # Reject a bridge needing a larger move instead of clipping each
            # arm independently and recreating a jagged corner.
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
