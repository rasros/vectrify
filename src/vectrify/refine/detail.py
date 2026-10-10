"""Supply local curve controls by subdividing the existing outline exactly."""

from __future__ import annotations

import heapq
from dataclasses import replace

import numpy as np

from vectrify.document import Geometry, PathNode
from vectrify.document.model import new_id
from vectrify.refine.snap import _Frame, _halves


def densified(
    geometry: Geometry,
    frame: _Frame,
    held: frozenset[str] = frozenset(),
    spacing: float = 4.0,
) -> Geometry:
    """Split the longest spans first, keeping every original endpoint and ID.

    Lengths are measured in reference pixels, so zoom and object transforms
    do not change the density. Held spans and implicit closing lines are left
    alone. Each call adds at most one original point count or sixteen points,
    with a total ceiling of 512; a larger input is never truncated.
    """
    count = sum(len(s.nodes) for s in geometry.subpaths)
    budget = max(0, min(512 - count, max(count, 16)))
    pieces, queue = {}, []
    serial = 0

    def enqueue(key, t, controls):
        nonlocal serial
        pieces[key][t] = controls
        pixels = frame.pixels(tuple(controls.ravel()))
        length = float(np.linalg.norm(np.diff(pixels, axis=0), axis=1).sum())
        if length > spacing:
            heapq.heappush(queue, (-length, serial, key, t))
            serial += 1

    for si, subpath in enumerate(geometry.subpaths):
        for ni, node in enumerate(subpath.nodes[1:], 1):
            previous = subpath.nodes[ni - 1]
            if (
                node.id in held
                or previous.id in held
                or node.pinned
                or previous.pinned
                or node.feature is not None
                or previous.feature is not None
            ):
                continue
            start, end = np.array(previous.endpoint), np.array(node.endpoint)
            control = (
                np.array(
                    [
                        start,
                        start + (end - start) / 3,
                        start + 2 * (end - start) / 3,
                        end,
                    ]
                )
                if node.command == "L"
                else np.vstack((start, np.asarray(node.values).reshape(-1, 2)))
            )
            key = (si, ni)
            pieces[key] = {}
            enqueue(key, (0.0, 1.0), control)

    while queue and budget:
        _, _, key, interval = heapq.heappop(queue)
        controls = pieces[key].pop(interval)
        first, second = _halves(controls, 0.5)
        left, right = interval
        middle = (left + right) / 2
        enqueue(key, (left, middle), np.vstack((controls[0], first)))
        enqueue(key, (middle, right), np.vstack((first[-1], second)))
        budget -= 1

    subpaths = []
    for si, subpath in enumerate(geometry.subpaths):
        nodes = [subpath.nodes[0]]
        for ni, node in enumerate(subpath.nodes[1:], 1):
            spans = pieces.get((si, ni), {})
            if len(spans) <= 1:
                nodes.append(node)
                continue
            values = [tuple(c[1:].ravel()) for _, c in sorted(spans.items())]
            nodes.extend(PathNode(new_id("node"), "C", v) for v in values[:-1])
            nodes.append(replace(node, command="C", values=values[-1]))
        subpaths.append(replace(subpath, nodes=tuple(nodes)))
    return replace(geometry, subpaths=tuple(subpaths))
