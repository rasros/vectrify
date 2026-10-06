"""Bounded local region proposals with finite feature protection."""

from __future__ import annotations

import heapq
import time

import numpy as np

from vectrify.refine.cel_plan.model import Graph, Options, Work
from vectrify.refine.colour_regions import simplified_indices


def merged_labels(graph: Graph, options: Options, work: Work):
    started = time.monotonic()
    regions = graph.regions
    count = len(regions)
    areas = np.array([max(1, r.area) for r in regions], dtype=float)
    colors = np.array([r.paint for r in regions])
    sums = colors * areas[:, None]
    textured = np.array([r.texture for r in regions]) * areas
    features = np.array([r.feature for r in regions])
    parent = np.arange(count)
    versions = np.zeros(count, dtype=int)
    edges: list[dict[int, tuple[float, float, float]]] = [{} for _ in regions]
    for boundary in graph.boundaries:
        a, b = boundary.left, boundary.right
        if min(a, b) < 0 or a in graph.hidden or b in graph.hidden:
            continue
        length = float(np.linalg.norm(np.diff(boundary.points, axis=0), axis=1).sum())
        old_length, support, nodes = edges[a].get(b, (0, 0, 0))
        fitted_nodes = len(simplified_indices(boundary.points, 2))
        value = (
            old_length + length,
            support + length * boundary.line_support,
            nodes + fitted_nodes,
        )
        edges[a][b] = edges[b][a] = value
    total = max(1, sum(r.area for r in regions))
    price = options.detail_cost / 2000
    decisions = []

    def proposal(a, b):
        length, lined, nodes = edges[a][b]
        difference = sums[a] / areas[a] - sums[b] / areas[b]
        ward = areas[a] * areas[b] / (areas[a] + areas[b])
        color_error = ward * float(difference @ difference) / (total * 255**2)
        # A strong, stable small feature pays a finite but substantial removal
        # cost. It still participates in search instead of bypassing the budget.
        small = a if areas[a] < areas[b] else b
        feature = features[small] * (1 - textured[small] / areas[small])
        protection = options.protection * feature * np.linalg.norm(difference) / 255
        protection *= min(1, float(np.linalg.norm(difference)) / 24)
        protection *= min(areas[small] / total, 0.01) * 8
        error = color_error * (1 + 2 * lined / max(length, 1)) + protection
        saved = 2 + nodes
        return error - price * saved, error, saved

    heap = []
    for a in range(count):
        for b in edges[a]:
            if a < b:
                change, _, _ = proposal(a, b)
                heap.append((change, a, b, 0, 0))
    heapq.heapify(heap)
    while heap and not work.interrupted:
        change, a, b, va, vb = heapq.heappop(heap)
        if versions[a] != va or versions[b] != vb or b not in edges[a]:
            continue
        if change >= 0:
            break
        _, error, saved = proposal(a, b)
        decisions.append(
            {
                "operator": "merge",
                "regions": [int(a), int(b)],
                "visual_cost": error,
                "complexity_saved": saved,
            }
        )
        if len(edges[a]) < len(edges[b]):
            a, b = b, a
        parent[b] = a
        areas[a] += areas[b]
        sums[a] += sums[b]
        textured[a] += textured[b]
        features[a] = max(features[a], features[b])
        del edges[a][b]
        for c, (length, support, nodes) in edges[b].items():
            if c == a:
                continue
            del edges[c][b]
            old_length, old_support, old_nodes = edges[a].get(c, (0, 0, 0))
            value = (old_length + length, old_support + support, old_nodes + nodes)
            edges[a][c] = edges[c][a] = value
        edges[b] = {}
        versions[a] += 1
        versions[b] += 1
        for c in edges[a]:
            change, _, _ = proposal(a, c)
            heapq.heappush(heap, (change, a, c, versions[a], versions[c]))
    for index in range(count):
        root = index
        while parent[root] != root:
            root = int(parent[root])
        parent[index] = root
    _, renumber = np.unique(parent, return_inverse=True)
    work.timings["planning"] = time.monotonic() - started
    return renumber[graph.labels].astype(np.int32), tuple(decisions)
