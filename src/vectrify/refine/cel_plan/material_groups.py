"""Connected material hypotheses with costs updated after every union.

The increase in constant-paint squared error guides discovery, not acceptance.
Export still fits complete source paint and the native evaluator judges the
drawing. Small disconnected supports keep their original owners and geometry.
"""

from __future__ import annotations

import heapq

import numpy as np

MAX_MATERIALS = 4096
MAX_EDGES = 16_384
REBUILD_QUEUE_AT = 65_536


def ink_paint_links(colors, kinds, work, *, tolerance=8):
    """Bounded co-paint hypotheses for disconnected source ink.

    Each bucket spans less than ``tolerance`` in each RGB channel. Links permit
    a compound paint model; they add no pixels, contours or physical adjacency.
    Complete source paint fitting and native admission still judge the result.
    """
    colors, kinds = np.asarray(colors), np.asarray(kinds)
    if (
        len(colors) > MAX_MATERIALS
        or colors.shape != (len(kinds), 3)
        or not np.isfinite(colors).all()
        or not np.isfinite(tolerance)
        or tolerance <= 0
    ):
        raise ValueError("Ink co-paint requires bounded finite source colors")
    buckets, links = {}, []
    for i in np.flatnonzero(kinds):
        if work.interrupted:
            return None
        key = (int(kinds[i]), *np.floor(colors[i] / tolerance).astype(int))
        first = buckets.setdefault(key, int(i))
        if first != i:
            links.append((first, int(i)))
    return links


def grouped(
    colors,
    sizes,
    edges,
    budget,
    work,
    *,
    minimum_component_area=256,
    kinds=None,
    paint_edges=(),
):
    """Return complete owner groups and eligibility, or discard interrupted work.

    The budget applies to materials in substantial connected supports. Tiny
    disconnected supports remain independently owned, outside that budget.
    They still count toward actual exported cost and are never silently lost.
    """
    count = len(sizes)
    if not 0 < count <= MAX_MATERIALS or len(edges) + len(paint_edges) > MAX_EDGES:
        raise ValueError("Material hierarchy exceeds discovery bounds")
    if budget < 1 or minimum_component_area < 1:
        raise ValueError("Material hierarchy requires positive bounds")
    parents = np.arange(count, dtype=np.int32)
    kinds = np.zeros(count, np.uint8) if kinds is None else np.asarray(kinds)
    if kinds.shape != (count,) or kinds.dtype.kind not in "biu":
        raise ValueError("Material hierarchy requires one discrete role per owner")
    means = np.asarray(colors, np.float64).copy()
    areas = np.asarray(sizes, np.float64).copy()
    if (
        means.shape != (count, 3)
        or not np.isfinite(means).all()
        or not np.isfinite(areas).all()
        or (areas <= 0).any()
    ):
        raise ValueError("Material hierarchy requires finite nonempty source paint")
    neighbors = [set() for _ in range(count)]
    for a, b in edges:
        if work.interrupted:
            return None
        if not 0 <= a < count or not 0 <= b < count or a == b:
            raise ValueError("Invalid material adjacency")
        neighbors[a].add(b)
        neighbors[b].add(a)
    eligible = np.zeros(count, bool)
    unseen = set(range(count))
    components = retained = 0
    while unseen:
        todo, members = [min(unseen)], []
        while todo:
            if work.interrupted:
                return None
            i = todo.pop()
            if i not in unseen:
                continue
            unseen.remove(i)
            members.append(i)
            todo.extend(sorted(neighbors[i].intersection(unseen)))
        if float(areas[members].sum()) >= minimum_component_area:
            eligible[members] = True
            components += 1
        else:
            retained += len(members)
    # Eligibility comes only from actual spatial support. Co-paint can combine
    # disconnected ink contours within that support, never promote tiny marks
    # into a large physical component or bridge their geometry.
    linked_paints = 0
    for a, b in paint_edges:
        if work.interrupted:
            return None
        if (
            not 0 <= a < count
            or not 0 <= b < count
            or a == b
            or not kinds[a]
            or kinds[a] != kinds[b]
        ):
            raise ValueError(
                "Co-paint links require distinct owners of the same ink role"
            )
        if eligible[a] and eligible[b] and b not in neighbors[a]:
            neighbors[a].add(b)
            neighbors[b].add(a)
            linked_paints += 1
    versions = np.zeros(count, np.int32)
    heap = []
    merges = rebuilds = 0

    def push(a, b):
        if kinds[a] != kinds[b]:
            return
        if b < a:
            a, b = b, a
        delta = means[a] - means[b]
        price = areas[a] * areas[b] / (areas[a] + areas[b]) * float(delta @ delta)
        heapq.heappush(heap, (price, a, b, int(versions[a]), int(versions[b])))

    def rebuild():
        heap.clear()
        for a in range(count):
            if work.interrupted:
                return False
            if parents[a] != a or not eligible[a]:
                continue
            for b in sorted(neighbors[a]):
                if b > a:
                    push(a, b)
        return True

    if not rebuild():
        return None
    peak = len(heap)
    remaining = int(eligible.sum())
    while heap and remaining > budget:
        if work.interrupted:
            return None
        _, a, b, va, vb = heapq.heappop(heap)
        if parents[a] != a or parents[b] != b or versions[a] != va or versions[b] != vb:
            continue
        means[a] = (means[a] * areas[a] + means[b] * areas[b]) / (areas[a] + areas[b])
        areas[a] += areas[b]
        parents[b] = a
        versions[a] += 1
        linked = (neighbors[a] | neighbors[b]) - {a, b}
        for i in sorted(linked):
            if work.interrupted:
                return None
            neighbors[i].discard(a)
            neighbors[i].discard(b)
            neighbors[i].add(a)
        neighbors[a], neighbors[b] = linked, set()
        for i in sorted(linked):
            push(a, i)
        remaining -= 1
        merges += 1
        peak = max(peak, len(heap))
        if len(heap) > REBUILD_QUEUE_AT:
            if not rebuild():
                return None
            rebuilds += 1
    for i in range(count):
        if work.interrupted:
            return None
        at = i
        while parents[at] != at:
            at = int(parents[at])
        parents[i] = at
    parents.flags.writeable = eligible.flags.writeable = False
    return (
        parents,
        eligible,
        {
            "merges": merges,
            "substantial_components": components,
            "retained_disconnected_owners": retained,
            "remaining_materials": remaining,
            "budget_unmet": remaining > budget,
            "ink_materials": int(np.count_nonzero(kinds[np.unique(parents[eligible])])),
            "ink_paint_links": linked_paints,
            "queue_peak": peak,
            "queue_rebuilds": rebuilds,
        },
    )
