"""Connected material hypotheses with costs updated after every union.

Flat RGB or complete common-axis RGBA fit error guides discovery, not acceptance.
Export still fits complete source paint and the native evaluator judges the
drawing. Small disconnected supports keep their original owners and geometry.
"""

from __future__ import annotations

import heapq

import numpy as np

from vectrify.refine.cel_plan.materials import model

MAX_MATERIALS = 4096
MAX_EDGES = 16_384
REBUILD_QUEUE_AT = 65_536
MAX_MODEL_EVALUATIONS = 32_768


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
    statistics=None,
    alpha_ranges=None,
    gradients=True,
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
    gram = None if statistics is None else np.asarray(statistics, np.float64).copy()
    ranges = None
    losses = np.zeros(count)
    evaluations = alpha_exclusions = 0
    limited = False
    if gram is not None:
        ranges = np.asarray(alpha_ranges, np.float64).copy()
        if (
            gram.shape != (count, 7, 7)
            or not np.isfinite(gram).all()
            or (gram[:, 0, 0] <= 0).any()
            or ranges.shape != (count, 2)
            or not np.isfinite(ranges).all()
            or (ranges < 0).any()
            or (ranges > 1).any()
            or (ranges[:, 0] > ranges[:, 1]).any()
        ):
            raise ValueError("Material fit requires complete finite RGBA statistics")
        for i, g in enumerate(gram):
            if work.interrupted:
                return None
            losses[i] = model(g, gradients=gradients, gradient_price=0).error
            evaluations += 1
    elif alpha_ranges is not None:
        raise ValueError("Opacity ranges require material fit statistics")
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
        nonlocal evaluations, alpha_exclusions, limited
        if kinds[a] != kinds[b]:
            return
        if b < a:
            a, b = b, a
        fitted = 0.0
        if gram is None:
            delta = means[a] - means[b]
            price = areas[a] * areas[b] / (areas[a] + areas[b]) * float(delta @ delta)
        else:
            assert ranges is not None
            if evaluations >= MAX_MODEL_EVALUATIONS:
                limited = True
                return
            evaluations += 1
            proposed = model(gram[a] + gram[b], gradients=gradients, gradient_price=0)
            low, high = min(ranges[a, 0], ranges[b, 0]), max(ranges[a, 1], ranges[b, 1])
            if high > low * 1.25 + 1e-7 and (
                not proposed.gradient or proposed.alpha_residual > 0.02
            ):
                alpha_exclusions += 1
                return
            fitted = proposed.error
            price = max(0.0, fitted - losses[a] - losses[b])
        heapq.heappush(heap, (price, a, b, int(versions[a]), int(versions[b]), fitted))

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
    while heap and remaining > budget and not limited:
        if work.interrupted:
            return None
        _, a, b, va, vb, fitted = heapq.heappop(heap)
        if parents[a] != a or parents[b] != b or versions[a] != va or versions[b] != vb:
            continue
        means[a] = (means[a] * areas[a] + means[b] * areas[b]) / (areas[a] + areas[b])
        areas[a] += areas[b]
        if gram is not None:
            assert ranges is not None
            gram[a] += gram[b]
            losses[a] = fitted
            ranges[a] = min(ranges[a, 0], ranges[b, 0]), max(ranges[a, 1], ranges[b, 1])
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
            "paint_cost": "source-rgba-fit" if gram is not None else "flat-rgb-ward",
            "model_evaluations": evaluations,
            "model_limit_hit": limited,
            "alpha_exclusions": alpha_exclusions,
        },
    )
