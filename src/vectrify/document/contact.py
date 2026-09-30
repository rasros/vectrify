"""Match nearby boundary spans, preserving curves and explicit edge adjacency."""

import math
from dataclasses import replace

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial import KDTree
from shapely import STRtree
from shapely.geometry import LineString, Point

from vectrify.document.hit_test import IDENTITY, multiply, transform
from vectrify.document.model import DocumentError, EdgeRef, SharedBoundary, new_id
from vectrify.document.topology import edge, inverse_matrix, mapped_point, split_edges


def edges(document, gid, matrix=IDENTITY):
    return [
        edge(document, EdgeRef(gid, n.id, matrix=matrix))
        for s in document.geometry(gid).subpaths
        for i, n in enumerate(s.nodes)
        if i or s.closed
    ]


def sample(points, ts):
    p = np.asarray(points)
    t = np.asarray(ts)[:, None]
    if len(p) == 2:
        return p[0] * (1 - t) + p[1] * t
    return (
        p[0] * (1 - t) ** 3
        + 3 * p[1] * (1 - t) ** 2 * t
        + 3 * p[2] * (1 - t) * t * t
        + p[3] * t**3
    )


def projection(point, points):
    """Find a parameter near the nearest sampled span, including endpoints."""
    p = np.asarray(point)
    ts = np.linspace(0, 1, 33 if len(points) == 4 else 2)
    samples = sample(points, ts)
    starts, directions = samples[:-1], np.diff(samples, axis=0)
    fractions = np.clip(
        np.sum((p - starts) * directions, axis=1)
        / np.maximum(np.sum(directions * directions, axis=1), 1e-30),
        0,
        1,
    )
    distances = np.sum((starts + directions * fractions[:, None] - p) ** 2, axis=1)
    i = int(np.argmin(distances))
    t = ts[i] + fractions[i] * (ts[i + 1] - ts[i])
    if len(points) == 4:
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = points
        px, py = point

        # Plain floats: the search evaluates the curve dozens of times.
        def squared(u):
            v = 1 - u
            a, b, c, d = v * v * v, 3 * v * v * u, 3 * v * u * u, u * u * u
            x = a * x0 + b * x1 + c * x2 + d * x3 - px
            y = a * y0 + b * y1 + c * y2 + d * y3 - py
            return x * x + y * y

        fit = minimize_scalar(
            squared,
            bounds=(ts[i], ts[i + 1]),
            method="bounded",
            options={"xatol": 1e-12},
        )
        t = min((0.0, 1.0, float(fit.x)), key=squared)
    return float(t), float(np.linalg.norm(sample(points, [t])[0] - p))


def box(points, margin=0.0):
    """Bounds of *points*, widened by *margin*: (left, top, right, bottom)."""
    p = np.asarray(points).reshape(-1, 2)
    return (*(p.min(axis=0) - margin), *(p.max(axis=0) + margin))


def inside(point, bounds):
    return bounds[0] <= point[0] <= bounds[2] and bounds[1] <= point[1] <= bounds[3]


def meets(first, second):
    return (
        first[0] <= second[2]
        and second[0] <= first[2]
        and first[1] <= second[3]
        and second[1] <= first[3]
    )


def split_at_contacts(document, first, second, tolerance, matrices, linked=()):
    originals = {gid: edges(document, gid, matrices[gid]) for gid in (first, second)}
    bounds = {
        gid: box([p for e in found for p in e.points], tolerance)
        for gid, found in originals.items()
    }
    cuts = {}
    for source, target in ((first, second), (second, first)):
        # Only the parts within reach of the other path can touch it. Linked
        # edges are never split: that would split their partners too, and
        # those may belong to paths outside this pair.
        targets = [
            e
            for e in originals[target]
            if (e.ref.geometry_id, e.ref.node_id) not in linked
            and meets(box(e.points), bounds[source])
        ]
        points = {
            p
            for e in originals[source]
            for p in (e.points[0], e.points[-1])
            if inside(p, bounds[target])
        }
        if not targets or not points:
            continue
        lines = [LineString(sample(e.points, np.linspace(0, 1, 33))) for e in targets]
        tree = STRtree(lines)
        for point in points:
            for index in tree.query(Point(point).buffer(tolerance)):
                e = targets[int(index)]
                t, distance = projection(point, e.points)
                if distance <= tolerance and 1e-6 < t < 1 - 1e-6:
                    cuts.setdefault(e.ref, set()).add(t)
    # No cut edge is linked, so splitting can skip the search for partners.
    boundaries = document.boundaries
    document = replace(document, boundaries=())
    # All projections refer to original curves; split the remaining suffix at
    # an adjusted parameter so subdivision introduces no flattening error.
    for ref, parameters in cuts.items():
        previous = 0.0
        for t in sorted(parameters):
            if t - previous < 1e-5:
                continue
            document, _ = split_edges(document, ref, (t - previous) / (1 - previous))
            previous = t
    return replace(document, boundaries=boundaries)


def regions(document, object_ids):
    """The selected paths' geometry IDs and root frames, back to front."""
    if len(object_ids) < 2:
        raise DocumentError("Select two or more paths to share boundaries")
    frames = {}
    for element in (e for e in document.elements() if e.id in object_ids):
        if element.tag != "path":
            raise DocumentError(
                "Select editable paths; detach instances or select group children."
            )
        geometry = document.geometry_for(element.id)
        if document.geometry_users(geometry.id) != frozenset({element.id}):
            raise DocumentError("Detach shared geometry before matching boundaries")
        if any(not s.closed for s in geometry.subpaths):
            raise DocumentError("Shared regions must have closed contours")
        matrix = IDENTITY
        for ancestor in document.ancestry(element.id):
            if ancestor.get("clip-path"):
                raise DocumentError(
                    "Clipped paths are not supported by boundary matching yet"
                )
            matrix = multiply(matrix, transform(ancestor.get("transform")))
        inverse_matrix(matrix)
        frames[geometry.id] = matrix
    return frames


def touching(document, frames, tolerance):
    """(front, rear) pairs whose root-space bounds come within *tolerance*,
    frontmost fronts first so the front contour is always the reference."""
    ids = list(frames)
    # Control points bound their curves, so their extremes bound each path.
    boxes = np.asarray(
        [
            box([p for e in edges(document, gid, frames[gid]) for p in e.points])
            for gid in ids
        ]
    ).reshape(-1, 4)
    low, high = boxes[:, :2], boxes[:, 2:] + tolerance
    near = np.all(low[:, None] <= high[None], axis=2) & np.all(
        low[None] <= high[:, None], axis=2
    )
    return [
        (ids[front], ids[rear])
        for front in reversed(range(len(ids)))
        for rear in reversed(range(front))
        if near[front, rear]
    ]


def match_pair(candidate, front, rear, tolerance, frames, linked, locked):
    """Link the touching, still unlinked edges of one pair, snapping *rear*.

    A span is skipped when snapping it would move a slot of an edge linked
    before, so every existing boundary stays coincident.
    """
    candidate = split_at_contacts(candidate, front, rear, tolerance, frames, linked)
    sources = [
        e
        for e in edges(candidate, front, frames[front])
        if (front, e.ref.node_id) not in linked
    ]
    targets = [
        e
        for e in edges(candidate, rear, frames[rear])
        if (rear, e.ref.node_id) not in linked
    ]
    if len(sources) + len(targets) > 30000:
        raise DocumentError(
            "Too many contact candidates; simplify the paths or reduce contact distance"
        )
    if not sources or not targets:
        return candidate, []
    tree = KDTree([e.points[0] for e in targets] + [e.points[-1] for e in targets])
    proposals = []
    for s in sources:
        if math.dist(s.points[0], s.points[-1]) < 1e-8:
            continue
        for i in tree.query_ball_point(s.points[0], tolerance):
            reverse = i >= len(targets)
            t = targets[i % len(targets)]
            points = t.points[::-1] if reverse else t.points
            if (
                len(points) != len(s.points)
                or math.dist(s.points[-1], points[-1]) > tolerance
            ):
                continue
            first_direction = np.asarray(s.points[-1]) - s.points[0]
            second_direction = np.asarray(points[-1]) - points[0]
            lengths = np.linalg.norm(first_direction) * np.linalg.norm(second_direction)
            if (
                not lengths
                or np.dot(first_direction, second_direction) < 0.95 * lengths
            ):
                continue
            # Matching endpoints alone is insufficient for curved edges.
            distances = np.linalg.norm(
                sample(s.points, np.linspace(0, 1, 17))
                - sample(points, np.linspace(0, 1, 17)),
                axis=1,
            )
            if float(max(distances)) <= tolerance:
                proposals.append((float(sum(distances)), s, t, reverse))
    current = {
        (rear, n.id, i): v
        for s in candidate.geometry(rear).subpaths
        for n in s.nodes
        for i, v in enumerate(n.values)
    }
    used_source, used_target, updates, boundaries = set(), set(), {}, []
    for _, s, t, reverse in sorted(proposals, key=lambda p: p[0]):
        if s.ref in used_source or t.ref in used_target:
            continue
        slots = edge(candidate, replace(t.ref, reversed=reverse)).slots
        values = tuple(
            v for p in s.points for v in mapped_point(p, inverse_matrix(t.ref.matrix))
        )
        if any(
            (slot in updates and abs(updates[slot] - v) > 1e-8)
            or (slot in locked and abs(current[slot] - v) > 1e-8)
            for slot, v in zip(slots, values, strict=True)
        ):
            continue
        updates.update(zip(slots, values, strict=True))
        boundaries.append(
            SharedBoundary(
                new_id("boundary"), (s.ref, replace(t.ref, reversed=reverse))
            )
        )
        used_source.add(s.ref)
        used_target.add(t.ref)
    if not boundaries:
        return candidate, []
    # Only the rear region snaps; the front contour remains the reference.
    geometry = candidate.geometry(rear)
    geometry = replace(
        geometry,
        subpaths=tuple(
            replace(
                s,
                nodes=tuple(
                    replace(
                        n,
                        values=tuple(
                            updates.get((rear, n.id, i), v)
                            for i, v in enumerate(n.values)
                        ),
                    )
                    for n in s.nodes
                ),
            )
            for s in geometry.subpaths
        ),
    )
    candidate = candidate.replace_geometry(geometry)
    candidate = replace(candidate, boundaries=(*candidate.boundaries, *boundaries))
    return candidate, boundaries


def match_boundaries(document, object_ids, tolerance):
    """Link the touching edges of every pair of the selected paths.

    Edges already in a boundary are kept as they are and never matched again.
    """
    if not math.isfinite(tolerance) or not 0 < tolerance <= 20:
        raise DocumentError(
            "Contact distance must be greater than zero and at most 20 SVG units"
        )
    frames = regions(document, object_ids)
    if len(frames) < 2:
        raise DocumentError("Select two or more paths to share boundaries")
    candidate, count = document, 0
    linked = {
        (m.geometry_id, m.node_id) for b in document.boundaries for m in b.members
    }
    locked = {
        slot
        for b in document.boundaries
        for m in b.members
        for slot in edge(document, m).slots
    }
    for front, rear in touching(document, frames, tolerance):
        candidate, boundaries = match_pair(
            candidate, front, rear, tolerance, frames, linked, locked
        )
        for boundary in boundaries:
            for member in boundary.members:
                linked.add((member.geometry_id, member.node_id))
                locked.update(edge(candidate, member).slots)
        count += len(boundaries)
    if not count:
        raise DocumentError(
            "No matching boundary spans found. Try a slightly larger contact distance."
        )
    candidate.validate()
    return candidate, count
