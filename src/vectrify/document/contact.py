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
        fit = minimize_scalar(
            lambda u: float(np.sum((sample(points, [u])[0] - p) ** 2)),
            bounds=(ts[i], ts[i + 1]),
            method="bounded",
            options={"xatol": 1e-12},
        )
        t = min(
            (0.0, 1.0, float(fit.x)),
            key=lambda u: float(np.sum((sample(points, [u])[0] - p) ** 2)),
        )
    return float(t), float(np.linalg.norm(sample(points, [t])[0] - p))


def split_at_contacts(document, first, second, tolerance, matrices):
    originals = {gid: edges(document, gid, matrices[gid]) for gid in (first, second)}
    cuts = {}
    for source, target in ((first, second), (second, first)):
        targets = originals[target]
        lines = [LineString(sample(e.points, np.linspace(0, 1, 33))) for e in targets]
        tree = STRtree(lines)
        points = {p for e in originals[source] for p in (e.points[0], e.points[-1])}
        for point in points:
            for index in tree.query(Point(point).buffer(tolerance)):
                e = targets[int(index)]
                t, distance = projection(point, e.points)
                if distance <= tolerance and 1e-6 < t < 1 - 1e-6:
                    cuts.setdefault(e.ref, set()).add(t)
    # All projections refer to original curves; split the remaining suffix at
    # an adjusted parameter so subdivision introduces no flattening error.
    for ref, parameters in cuts.items():
        previous = 0.0
        for t in sorted(parameters):
            if t - previous < 1e-5:
                continue
            document, _ = split_edges(document, ref, (t - previous) / (1 - previous))
            previous = t
    return document


def match_boundaries(document, object_ids, tolerance):
    if len(object_ids) != 2:
        raise DocumentError("Select exactly two paths to share a boundary")
    if not math.isfinite(tolerance) or not 0 < tolerance <= 20:
        raise DocumentError(
            "Contact distance must be greater than zero and at most 20 SVG units"
        )
    ordered = [e for e in document.elements() if e.id in object_ids]
    matrices = []
    for element in ordered:
        if element.tag != "path":
            raise DocumentError(
                "Select two editable paths; detach instances or select group children."
            )
        geometry = document.geometry_for(element.id)
        if document.geometry_users(geometry.id) != frozenset({element.id}):
            raise DocumentError("Detach shared geometry before matching boundaries")
        if any(not s.closed for s in geometry.subpaths):
            raise DocumentError("Shared regions must have closed contours")
        if any(
            m.geometry_id == geometry.id for b in document.boundaries for m in b.members
        ):
            raise DocumentError(
                "Unlink the existing shared boundaries before rematching these paths."
            )
        matrix = IDENTITY
        for ancestor in document.ancestry(element.id):
            if ancestor.get("clip-path"):
                raise DocumentError(
                    "Clipped paths are not supported by boundary matching yet"
                )
            matrix = multiply(matrix, transform(ancestor.get("transform")))
        matrices.append(matrix)
    for matrix in matrices:
        inverse_matrix(matrix)
    target, source = [document.geometry_for(e.id).id for e in ordered]
    frames = {
        document.geometry_for(e.id).id: matrix
        for e, matrix in zip(ordered, matrices, strict=True)
    }
    candidate = split_at_contacts(document, source, target, tolerance, frames)
    sources = edges(candidate, source, frames[source])
    targets = edges(candidate, target, frames[target])
    if len(sources) + len(targets) > 30000:
        raise DocumentError(
            "Too many contact candidates; simplify the paths or reduce contact distance"
        )
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
    used_source, used_target, updates, boundaries = set(), set(), {}, []
    for _, s, t, reverse in sorted(proposals, key=lambda p: p[0]):
        if s.ref in used_source or t.ref in used_target:
            continue
        slots = edge(candidate, replace(t.ref, reversed=reverse)).slots
        values = tuple(
            v for p in s.points for v in mapped_point(p, inverse_matrix(t.ref.matrix))
        )
        if any(
            slot in updates and abs(updates[slot] - v) > 1e-8
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
        raise DocumentError(
            "No matching boundary spans found. Try a slightly larger contact distance."
        )
    # Only the rear region snaps; the frontmost contour remains the reference.
    geometry = candidate.geometry(target)
    geometry = replace(
        geometry,
        subpaths=tuple(
            replace(
                s,
                nodes=tuple(
                    replace(
                        n,
                        values=tuple(
                            updates.get((target, n.id, i), v)
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
    candidate.validate()
    return candidate, len(boundaries)
