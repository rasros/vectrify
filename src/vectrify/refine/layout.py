"""Keep opaque regions inside a nearly symmetric, open enclosing outline.

An open outline with two matching runs from its base to its tip describes an
enclosed silhouette even though its base is not stroked. Project paired curve
controls about the tip/base axis, intersect each fill with that silhouette, and
put uncovered interior behind the existing fills. Paint and stacking stay put.
This applies only to a complete selection in one frame, with no pinned nodes,
locks, translucent paint, clips, or shared geometry. Other artwork is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pathops
from cairosvg.colors import color

from vectrify.document import Document, Geometry
from vectrify.document.join import curve_path, path_geometry, path_style
from vectrify.document.transforms import object_matrix


def axis_for(document: Document, oid: str):
    nodes = document.geometry_for(oid).subpaths[0].nodes
    base = (np.array(nodes[0].values[-2:]) + nodes[-1].values[-2:]) / 2
    tip = np.array(nodes[len(nodes) // 2].values[-2:])
    a, b, c, d, e, f = object_matrix(document, oid)
    return tuple((a * x + c * y + e, b * x + d * y + f) for x, y in (base, tip))


def reflection(axis):
    """Shapely affine coefficients for reflection about an arbitrary axis."""
    start, end = np.asarray(axis, dtype=float)
    direction = end - start
    direction /= np.linalg.norm(direction)
    linear = 2 * np.outer(direction, direction) - np.eye(2)
    offset = start - linear @ start
    return (*linear[0], *linear[1], *offset)


def _symmetric(geometry: Geometry) -> Geometry | None:
    if len(geometry.subpaths) != 1:
        return None
    sub = geometry.subpaths[0]
    nodes = sub.nodes
    # Two runs with matching commands in reverse. A closing base is implicit.
    if sub.closed or len(nodes) < 3 or len(nodes) % 2 != 1:
        return None
    commands = [n.command for n in nodes[1:]]
    if commands != commands[::-1]:
        return None
    points = np.array(
        [p for n in nodes for p in zip(n.values[::2], n.values[1::2], strict=True)]
    )
    base = (points[0] + points[-1]) / 2
    tip = np.array(nodes[len(nodes) // 2].values[-2:])
    if np.linalg.norm(tip - base) < 1e-6:
        return None
    a, b, c, d, e, f = reflection((base, tip))
    mirrored = points[::-1] @ np.array([[a, b], [c, d]]).T + (e, f)
    # Only infer this property for outlines already close to symmetric.
    direction = (tip - base) / np.linalg.norm(tip - base)
    normal = np.array([-direction[1], direction[0]])
    width = np.ptp(points @ normal)
    if width < 1e-6 or np.linalg.norm(points - mirrored, axis=1).max() > width * 0.15:
        return None
    projected = (points + mirrored) / 2
    at = 0
    fitted = []
    for node in nodes:
        count = len(node.values) // 2
        fitted.append(replace(node, values=tuple(projected[at : at + count].ravel())))
        at += count
    return replace(geometry, subpaths=(replace(sub, nodes=tuple(fitted)),))


@dataclass(frozen=True)
class Layout:
    outline: str
    fills: tuple[str, ...]
    # Seed topology for the symmetric outline. Its paired controls can follow
    # the reference together; an incompatible simplification keeps this seed.
    silhouette: Geometry

    def valid(self, document: Document) -> bool:
        """Reject even a Boolean result that leaks through numerical clipping.

        PathOps stores float32 controls. Near coincident long curves, an
        intersection can occasionally retain a thin outside lens. Check each
        final fill against the actual final outline before accepting a step.
        """
        boundary = curve_path(document.geometry_for(self.outline))
        covered = pathops.Path()
        for oid in self.fills:
            path = curve_path(
                document.geometry_for(oid),
                path_style(document, document.element(oid))["fill-rule"],
            )
            if abs(pathops.op(path, boundary, pathops.PathOp.DIFFERENCE).area) > 0.05:
                return False
            covered = pathops.op(covered, path, pathops.PathOp.UNION)
        return (
            abs(pathops.op(boundary, covered, pathops.PathOp.DIFFERENCE).area) <= 0.05
        )

    def project(self, document: Document) -> Document:
        silhouette = _symmetric(document.geometry_for(self.outline)) or self.silhouette
        document = document.replace_geometry(silhouette)
        boundary = curve_path(silhouette)
        clipped = {}
        for oid in self.fills:
            geometry = document.geometry_for(oid)
            rule = path_style(document, document.element(oid))["fill-rule"]
            clipped[oid] = pathops.op(
                curve_path(geometry, rule), boundary, pathops.PathOp.INTERSECTION
            )
        # A continuous opaque underpaint also closes antialiased seams. Merely
        # making neighbouring contours meet leaves partially transparent pixels
        # in Cairo's source-over composition. Use the backmost existing fill;
        # all later paint remains in its original order above it.
        clipped[self.fills[0]] = boundary
        for oid, path in clipped.items():
            original = document.geometry_for(oid)
            # An already-valid path keeps its point IDs, pins and topology.
            source = curve_path(
                original, path_style(document, document.element(oid))["fill-rule"]
            )
            delta = pathops.op(source, path, pathops.PathOp.XOR)
            if abs(delta.area) < 1e-6:
                continue
            geometry = replace(path_geometry(path), id=original.id)
            document = document.replace_geometry(geometry)
        return document


def infer(document: Document, oids, held=frozenset()) -> tuple[Layout, ...]:
    """Conservative structural priors for complete groups selected together."""
    selected = set(oids)
    found = []
    for group in document.elements():
        if group.tag != "g":
            continue
        if any(e.tag not in {"path", "defs"} for e in group.children):
            continue
        paths = [e for e in group.children if e.tag == "path"]
        if len(paths) < 3 or any(e.id not in selected for e in paths):
            continue
        frames = {object_matrix(document, e.id) for e in paths}
        if len(frames) != 1:
            continue
        a, b, c, d, _e, _f = next(iter(frames))
        linear = np.array([[a, c], [b, d]])
        metric = linear.T @ linear
        # Local reflection is a document-space reflection only for a
        # similarity transform (rotation/reflection and uniform scale).
        if not np.allclose(metric, np.eye(2) * metric[0, 0]) or metric[0, 0] <= 0:
            continue
        if any(
            a.locks
            or a.get("clip-path")
            or a.get("mask")
            or float(a.get("opacity", "1") or "1") != 1
            for e in paths
            for a in document.ancestry(e.id)
        ):
            continue
        if any(
            document.geometry_users(document.geometry_for(e.id).id) != {e.id}
            or any(
                n.pinned or n.id in held
                for s in document.geometry_for(e.id).subpaths
                for n in s.nodes
            )
            for e in paths
        ):
            continue
        styles = {e.id: path_style(document, e) for e in paths}
        outlines = [
            e
            for e in paths
            if styles[e.id]["fill"] == "none" and styles[e.id]["stroke"] != "none"
        ]
        fills = [e for e in paths if styles[e.id]["fill"] != "none"]
        if len(outlines) != 1 or len(fills) != len(paths) - 1:
            continue
        if any(
            styles[e.id]["stroke"] != "none"
            or "url(" in styles[e.id]["fill"]
            or float(styles[e.id]["fill-opacity"]) != 1
            or color(styles[e.id]["fill"])[3] != 1
            for e in fills
        ):
            continue
        silhouette = _symmetric(document.geometry_for(outlines[0].id))
        if silhouette is None:
            continue
        boundary = curve_path(silhouette)
        covered = pathops.Path()
        for e in fills:
            covered = pathops.op(
                covered,
                curve_path(document.geometry_for(e.id), styles[e.id]["fill-rule"]),
                pathops.PathOp.UNION,
            )
        common = pathops.op(covered, boundary, pathops.PathOp.INTERSECTION)
        if abs(common.area) < 0.9 * max(abs(covered.area), abs(boundary.area)):
            continue
        found.append(Layout(outlines[0].id, tuple(e.id for e in fills), silhouette))
    return tuple(found)
