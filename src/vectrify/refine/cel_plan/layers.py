"""Locally supported base continuation beneath compact closed overlays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

from vectrify.refine import cel
from vectrify.refine.cel_plan.geometry import Model, ellipse
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.ink import Ink, measure
from vectrify.refine.cel_plan.model import Evidence, Options, Work
from vectrify.refine.colour_regions import nearest_indices
from vectrify.refine.tracing import _loops


@dataclass(frozen=True)
class Overlay:
    region: int
    model: Model
    ink: Ink | None
    members: tuple[int, ...] = ()

    @property
    def data(self) -> str:
        contour = self.model.contour
        return cel._data(
            contour.nodes[0].endpoint,
            [(node.command, node.values) for node in contour.nodes[1:]],
            True,
        )


def _families(evidence: Evidence, labels: np.ndarray, work: Work):
    """Small adjacent shade families compete as one closed surface proposal."""
    graph = build(evidence, labels)
    parent = list(range(len(graph.regions)))

    def root(index):
        while parent[index] != index:
            index = parent[index]
        return index

    paints = np.array([region.paint for region in graph.regions])
    chroma = paints - paints.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(chroma, axis=1)
    for boundary in graph.boundaries:
        if work.interrupted:
            break
        a, b = boundary.left, boundary.right
        if min(a, b) < 0 or a in graph.hidden or b in graph.hidden:
            continue
        if (
            boundary.line_support > 0.75
            and measure(boundary.points, evidence.target, 1.5) is not None
        ):
            continue
        distance = np.linalg.norm(paints[a] - paints[b])
        cosine = float(chroma[a] @ chroma[b] / max(norms[a] * norms[b], 1e-6))
        alike = cosine >= 0.97 and distance <= 96
        alike |= min(norms[a], norms[b]) < 12 and distance <= 20
        if alike:
            parent[root(b)] = root(a)
    families: dict[int, list[int]] = {}
    for index in range(len(parent)):
        families.setdefault(root(index), []).append(index)
    result = []
    for members in families.values():
        if not 1 < len(members) <= 4:
            continue
        colors = paints[members]
        if np.max(np.linalg.norm(colors[:, None] - colors[None, :], axis=-1)) <= 96:
            result.append(tuple(members))
    return result


def continued(evidence: Evidence, labels: np.ndarray, options: Options, work: Work):
    """Offer isolated compact shapes above their immediate adjoining surfaces.

    A shape with holes, a silhouette contact or substantial unsupported
    boundary displacement is excluded. Hidden continuation is nearest-neighbor
    extension of existing adjacent surfaces; it never invents a semantic part.
    The resulting layer order is acyclic: bases, overlays, then ink.
    """
    scale = float(np.sqrt(np.prod(evidence.scale)))
    tolerance = options.boundary_tolerance * scale
    total = int((~evidence.empty).sum())
    sizes = np.bincount(labels.ravel())
    hidden = set(np.unique(labels[evidence.empty]))
    margin = np.asarray(distance_transform_edt(~evidence.empty))
    base = labels.copy()
    overlays = []
    used: set[int] = set()
    families = _families(evidence, labels, work)
    candidates = families + [(int(region),) for region in np.argsort(-sizes)]
    for members in candidates:
        if work.interrupted or len(overlays) >= 8:
            break
        region = members[0]
        if (
            any(member in hidden or member in used for member in members)
            or not 64 <= sum(sizes[member] for member in members) <= total * 0.05
        ):
            continue
        own = np.isin(labels, members)
        if margin[own].min() <= max(3, 2 * tolerance):
            continue
        loops = _loops(own)
        if len(loops) != 1:
            continue
        points = np.array([*loops[0], loops[0][0]])
        model = ellipse(points, tolerance)
        if model is None:
            continue
        # Exclude a nested transparent cutout even when threshold rounding
        # makes its raster topology ambiguous.
        outside = nearest_indices(own)
        neighbors = labels[outside][own]
        if np.any(np.isin(neighbors, list(hidden))):
            continue
        base[own] = base[outside][own]
        ink = measure(
            points, evidence.target, max(1, options.line_width * scale or 1.5)
        )
        if len(members) > 1 and ink is None:
            base[own] = labels[own]
            continue
        overlays.append(Overlay(int(region), model, ink, members))
        used.update(members)
    return base, tuple(overlays)
