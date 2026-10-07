"""Canonical boundary ownership and local region statistics."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.ndimage import find_objects, gaussian_filter, label

from vectrify.refine.cel_plan.model import (
    Boundary,
    Evidence,
    Graph,
    Region,
    StageInterruptedError,
    Work,
)
from vectrify.refine.colour_regions import boundary_chains


def build(
    evidence: Evidence, labels: np.ndarray | None = None, *, work: Work | None = None
) -> Graph:
    def check():
        if work is not None and work.interrupted:
            raise StageInterruptedError("Region graph interrupted")

    check()
    labels = evidence.labels if labels is None else labels
    if labels.flags.writeable:
        labels = labels.copy()
        labels.flags.writeable = False
    count = int(labels.max()) + 1
    components, _ = label(evidence.foreground)
    regions = []
    hidden = frozenset(int(i) for i in np.unique(labels[evidence.empty]))
    alpha_coarse = None
    if evidence.opacity is not None:
        support = (~evidence.empty).astype(np.float32)
        alpha_coarse = gaussian_filter(evidence.opacity * support, 4) / np.maximum(
            gaussian_filter(support, 4), 1e-6
        )
    # A label's statistics need only its bounding box. Full-canvas masks for
    # every region made noisy artwork scale as pixels times region count.
    boxes = find_objects(labels + 1, max_label=count)
    for index, box in enumerate(boxes):
        check()
        box = box or (slice(0, 0), slice(0, 0))
        mask = (labels[box] == index) & ~evidence.empty[box]
        paint = mask if evidence.opacity is not None else mask & ~evidence.line[box]
        color_field = (
            evidence.target if evidence.opacity is not None else evidence.smooth
        )
        pixels = color_field[box][paint if paint.any() else mask]
        median = np.median(pixels, axis=0) if len(pixels) else np.full(3, 255)
        residual = evidence.target[box][paint] - evidence.coarse[box][paint]
        contrast = (
            float(np.linalg.norm(np.median(residual, axis=0))) if len(residual) else 0
        )
        y, x = np.nonzero(mask)
        if len(x) > 2:
            covariance = np.cov(np.stack((x, y)))
            values = np.linalg.eigvalsh(covariance)
            elongated = float(np.sqrt((values[-1] + 1) / (values[0] + 1)))
        else:
            elongated = 1
        component = np.bincount(components[box][mask]).argmax() if mask.any() else 0
        feature = max(contrast / 60, elongated / 40)
        if alpha_coarse is not None and evidence.opacity is not None:
            # Continuous opacity ramps produce narrow quantization bands, not
            # thin ink features. Protect actual local opacity discontinuities.
            alpha_contrast = (
                float(
                    np.abs(evidence.opacity[box][mask] - alpha_coarse[box][mask]).mean()
                )
                if mask.any()
                else 0
            )
            feature = max(contrast / 60, alpha_contrast / 0.15)
        regions.append(
            Region(
                index,
                int(mask.sum()),
                (float(median[0]), float(median[1]), float(median[2])),
                float(evidence.texture[box][mask].mean()) if mask.any() else 0,
                min(1.0, feature),
                int(component),
                float(np.median(evidence.opacity[box][mask]))
                if evidence.opacity is not None and mask.any()
                else 1.0,
                evidence.filled_line_width > 0
                and bool(evidence.drawn[box][mask].any()),
                (
                    float(evidence.opacity[box][mask].min()),
                    float(evidence.opacity[box][mask].max()),
                )
                if evidence.opacity is not None and mask.any()
                else (1.0, 1.0),
            )
        )
    padded = np.pad(labels, 1, constant_values=-1)
    boundaries = []
    ends: dict[tuple[float, float], int] = {}
    for points in boundary_chains(padded, check=check):
        check()
        middle = (points[0] + points[1]) / 2
        delta = points[1] - points[0]
        normal = np.array([-delta[1], delta[0]]) * 0.25
        x, y = np.floor(middle + normal).astype(int)
        left = int(padded[y, x])
        x, y = np.floor(middle - normal).astype(int)
        right = int(padded[y, x])
        points = points - 1
        points.flags.writeable = False
        xy = np.floor(points).astype(int)
        x = np.clip(xy[:, 0], 0, labels.shape[1] - 1)
        y = np.clip(xy[:, 1], 0, labels.shape[0] - 1)
        support = float(evidence.line[y, x].mean())
        boundaries.append(Boundary(len(boundaries), left, right, points, support))
        for point in (points[0], points[-1]):
            key = (float(point[0]), float(point[1]))
            ends[key] = ends.get(key, 0) + 1
    return Graph(
        labels,
        tuple(regions),
        tuple(boundaries),
        tuple(p for p, count in ends.items() if count > 2),
        hidden,
    )


def shared(result: Graph, original: Graph, work: Work) -> Graph:
    """Reuse only equal immutable statistics and complete canonical chains.

    Unchanged boundaries keep their identity. New chains get fresh identities
    beyond the original range; graph consumers resolve identities rather than
    treating a boundary's identity as its position in the tuple.
    """
    regions = []
    for region in result.regions:
        if work.interrupted:
            raise StageInterruptedError("Shared source graph interrupted")
        previous = (
            original.regions[region.id] if region.id < len(original.regions) else None
        )
        regions.append(
            previous if previous is not None and region == previous else region
        )

    def key(edge):
        return (
            edge.left,
            edge.right,
            len(edge.points),
            tuple(edge.points[0]),
            tuple(edge.points[-1]),
        )

    candidates = {}
    for edge in original.boundaries:
        if work.interrupted:
            raise StageInterruptedError("Shared source graph interrupted")
        if not edge.points.flags.writeable:
            candidates.setdefault(key(edge), []).append(edge)
    boundaries = []
    next_id = max((b.id for b in original.boundaries), default=-1) + 1
    for edge in result.boundaries:
        if work.interrupted:
            raise StageInterruptedError("Shared source graph interrupted")
        matching = next(
            (
                old
                for old in candidates.get(key(edge), ())
                if old.line_support == edge.line_support
                and np.array_equal(old.points, edge.points)
            ),
            None,
        )
        if matching is not None:
            boundaries.append(matching)
        else:
            boundaries.append(replace(edge, id=next_id))
            next_id += 1
    return replace(result, regions=tuple(regions), boundaries=tuple(boundaries))
