"""Canonical boundary ownership and local region statistics."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import find_objects, label

from vectrify.refine.cel_plan.model import Boundary, Evidence, Graph, Region
from vectrify.refine.colour_regions import boundary_chains


def build(evidence: Evidence, labels: np.ndarray | None = None) -> Graph:
    labels = evidence.labels if labels is None else labels
    count = int(labels.max()) + 1
    components, _ = label(evidence.foreground)
    regions = []
    hidden = frozenset(int(i) for i in np.unique(labels[evidence.empty]))
    # A label's statistics need only its bounding box. Full-canvas masks for
    # every region made noisy artwork scale as pixels times region count.
    boxes = find_objects(labels + 1, max_label=count)
    for index, box in enumerate(boxes):
        box = box or (slice(0, 0), slice(0, 0))
        mask = (labels[box] == index) & ~evidence.empty[box]
        paint = mask & ~evidence.line[box]
        pixels = evidence.smooth[box][paint if paint.any() else mask]
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
        regions.append(
            Region(
                index,
                int(mask.sum()),
                (float(median[0]), float(median[1]), float(median[2])),
                float(evidence.texture[box][mask].mean()) if mask.any() else 0,
                min(1.0, max(contrast / 60, elongated / 40)),
                int(component),
            )
        )
    padded = np.pad(labels, 1, constant_values=-1)
    boundaries = []
    ends: dict[tuple[float, float], int] = {}
    for points in boundary_chains(padded):
        middle = (points[0] + points[1]) / 2
        delta = points[1] - points[0]
        normal = np.array([-delta[1], delta[0]]) * 0.25
        x, y = np.floor(middle + normal).astype(int)
        left = int(padded[y, x])
        x, y = np.floor(middle - normal).astype(int)
        right = int(padded[y, x])
        points = points - 1
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
