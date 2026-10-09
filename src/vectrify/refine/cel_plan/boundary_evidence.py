"""Bounded evidence for shade contacts too short to support a stroke model.

A monotone color step can be a surface boundary even when coarse CEL line
evidence touches it. A dark trough, incomplete support or opacity discontinuity
keeps the contact protected. This permits a competing merge; native validation
still decides whether its paint and geometry are acceptable.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from vectrify.refine.cel_plan.model import Evidence

MAX_SAMPLES = 64
REACH = 4.0
MONOTONE_NOISE = 2.0
TROUGH_NOISE = 3.0


def shade_fragment(points: np.ndarray, evidence: Evidence, light: np.ndarray) -> bool:
    """Require complete monotone cross-sections, never infer a missing ridge."""
    points = np.asarray(points, dtype=float)
    delta = np.diff(points, axis=0)
    length = np.linalg.norm(delta, axis=1)
    segments = np.flatnonzero(length > 0)
    if not len(segments) or len(segments) > MAX_SAMPLES:
        return False
    middle = (points[segments] + points[segments + 1]) * 0.5
    normal = np.column_stack((-delta[segments, 1], delta[segments, 0]))
    normal /= length[segments, None]
    offsets = np.arange(-REACH, REACH + 0.25, 0.5)
    samples = middle[:, None, :] + offsets[None, :, None] * normal[:, None, :]
    height, width = light.shape
    if (
        (samples[..., 0] < 0.5).any()
        or (samples[..., 0] > width - 0.5).any()
        or (samples[..., 1] < 0.5).any()
        or (samples[..., 1] > height - 0.5).any()
    ):
        return False
    coordinates = [samples[..., 1] - 0.5, samples[..., 0] - 0.5]
    x = np.floor(samples[..., 0]).astype(np.int32)
    y = np.floor(samples[..., 1]).astype(np.int32)
    if evidence.empty[y, x].any():
        return False
    if evidence.opacity is not None:
        alpha = map_coordinates(evidence.opacity, coordinates, order=1)
        if (alpha.min(axis=1) < alpha.max(axis=1) * 0.75 - 1e-7).any():
            return False
    levels = map_coordinates(light, coordinates, order=1)
    increments = np.diff(levels, axis=1)
    monotone = (increments >= -MONOTONE_NOISE).all(axis=1) | (
        increments <= MONOTONE_NOISE
    ).all(axis=1)
    surface = np.minimum(
        np.median(levels[:, offsets <= -2.5], axis=1),
        np.median(levels[:, offsets >= 2.5], axis=1),
    )
    trough = surface - levels[:, np.abs(offsets) <= 2].min(axis=1)
    return bool(monotone.all() and (trough <= TROUGH_NOISE).all())
