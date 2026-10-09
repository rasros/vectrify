"""Virtual source observations separate ink from paint before native cuts.

An existing SVG owner is a storage unit, not a material or ink classification.
These observations never change that owner or its atom namespace. The final
joint decoder partitions original atoms once, after choosing complete classes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vectrify.refine.cel_plan.local import Box
from vectrify.refine.cel_plan.material_groups import MAX_MATERIALS
from vectrify.refine.cel_plan.model import StageInterruptedError

CHUNK_PIXELS = 65_536


@dataclass(frozen=True)
class Roles:
    source: np.ndarray
    owners: np.ndarray
    ink: np.ndarray


def observations(source, own, ink, work):
    """Give each nonempty owner/role pair a bounded, immutable virtual index."""
    if work.interrupted:
        return None
    primary = source[own]
    if not len(primary) or primary.min() < 0 or primary.max() >= MAX_MATERIALS:
        raise ValueError("Source roles require bounded complete owner support")
    codes = np.unique(2 * primary + ink[own])
    if len(codes) > MAX_MATERIALS or work.interrupted:
        return None
    lookup = np.full(int(codes[-1]) + 1, -1, np.int32)
    lookup[codes] = np.arange(len(codes), dtype=np.int32)
    classified = np.full(source.shape, -1, np.int32)
    classified[own] = lookup[2 * primary + ink[own]]
    owners = (codes // 2).astype(np.int32)
    kinds = (codes % 2).astype(bool)
    if work.interrupted:
        return None
    for value in (classified, owners, kinds):
        value.flags.writeable = False
    return Roles(classified, owners, kinds)


def moments(roles, evidence, graph, options, work):
    """Stream intrinsic paint statistics with original atom feature weights.

    The existing coverage carrier supplies alpha. Virtual role observations
    inherit each pixel's actual source atom weight, including when an SVG
    owner contains several original atoms with different feature support.
    """
    count = len(roles.owners)
    result = np.zeros((count, 7, 7), np.float64)
    weights = np.array(
        [
            1 + 8 * options.protection * r.feature * (1 - r.texture)
            for r in graph.regions
        ]
    )
    height, width = graph.labels.shape
    scale = max(height, width, 1)
    for chunk in Box(0, 0, width, height).chunks(CHUNK_PIXELS):
        if work.interrupted:
            raise StageInterruptedError("Source role statistics interrupted")
        box = chunk.slices
        y, x = np.nonzero(roles.source[box] >= 0)
        ids = roles.source[box][y, x]
        basis = np.column_stack(
            (
                np.ones(len(ids)),
                (x + chunk.x + 0.5) / scale,
                (y + chunk.y + 0.5) / scale,
                evidence.target[box][y, x] / 255,
                np.ones(len(ids)),
            )
        )
        weight = weights[graph.labels[box][y, x]]
        for a in range(7):
            for b in range(a, 7):
                if work.interrupted:
                    raise StageInterruptedError("Source role statistics interrupted")
                value = np.bincount(
                    ids, weights=weight * basis[:, a] * basis[:, b], minlength=count
                )
                result[:, a, b] += value
                if a != b:
                    result[:, b, a] += value
    return result
