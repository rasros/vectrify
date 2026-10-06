"""Opaque cores inside opacity groups close adjacent-surface antialias seams.

The group uses the source component's modal opacity. Children normalize
their paint alpha by that value, preserving it outside the base. A base exists
only on a near-uniform native core; variable-alpha components remain eligible
for the adjacent-surface interpretation instead. Exact validation chooses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_erosion, find_objects

from vectrify.document.paint import hex_colour
from vectrify.refine import cel
from vectrify.refine.cel_plan.model import Evidence, StageInterruptedError, Work


@dataclass(frozen=True)
class Base:
    component: int
    members: tuple[int, ...]
    opacity: float
    data: str
    origin: tuple[int, int]
    color: str
    core_fraction: float


def propose(
    evidence: Evidence, labels: np.ndarray, components: np.ndarray, work: Work
) -> tuple[Base, ...]:
    assert evidence.opacity is not None

    def check():
        if work.interrupted:
            raise StageInterruptedError("Opacity base proposal interrupted")

    hidden = {int(i) for i in np.unique(labels[evidence.empty])}
    members: dict[int, list[int]] = {}
    for index, box in enumerate(find_objects(labels + 1)):
        check()
        if box is None or index in hidden:
            continue
        own = labels[box] == index
        ids = np.unique(components[box][own])
        if len(ids) == 1 and ids[0] > 0:
            members.setdefault(int(ids[0]), []).append(index)
    boxes = find_objects(components)
    result = []
    for component, indices in sorted(members.items()):
        check()
        box = boxes[component - 1]
        if box is None or len(indices) < 2:
            continue
        own = components[box] == component
        if not binary_erosion(own, iterations=2).any():
            continue
        samples = evidence.opacity[box][own]
        levels = np.rint(samples * 255).astype(np.int32)
        modal = int(np.bincount(levels).argmax())
        opacity = float(np.median(samples[levels == modal]))
        if float(samples.max()) > opacity * 1.02 + 1e-7:
            continue
        # This is a proposal to remove small opacity texture within a material,
        # not a hard exemption. Native error and feature checks pay for it.
        epsilon = opacity * 0.05
        core = (
            own
            & np.isin(labels[box], indices)
            & (evidence.opacity[box] >= opacity - epsilon - 1e-7)
        )
        fraction = float(core.sum() / max(1, int(own.sum())))
        if fraction < 0.5:
            continue

        def boundary(points, _bound):
            check()
            return [
                ("L", tuple(float(v) for v in point))
                for point in cel.simplify(points, 0)[1:]
            ]

        outlines = cel.region_outlines(
            core.astype(np.int32), 0, fit_boundary=boundary, check=check
        )
        if 1 not in outlines:
            continue
        paint_support = core & ~evidence.drawn[box]
        samples = evidence.target[box][paint_support if paint_support.any() else core]
        color = hex_colour(tuple(np.median(samples, axis=0) / 255))
        result.append(
            Base(
                component,
                tuple(indices),
                opacity,
                outlines[1],
                (box[1].start, box[0].start),
                color,
                fraction,
            )
        )
    check()
    return tuple(result)
