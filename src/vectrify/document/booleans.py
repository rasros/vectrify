"""Union, Subtract, Intersect and Exclude of filled areas, with curves kept."""

from __future__ import annotations

from collections.abc import Sequence

import pathops

# The area each operation keeps, of a target and the cutters in front of it.
BOOLEANS = {
    # Inside any of them.
    "union": pathops.PathOp.UNION,
    # Inside the target and outside every cutter.
    "subtract": pathops.PathOp.DIFFERENCE,
    # Inside all of them.
    "intersect": pathops.PathOp.INTERSECTION,
    # Inside an odd number of them: where two overlap is open.
    "exclude": pathops.PathOp.XOR,
}


def boolean_area(
    operation: str, target: pathops.Path, cutters: Sequence[pathops.Path]
) -> pathops.Path:
    """The *operation*'s area of outlines given in one frame.

    Subtract removes the cutters' combined area from the target; the other
    operations fold over the outlines in order, which gives the same area
    whatever the order is.
    """
    if operation == "subtract":
        cutter = cutters[0]
        for path in cutters[1:]:
            cutter = pathops.op(cutter, path, pathops.PathOp.UNION)
        return pathops.op(target, cutter, pathops.PathOp.DIFFERENCE)
    result = target
    for path in cutters:
        result = pathops.op(result, path, BOOLEANS[operation])
    return result


def overlaps(
    target: pathops.Path, cutters: Sequence[pathops.Path], tolerance: float
) -> bool:
    """Whether any cutter covers more than *tolerance* of the target's area."""
    return any(
        pathops.op(target, cutter, pathops.PathOp.INTERSECTION).area > tolerance
        for cutter in cutters
    )
