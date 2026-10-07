"""Bounded paired contours for a complete source-supported closed ink rim.

Both edges are fitted from their complete source perimeter. The RGB cavity is
left open; callers must restore neighboring paint and prove core coverage and
order before this geometry can compete with the original fragment union.
"""

from __future__ import annotations

import numpy as np

from vectrify.document import Geometry
from vectrify.document.join import transformed_geometry
from vectrify.document.redraw import root_matrix
from vectrify.document.topology import inverse_matrix
from vectrify.refine.cel_plan.geometry import fitted
from vectrify.refine.crossings import crossings
from vectrify.refine.tracing import _loops

MAX_PERIMETER = 4096


def rim(document, survivor, evidence, box, own, original, tolerance, work, diagnostics):
    if work.interrupted:
        return None
    loops = _loops(own)
    if len(loops) != 2 or not all(s.closed for s in original.subpaths):
        diagnostics["rim_topology_exclusions"] += 1
        return None
    if sum(len(loop) for loop in loops) > MAX_PERIMETER:
        diagnostics["rim_perimeter_limits"] += 1
        return None
    areas = []
    models = []
    for loop in loops:
        if work.interrupted:
            return None
        points = np.array([*loop, loop[0]], dtype=float)
        areas.append(
            float(
                np.sum(points[:-1, 0] * points[1:, 1] - points[1:, 0] * points[:-1, 1])
            )
        )
        points += (box.x, box.y)
        points = points / evidence.scale + evidence.offset
        models.append(fitted(points, tolerance))
    if work.interrupted:
        return None
    # A connected ring has opposite perimeter winding. Two unrelated closed
    # marks cannot justify inventing a cavity or joining their blank gap.
    if areas[0] * areas[1] >= 0:
        diagnostics["rim_topology_exclusions"] += 1
        return None
    shape = transformed_geometry(
        Geometry("source-rim", tuple(m.contour for m in models)),
        inverse_matrix(root_matrix(document, survivor)),
    )
    if crossings(shape):
        diagnostics["rim_topology_exclusions"] += 1
        return None
    old_nodes = sum(len(s.nodes) for s in original.subpaths)
    nodes = sum(len(s.nodes) for s in shape.subpaths)
    if nodes >= old_nodes:
        diagnostics["rim_no_compaction"] += 1
        return None
    return shape, tuple(m.kind for m in models)
