"""Conservative native paint bounds shared by planner edits and search."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from vectrify.document.join import path_style
from vectrify.document.transforms import root_matrix

if TYPE_CHECKING:
    from vectrify.document.model import Document


def conservative_bounds(
    document: Document, oid: str
) -> tuple[float, float, float, float]:
    """Conservative control-hull bounds, including transformed stroke width."""
    a, b, c, d, e, f = root_matrix(document, oid)
    points = np.array(
        [
            node.values[i : i + 2]
            for subpath in document.geometry_for(oid).subpaths
            for node in subpath.nodes
            for i in range(0, len(node.values), 2)
        ]
    ).reshape(-1, 2)
    if not len(points):
        return (0, 0, 0, 0)
    linear = np.array([[a, c], [b, d]])
    points = points @ linear.T + (e, f)
    style = path_style(document, document.element(oid))
    width = float(style["stroke-width"]) if style["stroke"] != "none" else 0
    # Include miter paint in the proposal crop, even at acute corners. Exact
    # full validation later includes all layers and all native pixels.
    pad = 4 + width * float(np.linalg.norm(linear, ord=2)) * 5
    low, high = points.min(axis=0) - pad, points.max(axis=0) + pad
    return (float(low[0]), float(low[1]), float(high[0]), float(high[1]))
