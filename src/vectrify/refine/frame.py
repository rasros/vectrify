"""A path's local coordinates against a reference crop's pixels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vectrify.document import Document
from vectrify.document.transforms import object_matrix
from vectrify.operations.generate import Region


@dataclass(frozen=True)
class Frame:
    """Maps a path's local coordinates to the crop's pixels and back."""

    matrix: np.ndarray
    offset: np.ndarray

    def pixels(self, values: tuple[float, ...]) -> np.ndarray:
        points = np.asarray(values, dtype=np.float64).reshape(-1, 2)
        return points @ self.matrix.T + self.offset

    def local(self, points: np.ndarray) -> tuple[float, ...]:
        inverse = np.linalg.inv(self.matrix)
        return tuple(float(v) for v in ((points - self.offset) @ inverse.T).ravel())


def object_frame(document: Document, oid: str, region: Region, size) -> Frame | None:
    """Object *oid*'s frame in *region* drawn at *size*, or None when it is
    flattened to nothing."""
    matrix = object_matrix(document, oid)
    a, b, c, d, e, f = matrix
    scale = np.diag([size[0] / region.width, size[1] / region.height])
    linear = scale @ np.array([[a, c], [b, d]], dtype=np.float64)
    if abs(np.linalg.det(linear)) < 1e-12:
        return None
    return Frame(linear, scale @ (np.array([e, f]) - [region.x, region.y]))
