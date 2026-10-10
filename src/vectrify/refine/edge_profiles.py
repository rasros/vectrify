"""Read visible fill boundaries in their frozen compositing context."""

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from vectrify.refine.snap import _Frame


class EdgeProfiles:
    """Find the lowest-error fill edge across a short, outward-facing normal.

    The frozen artwork determines where this path can change the image.
    Occluded edges and ambiguous normals keep their original positions.
    """

    def __init__(
        self,
        base: np.ndarray,
        response: np.ndarray,
        target: np.ndarray,
        coverage: np.ndarray,
        frame: _Frame,
        pixel_scale: float,
    ):
        self.frame = frame
        self.coverage = coverage
        self.power = np.square(response).sum(-1)
        self.raw_signal = self.power - 2 * ((target - base) * response).sum(-1)
        self.signal = gaussian_filter(self.raw_signal, 0.6)
        self.steps = np.arange(-6, 6.01, 0.25) / pixel_scale
        self.probe = 1.5 / pixel_scale
        self.threshold = max(1e-8, float(self.power.max()) * 0.02)

    @staticmethod
    def _read(field: np.ndarray, points: np.ndarray) -> np.ndarray:
        return map_coordinates(
            field, [points[..., 1] - 0.5, points[..., 0] - 0.5], order=1, mode="nearest"
        )

    def __call__(self, points: np.ndarray, tangents: np.ndarray) -> np.ndarray:
        position = self.frame.pixels(tuple(points.ravel()))
        direction = tangents @ self.frame.matrix.T
        normal = np.column_stack((direction[:, 1], -direction[:, 0]))
        size = np.linalg.norm(normal, axis=1)
        normal /= np.maximum(size[:, None], 1e-9)
        plus = self._read(self.coverage, position + self.probe * normal)
        minus = self._read(self.coverage, position - self.probe * normal)
        normal *= np.where(plus > minus, -1, 1)[:, None]
        samples = position[:, None] + self.steps[None, :, None] * normal[:, None]
        signal = self._read(self.signal, samples)
        # Painting before an outward-facing boundary incurs this cumulative
        # change relative to leaving the path absent along the whole profile.
        cost = np.cumsum(signal, axis=1)
        index = cost.argmin(axis=1)
        rows = np.arange(len(points))
        safe = index.clip(1, len(self.steps) - 2)
        left, middle, right = (
            cost[rows, safe - 1],
            cost[rows, safe],
            cost[rows, safe + 1],
        )
        curvature = left - 2 * middle + right
        fraction = np.divide(
            0.5 * (left - right),
            curvature,
            out=np.zeros_like(curvature),
            where=curvature > 1e-10,
        ).clip(-0.5, 0.5)
        # Costs lie between adjacent sample positions, not on a pixel centre.
        offset = self.steps[safe] + (fraction + 0.5) * (self.steps[1] - self.steps[0])
        known = (
            (self._read(self.power, position) > self.threshold)
            & (np.abs(plus - minus) > 0.25)
            & (size > 1e-6)
        )
        known &= (index > 0) & (index < len(self.steps) - 1)
        reached = position + np.where(known, offset, 0)[:, None] * normal
        return np.array(self.frame.local(reached)).reshape(-1, 2)
