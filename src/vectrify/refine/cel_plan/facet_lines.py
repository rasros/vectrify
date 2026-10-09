"""Bounded multiscale material-edge votes independent of source paint labels."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from vectrify.refine.cel_plan.model import StageInterruptedError

MAX_PIXELS = 1536**2
MAX_POINTS = 4096
MAX_LINES = 16
SCALES = (0.7, 1.4, 2.8)
MIN_CONTRAST = 8.0
MIN_SUPPORT = 0.5


class FacetLines:
    """Sample edge ridges once; vote and refine inside each current cell.

    Complete source-pixel paint fitting follows these restricted hypotheses.
    No role label, human feature box or fitted output geometry supplies votes.
    """

    def __init__(self, target, visible):
        self.target, self.visible = target, visible
        self.points: np.ndarray | None = None
        self.normals: np.ndarray | None = None
        self.diagnostics = {"points": 0, "votes": 0, "bounded": 0}

    def prepare(self, work):
        if self.points is not None:
            return
        if self.visible.size > MAX_PIXELS:
            self.diagnostics["bounded"] += 1
            self.points = self.normals = np.empty((0, 2))
            return
        positions, normals = [], []
        for sigma in SCALES:
            support = np.asarray(self.visible, np.float32)
            denominator = np.maximum(gaussian_filter(support, sigma), 1e-12)
            vx = gaussian_filter(support, sigma, order=(0, 1))
            vy = gaussian_filter(support, sigma, order=(1, 0))
            gx = np.zeros(self.visible.shape, np.float32)
            gy = np.zeros_like(gx)
            magnitude = np.zeros_like(gx)
            for channel in range(3):
                if work.interrupted:
                    raise StageInterruptedError("Facet edge preparation interrupted")
                source = np.asarray(self.target[..., channel], np.float32) * support
                average = gaussian_filter(source, sigma) / denominator
                # Derivatives of normalized visible paint: empty background
                # and alpha holes cannot supply a material contrast vote.
                dx = (
                    gaussian_filter(source, sigma, order=(0, 1)) - average * vx
                ) / denominator
                dy = (
                    gaussian_filter(source, sigma, order=(1, 0)) - average * vy
                ) / denominator
                strength = np.hypot(dx, dy)
                better = strength > magnitude
                gx[better], gy[better] = dx[better], dy[better]
                magnitude[better] = strength[better]
            # A step's derivative peak is jump / (sqrt(2 pi) sigma).
            # Equal contrast competes at each scale rather than requiring a
            # large one-pixel jump through a soft material edge.
            ridge = magnitude * (np.sqrt(2 * np.pi) * sigma) >= MIN_CONTRAST
            horizontal = np.abs(gx) >= np.abs(gy)
            peak_x = np.zeros_like(ridge)
            peak_y = np.zeros_like(ridge)
            peak_x[:, 1:-1] = (magnitude[:, 1:-1] >= magnitude[:, :-2]) & (
                magnitude[:, 1:-1] >= magnitude[:, 2:]
            )
            peak_y[1:-1] = (magnitude[1:-1] >= magnitude[:-2]) & (
                magnitude[1:-1] >= magnitude[2:]
            )
            ridge &= np.where(horizontal, peak_x, peak_y) & self.visible
            y, x = np.nonzero(ridge)
            step = max(1, (len(x) + MAX_POINTS - 1) // MAX_POINTS)
            x, y = x[::step], y[::step]
            positions.append(np.column_stack((x + 0.5, y + 0.5)))
            strength = magnitude[y, x]
            normals.append(np.column_stack((gx[y, x], gy[y, x])) / strength[:, None])
        if work.interrupted:
            raise StageInterruptedError("Facet edge preparation interrupted")
        self.points, self.normals = np.concatenate(positions), np.concatenate(normals)
        self.diagnostics["points"] = len(self.points)

    def __call__(self, own, work, *, scope=None):
        self.prepare(work)
        if work.interrupted:
            return
        assert self.points is not None
        assert self.normals is not None
        x, y = np.floor(self.points).astype(int).T
        selected = own[y, x]
        points, normals = self.points[selected], self.normals[selected]
        if len(points) < 8:
            return
        low, high = points.min(axis=0), points.max(axis=0)
        corners = np.array(
            ((low[0], low[1]), (low[0], high[1]), (high[0], low[1]), (high[0], high[1]))
        )
        candidates = []
        for theta in np.arange(32) * np.pi / 32:
            if work.interrupted:
                return
            normal = np.array((np.cos(theta), np.sin(theta)))
            aligned = np.abs(normals @ normal) >= np.cos(np.pi / 16)
            sample = points[aligned]
            if len(sample) < 8:
                continue
            along = sample @ normal
            bins = np.rint(along).astype(int)
            values, counts = np.unique(bins, return_counts=True)
            for index in np.argsort(-counts, kind="stable")[:4]:
                supported = sample[np.abs(along - values[index]) <= 1.5]
                if len(supported) < 8:
                    continue
                center = supported.mean(axis=0)
                values_xy, directions = np.linalg.eigh(np.cov(supported.T))
                fitted = directions[:, 0]
                if fitted @ normal < 0:
                    fitted = -fitted
                tangent = supported @ np.array((-fitted[1], fitted[0]))
                if (
                    np.ptp(tangent) < 8
                    or abs(fitted @ normal) < np.cos(np.pi / 16)
                    or values_xy[0] > 2.25
                ):
                    continue
                rho = float(center @ fitted)
                # The fitted ray must explain source edges across the actual
                # cut, not extrapolate a short patch through a large surface.
                if scope is not None:
                    band = np.abs(scope @ fitted - rho) <= 2
                    if not band.any():
                        continue
                    tangent_vector = np.array((-fitted[1], fitted[0]))
                    expected = np.unique(np.floor(scope[band] @ tangent_vector / 4))
                    supported_all = points[
                        (np.abs(points @ fitted - rho) <= 2)
                        & (np.abs(normals @ fitted) >= np.cos(np.pi / 16))
                    ]
                    present = np.unique(np.floor(supported_all @ tangent_vector / 4))
                    if len(np.intersect1d(expected, present)) < MIN_SUPPORT * len(
                        expected
                    ):
                        continue
                candidates.append((-len(supported), theta, rho, fitted))
        chosen = []
        for _support, _theta, rho, normal in sorted(candidates, key=lambda c: c[:3]):
            if work.interrupted:
                return

            # Compare both endpoints of the cell extent. Close parallel bins
            # may agree locally but predict distinct long material boundaries.
            def equivalent(a, b, normal=normal, rho=rho):
                if normal @ a < 0:
                    a, b = -a, -b
                return np.max(np.abs(corners @ (normal - a) - (rho - b))) < 1

            if any(equivalent(a, b) for a, b in chosen):
                continue
            chosen.append((normal, rho))
            self.diagnostics["votes"] += 1
            yield normal, rho
            if len(chosen) >= MAX_LINES:
                return
