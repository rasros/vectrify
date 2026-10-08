"""Native-phase stroke constraints at independently observed source gaps.

No repair or new endpoint comes from these observations. Test actual exported
stroke bodies, including caps, at raw source absence positions. Rasterize only
the required native tiles; no full native stroke or patched RGB image is made.
This inspected engineering contract is an explicit offline competitor, not a
calibrated guarantee of preserving every possible source gap.
"""

from __future__ import annotations

import numpy as np

from vectrify.refine.cel_plan.line_fidelity import SourceLineGuard
from vectrify.refine.cel_plan.model import StageInterruptedError
from vectrify.refine.cel_plan.score import render

MAX_NATIVE_PIXELS = 1536**2
MAX_PROFILE_SAMPLES = 4096
MAX_POINTS = 4096
MAX_TILES = 32
TILE_SIDE = 64
MAX_MASK_BYTES = 16 * 1024**2
ALPHA_TOLERANCE = 1 / 255 + 1e-7


def _check(work):
    if work.interrupted:
        raise StageInterruptedError("Source absence proof interrupted")


class SourceAbsence:
    def __init__(self, evidence, profiles, work):
        _check(work)
        h, w = evidence.rgba.shape[:2]
        if h * w > MAX_NATIVE_PIXELS or evidence.source_size != (w, h):
            raise ValueError("Source absence exceeds the native frame bound")
        for profile in profiles:
            _check(work)
            samples = (
                1
                + np.maximum(
                    1,
                    np.ceil(
                        2 * np.linalg.norm(np.diff(profile.points, axis=0), axis=1)
                    ),
                ).sum()
            )
            if samples > MAX_PROFILE_SAMPLES:
                raise ValueError("Source absence exceeds the per-profile sample bound")
        guard = SourceLineGuard(evidence.rgba, profiles, work=work)
        self.points = guard.gap_centres(limit=MAX_POINTS, work=work)
        self.size = (w, h)
        # Match scipy's full-frame bilinear constant sampling exactly. A query
        # crossing a tile seam still reads all four actual native neighbours.
        coordinates = self.points - 0.5
        valid = (
            (coordinates[:, 0] >= 0)
            & (coordinates[:, 1] >= 0)
            & (coordinates[:, 0] <= w - 1)
            & (coordinates[:, 1] <= h - 1)
        )
        self.indices = np.flatnonzero(valid)
        self.indices.flags.writeable = False
        coordinates = coordinates[valid]
        origin = np.floor(coordinates).astype(int)
        fraction = coordinates - origin
        self.queries = []
        tiles = set()
        for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
            pixels = np.add(origin, (dx, dy))
            weights = np.prod(np.abs((1 - dx, 1 - dy) - fraction), axis=1)
            active = (weights > 0) & (pixels[:, 0] < w) & (pixels[:, 1] < h)
            locations = pixels[active] // TILE_SIDE
            tiles.update(map(tuple, locations.tolist()))
            query = (np.flatnonzero(active), pixels[active], weights[active])
            for array in query:
                array.flags.writeable = False
            self.queries.append(query)
        if len(tiles) > MAX_TILES:
            raise ValueError("Source absence exceeds the native tile bound")
        if len(tiles) * TILE_SIDE**2 * np.dtype(np.float32).itemsize > MAX_MASK_BYTES:
            raise ValueError("Source absence exceeds the mask byte bound")
        self.tiles = tuple(sorted(tiles))
        _check(work)

    def permits(self, geometry, width, cap, work):
        """Publish a positive proof only after every native query completes."""
        _check(work)
        if not np.isfinite(width) or width <= 0 or cap not in {"round", "butt"}:
            raise ValueError("Source absence requires a finite supported stroke style")
        if not len(self.points):
            return True
        data = geometry.path_data()
        observed = np.zeros(len(self.indices), np.float64)
        w, h = self.size
        for tx, ty in self.tiles:
            _check(work)
            x, y = int(tx) * TILE_SIDE, int(ty) * TILE_SIDE
            tw, th = min(TILE_SIDE, w - x), min(TILE_SIDE, h - y)
            svg = (
                '<svg xmlns="http://www.w3.org/2000/svg" '
                f'width="{tw}" height="{th}" viewBox="{x} {y} {tw} {th}">'
                f'<path d="{data}" fill="none" stroke="black" '
                f'stroke-width="{width}" stroke-linecap="{cap}" '
                'stroke-linejoin="round"/></svg>'
            )
            alpha = render(svg, (tw, th))[..., 3]
            _check(work)
            for indices, pixels, weights in self.queries:
                active = np.all(pixels // TILE_SIDE == (tx, ty), axis=1)
                local = pixels[active] - (x, y)
                observed[indices[active]] += (
                    weights[active] * alpha[local[:, 1], local[:, 0]]
                )
        _check(work)
        return bool(np.all(observed <= ALPHA_TOLERANCE))
