"""Select complete original source chains for a separated filled ink band.

An original physical profile, its copied anchors, measured qualification and
negative observations supply the seed. Neither a human redraw nor a carrier
defines this source chain. Ambiguous matches and any observed internal gap
exclude this complete-chain interpretation. Native painted/body/ownership
validation is still required for the resulting competitor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

from vectrify.document import Geometry
from vectrify.document.join import curve_path, transformed_geometry
from vectrify.document.redraw import root_matrix
from vectrify.refine.cel_plan.filled_bands import MAX_NODES, MAX_WIDTH, Band, _check
from vectrify.refine.cel_plan.geometry import fitted
from vectrify.refine.cel_plan.ink import measure
from vectrify.refine.cel_plan.local import Box
from vectrify.refine.cel_plan.score import render
from vectrify.refine.crossings import crossings

MAX_PROFILES = 128
MAX_POINTS = 2048
MAX_PIXELS = 128 * 128
MAX_EXTENT = 128
MAX_DISTANCE = 3


@dataclass(frozen=True)
class SourceBand:
    band: Band
    paint: str
    profile: int
    anchors: np.ndarray
    bounds: tuple[int, int, int, int]
    support: float


class SourceBands:
    def __init__(self, evidence, guard):
        self.evidence, self.guard = evidence, guard

    def seed(self, document, oid, geometry, rule, work, *, tolerance=0.75, width=0):
        """One uniquely matching complete physical source chain in native space.

        Source centres must be near the entire band, mostly within its coverage,
        and span its extent. A nearest fragment cannot seed a complete outline.
        This source competitor does not relax uniform intrinsic band inversion.
        """
        _check(work)
        if (
            rule not in {"nonzero", "evenodd"}
            or not 0 < tolerance <= 3
            or not 0 <= width <= MAX_WIDTH
        ):
            raise ValueError("Source band fitting requires bounded settings")
        if (
            not geometry.subpaths
            or any(not s.closed for s in geometry.subpaths)
            or sum(len(s.nodes) for s in geometry.subpaths) > MAX_NODES
        ):
            return None
        frame = root_matrix(document, oid)
        native = transformed_geometry(geometry, frame)
        bounds = np.asarray(curve_path(native, rule).bounds)
        if (
            not np.isfinite(bounds).all()
            or np.max(np.abs(bounds)) > 1_000_000
            or np.max(bounds[2:] - bounds[:2]) > MAX_EXTENT
        ):
            return None
        lo = np.maximum(0, np.floor(bounds[:2]).astype(int) - 40)
        hi = np.minimum(self.evidence.source_size, np.ceil(bounds[2:]).astype(int) + 40)
        box = Box(*map(int, (*lo, *hi)))
        if not 0 < box.area <= MAX_PIXELS:
            return None
        size = (box.right - box.x, box.bottom - box.y)
        alpha = render(
            f'<svg width="{size[0]}" height="{size[1]}" '
            f'viewBox="{box.x} {box.y} {size[0]} {size[1]}">'
            f'<path d="{native.path_data()}" fill="white" fill-rule="{rule}"/></svg>',
            size,
        )[..., 3]
        _check(work)
        distance = distance_transform_edt(alpha <= 1 / 255)
        profiles = self.guard.original_profiles(work=work)
        if len(profiles) > MAX_PROFILES:
            return None
        matches = []
        for index, profile in enumerate(profiles):
            _check(work)
            observed = self.guard.source_breaks(profile, work=work)
            if (
                observed is None
                or observed.anchors is None
                or len(observed.points) > MAX_POINTS
                or observed.gaps.any()
                or observed.qualified.mean() < 0.6
                or np.array_equal(observed.points[0], observed.points[-1])
            ):
                continue
            points = observed.points
            q = points - lo - 0.5
            d = map_coordinates(
                distance, [q[:, 1], q[:, 0]], order=1, mode="constant", cval=999
            )
            cover = map_coordinates(
                alpha, [q[:, 1], q[:, 0]], order=1, mode="constant", cval=0
            )
            if (
                d.max() > MAX_DISTANCE
                or (cover > 0.05).mean() < 0.5
                or np.linalg.norm(np.ptp(points, axis=0))
                < 0.7 * np.linalg.norm(bounds[2:] - bounds[:2])
            ):
                continue
            matches.append((index, observed))
        if len(matches) != 1:
            return None
        index, observed = matches[0]
        anchors = observed.anchors
        length = float(np.linalg.norm(np.diff(anchors, axis=0), axis=1).sum())
        hint = float(np.clip(alpha.sum() / max(length, 1e-12), 0.8, MAX_WIDTH))
        source = self.evidence.rgba[box.slices]
        measured = measure(
            anchors - lo,
            source[..., :3] * 255,
            hint,
            visible=source[..., 3] > 1 / 255,
            opacity=source[..., 3],
        )
        _check(work)
        if measured is None:
            return None
        model = fitted(anchors, tolerance)
        shape = Geometry("source-band", (model.contour,))
        if (
            len(model.contour.nodes) > 8
            or crossings(shape)
            or not 0.8 <= measured.width <= MAX_WIDTH
        ):
            return None
        paint = "#" + "".join(
            f"{int(v):02x}" for v in np.rint(np.clip(measured.paint, 0, 255))
        )
        _check(work)
        return SourceBand(
            Band(shape, width or measured.width, len(anchors)),
            paint,
            index,
            anchors,
            (box.x, box.y, box.right, box.bottom),
            float(observed.qualified.mean()),
        )
