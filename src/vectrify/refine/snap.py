"""Snap to the reference: move the selected paths' points onto its edges.

Where the node search tries random moves and keeps what scores better, this
reads the edge off the reference and puts the outline on it. For each filled
path a target mask is built: the path's own coverage, except in a band around
its outline, where each reference pixel belongs to the shape when its colour
is closer to the colour inside the path than to the colour just outside.

Each pass samples every segment and looks along its normal for the mask's
edge. A point moves by the shift that best carries both of its segments onto
the edge where they meet it, so corners follow their two sides and smooth
points only move across the outline. Cubic handles are then refitted to the
edge between their ends. With detail on, segments that still miss the edge
are split where they miss it most.

All of it works in the reference's pixels, so steps and tolerances are the
same for a transformed or tiny path. Pinned endpoints and linked edges stay.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field, replace
from typing import cast

import cairosvg
import numpy as np
from PIL import Image
from scipy import ndimage

from vectrify.document import Document, Geometry, PathNode
from vectrify.document.hit_test import IDENTITY, multiply, transform
from vectrify.document.join import path_style
from vectrify.document.model import new_id
from vectrify.image_utils import resize_long_side
from vectrify.operations.generate import Region
from vectrify.vector.nodes import Frozen, Paths

# The band around the outline the edge is looked for in, as a share of the
# path's larger side in pixels, and never less than a few pixels.
BAND = 0.1
MIN_BAND = 3.0
# Reference crops larger than this are shrunk first.
LONG_SIDE = 512
PASSES = 6
# How far a point may go in all, in bands.
REACH = 2.0
# Stop once no point moves further than this, in pixels.
SETTLED = 0.2
# Inside and outside colours closer than this (RGB, 0-1) cannot tell the
# edge apart, so the path's own coverage stands there.
CONTRAST = 0.08
# Detail: a segment is split when the edge is further than this from it.
SPLIT_ERROR = 1.5
SPLIT_PASSES = 6
# How finely the normal is searched, in pixels.
STEP = 0.5


@dataclass(frozen=True)
class _Frame:
    """Maps a path's local coordinates to the crop's pixels and back."""

    matrix: np.ndarray
    offset: np.ndarray

    def pixels(self, values: tuple[float, ...]) -> np.ndarray:
        points = np.asarray(values, dtype=np.float64).reshape(-1, 2)
        return points @ self.matrix.T + self.offset

    def local(self, points: np.ndarray) -> tuple[float, ...]:
        inverse = np.linalg.inv(self.matrix)
        return tuple(float(v) for v in ((points - self.offset) @ inverse.T).ravel())


def _frame(document: Document, oid: str, region: Region, size) -> _Frame | None:
    matrix = IDENTITY
    for ancestor in document.ancestry(oid):
        matrix = multiply(matrix, transform(ancestor.get("transform")))
    a, b, c, d, e, f = matrix
    scale = np.diag([size[0] / region.width, size[1] / region.height])
    linear = scale @ np.array([[a, c], [b, d]], dtype=np.float64)
    if abs(np.linalg.det(linear)) < 1e-12:
        return None
    return _Frame(linear, scale @ (np.array([e, f]) - [region.x, region.y]))


@dataclass
class _Node:
    id: str
    command: str
    # Pixel coordinates: one row for M and L, three (c1, c2, end) for C.
    points: np.ndarray
    # The whole node stays (a linked edge's end) or only its endpoint does.
    fixed: bool
    pinned: bool
    original: PathNode | None
    # Where the endpoint started, which it never leaves by more than REACH.
    origin: np.ndarray = field(init=False)

    def __post_init__(self):
        self.origin = self.points[-1].copy()

    @property
    def end(self) -> np.ndarray:
        return self.points[-1]


@dataclass
class _Contour:
    nodes: list[_Node]
    closed: bool


def _bezier(control: np.ndarray, t: np.ndarray):
    """Points and tangents along a line (two controls) or cubic (four)."""
    t = t[:, None]
    if len(control) == 2:
        a, b = control
        return a + (b - a) * t, np.repeat((b - a)[None], len(t), axis=0)
    a, b, c, d = control
    u = 1 - t
    points = u**3 * a + 3 * u**2 * t * b + 3 * u * t**2 * c + t**3 * d
    tangents = 3 * u**2 * (b - a) + 6 * u * t * (c - b) + 3 * t**2 * (d - c)
    return points, tangents


def _segments(contour: _Contour) -> list[tuple[int, np.ndarray]]:
    """(index of the node the segment ends at, its controls) for each segment.

    The closing line of a closed contour ends at node 0 and is left out when
    the last node already sits on the first.
    """
    nodes = contour.nodes
    found = []
    for i in range(1, len(nodes)):
        start = nodes[i - 1].end
        found.append((i, np.vstack([start, nodes[i].points])))
    if (
        contour.closed
        and len(nodes) > 1
        and np.linalg.norm(nodes[-1].end - nodes[0].end) > 1e-6
    ):
        found.append((0, np.vstack([nodes[-1].end, nodes[0].end])))
    return found


def _path_data(contours: list[_Contour]) -> str:
    parts = []
    for contour in contours:
        for node in contour.nodes:
            parts.append(
                node.command + " ".join(f"{v:.4f}" for v in node.points.ravel())
            )
        if contour.closed:
            parts.append("Z")
    return " ".join(parts)


def _coverage(contours: list[_Contour], size, rule: str) -> np.ndarray:
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size[0]}" '
        f'height="{size[1]}"><path d="{_path_data(contours)}" fill="#000" '
        f'fill-rule="{rule}"/></svg>'
    )
    png = cairosvg.svg2png(bytestring=svg.encode(), background_color="white")
    assert png is not None
    with Image.open(io.BytesIO(png)) as image:
        return np.asarray(image.convert("L")) < 128


def _local_colour(reference, where, window):
    """The mean colour of *where* around each pixel, or its median overall."""
    weight = where.astype(np.float64)
    if not weight.any():
        return None
    total = ndimage.uniform_filter(weight, window, mode="constant")
    fallback = np.median(reference[where], axis=0)
    colour = np.empty_like(reference)
    for channel in range(3):
        summed = ndimage.uniform_filter(
            reference[..., channel] * weight, window, mode="constant"
        )
        colour[..., channel] = np.where(
            total > 1e-3, summed / np.maximum(total, 1e-9), fallback[channel]
        )
    return colour


def distance_transform_edt(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance to the nearest zero, typed for the one return value."""
    return cast(np.ndarray, ndimage.distance_transform_edt(mask))


def nearest_indices(mask: np.ndarray) -> tuple[np.ndarray, ...]:
    """Index arrays of the nearest zero for every pixel, for fancy indexing."""
    indices = ndimage.distance_transform_edt(
        mask, return_distances=False, return_indices=True
    )
    return tuple(cast(np.ndarray, indices))


def _nearest_colour(reference, where):
    """The colour of the nearest pixel in *where*, lightly smoothed.

    Unlike a mean it never blends two backgrounds into a colour neither has.
    """
    if not where.any():
        return None
    smooth = ndimage.gaussian_filter(reference, (1.0, 1.0, 0))
    rows, columns = nearest_indices(~where)
    return smooth[rows, columns]


class _Target:
    """The shape's pixels as the reference shows them, near the outline."""

    def __init__(self, reference, coverage, band):
        self.band = band
        inside = distance_transform_edt(coverage)
        outside = distance_transform_edt(~coverage)
        signed = outside - inside
        near = np.abs(signed) <= band
        deep = signed < -band
        if deep.sum() < 10:
            deep = coverage
        ring = (signed > band) & (signed <= 2 * band)
        if ring.sum() < 10:
            ring = ~coverage
        window = int(4 * band) | 1
        colour_in = _local_colour(reference, deep, window)
        colour_out = _nearest_colour(reference, ring)
        mask = coverage.copy()
        if colour_in is not None and colour_out is not None:
            closer = np.linalg.norm(reference - colour_in, axis=2) < np.linalg.norm(
                reference - colour_out, axis=2
            )
            telling = np.linalg.norm(colour_in - colour_out, axis=2) > CONTRAST
            classify = near & telling
            mask[classify] = closer[classify]
            # Drop single-pixel specks.
            mask = ndimage.uniform_filter(mask.astype(np.float64), 3) > 0.5
        self.mask = ndimage.gaussian_filter(mask.astype(np.float64), 0.7)
        self.coverage = ndimage.gaussian_filter(coverage.astype(np.float64), 1.0)

    def sample(self, field, points):
        return ndimage.map_coordinates(
            field, [points[..., 1] - 0.5, points[..., 0] - 0.5], order=1, mode="nearest"
        )

    def outward(self, points, normals) -> float:
        """+1 when *normals* point out of the path, -1 when in, 0 unknown."""
        inner = self.sample(self.coverage, points - 2 * normals)
        outer = self.sample(self.coverage, points + 2 * normals)
        balance = float(np.sum(inner - outer))
        return float(np.sign(balance)) if abs(balance) > 0.1 * len(points) else 0.0

    def offsets(self, points, normals, reach) -> np.ndarray:
        """How far along each outward normal the edge is; NaN where none is."""
        steps = np.arange(-reach, reach + 1e-9, STEP)
        probe = points[:, None, :] + steps[None, :, None] * normals[:, None, :]
        value = self.sample(self.mask, probe) - 0.5
        found = np.full(len(points), np.nan)
        for k, row in enumerate(value):
            # Leaving the shape going outward.
            index = np.nonzero((row[:-1] > 0) & (row[1:] <= 0))[0]
            if index.size:
                at = steps[index] + row[index] / (row[index] - row[index + 1]) * STEP
                found[k] = at[np.argmin(np.abs(at))]
        # Off the crop there is nothing to see, only its border.
        edge = points + np.nan_to_num(found)[:, None] * normals
        height, width = self.mask.shape
        outside = (
            (edge[:, 0] < 0.5)
            | (edge[:, 1] < 0.5)
            | (edge[:, 0] > width - 0.5)
            | (edge[:, 1] > height - 0.5)
        )
        found[outside] = np.nan
        return found


@dataclass
class _Samples:
    t: np.ndarray
    points: np.ndarray
    normals: np.ndarray
    offsets: np.ndarray

    @property
    def valid(self) -> np.ndarray:
        return ~np.isnan(self.offsets)

    def edge(self) -> np.ndarray:
        """The edge points found, in order along the segment."""
        ok = self.valid
        return self.points[ok] + self.offsets[ok, None] * self.normals[ok]


class _Snapper:
    def __init__(self, frame, reference, rule, fixed: Frozen, detail: bool):
        self.frame = frame
        self.reference = reference
        self.size = (reference.shape[1], reference.shape[0])
        self.rule = rule
        self.fixed = fixed
        self.detail = detail
        self.band = MIN_BAND

    def run(self, geometry: Geometry) -> Geometry:
        contours = [
            _Contour(
                [
                    _Node(
                        n.id,
                        n.command,
                        self.frame.pixels(n.values),
                        n.id in self.fixed.nodes,
                        n.id in self.fixed.endpoints,
                        n,
                    )
                    for n in s.nodes
                ],
                s.closed,
            )
            for s in geometry.subpaths
        ]
        every = np.vstack([n.points for c in contours for n in c.nodes])
        extent = float(np.max(every.max(axis=0) - every.min(axis=0)))
        coverage = self._coverage(contours)
        if not coverage.any():
            return geometry
        # No wider than the shape is thick, or one side's band reaches the
        # other side's edge.
        thickness = float(np.max(distance_transform_edt(coverage)))
        self.band = max(MIN_BAND, min(BAND * extent, thickness))
        self._settle(contours)
        if self.detail:
            self._add_detail(contours)
        return self._geometry(geometry, contours)

    def _settle(self, contours: list[_Contour]) -> None:
        """Move points and refit handles until the points stop moving."""
        for _ in range(PASSES):
            target = self._target(contours)
            moved = max((self._move_points(c, target) for c in contours), default=0.0)
            for contour in contours:
                self._refit(contour, target)
            if moved < SETTLED:
                break
        # Only once the corners are in place can a line tell it should bow.
        target = self._target(contours)
        for contour in contours:
            self._refit(contour, target, bend=True)

    def _coverage(self, contours):
        return _coverage(contours, self.size, self.rule)

    def _target(self, contours) -> _Target:
        return _Target(self.reference, self._coverage(contours), self.band)

    def _sample(self, control, target, direction, count=None) -> _Samples:
        length = float(np.sum(np.linalg.norm(np.diff(control, axis=0), axis=1)))
        count = count or int(np.clip(length / 1.5, 4, 64))
        t = (np.arange(count) + 0.5) / count
        points, tangents = _bezier(control, t)
        norm = np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-12)
        normals = direction * np.column_stack([tangents[:, 1], -tangents[:, 0]]) / norm
        return _Samples(t, points, normals, target.offsets(points, normals, self.band))

    def _direction(self, contour, target) -> float:
        points, normals = [], []
        for _, control in _segments(contour):
            t = np.linspace(0.1, 0.9, 5)
            p, tangent = _bezier(control, t)
            norm = np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
            points.append(p)
            normals.append(np.column_stack([tangent[:, 1], -tangent[:, 0]]) / norm)
        if not points:
            return 0.0
        return target.outward(np.vstack(points), np.vstack(normals))

    def _move_points(self, contour: _Contour, target: _Target) -> float:
        """Move each free point onto the edge; the furthest move, in pixels."""
        direction = self._direction(contour, target)
        if not direction:
            return 0.0
        nodes = contour.nodes
        segments = _segments(contour)
        # Per node: the edge's offset where each of its segments meets it,
        # along that segment's normal there, and how much to trust it.
        constraints: dict[int, list[tuple[np.ndarray, float, float]]] = {}
        for end, control in segments:
            samples = self._sample(control, target, direction)
            ok = samples.valid
            if ok.sum() < 3:
                continue
            fit = _robust_line(samples.t[ok], samples.offsets[ok])
            # Never beyond what was seen: a curve's offsets extrapolate badly.
            low, high = np.min(samples.offsets[ok]), np.max(samples.offsets[ok])
            for index, at in (((end - 1) % len(nodes), 0.0), (end, 1.0)):
                _, tangent = _bezier(control, np.array([at]))
                normal = direction * np.array([tangent[0, 1], -tangent[0, 0]])
                size = np.linalg.norm(normal)
                if size < 1e-9:
                    continue
                weight = ok.mean()
                constraints.setdefault(index, []).append(
                    (
                        normal / size,
                        float(np.clip(fit[0] + fit[1] * at, low, high)),
                        weight,
                    )
                )
        # A closed contour whose last node sits on its first moves them together.
        twin = (
            contour.closed
            and len(nodes) > 1
            and np.linalg.norm(nodes[-1].end - nodes[0].end) <= 1e-6
        )
        if twin:
            merged = constraints.get(0, []) + constraints.get(len(nodes) - 1, [])
            constraints[0] = constraints[len(nodes) - 1] = merged
        furthest = 0.0
        shifts = {}
        for index, rows in constraints.items():
            node = nodes[index]
            if node.fixed or node.pinned:
                continue
            system = 0.1 * np.eye(2)
            rhs = np.zeros(2)
            for normal, offset, weight in rows:
                system += weight * np.outer(normal, normal)
                rhs += weight * normal * offset
            shift = np.linalg.solve(system, rhs)
            away = node.end + shift - node.origin
            size = np.linalg.norm(away)
            if size > REACH * self.band:
                shift = node.origin + away * (REACH * self.band / size) - node.end
            shifts[index] = shift
        if twin and (
            nodes[0].fixed or nodes[0].pinned or nodes[-1].fixed or nodes[-1].pinned
        ):
            shifts.pop(0, None)
            shifts.pop(len(nodes) - 1, None)
        for index, shift in shifts.items():
            self._shift(contour, index, shift)
            furthest = max(furthest, float(np.linalg.norm(shift)))
        return furthest

    def _shift(self, contour: _Contour, index: int, shift: np.ndarray) -> None:
        """Move an endpoint, taking its neighbouring handles along."""
        nodes = contour.nodes
        node = nodes[index]
        node.points = node.points.copy()
        node.points[-1] += shift
        if node.command == "C":
            node.points[1] += shift
        if index + 1 < len(nodes):
            after = nodes[index + 1]
            if after.command == "C" and not after.fixed:
                after.points = after.points.copy()
                after.points[0] += shift

    def _refit(self, contour: _Contour, target: _Target, bend=False) -> None:
        """Fit each segment's handles to the edge between its ends.

        With *bend*, a line alongside a clearly curved edge becomes a cubic.
        """
        direction = self._direction(contour, target)
        if not direction:
            return
        for end, control in _segments(contour):
            if end == 0:
                continue
            node = contour.nodes[end]
            if node.fixed:
                continue
            samples = self._sample(control, target, direction)
            ok = samples.valid
            if ok.sum() < max(4, 0.5 * len(ok)):
                continue
            if node.command == "C" or (bend and _curved(samples, control)):
                fitted = _cubic_through(
                    control[0], control[-1], samples.t[ok], samples.edge()
                )
                if fitted is not None:
                    node.command = "C"
                    node.points = fitted

    def _add_detail(self, contours: list[_Contour]) -> None:
        """Split segments where the edge is still far from them.

        After each round of splits the points settle again, so the corners
        the new points make find their place.
        """
        limit = 2 * sum(len(c.nodes) for c in contours)
        for _ in range(SPLIT_PASSES):
            target = self._target(contours)
            split = False
            for contour in contours:
                direction = self._direction(contour, target)
                if not direction:
                    continue
                for end, control in reversed(_segments(contour)):
                    if sum(len(c.nodes) for c in contours) >= limit:
                        break
                    if not contour.nodes[end].fixed:
                        split |= self._split(contour, end, control, target, direction)
            if not split:
                return
            self._settle(contours)

    def _split(self, contour, end, control, target, direction) -> bool:
        length = float(np.sum(np.linalg.norm(np.diff(control, axis=0), axis=1)))
        count = int(np.clip(length, 8, 128))
        samples = self._sample(control, target, direction, count)
        ok = samples.valid
        if ok.sum() < max(4, 0.5 * count):
            return False
        error = np.where(ok, np.abs(samples.offsets), 0.0)
        # Not so near an end that a sliver is left.
        room = (samples.t > 0.1) & (samples.t < 0.9)
        error[~room] = 0.0
        k = int(np.argmax(error))
        if error[k] <= SPLIT_ERROR or length * min(samples.t[k], 1 - samples.t[k]) < 2:
            return False
        middle = samples.points[k] + samples.offsets[k] * samples.normals[k]
        before = ok & (np.arange(count) < k)
        after = ok & (np.arange(count) > k)
        edge = (
            samples.points + np.nan_to_num(samples.offsets)[:, None] * samples.normals
        )
        t, at = samples.t, samples.t[k]
        node = contour.nodes[end]
        if node.command == "C":
            # Each half follows the edge where it can, or the curve it was.
            first, second = _halves(control, at, middle)
            fitted = _cubic_through(control[0], middle, t[before] / at, edge[before])
            first = first if fitted is None else fitted
            fitted = _cubic_through(
                middle, control[-1], (t[after] - at) / (1 - at), edge[after]
            )
            second = second if fitted is None else fitted
            new = _Node(new_id("node"), "C", first, False, False, None)
            node.points = second
        else:
            new = _Node(new_id("node"), "L", middle[None].copy(), False, False, None)
        if end == 0:
            contour.nodes.append(new)
        else:
            contour.nodes.insert(end, new)
        return True

    def _geometry(self, geometry: Geometry, contours: list[_Contour]) -> Geometry:
        subpaths = []
        for subpath, contour in zip(geometry.subpaths, contours, strict=True):
            nodes = []
            for node in contour.nodes:
                values = self.frame.local(node.points)
                original = node.original
                if original is None:
                    nodes.append(PathNode(node.id, node.command, values))
                    continue
                if original.command == node.command and np.allclose(
                    self.frame.pixels(original.values), node.points, atol=1e-9
                ):
                    nodes.append(original)
                else:
                    nodes.append(replace(original, command=node.command, values=values))
            subpaths.append(replace(subpath, nodes=tuple(nodes)))
        return replace(geometry, subpaths=tuple(subpaths))


def _robust_line(t: np.ndarray, offsets: np.ndarray) -> tuple[float, float]:
    """offset = a + b t, fitted with outliers weighed down."""
    weight = np.ones_like(t)
    design = np.column_stack([np.ones_like(t), t])
    a = b = 0.0
    for _ in range(3):
        root = np.sqrt(weight)
        (a, b), *_ = np.linalg.lstsq(design * root[:, None], offsets * root, rcond=None)
        residual = np.abs(offsets - a - b * t)
        weight = 1 / np.maximum(1.0, residual / 1.5)
    return float(a), float(b)


def _curved(samples: _Samples, control: np.ndarray) -> bool:
    """Whether the edge alongside a line clearly bows away from it."""
    offsets = samples.offsets[samples.valid]
    length = float(np.linalg.norm(control[-1] - control[0]))
    # The bow, once a tilt of the whole line is taken out.
    a, b = _robust_line(samples.t[samples.valid], offsets)
    bow = offsets - a - b * samples.t[samples.valid]
    middle = np.abs(samples.t[samples.valid] - 0.5) < 0.25
    if not middle.any():
        return False
    depth = float(np.median(bow[middle]))
    return abs(depth) > max(1.5, 0.04 * length)


def _cubic_through(start, end, t, edge) -> np.ndarray | None:
    """Cubic controls (c1, c2, end) from *start* to *end* following *edge*.

    *t* are the edge points' parameters along the segment they were found
    from, a far better start than spacing them evenly when some are missing.
    A few Newton steps then move each parameter to its point's closest spot
    on the curve before the final least-squares fit. None when the points do
    not pin a curve down, or the curve that fits them runs wild.
    """
    if len(edge) < 3 or np.ptp(t) < 0.5:
        return None
    t = np.clip(t, 0.0, 1.0)
    for step in range(4):
        u = 1 - t
        basis = np.column_stack([3 * u**2 * t, 3 * u * t**2])
        base = u[:, None] ** 3 * start + t[:, None] ** 3 * end
        (c1, c2), *_ = np.linalg.lstsq(basis, edge - base, rcond=None)
        points, first = _bezier(np.vstack([start, c1, c2, end]), t)
        if step == 3:
            break
        second = 6 * u[:, None] * (c2 - 2 * c1 + start) + 6 * t[:, None] * (
            end - 2 * c2 + c1
        )
        miss = points - edge
        slope = np.sum(first * first, axis=1) + np.sum(miss * second, axis=1)
        change = np.sum(miss * first, axis=1) / np.where(
            np.abs(slope) > 1e-9, slope, np.inf
        )
        t = np.sort(np.clip(t - change, 0.0, 1.0))
    trail = np.vstack([start, edge, end])
    length = float(np.sum(np.linalg.norm(np.diff(trail, axis=0), axis=1)))
    miss = float(np.max(np.linalg.norm(points - edge, axis=1)))
    if (
        np.linalg.norm(c1 - start) > length
        or np.linalg.norm(c2 - end) > length
        or miss > max(2.0, 0.1 * length)
    ):
        return None
    return np.vstack([c1, c2, end])


def _halves(control, at, middle) -> tuple[np.ndarray, np.ndarray]:
    """A cubic split at *at* (de Casteljau), the split point moved to *middle*."""
    a, b, c, d = control
    ab, bc, cd = a + (b - a) * at, b + (c - b) * at, c + (d - c) * at
    left, right = ab + (bc - ab) * at, bc + (cd - bc) * at
    shift = middle - (left + (right - left) * at)
    return np.vstack([ab, left + shift, middle]), np.vstack([right + shift, cd, d])


def snap(
    document: Document,
    paths: Paths,
    region: Region,
    fixed: Frozen,
    *,
    detail: bool = False,
    long_side: int = LONG_SIDE,
) -> Paths:
    """*paths* with each filled path's points moved onto the reference's edges.

    *region* is the reference crop around the paths. Stroke-only paths are
    left as they are. With *detail*, segments may be split where one curve
    cannot follow the edge; otherwise every path keeps its nodes.
    """
    image = resize_long_side(region.image.convert("RGB"), long_side)
    reference = np.asarray(image, dtype=np.float64) / 255
    geometries = dict(paths.geometries)
    for oid, geometry in paths.geometries.items():
        style = path_style(document, document.element(oid))
        if style["fill"] == "none":
            continue
        frame = _frame(document, oid, region, image.size)
        if frame is None:
            continue
        snapper = _Snapper(frame, reference, style["fill-rule"], fixed, detail)
        geometries[oid] = snapper.run(geometry)
    return replace(paths, geometries=geometries)
