"""Snap to the reference: move the selected paths' points onto its edges.

For each free point, the reference is read along the outline's normal, a few
pixels either way. The colour just inside the path there and the colour just
outside it are taken from that same line, and the edge is where the line
stops looking like the inside and starts looking like the outside, nearest
to where the point already is. The point moves there and takes its two
handles along, so the curve keeps its shape. Segment middles are then pulled
onto the edge the same way by their handles.

The reach is small on purpose: the job is to put an outline on the edge that
is right there, not to hunt for one, since the nearest edge further away is
as often a neighbouring shape's. A pass that would make an outline cross
itself is scaled back until it does not. With detail on, points are added
where the path and the reference disagree most: each large blob of pixels
the path misses or covers wrongly gets new points on the segment beside it,
one, two or a spike of three, kept when they pay for themselves, and the
passes run again.

All of it works in the reference's pixels, so steps and tolerances are the
same for a transformed or tiny path. Pinned endpoints stay.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field, replace

import cairosvg
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import KDTree

from vectrify.document import Document, Geometry, PathNode
from vectrify.document.join import path_style
from vectrify.document.model import new_id
from vectrify.document.transforms import root_matrix
from vectrify.image_utils import resize_long_side
from vectrify.operations.generate import Region
from vectrify.refine.crossings import bezier as _bezier
from vectrify.refine.crossings import contour_line, polyline_crossings
from vectrify.refine.frozen import Frozen, Paths

# How far a point looks for the edge either way, in reference pixels, and
# the share of the path's size that caps it for small paths.
REACH = 6.0
REACH_SHARE = 0.05
# Reference crops larger than this are shrunk first.
LONG_SIDE = 512
PASSES = 6
# How far a point may travel in all, in reaches.
TRAVEL = 3.0
# Stop once no point moves further than this, in pixels.
SETTLED = 0.2
# Inside and outside colours closer than this (RGB, 0-1) cannot tell the
# edge apart, so the point stays.
CONTRAST = 0.08
# Detail: how many pixels each added point has to fix, how many of the
# largest wrong blobs each round looks at, and how finely each segment is
# sampled to find the one a blob sits on. The path grows to at most
# SPLIT_GROWTH times its points, or SPLIT_LEAST more if that is more.
SPLIT_GAIN = 12
SPLIT_BLOBS = 8
SAMPLES = 32
# How far out, in reaches, detail also aims into a blob, besides its far end.
CREEP = (1.0, 2.0, 4.0)
# How close to a segment's end, in t, a blob is taken to sit at that point.
AT_NODE = 0.05
# Where, in t along the segments either side, a moved point is held.
HOLD = (0.1, 0.4)
SPLIT_GROWTH = 2.0
SPLIT_LEAST = 16
# The most tries detail scores in one call, over all its rounds.
SPLIT_TRIES = 160
# How finely the normal is read, in pixels.
STEP = 0.5
# How many points each segment is drawn with when checking for crossings.
CROSSING_SAMPLES = 8
# Detail's tries are scored on outlines drawn as straight edges this long, in
# pixels, with at most so many to a curve.
FILL_CHORD = 2.0
FILL_SAMPLES = 32


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
    matrix = root_matrix(document, oid)
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
    # A pinned endpoint stays; its handles may still move.
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


def _coverage(contours: list[_Contour], size, rule: str, window=None) -> np.ndarray:
    """Which pixels the contours fill, over the whole crop or a *window* of it."""
    x0, y0, x1, y1 = window or (0, 0, size[0], size[1])
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{x1 - x0}" '
        f'height="{y1 - y0}" viewBox="{x0} {y0} {x1 - x0} {y1 - y0}">'
        f'<path d="{_path_data(contours)}" fill="#000" '
        f'fill-rule="{rule}"/></svg>'
    )
    png = cairosvg.svg2png(bytestring=svg.encode(), background_color="white")
    assert png is not None
    with Image.open(io.BytesIO(png)) as image:
        return np.asarray(image.convert("L")) < 128


def _edges(contours: list[_Contour]) -> np.ndarray:
    """The contours as straight edges (n, 2, 2), curves flattened, each closed
    as a fill closes it."""
    lines, cubics = [], []
    for contour in contours:
        segments = _segments(contour)
        if not segments:
            continue
        for _end, control in segments:
            (cubics if len(control) == 4 else lines).append(control)
        first, last = segments[0][1][0], segments[-1][1][-1]
        if np.linalg.norm(last - first) > 1e-9:
            lines.append(np.vstack([last, first]))
    parts = [np.asarray(lines, dtype=np.float64).reshape(-1, 2, 2)]
    if cubics:
        control = np.asarray(cubics, dtype=np.float64)
        hull = np.linalg.norm(np.diff(control, axis=1), axis=2).sum(axis=1).max()
        count = int(np.clip(np.ceil(hull / FILL_CHORD), 2, FILL_SAMPLES))
        t = np.linspace(0, 1, count + 1)[:, None]
        u = 1 - t
        basis = np.hstack([u**3, 3 * u**2 * t, 3 * u * t**2, t**3])
        points = np.einsum("sk,nkc->nsc", basis, control)
        parts.append(
            np.stack([points[:, :-1], points[:, 1:]], axis=2).reshape(-1, 2, 2)
        )
    return np.concatenate(parts)


def _fill(contours: list[_Contour], rule: str, window) -> np.ndarray:
    """Which pixels of *window* the contours fill, judged at pixel centres.

    What _coverage gives, without drawing and decoding an image: each row's
    centre line is crossed with the outline's edges, and the winding counted
    from the left.
    """
    x0, y0, x1, y1 = window
    width, height = x1 - x0, y1 - y0
    edges = _edges(contours)
    a, b = edges[:, 0], edges[:, 1]
    low, high = np.minimum(a[:, 1], b[:, 1]), np.maximum(a[:, 1], b[:, 1])
    # Only edges that cross a row of the window and start left of its right
    # side can change the winding inside it.
    keep = (high > y0) & (low < y1) & (low < high)
    keep &= np.minimum(a[:, 0], b[:, 0]) < x1
    a, b, low, high = a[keep], b[keep], low[keep], high[keep]
    # The rows whose centre, at y0 + row + 0.5, lies in [low, high).
    first = np.maximum(0, np.ceil(low - y0 - 0.5)).astype(int)
    last = np.minimum(height - 1, np.ceil(high - y0 - 0.5).astype(int) - 1)
    counts = np.maximum(0, last - first + 1)
    edge = np.repeat(np.arange(len(a)), counts)
    rows = (
        first[edge]
        + np.arange(len(edge))
        - np.repeat(np.cumsum(counts) - counts, counts)
    )
    a, b = a[edge], b[edge]
    y = y0 + rows + 0.5
    x = a[:, 0] + (y - a[:, 1]) / (b[:, 1] - a[:, 1]) * (b[:, 0] - a[:, 0])
    # A crossing counts for the pixel centres to its right.
    columns = np.clip(np.floor(x - x0 - 0.5).astype(int) + 1, 0, width)
    direction = np.where(b[:, 1] > a[:, 1], 1.0, -1.0)
    if rule == "evenodd":
        direction = np.ones_like(direction)
    winding = np.bincount(
        rows * (width + 1) + columns, weights=direction, minlength=height * (width + 1)
    ).reshape(height, width + 1)
    winding = np.cumsum(winding, axis=1)[:, :width].round().astype(int)
    return winding % 2 == 1 if rule == "evenodd" else winding != 0


def _unit(vector: np.ndarray) -> np.ndarray | None:
    size = float(np.linalg.norm(vector))
    return None if size < 1e-9 else vector / size


class _Snapper:
    def __init__(
        self,
        frame,
        reference,
        rule,
        fixed: Frozen,
        detail: bool,
        split_gain: float,
        deadline: float = float("inf"),
        tries: int = SPLIT_TRIES,
    ):
        self.frame = frame
        self.reference = ndimage.gaussian_filter(reference, (0.6, 0.6, 0))
        self.size = (reference.shape[1], reference.shape[0])
        self.rule = rule
        self.fixed = fixed
        self.detail = detail
        self.split_gain = split_gain
        self.reach = REACH
        # When to stop (time.monotonic()), and how many tries detail has left.
        self.deadline = deadline
        self.tries = tries

    def run(self, geometry: Geometry) -> Geometry:
        contours = [
            _Contour(
                [
                    _Node(
                        n.id,
                        n.command,
                        self.frame.pixels(n.values),
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
        self.reach = max(2.0, min(REACH, REACH_SHARE * extent))
        start = sum(len(c.nodes) for c in contours)
        for _ in range(PASSES):
            if self._pass(contours) < SETTLED or self._late():
                break
        if self.detail:
            limit = max(SPLIT_GROWTH * start, start + SPLIT_LEAST)
            while sum(len(c.nodes) for c in contours) < limit and not self._spent():
                if not self._split(contours, limit):
                    break
                self._pass(contours)
        return self._geometry(geometry, contours)

    def _late(self) -> bool:
        return time.monotonic() >= self.deadline

    def _spent(self) -> bool:
        """Whether detail has used up its tries or the time."""
        return self.tries <= 0 or self._late()

    # Reading the reference.

    def _coverage(self, contours) -> np.ndarray:
        return _coverage(contours, self.size, self.rule)

    def _read(self, points: np.ndarray) -> np.ndarray:
        rows = points[..., 1] - 0.5
        columns = points[..., 0] - 0.5
        return np.stack(
            [
                ndimage.map_coordinates(
                    self.reference[..., c], [rows, columns], order=1, mode="nearest"
                )
                for c in range(3)
            ],
            axis=-1,
        )

    def _inside_colour(self, coverage) -> np.ndarray | None:
        """The reference's colour over the path's interior, away from its edge."""
        core = ndimage.binary_erosion(coverage, iterations=2)
        if core.sum() < 10:
            core = coverage
        if not core.any():
            return None
        return np.median(self.reference[core], axis=0)

    def _edge(self, point, normal, colour_in) -> float | None:
        """Offset along *normal* (pointing out) to the edge nearest *point*.

        Where the whole line still looks like the inside, the edge is further
        out than the reach, and the answer is a full step outward so the next
        pass can look again from there; likewise inward.
        """
        steps = np.arange(-self.reach, self.reach + 1e-9, STEP)
        probe = point[None, :] + steps[:, None] * normal[None, :]
        width, height = self.size
        # Near the crop's border the line is cut short where it leaves it.
        within = (
            (probe[:, 0] >= 0.5)
            & (probe[:, 1] >= 0.5)
            & (probe[:, 0] <= width - 0.5)
            & (probe[:, 1] <= height - 0.5)
        )
        centre = len(steps) // 2
        if not within[centre]:
            return None
        outside = np.nonzero(~within)[0]
        low = outside[outside < centre].max() + 1 if (outside < centre).any() else 0
        high = (
            outside[outside > centre].min() if (outside > centre).any() else len(steps)
        )
        steps, probe = steps[low:high], probe[low:high]
        if len(steps) < 4:
            return None
        colours = self._read(probe)
        distance = np.linalg.norm(colours - colour_in, axis=1)
        end = max(2, len(steps) // 5)
        colour_out = np.median(colours[-end:], axis=0)
        contrast = float(np.linalg.norm(colour_out - colour_in))
        inner_like = float(np.median(distance[:end])) < CONTRAST
        if contrast < CONTRAST:
            # The far end looks like the inside too: step out if the near end
            # agrees, since the shape then carries on past the reach.
            return self.reach if inner_like else None
        if not inner_like:
            # Even inside the outline the reference is not the shape's colour:
            # the outline sits outside the shape, so step in.
            return -self.reach
        # Positive where the reference looks like the inside.
        likeness = contrast / 2 - distance
        crossings = np.nonzero((likeness[:-1] > 0) & (likeness[1:] <= 0))[0]
        if not crossings.size:
            return None
        at = (
            steps[crossings]
            + likeness[crossings]
            / (likeness[crossings] - likeness[crossings + 1])
            * STEP
        )
        return float(at[np.argmin(np.abs(at))])

    def _outward(self, point, normal, coverage) -> np.ndarray | None:
        """*normal* turned to point out of the path, or None if unclear."""
        height, width = coverage.shape

        def filled(p):
            x, y = int(np.floor(p[0])), int(np.floor(p[1]))
            return 0 <= x < width and 0 <= y < height and bool(coverage[y, x])

        ahead, behind = filled(point + 1.5 * normal), filled(point - 1.5 * normal)
        if ahead == behind:
            return None
        return normal if behind else -normal

    # Moving points.

    def _pass(self, contours) -> float:
        """One round of moving points and segment middles; the furthest move."""
        coverage = self._coverage(contours)
        if not coverage.any():
            return 0.0
        before = [[n.points.copy() for n in c.nodes] for c in contours]
        crossings = sum(_crossings(c) for c in contours)
        colour_in = self._inside_colour(coverage)
        if colour_in is None:
            return 0.0
        moved = 0.0
        for contour in contours:
            moved = max(moved, self._move_points(contour, coverage, colour_in))
            self._move_middles(contour, coverage, colour_in)
        # Scale the whole pass back until no outline crosses itself more.
        scale = 1.0
        while sum(_crossings(c) for c in contours) > crossings and scale > 0.1:
            scale /= 2
            for contour, saved in zip(contours, before, strict=True):
                for node, old in zip(contour.nodes, saved, strict=True):
                    node.points = old + (node.points - old) * scale
        if sum(_crossings(c) for c in contours) > crossings:
            for contour, saved in zip(contours, before, strict=True):
                for node, old in zip(contour.nodes, saved, strict=True):
                    node.points = old
            return 0.0
        return moved * scale

    def _move_points(self, contour: _Contour, coverage, colour_in) -> float:
        nodes = contour.nodes
        count = len(nodes)
        closed = contour.closed and count > 2
        # A closed contour whose last node repeats its first has one point
        # there: it moves as node 0, whose neighbours are node 1 and the one
        # before the last.
        repeats = closed and np.linalg.norm(nodes[-1].end - nodes[0].end) < 1e-6
        ring = count - 1 if repeats else count
        offsets: dict[int, tuple[np.ndarray, float]] = {}
        for i, node in enumerate(nodes[:ring]):
            # Out of time, the points looked at so far still move.
            if self._late():
                break
            if node.pinned:
                continue
            if not closed and i in (0, count - 1):
                continue
            before = nodes[(i - 1) % ring].end
            after = nodes[(i + 1) % ring].end
            tangent = _unit(after - before)
            if tangent is None:
                continue
            normal = self._outward(
                node.end, np.array([tangent[1], -tangent[0]]), coverage
            )
            if normal is None:
                continue
            found = self._edge(node.end, normal, colour_in)
            if found is not None and abs(found) > 1e-3:
                offsets[i] = (normal, found)
        # A lone point jumping far from both neighbours is more likely to have
        # found a different edge than to be right.
        moved = 0.0
        for i, (normal, found) in offsets.items():
            near = [
                offsets[j][1] for j in ((i - 1) % ring, (i + 1) % ring) if j in offsets
            ]
            if near and all(abs(found - other) > self.reach / 2 for other in near):
                found = float(np.median([found, *near]))
            delta = normal * found
            # Never further than a few reaches from where the point started.
            travel = nodes[i].end + delta - nodes[i].origin
            limit = TRAVEL * self.reach
            if np.linalg.norm(travel) > limit:
                delta = (
                    nodes[i].origin
                    + travel * (limit / np.linalg.norm(travel))
                    - nodes[i].end
                )
            self._shift_point(contour, i, delta)
            moved = max(moved, abs(found))
        return moved

    def _shift_point(self, contour: _Contour, i: int, delta: np.ndarray) -> None:
        """Move node *i*'s endpoint and both handles beside it by *delta*."""
        nodes = contour.nodes
        node = nodes[i]
        points = node.points.copy()
        points[-1] += delta
        if node.command == "C":
            points[1] += delta
        node.points = points
        following = (i + 1) % len(nodes)
        if (i + 1 < len(nodes) or contour.closed) and following != i:
            nxt = nodes[following]
            if nxt.command == "C":
                shifted = nxt.points.copy()
                shifted[0] += delta
                nxt.points = shifted
        # A closed contour whose last node repeats its first moves them as one.
        if contour.closed and i == 0 and len(nodes) > 1:
            last = nodes[-1]
            if np.linalg.norm(last.end - (node.end - delta)) < 1e-6:
                shifted = last.points.copy()
                shifted[-1] += delta
                last.points = shifted

    def _middle(self, control, coverage, colour_in):
        """The edge's offset from the segment's middle, along its normal."""
        point, tangent = _bezier(control, np.array([0.5]))
        unit = _unit(tangent[0])
        if unit is None:
            return None, None
        normal = self._outward(point[0], np.array([unit[1], -unit[0]]), coverage)
        if normal is None:
            return None, None
        return self._edge(point[0], normal, colour_in), normal

    def _move_middles(self, contour: _Contour, coverage, colour_in) -> None:
        for end, control in _segments(contour):
            if self._late():
                break
            node = contour.nodes[end]
            if node.command != "C" or len(control) != 4:
                continue
            found, normal = self._middle(control, coverage, colour_in)
            if found is None or normal is None or abs(found) < 0.25:
                continue
            # Both handles moved by d move the curve's middle by 3/4 d.
            points = node.points.copy()
            points[0] += normal * found * 4 / 3
            points[1] += normal * found * 4 / 3
            node.points = points

    def _split(self, contours, limit: float) -> bool:
        """Add points where the path misses a piece of the shape, or covers too much.

        The pixels where path and reference disagree are grouped into blobs,
        largest first. For each, the segment nearest to it is split there: a
        point at the blob's far end, two spanning it, or a spike of three whose
        base points stay on the outline so the rest of the contour keeps its
        shape. A try is kept when each point it adds fixes *split_gain* pixels.
        """
        coverage = self._coverage(contours)
        colour_in = self._inside_colour(coverage)
        if colour_in is None:
            return False
        inside = self._inside(coverage, colour_in)
        if inside is None:
            return False
        wrong = inside != coverage
        # One pixel slivers along the outline are antialiasing, not shape.
        wrong = ndimage.binary_opening(wrong)
        labels, count = ndimage.label(wrong)
        if not count:
            return False
        areas = ndimage.sum(wrong, labels, range(1, count + 1))
        order = np.argsort(-areas)[:SPLIT_BLOBS]
        samples = self._samples(contours)
        points_now = sum(len(c.nodes) for c in contours)
        done: set[int] = set()
        for index in order:
            if areas[index] < self.split_gain or points_now >= limit or self._spent():
                break
            ys, xs = np.nonzero(labels == index + 1)
            blob = np.column_stack([xs, ys]) + 0.5
            depth = np.asarray(ndimage.distance_transform_edt(labels == index + 1))
            centre = depth[ys, xs]
            window = self._window(blob)
            if window is None:
                continue
            truth = _crop(inside, window)
            base = int((_fill(contours, self.rule, window) != truth).sum())
            best: tuple[float, int, int, _Contour] | None = None
            for c, trials in self._targets(blob, centre, samples, contours, done):
                for added, trial in trials:
                    if self._spent():
                        break
                    if points_now + added > limit:
                        continue
                    self.tries -= 1
                    trying = [
                        trial if k == c else other for k, other in enumerate(contours)
                    ]
                    error = int((_fill(trying, self.rule, window) != truth).sum())
                    # Each new point has to pay for itself.
                    gain = (base - error) / max(added, 1)
                    # Moving a point adds none, so it only has to help.
                    needed = self.split_gain if added else self.split_gain / 4
                    if gain >= needed and (best is None or gain > best[0]):
                        best = (gain, added, c, trial)
            if best is None:
                continue
            # One split per contour per round: a split renumbers its nodes,
            # and the passes after settle the rest.
            _gain, added, c, trial = best
            contours[c] = trial
            done.add(c)
            points_now += added
        return bool(done)

    def _samples(self, contours):
        """Points along every free segment: (contour, end node, t, point) rows."""
        t = np.linspace(0, 1, SAMPLES + 1)
        rows = []
        for c, contour in enumerate(contours):
            for end, control in _segments(contour):
                points, _ = _bezier(control, t)
                rows.append((c, end, control, t, points))
        return rows

    def _targets(self, blob, centre, samples, contours, done):
        """(contour, its tries) for each way of reaching into a blob.

        Each aim is a pixel of the blob, reached from the nearest point of the
        outline: by splitting the segment there, or, where that point is one
        of the path's, by moving it out and holding the outline either side.
        The blob's furthest pixel is one aim. A strand that curls away is not
        reached by a straight spike to its end, so the others creep up on it:
        the middle of the blob a few reaches out, from where the next round
        can reach further.
        """
        if not samples:
            return
        every = np.vstack([points for *_rest, points in samples])
        owner = np.concatenate(
            [np.full(len(points), k) for k, (*_rest, points) in enumerate(samples)]
        )
        spots = np.concatenate([t for *_rest, t, _points in samples])
        distance, nearest = KDTree(every).query(blob)
        far = int(distance.argmax())
        aims = [far]
        for steps in CREEP:
            out = steps * self.reach
            band = np.abs(distance - out) <= 1.0
            if out < distance[far] - self.reach and band.any():
                aims.append(int(np.flatnonzero(band)[centre[band].argmax()]))
        for aim in aims:
            k = int(owner[nearest[aim]])
            c, end, control, _t, _points = samples[k]
            if c in done:
                continue
            # The blob's footprint on that segment: where its near pixels touch.
            mine = owner[nearest] == k
            touching = mine & (distance <= max(2.0, 0.25 * distance[aim]))
            if not touching.any():
                touching = mine
            at = spots[nearest[touching]]
            tip = float(spots[nearest[aim]])
            point, _tangent = _bezier(control, np.array([tip]))
            delta = blob[aim] - point[0]
            if tip >= 1 - AT_NODE or tip <= AT_NODE:
                count = len(contours[c].nodes)
                node = end if tip >= 1 - AT_NODE else (end - 1) % count
                yield c, self._extended(contours[c], node, delta)
                continue
            normal = _unit(delta)
            if normal is None:
                continue
            low = max(min(float(at.min()), tip - 0.02), 0.01)
            high = min(max(float(at.max()), tip + 0.02), 0.99)
            yield (
                c,
                _targeted(
                    contours[c],
                    end,
                    control,
                    low,
                    tip,
                    high,
                    float(distance[aim]),
                    normal,
                ),
            )

    def _extended(self, contour: _Contour, node: int, delta):
        """(points added, *contour* with point *node* moved by *delta*), alone
        and with new points either side keeping the outline beside it."""
        nodes = contour.nodes
        if (
            contour.closed
            and node == len(nodes) - 1
            and np.linalg.norm(nodes[-1].end - nodes[0].end) < 1e-6
        ):
            node = 0
        if nodes[node].pinned:
            return
        contour = _copy(contour)
        # Lines either side become curves the passes can bend.
        _curved(contour, node)
        _curved(contour, (node + 1) % len(nodes))
        for share in (0.5, 1.0):
            moved = _copy(contour)
            self._shift_point(moved, node, delta * share)
            yield 0, moved
            # Held far back the whole spike swings; held close, a thin
            # extension grows from its tip.
            for hold in HOLD:
                held = _copy(contour)
                at = node
                incoming = dict(_segments(held)).get(node)
                if incoming is not None:
                    held = _split_segment(
                        held, node, incoming, [(1 - hold, np.zeros(2))]
                    )
                    at = node + 1 if node > 0 else node
                following = (at + 1) % len(held.nodes)
                outgoing = dict(_segments(held)).get(following)
                if outgoing is not None:
                    held = _split_segment(
                        held, following, outgoing, [(hold, np.zeros(2))]
                    )
                added = len(held.nodes) - len(nodes)
                if added:
                    self._shift_point(held, at, delta * share)
                    yield added, held

    def _inside(self, coverage, colour_in) -> np.ndarray | None:
        """Where the reference looks like the path's inside rather than around it."""
        distance = np.linalg.norm(self.reference - colour_in, axis=2)
        ring = (
            ndimage.binary_dilation(coverage, iterations=int(np.ceil(self.reach)))
            & ~coverage
        )
        if not ring.any():
            return None
        contrast = float(np.median(distance[ring]))
        if contrast < CONTRAST:
            return None
        return distance < contrast / 2

    def _window(self, control) -> tuple[int, int, int, int] | None:
        """The pixels around a segment, padded by twice the reach."""
        pad = 2 * self.reach
        low = np.floor(control.min(axis=0) - pad).astype(int)
        high = np.ceil(control.max(axis=0) + pad).astype(int)
        width, height = self.size
        x0, y0 = max(0, low[0]), max(0, low[1])
        x1, y1 = min(width, high[0]), min(height, high[1])
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None
        return int(x0), int(y0), int(x1), int(y1)

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


def _halves(control: np.ndarray, t: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """A cubic split at *t*: each part's c1, c2 and end."""
    a, b, c, d = control
    ab, bc, cd = a + (b - a) * t, b + (c - b) * t, c + (d - c) * t
    abc, bcd = ab + (bc - ab) * t, bc + (cd - bc) * t
    middle = abc + (bcd - abc) * t
    return np.vstack([ab, abc, middle]), np.vstack([bcd, cd, d])


def _crop(image: np.ndarray, window) -> np.ndarray:
    x0, y0, x1, y1 = window
    return image[y0:y1, x0:x1]


def _copy(contour: _Contour) -> _Contour:
    nodes = [replace(n, points=n.points.copy()) for n in contour.nodes]
    for new, old in zip(nodes, contour.nodes, strict=True):
        new.origin = old.origin
    return _Contour(nodes, contour.closed)


def _split_segment(
    contour: _Contour,
    end: int,
    control,
    at: list[tuple[float, np.ndarray]],
    sharp=False,
) -> _Contour:
    """A copy of *contour* with the segment ending at node *end* split.

    *at* lists (t, delta) in order: each new point sits at t on the segment
    and moves by delta, taking the handles beside it along. *sharp* draws the
    two segments beside each moved point straight, for a spike's tip.
    A line that gets a moved point becomes curves, still straight, so the
    passes after can bend them onto the edge.
    """
    nodes = _copy(contour).nodes
    if len(control) == 2 and any(np.any(delta) for _t, delta in at):
        control = _curve(control)
    pieces: list[np.ndarray] = []
    rest, done = control, 0.0
    for t, _delta in at:
        local = (t - done) / (1 - done)
        if len(control) == 4:
            first, second = _halves(rest, local)
            pieces.append(first)
            rest = np.vstack([first[-1], second])
        else:
            point = rest[0] + (rest[1] - rest[0]) * local
            pieces.append(point[None, :])
            rest = np.vstack([point, rest[1]])
        done = t
    pieces.append(rest[1:])
    for k, (_t, delta) in enumerate(at):
        pieces[k][-1] += delta
        if len(control) == 4:
            pieces[k][1] += delta
            pieces[k + 1][0] += delta
    if sharp and len(control) == 4:
        moved = {k for k, (_t, delta) in enumerate(at) if np.any(delta)}
        for k in sorted(moved | {k + 1 for k in moved}):
            begin = control[0] if k == 0 else pieces[k - 1][-1]
            pieces[k][0], pieces[k][1] = begin, pieces[k][-1]
    command = "C" if len(control) == 4 else "L"
    added = [
        _Node(new_id("node"), command, piece, False, None) for piece in pieces[:-1]
    ]
    if end > 0:
        if command == "C":
            nodes[end].command, nodes[end].points = "C", pieces[-1]
        nodes[end:end] = added
    else:
        # The closing line ends at node 0, so its new points go last; as a
        # curve its last piece needs a node of its own, back on the first.
        if command == "C":
            added.append(_Node(new_id("node"), "C", pieces[-1], False, None))
        nodes.extend(added)
    return _Contour(nodes, contour.closed)


def _curve(line: np.ndarray) -> np.ndarray:
    """A straight cubic along *line*, its handles at the thirds."""
    a, b = line
    return np.vstack([a, a + (b - a) / 3, a + 2 * (b - a) / 3, b])


def _curved(contour: _Contour, i: int) -> None:
    """Make the line ending at node *i* a straight cubic, in place."""
    node = contour.nodes[i]
    if i == 0 or node.command != "L":
        return
    start = contour.nodes[i - 1].end
    node.command, node.points = "C", _curve(np.vstack([start, node.end]))[1:]


def _targeted(contour, end, control, low, tip, high, depth, normal):
    """(points added, *contour* split) for each way of reaching a blob.

    The blob touches the segment from *low* to *high* (in t) and reaches
    *depth* pixels along *normal* at *tip*.
    """
    none = np.zeros(2)
    for share in (0.75, 1.0):
        reach = normal * depth * share
        yield 1, _split_segment(contour, end, control, [(tip, reach)])
        if high - low > 0.02:
            yield (
                2,
                _split_segment(contour, end, control, [(low, reach), (high, reach)]),
            )
        if low < tip < high:
            for sharp in (False, True):
                at = [(low, none), (tip, reach), (high, none)]
                yield 3, _split_segment(contour, end, control, at, sharp)


def _crossings(contour: _Contour) -> int:
    """How many times the outline crosses itself, drawn as a polyline."""
    controls = [control for _end, control in _segments(contour)]
    line, _segment = contour_line(controls, CROSSING_SAMPLES)
    return polyline_crossings(line, contour.closed)


def snap(
    document: Document,
    paths: Paths,
    region: Region,
    fixed: Frozen,
    *,
    detail: bool = False,
    split_gain: float = SPLIT_GAIN,
    long_side: int = LONG_SIDE,
    deadline: float = float("inf"),
    tries: int = SPLIT_TRIES,
) -> Paths:
    """*paths* with each filled path's points moved onto the reference's edges.

    *region* is the reference crop around the paths. Stroke-only paths are
    left as they are. With *detail*, segments may be split where one curve
    cannot follow the edge, scoring at most *tries* of them per path;
    otherwise every path keeps its nodes. Past *deadline* (time.monotonic())
    each path keeps how far it got.
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
        snapper = _Snapper(
            frame,
            reference,
            style["fill-rule"],
            fixed,
            detail,
            split_gain,
            deadline,
            tries,
        )
        geometries[oid] = snapper.run(geometry)
    return replace(paths, geometries=geometries)
