"""Cel tracer: flat regions bounded by the drawn lines, with the lines on top.

Cel and anime art is flat colour inside drawn outlines. The tracer finds the
lines first, fills the space between them with a shrinking ball so a small
gap in a line does not join the regions either side, splits each region by
colour where a shade edge has no line, and merges regions down to a target
count, sparing the boundaries a line runs along. The line pixels go to the
regions either side, so neighbours meet at the line's middle and share one
traced edge. The lines are thinned to centrelines and drawn over the fills as
strokes, one path per line colour (two when some are much bolder).

CPU only, with numpy and scipy.
"""

from __future__ import annotations

import heapq
import time
from itertools import pairwise
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    binary_propagation,
    center_of_mass,
    gaussian_filter,
    gaussian_filter1d,
    grey_closing,
    label,
    median,
    minimum_filter,
)

from vectrify.document.lines import contour_ends, end_pairs, joined
from vectrify.document.model import Geometry, PathNode, Subpath
from vectrify.refine.colour_regions import (
    boundary_chains,
    colour,
    fit_palette,
    nearest_indices,
    remove_fragments,
    simplified_indices,
    simplify,
)
from vectrify.refine.frozen import Frozen
from vectrify.refine.samvg import _fit_cubic, _simplified_data, mask_path
from vectrify.refine.simplify import simplified_geometry

LUMINANCE = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
# A line is darker than the surface either side by this much at its core, in
# 0-255 luminance, and its antialiased edge by a third of it.
LINE_CONTRAST = 12
# Line pieces smaller than this many pixels are specks, not lines, and holes
# in a line this small are closed.
LINE_SPECK = 20
LINE_HOLE = 12
# A piece taking less than this share of the light that the image's lines
# typically take is shading, not a line.
LINE_INK = 0.5
# How far, in pixels, a surface is followed into the notches it opens into.
NOTCH_REACH = 32
# The balls the free space is filled with, largest first, in pixels. A gap in
# a line narrower than a ball keeps it out: the largest closes the most.
BALLS = (6, 4, 2, 1)
# Colours the regions are split by, and the smallest split-off piece kept.
PALETTE = 12
PIECE = 24
# What merging across a drawn line costs, as a colour difference in 0-255 RGB
# along a boundary that a line runs all the way along.
LINE_PENALTY = 60.0
# Curves per this many traced pixels of a filled line shape before it is
# simplified, and the smoothing that takes the pixel staircase out of any
# traced run first, in pixels.
DENSITY = 6
SMOOTH = 1.0
# A line whose width varies more than this along it, its 90th percentile
# over its 10th, is tapered; when this share of the lines is, they are drawn
# as filled shapes rather than strokes. The strokes of one colour are split
# into thin and bold paths when their widths differ by more than this ratio.
WIDTH_SPREAD = 2.5
TAPERED_SHARE = 0.5
WIDTH_STEP = 1.6
# At most this many line colours, one stroked path each; closer colours, in
# 0-255 RGB, are one.
LINE_COLOURS = 3
LINE_COLOUR_DIFFERENCE = 40.0
# The longest gap, in line widths, bridged between two runs of one stroke
# that carry on from each other: a line the detection broke.
LINE_GAP = 1.5


def _disk(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return x * x + y * y <= radius * radius


def line_darkness(target: np.ndarray, radius: int) -> np.ndarray:
    """How much darker each pixel is than the surface around it: a black
    top-hat, which answers only on dark marks narrower than *radius* * 2."""
    luminance = target @ LUMINANCE
    return grey_closing(luminance, size=2 * radius + 1) - luminance


def detect_lines(target: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
    """The drawn lines, with their antialiased edges, and each pixel's
    darkness against the surface around it.

    Hysteresis keeps a faint pixel only when it joins a clearly dark one.
    Shading can be as narrow as a line, but it takes away less of the light
    beneath: a piece much fainter than the image's lines is left to the fills.
    A dark notch between two light spikes is as narrow as a line too, but it
    opens into a surface as dark as itself; it is given back to that surface.
    """
    darkness = line_darkness(target, radius)
    core = darkness >= LINE_CONTRAST
    mask = binary_propagation(core, mask=darkness >= LINE_CONTRAST / 3)
    mask = _without_notches(target @ LUMINANCE, mask)
    core &= mask
    pieces, count = label(mask, np.ones((3, 3)))
    if not count:
        return mask, darkness
    # How much of the surface's light a line takes: 1 for black on white.
    share = darkness / np.maximum(darkness + target @ LUMINANCE, 1)
    typical = float(np.percentile(share[core], 75))
    shares = np.zeros(count + 1)
    shares[1:] = median(share, np.where(core, pieces, 0), np.arange(1, count + 1))
    sizes = np.bincount(pieces.ravel(), minlength=count + 1)
    keep = (sizes >= LINE_SPECK) & (shares >= LINE_INK * typical)
    keep[0] = False
    mask = keep[pieces]
    # Pinholes in a line would thin into loops.
    holes, found = label(binary_fill_holes(mask) & ~mask)
    if found:
        small = np.bincount(holes.ravel()) <= LINE_HOLE
        small[0] = False
        mask |= small[holes]
    return mask, darkness


def _without_notches(luminance: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """*mask* less the pixels no darker than the surface they join.

    The surface grows into the marks pixel by pixel, taking each that is at
    most a little darker than the surface it grew from, and carrying that
    surface's brightness on: down a notch as dark as the surface it opens
    into, but only onto the faint rim of a line darker than both its sides.
    """
    mask = mask.copy()
    surface = np.where(mask, np.inf, luminance)
    for _ in range(NOTCH_REACH):
        beside = minimum_filter(surface, size=3)
        joined = mask & (luminance >= beside - LINE_CONTRAST / 2)
        if not joined.any():
            break
        surface[joined] = beside[joined]
        mask &= ~joined
    return mask


def trapped_ball_fill(free: np.ndarray, balls: tuple[int, ...] = BALLS) -> np.ndarray:
    """Regions of *free* that a ball can fill, 1 up; 0 where none reached.

    Each stage fills what the ball of its radius can reach without passing
    through a gap narrower than itself, grown back out by the radius. The
    ball shrinks stage by stage into the corners the larger one left, and a
    region too small for its stage is left to the next.
    """
    labels = np.zeros(free.shape, dtype=np.int32)
    count = 0
    for radius in balls:
        available = free & (labels == 0)
        disk = _disk(radius)
        core = binary_erosion(available, disk)
        if not core.any():
            continue
        cores, found = label(core)
        grown = binary_dilation(core, disk) & available
        # Each grown pixel belongs to the core it grew from, the nearest.
        grown_labels = np.where(grown, cores[nearest_indices(~core)], 0)
        sizes = np.bincount(grown_labels.ravel(), minlength=found + 1)
        keep = np.flatnonzero(sizes >= max(30, 10 * radius * radius))
        keep = keep[keep > 0]
        renumber = np.zeros(found + 1, dtype=np.int32)
        renumber[keep] = np.arange(count + 1, count + 1 + len(keep))
        count += len(keep)
        labels = np.where(labels == 0, renumber[grown_labels], labels)
    return labels


def thin(mask: np.ndarray) -> np.ndarray:
    """*mask* thinned to one-pixel centrelines (Zhang and Suen, 1984)."""
    image = np.pad(np.asarray(mask, dtype=bool), 1)
    while True:
        changed = False
        for step in (0, 1):
            p = _ring(image)
            neighbours = sum(p)
            # 0 -> 1 steps going round the ring: one where it is a line.
            crossings = sum(~a & b for a, b in pairwise((*p, p[0])))
            p2, _, p4, _, p6, _, p8, _ = p
            if step == 0:
                sides = ~(p2 & p4 & p6) & ~(p4 & p6 & p8)
            else:
                sides = ~(p2 & p4 & p8) & ~(p2 & p6 & p8)
            remove = (
                image[1:-1, 1:-1]
                & (neighbours >= 2)
                & (neighbours <= 6)
                & (crossings == 1)
                & sides
            )
            if remove.any():
                image[1:-1, 1:-1] &= ~remove
                changed = True
        if not changed:
            return image[1:-1, 1:-1]


def _ring(image: np.ndarray) -> tuple[np.ndarray, ...]:
    """The eight neighbours of each inner pixel of padded *image*, clockwise
    from above: N, NE, E, SE, S, SW, W, NW."""
    return (
        image[:-2, 1:-1],
        image[:-2, 2:],
        image[1:-1, 2:],
        image[2:, 2:],
        image[2:, 1:-1],
        image[2:, :-2],
        image[1:-1, :-2],
        image[:-2, :-2],
    )


def split_by_colour(
    target: np.ndarray, regions: np.ndarray, line: np.ndarray
) -> np.ndarray:
    """*regions* cut where the colour changes with no line, 0 still unset.

    A palette quantizes the surface; each region's pixels of one palette
    colour, connected, become a region of their own. Pieces too small to be
    a shape go to their neighbours.
    """
    fill = ~line & (regions > 0)
    if not fill.any():
        return regions
    smooth = gaussian_filter(target, (1, 1, 0))
    colours = min(PALETTE, int(fill.sum()))
    palette = np.zeros(regions.shape, dtype=np.int32)
    palette[fill] = fit_palette(smooth[fill][:, None, :], colours, 24, gpu=False)[:, 0]
    pieces = np.zeros(regions.shape, dtype=np.int64)
    count = 0
    for index in range(colours):
        found, added = label(fill & (palette == index))
        pieces = np.where(found > 0, found + count, pieces)
        count += added
    # A palette piece can span two regions that touch without a line between.
    _, split = np.unique(
        pieces * (int(regions.max()) + 1) + regions, return_inverse=True
    )
    split = split.reshape(regions.shape).astype(np.int32)
    split[~fill] = 0
    sizes = np.bincount(split.ravel())
    sizes[0] = 0
    kept = sizes[split] >= PIECE
    if not kept.any():
        return regions
    # A speck takes the region nearest it, most often the one around it.
    split = np.where(fill & ~kept, split[nearest_indices(~kept)], split)
    return np.where(fill, split, 0)


def merge_regions(
    labels: np.ndarray,
    target: np.ndarray,
    line: np.ndarray,
    count: int,
    penalty: float = LINE_PENALTY,
) -> np.ndarray:
    """*labels* merged greedily down to *count* regions, renumbered from 0.

    Merging two regions costs their colour difference squared, weighted by
    the smaller region's size (Ward), plus *penalty* squared for the share
    of their boundary a drawn line runs along: regions a line separates stay
    apart unless they are small or there is nothing else left to merge.
    """
    _, labels = np.unique(labels, return_inverse=True)
    labels = labels.reshape(line.shape)
    n = int(labels.max()) + 1
    if n <= count:
        return labels
    flat = labels.ravel()
    paint = ~line.ravel()
    area = np.bincount(flat[paint], minlength=n).astype(np.float64)
    sums = np.stack(
        [
            np.bincount(flat[paint], target[..., c].ravel()[paint], minlength=n)
            for c in range(3)
        ],
        1,
    )
    # A region all line has no paint of its own; its pixels stand in.
    bare = area == 0
    if bare.any():
        area_all = np.bincount(flat, minlength=n).astype(np.float64)
        sums_all = np.stack(
            [np.bincount(flat, target[..., c].ravel(), minlength=n) for c in range(3)],
            1,
        )
        area[bare] = area_all[bare]
        sums[bare] = sums_all[bare]
    area = list(area)
    sums = list(sums)
    first = np.concatenate((labels[:, :-1].ravel(), labels[:-1].ravel()))
    second = np.concatenate((labels[:, 1:].ravel(), labels[1:].ravel()))
    drawn = np.concatenate(
        ((line[:, :-1] | line[:, 1:]).ravel(), (line[:-1] | line[1:]).ravel())
    )
    apart = first != second
    first, second, drawn = first[apart], second[apart], drawn[apart]
    low, high = np.minimum(first, second), np.maximum(first, second)
    keys, inverse, length = np.unique(
        low.astype(np.int64) * n + high, return_inverse=True, return_counts=True
    )
    lined = np.bincount(inverse, drawn, minlength=len(keys))
    # Each region's neighbours: (boundary length, of it along a line).
    edges: list[dict[int, list[float]]] = [{} for _ in range(n)]
    for key, total, along in zip(
        keys.tolist(), length.tolist(), lined.tolist(), strict=True
    ):
        a, b = divmod(key, n)
        edges[a][b] = edges[b][a] = [total, along]

    def cost(a: int, b: int) -> float:
        total, along = edges[a][b]
        difference = sums[a] / area[a] - sums[b] / area[b]
        weight = area[a] * area[b] / (area[a] + area[b])
        return weight * (float(difference @ difference) + penalty**2 * along / total)

    version = [0] * n
    heap = [(cost(a, b), a, b, 0, 0) for a in range(n) for b in edges[a] if a < b]
    heapq.heapify(heap)
    parent = list(range(n))
    regions = n
    while regions > count and heap:
        _, a, b, va, vb = heapq.heappop(heap)
        if version[a] != va or version[b] != vb:
            continue
        # The larger keeps its number; the smaller's neighbours join it.
        if len(edges[a]) < len(edges[b]):
            a, b = b, a
        parent[b] = a
        area[a] += area[b]
        sums[a] = sums[a] + sums[b]
        del edges[a][b]
        for c, (total, along) in edges[b].items():
            if c == a:
                continue
            del edges[c][b]
            joined = edges[a].setdefault(c, [0.0, 0.0])
            joined[0] += total
            joined[1] += along
            edges[c][a] = joined
        edges[b] = {}
        version[a] += 1
        version[b] += 1
        regions -= 1
        for c in edges[a]:
            heapq.heappush(heap, (cost(a, c), a, c, version[a], version[c]))
    roots = np.array([_root(parent, i) for i in range(n)])
    _, renumbered = np.unique(roots, return_inverse=True)
    return renumbered[labels]


def _root(parent: list[int], index: int) -> int:
    while parent[index] != index:
        parent[index] = parent[parent[index]]
        index = parent[index]
    return index


def _smoothed(points: np.ndarray, sigma: float, closed: bool) -> np.ndarray:
    """*points* smoothed along their length, the ends of an open run kept."""
    if sigma <= 0 or len(points) < 5:
        return points
    if closed:
        return np.concatenate(
            (gaussian_filter1d(points[:-1], sigma, axis=0, mode="wrap"), points[:1])
        )
    smooth = gaussian_filter1d(points, sigma, axis=0, mode="nearest")
    smooth[[0, -1]] = points[[0, -1]]
    return smooth


def curve_nodes(
    points: np.ndarray, tolerance: float, *, smooth: float = SMOOTH
) -> list[tuple[str, tuple[float, ...]]]:
    """The run *points* as curves within *tolerance* pixels, after the start.

    The run is smoothed and cut at the points a polyline within half the
    tolerance needs; a cubic fitted to the run between each two follows it,
    so a corner stays sharp. Then it is simplified: each curve goes as far
    as it can without moving the outline more than the tolerance. The ends
    stay where they are, so runs that meet keep meeting.
    """
    closed = len(points) > 3 and np.array_equal(points[0], points[-1])
    points = _smoothed(np.asarray(points, dtype=np.float64), smooth, closed)
    kept = simplified_indices(points, tolerance / 2) if len(points) > 2 else [0, 1]
    if len(kept) <= 2 and len(points) <= 3:
        return [("L", tuple(float(v) for v in points[i])) for i in kept[1:]]
    nodes = [PathNode("n0", "M", tuple(float(v) for v in points[0]))]
    for i, (first, last) in enumerate(pairwise(kept), start=1):
        sample = points[first : last + 1]
        if len(sample) < 4:
            nodes.append(PathNode(f"n{i}", "L", tuple(float(v) for v in sample[-1])))
            continue
        a, b = _fit_cubic(sample, reparameterize=False)
        values = tuple(float(v) for v in (*a, *b, *sample[-1]))
        nodes.append(PathNode(f"n{i}", "C", values))
    geometry = simplified_geometry(
        Geometry("run", (Subpath("s", tuple(nodes)),)),
        Frozen(frozenset()),
        tolerance,
    )
    return [(node.command, node.values) for node in geometry.subpaths[0].nodes[1:]]


def _reversed(
    start: tuple[float, float], nodes: list[tuple[str, tuple[float, ...]]]
) -> list[tuple[str, tuple[float, ...]]]:
    """*nodes*, which run on from *start*, run backwards to it."""
    ends = [start, *(values[-2:] for _, values in nodes)]
    backwards = []
    for (command, values), end in zip(
        reversed(nodes), reversed(ends[:-1]), strict=True
    ):
        if command == "C":
            backwards.append(("C", (*values[2:4], *values[0:2], *end)))
        else:
            backwards.append(("L", tuple(end)))
    return backwards


def _data(start, nodes, closed: bool) -> str:
    parts = [f"M{start[0]:.2f} {start[1]:.2f}"]
    for command, values in nodes:
        parts.append(command + " ".join(f"{v:.2f}" for v in values))
    return " ".join(parts) + (" Z" if closed else "")


def region_outlines(labels: np.ndarray, tolerance: float) -> dict[int, str]:
    """Each region's outline as path data, traced once per shared edge.

    Every edge between two regions is fitted once and used, forwards and
    backwards, by both, so neighbours meet exactly. Edges on the canvas
    border stay straight.
    """
    padded = np.pad(labels, 1, constant_values=-1)
    pieces: dict[int, list[tuple[tuple, list]]] = {}
    for points in boundary_chains(padded):
        middle = (points[0] + points[1]) / 2
        direction = points[1] - points[0]
        normal = np.array([-direction[1], direction[0]]) * 0.25
        x, y = np.floor(middle + normal).astype(int)
        left = int(padded[y, x])
        x, y = np.floor(middle - normal).astype(int)
        right = int(padded[y, x])
        points = points - 1
        start = (float(points[0, 0]), float(points[0, 1]))
        if min(left, right) < 0:
            nodes: list[tuple[str, tuple[float, ...]]] = [
                ("L", (float(x), float(y))) for x, y in simplify(points, 0)[1:]
            ]
        else:
            nodes = curve_nodes(points, tolerance)
        end = (float(points[-1, 0]), float(points[-1, 1]))
        if left >= 0:
            pieces.setdefault(left, []).append((start, nodes))
        if right >= 0:
            pieces.setdefault(right, []).append((end, _reversed(start, nodes)))
    outlines = {}
    for index, segments in pieces.items():
        starts: dict[tuple, list[int]] = {}
        for n, (start, _) in enumerate(segments):
            starts.setdefault(_key(start), []).append(n)
        unused = set(range(len(segments)))
        parts = []
        while unused:
            n = min(unused)
            unused.remove(n)
            start, nodes = segments[n]
            loop = list(nodes)
            while _key(loop[-1][1][-2:]) != _key(start):
                following = next(
                    m for m in starts[_key(loop[-1][1][-2:])] if m in unused
                )
                unused.remove(following)
                loop.extend(segments[following][1])
            parts.append(_data(start, loop[:-1] if loop[-1][0] == "L" else loop, True))
        outlines[index] = " ".join(parts)
    return outlines


def _key(point) -> tuple[int, int]:
    """A traced corner, which lies on the pixel grid, as a dictionary key."""
    return round(point[0]), round(point[1])


def line_runs(skeleton: np.ndarray, spur: float) -> list[np.ndarray]:
    """The centrelines as runs of pixel centres between their ends and
    junctions, closed loops ending where they start.

    A junction is where more than two runs meet; each run ends at its middle
    so the strokes join. Branches shorter than *spur* off a junction, and
    loops that short from a junction back to it, are thinning's whiskers and
    go.
    """
    height, width = skeleton.shape
    padded = np.pad(skeleton, 1)
    ring = _ring(padded)
    crossings = sum(~a & b for a, b in pairwise((*ring, ring[0])))
    node = skeleton & (crossings != 2)
    junctions, _ = label(node & (crossings > 2), np.ones((3, 3)))
    found = int(junctions.max())
    centres = {
        index: (x + 0.5, y + 0.5)
        for index, (y, x) in enumerate(
            center_of_mass(junctions > 0, junctions, range(1, found + 1)), start=1
        )
    }
    offsets = ((0, -1), (1, 0), (0, 1), (-1, 0), (1, -1), (1, 1), (-1, 1), (-1, -1))
    visited = np.zeros_like(skeleton)

    def neighbours(x: int, y: int):
        for dx, dy in offsets:
            u, v = x + dx, y + dy
            if 0 <= u < width and 0 <= v < height and skeleton[v, u]:
                yield u, v

    def point(x: int, y: int) -> tuple[float, float]:
        junction = int(junctions[y, x])
        return centres[junction] if junction else (x + 0.5, y + 0.5)

    # Each run, how many of its ends are at a junction, and whether it comes
    # back to the junction it left.
    runs: list[tuple[np.ndarray, int, bool]] = []

    def walk(x: int, y: int, u: int, v: int) -> None:
        """Follow the run leaving node (x, y) through (u, v)."""
        start = int(junctions[y, x])
        run = [point(x, y)]
        previous = (x, y)
        ended = 0
        while True:
            if node[v, u]:
                run.append(point(u, v))
                ended = int(junctions[v, u])
                break
            visited[v, u] = True
            run.append((u + 0.5, v + 0.5))
            following = [
                p
                for p in neighbours(u, v)
                if p != previous
                and (node[p[1], p[0]] or not visited[p[1], p[0]])
                # Leaving a junction passes beside its other pixels.
                and not (len(run) <= 2 and start and junctions[p[1], p[0]] == start)
            ]
            if not following:
                break
            # A node next to it ends the run; else straight on before
            # diagonally, since a staircase corner is not a branch.
            nodes = [p for p in following if node[p[1], p[0]]]
            ahead = [p for p in following if abs(p[0] - u) + abs(p[1] - v) == 1]
            previous = (u, v)
            u, v = (nodes or ahead or following)[0]
        runs.append((np.array(run), (start > 0) + (ended > 0), start == ended > 0))

    for y, x in zip(*np.nonzero(node), strict=True):
        for u, v in neighbours(int(x), int(y)):
            if node[v, u] or visited[v, u]:
                continue
            walk(int(x), int(y), u, v)
    # What is left are loops with no node on them.
    for y, x in zip(*np.nonzero(skeleton & ~node), strict=True):
        if visited[y, x]:
            continue
        visited[y, x] = True
        run = [(x + 0.5, y + 0.5)]
        previous = u, v = int(x), int(y)
        while True:
            unvisited = [p for p in neighbours(u, v) if not visited[p[1], p[0]]]
            if not unvisited:
                break
            ahead = [p for p in unvisited if abs(p[0] - u) + abs(p[1] - v) == 1]
            previous = (u, v)
            u, v = (ahead or unvisited)[0]
            visited[v, u] = True
            run.append((u + 0.5, v + 0.5))
        del previous
        if len(run) > 2:
            run.append(run[0])
        runs.append((np.array(run), 0, False))
    kept = []
    for run, at_junctions, returns in runs:
        length = float(np.linalg.norm(np.diff(run, axis=0), axis=1).sum())
        # A whisker: off a junction to a free end, and short; or a short way
        # round a pinhole in a junction's blob, back to the junction.
        if length < 2 or (length < spur and (at_junctions == 1 or returns)):
            continue
        kept.append(run)
    return kept


def line_colours(target: np.ndarray, skeleton: np.ndarray) -> np.ndarray:
    """A few colours for the lines, from the pixels down their middles."""
    pixels = target[skeleton]
    count = min(LINE_COLOURS, len(pixels))
    assigned = fit_palette(pixels[:, None, :], count, 24, gpu=False)[:, 0]
    palette = [
        np.median(pixels[assigned == i], axis=0)
        for i in range(count)
        if (assigned == i).any()
    ]
    # Colours too close to tell apart are one, weighted by use.
    palette.sort(key=lambda c: float(c @ LUMINANCE))
    merged: list[np.ndarray] = []
    for c in palette:
        if merged and np.linalg.norm(merged[-1] - c) < LINE_COLOUR_DIFFERENCE:
            continue
        merged.append(c)
    return np.array(merged)


def vectorize(
    image: Image.Image,
    *,
    regions: int = 50,
    line_width: float = 0.0,
    tolerance: float = 0.75,
    strokes: bool = True,
) -> tuple[str, dict]:
    """Trace *image* as flat regions and drawn lines, as SVG in its pixels.

    *line_width* 0 measures the lines; any other fixes their stroke width.
    Without *strokes*, or when the widths vary too much for one, the lines
    are filled shapes instead.
    """
    if regions < 1 or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("invalid region count or tolerance")
    started = time.monotonic()
    target = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width = target.shape[:2]
    radius = max(3, int(np.ceil(line_width)))
    line, darkness = detect_lines(target, radius)
    # 1. Regions the lines bound, then split where only the colour changes.
    filled = trapped_ball_fill(~line)
    split = split_by_colour(target, filled, line)
    # 2. Line pixels, and corners no ball reached, go to the nearest region:
    # the boundary runs down the middle of each line.
    if split.any():
        split = split[nearest_indices(split == 0)]
    labels = merge_regions(split, target, line, regions)
    labels = remove_fragments(labels, PIECE)
    _, labels = np.unique(labels, return_inverse=True)
    labels = labels.reshape(line.shape)
    count = int(labels.max()) + 1
    # Each region's colour is the median of its own paint, not of the lines.
    paint = np.where(line, 0, labels + 1)
    bare = np.bincount(paint.ravel(), minlength=count + 1)[1:] == 0
    indices = np.arange(1, count + 1)
    medians = np.stack([median(target[..., c], paint, indices) for c in range(3)], 1)
    if bare.any():
        medians[bare] = np.stack(
            [median(target[..., c], labels + 1, indices[bare]) for c in range(3)], 1
        )
    fills = {index: colour(medians[index]) for index in range(count)}
    outlines = region_outlines(labels, tolerance)
    order = np.argsort(-np.bincount(labels.ravel(), minlength=count))
    # Beneath them all, the largest region's colour shows at any seam.
    backdrop = fills[int(order[0])]
    parts = [f'<rect width="{width}" height="{height}" fill="{backdrop}"/>']
    parts.extend(
        f'<path d="{outlines[int(i)]}" fill="{fills[int(i)]}" fill-rule="evenodd"/>'
        for i in order
        if int(i) in outlines
    )
    details: dict[str, Any] = {
        "regions": len(outlines),
        "line_pixels": int(line.sum()),
    }
    if line.any():
        line_parts, line_details = _line_paths(
            target, line, darkness, line_width, tolerance, strokes
        )
        parts.extend(line_parts)
        details.update(line_details)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">' + "".join(parts) + "</svg>"
    )
    details["seconds"] = time.monotonic() - started
    return svg, details


def _line_paths(
    target: np.ndarray,
    line: np.ndarray,
    darkness: np.ndarray,
    line_width: float,
    tolerance: float,
    strokes: bool,
) -> tuple[list[str], dict]:
    """The lines as stroked paths, one per colour and width, or as filled
    shapes."""
    skeleton = thin(line)
    if not skeleton.any():
        return [], {}
    # Each line pixel's ink, from none to all of the darkness down the middle
    # of its line, counted to the centreline pixel nearest it: the ink across
    # the line there, its width.
    nearest = nearest_indices(~skeleton)
    ink = np.clip(darkness / np.maximum(darkness[nearest], 1e-6), 0, 1) * line
    flat = np.ravel_multi_index(nearest, line.shape).ravel()
    across = np.bincount(flat, ink.ravel(), minlength=line.size).reshape(line.shape)
    palette = line_colours(target, skeleton)
    runs = line_runs(skeleton, spur=2 * float(np.median(across[skeleton])) + 2)
    measured = []
    for run in runs:
        xs = np.clip(run[:, 0].astype(int), 0, line.shape[1] - 1)
        ys = np.clip(run[:, 1].astype(int), 0, line.shape[0] - 1)
        # The ends sit in junctions, where the ink of several lines meets.
        inner = slice(1, -1) if len(run) > 4 else slice(None)
        widths = across[ys[inner], xs[inner]]
        middle = np.median(target[ys, xs], axis=0)
        measured.append(
            (
                int(np.square(palette - middle).sum(1).argmin()),
                float(np.median(widths)),
                float(np.percentile(widths, 90) / max(np.percentile(widths, 10), 0.5)),
                len(run),
            )
        )
    lengths = np.array([m[3] for m in measured], dtype=float)
    tapered = np.array([m[2] > WIDTH_SPREAD for m in measured], dtype=bool)
    share = float(lengths[tapered].sum() / max(lengths.sum(), 1))
    details = {
        "line_colours": len(palette),
        "line_runs": len(runs),
        "tapered_share": round(share, 3),
    }
    if not strokes or (not line_width and share > TAPERED_SHARE):
        # Filled shapes: where the ink is at least half a line's darkness.
        shape = ink >= 0.5
        colours = np.square(target[..., None, :] - palette).sum(-1).argmin(-1)
        paths = []
        for index, value in enumerate(palette):
            data = mask_path(shape & (colours == index), density=DENSITY, smooth=SMOOTH)
            if data is None:
                continue
            paths.append(
                f'<path d="{_simplified_data(data, tolerance)}" '
                f'fill="{colour(value)}" fill-rule="nonzero"/>'
            )
        return paths, {**details, "line_style": "filled", "line_paths": len(paths)}
    # The runs of one colour share a path, or two when some are much bolder.
    widths = np.array([max(1.0, m[1]) for m in measured])
    colours = np.array([m[0] for m in measured])
    bold = np.zeros(len(runs), dtype=bool)
    for index in np.unique(colours):
        own = colours == index
        low, high = _weighted_percentiles(widths[own], lengths[own], (25, 75))
        if not line_width and high > WIDTH_STEP * low:
            bold[own] = widths[own] > np.sqrt(low * high)
    grouped: dict[tuple[int, bool], list[tuple[Subpath, float, int]]] = {}
    for run, index, width, step in zip(runs, colours, widths, bold, strict=True):
        width = line_width or width
        closed = np.array_equal(run[0], run[-1])
        nodes = curve_nodes(run, tolerance)
        if closed and nodes and nodes[-1][0] == "L":
            nodes = nodes[:-1]
        contour = Subpath(
            "s",
            (
                PathNode("n", "M", tuple(float(v) for v in run[0])),
                *(PathNode("n", c, v) for c, v in nodes),
            ),
            closed,
        )
        # Each run counts toward its path's width by the size of its data.
        size = len(_data(run[0], nodes, closed))
        grouped.setdefault((int(index), bool(step)), []).append((contour, width, size))
    paths = []
    pieces_before = pieces_after = 0
    for index, step in sorted(grouped):
        pieces = grouped[index, step]
        width = _weighted_percentiles(
            np.array([w for _, w, _ in pieces]),
            np.array([size for _, _, size in pieces], dtype=float),
            (50,),
        )[0]
        contours = _joined_runs([c for c, _, _ in pieces], LINE_GAP * width)
        pieces_before += len(pieces)
        pieces_after += len(contours)
        data = " ".join(
            _data(
                c.nodes[0].endpoint,
                [(n.command, n.values) for n in c.nodes[1:]],
                c.closed,
            )
            for c in contours
        )
        paths.append(
            f'<path d="{data}" fill="none" '
            f'stroke="{colour(palette[index])}" stroke-width="{width:.2f}" '
            'stroke-linecap="round" stroke-linejoin="round"/>'
        )
    return paths, {
        **details,
        "line_style": "strokes",
        "line_paths": len(paths),
        "line_pieces": pieces_after,
        "line_runs_joined": pieces_before - pieces_after,
    }


def _joined_runs(contours: list[Subpath], reach: float) -> list[Subpath]:
    """The runs of one stroke joined where one carries on from another: at
    a junction, the straightest way through, and across a gap at most
    *reach* long, as the line was heading."""
    lines = [
        tuple(PathNode(f"n{i}_{j}", n.command, n.values) for j, n in enumerate(c.nodes))
        for i, c in enumerate(contours)
        if not c.closed
    ]
    ends = [e for i, nodes in enumerate(lines) for e in contour_ends(i, nodes)]
    chains, _ = joined(lines, end_pairs(ends, reach))
    used = {i for members, _ in chains for i in members}
    return [
        *(c for c in contours if c.closed),
        *(Subpath("s", nodes) for i, nodes in enumerate(lines) if i not in used),
        *(subpath for _, subpath in chains),
    ]


def _weighted_percentiles(
    values: np.ndarray, weights: np.ndarray, percentiles: tuple[float, ...]
) -> list[float]:
    """Percentiles of *values*, each counted *weights* times."""
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return [
        float(values[order][np.searchsorted(cumulative, q / 100 * cumulative[-1])])
        for q in percentiles
    ]
