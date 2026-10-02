"""Cel tracer: flat regions bounded by the drawn lines, with the lines on top.

Cel and anime art is flat colour inside drawn outlines. The tracer finds the
lines first, as marks darker in their brightest channel than the surface
around them (so a black line on a navy fill counts), in a grainy image after
a median smooths the grain away; dark shapes much wider than a line are left
to the fills. It fills the space between the lines with a shrinking ball so
a small gap in a line does not join the regions either side, splits each
region by colour where a shade edge has no line, and merges regions down to
a target count, keeping apart regions of different colours a line runs
between and, past the count, shadows a step darker than their surface. The
line pixels go to the regions either side, so neighbours meet at the line's
middle and share one traced edge, smoothed between its corners
before it is fitted. Each region's colour is then fitted in closed form to
the image under the lines as drawn, and a region whose colour clearly ramps
takes a linear gradient. The lines are thinned to centrelines and drawn over the
fills as strokes in their ink: a thin line's antialiased middle is a mix of
ink and surface, so it is drawn darker and thinner than its pixels look.
There is one path per line colour and width, a line cut where its width
steps so each part has its own.
Optionally one unbroken stroke runs round the drawing's silhouette in place
of the traced outer line, in its ink, down the middle of that ink at its
width, and open where the edge has none.

CPU only, with numpy and scipy.
"""

from __future__ import annotations

import heapq
import time
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    binary_opening,
    binary_propagation,
    center_of_mass,
    gaussian_filter,
    gaussian_filter1d,
    grey_closing,
    label,
    map_coordinates,
    median,
    median_filter,
    minimum_filter,
)

from vectrify.document.lines import contour_ends, end_pairs, joined
from vectrify.document.model import Geometry, PathNode, Subpath
from vectrify.document.paint import hex_colour
from vectrify.refine.colour_regions import (
    boundary_chains,
    colour,
    distance_transform_edt,
    fit_palette,
    nearest_indices,
    remove_fragments,
    simplified_indices,
    simplify,
)
from vectrify.refine.frozen import Frozen
from vectrify.refine.redraw import _corners
from vectrify.refine.redraw import _smoothed as _smoothed_between
from vectrify.refine.samvg import _fit_cubic, _simplified_data, mask_path
from vectrify.refine.simplify import simplified_geometry

if TYPE_CHECKING:
    from vectrify.operations.methods.colours import Ramp

LUMINANCE = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
# A line is darker than the surface either side by this much at its core, in
# 0-255 of the brightest channel, and its antialiased edge by a third of it.
LINE_CONTRAST = 12
# Line pieces smaller than this many pixels are specks, not lines, and holes
# in a line this small are closed.
LINE_SPECK = 20
LINE_HOLE = 12
# A pixel taking less than this share of the light that the image's lines
# typically take is shading, not a line.
LINE_INK = 0.5
# A mark as dark as ink, at most this share of the surface's light, is a
# line up to twice as wide as other marks: a bold outline.
BOLD_INK = 0.3
# How far, in pixels, a surface is followed into the notches it opens into.
# A notch is within NOTCH_BAND above to LINE_CONTRAST / 2 below the surface,
# that margin shrinking on a surface darker than NOTCH_DARK down to a
# quarter: a line on a dark fill is darker than it by less.
NOTCH_REACH = 32
NOTCH_BAND = 6
NOTCH_DARK = 128
# Above this much grain (see noise_level), lines are found in the image
# with a 3 x 3 median taken.
NOISE = 1.5
# Dark shapes deeper than this many times the typical line's half width, and
# than SHAPE_LEAST pixels (so bold strokes of lettering stay lines), are
# filled, not stroked.
SHAPE_DEPTH = 3.0
SHAPE_LEAST = 4.5
# The balls the free space is filled with, largest first, in pixels. A gap in
# a line narrower than a ball keeps it out: the largest closes the most.
BALLS = (6, 4, 2, 1)
# Colours the regions are split by, and the smallest split-off piece kept.
PALETTE = 16
PIECE = 24
# Merging across a drawn line costs this many times more for the share of
# the boundary a line runs along: regions a line separates and that differ
# stay apart, while regions of one colour either side of a line merge
# freely, since the line is drawn over them anyway.
LINE_PENALTY = 2.0
# Lightness counts this many times more than the rest of a colour difference
# when regions merge: a shadow is a step in value more than in hue.
SHADE = 2.0
# Two regions a step of SHADOW_STEP apart in luminance, each with at least
# SHADOW_LEAST pixels of paint, are a shadow and the surface it falls on: they
# are kept apart, and the shadow does not count toward the target count.
SHADOW_STEP = 8.0
SHADOW_LEAST = 150
# Curves per this many traced pixels of a filled line shape before it is
# simplified, and the smoothing that takes the pixel staircase out of any
# traced run first, in pixels.
DENSITY = 6
SMOOTH = 1.0
# A region boundary is smoothed more than a line before it is fitted, and
# cut into curves at the points a polyline within this many times the
# tolerance needs (a line's within half): a fill's edge has only the pixel
# staircase to lose, where a line's centreline wavers with its ink. The
# cubics between the cuts are then simplified within the tolerance itself,
# so cutting this coarsely leaves fewer points without moving the outline.
FILL_SMOOTH = 2.0
FILL_CUT = 2.0
# A run turning more than CORNER degrees over CORNER_SPAN points either side
# has a corner there: each stretch between corners is smoothed on its own
# and a curve ends at each, so the corner stays sharp.
CORNER = 50.0
CORNER_SPAN = 4
# A line whose width varies more than this along it, its 90th percentile
# over its 10th, is tapered (reported only: it is still a stroke).
WIDTH_SPREAD = 2.5
WIDTH_STEP = 1.6
# The strokes of one colour are grouped into paths of about one width, the
# widest of a group at most WIDTH_GROUP times its narrowest, at most
# WIDTH_GROUPS of them; no stroke is thinner than THINNEST pixels.
WIDTH_GROUP = 1.4
WIDTH_GROUPS = 4
THINNEST = 0.4
# A thin line's middle mixes its ink with the surface: of the inks that
# explain its colour within this distance, in 0-255 RGB, of the best, the
# one covering the least is its ink.
INK_SLACK = 12.0
# A line is cut where its width steps by WIDTH_STEP, into pieces at least
# this many points long, or this many of the typical line's widths.
LINE_PIECE = 12
LINE_PIECE_WIDTHS = 4.0
# At most this many line colours, one stroked path each; closer colours, in
# 0-255 RGB, are one.
LINE_COLOURS = 3
LINE_COLOUR_DIFFERENCE = 40.0
# The longest gap, in line widths, bridged between two runs of one stroke
# that carry on from each other: a line the detection broke.
LINE_GAP = 1.5
# The outer outline (with *outline*): the background is the canvas border's
# commonest colour when at least BACKGROUND_SHARE of the border has it, those
# pixels within BACKGROUND_FLAT of it on average (a plain backdrop), and
# every pixel within BACKGROUND_DIFFERENCE of it, in 0-255 RGB, that the
# border reaches; the drawing is the rest, less specks under SILHOUETTE_LEAST
# pixels. Centreline pixels no further in from the background than their
# line is wide, give or take OUTLINE_GAP, are the outer line's; when there
# are at least OUTLINE_INKED as many as the silhouette has edge pixels, the
# outline is drawn at their typical depth, width and ink. Without outer ink
# it is the lines' typical width, or OUTLINE_WIDTH with no lines, its middle
# OUTLINE_INSET beyond half that in. A traced line within OUTLINE_SLACK
# beyond half the outline's width of its middle is the outline itself. The
# background's regions keep out of the silhouette's rim, OUTLINE_RIM deep.
# With outer ink, the outline is drawn only where the ink across the edge,
# from where it first covers half its most (at least OUTLINE_PEAK) until
# its cover falls below OUTLINE_CLEAR, is at least OUTLINE_INK_LEAST of a
# stroke wide, across gaps
# up to OUTLINE_BRIDGE strokes long; it is cut where that width steps by
# OUTLINE_STEP, each piece down its own ink's middle.
BACKGROUND_SHARE = 0.5
BACKGROUND_FLAT = 12.0
BACKGROUND_DIFFERENCE = 40.0
SILHOUETTE_LEAST = 400
OUTLINE_INKED = 0.2
OUTLINE_GAP = 2.5
OUTLINE_WIDTH = 2.0
OUTLINE_INSET = 0.5
OUTLINE_SLACK = 1.5
OUTLINE_RIM = 12
OUTLINE_BRIDGE = 6.0
OUTLINE_STEP = 1.25
OUTLINE_INK_LEAST = 0.25
OUTLINE_CLEAR = 0.1
OUTLINE_PEAK = 0.25
# Ink beyond the strokes' reach, mostly one ink's colour, in pieces at least
# this many pixels across, is filled beneath them.
UNCOVERED_WIDE = 3
UNCOVERED_LEAST = 8
# A region's colour is fitted where the lines leave at least this many
# pixels' worth of it showing.
FIT_LEAST = 4.0
# A region of at least RAMP_PIXELS takes a gradient when one lowers its
# squared error under the lines by RAMP_GAIN of the flat fill's, and by
# RAMP_LEAST per pixel in 0-255 RGB squared; it is fitted on about
# RAMP_SAMPLES of its pixels.
RAMP_PIXELS = 400
RAMP_GAIN = 0.25
RAMP_LEAST = 4.0
RAMP_SAMPLES = 4000


def _disk(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return x * x + y * y <= radius * radius


def lightness(target: np.ndarray) -> np.ndarray:
    """How far each pixel is from black ink: its brightest channel, so a
    neutral line on a navy fill of the same luminance is still darker."""
    return target.max(-1)


def line_darkness(target: np.ndarray, radius: int) -> np.ndarray:
    """How much darker each pixel is than the surface around it: a black
    top-hat, which answers only on dark marks narrower than *radius* * 2."""
    light = lightness(target)
    return grey_closing(light, size=2 * radius + 1) - light


def detect_lines(target: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
    """The drawn lines, with their antialiased edges, and each pixel's
    darkness against the surface around it.

    Darkness is measured in the brightest channel, against the surface
    within *radius*, or twice that for a mark as dark as ink. Hysteresis
    keeps a faint pixel only when it joins a clearly dark one. Shading can
    be as narrow as a line, but it takes away less of the light beneath: a
    pixel much fainter than the image's lines is left to the fills, so a line
    keeps going where shading runs into it. A dark notch between two light
    spikes is as narrow as a line too, but it opens into a surface as dark as
    itself; it is given back to that surface.
    """
    light = lightness(target)
    darkness = line_darkness(target, radius)
    bold = line_darkness(target, 2 * radius)
    darkness = np.maximum(
        darkness, np.where(light <= BOLD_INK * (bold + light), bold, 0)
    )
    core = darkness >= LINE_CONTRAST
    if not core.any():
        return core, darkness
    # How much of the surface's light a line takes: 1 for black on white.
    share = darkness / np.maximum(darkness + light, 1)
    inked = share >= LINE_INK * float(np.percentile(share[core], 75))
    faint = darkness >= LINE_CONTRAST / 3
    mask = binary_propagation(core & inked, mask=faint & inked)
    # With the antialiased rim either side.
    mask |= binary_dilation(mask, np.ones((3, 3))) & faint
    mask = _without_notches(light, mask)
    pieces, count = label(mask, np.ones((3, 3)))
    keep = np.bincount(pieces.ravel(), minlength=count + 1) >= LINE_SPECK
    keep[0] = False
    mask = keep[pieces]
    # Pinholes in a line would thin into loops.
    holes, found = label(binary_fill_holes(mask) & ~mask)
    if found:
        small = np.bincount(holes.ravel()) <= LINE_HOLE
        small[0] = False
        mask |= small[holes]
    return mask, darkness


def noise_level(target: np.ndarray) -> float:
    """How grainy *target* is: the mean step between neighbouring pixels'
    brightest channels, leaving out the edges. Flat cel art steps by under
    one; noise and JPEG, by two or more."""
    light = target.max(-1)
    steps = np.abs(np.diff(light, axis=1)).ravel()
    small = steps[steps < 24]
    return float(small.mean()) if small.size else 0.0


def without_shapes(line: np.ndarray, times: float = SHAPE_DEPTH) -> np.ndarray:
    """*line* less its dark shapes, the parts deeper than *times* the
    typical line's half width, which are filled, not stroked: each such
    pixel's disk, as far as the shape is deep there."""
    skeleton = thin(line)
    if not skeleton.any():
        return line
    depth = distance_transform_edt(line)
    inside = depth > max(SHAPE_LEAST, times * float(np.median(depth[skeleton])))
    if not inside.any():
        return line
    y, x = nearest_indices(~inside)
    return line & ~(distance_transform_edt(~inside) <= depth[y, x] + 0.5)


def _without_notches(luminance: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """*mask* less the pixels no darker than the surface they join.

    The surface grows into the marks pixel by pixel, taking each that is
    about as bright as the surface it grew from, at most a little darker
    (less so on a dark surface) and not much lighter, and carrying that
    surface's brightness on: down a notch as dark as the surface it opens
    into, but not along a line darker than its light side that runs into
    a dark fill.
    """
    mask = mask.copy()
    surface = np.where(mask, np.inf, luminance)
    for _ in range(NOTCH_REACH):
        beside = minimum_filter(surface, size=3)
        margin = LINE_CONTRAST / 2 * np.clip(beside / NOTCH_DARK, 0.25, 1)
        joined = (
            mask & (luminance >= beside - margin) & (luminance <= beside + NOTCH_BAND)
        )
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
    *,
    shade: float = SHADE,
    shadow_step: float = SHADOW_STEP,
    shadow_least: int = SHADOW_LEAST,
) -> np.ndarray:
    """*labels* merged greedily down to *count* regions, renumbered from 0.

    Merging two regions costs their colour difference squared, the step in
    lightness counted *shade* times more, weighted by the smaller region's
    size (Ward), and *penalty* times more for the share of their boundary a
    drawn line runs along: different regions a line separates stay apart
    unless they are small or there is nothing else left to merge.

    Two regions with at least *shadow_least* pixels of paint each and
    *shadow_step* or more apart in luminance are never merged: a shadow and
    the surface it falls on. The darker of each pair kept apart this way
    does not count toward *count*, so more regions may be left.
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
    # Paint of its own: a region all line has none, and its pixels stand in.
    own = list(area)
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
        lined = along / total
        step = float(difference @ LUMINANCE)
        spread = float(difference @ difference) + shade * step * step
        # Between merges that cost nothing, the one across no line first.
        return weight * (spread * (1 + penalty * lined) + lined)

    def shadow(a: int, b: int) -> int | None:
        """The shadow merging *a* and *b* would wash out, if any: the darker
        of two regions with enough paint of their own (not lines) a step of
        *shadow_step* in lightness apart."""
        if min(own[a], own[b]) < shadow_least:
            return None
        light = [float(sums[i] / area[i] @ LUMINANCE) for i in (a, b)]
        if abs(light[0] - light[1]) < shadow_step:
            return None
        return a if light[0] < light[1] else b

    version = [0] * n
    heap = [(cost(a, b), a, b, 0, 0) for a in range(n) for b in edges[a] if a < b]
    heapq.heapify(heap)
    parent = list(range(n))
    regions = n
    # Shadows a merge was refused for: they do not count toward *count*.
    shadows: set[int] = set()
    while regions - len(shadows) > count and heap:
        _, a, b, va, vb = heapq.heappop(heap)
        if version[a] != va or version[b] != vb:
            continue
        kept = shadow(a, b)
        if kept is not None:
            shadows.add(kept)
            continue
        shadows.difference_update((a, b))
        # The larger keeps its number; the smaller's neighbours join it.
        if len(edges[a]) < len(edges[b]):
            a, b = b, a
        parent[b] = a
        area[a] += area[b]
        own[a] += own[b]
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


def run_corners(points: np.ndarray, closed: bool) -> list[int]:
    """Where the run *points* turns sharply, by index; a closed run's turns
    are found across its start too."""
    span = CORNER_SPAN
    if not closed:
        return _corners(points, span, CORNER)
    loop = points[:-1]
    if len(loop) < 2 * span + 1:
        return []
    around = np.concatenate((loop[-span:], loop, loop[:span]))
    found = {(i - span) % len(loop) for i in _corners(around, span, CORNER)}
    return sorted(found)


def _smoothed(
    points: np.ndarray, sigma: float, closed: bool, corners: list[int]
) -> np.ndarray:
    """*points* smoothed along their length, each piece between *corners* on
    its own so they stay sharp; the ends of an open run, and the start of a
    closed one, are kept."""
    if sigma <= 0 or len(points) < 5:
        return points
    if not closed:
        return _smoothed_between(points, sigma, corners)
    loop = points[:-1]
    if not corners:
        smooth = gaussian_filter1d(loop, sigma, axis=0, mode="wrap")
    else:
        # From the first corner round to it again, as an open run.
        first = corners[0]
        rolled = np.roll(loop, -first, axis=0)
        rolled = np.concatenate((rolled, rolled[:1]))
        inner = [c - first for c in corners[1:]]
        smooth = np.roll(_smoothed_between(rolled, sigma, inner)[:-1], first, axis=0)
    return np.concatenate((points[:1], smooth[1:], points[:1]))


def curve_nodes(
    points: np.ndarray, tolerance: float, *, smooth: float = SMOOTH, cut: float = 0.5
) -> list[tuple[str, tuple[float, ...]]]:
    """The run *points* as curves within *tolerance* pixels, after the start.

    The run is smoothed over *smooth* points between its corners, and cut
    at its corners and at the points a polyline within *cut* of the
    tolerance needs; a cubic fitted to the run between each two follows it,
    so a corner stays sharp. Then it is simplified: each curve goes as far
    as it can without moving the outline more than the tolerance. The ends
    stay where they are, so runs that meet keep meeting.
    """
    closed = len(points) > 3 and np.array_equal(points[0], points[-1])
    points = np.asarray(points, dtype=np.float64)
    corners = run_corners(points, closed) if smooth > 0 else []
    points = _smoothed(points, smooth, closed, corners)
    kept = simplified_indices(points, tolerance * cut) if len(points) > 2 else [0, 1]
    kept = sorted({*kept, *corners})
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
            nodes = curve_nodes(points, tolerance, smooth=FILL_SMOOTH, cut=FILL_CUT)
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

    def beside(x: int, y: int, junction: int) -> bool:
        return any(junctions[v, u] == junction for u, v in neighbours(x, y))

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
        away = False
        while True:
            if node[v, u]:
                run.append(point(u, v))
                ended = int(junctions[v, u])
                break
            visited[v, u] = True
            run.append((u + 0.5, v + 0.5))
            away = away or not beside(u, v, start)
            following = [
                p
                for p in neighbours(u, v)
                if p != previous
                and (node[p[1], p[0]] or not visited[p[1], p[0]])
                # Leaving a junction passes beside its other pixels.
                and not (len(run) <= 2 and start and junctions[p[1], p[0]] == start)
            ]
            if start and not away:
                # Still beside the junction it left, the run goes back into
                # it only when there is no other way on: a line leaving it
                # along a staircase passes by it once more.
                onward = [p for p in following if junctions[p[1], p[0]] != start]
                following = onward or following
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

    def onward(x: int, y: int) -> list[tuple[float, float]]:
        """The unvisited pixels on from (x, y), straight on before diagonally."""
        found = []
        while True:
            unvisited = [p for p in neighbours(x, y) if not visited[p[1], p[0]]]
            if not unvisited:
                return found
            ahead = [p for p in unvisited if abs(p[0] - x) + abs(p[1] - y) == 1]
            x, y = (ahead or unvisited)[0]
            visited[y, x] = True
            found.append((x + 0.5, y + 0.5))

    # What is left are loops with no node on them.
    for y, x in zip(*np.nonzero(skeleton & ~node), strict=True):
        if visited[y, x]:
            continue
        visited[y, x] = True
        run = [(x + 0.5, y + 0.5), *onward(int(x), int(y))]
        end = np.subtract(run[-1], run[0])
        if len(run) > 2 and np.abs(end).max() <= 1:
            run.append(run[0])
        else:
            # Not round to its start: an open stretch, followed both ways.
            run = [*reversed(onward(int(x), int(y))), *run]
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


def _ink_of(palette: np.ndarray, middle: np.ndarray, surface: np.ndarray) -> int:
    """Which of the *palette* inks, covering some of the pixel, mixes with
    *surface* into the colour *middle* down a line."""
    fits = []
    for index, ink in enumerate(palette):
        away = surface - ink
        span = float(away @ away)
        if span < 1:
            continue
        cover = float(np.clip((surface - middle) @ away / span, 0.05, 1))
        miss = float(np.linalg.norm(surface - cover * away - middle))
        fits.append((miss, cover, index))
    if not fits:
        return int(np.square(palette - middle).sum(1).argmin())
    least = min(f[0] for f in fits)
    # Of the inks that explain it about as well, the one covering least: a
    # thin line of dark ink, not a grey one as wide as its antialiasing.
    return min((f for f in fits if f[0] <= least + INK_SLACK), key=lambda f: f[1])[2]


def silhouette(target: np.ndarray) -> np.ndarray | None:
    """The drawing's silhouette, its holes filled: what the background does
    not cover, reaching in from the canvas border. The background is the
    border's commonest colour, when at least BACKGROUND_SHARE of the border
    has it and it is plain there; with none, None."""
    border = np.concatenate((target[0], target[-1], target[:, 0], target[:, -1]))
    # The commonest colour, to within 16 levels a channel.
    bins, counts = np.unique(
        (border // 16).astype(np.int32), axis=0, return_counts=True
    )
    common = bins[counts.argmax()]
    background = np.median(border[((border // 16) == common).all(-1)], axis=0)
    near = np.linalg.norm(target - background, axis=-1) < BACKGROUND_DIFFERENCE
    edge = np.zeros(near.shape, dtype=bool)
    edge[[0, -1], :] = edge[:, [0, -1]] = True
    if near[edge].mean() < BACKGROUND_SHARE:
        return None
    spread = np.linalg.norm(target[edge & near] - background, axis=-1).mean()
    if spread > BACKGROUND_FLAT:
        return None
    outside = binary_propagation(edge & near, mask=near)
    pieces, count = label(~outside)
    sizes = np.bincount(pieces.ravel(), minlength=count + 1)
    sizes[0] = 0
    drawing = np.asarray(binary_fill_holes(sizes[pieces] >= SILHOUETTE_LEAST))
    return drawing if drawing.any() else None


def outer_line(
    drawing: np.ndarray, line: np.ndarray, skeleton: np.ndarray
) -> tuple[np.ndarray | None, float]:
    """The centreline pixels of the drawing's outer ink, if it has any, and
    how far in from the background they typically lie.

    A centreline pixel no further in from the background than its line is
    wide, give or take OUTLINE_GAP, is the outer line's: its ink runs out to
    the edge. The drawing has outer ink when there is at least OUTLINE_INKED
    as much of it as there is silhouette edge.
    """
    inside = distance_transform_edt(drawing)
    rim = drawing & (inside < 1.5)
    outer = skeleton & (inside <= 2 * distance_transform_edt(line) + OUTLINE_GAP)
    if not rim.any() or outer.sum() < max(LINE_SPECK, OUTLINE_INKED * rim.sum()):
        return None, 0.0
    return outer, float(np.median(inside[outer]))


def outline_runs(
    drawing: np.ndarray, depth: float, width: float
) -> tuple[list[np.ndarray], np.ndarray]:
    """The middle of a closed stroke round the silhouette *drawing*, *depth*
    pixels in from the background, as runs of points; and the pixels along
    it. Where the silhouette runs off the canvas, it stops; pieces shorter
    than four times *width* go.

    The traced pixel edge is moved onto the level where the distance from
    the background, between pixel centres, is *depth*: through the outer
    line's centreline pixels, not half a pixel beside them.
    """
    height, wide = drawing.shape
    inside = distance_transform_edt(drawing)
    middle = binary_fill_holes(inside > depth - 0.5)
    pieces, count = label(middle)
    sizes = np.bincount(pieces.ravel(), minlength=count + 1)
    sizes[0] = 0
    middle = sizes[pieces] >= SILHOUETTE_LEAST
    edge = middle & ~binary_erosion(middle, border_value=1)
    slopes = tuple(np.gradient(inside))
    runs = []
    for points in boundary_chains(np.pad(middle.astype(np.int32), 1)):
        points = points - 1
        # Off the canvas edge: the parts that run along it go.
        on_edge = (
            (points[:, 0] <= 0)
            | (points[:, 0] >= wide)
            | (points[:, 1] <= 0)
            | (points[:, 1] >= height)
        )
        segments = [points]
        if on_edge.any():
            if np.array_equal(points[0], points[-1]):
                # A loop: from a point on the canvas edge round to it again.
                start = int(np.argmax(on_edge))
                points = np.concatenate((points[start:-1], points[: start + 1]))
                on_edge = np.concatenate((on_edge[start:-1], on_edge[: start + 1]))
            segments = [points[a:b] for a, b in _stretches(~on_edge)]
        for run in segments:
            if len(run) >= max(12, 4 * width):
                runs.append(_onto_level(run, inside, slopes, depth))
    return runs, edge


def _contour(run: np.ndarray, tolerance: float) -> Subpath:
    """The run *points* of an outline fitted with curves, as a contour."""
    closed = len(run) > 3 and np.array_equal(run[0], run[-1])
    nodes = curve_nodes(run, tolerance, smooth=FILL_SMOOTH, cut=FILL_CUT)
    if closed and nodes and nodes[-1][0] == "L":
        nodes = nodes[:-1]
    return Subpath(
        "s",
        (
            PathNode("n", "M", tuple(float(v) for v in run[0])),
            *(PathNode("n", c, v) for c, v in nodes),
        ),
        closed,
    )


def inked_outline(
    runs: list[np.ndarray],
    cover: np.ndarray,
    line: np.ndarray,
    drawing: np.ndarray,
    stroke: float,
) -> list[tuple[np.ndarray, float]]:
    """The stretches of the outline *runs* that run along drawn ink, each
    moved onto the ink's middle there and with its width.

    Across the silhouette *drawing* at each point, from just outside it to
    twice *stroke* in, the ink's *cover* of each pixel (0-1) sums to its
    width there and centres on its middle. Where there is less than
    OUTLINE_INK_LEAST of a stroke of ink, and no pixel of the drawn *line*
    within OUTLINE_SLACK beyond half the stroke, across gaps up to OUTLINE_BRIDGE
    strokes long, the outline is left open, as along a cropped hem with no
    ink. Each stretch is cut where the ink's width steps by OUTLINE_STEP,
    as the lines are, and each piece lies at its own ink's middle.
    """
    inside = distance_transform_edt(drawing)
    slopes = tuple(np.gradient(inside))
    gap = distance_transform_edt(~line)
    reach = stroke / 2 + OUTLINE_SLACK
    # Depths from the background across the silhouette, in pixels.
    depths = np.arange(-1.0, 2 * stroke + OUTLINE_GAP, 0.5)
    bridge = OUTLINE_BRIDGE * stroke
    shortest = max(12.0, 4 * stroke)
    span = max(LINE_PIECE, LINE_PIECE_WIDTHS * stroke)
    kept = []
    for run in runs:
        at = [run[:, 1] - 0.5, run[:, 0] - 0.5]
        start_depth = map_coordinates(inside, at, order=1, mode="nearest")
        dy, dx = (map_coordinates(s, at, order=1, mode="nearest") for s in slopes)
        norm = np.maximum(np.hypot(dx, dy), 1e-6)
        dx, dy = dx / norm, dy / norm
        # Each point's ray, from its depth *start_depth* to each of *depths*.
        offset = depths[None, :] - start_depth[:, None]
        xs = run[:, 0, None] + offset * dx[:, None] - 0.5
        ys = run[:, 1, None] + offset * dy[:, None] - 0.5
        ink = map_coordinates(cover, [ys.ravel(), xs.ravel()], order=1, mode="constant")
        ink = ink.reshape(xs.shape)
        # Only the outer ink: from its first sample covering half its most,
        # until the cover falls away, with the partial cover before it.
        index = np.arange(len(depths))[None, :]
        peak = ink.max(1, keepdims=True)
        strong = (ink >= 0.5 * peak) & (peak >= OUTLINE_PEAK)
        first = np.where(strong.any(1), strong.argmax(1), len(depths))[:, None]
        clear = (ink < OUTLINE_CLEAR) & (index > first)
        stop = np.where(clear.any(1), clear.argmax(1), len(depths))[:, None]
        ink = np.where((index >= first - 1) & (index < stop), ink, 0.0)
        width = ink.sum(1) * 0.5
        middle = (ink * depths).sum(1) / np.maximum(ink.sum(1), 1e-6)
        lined = gap[
            np.clip(run[:, 1].astype(int), 0, line.shape[0] - 1),
            np.clip(run[:, 0].astype(int), 0, line.shape[1] - 1),
        ]
        # Where the ink was measured, and where a drawn line at least runs.
        found = width >= OUTLINE_INK_LEAST * stroke
        inked = found | (lined <= reach)
        if not inked.any():
            continue
        closed = len(run) > 3 and np.array_equal(run[0], run[-1])
        for start in (np.argmax, np.argmin):
            if closed and not inked.all():
                # A loop: from a point with (then without) ink round to it.
                at = int(start(inked))
                order = np.concatenate((np.arange(at, len(run) - 1), np.arange(at + 1)))
                run, inked, found = run[order], inked[order], found[order]
                width, middle = width[order], middle[order]
            steps = np.linalg.norm(np.diff(run, axis=0), axis=1)
            along = np.concatenate(([0.0], np.cumsum(steps)))
            for first, last in _stretches(~inked):
                # A short break in the ink between two inked stretches.
                if (
                    first > 0
                    and last < len(run)
                    and along[last] - along[first] <= bridge
                ):
                    inked[first:last] = True
        for first, last in _stretches(inked):
            if along[last - 1] - along[first] < shortest:
                continue
            piece = run[first:last]
            # Across a break, and along a line too faint to measure, the
            # ink's width and middle either side, or the outline's own.
            own = found[first:last]
            wide, deep = width[first:last], middle[first:last]
            if own.any():
                wide = np.where(own, wide, np.median(wide[own]))
                deep = np.where(own, deep, np.median(deep[own]))
            else:
                wide = np.full(len(piece), stroke)
                deep = np.full(len(piece), float(np.median(start_depth)))
            parts = []
            for a, b in width_pieces(wide, span, OUTLINE_STEP):
                middle_depth = float(np.median(deep[a : b + 1]))
                part = _onto_level(piece[a : b + 1], inside, slopes, middle_depth)
                if parts:
                    # Each piece starts where the one before ends.
                    part[0] = parts[-1][0][-1]
                parts.append(
                    (part, max(stroke / WIDTH_STEP, float(np.median(wide[a : b + 1]))))
                )
            if len(parts) > 1 and np.array_equal(piece[0], piece[-1]):
                parts[-1][0][-1] = parts[0][0][0]
            kept.extend(parts)
    return kept


def _onto_level(
    points: np.ndarray,
    field: np.ndarray,
    slopes: tuple[np.ndarray, ...],
    level: float,
) -> np.ndarray:
    """*points*, in pixel-corner coordinates, each moved along the slope of
    *field* (sampled at pixel centres) onto where it is *level*, at most a
    pixel and a half."""
    points = np.asarray(points, dtype=np.float64)
    for _ in range(2):
        at = [points[:, 1] - 0.5, points[:, 0] - 0.5]
        value = map_coordinates(field, at, order=1, mode="nearest")
        dy, dx = (map_coordinates(s, at, order=1, mode="nearest") for s in slopes)
        norm = np.maximum(dx * dx + dy * dy, 0.25)
        step = np.clip((level - value) / norm, -1.5, 1.5)
        points = points + np.stack((dx * step, dy * step), 1)
    return points


def _within(labels: np.ndarray, drawing: np.ndarray, depth: float) -> np.ndarray:
    """*labels* with the background's regions, those mostly outside the
    silhouette *drawing*, kept out of its rim from *depth* in, the outline's
    middle, to OUTLINE_RIM further: the drawing's own regions meet the
    background beneath the outline's middle, wherever the traced outer line
    split them before."""
    flat = labels.ravel()
    total = np.bincount(flat)
    within = np.bincount(flat, drawing.ravel())
    background = within < total / 2
    inside = distance_transform_edt(drawing)
    rim = drawing & (inside > depth - 0.5) & (inside <= depth + OUTLINE_RIM)
    taken = rim & background[labels]
    own = drawing & ~background[labels]
    if not taken.any() or not own.any():
        return labels
    y, x = nearest_indices(~own)
    return np.where(taken, labels[y, x], labels)


def _stretches(mask: np.ndarray) -> list[tuple[int, int]]:
    """The (first, last + 1) of each run of True in *mask*."""
    padded = np.concatenate(([False], mask, [False]))
    steps = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return list(zip(steps[::2].tolist(), steps[1::2].tolist(), strict=True))


def off_outline(
    runs: list[np.ndarray], edge: np.ndarray, reach: float, shortest: float
) -> list[np.ndarray]:
    """*runs* less their stretches along the outer outline, whose middle is
    *edge*: points within *reach* of it are the outline drawn again. What is
    left of a run, if at least *shortest* long, runs on to the outline where
    it was cut, and from a free end within twice *reach* of it, so the
    lines meeting the outline join it."""
    away = distance_transform_edt(~edge)
    ys, xs = nearest_indices(~edge)
    height, width = edge.shape

    def nearest(point: np.ndarray) -> np.ndarray:
        x = min(max(int(point[0]), 0), width - 1)
        y = min(max(int(point[1]), 0), height - 1)
        return np.array([xs[y, x] + 0.5, ys[y, x] + 0.5])

    def distance(point: np.ndarray) -> float:
        x = min(max(int(point[0]), 0), width - 1)
        y = min(max(int(point[1]), 0), height - 1)
        return float(away[y, x])

    kept = []
    for run in runs:
        on = np.array([distance(p) <= reach for p in run])
        if not on.any():
            kept.append(run)
            continue
        if np.array_equal(run[0], run[-1]):
            # A loop: from a point on the outline round to it again.
            start = int(np.argmax(on))
            loop = run[:-1]
            run = np.concatenate((loop[start:], loop[: start + 1]))
            on = np.concatenate((on[:-1][start:], on[:-1][: start + 1]))
        for first, last in _stretches(~on):
            piece = run[first:last]
            if len(piece) < 2:
                continue
            length = float(np.linalg.norm(np.diff(piece, axis=0), axis=1).sum())
            if length < shortest:
                continue
            if first > 0 or distance(piece[0]) <= 2 * reach:
                piece = np.concatenate(([nearest(piece[0])], piece))
            if last < len(run) or distance(piece[-1]) <= 2 * reach:
                piece = np.concatenate((piece, [nearest(piece[-1])]))
            kept.append(piece)
    return kept


def vectorize(
    image: Image.Image,
    *,
    regions: int = 50,
    line_width: float = 0.0,
    tolerance: float = 0.75,
    strokes: bool = True,
    outline: bool = False,
    fit_colours: bool = True,
    gradients: bool = True,
) -> tuple[str, dict]:
    """Trace *image* as flat regions and drawn lines, as SVG in its pixels.

    *line_width* 0 measures the lines; any other fixes their stroke width.
    Without *strokes* the lines are filled shapes instead. With *outline*,
    one unbroken stroke runs round the drawing's silhouette (see
    :func:`silhouette`) in place of the traced outer line. With
    *fit_colours*, each region's colour is the one that best matches the
    image under the lines as drawn (see :func:`fitted_fills`), not its
    median; with *gradients*, a region whose colour clearly ramps takes a
    linear gradient (see :func:`ramps`).
    """
    if regions < 1 or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("invalid region count or tolerance")
    started = time.monotonic()
    target = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width = target.shape[:2]
    radius = max(3, int(np.ceil(line_width)))
    # Grain and JPEG noise read as faint marks everywhere: the lines are
    # found in the image with it smoothed away first.
    grainy = noise_level(target) > NOISE
    found = median_filter(target, size=(3, 3, 1)) if grainy else target
    line, darkness = detect_lines(found, radius)
    line = without_shapes(line)
    # 1. Regions the lines bound, then split where only the colour changes.
    filled = trapped_ball_fill(~line)
    split = split_by_colour(target, filled, line)
    # 2. Line pixels, and corners no ball reached, go to the nearest region:
    # the boundary runs down the middle of each line.
    if split.any():
        split = split[nearest_indices(split == 0)]
    labels = merge_regions(split, target, line, regions)
    labels = remove_fragments(labels, PIECE)
    drawing = silhouette(found) if outline else None
    outer, depth = None, 0.0
    if drawing is not None:
        outer, depth = outer_line(drawing, line, thin(line))
        if outer is not None:
            labels = _within(labels, drawing, depth)
    _, labels = np.unique(labels, return_inverse=True)
    labels = labels.reshape(line.shape)
    count = int(labels.max()) + 1
    medians = region_medians(target, labels, line)
    outlines = region_outlines(labels, tolerance)
    order = np.argsort(-np.bincount(labels.ravel(), minlength=count))
    details: dict[str, Any] = {
        "regions": len(outlines),
        "line_pixels": int(line.sum()),
    }
    line_parts: list[str] = []
    if line.any():
        line_parts, line_details = _line_paths(
            target,
            line,
            darkness,
            line_width,
            tolerance,
            strokes,
            medians[labels],
            drawing=drawing,
            outer=(outer, depth),
        )
        details.update(line_details)
    elif drawing is not None:
        # No lines to take its ink and width from.
        stroke = line_width or OUTLINE_WIDTH
        contours, _ = outline_runs(drawing, stroke / 2 + OUTLINE_INSET, stroke)
        if contours:
            line_parts.append(
                _stroke([_contour(c, tolerance) for c in contours], "#000000", stroke)
            )
        details["outline"] = len(contours)
    # Each region's colour, fitted under the lines as drawn; a region whose
    # colour ramps clearly better than it stays flat takes a gradient.
    fitted = medians
    gradient_defs: list[str] = []
    paints: dict[int, str] = {}
    if fit_colours or gradients:
        cover, painted = line_layer(line_parts, width, height)
        if fit_colours:
            fitted = fitted_fills(target, labels, cover, painted, medians)
        if gradients:
            for index, ramp in ramps(target, labels, cover, painted, fitted).items():
                gradient_defs.append(_gradient(f"ramp{index}", ramp))
                paints[index] = f"url(#ramp{index})"
        details["gradients"] = len(gradient_defs)
    fills = {i: paints.get(i) or colour(fitted[i]) for i in range(count)}
    # Beneath them all, the largest region's colour shows at any seam.
    backdrop = colour(fitted[int(order[0])])
    parts = [f"<defs>{''.join(gradient_defs)}</defs>"] if gradient_defs else []
    parts.append(f'<rect width="{width}" height="{height}" fill="{backdrop}"/>')
    parts.extend(
        f'<path d="{outlines[int(i)]}" fill="{fills[int(i)]}" fill-rule="evenodd"/>'
        for i in order
        if int(i) in outlines
    )
    parts.extend(line_parts)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">' + "".join(parts) + "</svg>"
    )
    details["seconds"] = time.monotonic() - started
    return svg, details


def line_layer(
    parts: list[str], width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """The lines *parts* rendered alone: each pixel's cover by them, 0-1,
    and their colour times that cover, 0-255 RGB."""
    import io

    import cairosvg

    if not parts:
        return np.zeros((height, width, 1), np.float32), np.zeros(
            (height, width, 3), np.float32
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">' + "".join(parts) + "</svg>"
    )
    png = cairosvg.svg2png(
        bytestring=svg.encode(), output_width=width, output_height=height
    )
    assert png is not None
    with Image.open(io.BytesIO(png)) as rendered:
        rgba = np.asarray(rendered.convert("RGBA"), dtype=np.float32)
    cover = rgba[..., 3:] / 255
    return cover, rgba[..., :3] * cover


def fitted_fills(
    target: np.ndarray,
    labels: np.ndarray,
    cover: np.ndarray,
    painted: np.ndarray,
    fallback: np.ndarray,
) -> np.ndarray:
    """Each region's flat colour, by label, that best matches *target* under
    the lines: a pixel shows ``(1 - cover) * fill + painted`` (see
    :func:`line_layer`), so the least-squares fill per channel is the sum of
    ``(1 - cover) * (target - painted)`` over the region's pixels, over that
    of ``(1 - cover) ** 2``, as Fit colours solves it. Pixels on a region's
    edge are left out: the outline's antialiasing already mixes them with
    the neighbour, so counting them would mix it in twice. A region the
    lines all but hide keeps its *fallback* colour."""
    count = len(fallback)
    flat = labels.ravel()
    inside = np.ones(labels.shape, dtype=bool)
    across = labels[1:] != labels[:-1]
    inside[1:] &= ~across
    inside[:-1] &= ~across
    across = labels[:, 1:] != labels[:, :-1]
    inside[:, 1:] &= ~across
    inside[:, :-1] &= ~across
    seen = (1 - cover[..., 0]).astype(np.float64) * inside
    weight = np.bincount(flat, (seen * seen).ravel(), minlength=count)
    sums = np.stack(
        [
            np.bincount(
                flat, (seen * (target[..., c] - painted[..., c])).ravel(), count
            )
            for c in range(3)
        ],
        1,
    )
    fitted = np.clip(sums / np.maximum(weight, 1e-9)[:, None], 0, 255)
    return np.where((weight >= FIT_LEAST)[:, None], fitted, fallback)


def ramps(
    target: np.ndarray,
    labels: np.ndarray,
    cover: np.ndarray,
    painted: np.ndarray,
    fitted: np.ndarray,
) -> dict[int, Ramp]:
    """The regions, by label, whose colour ramps: the linear gradient Fit
    gradients would give each (see :func:`fit_ramp`), fitted on at most
    about RAMP_SAMPLES of its pixels, kept where it lowers the region's
    squared error under the lines by at least RAMP_GAIN of its flat
    *fitted* colour's and by RAMP_LEAST per pixel."""
    from scipy.ndimage import find_objects

    from vectrify.operations.generate import Region
    from vectrify.operations.methods.colours import fit_ramp

    found = {}
    seen = 1 - cover[..., 0]
    for index, box in enumerate(find_objects(labels + 1)):
        if box is None:
            continue
        mask = labels[box] == index
        size = int(mask.sum())
        if size < RAMP_PIXELS:
            continue
        step = max(1, int(np.ceil(np.sqrt(size / RAMP_SAMPLES))))
        sample = (slice(step // 2, None, step), slice(step // 2, None, step))
        own = mask[sample]
        rows, columns = own.shape
        if own.sum() < 3:
            continue
        reference = target[box][sample] / 255
        under = np.where(own[..., None], painted[box][sample] / 255, reference)
        coverage = np.where(own, seen[box][sample], 0)[..., None].repeat(3, -1)
        region = Region(
            float(box[1].start),
            float(box[0].start),
            float(columns * step),
            float(rows * step),
            Image.new("RGB", (columns, rows)),
        )
        xs, ys = np.meshgrid(
            region.x + (np.arange(columns) + 0.5) * step,
            region.y + (np.arange(rows) + 0.5) * step,
        )
        ramp = fit_ramp(under, coverage, reference, region)
        if ramp is None or ramp.flat():
            continue
        (x0, y0), (x1, y1) = ramp.start, ramp.end
        dx, dy = x1 - x0, y1 - y0
        u = np.clip(((xs - x0) * dx + (ys - y0) * dy) / (dx * dx + dy * dy), 0, 1)
        start, end = (np.asarray(c) for c in ramp.colours)
        before, after = (
            float(((coverage * fill + under - reference)[own] ** 2).sum())
            for fill in (fitted[index] / 255, start + u[..., None] * (end - start))
        )
        pixels = float(own.sum())
        if (
            after <= before * (1 - RAMP_GAIN)
            and (before - after) / pixels >= RAMP_LEAST / 255**2
        ):
            found[index] = ramp
    return found


def _gradient(name: str, ramp: Ramp) -> str:
    """*ramp* as a linear gradient in the trace's pixels, named *name*."""
    (x1, y1), (x2, y2) = ramp.start, ramp.end
    stops = "".join(
        f'<stop offset="{offset}" stop-color="{hex_colour(c)}"/>'
        for offset, c in zip((0, 1), ramp.colours, strict=True)
    )
    return (
        f'<linearGradient id="{name}" gradientUnits="userSpaceOnUse" '
        f'x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}">{stops}'
        "</linearGradient>"
    )


def region_medians(
    target: np.ndarray, labels: np.ndarray, line: np.ndarray
) -> np.ndarray:
    """Each region's colour, by label from 0: the median of its own paint,
    not of the lines, or of all its pixels when it is all line."""
    count = int(labels.max()) + 1
    paint = np.where(line, 0, labels + 1)
    bare = np.bincount(paint.ravel(), minlength=count + 1)[1:] == 0
    indices = np.arange(1, count + 1)
    medians = np.stack([median(target[..., c], paint, indices) for c in range(3)], 1)
    if bare.any():
        medians[bare] = np.stack(
            [median(target[..., c], labels + 1, indices[bare]) for c in range(3)], 1
        )
    return medians


def _line_paths(
    target: np.ndarray,
    line: np.ndarray,
    darkness: np.ndarray,
    line_width: float,
    tolerance: float,
    strokes: bool,
    surface: np.ndarray | None = None,
    *,
    drawing: np.ndarray | None = None,
    outer: tuple[np.ndarray | None, float] = (None, 0.0),
) -> tuple[list[str], dict]:
    """The lines as stroked paths, one per colour and width, or as filled
    shapes; with the silhouette *drawing*, a stroke round it first, in the
    ink and at the width of its outer line, which it replaces: *outer* is
    that line's centreline pixels and depth (see :func:`outer_line`)."""
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
    # Down the middle of a line at least three pixels wide the ink covers
    # the pixel whole: the inks are those colours.
    solid = skeleton & (distance_transform_edt(line) >= 1.5)
    palette = line_colours(target, solid if solid.sum() >= 50 else skeleton)
    if surface is None:
        surface = target
    # Each line pixel's share covered by each ink, over the surface of the
    # region it lies in, summed across the line: its width in that ink.
    widths_by_ink = []
    covers = []
    for value in palette:
        away = surface - value
        span = (away * away).sum(-1)
        cover = ((surface - target) * away).sum(-1) / np.maximum(span, 1)
        # On a surface as dark as the ink, the line pixel is all ink. Every
        # pixel's cover, line or not, is kept: the outline goes by the ink
        # the picture has, whether or not it was found as a line.
        cover = np.where(span < 30**2, line, np.clip(cover, 0, 1))
        covers.append(cover)
        cover = cover * line
        widths_by_ink.append(
            np.bincount(flat, cover.ravel(), minlength=line.size).reshape(line.shape)
        )
    typical = float(np.median(across[skeleton]))
    runs = line_runs(skeleton, spur=2 * typical + 2)

    def style(run: np.ndarray) -> tuple[int, np.ndarray]:
        """The ink of *run*, and its width in that ink along it."""
        xs = np.clip(run[:, 0].astype(int), 0, line.shape[1] - 1)
        ys = np.clip(run[:, 1].astype(int), 0, line.shape[0] - 1)
        # The ends sit in junctions, where the ink of several lines meets.
        inner = slice(1, -1) if len(run) > 4 else slice(None)
        index = _ink_of(
            palette,
            np.median(target[ys, xs], axis=0),
            np.median(surface[ys, xs], axis=0),
        )
        return index, widths_by_ink[index][ys[inner], xs[inner]]

    outer_parts: list[str] = []
    outer_pieces: list[tuple[np.ndarray, float]] = []
    outer_details: dict[str, Any] = {}
    off = None
    if drawing is not None:
        outer_pixels, depth = outer
        if outer_pixels is None:
            depth = max(THINNEST, line_width or typical) / 2 + OUTLINE_INSET
        contours, edge = outline_runs(drawing, depth, 2 * depth)
        stroke = max(THINNEST, line_width or typical)
        outer_ink = int(np.argmin(palette @ LUMINANCE))
        if contours:
            # The traced runs along it are its ink and width.
            away = distance_transform_edt(~edge)
            along = [
                run
                for run in runs
                if np.mean(_widths_along(away, run) <= depth + OUTLINE_SLACK) >= 0.5
            ]
            if outer_pixels is not None and along:
                styles = [style(run) for run in along]
                lengths = np.array([len(run) for run in along], dtype=float)
                inks = np.array([index for index, _ in styles])
                outer_ink = int(np.bincount(inks, lengths).argmax())
                own = inks == outer_ink
                stroke = max(
                    THINNEST,
                    line_width
                    or _weighted_percentiles(
                        np.array([float(np.median(w)) for _, w in styles])[own],
                        lengths[own],
                        (50,),
                    )[0],
                )
            if outer_pixels is not None:
                # Only along the drawn ink, at its width there.
                pieces = inked_outline(
                    contours, covers[outer_ink], line, drawing, stroke
                )
                if line_width:
                    pieces = [(run, line_width) for run, _ in pieces]
                drawn = np.zeros(line.shape, dtype=bool)
                for run, _ in pieces:
                    xs = np.clip(run[:, 0].astype(int), 0, line.shape[1] - 1)
                    ys = np.clip(run[:, 1].astype(int), 0, line.shape[0] - 1)
                    drawn[ys, xs] = True
                edge &= binary_dilation(drawn, np.ones((3, 3)), iterations=2)
                away = distance_transform_edt(~edge)
            else:
                pieces = [(run, stroke) for run in contours]
            reach = stroke / 2 + OUTLINE_SLACK
            runs = off_outline(runs, edge, reach, max(3.0, stroke))
            off = away > reach
            outer_parts.extend(
                _outline_strokes(pieces, colour(palette[outer_ink]), tolerance)
            )
            outer_pieces = pieces
            contours = [run for run, _ in pieces]
        outer_details = {
            "outline": len(contours),
            "outline_width": round(stroke, 2),
            "outline_inked": outer_pixels is not None,
        }
    if not line_width and strokes:
        # A line whose width changes a lot is drawn as a stroke per width.
        runs = [
            run[first : last + 1]
            for run in runs
            for first, last in width_pieces(
                _widths_along(across, run), max(LINE_PIECE, LINE_PIECE_WIDTHS * typical)
            )
        ]
    measured = []
    for run in runs:
        index, widths = style(run)
        width = float(np.median(widths))
        measured.append(
            (
                index,
                width,
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
    if not strokes:
        # Filled shapes: where the ink is at least half a line's darkness.
        shape = ink >= 0.5
        if off is not None:
            shape &= off
        colours = np.square(target[..., None, :] - palette).sum(-1).argmin(-1)
        paths = list(outer_parts)
        for index, value in enumerate(palette):
            data = mask_path(shape & (colours == index), density=DENSITY, smooth=SMOOTH)
            if data is None:
                continue
            paths.append(
                f'<path d="{_simplified_data(data, tolerance)}" '
                f'fill="{colour(value)}" fill-rule="nonzero"/>'
            )
        return paths, {
            **details,
            **outer_details,
            "line_style": "filled",
            "line_paths": len(paths),
        }
    # The runs of one colour and about one width share a path.
    widths = np.array([max(THINNEST, m[1]) for m in measured])
    colours = np.array([m[0] for m in measured])
    group = np.zeros(len(runs), dtype=np.int64)
    if not line_width:
        for index in np.unique(colours):
            own = np.flatnonzero(colours == index)
            for number, members in enumerate(width_groups(widths[own], lengths[own])):
                group[own[members]] = number
    grouped: dict[tuple[int, int], list[tuple[Subpath, float, int]]] = {}
    group_runs: dict[tuple[int, int], list[np.ndarray]] = {}
    for run, index, width, step in zip(runs, colours, widths, group, strict=True):
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
        grouped.setdefault((int(index), int(step)), []).append((contour, width, size))
        group_runs.setdefault((int(index), int(step)), []).append(run)
    paths = list(outer_parts)
    pieces_before = pieces_after = 0
    # How far each centreline pixel's stroke reaches either side of it.
    reaches = np.full(line.shape, -1.0)
    # The outline covers the outer ink as the strokes do theirs.
    for run, width in outer_pieces:
        xs = np.clip(run[:, 0].astype(int), 0, line.shape[1] - 1)
        ys = np.clip(run[:, 1].astype(int), 0, line.shape[0] - 1)
        reaches[ys, xs] = np.maximum(reaches[ys, xs], width / 2)
    for index, step in sorted(grouped):
        pieces = grouped[index, step]
        width = _weighted_percentiles(
            np.array([w for _, w, _ in pieces]),
            np.array([size for _, _, size in pieces], dtype=float),
            (50,),
        )[0]
        for run in group_runs[index, step]:
            xs = np.clip(run[:, 0].astype(int), 0, line.shape[1] - 1)
            ys = np.clip(run[:, 1].astype(int), 0, line.shape[0] - 1)
            reaches[ys, xs] = np.maximum(reaches[ys, xs], width / 2)
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
    filled = _uncovered_ink(target, line, palette, covers, reaches, tolerance)
    return filled + paths, {
        **details,
        **outer_details,
        "ink_fills": len(filled),
        "line_style": "strokes",
        "line_paths": len(paths),
        "line_pieces": pieces_after,
        "line_runs_joined": pieces_before - pieces_after,
    }


def _outline_strokes(
    pieces: list[tuple[np.ndarray, float]], paint: str, tolerance: float
) -> list[str]:
    """The outline's *pieces*, runs and their widths, fitted with curves and
    stroked in *paint*, one path per group of about one width."""
    if not pieces:
        return []
    widths = np.array([width for _, width in pieces])
    lengths = np.array([len(run) for run, _ in pieces], dtype=float)
    paths = []
    for members in width_groups(widths, lengths):
        width = _weighted_percentiles(widths[members], lengths[members], (50,))[0]
        contours = [_contour(pieces[i][0], tolerance) for i in members]
        paths.append(_stroke(contours, paint, width))
    return paths


def _stroke(contours: list[Subpath], paint: str, width: float) -> str:
    """*contours* as one round-capped stroke of *paint*, *width* wide."""
    data = " ".join(
        _data(
            c.nodes[0].endpoint,
            [(n.command, n.values) for n in c.nodes[1:]],
            c.closed,
        )
        for c in contours
    )
    return (
        f'<path d="{data}" fill="none" stroke="{paint}" stroke-width="{width:.2f}" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
    )


def _uncovered_ink(
    target: np.ndarray,
    line: np.ndarray,
    palette: np.ndarray,
    covers: list[np.ndarray],
    reaches: np.ndarray,
    tolerance: float,
) -> list[str]:
    """The ink the strokes leave out, as filled shapes beneath them.

    A dark mark wider than its stroke but too small to be a filled shape,
    such as an eye or an eyebrow, would otherwise keep only the stroke down
    its middle, its other pixels going to the region around it. Where line
    pixels beyond every stroke's reach (*reaches*, half its width at each
    centreline pixel) are mostly one of the *palette* inks, by *covers*,
    the mark of that ink around them, where it is at least UNCOVERED_WIDE
    across, is filled, in pieces of at least UNCOVERED_LEAST pixels.
    """
    drawn = reaches >= 0
    if not drawn.any():
        return []
    y, x = nearest_indices(~drawn)
    rows, columns = np.indices(line.shape)
    distance = np.hypot(rows - y, columns - x)
    uncovered = line & (distance > reaches[y, x] + 0.5)
    colours = np.square(target[..., None, :] - palette).sum(-1).argmin(-1)
    paths = []
    for index, value in enumerate(palette):
        # The marks of this ink wide enough to fill, where the strokes leave
        # some of them out.
        inked = line & (colours == index) & (covers[index] >= 0.5)
        wide = binary_opening(inked, _disk(UNCOVERED_WIDE // 2))
        mask = wide & binary_dilation(uncovered & inked, np.ones((3, 3)))
        pieces, count = label(mask, np.ones((3, 3)))
        if not count:
            continue
        keep = np.bincount(pieces.ravel(), minlength=count + 1) >= UNCOVERED_LEAST
        keep[0] = False
        data = mask_path(keep[pieces], density=DENSITY, smooth=SMOOTH)
        if data is None:
            continue
        paths.append(
            f'<path d="{_simplified_data(data, tolerance)}" '
            f'fill="{colour(value)}" fill-rule="nonzero"/>'
        )
    return paths


def width_groups(
    widths: np.ndarray,
    lengths: np.ndarray,
    spread: float = WIDTH_GROUP,
    most: int = WIDTH_GROUPS,
) -> list[np.ndarray]:
    """The runs *widths* wide and *lengths* long in groups of about one
    width, by index: each widest at most *spread* times its narrowest, then
    the least used merged into a neighbour until there are at most *most*
    and each holds a twentieth of the length."""
    order = np.argsort(widths)
    groups: list[list[int]] = []
    for i in order:
        if groups and widths[i] <= spread * widths[groups[-1][0]]:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    total = float(lengths.sum())

    def used(group: list[int]) -> float:
        return float(lengths[group].sum())

    while len(groups) > 1:
        least = min(range(len(groups)), key=lambda g: used(groups[g]))
        if len(groups) <= most and used(groups[least]) >= total / 20:
            break
        # Into the neighbour nearest its width.
        middle = float(np.median(widths[groups[least]]))
        sides = [g for g in (least - 1, least + 1) if 0 <= g < len(groups)]
        into = min(
            sides,
            key=lambda g: abs(np.log(np.median(widths[groups[g]]) / middle)),
        )
        groups[into] = sorted(groups[into] + groups[least], key=lambda i: widths[i])
        del groups[least]
    return [np.array(g) for g in groups]


def _widths_along(across: np.ndarray, run: np.ndarray) -> np.ndarray:
    """The width of the line at each point of *run*."""
    xs = np.clip(run[:, 0].astype(int), 0, across.shape[1] - 1)
    ys = np.clip(run[:, 1].astype(int), 0, across.shape[0] - 1)
    return across[ys, xs]


def width_pieces(
    widths: np.ndarray, shortest: float, ratio: float = WIDTH_STEP
) -> list[tuple[int, int]]:
    """The run whose points are *widths* wide, as (first, last) pieces cut
    where its width steps by more than *ratio*, each at least *shortest*
    points long; adjacent pieces share the point between them.

    Each cut is where the typical widths either side differ the most, and
    the pieces are cut again until no step that large is left.
    """
    logs = np.log(np.maximum(np.asarray(widths, dtype=np.float64), 0.5))
    # The run's ends sit in junctions, where the ink of several lines meets.
    if len(logs) > 4:
        logs[[0, -1]] = logs[[1, -2]]
    minimum = max(2, int(np.ceil(shortest)))

    def cut(first: int, last: int) -> list[tuple[int, int]]:
        count = last - first + 1
        if count < 2 * minimum:
            return [(first, last)]
        sums = np.concatenate(([0.0], np.cumsum(logs[first : last + 1])))
        at = np.arange(minimum, count - minimum + 1)
        before = sums[at] / at
        after = (sums[-1] - sums[at]) / (count - at)
        step = np.abs(before - after)
        best = int(step.argmax())
        if step[best] <= np.log(ratio):
            return [(first, last)]
        middle = first + int(at[best])
        return [*cut(first, middle), *cut(middle, last)]

    return cut(0, len(logs) - 1)


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
