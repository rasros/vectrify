"""Redraw outline: fit a stretch of outline to the reference edge beside a stroke.

The stroke is drawn roughly along an edge of the reference. Within a band a
few pixels either side of it, the reference's colour gradient makes a cost
image: cheap along strong edges, dear across flat colour and further from the
stroke. The cheapest path through the band between the stroke's two ends,
Dijkstra over the band's pixels, follows the edge where there is one and the
stroke where there is none. Each of its points then moves to the peak of the
gradient across it, for the edge's place within a pixel.

The path is fitted with a short cubic every few pixels, split at sharp turns
so a spike's tip stays a tip, and simplified to a pixel tolerance, the dense
trace then simplify of ``tracing.mask_path``. Without a reference the stroke
itself is fitted, smoothed a little, to a tolerance in screen pixels.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from vectrify.document import Document, Geometry, PathNode, Subpath
from vectrify.document.model import new_id
from vectrify.document.transforms import root_matrix
from vectrify.image_utils import on_white
from vectrify.refine.frozen import Frozen
from vectrify.refine.simplify import simplified_geometry, straightened
from vectrify.refine.snap import _Frame

# How far from the stroke the edge may be, in screen pixels, and never less
# than this many reference pixels.
BAND = 8.0
LEAST_BAND = 3.0
# Blur, in reference pixels, before the gradient is read.
SIGMA = 0.6
# Gradient (RGB 0-1 per pixel) that counts as a full edge.
FULL_EDGE = 0.12
# Cost of a pixel on a full edge, and of one a full band from the stroke, both
# against one across flat colour on the stroke.
ON_EDGE = 0.05
STRAY = 0.5
# How far a path point may move onto the gradient's peak, in pixels.
PEAK = 1.5
# Points each dense cubic spans, and the turn, in degrees over TURN_SPAN
# points either side, that makes a corner.
SPACING = 6
CORNER = 45.0
TURN_SPAN = 3
# Points either side a direction is taken over.
TANGENT = 2
# How far the fit may stray: reference pixels, or screen pixels freehand.
TOLERANCE = 0.6
FREEHAND_TOLERANCE = 1.0
# Smoothing along the traced edge, in reference pixels, and along a
# freehand stroke, in screen pixels.
SMOOTH = 0.7
FREEHAND_SMOOTH = 1.5


def redraw_stretch(
    document: Document,
    object_id: str,
    stroke: list[tuple[float, float]],
    reference: Image.Image | None,
    pixel: float,
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    """New segments along *stroke*, in the object's local coordinates.

    *stroke* is in root user space and runs between the two places it
    attaches to, its first and last points; *pixel* is a screen pixel's size
    there. Follows the reference's edge near the stroke when there is a
    reference. Returns (command, values) for each segment after the first
    point.
    """
    line = np.asarray(stroke, dtype=np.float64)
    if line.ndim != 2 or line.shape[1] != 2 or len(line) < 2:
        raise ValueError("A stroke needs at least two points")
    a, b, c, d, e, f = root_matrix(document, object_id)
    linear = np.array([[a, c], [b, d]], dtype=np.float64)
    offset = np.array([e, f], dtype=np.float64)
    # Freehand, in screen pixels, unless the reference has an edge to follow.
    scale, origin = np.eye(2) / pixel, np.zeros(2)
    traced = line / pixel
    tolerance, sigma = FREEHAND_TOLERANCE, FREEHAND_SMOOTH
    if reference is not None:
        vx, vy, vw, vh = document.artboard()
        image = on_white(reference)
        to_pixels = np.diag([image.width / vw, image.height / vh])
        pixels = (line - [vx, vy]) @ to_pixels.T
        band = max(LEAST_BAND, BAND * pixel * float(to_pixels.max()))
        edge = _edge_path(image, pixels, band)
        if edge is not None:
            scale, origin, traced = to_pixels, np.array([vx, vy]), edge
            tolerance, sigma = TOLERANCE, SMOOTH
    frame = _Frame(scale @ linear, scale @ (offset - origin))
    geometry = _fitted(traced, tolerance, sigma)
    return tuple(
        (node.command, frame.local(np.asarray(node.values).reshape(-1, 2)))
        for node in geometry.subpaths[0].nodes[1:]
    )


def _edge_path(image: Image.Image, stroke: np.ndarray, band: float):
    """The cheapest path along the edge beside *stroke*, in reference pixels,
    or None where the stroke's ends are off the reference."""
    width, height = image.size
    margin = int(np.ceil(band + 3 * SIGMA + 2))
    low = np.floor(stroke.min(axis=0)).astype(int) - margin
    high = np.ceil(stroke.max(axis=0)).astype(int) + margin
    x0, y0 = max(0, low[0]), max(0, low[1])
    x1, y1 = min(width, high[0]), min(height, high[1])
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    local = stroke - [x0, y0]
    size = np.array([x1 - x0, y1 - y0])
    ends = np.floor(local[[0, -1]]).astype(int)
    if (ends < 0).any() or (ends >= size).any():
        return None
    crop = np.asarray(image.crop((x0, y0, x1, y1)), dtype=np.float64) / 255
    gradient = np.sqrt(
        sum(
            ndimage.gaussian_gradient_magnitude(crop[..., c], SIGMA) ** 2
            for c in range(3)
        )
    )
    # How far each pixel's centre is from the stroke.
    marked = np.zeros(crop.shape[:2], dtype=bool)
    dense = _resampled(local, 0.25)
    columns = np.clip(np.floor(dense[:, 0]).astype(int), 0, size[0] - 1)
    rows = np.clip(np.floor(dense[:, 1]).astype(int), 0, size[1] - 1)
    marked[rows, columns] = True
    distance = np.asarray(ndimage.distance_transform_edt(~marked), dtype=np.float64)
    inside = distance <= band
    edge = np.clip(gradient / FULL_EDGE, 0, 1)
    cost = ON_EDGE + (1 - edge) + STRAY * (distance / band) ** 2
    rows, columns = np.nonzero(inside)
    count = len(rows)
    index = np.full(crop.shape[:2], -1)
    index[rows, columns] = np.arange(count)
    sources, targets, weights = [], [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        near_rows, near_columns = rows + dy, columns + dx
        valid = (near_rows < size[1]) & (near_columns >= 0) & (near_columns < size[0])
        valid[valid] = inside[near_rows[valid], near_columns[valid]]
        here = (rows[valid], columns[valid])
        there = (near_rows[valid], near_columns[valid])
        sources.append(index[here])
        targets.append(index[there])
        weights.append((cost[here] + cost[there]) / 2 * np.hypot(dx, dy))
    graph = csr_matrix(
        (np.concatenate(weights), (np.concatenate(sources), np.concatenate(targets))),
        shape=(count, count),
    )
    start = index[ends[0, 1], ends[0, 0]]
    goal = index[ends[1, 1], ends[1, 0]]
    _, before = dijkstra(graph, directed=False, indices=start, return_predecessors=True)
    if goal != start and before[goal] < 0:
        return None
    chain = [goal]
    while chain[-1] != start:
        chain.append(before[chain[-1]])
    path = np.column_stack([columns, rows])[chain[::-1]] + 0.5
    # The ends are the places the stroke attaches to, exactly; the pixels
    # right beside them would only double back.
    gaps = np.minimum(
        np.linalg.norm(path - local[0], axis=1),
        np.linalg.norm(path - local[-1], axis=1),
    )
    path = np.vstack([local[0], path[gaps > 1], local[-1]])
    return _on_peaks(path, gradient) + np.array([x0, y0])


def _on_peaks(path: np.ndarray, gradient: np.ndarray) -> np.ndarray:
    """Each point moved across the path onto the gradient's peak, if near.

    The ends and corners stay: across a sharp tip there is no one way across.
    """
    if len(path) < 3:
        return path
    tangent = np.gradient(ndimage.gaussian_filter1d(path, 1.0, axis=0), axis=0)
    size = np.linalg.norm(tangent, axis=1, keepdims=True)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]]) / np.maximum(size, 1e-9)
    steps = np.linspace(-PEAK, PEAK, int(4 * PEAK) * 2 + 1)
    probe = path[:, None, :] + steps[None, :, None] * normal[:, None, :]
    values = ndimage.map_coordinates(
        gradient, [probe[..., 1] - 0.5, probe[..., 0] - 0.5], order=1, mode="nearest"
    )
    best = np.argmax(values, axis=1)
    moving = values[np.arange(len(path)), best] >= FULL_EDGE / 3
    moving[[0, -1]] = False
    for corner in _corners(path):
        moving[max(0, corner - TURN_SPAN) : corner + TURN_SPAN + 1] = False
    return path + np.where(moving, steps[best], 0)[:, None] * normal


def _resampled(line: np.ndarray, spacing: float) -> np.ndarray:
    """*line* with points every *spacing* along it, its ends kept."""
    steps = np.linalg.norm(np.diff(line, axis=0), axis=1)
    along = np.concatenate([[0], np.cumsum(steps)])
    if along[-1] < 1e-9:
        return line[[0, -1]]
    count = max(2, int(np.ceil(along[-1] / spacing)) + 1)
    at = np.linspace(0, along[-1], count)
    return np.column_stack(
        [np.interp(at, along, line[:, 0]), np.interp(at, along, line[:, 1])]
    )


def _smoothed(line: np.ndarray, sigma: float, kept: list[int]) -> np.ndarray:
    """*line* smoothed along its length over *sigma* points, each piece
    between its ends and the *kept* points on its own, those staying."""
    smooth = line.copy()
    for i, j in pairwise(sorted({0, len(line) - 1, *kept})):
        if j - i < 3:
            continue
        piece = ndimage.gaussian_filter1d(
            line[i : j + 1], sigma, axis=0, mode="nearest"
        )
        smooth[i + 1 : j] = piece[1:-1]
    return smooth


def _corners(
    line: np.ndarray, span: int = TURN_SPAN, corner: float = CORNER
) -> list[int]:
    """Where *line* turns sharply, more than *corner* degrees over *span*
    points either side, the sharpest point of each turn."""
    count = len(line)
    k = span
    if count < 2 * k + 1:
        return []
    before = line[k:-k] - line[: -2 * k]
    after = line[2 * k :] - line[k:-k]
    cosine = (before * after).sum(axis=1) / np.maximum(
        np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1), 1e-12
    )
    turn = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    found: list[int] = []
    for i in np.argsort(-turn):
        if turn[i] < corner:
            break
        if all(abs(i + k - j) > k for j in found):
            found.append(int(i + k))
    return sorted(found)


def _fitted(line: np.ndarray, tolerance: float, sigma: float) -> Geometry:
    """*line*, in pixels, as a simplified run of cubics from its first point.

    A cubic every SPACING points, meeting smoothly except at corners, where
    each side keeps its own direction.
    """
    line = _resampled(line, 1.0)
    corners = _corners(line)
    line = _smoothed(line, sigma, corners)
    last = len(line) - 1
    sharp = {0, last, *corners}
    breaks = sorted({*range(0, last, SPACING), *sharp, last})

    def direction(i: int, j: int) -> np.ndarray | None:
        step = line[max(0, min(last, j))] - line[i]
        size = float(np.linalg.norm(step))
        return step / size if size > 1e-9 else None

    nodes = [PathNode(new_id("node"), "M", tuple(line[0]))]
    for i, j in pairwise(breaks):
        chunk = line[i : j + 1]
        leave = (
            direction(i, i + TANGENT)
            if i in sharp
            else direction(i - TANGENT, i + TANGENT)
        )
        arrive = (
            direction(j, j - TANGENT)
            if j in sharp
            else direction(j + TANGENT, j - TANGENT)
        )
        handles = None
        if len(chunk) >= 3 and leave is not None and arrive is not None:
            handles = _cubic(chunk, leave, arrive)
        if handles is None:
            nodes.append(PathNode(new_id("node"), "L", tuple(chunk[-1])))
        else:
            nodes.append(PathNode(new_id("node"), "C", (*handles.ravel(), *chunk[-1])))
    geometry = Geometry(new_id("geometry"), (Subpath(new_id("subpath"), tuple(nodes)),))
    geometry = simplified_geometry(geometry, Frozen(frozenset()), tolerance)
    return straightened(geometry, tolerance / 2)


def _cubic(
    points: np.ndarray, leave: np.ndarray, arrive: np.ndarray
) -> np.ndarray | None:
    """The two handles of the cubic through *points*' ends, leaving along
    *leave* and arriving from *arrive*, whose lengths follow them closest."""
    start, end = points[0], points[-1]
    chord = float(np.linalg.norm(end - start))
    if chord < 1e-9:
        return None
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    t = np.concatenate([[0], np.cumsum(steps)]) / steps.sum()
    u = 1 - t
    fixed = (u**3 + 3 * u**2 * t)[:, None] * start + (3 * u * t**2 + t**3)[
        :, None
    ] * end
    basis = np.stack(
        [(3 * u**2 * t)[:, None] * leave, (3 * u * t**2)[:, None] * arrive], axis=-1
    ).reshape(-1, 2)
    lengths, *_ = np.linalg.lstsq(basis, (points - fixed).reshape(-1), rcond=None)
    if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
        lengths = np.array([chord / 3, chord / 3])
    lengths = np.minimum(lengths, chord)
    return np.vstack([start + lengths[0] * leave, end + lengths[1] * arrive])
