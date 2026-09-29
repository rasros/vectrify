"""No-LLM colour-region vectorizer with CUDA palette fitting.

A representation separate from SAMVG: a CUDA-fitted palette divides the image
into colour regions, which are traced into shared-contour SVG regions, with
optional preserved dark outlines and a final conservative geometry cleanup.
Needs PyTorch with CUDA and SciPy (the ``vision`` extra).
"""

from __future__ import annotations

import logging
import time
from heapq import heappop, heappush
from itertools import pairwise
from typing import cast
from xml.etree import ElementTree as ET

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import (
    binary_propagation,
    gaussian_filter,
    gaussian_filter1d,
    grey_closing,
    label,
    map_coordinates,
    maximum_filter,
    maximum_filter1d,
    maximum_position,
    minimum_filter,
    watershed_ift,
)
from scipy.ndimage import (
    distance_transform_edt as _distance_transform_edt,
)

from vectrify.formats.svg.cleanup import cleanup_svg_geometry
from vectrify.refine.samvg import _loops

log = logging.getLogger(__name__)


def distance_transform_edt(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance to the nearest zero, typed for the one return value."""
    return cast(np.ndarray, _distance_transform_edt(mask))


def nearest_indices(mask: np.ndarray) -> tuple[np.ndarray, ...]:
    """Index arrays of the nearest zero for every pixel, for fancy indexing."""
    indices = _distance_transform_edt(mask, return_distances=False, return_indices=True)
    return tuple(cast(np.ndarray, indices))


def simplify(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Keep endpoints and bound sample-to-segment error without recursion."""
    keep = {0, len(points) - 1}
    pending = [(0, len(points) - 1)]
    while pending:
        first, last = pending.pop()
        if last - first < 2:
            continue
        sample = points[first : last + 1]
        start, end = sample[0], sample[-1]
        delta = end - start
        length = float(delta @ delta)
        if length:
            parameter = np.clip((sample - start) @ delta / length, 0, 1)
            error = np.square(sample - start - parameter[:, None] * delta).sum(1)
        else:
            error = np.square(sample - start).sum(1)
        split = int(error.argmax())
        if error[split] > tolerance * tolerance:
            middle = first + split
            keep.add(middle)
            pending.extend(((first, middle), (middle, last)))
    return points[sorted(keep)]


def simplify_loop(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Simplify a closed polygon without deleting its fill or hole."""
    split = int(np.square(points - points[0]).sum(1).argmax())
    if not split:
        return points
    first = simplify(points[: split + 1], tolerance)
    second = simplify(np.concatenate((points[split:], points[:1])), tolerance)
    selected = np.concatenate((first[:-1], second[:-1]))
    return selected if len(selected) >= 3 else points


def trace(
    mask: np.ndarray, tolerance: float, *, baseline_tolerance: float | None = None
) -> str:
    parts = []
    for loop in _loops(mask):
        points = np.asarray(loop)
        if baseline_tolerance is not None and baseline_tolerance < tolerance:
            # Keep a compact fallback for small texture islands. Falling back
            # to every raster step would make stronger simplification larger.
            points = simplify_loop(points, baseline_tolerance)
        selected = simplify_loop(points, tolerance)
        if len(selected) >= 3:
            parts.append("M" + " L".join(f"{x:g} {y:g}" for x, y in selected) + " Z")
    return " ".join(parts)


def fit_palette(pixels: np.ndarray, count: int, steps: int) -> np.ndarray:
    """Fit representative colours on CUDA using deterministic farthest seeds."""
    # Bound training memory independently of input size. Labels below still
    # use every original-resolution pixel, not an upscaled small label map.
    flat = pixels.reshape(-1, 3)
    sample = torch.tensor(flat[:: max(1, len(flat) // 131072)] / 255, device="cuda")
    centers = [sample.mean(0)]
    nearest = torch.full((len(sample),), float("inf"), device="cuda")
    for _ in range(count - 1):
        nearest = torch.minimum(nearest, (sample - centers[-1]).square().sum(1))
        centers.append(sample[nearest.argmax()])
    centers = torch.stack(centers)
    for _ in range(steps):
        distance = (
            sample.square().sum(1, keepdim=True)
            + centers.square().sum(1)[None]
            - 2 * sample @ centers.T
        )
        assigned = distance.argmin(1)
        sums = torch.zeros_like(centers).index_add_(0, assigned, sample)
        counts = torch.bincount(assigned, minlength=count)
        centers = torch.where(
            counts[:, None] > 0, sums / counts[:, None].clamp_min(1), centers
        )
    labels = np.empty(len(flat), dtype=np.int32)
    for start in range(0, len(flat), 131072):
        batch = torch.tensor(flat[start : start + 131072] / 255, device="cuda")
        distance = (
            batch.square().sum(1, keepdim=True)
            + centers.square().sum(1)[None]
            - 2 * batch @ centers.T
        )
        labels[start : start + len(batch)] = distance.argmin(1).cpu().numpy()
    return labels.reshape(pixels.shape[:2])


def remove_fragments(labels: np.ndarray, minimum: int) -> np.ndarray:
    valid = np.zeros(labels.shape, dtype=bool)
    for index in np.unique(labels):
        components, _ = label(labels == index)
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        valid |= sizes[components] >= minimum
    if not valid.any():
        raise ValueError("minimum component size removes every region")
    if valid.all():
        return labels
    nearest = nearest_indices(~valid)
    return labels[nearest]


def colour(value: np.ndarray) -> str:
    red, green, blue = np.clip(np.rint(value), 0, 255).astype(int)
    return f"#{red:02x}{green:02x}{blue:02x}"


def dark_outline_mask(
    target: np.ndarray, *, radius: int = 3, contrast: float = 6
) -> np.ndarray:
    """Find narrow dark ridges, retaining their connected antialiased edges.

    Grayscale closing estimates the local surface without a narrow dark line.
    Hysteresis keeps weaker pixels only when connected to a strong ridge.
    This targets dark linework; it is not a general coloured-outline detector.
    """
    if radius < 1 or not np.isfinite(contrast) or contrast <= 0:
        raise ValueError("outline radius and contrast must be positive")
    luminance = target @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    ridge = grey_closing(luminance, size=2 * radius + 1) - luminance
    return binary_propagation(ridge >= contrast, mask=ridge >= contrast / 3)


def append_outlines(
    svg: str,
    target: np.ndarray,
    *,
    radius: int = 3,
    contrast: float = 6,
    colours: int = 12,
) -> tuple[str, dict]:
    """Overlay fine filled vector contours without fragment pruning or strokes."""
    mask = dark_outline_mask(target, radius=radius, contrast=contrast)
    metadata = {"outline_pixels": int(mask.sum()), "outline_colours": 0}
    if not mask.any():
        return svg, metadata
    assigned = fit_palette(target[mask][:, None, :], colours, 24).ravel()
    root = ET.fromstring(svg)
    group = ET.SubElement(
        root, "{http://www.w3.org/2000/svg}g", {"id": "preserved-outlines"}
    )
    for index in np.unique(assigned):
        region = np.zeros(mask.shape, dtype=bool)
        region[mask] = assigned == index
        ET.SubElement(
            group,
            "{http://www.w3.org/2000/svg}path",
            {
                "d": trace(region, 0.25),
                "fill": colour(target[region].mean(0)),
                "fill-rule": "evenodd",
            },
        )
    metadata["outline_colours"] = len(group)
    return ET.tostring(root, encoding="unicode"), metadata


def boundary_chains(labels: np.ndarray) -> list[np.ndarray]:
    """Trace every shared region edge once, including loops and junctions."""
    _height, width = labels.shape
    graph: dict[int, list[int]] = {}

    def add(first: int, second: int) -> None:
        graph.setdefault(first, []).append(second)
        graph.setdefault(second, []).append(first)

    for y, x in zip(*np.nonzero(labels[:, :-1] != labels[:, 1:]), strict=True):
        first = int(y) * (width + 1) + int(x) + 1
        add(first, first + width + 1)
    for y, x in zip(*np.nonzero(labels[:-1] != labels[1:]), strict=True):
        first = (int(y) + 1) * (width + 1) + int(x)
        add(first, first + 1)
    visited: set[tuple[int, int]] = set()
    traces = []

    def edge(first: int, second: int) -> tuple[int, int]:
        return min(first, second), max(first, second)

    nodes = sorted(point for point, neighbours in graph.items() if len(neighbours) != 2)
    for start in [*nodes, *sorted(graph)]:
        for following in graph[start]:
            if edge(start, following) in visited:
                continue
            trace, previous, current = [start], start, following
            visited.add(edge(start, following))
            while True:
                trace.append(current)
                if current == start or len(graph[current]) != 2:
                    break
                following = next(point for point in graph[current] if point != previous)
                if edge(current, following) in visited:
                    break
                visited.add(edge(current, following))
                previous, current = current, following
            traces.append(
                np.array(
                    [(p % (width + 1), p // (width + 1)) for p in trace], dtype=float
                )
            )
    return traces


def simplify_detail_chain(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Suppress short raster wobble while anchoring persistent sharp corners."""
    closed = np.array_equal(points[0], points[-1])
    raw = points[:-1] if closed else points
    if len(raw) < 15:
        return points
    indices = np.arange(len(raw))
    bends = []
    for scale in (3, 7):
        before = (
            raw[(indices - scale) % len(raw)]
            if closed
            else raw[np.maximum(indices - scale, 0)]
        )
        after = (
            raw[(indices + scale) % len(raw)]
            if closed
            else raw[np.minimum(indices + scale, len(raw) - 1)]
        )
        left, right = raw - before, after - raw
        norm = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        bends.append(-(left * right).sum(1) / np.maximum(norm, 1e-8))
    score = np.minimum(bends[0], bends[1])
    peaks = np.flatnonzero(
        (score > -0.15)
        & (score == maximum_filter1d(score, 7, mode="wrap" if closed else "nearest"))
    )
    pins = {0, len(raw) - 1} if not closed else {0}
    for index in sorted(peaks, key=lambda i: -score[i]):
        if all(
            min(abs(index - other), len(raw) - abs(index - other)) > 3 for other in pins
        ):
            pins.add(int(index))
    smooth = gaussian_filter1d(raw, 1.15, axis=0, mode="wrap" if closed else "nearest")
    smooth[sorted(pins)] = raw[sorted(pins)]
    if closed:
        smooth = np.concatenate((smooth, smooth[:1]))
        pins.add(len(raw))
        if len(pins) == 2:
            pins.add(int(np.square(raw - raw[0]).sum(1).argmax()))
    ordered = sorted(pins)
    result = [simplify(smooth[a : b + 1], tolerance)[:-1] for a, b in pairwise(ordered)]
    result.append(smooth[ordered[-1] : ordered[-1] + 1])
    selected = np.concatenate(result)
    return selected if not closed or len(selected) >= 4 else points


def shared_region_geometry(
    labels: np.ndarray, tolerance: float = 0.65, *, denoise: bool = False
):
    """Simplify each interface once and reuse it in both neighbouring fills.

    Padding includes the canvas perimeter in the mesh. Optional denoising
    anchors sharp corners before smoothing and simplifying between them.
    """
    padded = np.pad(labels, 1, constant_values=-1)
    chains = boundary_chains(padded)
    pieces: dict[int, list[np.ndarray]] = {}
    interfaces = []
    for points in chains:
        middle = (points[0] + points[1]) / 2
        direction = points[1] - points[0]
        normal = np.array([-direction[1], direction[0]]) * 0.25
        x, y = np.floor(middle + normal).astype(int)
        first = int(padded[y, x])
        x, y = np.floor(middle - normal).astype(int)
        second = int(padded[y, x])
        points = points - 1
        if denoise:
            selected = simplify_detail_chain(points, tolerance)
        elif np.array_equal(points[0], points[-1]):
            split = int(np.square(points - points[0]).sum(1).argmax())
            selected = np.concatenate(
                (
                    simplify(points[: split + 1], tolerance)[:-1],
                    simplify(points[split:], tolerance),
                )
            )
            if len(selected) < 4:
                selected = points
        else:
            selected = simplify(points, tolerance)
        if first >= 0:
            pieces.setdefault(first, []).append(selected)
        if second >= 0:
            pieces.setdefault(second, []).append(selected[::-1])
        if min(first, second) >= 0:
            interfaces.append((points, selected))
    contours = {}
    for index, segments in pieces.items():
        starts: dict[tuple, list[int]] = {}
        for n, segment in enumerate(segments):
            starts.setdefault(tuple(segment[0]), []).append(n)
        unused = set(range(len(segments)))
        loops = []
        while unused:
            n = min(unused)
            current = [segments[n]]
            unused.remove(n)
            start, end = tuple(current[0][0]), tuple(current[-1][-1])
            while end != start:
                following = next(n for n in starts[end] if n in unused)
                unused.remove(following)
                current.append(segments[following][1:])
                end = tuple(current[-1][-1])
            loops.append(np.concatenate(current))
        contours[index] = loops
    return contours, interfaces


def polygon_path(points: np.ndarray) -> str:
    data = "M" + " L".join(f"{x:g} {y:g}" for x, y in points)
    return data + (" Z" if np.array_equal(points[0], points[-1]) else "")


def follow_ink_boundaries(target: np.ndarray, labels: np.ndarray, minimum: int):
    """Grow fixed region interiors to ink ridges without optimizing the SVG."""
    boundary = minimum_filter(labels, size=3) != maximum_filter(labels, size=3)
    interior = distance_transform_edt(~boundary) >= 3
    seeds = np.where(interior, labels + 1, 0).astype(np.int32)
    for index in np.unique(labels):
        region = labels == index
        components, count = label(region)
        seeded = np.bincount(components[seeds > 0], minlength=count + 1)
        missing = np.flatnonzero(seeded[1:] == 0) + 1
        if len(missing):
            # A thin island may have no pixel three pixels from its edge.
            # Retain an interior seed instead of silently deleting the island.
            positions = maximum_position(
                distance_transform_edt(region), components, missing
            )
            for y, x in positions:
                seeds[y, x] = int(index) + 1
    luminance = target @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    ridge = grey_closing(luminance, size=7) - luminance
    height_map = np.clip(ridge * 8, 0, 255).astype(np.uint8)
    return remove_fragments(watershed_ift(height_map, seeds) - 1, minimum)


def supported_stroke_runs(
    points: np.ndarray, luminance: np.ndarray, contrast: float
) -> list[np.ndarray]:
    """Keep locally ink-supported edges, bridging only short weak gaps.

    Proximity to another ink line is insufficient: the candidate itself must
    be darker than both neighbouring surfaces. This rejects the inner side
    of colour slivers that previously acquired a second outline.
    """
    delta = np.diff(points, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    counts = np.maximum(2, np.ceil(lengths / 0.75).astype(int))
    segment = np.repeat(np.arange(len(lengths)), counts)
    parameter = np.concatenate([(np.arange(n) + 0.5) / n for n in counts])
    centers = points[segment] + parameter[:, None] * delta[segment]
    normal = np.column_stack((-delta[:, 1], delta[:, 0])) / np.maximum(
        lengths[:, None], 1e-8
    )
    offsets = np.array([-4, -3, -2, -0.5, 0, 0.5, 2, 3, 4])
    samples = centers[:, None] + offsets[None, :, None] * normal[segment, None]
    values = map_coordinates(
        luminance,
        [samples[:, :, 1] - 0.5, samples[:, :, 0] - 0.5],
        order=1,
        mode="nearest",
    )
    surroundings = np.minimum(values[:, :3].mean(1), values[:, -3:].mean(1))
    ink = surroundings - values[:, 3:6].min(1) >= contrast / 3
    supported = (
        np.bincount(segment, weights=ink, minlength=len(lengths)) / counts >= 0.45
    )
    closed = np.array_equal(points[0], points[-1])
    # Anchor a circular chain on a supported segment so weak runs wrapping
    # across its arbitrary first vertex are handled just like all others.
    if closed and supported.any():
        start = int(np.flatnonzero(supported)[0])
        points = np.concatenate((points[start:-1], points[: start + 1]))
        supported = np.roll(supported, -start)
        lengths = np.roll(lengths, -start)
    position = 0
    while position < len(supported):
        if supported[position]:
            position += 1
            continue
        end = position + 1
        while end < len(supported) and not supported[end]:
            end += 1
        left = position > 0 and supported[position - 1]
        right = (end < len(supported) and supported[end]) or (
            closed and end == len(supported) and supported[0]
        )
        if left and right and lengths[position:end].sum() <= 4:
            supported[position:end] = True
        position = end
    runs = []
    position = 0
    while position < len(supported):
        if not supported[position]:
            position += 1
            continue
        end = position + 1
        while end < len(supported) and supported[end]:
            end += 1
        runs.append(points[position : end + 1])
        position = end
    if closed and len(runs) > 1 and supported[0] and supported[-1]:
        runs[0] = np.concatenate((runs[-1][:-1], runs[0]))
        runs.pop()
    return [
        run for run in runs if np.linalg.norm(np.diff(run, axis=0), axis=1).sum() >= 4
    ]


def deduplicate_outline_branches(
    candidates: list[np.ndarray], luminance: np.ndarray, contrast: float
) -> list[np.ndarray]:
    """Reject weakly inked detours competing with a nearby real contour.

    A thin third-colour patch creates two paths between the same junctions.
    Keep both when the source inks both sides; otherwise remove only the
    unsupported detour. Unpaired/faint contours retain their continuity.
    """
    lengths = [np.linalg.norm(np.diff(p, axis=0), axis=1).sum() for p in candidates]
    scores = [
        sum(
            np.linalg.norm(np.diff(run, axis=0), axis=1).sum()
            for run in supported_stroke_runs(points, luminance, contrast)
        )
        / max(lengths[index], 1e-8)
        for index, points in enumerate(candidates)
    ]
    graph: dict[tuple, list[tuple]] = {}
    for index, points in enumerate(candidates):
        if scores[index] < 0.8 or np.array_equal(points[0], points[-1]):
            continue
        start, end = tuple(points[0]), tuple(points[-1])
        graph.setdefault(start, []).append((end, index, False))
        graph.setdefault(end, []).append((start, index, True))
    removed = set()
    for index, weak in enumerate(candidates):
        if scores[index] >= 0.8 or np.array_equal(weak[0], weak[-1]):
            continue
        start, end = tuple(weak[0]), tuple(weak[-1])
        queue = [(0.0, start)]
        distances = {start: 0.0}
        parents = {}
        while queue:
            distance, node = heappop(queue)
            if distance != distances[node]:
                continue
            if node == end:
                break
            for following, edge, reverse in graph.get(node, []):
                cost = distance + lengths[edge]
                if cost > max(50, lengths[index] * 4):
                    continue
                if cost < distances.get(following, np.inf):
                    distances[following] = cost
                    parents[following] = (node, edge, reverse)
                    heappush(queue, (cost, following))
        if end not in parents:
            continue
        route = []
        support = 0.0
        node = end
        while node != start:
            node, edge, reverse = parents[node]
            support += scores[edge] * lengths[edge]
            points = candidates[edge][::-1] if reverse else candidates[edge]
            route.append(points[:-1])
        if support / max(distances[end], 1e-8) < scores[index] + 0.15:
            continue
        strong = np.concatenate([*reversed(route), weak[-1:]])
        ring = np.concatenate((strong, weak[::-1]))
        area = abs(np.sum(ring[:-1, 0] * ring[1:, 1] - ring[1:, 0] * ring[:-1, 1])) / 2
        # Area/perimeter excludes substantial regions with fainter ink on
        # one side. The alternate contour can span several junctions.
        mean_width = 2 * area / max(distances[end] + lengths[index], 1e-8)
        if mean_width <= 12:
            removed.add(index)
    return [points for index, points in enumerate(candidates) if index not in removed]


def clean_region_svg(
    target: np.ndarray,
    *,
    colours: int,
    steps: int,
    minimum: int,
    radius: int,
    contrast: float,
    width: float,
    regions: int = 3,
    texture_tolerance: float = 5.0,
) -> tuple[str, dict]:
    """Build outlines and independently simplified texture on shared contours.

    Texture tolerance does not affect structural segmentation, silhouettes,
    or ink strokes. Each region contour is defined once and reused for its
    filled underlay and clipping path.
    """
    if not 2 <= regions <= 256 or not np.isfinite(width) or width <= 0:
        raise ValueError("invalid clean-outline region count or stroke width")
    if not np.isfinite(texture_tolerance) or texture_tolerance < 0:
        raise ValueError("texture tolerance must be finite and nonnegative")
    mask = dark_outline_mask(target, radius=radius, contrast=contrast)
    if mask.any() and not mask.all():
        nearest = nearest_indices(mask)
        surface = target[nearest]
    else:
        surface = target
    # Source ink is removed before palette fitting, preventing two parallel
    # boundaries around the ink. Light smoothing suppresses pixel texture but
    # retains narrow grass tips that the previous 2px fill blur erased.
    labels = remove_fragments(
        fit_palette(gaussian_filter(surface, (0.35, 0.35, 0)), regions, steps), minimum
    )
    # Extend labelled interiors to the source ink ridge. Assigning all ink
    # to the darker neighbour previously ate light grass tips and made steps.
    labels = follow_ink_boundaries(target, labels, minimum)
    contours, interfaces = shared_region_geometry(labels, 0.85, denoise=True)
    height, image_width = labels.shape
    texture = remove_fragments(
        fit_palette(gaussian_filter(surface, (2, 2, 0)), colours, steps), minimum
    )
    texture_fills = {
        int(i): colour(surface[texture == i].mean(0)) for i in np.unique(texture)
    }
    parts = [
        f'<rect width="{image_width}" height="{height}" '
        f'fill="{colour(surface.mean((0, 1)))}"/>'
    ]
    texture_vertices = 0
    for index, loops in contours.items():
        data = " ".join(polygon_path(points) for points in loops)
        region = labels == index
        fill = colour(surface[region].mean(0))
        # The clip and fill reuse the exact same contour as the outline mesh.
        # Extending texture paths through the clip covers simplification slivers
        # without exposing unrelated colours from the neighbouring region.
        parts.append(
            f'<defs><path id="region-shape-{index}" d="{data}" '
            'fill-rule="evenodd" clip-rule="evenodd"/>'
            f'<clipPath id="region-{index}">'
            f'<use xlink:href="#region-shape-{index}"/></clipPath></defs>'
        )
        parts.append(
            f'<use xlink:href="#region-shape-{index}" fill="{fill}" '
            f'stroke="{fill}" stroke-width="0.25"/>'
        )
        parts.append(f'<g clip-path="url(#region-{index})">')
        # Pull boundary colours from the region interior so the previous
        # raster edge cannot remain visible beside the new vector contour.
        interior = distance_transform_edt(region) > 4
        if interior.any():
            nearest = nearest_indices(~interior)
            region_texture = np.where(interior, texture, texture[nearest])
        else:
            region_texture = np.full_like(
                texture, np.bincount(texture[region]).argmax()
            )
        for shade in np.unique(region_texture[region]):
            texture_path = trace(
                region & (region_texture == shade),
                texture_tolerance,
                baseline_tolerance=1.25,
            )
            texture_vertices += texture_path.count("M") + texture_path.count("L")
            tint = texture_fills[int(shade)]
            parts.append(
                f'<path d="{texture_path}" fill="{tint}" fill-rule="evenodd" '
                f'stroke="{tint}" stroke-width="3" stroke-linejoin="round"/>'
            )
        parts.append("</g>")
    ink = colour(np.percentile(target[mask], 10, axis=0)) if mask.any() else "#000000"
    distance = distance_transform_edt(~mask)
    candidates = []
    luminance = target @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    for original, selected in interfaces:
        xs = np.clip(np.rint(original[:, 0]).astype(int), 0, image_width - 1)
        ys = np.clip(np.rint(original[:, 1]).astype(int), 0, height - 1)
        if mask.any() and len(original) >= 8 and np.mean(distance[ys, xs] <= 2) >= 0.65:
            candidates.append(selected)
    selected_paths = [
        polygon_path(points)
        for points in deduplicate_outline_branches(candidates, luminance, contrast)
    ]
    parts.append(
        f'<g id="clean-outlines" fill="none" stroke="{ink}" '
        f'stroke-width="{width:g}" stroke-linejoin="miter" '
        'stroke-miterlimit="6" stroke-linecap="round">'
    )
    parts.extend(f'<path d="{data}"/>' for data in selected_paths)
    parts.append("</g>")
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{image_width}" height="{height}" '
        f'viewBox="0 0 {image_width} {height}">' + "".join(parts) + "</svg>"
    )
    return svg, {
        "outline_style": "clean",
        "outline_support": "local contrast resolves competing narrow branches",
        "outline_branches_removed": len(candidates) - len(selected_paths),
        "outline_paths": len(selected_paths),
        "outline_width": width,
        "outline_colour": ink,
        "shared_fill_stroke_geometry": True,
        "palette_colours": len(texture_fills),
        "texture_tolerance_pixels": texture_tolerance,
        "texture_vertices": texture_vertices,
        "shared_region_definitions": len(contours),
        "outline_regions": regions,
        "contour_tolerance_pixels": 0.85,
        "contour_denoise_sigma": 1.15,
        "contour_assignment": "source ink watershed",
        "contour_smoothing_sigma": 0.35,
    }


def vectorize(
    image: Image.Image,
    *,
    colours: int = 32,
    steps: int = 24,
    min_pixels: int = 64,
    tolerance: float = 1.25,
    smooth_sigma: float = 1.5,
    preserve_outlines: bool = False,
    outline_radius: int = 3,
    outline_contrast: float = 6,
    outline_style: str = "preserve",
    outline_regions: int = 3,
    outline_width: float = 1.5,
    texture_tolerance: float = 5.0,
    geometry_cleanup: bool = True,
) -> tuple[str, dict]:
    if not torch.cuda.is_available():
        raise RuntimeError("Colour regions need CUDA; no CPU fit is substituted")
    if not 2 <= colours <= 256 or steps < 1 or min_pixels < 1:
        raise ValueError("invalid palette, step count, or minimum component size")
    if outline_style not in {"preserve", "clean"}:
        raise ValueError("outline_style must be preserve or clean")
    if (
        not np.isfinite([tolerance, smooth_sigma]).all()
        or min(tolerance, smooth_sigma) < 0
    ):
        raise ValueError("tolerance and smoothing must be finite and nonnegative")
    target = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width = target.shape[:2]
    started = time.monotonic()
    if preserve_outlines and outline_style == "clean":
        svg, metadata = clean_region_svg(
            target,
            colours=colours,
            steps=steps,
            minimum=min_pixels,
            radius=outline_radius,
            contrast=outline_contrast,
            width=outline_width,
            regions=outline_regions,
            texture_tolerance=texture_tolerance,
        )
        if geometry_cleanup:
            svg, cleanup_metrics = cleanup_svg_geometry(svg)
            metadata["geometry_cleanup"] = cleanup_metrics
        return svg, {
            **metadata,
            "preserve_outlines": True,
            "source_size": [width, height],
            "min_component_pixels": min_pixels,
            "palette_steps": steps,
            "seconds": time.monotonic() - started,
            "bytes": len(svg.encode()),
            "gpu": torch.cuda.get_device_name(),
            "llm_calls": 0,
            "evolutionary_steps": 0,
            "method": (
                "experimental shared-contour colour regions; "
                "not the two-phase SAMVG pipeline"
            ),
        }
    smooth = gaussian_filter(target, (smooth_sigma, smooth_sigma, 0))
    labels = remove_fragments(fit_palette(smooth, colours, steps), min_pixels)
    log.debug("Palette and region cleanup complete")
    order = np.argsort(-np.bincount(labels.ravel(), minlength=colours))
    fills = {
        int(index): colour(target[labels == index].mean(0))
        for index in order
        if (labels == index).any()
    }
    # The underlay makes coverage unconditional. Small same-colour strokes
    # hide antialias seams between independently simplified adjacent regions.
    parts = [f'<rect width="{width}" height="{height}" fill="{fills[int(order[0])]}"/>']
    for index, fill in fills.items():
        data = trace(labels == index, tolerance)
        parts.append(
            f'<path d="{data}" fill="{fill}" fill-rule="evenodd" stroke="{fill}" '
            f'stroke-width="{tolerance:g}" stroke-linejoin="round"/>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">' + "".join(parts) + "</svg>"
    )
    outline_metadata = {}
    if preserve_outlines:
        svg, outline_metadata = append_outlines(
            svg, target, radius=outline_radius, contrast=outline_contrast
        )
    if geometry_cleanup:
        svg, cleanup_metrics = cleanup_svg_geometry(svg)
        outline_metadata["geometry_cleanup"] = cleanup_metrics
    return svg, {
        **outline_metadata,
        "preserve_outlines": preserve_outlines,
        "source_size": [width, height],
        "palette_colours": len(fills),
        "min_component_pixels": min_pixels,
        "tolerance_pixels": tolerance,
        "smooth_sigma": smooth_sigma,
        "palette_steps": steps,
        "seconds": time.monotonic() - started,
        "bytes": len(svg.encode()),
        "gpu": torch.cuda.get_device_name(),
        "llm_calls": 0,
        "evolutionary_steps": 0,
        "method": "experimental colour regions; not the two-phase SAMVG pipeline",
    }
