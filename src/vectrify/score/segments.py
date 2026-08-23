"""Edge-aware Voronoi masks for retaining locally good candidates."""

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from vectrify.score.compare import Comparison
from vectrify.score.edges import edge_map, overlap_distance
from vectrify.score.utils import clamp01

PALETTE_SIZE = 16
DETAIL_SLOTS = 2
DETAIL_EDGE_THRESHOLD = 0.25
DETAIL_GROUP_RADIUS = 4
DETAIL_PADDING = 6
DETAIL_MAX_AREA_FRACTION = 0.12
MIN_SEGMENT_PIXELS = 64
SEGMENT_COLOURS = np.array(
    [
        (222, 82, 83),
        (65, 160, 212),
        (94, 184, 118),
        (239, 178, 72),
        (166, 103, 190),
        (68, 187, 178),
        (220, 118, 160),
        (144, 144, 144),
    ],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class Segment:
    """One soft, edge-centred attention field at scoring resolution."""

    index: int
    label_id: int | None
    mask: np.ndarray
    detail: bool = False

    @property
    def metric_name(self) -> str:
        return f"segment_{self.index}"


def _split(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Bisect a tile through its median occupied row or column."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 2:
        return None
    by_x = xs.max() - xs.min() >= ys.max() - ys.min()
    values = xs if by_x else ys
    pivot = int(np.median(values))
    first = mask.copy()
    second = mask.copy()
    if by_x:
        first[:, pivot + 1 :] = False
        second[:, : pivot + 1] = False
    else:
        first[pivot + 1 :, :] = False
        second[: pivot + 1, :] = False
    return (first, second) if first.any() and second.any() else None


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill background components that cannot reach the canvas boundary."""
    height, width = mask.shape
    exterior = np.zeros(mask.shape, dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for x in range(width):
        for y in (0, height - 1):
            if not mask[y, x] and not exterior[y, x]:
                exterior[y, x] = True
                queue.append((y, x))
    for y in range(height):
        for x in (0, width - 1):
            if not mask[y, x] and not exterior[y, x]:
                exterior[y, x] = True
                queue.append((y, x))
    while queue:
        y, x = queue.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if (
                0 <= ny < height
                and 0 <= nx < width
                and not mask[ny, nx]
                and not exterior[ny, nx]
            ):
                exterior[ny, nx] = True
                queue.append((ny, nx))
    return ~exterior


def _remove_tile_holes(
    pieces: list[tuple[int | None, np.ndarray]],
) -> list[tuple[int | None, np.ndarray]]:
    """Give each enclosed void to the tile that surrounds it."""
    for index in sorted(
        range(len(pieces)), key=lambda item: int(pieces[item][1].sum()), reverse=True
    ):
        label, mask = pieces[index]
        # The canvas tile necessarily surrounds every drawing element. It is
        # not a meaningful hole in the target partition, so never absorb it.
        if (
            mask[0, :].any()
            or mask[-1, :].any()
            or mask[:, 0].any()
            or mask[:, -1].any()
        ):
            continue
        filled = _fill_holes(mask)
        gained = filled & ~mask
        if not gained.any():
            continue
        pieces[index] = (label, filled)
        for other, (other_label, other_mask) in enumerate(pieces):
            if other != index:
                pieces[other] = (other_label, other_mask & ~gained)
    return [(label, mask) for label, mask in pieces if mask.any()]


def _merge_weak_boundaries(
    pieces: list[tuple[int, np.ndarray]], edges: np.ndarray, count: int
) -> list[tuple[int, np.ndarray]]:
    """Merge adjacent palette regions across the weakest visible boundary."""
    while len(pieces) > count:
        owner = np.full(edges.shape, -1, dtype=np.int16)
        for index, (_label, mask) in enumerate(pieces):
            owner[mask] = index

        pairs: list[tuple[np.ndarray, np.ndarray]] = []
        for left, right, left_edge, right_edge in (
            (owner[:, :-1], owner[:, 1:], edges[:, :-1], edges[:, 1:]),
            (owner[:-1, :], owner[1:, :], edges[:-1, :], edges[1:, :]),
        ):
            different = left != right
            a, b = left[different], right[different]
            keys = np.minimum(a, b) * len(pieces) + np.maximum(a, b)
            strength = np.maximum(left_edge[different], right_edge[different])
            pairs.append((keys, strength))

        keys = np.concatenate([pair[0] for pair in pairs])
        strengths = np.concatenate([pair[1] for pair in pairs])
        unique, inverse = np.unique(keys, return_inverse=True)
        totals = np.bincount(inverse, weights=strengths)
        sizes = np.bincount(inverse)
        weakest = int(unique[np.argmin(totals / sizes)])
        first, second = divmod(weakest, len(pieces))
        label, mask = pieces[first]
        pieces[first] = (label, mask | pieces[second][1])
        del pieces[second]
    return pieces


def _grow_small_tiles(
    pieces: list[tuple[int | None, np.ndarray]],
    edges: np.ndarray,
    minimum_pixels: int,
) -> list[tuple[int | None, np.ndarray]]:
    """Expand undersized fragments through their lowest-edge boundary."""
    while len(pieces) > 1:
        source = min(range(len(pieces)), key=lambda index: int(pieces[index][1].sum()))
        if int(pieces[source][1].sum()) >= minimum_pixels:
            break

        owner = np.full(edges.shape, -1, dtype=np.int16)
        for index, (_label, mask) in enumerate(pieces):
            owner[mask] = index
        source_mask = owner == source
        touching = np.zeros(owner.shape, dtype=bool)
        touching[1:, :] |= source_mask[:-1, :]
        touching[:-1, :] |= source_mask[1:, :]
        touching[:, 1:] |= source_mask[:, :-1]
        touching[:, :-1] |= source_mask[:, 1:]
        candidates = touching & ~source_mask
        ys, xs = np.nonzero(candidates)
        if len(xs) == 0:
            break
        needed = minimum_pixels - int(source_mask.sum())
        order = np.argsort(edges[ys, xs], kind="stable")[:needed]
        grown = np.zeros(owner.shape, dtype=bool)
        grown[ys[order], xs[order]] = True
        label, mask = pieces[source]
        pieces[source] = (label, mask | grown)
        for index, (other_label, other_mask) in enumerate(pieces):
            if index != source:
                pieces[index] = (other_label, other_mask & ~grown)
    return pieces


def _detail_masks(image: Image.Image, count: int) -> list[np.ndarray]:
    """Find compact, ink-dense neighbourhoods such as text and small features."""
    if count <= 0:
        return []
    edges = edge_map(image, tolerance=0)
    ink = (edges >= DETAIL_EDGE_THRESHOLD).astype(np.uint8) * 255
    grouped = (
        np.asarray(
            Image.fromarray(ink, mode="L").filter(
                ImageFilter.MaxFilter(DETAIL_GROUP_RADIUS * 2 + 1)
            )
        )
        > 0
    )
    height, width = grouped.shape
    seen = np.zeros(grouped.shape, dtype=bool)
    candidates: list[tuple[float, np.ndarray]] = []
    max_area = height * width * DETAIL_MAX_AREA_FRACTION

    for y, x in np.ndindex(grouped.shape):
        if seen[y, x] or not grouped[y, x]:
            continue
        points: list[tuple[int, int]] = []
        queue = deque([(y, x)])
        seen[y, x] = True
        while queue:
            cy, cx = queue.popleft()
            points.append((cy, cx))
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if (
                    0 <= ny < height
                    and 0 <= nx < width
                    and not seen[ny, nx]
                    and grouped[ny, nx]
                ):
                    seen[ny, nx] = True
                    queue.append((ny, nx))

        component = np.zeros(grouped.shape, dtype=np.uint8)
        ys, xs = zip(*points, strict=True)
        component[ys, xs] = 255
        mask = (
            np.asarray(
                Image.fromarray(component, mode="L").filter(
                    ImageFilter.MaxFilter(DETAIL_PADDING * 2 + 1)
                )
            )
            > 0
        )
        mask = _fill_holes(mask)
        area = int(mask.sum())
        mass = float(edges[mask].sum())
        if 8 <= area <= max_area and mass >= 4.0:
            candidates.append((mass, mask))

    selected: list[np.ndarray] = []
    for _mass, mask in sorted(
        candidates, key=lambda candidate: candidate[0], reverse=True
    ):
        if all(not (mask & existing).any() for existing in selected):
            selected.append(mask)
        if len(selected) == count:
            break
    return selected


def _cluster_masks(image: Image.Image, count: int) -> list[np.ndarray]:
    """Fit edge clusters, then turn them into Voronoi masks."""
    edges = edge_map(image, tolerance=0)
    points = np.argwhere(edges >= DETAIL_EDGE_THRESHOLD)
    if len(points) < count:
        points = np.argwhere(np.ones(edges.shape, dtype=bool))
    weights = edges[points[:, 0], points[:, 1]] + 0.05
    centers = [points[int(np.argmax(weights))].astype(float)]
    nearest = np.sum((points - centers[0]) ** 2, axis=1)
    for _ in range(1, count):
        index = int(np.argmax(nearest * weights))
        centers.append(points[index].astype(float))
        nearest = np.minimum(nearest, np.sum((points - centers[-1]) ** 2, axis=1))
    centers_array = np.asarray(centers)
    for _ in range(12):
        distances = ((points[:, None, :] - centers_array[None, :, :]) ** 2).sum(axis=2)
        assignment = distances.argmin(axis=1)
        for index in range(count):
            assigned = assignment == index
            if assigned.any():
                centers_array[index] = np.average(
                    points[assigned], axis=0, weights=weights[assigned]
                )

    yy, xx = np.indices(edges.shape)
    distances = (yy[..., None] - centers_array[None, None, :, 0]) ** 2 + (
        xx[..., None] - centers_array[None, None, :, 1]
    ) ** 2
    ownership = np.argmin(distances, axis=2)
    return [ownership == index for index in range(count)]


def _balanced_region_labels(masks: list[np.ndarray]) -> np.ndarray:
    """Expand small SAM regions into adjacent background territory."""
    from scipy.ndimage import distance_transform_edt

    regions = [~np.logical_or.reduce(masks), *masks]
    distances = np.stack(
        [np.asarray(distance_transform_edt(~mask)) for mask in regions]
    )
    labels = np.argmin(distances, axis=0)
    areas = np.bincount(labels.ravel(), minlength=len(regions)).astype(float)
    target = areas + 0.35 * (areas.mean() - areas)
    bias = np.zeros(len(regions))
    for _ in range(40):
        labels = np.argmax(-distances + bias[:, None, None], axis=0)
        areas = np.bincount(labels.ravel(), minlength=len(regions))
        bias += 0.08 * (target - areas) / np.maximum(target, 1)
    return labels


def _merge_indistinguishable(labels: np.ndarray, image: Image.Image) -> np.ndarray:
    """Merge touching regions with a weak, low-contrast shared boundary."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    edges = edge_map(image, tolerance=0)
    changed = True
    while changed:
        changed = False
        for first, second in (
            (labels[:, :-1], labels[:, 1:]),
            (labels[:-1], labels[1:]),
        ):
            different = first != second
            for left, right in zip(first[different], second[different], strict=True):
                a, b = sorted((int(left), int(right)))
                if a == b:
                    continue
                mask_a, mask_b = labels == a, labels == b
                colour = float(
                    np.abs(rgb[mask_a].mean(axis=0) - rgb[mask_b].mean(axis=0)).mean()
                )
                boundary = (labels == a) & np.asarray(
                    Image.fromarray(mask_b.astype(np.uint8) * 255).filter(
                        ImageFilter.MaxFilter(3)
                    )
                ).astype(bool)
                edge_strength = float(edges[boundary].mean()) if boundary.any() else 0.0
                if colour < 0.035 and edge_strength < 0.12:
                    labels[labels == b] = a
                    changed = True
                    break
            if changed:
                break
    return labels


def segment_target(image: Image.Image, *, max_regions: int = 8) -> list[Segment]:
    """Return meaningful SAM regions after merging indistinguishable neighbours."""
    if max_regions < 1:
        return []
    from transformers import pipeline

    generated = pipeline("mask-generation", model="facebook/sam-vit-base", device=0)(
        image, points_per_batch=32, points_per_crop=16
    )["masks"]
    masks = [np.asarray(mask, dtype=bool) for mask in generated]
    labels = _merge_indistinguishable(_balanced_region_labels(masks), image)
    candidates = [labels == label for label in np.unique(labels)]
    edges = edge_map(image, tolerance=0)
    candidates.sort(
        key=lambda mask: float(edges[mask].sum()) + 0.05 * np.sqrt(mask.sum()),
        reverse=True,
    )
    return [
        Segment(index=index, label_id=None, mask=mask)
        for index, mask in enumerate(candidates[:max_regions])
    ]


def segment_error(
    comparison: Comparison, mask: np.ndarray, *, detail: bool = False
) -> float:
    """Colour-and-structure error weighted by one local attention field."""
    weights = mask.astype(np.float32)
    total_weight = float(weights.sum())
    if total_weight == 0.0:
        return 1.0
    colour = float((comparison.colour * weights).sum() / total_weight)
    structure = overlap_distance(
        comparison.reference_edges * weights, comparison.candidate_edges * weights
    )
    edge_weight = 0.75 if detail else 0.5
    return clamp01(edge_weight * structure + (1.0 - edge_weight) * colour)


def save_segments(segments: list[Segment], run_dir: Path, target: Image.Image) -> None:
    """Persist translucent Voronoi masks over the scoring target."""
    run_dir.mkdir(parents=True, exist_ok=True)
    if not segments:
        return
    height, width = segments[0].mask.shape
    image = np.asarray(target.convert("RGB").resize((width, height)), dtype=np.float32)
    for segment in segments:
        image[segment.mask] = (
            0.62 * image[segment.mask]
            + 0.38 * SEGMENT_COLOURS[segment.index % len(SEGMENT_COLOURS)]
        )
    Image.fromarray(image.astype(np.uint8), mode="RGB").save(run_dir / "segments.png")
