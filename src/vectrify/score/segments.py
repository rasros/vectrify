"""Approximate, disjoint cartoon tiles for retaining locally good candidates."""

import json
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
DETAIL_MAX_AREA_FRACTION = 0.12


@dataclass(frozen=True)
class Segment:
    """One flat-colour connected component at scoring resolution."""

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

        ys, xs = zip(*points, strict=True)
        top = max(0, min(ys) - DETAIL_GROUP_RADIUS)
        bottom = min(height, max(ys) + DETAIL_GROUP_RADIUS + 1)
        left = max(0, min(xs) - DETAIL_GROUP_RADIUS)
        right = min(width, max(xs) + DETAIL_GROUP_RADIUS + 1)
        area = (bottom - top) * (right - left)
        mass = float(edges[top:bottom, left:right].sum())
        if 8 <= area <= max_area and mass >= 4.0:
            mask = np.zeros(grouped.shape, dtype=bool)
            mask[top:bottom, left:right] = True
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


def segment_target(image: Image.Image, *, max_regions: int = 8) -> list[Segment]:
    """Return exactly ``max_regions`` disjoint, non-empty cartoon tiles.

    Palette quantisation absorbs antialiasing and shading. Adjacent palette
    regions merge across weak target edges first, so a smooth gradient stays
    whole while real outlines survive. Up to two compact high-edge regions are
    protected as detail tiles; broad tiles are bisected only if needed.
    """
    if max_regions < 1:
        return []
    detail_masks = _detail_masks(image, min(DETAIL_SLOTS, max_regions - 1))
    base_count = max_regions - len(detail_masks)
    labels = np.asarray(
        image.convert("RGB").quantize(
            colors=PALETTE_SIZE, method=Image.Quantize.MEDIANCUT
        )
    )
    pieces = [(int(label), labels == label) for label in np.unique(labels)]
    if len(pieces) > base_count:
        pieces = _merge_weak_boundaries(
            pieces, edge_map(image, tolerance=0), base_count
        )
    while len(pieces) < base_count:
        index = max(range(len(pieces)), key=lambda item: int(pieces[item][1].sum()))
        label, mask = pieces[index]
        split = _split(mask)
        if split is None:
            break
        pieces[index : index + 1] = [(label, split[0]), (label, split[1])]
    detail_coverage = np.logical_or.reduce(detail_masks)
    pieces = [
        (label, remainder)
        for label, mask in pieces
        if (remainder := mask & ~detail_coverage).any()
    ]
    # A protected detail box can entirely consume a small colour region. Split
    # a broad remainder to keep the promised tile count without discarding it.
    while len(pieces) < base_count:
        index = max(range(len(pieces)), key=lambda item: int(pieces[item][1].sum()))
        label, mask = pieces[index]
        split = _split(mask)
        if split is None:
            break
        pieces[index : index + 1] = [(label, split[0]), (label, split[1])]
    all_pieces: list[tuple[int | None, np.ndarray]] = [
        *pieces,
        *((None, mask) for mask in detail_masks),
    ]
    all_pieces.sort(key=lambda item: int(item[1].sum()), reverse=True)
    return [
        Segment(index=index, label_id=label, mask=mask, detail=label is None)
        for index, (label, mask) in enumerate(all_pieces[:max_regions])
    ]


def segment_error(
    comparison: Comparison, mask: np.ndarray, *, detail: bool = False
) -> float:
    """Colour-and-structure error reduced within one tile."""
    if not mask.any():
        return 1.0
    colour = float(comparison.colour[mask].mean())
    structure = overlap_distance(
        comparison.reference_edges * mask, comparison.candidate_edges * mask
    )
    edge_weight = 0.75 if detail else 0.5
    return clamp01(edge_weight * structure + (1.0 - edge_weight) * colour)


def save_segments(segments: list[Segment], run_dir: Path) -> None:
    """Persist tile masks and their manifest alongside ``lineage.csv``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for segment in segments:
        filename = f"segment-{segment.index:02d}.png"
        Image.fromarray(segment.mask.astype(np.uint8) * 255, mode="L").save(
            run_dir / filename
        )
        manifest.append(
            {
                "index": segment.index,
                "label_id": segment.label_id,
                "pixels": int(segment.mask.sum()),
                "detail": segment.detail,
                "mask": filename,
            }
        )
    (run_dir / "segments.json").write_text(json.dumps(manifest, indent=2) + "\n")
